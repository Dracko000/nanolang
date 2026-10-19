"""SmolLM2-135M NanoLang Inference Runner.

Loads the ternary .nanow model and generates real text using pure NumPy.
Implements the full LLaMA-2 architecture: RoPE, GQA, SwiGLU, RMSNorm.
Ternary linear projections use ADD/SUB only — zero MatMul for 79% of params.

Usage:
    python3 run_smollm2.py [--prompt "text"] [--tokens 64] [--temp 0.8]
"""

from __future__ import annotations
import json
import struct
import sys
import os
import time
import mmap
import argparse
import ctypes
import re
import html
import urllib.request
import urllib.parse

from nanow_tools import ToolRegistry, run_agent

# Threading defaults for a small, memory-bound box: keep one OpenMP pool for the
# C kernels and stop OpenBLAS from spinning up a second, contending pool (the
# two together measured ~3x slower than a single pool). Set before numpy loads.
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

# ---------------------------------------------------------------------------
# .nanow Loader
# ---------------------------------------------------------------------------

NANOW_MAGIC = 0x4E414E4F  # 'NANO'

def load_nanow(path: str) -> dict:
    """Memory-map a .nanow file and return a dict of tensor name → (data, scale, shape, is_ternary)."""
    fd = os.open(path, os.O_RDONLY)
    fsize = os.fstat(fd).st_size
    mm = mmap.mmap(fd, fsize, access=mmap.ACCESS_READ)

    header_fmt = "<IIII IIII QQ I 3I"
    header_size = struct.calcsize(header_fmt)
    header = struct.unpack(header_fmt, mm[:header_size])

    magic, version, flags, num_tensors = header[0:4]
    n_layers, hidden, n_heads, vocab = header[4:8]
    total_params, total_size, directory_size = header[8:11]

    assert magic == NANOW_MAGIC, f"Bad magic: {magic:#x}"

    desc_fmt = "<64s III 4I Q Q Q f"
    desc_size = struct.calcsize(desc_fmt)

    # Dtype enum must match phase1/python/export_to_nanow.py
    DTYPE_TERNARY = 0
    DTYPE_INT8 = 1
    DTYPE_FP16 = 2
    DTYPE_FP32 = 3

    tensors = {}
    for i in range(num_tensors):
        offset = header_size + i * desc_size
        desc = struct.unpack(desc_fmt, mm[offset:offset + desc_size])
        name = desc[0].split(b"\x00")[0].decode()
        dtype_id = desc[1]
        packing = desc[2]
        ndims = desc[3]
        shape = list(desc[4:8])[:ndims] if ndims > 0 else [desc[4]]
        n_elements = desc[8]
        data_offset = desc[9]
        data_size = desc[10]
        scale = desc[11]

        raw = mm[data_offset:data_offset + data_size]

        if dtype_id == DTYPE_TERNARY:
            # Keep the on-disk 2-bit packed layout (4 weights/byte) so the packed
            # kernel can stream 4x fewer weight bytes from DRAM. Shape is
            # (M, K // 4) when K % 4 == 0 (true for every BitNet matrix here).
            packed = np.frombuffer(raw, dtype=np.uint8).copy()
            tensors[name] = {"data": packed, "scale": scale, "ternary": True,
                             "packed": True, "shape": tuple(shape),
                             "n_elements": n_elements}
        elif dtype_id == DTYPE_FP32:
            data = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
            tensors[name] = {"data": data, "scale": 1.0, "ternary": False}
        else:
            print(f"  [WARN] Unknown dtype {dtype_id} for {name}, skipping")

    info = {
        "n_layers": n_layers, "hidden": hidden, "state_dim": n_heads,
        "vocab": vocab, "total_params": total_params,
    }
    return tensors, info, mm, fd


# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """BPE tokenizer that reads tokenizer.json directly.

    Auto-detects two families:
      * GPT-2 byte-level BPE (SmolLM2, Qwen, ...): a byte encoder maps raw
        bytes to unicode code points, so a space becomes "Ġ".
      * SentencePiece BPE (LLaMA, BitNet, ...): the normalizer prepends "▁" and
        turns spaces into "▁"; out-of-vocabulary bytes fall back to <0xNN>.
    """

    def __init__(self, tokenizer_path: str):
        with open(tokenizer_path, "r") as f:
            tok = json.load(f)

        model = tok["model"]
        self.vocab = model["vocab"]  # str → int
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = []
        for m in model.get("merges", []):
            if isinstance(m, str):
                parts = m.split(" ", 1)
                if len(parts) == 2:
                    self.merges.append((parts[0], parts[1]))
            else:
                self.merges.append((m[0], m[1]))

        # Build merge priority (lower index = higher priority)
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # SentencePiece style if the vocabulary contains the "▁" token.
        self.spm = "\u2581" in self.vocab

        # Byte-to-unicode mapping (GPT-2 style byte encoding)
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        if self.spm:
            self.eos_id = self.vocab.get("</s>", self.vocab.get("<|endoftext|>", 2))
            self.bos_id = self.vocab.get("<s>", self.vocab.get("<|startoftext|>", 1))
        else:
            self.eos_id = 0  # <|endoftext|>
            self.bos_id = 1  # <|im_start|>

    @staticmethod
    def _bytes_to_unicode():
        """GPT-2 byte encoder: maps bytes 0-255 to unicode chars."""
        bs = list(range(ord("!"), ord("~") + 1))
        bs += list(range(ord("¡"), ord("¬") + 1))
        bs += list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}

    def _bpe(self, symbols: list[str]) -> list[str]:
        """Greedily apply BPE merges by rank until no merge rule applies."""
        while len(symbols) > 1:
            best_pair = None
            best_rank = float("inf")
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[i], symbols[i + 1]), float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = (symbols[i], symbols[i + 1])
            if best_pair is None or best_rank == float("inf"):
                break
            merged = best_pair[0] + best_pair[1]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self.spm:
            # SentencePiece normalizer: prepend "▁" and turn spaces into "▁".
            text = "\u2581" + text.replace(" ", "\u2581")
            symbols = list(text)
        else:
            symbols = [self.byte_encoder[b] for b in text.encode("utf-8")]

        symbols = self._bpe(symbols)

        ids = []
        for token in symbols:
            if token in self.vocab:
                ids.append(self.vocab[token])
            elif self.spm:
                # Byte fallback: OOV characters become <0xNN> byte tokens.
                for b in token.encode("utf-8"):
                    ids.append(self.vocab.get(f"<0x{b:02X}>", self.vocab.get("<unk>", 0)))
            else:
                for ch in token:
                    byte_val = self.byte_decoder.get(ch)
                    byte_char = self.byte_encoder.get(byte_val, ch) if byte_val is not None else ch
                    ids.append(self.vocab.get(byte_char, 0))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        if self.spm:
            parts = []
            for tid in ids:
                token = self.id_to_token.get(tid, "")
                if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
                    parts.append(bytes([int(token[3:5], 16)]))
                else:
                    parts.append(token.replace("\u2581", " ").encode("utf-8"))
            return b"".join(parts).decode("utf-8", errors="replace")

        tokens = [self.id_to_token.get(tid, "") for tid in ids]
        text = "".join(tokens)
        # Convert byte-level chars back to actual bytes
        byte_arr = bytearray()
        for ch in text:
            if ch in self.byte_decoder:
                byte_arr.append(self.byte_decoder[ch])
            else:
                byte_arr.extend(ch.encode("utf-8"))
        return byte_arr.decode("utf-8", errors="replace")

    def decode_token(self, tid: int) -> str:
        """Decode a single token ID to text."""
        return self.decode([tid])

    def token_text(self, tid: int) -> str:
        """Cached single-token text, for constrained (grammar) decoding."""
        cache = getattr(self, "_text_cache", None)
        if cache is None:
            cache = self._text_cache = {}
        if tid not in cache:
            cache[tid] = self.decode([tid])
        return cache[tid]

    def name_token_ids(self) -> list[int]:
        """Token ids usable inside a tool name (lowercase letters/underscore).

        Grammar-constrained tool-name decoding otherwise has to test the whole
        ~32k vocabulary at every step; this shrinks the candidate set to the few
        thousand tokens that could possibly appear in a name.
        """
        cached = getattr(self, "_name_ids", None)
        if cached is not None:
            return cached
        keep = []
        for tid in self.id_to_token:
            text = self.token_text(tid).lstrip(" ")
            if text and len(text) <= 16 and re.fullmatch(r"[a-z_]+", text):
                keep.append(tid)
        self._name_ids = keep
        return keep


# ---------------------------------------------------------------------------
# Online grounding (standard library only, no API keys)
# ---------------------------------------------------------------------------

class WebRetriever:
    """Fetch live web context so a model with a frozen knowledge cutoff can
    answer current questions instead of hallucinating.

    Two keyless sources:
      * DuckDuckGo HTML results — titles + snippets for general queries.
      * Wikipedia intro extracts — encyclopedic grounding for entities.

    This is deliberately tiny (stdlib + regex). It replaces the model's stale
    parametric memory with fresh, citable text; the model's job is then only to
    weave the retrieved facts into an answer.
    """

    # DuckDuckGo serves an empty result page to non-browser user agents, so we
    # present a normal desktop browser UA. No cookies or keys are involved.
    UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout

    def _get(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": self.UA})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")

    @staticmethod
    def _strip(markup: str) -> str:
        return html.unescape(re.sub(r"<[^>]+>", "", markup)).strip()

    @classmethod
    def _pairs(cls, page: str, title_pattern: str,
               snippet_pattern: str, n: int) -> list[tuple[str, str]]:
        """Pair the first ``n`` title/snippet matches of a results page."""
        titles = re.findall(title_pattern, page, re.S)
        snippets = re.findall(snippet_pattern, page, re.S)
        out = []
        for i in range(min(n, len(titles), len(snippets))):
            out.append((cls._strip(titles[i]), cls._strip(snippets[i])))
        return out

    def duckduckgo(self, query: str, n: int = 5) -> list[str]:
        """Return up to ``n`` 'Title: snippet' strings from DDG HTML results.

        Tries the full HTML endpoint first, then the lite endpoint whose markup
        uses different class names.
        """
        q = urllib.parse.urlencode({"q": query})
        pairs: list[tuple[str, str]] = []
        try:
            page = self._get("https://html.duckduckgo.com/html/?" + q)
            pairs = self._pairs(page, r'class="result__a"[^>]*>(.*?)</a>',
                                r'class="result__snippet"[^>]*>(.*?)</a>', n)
        except Exception:  # noqa: BLE001 - fall through to the lite endpoint
            pass
        if not pairs:
            page = self._get("https://lite.duckduckgo.com/lite/?" + q)
            pairs = self._pairs(page, r'class="result-link"[^>]*>(.*?)</a>',
                                r'class="result-snippet"[^>]*>(.*?)</td>', n)
        return [f"{t}: {s}" for t, s in pairs if s]

    def wikipedia(self, query: str):
        """Return (title, intro_extract) for the best-matching article."""
        api = "https://en.wikipedia.org/w/api.php"
        s = urllib.parse.urlencode({"action": "opensearch", "format": "json",
                                    "limit": 1, "search": query})
        hits = json.loads(self._get(f"{api}?{s}"))
        if not hits or not hits[1]:
            return None, None
        title = hits[1][0]
        p = urllib.parse.urlencode({"action": "query", "format": "json",
                                    "prop": "extracts", "exintro": 1,
                                    "explaintext": 1, "redirects": 1, "titles": title})
        data = json.loads(self._get(f"{api}?{p}"))
        page = next(iter(data["query"]["pages"].values()))
        return title, page.get("extract", "")

    def ground(self, query: str, max_chars: int = 900):
        """Return (context_text, sources) retrieved live for ``query``."""
        chunks, sources = [], []
        try:
            title, extract = self.wikipedia(query)
            if extract:
                chunks.append(f"Wikipedia ({title}): {extract}")
                sources.append(f"wikipedia:{title}")
        except Exception as exc:  # noqa: BLE001 - network best-effort
            sources.append(f"wikipedia:error({type(exc).__name__})")
        try:
            for s in self.duckduckgo(query, n=5):
                chunks.append(s)
            sources.append("duckduckgo:html")
        except Exception as exc:  # noqa: BLE001 - network best-effort
            sources.append(f"duckduckgo:error({type(exc).__name__})")
        return "\n".join(chunks).strip()[:max_chars], sources


# ---------------------------------------------------------------------------
# Model Components (Pure NumPy)
# ---------------------------------------------------------------------------

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm: x * weight / sqrt(mean(x²) + eps)."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def unpack_ternary_2bit(packed: np.ndarray, n_elements: int, shape=None) -> np.ndarray:
    """Decode a .nanow 2-bit ternary payload to int8 {-1, 0, +1}.

    Matches ``phase1/python/export_to_nanow.pack_ternary_2bit``:
    code 0 -> 0, 1 -> +1, 3 -> -1, packed 4 weights per byte (field t at shift 2t).
    """
    vals = [((packed >> (s * 2)) & 0x03) for s in range(4)]
    flat = np.column_stack(vals).ravel()[:n_elements].astype(np.int8)
    out = np.where(flat == 1, 1, np.where(flat == 3, -1, 0)).astype(np.int8)
    return out.reshape(shape) if shape is not None else out


def ternary_matmul(x: np.ndarray, W_ternary: np.ndarray, scale: float,
                   kernel=None) -> np.ndarray:
    """Zero-multiply matmul: y = scale * (x @ W^T) using only add/sub.

    W_ternary has values in {-1, 0, +1}.
    For each element: +1 → add, -1 → subtract, 0 → skip.
    Result equivalent to: scale * (x @ W^T)

    When a compiled ``TernaryKernel`` is provided, the real phase0 C kernel
    (``ternary_gemv_avx2``, the ``_mm256_sign_epi8`` add/sub path) is used with
    int8 activation quantization, matching BitNet's BitLinear exactly.
    """
    if kernel is not None:
        return kernel.gemv(W_ternary, x, scale)
    # NumPy fallback: materialize float weights and let BLAS do the matmul.
    W_f32 = W_ternary.astype(np.float32)
    return scale * (x @ W_f32.T)


# Elementwise ops carried by the phase0 C library. All four are compiled in;
# the runner may still route any of them to NumPy by trimming ``ew_ops``.
DEFAULT_EW_OPS = frozenset({"rmsnorm", "rope", "silu", "softmax"})


class TernaryKernel:
    """ctypes binding to phase0's ternary GEMV kernels.

    Preferred path ``ternary_gemv_f32`` does the whole BitNet BitLinear op in C:
    int8 activation quantization (per-row ``127 / max|X|``), the zero-multiply
    add/sub ternary GEMV (``_mm256_sign_epi8``), and dequantization by
    ``weight_scale / act_scale``. The int8-only ``ternary_gemv_avx2`` path is kept
    as a fallback for older builds.
    """

    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)

        fn8 = self.lib.ternary_gemv_avx2
        fn8.restype = None
        fn8.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int, ctypes.c_int]
        self._fn8 = fn8

        self._has_f32 = hasattr(self.lib, "ternary_gemv_f32")
        if self._has_f32:
            fnf = self.lib.ternary_gemv_f32
            fnf.restype = None
            fnf.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_int, ctypes.c_int, ctypes.c_float]
            self._fnf = fnf

        # 2-bit packed path: reads 4x fewer weight bytes (the DRAM bottleneck).
        self.has_packed = hasattr(self.lib, "ternary_gemv_packed_f32")
        if self.has_packed:
            fnp = self.lib.ternary_gemv_packed_f32
            fnp.restype = None
            fnp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_int, ctypes.c_int, ctypes.c_float]
            self._fnpack = fnp

        # Parallel FP32 GEMV (LM head) so the hot path avoids OpenBLAS threads.
        self.has_fp32gemv = hasattr(self.lib, "fp32_gemv_par")
        if self.has_fp32gemv:
            fng = self.lib.fp32_gemv_par
            fng.restype = None
            fng.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_int, ctypes.c_int]
            self._fng = fng

        # Elementwise transformer ops (RMSNorm, SiLU·gate, RoPE, softmax axis=1).
        # Optional: older builds simply lack them and the runner falls back to
        # the NumPy reference.
        self.has_ew = hasattr(self.lib, "nano_rmsnorm_f32")
        if self.has_ew:
            self.lib.nano_rmsnorm_f32.restype = None
            self.lib.nano_rmsnorm_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.c_void_p, ctypes.c_int,
                                                  ctypes.c_float]
            self.lib.nano_silu_mul_f32.restype = None
            self.lib.nano_silu_mul_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_int]
            self.lib.nano_rope_f32.restype = None
            self.lib.nano_rope_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_void_p, ctypes.c_int,
                                               ctypes.c_int]
            self.lib.nano_softmax_axis1_f32.restype = None
            self.lib.nano_softmax_axis1_f32.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                        ctypes.c_int, ctypes.c_int]

    def gemv(self, W: np.ndarray, x: np.ndarray, scale: float) -> np.ndarray:
        M, K = W.shape
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.empty(M, dtype=np.float32)

        if self._has_f32:
            # Single ctypes call: quantize + ternary GEMV + scale, all in C.
            self._fnf(y.ctypes.data, W.ctypes.data, x.ctypes.data, M, K, float(scale))
            return y

        # Fallback: quantize activations in Python, run the int8 kernel.
        amax = float(np.abs(x).max())
        if amax < 1e-8:
            return y  # zeros
        s = 127.0 / amax
        xq = np.clip(np.rint(x * s), -127.0, 127.0).astype(np.int8)
        yi = np.empty(M, dtype=np.int32)
        self._fn8(yi.ctypes.data, W.ctypes.data, xq.ctypes.data, M, K)
        return yi.astype(np.float32) * (scale / s)

    def gemv_packed(self, Wp: np.ndarray, x: np.ndarray, scale: float,
                    M: int, K: int) -> np.ndarray:
        """GEMV straight over 2-bit packed weights (Wp shape (M, K // 4))."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.empty(M, dtype=np.float32)
        self._fnpack(y.ctypes.data, Wp.ctypes.data, x.ctypes.data, M, K, float(scale))
        return y

    def fp32_gemv(self, W: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Parallel FP32 GEMV Y = W @ x (W row-major (M, K))."""
        M, K = W.shape
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.empty(M, dtype=np.float32)
        self._fng(y.ctypes.data, W.ctypes.data, x.ctypes.data, M, K)
        return y

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        w = np.ascontiguousarray(weight, dtype=np.float32)
        out = np.empty_like(x)
        self.lib.nano_rmsnorm_f32(out.ctypes.data, x.ctypes.data, w.ctypes.data,
                                  x.size, float(eps))
        return out

    def silu_mul(self, gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        gate = np.ascontiguousarray(gate, dtype=np.float32)
        up = np.ascontiguousarray(up, dtype=np.float32)
        out = np.empty_like(gate)
        self.lib.nano_silu_mul_f32(out.ctypes.data, gate.ctypes.data, up.ctypes.data,
                                   gate.size)
        return out

    def rope_inplace(self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray,
                     n_heads: int, head_dim: int) -> None:
        cos = np.ascontiguousarray(cos, dtype=np.float32)
        sin = np.ascontiguousarray(sin, dtype=np.float32)
        self.lib.nano_rope_f32(x.ctypes.data, cos.ctypes.data, sin.ctypes.data,
                               n_heads, head_dim)

    def softmax_axis1(self, x: np.ndarray) -> None:
        axis0, n, axis2 = x.shape
        self.lib.nano_softmax_axis1_f32(x.ctypes.data, axis0, n, axis2)


def rope_frequencies(dim: int, max_len: int, theta: float = 100000.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute RoPE frequency components."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(max_len, dtype=np.float32)
    angles = np.outer(t, freqs)  # (max_len, dim/2)
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)
    return cos_vals, sin_vals


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply Rotary Position Embedding to x.

    x shape: (n_heads, head_dim)
    cos, sin shape: (head_dim/2,)
    """
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return np.concatenate([out1, out2], axis=-1)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish activation."""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Transformer Forward Pass
# ---------------------------------------------------------------------------

class SmolLM2Runner:
    """Runs inference on a SmolLM2-135M loaded from .nanow format."""

    def __init__(self, tensors: dict, config: dict, tokenizer: BPETokenizer,
                 kernel: "TernaryKernel | None" = None):
        self.t = tensors
        self.tokenizer = tokenizer
        self.kernel = kernel
        self.n_layers = config["n_layers"]
        self.hidden = config["hidden"]
        self.n_heads = config["n_heads"]  # 9 query heads for SmolLM2-135M
        self.n_kv_heads = config.get("n_kv_heads", 3)
        self.head_dim = self.hidden // self.n_heads  # 576/9 = 64
        self.n_groups = self.n_heads // self.n_kv_heads  # 3 (heads per KV group)
        self.vocab = config["vocab"]
        self.eps = config.get("rms_norm_eps", 1e-5)
        self.rope_theta = config.get("rope_theta", 100000.0)
        self.max_len = config.get("max_position_embeddings", 2048)
        self.attn_scale = np.float32(1.0 / np.sqrt(self.head_dim))

        # Precompute RoPE
        self.rope_cos, self.rope_sin = rope_frequencies(
            self.head_dim, max_len=self.max_len, theta=self.rope_theta
        )

        # KV cache (list of (K, V) per layer, each shape grows with seq_len)
        self.kv_cache = [{"k": None, "v": None} for _ in range(self.n_layers)]
        self.pos = 0
        self.fed_ids: list[int] = []
        self._last_logits = None
        # All four elementwise ops route to C. SiLU and softmax only became a
        # win once the kernel carried its own AVX2 exp (poly exp, ~1e-7 rel
        # error, i.e. the float32 floor) rather than scalar expf, which lost to
        # NumPy's vectorised exp. Measured 1.9x-2.6x per op; see ROADMAP 7.7.
        self.ew_ops = set(DEFAULT_EW_OPS)

    def reset(self):
        self.kv_cache = [{"k": None, "v": None} for _ in range(self.n_layers)]
        self.pos = 0
        self.fed_ids: list[int] = []
        self._last_logits = None

    def _linear(self, x: np.ndarray, name: str) -> np.ndarray:
        """Apply linear projection (ternary or FP32)."""
        t = self.t[name]
        if not t["ternary"]:
            return self._fp32_linear(t["data"], x)
        if t.get("packed"):
            M, K = t["shape"]
            if self.kernel is not None and getattr(self.kernel, "has_packed", False) and K % 4 == 0:
                return self.kernel.gemv_packed(t["data"], x, t["scale"], M, K)
            # No packed kernel: unpack once, then reuse the int8/NumPy path.
            if "unpacked" not in t:
                t["unpacked"] = unpack_ternary_2bit(t["data"], t["n_elements"], t["shape"])
            return ternary_matmul(x, t["unpacked"], t["scale"], self.kernel)
        return ternary_matmul(x, t["data"], t["scale"], self.kernel)

    def _fp32_linear(self, W: np.ndarray, x: np.ndarray) -> np.ndarray:
        """FP32 projection y = W @ x, using the parallel C GEMV when available."""
        if self.kernel is not None and getattr(self.kernel, "has_fp32gemv", False):
            return self.kernel.fp32_gemv(W, x)
        return x @ W.T

    def _rms(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """RMSNorm via the C kernel when present, else the NumPy reference."""
        k = self.kernel
        if "rmsnorm" in self.ew_ops and k is not None and getattr(k, "has_ew", False):
            return k.rmsnorm(x, weight, self.eps)
        return rms_norm(x, weight, self.eps)

    def _silu_mul(self, gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        k = self.kernel
        if "silu" in self.ew_ops and k is not None and getattr(k, "has_ew", False):
            return k.silu_mul(gate, up)
        return silu(gate) * up

    def _rope(self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        k = self.kernel
        if "rope" in self.ew_ops and k is not None and getattr(k, "has_ew", False):
            k.rope_inplace(x, cos, sin, x.shape[0], x.shape[1])
            return x
        return apply_rope(x, cos, sin)

    def _softmax_axis1(self, scores: np.ndarray) -> np.ndarray:
        k = self.kernel
        if "softmax" in self.ew_ops and k is not None and getattr(k, "has_ew", False):
            k.softmax_axis1(scores)
            return scores
        return softmax(scores, axis=1)

    def forward_token(self, token_id: int) -> np.ndarray:
        """Forward pass for a single token, returns logits over vocabulary."""
        # Embedding lookup
        emb_data = self.t["embed_tokens"]["data"]  # (vocab, hidden)
        x = emb_data[token_id].copy()  # (hidden,)

        pos = self.pos

        for layer in range(self.n_layers):
            residual = x.copy()

            # --- Pre-attention RMSNorm ---
            norm_w = self.t[f"layers.{layer}.input_layernorm"]["data"]
            x = self._rms(x, norm_w)

            # --- Multi-Head Attention with GQA ---
            q = self._linear(x, f"layers.{layer}.attn.q_proj")  # (n_heads * head_dim,)
            k = self._linear(x, f"layers.{layer}.attn.k_proj")  # (n_kv_heads * head_dim,)
            v = self._linear(x, f"layers.{layer}.attn.v_proj")  # (n_kv_heads * head_dim,)

            # Reshape to heads
            q = q.reshape(self.n_heads, self.head_dim)      # (9, 64)
            k = k.reshape(self.n_kv_heads, self.head_dim)   # (3, 64)
            v = v.reshape(self.n_kv_heads, self.head_dim)   # (3, 64)

            # Apply RoPE to Q and K
            cos_pos = self.rope_cos[pos]  # (head_dim/2,)
            sin_pos = self.rope_sin[pos]
            q = self._rope(q, cos_pos, sin_pos)
            k = self._rope(k, cos_pos, sin_pos)

            # Update KV cache
            cache = self.kv_cache[layer]
            if cache["k"] is None:
                cache["k"] = k[np.newaxis, :, :]  # (1, n_kv_heads, head_dim)
                cache["v"] = v[np.newaxis, :, :]
            else:
                cache["k"] = np.concatenate([cache["k"], k[np.newaxis, :, :]], axis=0)
                cache["v"] = np.concatenate([cache["v"], v[np.newaxis, :, :]], axis=0)

            K = cache["k"]  # (seq_len, n_kv_heads, head_dim)
            V = cache["v"]  # (seq_len, n_kv_heads, head_dim)
            seq_len = K.shape[0]

            # Batched GQA attention: BLAS matmuls instead of a per-head Python
            # loop. Query head h reads KV head h // n_groups.
            qg = q.reshape(self.n_kv_heads, self.n_groups, self.head_dim)
            Kt = np.ascontiguousarray(K.transpose(1, 0, 2))  # (n_kv, seq, head_dim)
            Vt = np.ascontiguousarray(V.transpose(1, 0, 2))  # (n_kv, seq, head_dim)
            # scores[kv] = Kt[kv] @ qg[kv]^T  -> (n_kv, seq, n_groups)
            scores = (Kt @ qg.transpose(0, 2, 1)) * self.attn_scale
            probs = self._softmax_axis1(scores)  # over seq_len
            # out[kv] = probs[kv]^T @ Vt[kv]  -> (n_kv, n_groups, head_dim)
            attn_out = (probs.transpose(0, 2, 1) @ Vt).reshape(self.n_heads * self.head_dim)

            # Optional BitNet sub-norm on the attention output (before o_proj)
            sub = f"layers.{layer}.attn.inner_attn_ln"
            if sub in self.t:
                attn_out = self._rms(attn_out, self.t[sub]["data"])

            # Output projection
            x = self._linear(attn_out, f"layers.{layer}.attn.o_proj")

            # Residual
            x = x + residual
            residual = x.copy()

            # --- Post-attention RMSNorm ---
            norm_w2 = self.t[f"layers.{layer}.post_attention_layernorm"]["data"]
            x = self._rms(x, norm_w2)

            # --- SwiGLU MLP ---
            gate = self._linear(x, f"layers.{layer}.mlp.gate_proj")  # (inter,)
            up = self._linear(x, f"layers.{layer}.mlp.up_proj")      # (inter,)
            x = self._silu_mul(gate, up)
            # Optional BitNet sub-norm on the MLP intermediate (before down_proj)
            sub = f"layers.{layer}.mlp.ffn_layernorm"
            if sub in self.t:
                x = self._rms(x, self.t[sub]["data"])
            x = self._linear(x, f"layers.{layer}.mlp.down_proj")     # (hidden,)

            # Residual
            x = x + residual

        # Final RMSNorm
        norm_w = self.t["final_norm"]["data"]
        x = self._rms(x, norm_w)

        # LM Head: use a dedicated lm_head when the model does not tie weights,
        # otherwise reuse the token embedding matrix.
        if "lm_head" in self.t and not self.t["lm_head"].get("tied", False):
            lm_w = self.t["lm_head"]["data"]
        else:
            lm_w = self.t["embed_tokens"]["data"]
        logits = self._fp32_linear(lm_w, x)  # (vocab,)

        self.pos += 1
        self.fed_ids.append(token_id)
        self._last_logits = logits
        return logits

    def _sample(self, logits: np.ndarray, temperature: float, top_p: float) -> int:
        """Sample one token id from logits (greedy when temperature < 0.01)."""
        if temperature < 0.01:
            return int(np.argmax(logits))
        probs = softmax(logits / temperature)
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p) + 1
        top_idx = sorted_idx[:cutoff]
        top_probs = probs[top_idx]
        top_probs = top_probs / top_probs.sum()
        return int(np.random.choice(top_idx, p=top_probs))

    def _encode(self, prompt: str) -> list[int]:
        """Encode, prepending BOS for SentencePiece/LLaMA-style models."""
        ids = self.tokenizer.encode(prompt)
        if getattr(self.tokenizer, "spm", False) and (not ids or ids[0] != self.tokenizer.bos_id):
            ids = [self.tokenizer.bos_id] + ids
        return ids

    def generate(self, prompt: str, max_tokens: int = 64,
                 temperature: float = 0.8, top_p: float = 0.9,
                 verbose: bool = True, display: "str | None" = None) -> str:
        """Generate text from a prompt.

        ``display`` overrides the prompt shown to the user (useful when the
        actual prompt is a long grounded context and we only want to echo the
        original question).
        """
        self.reset()
        shown = prompt if display is None else display

        input_ids = self._encode(prompt)
        if verbose:
            print(f"\n  Prompt tokens: {len(input_ids)}")
            print(f"  Prompt: {shown}")
            print(f"  Generating {max_tokens} tokens...\n")
            print(f"  {'─' * 60}")
            sys.stdout.write(f"  {shown}")
            sys.stdout.flush()

        # Prefill: process all prompt tokens
        t_start = time.time()
        for i, tid in enumerate(input_ids):
            logits = self.forward_token(tid)
        t_prefill = time.time() - t_start

        if verbose:
            prefill_tps = len(input_ids) / t_prefill if t_prefill > 0 else 0
            print(f"\n  [Prefill: {len(input_ids)} tokens in {t_prefill:.2f}s = {prefill_tps:.1f} tok/s]")
            sys.stdout.write("  ")
            sys.stdout.flush()

        # Decode: generate new tokens
        generated_ids = []
        t_decode_start = time.time()

        for step in range(max_tokens):
            # Sample from logits
            next_id = self._sample(logits, temperature, top_p)

            if next_id == self.tokenizer.eos_id:
                if verbose:
                    print(" [EOS]", end="")
                break

            generated_ids.append(next_id)
            token_text = self.tokenizer.decode_token(next_id)
            if verbose:
                sys.stdout.write(token_text)
                sys.stdout.flush()

            # Forward pass for next token
            logits = self.forward_token(next_id)

        t_decode = time.time() - t_decode_start
        t_total = time.time() - t_start

        if verbose:
            decode_tps = len(generated_ids) / t_decode if t_decode > 0 else 0
            print(f"\n  {'─' * 60}")
            print(f"\n  📊 Performance Metrics:")
            print(f"     Prefill:    {len(input_ids)} tokens in {t_prefill:.2f}s ({len(input_ids)/t_prefill:.1f} tok/s)")
            print(f"     Decode:     {len(generated_ids)} tokens in {t_decode:.2f}s ({decode_tps:.2f} tok/s)")
            print(f"     Total:      {t_total:.2f}s")
            print(f"     KV Cache:   {self.pos} positions × {self.n_layers} layers × {self.n_kv_heads} heads × {self.head_dim} dim")
            cache_bytes = self.pos * self.n_layers * self.n_kv_heads * self.head_dim * 4 * 2
            print(f"     Cache Size: {cache_bytes / 1024:.1f} KB")
            print()

        return self.tokenizer.decode(generated_ids)

    def generate_constrained(self, prompt: str, allowed_names, max_name_tokens: int = 6,
                             max_tokens: int = 48, temperature: float = 0.2,
                             top_p: float = 0.9) -> str:
        """Generate a tool call whose *name* is forced into ``allowed_names``.

        Grammar-constrained decoding over the tool-name prefix: at each of the
        first few tokens the logits are masked to the token ids that keep the
        emitted text a valid prefix of some allowed name. Once a name is
        complete, the argument list is generated unconstrained.

        This is the cheapest useful case of constrained decoding and it fixes
        the dominant failure of a small model (choosing the wrong tool or
        emitting free prose instead of a call).
        """
        ids = self._encode(prompt) if prompt else []
        fed = getattr(self, "fed_ids", [])
        if ids and fed and len(ids) >= len(fed) and ids[:len(fed)] == fed:
            new_ids, reused = ids[len(fed):], True
        elif ids:
            self.reset()
            new_ids, reused = ids, False
        else:  # empty prompt: continue from whatever is already cached
            new_ids, reused = [], False
        logits = getattr(self, "_last_logits", None)
        for tid in new_ids:
            logits = self.forward_token(tid)
        if logits is None:
            logits = self.forward_token(self.tokenizer.eos_id)
        if ids:
            self.last_prefill = (len(ids), len(new_ids), reused)

        names = list(allowed_names)
        name = self._constrained_name(names, max_name_tokens, temperature, top_p)

        # --- argument phase: free decoding ---
        generated = []
        logits = self._last_logits
        for _ in range(max_tokens):
            tid = self._sample(logits, temperature, top_p)
            if tid == self.tokenizer.eos_id:
                break
            generated.append(tid)
            logits = self.forward_token(tid)

        return name + self.tokenizer.decode(generated)

    def _name_candidates(self, names):
        """(id, text) pairs whose text is a substring of some allowed name."""
        cache = getattr(self, "_name_cand_cache", None)
        if cache is None:
            cache = self._name_cand_cache = {}
        key = tuple(sorted(names))
        cand = cache.get(key)
        if cand is None:
            cand = [(tid, text) for tid in self.tokenizer.name_token_ids()
                    for text in [self.tokenizer.token_text(tid).lstrip(" ")]
                    if text and any(text in n for n in names)]
            cache[key] = cand
        return cand

    def _constrained_name(self, names, max_name_tokens, temperature, top_p) -> str:
        """Generate a tool name forced into ``names`` (cache ends after it)."""
        cand = self._name_candidates(names)
        logits = self._last_logits
        emitted = ""
        for _ in range(max_name_tokens):
            allowed = [tid for tid, text in cand
                       if any(n.startswith(emitted + text) for n in names)]
            if not allowed:
                break
            masked = np.full_like(logits, -np.inf)
            masked[allowed] = logits[allowed]
            tid = self._sample(masked, temperature, top_p)
            emitted += self.tokenizer.token_text(tid).lstrip(" ")
            logits = self.forward_token(tid)
            if emitted in names:
                return emitted
        for name in names:
            if name.startswith(emitted):
                return name
        return names[0]

    def _constrained_value(self, max_tokens: int, temperature: float, top_p: float,
                           stop: str = '"'):
        """Generate a free string value, stopping at ``stop``; return (text, closed)."""
        logits = self._last_logits
        out = ""
        for _ in range(max_tokens):
            tid = self._sample(logits, temperature, top_p)
            if tid == self.tokenizer.eos_id:
                return out, False
            text = self.tokenizer.token_text(tid)
            logits = self.forward_token(tid)
            if stop and stop in text:
                return out + text.split(stop, 1)[0], True
            out += text
        return out, False

    def generate_tool_call(self, names, schemas, max_name_tokens: int = 6,
                           max_value_tokens: int = 16, temperature: float = 0.2,
                           top_p: float = 0.9) -> str:
        """Generate a *schema-valid* tool call: constrained name + forced args.

        The tool name is grammar-constrained (as in ``generate_constrained``),
        and the argument list is forced to that tool's schema keys in order:
        the structure (``{"key": "``) is emitted literally while each string
        value is generated freely until its closing quote. The result always
        parses to a call with exactly the right keys, which is the other half of
        the reliability problem after the name.

        Assumes the cache already ends at the call site (e.g. after a ``TOOL:``
        cue was fed); call ``feed`` first.
        """
        names = list(names)
        name = self._constrained_name(names, max_name_tokens, temperature, top_p)
        keys = list(schemas.get(name, {}))
        emitted = name + " {"
        self.feed(" {")
        for i, key in enumerate(keys):
            prefix = f'"{key}": "' if i == 0 else f', "{key}": "'
            self.feed(prefix)
            emitted += prefix
            value, closed = self._constrained_value(max_value_tokens, temperature, top_p)
            # The model often keeps writing the JSON inside a value (e.g.
            # "todo.txt, "), so drop trailing separators before the real quote.
            value = value.strip().rstrip(",").strip()
            emitted += value + '"'
            if not closed:
                # The value phase ran out of tokens before a quote; close it.
                self.feed('"')
        self.feed("}")
        emitted += "}"
        return emitted

    def generate_continue(self, prompt: str, max_tokens: int = 64,
                          temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Like ``generate`` but reuse the KV cache across calls.

        The agent loop keeps extending one prompt, so re-prefilling it from
        scratch at every step is O(n^2). Here only the tokens past what is
        already cached are fed. If the new tokenization does not extend the
        cached prefix exactly (a rare BPE boundary shift), we fall back to a
        full prefill so the output stays numerically identical.

        ``self.last_prefill`` records ``(total, newly_fed, reused)``.
        """
        ids = self._encode(prompt)
        fed = getattr(self, "fed_ids", [])
        if fed and len(ids) >= len(fed) and ids[:len(fed)] == fed:
            new_ids, reused = ids[len(fed):], True
        else:
            self.reset()
            new_ids, reused = ids, False

        logits = getattr(self, "_last_logits", None)
        for tid in new_ids:
            logits = self.forward_token(tid)
        if logits is None:
            logits = self.forward_token(self.tokenizer.eos_id)

        generated = []
        for _ in range(max_tokens):
            next_id = self._sample(logits, temperature, top_p)
            if next_id == self.tokenizer.eos_id:
                break
            generated.append(next_id)
            logits = self.forward_token(next_id)

        self.last_prefill = (len(ids), len(new_ids), reused)
        return self.tokenizer.decode(generated)

    def feed(self, text: str, add_bos: bool = False) -> int:
        """Feed text into the KV cache without generating; return tokens fed.

        This is the cooperative half of cache reuse: the agent loop appends its
        observation text here instead of rebuilding and re-encoding the whole
        prompt, so only the new tokens are prefilled.
        """
        ids = self.tokenizer.encode(text)
        if add_bos and getattr(self.tokenizer, "spm", False):
            ids = [self.tokenizer.bos_id] + ids
        for tid in ids:
            self.forward_token(tid)
        self.last_feed = len(ids)
        return len(ids)

    def generate_next(self, max_tokens: int = 64, temperature: float = 0.8,
                      top_p: float = 0.9) -> str:
        """Generate from the current cache state (used after ``feed``)."""
        logits = getattr(self, "_last_logits", None)
        if logits is None:
            logits = self.forward_token(self.tokenizer.eos_id)
        generated = []
        for _ in range(max_tokens):
            next_id = self._sample(logits, temperature, top_p)
            if next_id == self.tokenizer.eos_id:
                break
            generated.append(next_id)
            logits = self.forward_token(next_id)
        return self.tokenizer.decode(generated)

    def save_state(self, path: str) -> None:
        """Persist the KV cache and position so a later run can skip prefill.

        A fixed prompt prefix (the agent's tool preamble + few-shot) costs tens
        of seconds to prefill on this hardware. The cache is a pure function of
        the token ids and the weights, so it is safe to serialize and replay.
        """
        arrays = {
            "pos": np.array(self.pos, dtype=np.int64),
            "fed_ids": np.asarray(self.fed_ids, dtype=np.int32),
        }
        if self._last_logits is not None:
            arrays["logits"] = np.asarray(self._last_logits, dtype=np.float32)
        for i, cache in enumerate(self.kv_cache):
            if cache["k"] is not None:
                arrays[f"k{i}"] = cache["k"]
                arrays[f"v{i}"] = cache["v"]
        np.savez(path, **arrays)

    def load_state(self, path: str) -> bool:
        """Restore a cache written by ``save_state``; returns False if absent."""
        if not os.path.exists(path):
            return False
        with np.load(path, allow_pickle=False) as z:
            pos = int(z["pos"])
            fed_ids = [int(x) for x in z["fed_ids"]]
            kv_cache = [{"k": None, "v": None} for _ in range(self.n_layers)]
            for i in range(self.n_layers):
                if f"k{i}" in z:
                    kv_cache[i] = {"k": z[f"k{i}"], "v": z[f"v{i}"]}
            logits = z["logits"] if "logits" in z else None
        self.kv_cache = kv_cache
        self.pos = pos
        self.fed_ids = fed_ids
        self._last_logits = logits
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def resolve_compact(style: str, constrained: bool, constrained_args: bool) -> bool:
    """Whether the agent should use the compact prompt prefix.

    Prompt size is tied to the decoding mode because that is what the
    measurement supports: with a constrained or schema-forced call the compact
    prefix matches the full one (9/10 vs 9/10 tool-name accuracy) at roughly a
    third of the prefill, while free generation still needs the long few-shot
    (5/8 vs 1/8). See ROADMAP section 7.7.
    """
    if style == "auto":
        return bool(constrained or constrained_args)
    return style == "compact"


def main():
    parser = argparse.ArgumentParser(description="SmolLM2-135M NanoLang Inference")
    parser.add_argument("--model", default="models/smollm2/smollm2-135m.nanow",
                        help="Path to .nanow model file")
    parser.add_argument("--tokenizer", default="models/smollm2/tokenizer.json",
                        help="Path to tokenizer.json")
    parser.add_argument("--config", default="models/smollm2/config.json",
                        help="Path to Hugging Face config.json (for head counts)")
    parser.add_argument("--prompt", default="The meaning of life is",
                        help="Input prompt")
    parser.add_argument("--tokens", type=int, default=64,
                        help="Maximum tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling threshold")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-kernel", action="store_true",
                        help="Disable the compiled phase0 C kernel (use NumPy fallback)")
    parser.add_argument("--online", action="store_true",
                        help="Retrieve live web context and ground the answer in it")
    parser.add_argument("--ctx-chars", type=int, default=900,
                        help="Max characters of retrieved web context when --online")
    parser.add_argument("--agent", action="store_true",
                        help="Run the tool-use agent loop (file CRUD, bash, HTTP, search)")
    parser.add_argument("--workspace", default="nanow_workspace",
                        help="Sandbox root for every file/bash tool (default: nanow_workspace)")
    parser.add_argument("--allow-bash", action="store_true",
                        help="Permit the run_bash tool (screen by deny-list)")
    parser.add_argument("--max-steps", type=int, default=3,
                        help="Maximum tool-use steps in --agent mode")
    parser.add_argument("--step-tokens", type=int, default=48,
                        help="Tokens generated per agent step")
    parser.add_argument("--prime-tool", action="store_true",
                        help="Append a 'TOOL:' cue each step (nudges weak models into the call grammar)")
    parser.add_argument("--constrained", action="store_true",
                        help="Grammar-constrain the tool name via logit masking (needs the C kernel path)")
    parser.add_argument("--constrained-args", action="store_true",
                        help="Force the argument list to the tool schema (name + keys + string values)")
    parser.add_argument("--planner", choices=["model", "route", "hybrid"], default="model",
                        help="Who chooses the tool: model | route (deterministic) | hybrid")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable KV-cache reuse across agent steps")
    parser.add_argument("--ew", choices=["auto", "all", "off"], default="auto",
                        help="Elementwise kernels: auto/all = C (RMSNorm/RoPE/SiLU/softmax), off = NumPy reference")
    parser.add_argument("--kv-cache-dir", default=None,
                        help="Persist the static prompt-prefix KV cache here to skip its prefill on later runs")
    parser.add_argument("--prompt-style", choices=["auto", "full", "compact"], default="auto",
                        help="Agent preamble size: auto = compact under constrained decoding, full otherwise")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 70)
    print(" NANOLANG INFERENCE ENGINE — SmolLM2-135M (Ternary)")
    print("=" * 70)

    # Load model
    print("\n  📦 Loading .nanow model...", flush=True)
    t0 = time.time()
    tensors, config, mm, fd = load_nanow(args.model)
    t_load = time.time() - t0
    # Every tensor is copied out of the mapping, so release the file mapping
    # (frees ~350 MB of resident pages for bitnet700m).
    try:
        mm.close()
    except Exception:
        pass
    os.close(fd)

    # The .nanow header carries SSM state_dim in the head slot; the true
    # attention geometry lives in the Hugging Face config.json.
    with open(args.config) as f:
        hf_cfg = json.load(f)
    config["n_heads"] = hf_cfg["num_attention_heads"]
    config["n_kv_heads"] = hf_cfg["num_key_value_heads"]
    # RoPE frequency base and norm epsilon differ per model family
    # (SmolLM2 100000/1e-5, BitNet LLaMA 10000/1e-5).
    config["rope_theta"] = hf_cfg.get("rope_theta", 100000.0)
    config["rms_norm_eps"] = hf_cfg.get("rms_norm_eps", 1e-5)
    config["max_position_embeddings"] = hf_cfg.get("max_position_embeddings", 2048)

    print(f"     Loaded {len(tensors)} tensors in {t_load:.2f}s")
    print(f"     Config: {config}")

    # Count ternary vs fp32
    n_ternary = sum(1 for t in tensors.values() if t["ternary"])
    n_fp32 = sum(1 for t in tensors.values() if not t["ternary"])
    print(f"     Ternary tensors: {n_ternary}")
    print(f"     FP32 tensors:    {n_fp32}")

    # Load tokenizer
    print("\n  📝 Loading BPE tokenizer...", flush=True)
    tokenizer = BPETokenizer(args.tokenizer)
    print(f"     Vocab size: {len(tokenizer.vocab)}")

    # Load the compiled phase0 ternary C kernel (AVX2/NEON add-sub GEMV).
    kernel = None
    if not args.no_kernel:
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "phase0", "bin", "libternary.so")
        if os.path.exists(lib_path):
            try:
                kernel = TernaryKernel(lib_path)
                print("\n  ⚡ Ternary C kernel:  ACTIVE (phase0 sign-add/sub GEMV)")
            except OSError as exc:
                print(f"\n  ⚡ Ternary C kernel:  unavailable ({exc})")
        else:
            print("\n  ⚡ Ternary C kernel:  not built (run: make -C phase0 lib)")

    # Create runner
    runner = SmolLM2Runner(tensors, config, tokenizer, kernel=kernel)
    runner.model_tag = args.model
    if args.ew == "off":
        runner.ew_ops = set()

    compact = resolve_compact(args.prompt_style, args.constrained,
                              args.constrained_args)

    # --- Agent mode: tool-use loop (file CRUD, bash, HTTP, web search) ---
    if args.agent:
        retriever = WebRetriever() if args.online else None
        search = (lambda q: retriever.ground(q)[0]) if retriever else None
        registry = ToolRegistry(root=args.workspace, allow_bash=args.allow_bash,
                                search=search)
        print(f"\n  🤖 Agent mode: workspace={registry.root}")
        print(f"     Bash: {'enabled' if args.allow_bash else 'disabled'}   "
              f"Search: {'enabled' if search else 'disabled'}")
        print(f"     Planner: {args.planner}   Cache reuse: "
              f"{'off' if args.no_cache else 'on'}")
        print(f"     Constrained: {'args' if args.constrained_args else ('name' if args.constrained else 'off')}")
        print(f"     Tools: {', '.join(registry.names)}", flush=True)
        print(f"     Prompt: {'compact' if compact else 'full'}"
              f"   Prefix cache: {args.kv_cache_dir or 'off'}", flush=True)

        outcome = run_agent(runner, registry, args.prompt, max_steps=args.max_steps,
                            step_tokens=args.step_tokens, temperature=args.temp,
                            top_p=args.top_p, prime=args.prime_tool,
                            constrained=args.constrained,
                            constrained_args=args.constrained_args,
                            planner=args.planner,
                            reuse_cache=not args.no_cache,
                            kv_cache_dir=args.kv_cache_dir,
                            compact=compact)

        print("\n" + "=" * 70)
        print(" AGENT COMPLETE")
        print("=" * 70)
        if outcome["calls"]:
            print(" Tool calls:")
            for call in outcome["calls"]:
                print(f"   - {call['name']}({call['args']}) ok={call['ok']}")
        else:
            print(" No tool calls were made.")
        print(f"\n Final answer:\n {outcome['answer'] or '(none)'}")
        return

    # Generate
    display_prompt = args.prompt
    if args.online:
        print("\n  🌐 Online mode: retrieving live web context...", flush=True)
        retriever = WebRetriever()
        t0 = time.time()
        context, sources = retriever.ground(args.prompt, max_chars=args.ctx_chars)
        dt = time.time() - t0
        if context:
            preview = context[:200].replace("\n", " ")
            print(f"     Retrieved {len(context)} chars in {dt:.2f}s")
            print(f"     Sources: {', '.join(sources)}")
            print(f"     Context: {preview}…")
            prompt = (f"Context:\n{context}\n\n"
                      f"Question: {args.prompt}\nAnswer:")
        else:
            print(f"     No context retrieved ({', '.join(sources)}); using model memory.")
            prompt = args.prompt
    else:
        prompt = args.prompt

    print(f"\n  🚀 Generating text (temp={args.temp}, top_p={args.top_p})...")
    result = runner.generate(
        prompt=prompt,
        max_tokens=args.tokens,
        temperature=args.temp,
        top_p=args.top_p,
        display=display_prompt,
    )

    print("=" * 70)
    print(" GENERATION COMPLETE")
    print("=" * 70)

    # The mapping was already released right after loading (all tensors copied),
    # so there is nothing to close here.


if __name__ == "__main__":
    main()
