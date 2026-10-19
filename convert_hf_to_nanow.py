"""Hugging Face → NanoLang .nanow converter (general LLaMA/BitNet).

Reads an HF checkpoint (BF16 / F16 / F32 safetensors) plus its config.json and
writes a page-aligned, bit-packed .nanow container for zero-copy mmap loading.

Linear projections (q/k/v/o_proj, gate/up/down_proj) are ternarized to BitNet
b1.58 {-1, 0, +1} with a per-tensor absmean scale. Norms, the token embedding,
and an untied lm_head stay in FP32 because BitNet keeps those in higher
precision. Sub-norms used by some BitNet releases (attn.inner_attn_ln,
mlp.ffn_layernorm) are passed through unchanged.

Usage:
    python3 convert_hf_to_nanow.py \
        --model-dir models/bitnet700m \
        --output   models/bitnet700m/bitnet700m.nanow
"""

from __future__ import annotations
import argparse
import json
import os
import struct
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase1", "python"))
from export_to_nanow import (pack_ternary_2bit, NANOW_MAGIC, NANOW_VERSION,
                             PAGE_ALIGNMENT, PACKING_2BIT, PACKING_RAW_INT8,
                             DTYPE_TERNARY, DTYPE_FP32)

# Linear projections that get ternarized (short names used by run_smollm2.py).
TERNARY_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# safetensors reader (BF16 / F16 / F32, no torch dependency)
# ---------------------------------------------------------------------------

def read_safetensors(path: str):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(header_len).decode())
    return meta, 8 + header_len


def read_tensor(path: str, meta: dict, name: str, data_offset: int) -> np.ndarray:
    info = meta[name]
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_offset + start)
        raw = f.read(end - start)
    dtype = info["dtype"]
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        return ((u16.astype(np.uint32) << 16).view(np.float32)).reshape(info["shape"])
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(info["shape"])
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])
    raise ValueError(f"Unsupported dtype {dtype} for {name}")


# ---------------------------------------------------------------------------
# Ternarization (BitNet b1.58 absmean, matches 1bitLLM weight_quant)
# ---------------------------------------------------------------------------

def ternarize(W: np.ndarray) -> tuple[np.ndarray, float]:
    """W_eff = round(W / mean|W|).clamp(-1, 1) * mean|W|; returns codes + scale."""
    gamma = float(np.abs(W).mean()) + 1e-7
    Wt = np.clip(np.round(W / gamma), -1.0, 1.0).astype(np.int8)
    return Wt, gamma


# ---------------------------------------------------------------------------
# Container writer
# ---------------------------------------------------------------------------

HEADER_FMT = "<IIII IIII QQ I 3I"
DESC_FMT = "<64s III 4I Q Q Q f"


def add_tensor(tensors: list, name: str, data: np.ndarray, ternary: bool, scale: float = 1.0):
    if ternary:
        payload = pack_ternary_2bit(data)
        dtype, packing = DTYPE_TERNARY, PACKING_2BIT
    else:
        payload = data.astype(np.float32).tobytes()
        dtype, packing = DTYPE_FP32, PACKING_RAW_INT8
    tensors.append({
        "name": name.encode()[:63],
        "dtype": dtype,
        "packing": packing,
        "shape": list(data.shape) + [1] * (4 - len(data.shape)),
        "num_elements": int(data.size),
        "scale": float(scale),
        "payload": payload,
    })


def write_container(output_path: str, tensors: list, layers: int, hidden: int, vocab: int):
    num_tensors = len(tensors)
    header_size = struct.calcsize(HEADER_FMT)
    desc_size = struct.calcsize(DESC_FMT)
    directory_size = header_size + num_tensors * desc_size
    initial_pad = (PAGE_ALIGNMENT - (directory_size % PAGE_ALIGNMENT)) % PAGE_ALIGNMENT
    current = directory_size + initial_pad

    total_params = 0
    for t in tensors:
        payload_len = len(t["payload"])
        t["offset"] = current
        t["size"] = payload_len
        t["pad"] = (PAGE_ALIGNMENT - (payload_len % PAGE_ALIGNMENT)) % PAGE_ALIGNMENT
        current += payload_len + t["pad"]
        total_params += t["num_elements"]
    total_file_size = current

    header = struct.pack(HEADER_FMT, NANOW_MAGIC, NANOW_VERSION, 0, num_tensors,
                         layers, hidden, 16, vocab, total_params, total_file_size,
                         directory_size, 0, 0, 0)
    with open(output_path, "wb") as f:
        f.write(header)
        for t in tensors:
            f.write(struct.pack(DESC_FMT, t["name"].ljust(64, b"\x00"), t["dtype"],
                                t["packing"], len([s for s in t["shape"] if s > 1]),
                                *t["shape"], t["num_elements"], t["offset"],
                                t["size"], t["scale"]))
        if initial_pad:
            f.write(b"\x00" * initial_pad)
        for t in tensors:
            f.write(t["payload"])
            if t["pad"]:
                f.write(b"\x00" * t["pad"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def short_name(hf_name: str) -> str | None:
    """Map an HF tensor name to the runner's short name, or None to skip."""
    n = hf_name
    if n.endswith(".weight"):
        n = n[:-len(".weight")]
    # Rotary-embedding frequency buffers are recomputed by the runner.
    if "rotary_emb" in n or n.endswith("inv_freq"):
        return None
    if n == "model.embed_tokens":
        return "embed_tokens"
    if n == "model.norm":
        return "final_norm"
    if n == "lm_head":
        return "lm_head"
    if n.startswith("model.layers."):
        parts = n.split(".")
        layer = parts[2]
        rest = ".".join(parts[3:])
        rest = rest.replace("self_attn.", "attn.")
        return f"layers.{layer}.{rest}"
    return None


def convert(model_dir: str, output_path: str, tie_embeddings: bool | None = None):
    cfg_path = os.path.join(model_dir, "config.json")
    st_path = os.path.join(model_dir, "model.safetensors")
    with open(cfg_path) as f:
        cfg = json.load(f)

    hidden = cfg["hidden_size"]
    layers = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]
    tie = cfg.get("tie_word_embeddings", True) if tie_embeddings is None else tie_embeddings

    print("=" * 70)
    print(" HF → NANOLANG .nanow CONVERTER")
    print("=" * 70)
    print(f"  Model:    {model_dir}")
    print(f"  Hidden:   {hidden}  Layers: {layers}  Vocab: {vocab}  Tied: {tie}")

    meta, data_off = read_safetensors(st_path)
    keys = [k for k in meta if k != "__metadata__"]

    t_start = time.time()
    tensors: list = []
    total = ternary_n = fp32_n = 0

    # Embedding + final norm first
    for hf_name, is_ternary in [("model.embed_tokens.weight", False), ("model.norm.weight", False)]:
        if hf_name in meta:
            data = read_tensor(st_path, meta, hf_name, data_off)
            add_tensor(tensors, short_name(hf_name), data, ternary=False)
            total += data.size
            fp32_n += data.size

    # Untied lm_head (kept FP32)
    if not tie and "lm_head.weight" in meta:
        data = read_tensor(st_path, meta, "lm_head.weight", data_off)
        add_tensor(tensors, "lm_head", data, ternary=False)
        total += data.size
        fp32_n += data.size

    # Per-layer tensors, in stable order
    for l in range(layers):
        prefix = f"model.layers.{l}."
        for hf_key in sorted(k for k in keys if k.startswith(prefix)):
            short = short_name(hf_key)
            if short is None:
                continue
            leaf = hf_key.split(".")[-2]  # e.g. input_layernorm, q_proj, inner_attn_ln
            is_ternary = leaf in TERNARY_SUFFIXES
            data = read_tensor(st_path, meta, hf_key, data_off)
            if is_ternary:
                codes, gamma = ternarize(data)
                add_tensor(tensors, short, codes, ternary=True, scale=gamma)
                ternary_n += data.size
            else:
                add_tensor(tensors, short, data, ternary=False)
                fp32_n += data.size
            total += data.size
        if (l + 1) % 8 == 0:
            print(f"    ... layer {l + 1}/{layers}", flush=True)

    write_container(output_path, tensors, layers, hidden, vocab)

    file_mb = os.path.getsize(output_path) / (1024 * 1024)
    print()
    print("=" * 70)
    print(" CONVERSION COMPLETE")
    print("=" * 70)
    print(f"  Tensors:            {len(tensors)}")
    print(f"  Total parameters:   {total:,}")
    print(f"  Ternary (1.58-bit): {ternary_n:,} ({100 * ternary_n / total:.1f}%)")
    print(f"  FP32 (kept):        {fp32_n:,} ({100 * fp32_n / total:.1f}%)")
    print(f"  .nanow size:        {file_mb:.1f} MB")
    print(f"  Elapsed:            {time.time() - t_start:.1f}s")
    print(f"  Output:             {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HF → .nanow converter")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tie-embeddings", default=None, action="store_true")
    args = parser.parse_args()
    convert(args.model_dir, args.output,
            tie_embeddings=True if args.tie_embeddings else None)
