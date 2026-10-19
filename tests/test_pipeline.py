"""Lightweight regression tests for the NanoLang conversion/inference pipeline.

These avoid loading full models (memory is tight), and instead exercise:
  * BitNet b1.58 per-tensor ternarization
  * ternary bit-packing round-trip through the exporter
  * BPE tokenizer auto-detection (GPT-2 byte-level vs SentencePiece)
  * HF tensor-name mapping used by the converter

Run:
    python3 -m unittest tests.test_pipeline -v
"""

from __future__ import annotations
import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "phase1", "python"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load("run_smollm2", os.path.join(REPO, "run_smollm2.py"))
conv = _load("convert_hf_to_nanow", os.path.join(REPO, "convert_hf_to_nanow.py"))
from export_to_nanow import pack_ternary_2bit, unpack_ternary_2bit  # noqa: E402


class TestTernarization(unittest.TestCase):
    def test_scale_is_absmean(self):
        rng = np.random.default_rng(0)
        W = rng.normal(0, 0.1, size=(64, 128)).astype(np.float32)
        codes, gamma = conv.ternarize(W)
        self.assertAlmostEqual(gamma, float(np.abs(W).mean()), places=6)

    def test_codes_are_ternary(self):
        rng = np.random.default_rng(1)
        W = rng.normal(0, 0.1, size=(32, 32)).astype(np.float32)
        codes, _ = conv.ternarize(W)
        self.assertTrue(set(np.unique(codes).tolist()).issubset({-1, 0, 1}))

    def test_reconstruction_beats_zero(self):
        rng = np.random.default_rng(2)
        W = rng.normal(0, 0.1, size=(64, 64)).astype(np.float32)
        codes, gamma = conv.ternarize(W)
        err = np.linalg.norm(W - codes.astype(np.float32) * gamma) / np.linalg.norm(W)
        # Ternary PTQ is inherently lossy; just assert it is a real approximation
        # (< 1.0) and not accidental identity or collapse.
        self.assertLess(err, 0.9)
        self.assertGreater(err, 0.01)


class TestTernaryPacking(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(3)
        arr = rng.integers(-1, 2, size=1000).astype(np.int8)
        packed = pack_ternary_2bit(arr)
        out = unpack_ternary_2bit(packed, arr.size)
        self.assertTrue(np.array_equal(arr, out))

    def test_loader_unpack_matches_exporter(self):
        """run_smollm2's loader must decode the same code map as the exporter."""
        rng = np.random.default_rng(4)
        arr = rng.integers(-1, 2, size=256).astype(np.int8)
        packed = pack_ternary_2bit(arr)
        by_exporter = np.asarray(unpack_ternary_2bit(packed, arr.size), dtype=np.int8)
        # Reproduce the loader's branch: codes 0->0, 1->+1, 3->-1
        buf = np.frombuffer(packed, dtype=np.uint8)
        vals = [((buf >> (s * 2)) & 0x03) for s in range(4)]
        flat = np.column_stack(vals).ravel()[: arr.size].astype(np.int8)
        by_loader = np.where(flat == 1, 1, np.where(flat == 3, -1, 0)).astype(np.int8)
        self.assertTrue(np.array_equal(by_exporter, by_loader))


class TestTensorNameMapping(unittest.TestCase):
    def test_known_names(self):
        self.assertEqual(conv.short_name("model.embed_tokens.weight"), "embed_tokens")
        self.assertEqual(conv.short_name("model.norm.weight"), "final_norm")
        self.assertEqual(
            conv.short_name("model.layers.3.self_attn.q_proj.weight"),
            "layers.3.attn.q_proj")
        self.assertEqual(
            conv.short_name("model.layers.3.self_attn.inner_attn_ln.weight"),
            "layers.3.attn.inner_attn_ln")
        self.assertEqual(
            conv.short_name("model.layers.3.mlp.ffn_layernorm.weight"),
            "layers.3.mlp.ffn_layernorm")

    def test_ternary_suffix_selection(self):
        self.assertIn("q_proj", conv.TERNARY_SUFFIXES)
        self.assertIn("down_proj", conv.TERNARY_SUFFIXES)
        self.assertNotIn("inner_attn_ln", conv.TERNARY_SUFFIXES)
        self.assertNotIn("ffn_layernorm", conv.TERNARY_SUFFIXES)


class TestTernaryKernel(unittest.TestCase):
    LIB = os.path.join(REPO, "phase0", "bin", "libternary.so")

    @unittest.skipUnless(os.path.exists(LIB), "phase0 libternary.so not built")
    def test_kernel_matches_reference(self):
        ker = run.TernaryKernel(self.LIB)
        rng = np.random.default_rng(7)
        for M, K in [(8, 7), (64, 96), (128, 512)]:
            W = rng.integers(-1, 2, size=(M, K)).astype(np.int8)
            x = rng.normal(0, 1, size=K).astype(np.float32)
            scale = 0.137
            got = ker.gemv(W, x, scale)
            amax = float(np.abs(x).max())
            s = 127.0 / amax
            xq = np.clip(np.rint(x * s), -127, 127).astype(np.int8)
            ref = (W.astype(np.int32) @ xq.astype(np.int32)).astype(np.float32) * (scale / s)
            np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(os.path.exists(LIB), "phase0 libternary.so not built")
    def test_kernel_handles_zero_activation(self):
        ker = run.TernaryKernel(self.LIB)
        W = np.ones((4, 8), dtype=np.int8)
        got = ker.gemv(W, np.zeros(8, dtype=np.float32), 0.5)
        self.assertTrue(np.array_equal(got, np.zeros(4, dtype=np.float32)))


class TestPackedKernel(unittest.TestCase):
    LIB = os.path.join(REPO, "phase0", "bin", "libternary.so")

    def test_loader_unpack_helper_matches_exporter(self):
        rng = np.random.default_rng(11)
        arr = rng.integers(-1, 2, size=(24, 8)).astype(np.int8)
        packed = np.frombuffer(pack_ternary_2bit(arr), dtype=np.uint8)
        got = run.unpack_ternary_2bit(packed, arr.size, arr.shape)
        self.assertTrue(np.array_equal(got, arr))

    @unittest.skipUnless(os.path.exists(LIB), "phase0 libternary.so not built")
    def test_packed_kernel_matches_int8_kernel(self):
        ker = run.TernaryKernel(self.LIB)
        if not getattr(ker, "has_packed", False):
            self.skipTest("packed kernel symbol not present")
        rng = np.random.default_rng(12)
        for M, K in [(8, 8), (64, 96), (192, 192)]:
            W = rng.integers(-1, 2, size=(M, K)).astype(np.int8)
            packed = np.frombuffer(pack_ternary_2bit(W), dtype=np.uint8).reshape(M, K // 4)
            x = rng.normal(0, 1, size=K).astype(np.float32)
            y_packed = ker.gemv_packed(packed, x, 0.2, M, K)
            y_int8 = ker.gemv(W, x, 0.2)
            np.testing.assert_allclose(y_packed, y_int8, rtol=1e-5, atol=1e-4)

    @unittest.skipUnless(os.path.exists(LIB), "phase0 libternary.so not built")
    def test_packed_kernel_zero_activation(self):
        ker = run.TernaryKernel(self.LIB)
        if not getattr(ker, "has_packed", False):
            self.skipTest("packed kernel symbol not present")
        W = np.ones((4, 8), dtype=np.int8)
        packed = np.frombuffer(pack_ternary_2bit(W), dtype=np.uint8).reshape(4, 2)
        got = ker.gemv_packed(packed, np.zeros(8, dtype=np.float32), 0.5, 4, 8)
        self.assertTrue(np.array_equal(got, np.zeros(4, dtype=np.float32)))


class TestFP32Gemv(unittest.TestCase):
    LIB = os.path.join(REPO, "phase0", "bin", "libternary.so")

    @unittest.skipUnless(os.path.exists(LIB), "phase0 libternary.so not built")
    def test_fp32_gemv_matches_numpy(self):
        ker = run.TernaryKernel(self.LIB)
        if not getattr(ker, "has_fp32gemv", False):
            self.skipTest("fp32_gemv_par symbol not present")
        rng = np.random.default_rng(13)
        for M, K in [(8, 16), (128, 96), (1024, 512)]:
            W = rng.normal(0, 0.1, size=(M, K)).astype(np.float32)
            x = rng.normal(0, 1, size=K).astype(np.float32)
            got = ker.fp32_gemv(W, x)
            np.testing.assert_allclose(got, W @ x, rtol=1e-4, atol=1e-4)


class TestTokenizer(unittest.TestCase):
    SPM = os.path.join(REPO, "models/bitnet700m/tokenizer.json")
    GPT2 = os.path.join(REPO, "models/smollm2/tokenizer.json")

    @unittest.skipUnless(os.path.exists(SPM), "BitNet tokenizer not downloaded")
    def test_sentencepiece_detection_and_ids(self):
        tok = run.BPETokenizer(self.SPM)
        self.assertTrue(tok.spm)
        ids = tok.encode("The capital of France is")
        self.assertEqual(ids, [450, 7483, 310, 3444, 338])
        self.assertIn("France", tok.decode(ids))

    @unittest.skipUnless(os.path.exists(GPT2), "SmolLM2 tokenizer not downloaded")
    def test_gpt2_detection_and_roundtrip(self):
        tok = run.BPETokenizer(self.GPT2)
        self.assertFalse(tok.spm)
        text = "def fibonacci(n):"
        self.assertEqual(tok.decode(tok.encode(text)), text)


class TestWebRetriever(unittest.TestCase):
    """Online-mode parsing, exercised offline via fixture HTML/JSON."""

    DDG_HTML = (
        '<a class="result__a" href="x">Best LLMs 2026</a>\n'
        '<a class="result__snippet">ChatGPT, Claude, Gemini and Grok lead.</a>\n'
        '<a class="result__a" href="y">Second &amp; result</a>\n'
        '<a class="result__snippet">More &lt;useful&gt; text here.</a>\n'
    )

    def test_strip_removes_tags_and_unescapes(self):
        self.assertEqual(
            run.WebRetriever._strip("<b>GPU</b> &amp; CPU &lt;ok&gt;"),
            "GPU & CPU <ok>")

    def test_pairs_extraction(self):
        pairs = run.WebRetriever._pairs(
            self.DDG_HTML, r'class="result__a"[^>]*>(.*?)</a>',
            r'class="result__snippet"[^>]*>(.*?)</a>', 5)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(
            pairs[0], ("Best LLMs 2026", "ChatGPT, Claude, Gemini and Grok lead."))
        self.assertEqual(pairs[1], ("Second & result", "More <useful> text here."))

    def test_duckduckgo_parses_fixture(self):
        retriever = run.WebRetriever()
        retriever._get = lambda url: self.DDG_HTML  # no network
        out = retriever.duckduckgo("anything", n=5)
        self.assertEqual(
            out[0], "Best LLMs 2026: ChatGPT, Claude, Gemini and Grok lead.")
        self.assertEqual(len(out), 2)

    def test_ground_combines_wikipedia_and_duckduckgo(self):
        import json as _json
        opensearch = _json.dumps(["q", ["Large language model"], [""], [""]])
        extract = _json.dumps(
            {"query": {"pages": {"1": {"extract": "An LLM is a neural network."}}}})

        def fake_get(url):
            if "opensearch" in url:
                return opensearch
            if "prop=extracts" in url:
                return extract
            return self.DDG_HTML

        retriever = run.WebRetriever()
        retriever._get = fake_get
        ctx, srcs = retriever.ground("large language model")
        self.assertIn("An LLM is a neural network.", ctx)
        self.assertIn("Best LLMs 2026", ctx)
        self.assertTrue(any(s.startswith("wikipedia:") for s in srcs))
        self.assertIn("duckduckgo:html", srcs)

    def test_ground_survives_network_failure(self):
        def boom(url):
            raise OSError("no network")

        retriever = run.WebRetriever()
        retriever._get = boom
        ctx, srcs = retriever.ground("anything")
        self.assertEqual(ctx, "")
        self.assertTrue(any("error" in s for s in srcs))


class TestKVPersistence(unittest.TestCase):
    """The cross-request prefix cache must round-trip exactly."""

    def _stub(self, n_layers=2, seq=3, heads=2, hd=4):
        r = object.__new__(run.SmolLM2Runner)
        r.n_layers = n_layers
        r.pos = seq
        r.fed_ids = [1, 5, 9][:seq]
        r.kv_cache = [
            {"k": np.arange(seq * heads * hd, dtype=np.float32).reshape(seq, heads, hd) + i,
             "v": np.ones((seq, heads, hd), np.float32) * (i + 1)}
            for i in range(n_layers)
        ]
        r._last_logits = np.arange(7, dtype=np.float32)
        return r

    def test_round_trip(self):
        src = self._stub()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prefix.npz")
            src.save_state(path)
            dst = object.__new__(run.SmolLM2Runner)
            dst.n_layers = src.n_layers
            self.assertTrue(dst.load_state(path))
        self.assertEqual(dst.pos, src.pos)
        self.assertEqual(dst.fed_ids, src.fed_ids)
        np.testing.assert_allclose(dst._last_logits, src._last_logits)
        for got, want in zip(dst.kv_cache, src.kv_cache):
            np.testing.assert_allclose(got["k"], want["k"])
            np.testing.assert_allclose(got["v"], want["v"])

    def test_missing_file_returns_false(self):
        stub = object.__new__(run.SmolLM2Runner)
        stub.n_layers = 1
        self.assertFalse(stub.load_state("/nonexistent/nanow-prefix.npz"))


_LIB = os.path.join(REPO, "phase0", "bin", "libternary.so")


@unittest.skipUnless(os.path.exists(_LIB), "libternary.so not built")
class TestElementwiseC(unittest.TestCase):
    """The C elementwise kernels must match the NumPy reference."""

    def setUp(self):
        self.k = run.TernaryKernel(_LIB)
        if not getattr(self.k, "has_ew", False):
            self.skipTest("libternary.so lacks the elementwise kernels")
        self.rng = np.random.default_rng(0)

    def test_rmsnorm_matches_numpy(self):
        x = self.rng.standard_normal(96).astype(np.float32)
        w = self.rng.standard_normal(96).astype(np.float32)
        np.testing.assert_allclose(self.k.rmsnorm(x, w, 1e-5),
                                   run.rms_norm(x, w, 1e-5), rtol=1e-5, atol=1e-5)

    def test_silu_mul_matches_numpy(self):
        g = (self.rng.standard_normal(64).astype(np.float32) * 5)
        u = self.rng.standard_normal(64).astype(np.float32)
        np.testing.assert_allclose(self.k.silu_mul(g, u), run.silu(g) * u,
                                   rtol=1e-5, atol=1e-6)

    def test_rope_matches_numpy(self):
        x = self.rng.standard_normal((4, 96)).astype(np.float32)
        ref = x.copy()
        cos = self.rng.standard_normal(48).astype(np.float32)
        sin = self.rng.standard_normal(48).astype(np.float32)
        self.k.rope_inplace(x, cos, sin, 4, 96)
        np.testing.assert_allclose(x, run.apply_rope(ref, cos, sin),
                                   rtol=1e-5, atol=1e-6)

    def test_softmax_axis1_matches_numpy(self):
        scores = self.rng.standard_normal((3, 17, 2)).astype(np.float32)
        got = scores.copy()
        self.k.softmax_axis1(got)
        np.testing.assert_allclose(got, run.softmax(scores, axis=1),
                                   rtol=1e-5, atol=1e-6)

    def test_silu_mul_matches_across_a_wide_range(self):
        """Guards the SIMD exp polynomial across the whole clip window."""
        g = np.linspace(-30.0, 30.0, 513, dtype=np.float32)
        up = np.ones_like(g)
        np.testing.assert_allclose(self.k.silu_mul(g, up), run.silu(g) * up,
                                   rtol=1e-5, atol=1e-6)

    def test_softmax_matches_across_a_wide_range(self):
        scores = np.linspace(-60.0, 0.0, 257, dtype=np.float32).reshape(1, 257, 1)
        got = scores.copy()
        self.k.softmax_axis1(got)
        np.testing.assert_allclose(got, run.softmax(scores, axis=1),
                                   rtol=1e-5, atol=1e-7)


class FakeEW:
    """Records which elementwise ops the runner routes to C."""

    has_ew = True
    calls: list = []

    def rmsnorm(self, x, w, eps):
        FakeEW.calls.append("rmsnorm")
        return np.asarray(x, dtype=np.float32)

    def silu_mul(self, g, u):
        FakeEW.calls.append("silu")
        return np.asarray(g, dtype=np.float32)

    def rope_inplace(self, *a):
        FakeEW.calls.append("rope")

    def softmax_axis1(self, s):
        FakeEW.calls.append("softmax")


class TestElementwiseSelection(unittest.TestCase):
    """C is used only where it beats NumPy's vectorised exp."""

    def _stub(self, ew_ops):
        r = object.__new__(run.SmolLM2Runner)
        r.eps = 1e-5
        r.ew_ops = ew_ops
        r.kernel = FakeEW()
        return r

    def test_default_routes_all_four_ops_to_c(self):
        self.assertEqual(set(run.DEFAULT_EW_OPS),
                         {"rmsnorm", "rope", "silu", "softmax"})
        FakeEW.calls = []
        r = self._stub(set(run.DEFAULT_EW_OPS))
        x = np.ones(8, np.float32)
        r._rms(x, x)
        r._rope(np.ones((2, 4), np.float32), np.ones(2, np.float32),
                np.ones(2, np.float32))
        r._silu_mul(x, x)
        r._softmax_axis1(np.ones((2, 3, 1), np.float32))
        self.assertEqual(sorted(set(FakeEW.calls)),
                         ["rmsnorm", "rope", "silu", "softmax"])

    def test_empty_set_uses_numpy_only(self):
        FakeEW.calls = []
        r = self._stub(set())
        r._rms(np.ones(8, np.float32), np.ones(8, np.float32))
        self.assertEqual(FakeEW.calls, [])


class TestPromptStyle(unittest.TestCase):
    """Compact prefix is used only where it measured safe."""

    def test_auto_is_compact_only_when_constrained(self):
        self.assertTrue(run.resolve_compact("auto", True, False))
        self.assertTrue(run.resolve_compact("auto", False, True))
        self.assertFalse(run.resolve_compact("auto", False, False))

    def test_explicit_style_overrides_auto(self):
        self.assertTrue(run.resolve_compact("compact", False, False))
        self.assertFalse(run.resolve_compact("full", True, True))


if __name__ == "__main__":
    unittest.main()
