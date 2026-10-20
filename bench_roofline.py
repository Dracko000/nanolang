#!/usr/bin/env python3
"""T1 — where does a decoded token's time actually go?

The microbenchmark in `phase0/bench/stream_probe.c` answers whether a GEMV is
bandwidth- or compute-bound by comparing its warm-cache marginal cost against
its DRAM marginal cost. This script answers the other half of the question: how
much of a real token is each kind of GEMV, and how much is neither.

It loads the release container, wraps the runner's own methods with timers, and
runs a prefill plus N decode steps. Nothing is reimplemented: the numbers come
from the shipped kernels through the shipped code path.

    python3 bench_roofline.py --tokens 8

Reading the output:

  MB/token   bytes the stage must stream to produce one token
  GB/s       MB/token / ms/token — compare against `stream_probe`'s ceiling.
             Far below it means the stage is compute-bound or has overhead.
  remainder  time in forward_token() that is not in any wrapped method:
             attention matmuls, KV concatenation, argmax, Python bookkeeping.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_smollm2 as R  # noqa: E402

STATS: dict[str, dict] = {}
MATS: dict[tuple, dict] = {}
DEPTH = {"linear": 0}


def bucket(name: str, dt: float, nbytes: int = 0) -> None:
    s = STATS.setdefault(name, {"t": 0.0, "bytes": 0, "calls": 0})
    s["t"] += dt
    s["bytes"] += nbytes
    s["calls"] += 1


def mat(name: str, shape: tuple, dt: float) -> None:
    s = MATS.setdefault((name, shape), {"t": 0.0, "n": 0, "min": 1e9})
    s["t"] += dt
    s["n"] += 1
    s["min"] = min(s["min"], dt)


def wrap(name: str, label: str, nbytes) -> None:
    orig = getattr(R.SmolLM2Runner, name)

    def wrapper(self, *a, **kw):
        t0 = time.perf_counter()
        out = orig(self, *a, **kw)
        dt = time.perf_counter() - t0
        bucket(label, dt, nbytes(self, a) if nbytes else 0)
        if name == "_linear" and len(a) > 1:
            t = self.t.get(a[1])
            if t is not None and t.get("packed"):
                mat(a[1], tuple(t["shape"]), dt)
        return out

    setattr(R.SmolLM2Runner, name, wrapper)


def _linear_bytes(self, args):
    t = self.t.get(args[1])
    return int(t["data"].nbytes) if t else 0


def _fp32_bytes(self, args):
    # Only count the top-level call: _linear() may delegate here, and the time
    # would otherwise be attributed twice and the parts would sum past the whole.
    return 0 if DEPTH["linear"] else int(np.asarray(args[0]).nbytes)


def _rms_bytes(self, args):
    return int(np.asarray(args[0]).nbytes + np.asarray(args[1]).nbytes)


def _silu_bytes(self, args):
    return int(np.asarray(args[0]).nbytes + np.asarray(args[1]).nbytes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/bitnet700m/bitnet700m.nanow")
    ap.add_argument("--tokenizer", default="models/bitnet700m/tokenizer.json")
    ap.add_argument("--config", default="models/bitnet700m/config.json")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"model not found: {args.model}\n"
              "The .nanow container ships with the model repo, not with the "
              "engine:\n"
              "  https://huggingface.co/dracko14/nanolang-bitnet-b158-700m\n"
              "Pass --model /path/to/bitnet700m.nanow (plus --tokenizer and "
              "--config).", file=sys.stderr)
        return 2

    tensors, config, mm, fd = R.load_nanow(args.model)
    try:
        mm.close()
    except Exception:
        pass
    os.close(fd)

    with open(args.config) as f:
        hf = json.load(f)
    config["n_heads"] = hf["num_attention_heads"]
    config["n_kv_heads"] = hf["num_key_value_heads"]
    config["rope_theta"] = hf.get("rope_theta", 100000.0)
    config["rms_norm_eps"] = hf.get("rms_norm_eps", 1e-5)
    config["max_position_embeddings"] = hf.get("max_position_embeddings", 2048)

    tokenizer = R.BPETokenizer(args.tokenizer)
    kernel = R.TernaryKernel(os.path.join(HERE, "phase0", "bin", "libternary.so"))
    runner = R.SmolLM2Runner(tensors, config, tokenizer, kernel=kernel)
    runner.model_tag = args.model

    ternary_bytes = sum(int(t["data"].nbytes) for t in tensors.values() if t["ternary"])
    fp32_bytes = sum(int(t["data"].nbytes) for t in tensors.values() if not t["ternary"])
    print(f"container: {len(tensors)} tensors, "
          f"{ternary_bytes / 1e6:.1f} MB ternary + {fp32_bytes / 1e6:.1f} MB fp32")

    # --- instrument -------------------------------------------------------
    wrap("_linear", "ternary GEMV", _linear_bytes)
    orig_linear = R.SmolLM2Runner.__dict__["_linear"]
    def linear_with_depth(self, *a, **kw):
        DEPTH["linear"] += 1
        try:
            return orig_linear(self, *a, **kw)
        finally:
            DEPTH["linear"] -= 1
    R.SmolLM2Runner._linear = linear_with_depth
    wrap("_fp32_linear", "fp32 GEMV (lm head)", _fp32_bytes)
    wrap("_rms", "rmsnorm", _rms_bytes)
    wrap("_silu_mul", "silu*up", _silu_bytes)
    wrap("_rope", "rope", lambda s, a: 0)
    wrap("_softmax_axis1", "softmax", lambda s, a: 0)

    # --- prefill ----------------------------------------------------------
    ids = tokenizer.encode(args.prompt)
    if getattr(tokenizer, "spm", False):
        ids = [tokenizer.bos_id] + ids
    t0 = time.perf_counter()
    for tok in ids:
        runner.forward_token(tok)
    prefill = time.perf_counter() - t0
    STATS.clear()          # report decode only
    MATS.clear()

    # --- decode -----------------------------------------------------------
    total = 0.0
    tok_id = ids[-1]
    for _ in range(args.tokens):
        t0 = time.perf_counter()
        logits = runner.forward_token(tok_id)
        total += time.perf_counter() - t0
        tok_id = int(np.argmax(logits))
    per_tok = total / args.tokens

    print(f"prefill {len(ids)} tokens: {prefill * 1e3:.1f} ms "
          f"({len(ids) / prefill:.1f} tok/s)")
    print(f"decode: {per_tok * 1e3:.2f} ms/token = {1 / per_tok:.1f} tok/s\n")

    print(f"{'stage':<22} {'ms/token':>9} {'MB/token':>9} {'GB/s':>7} "
          f"{'share':>7}")
    print("-" * 60)

    rows = []
    for name, s in sorted(STATS.items(), key=lambda kv: -kv[1]["t"]):
        ms = s["t"] / args.tokens * 1e3
        mb = s["bytes"] / args.tokens / 1e6
        gbs = (s["bytes"] / s["t"] / 1e9) if s["t"] > 0 else 0.0
        rows.append((name, ms, mb, gbs))

    accounted = sum(r[1] for r in rows)
    for name, ms, mb, gbs in rows:
        print(f"{name:<22} {ms:9.2f} {mb:9.1f} {gbs:7.2f} "
              f"{100 * ms / (per_tok * 1e3):6.1f}%")

    rest = per_tok * 1e3 - accounted
    print(f"{'attention + overhead':<22} {rest:9.2f} {'':>9} {'':>7} "
          f"{100 * rest / (per_tok * 1e3):6.1f}%")
    print("-" * 60)
    print(f"{'total':<22} {per_tok * 1e3:9.2f}")

    # --- per shape, to explain the in-situ cost of the ternary GEMV ---------
    # The probe measured 21.8 ns/row warm and 28.0 ns/row DRAM-cold for K=1536.
    # If the model's calls are far above that, the difference is not the kernel
    # but what surrounds 168 separate calls per token.
    print("\nternary GEMV by shape, ns per row and equivalent at K=1536:")
    print(f"{'M x K':<14} {'calls/tok':>10} {'mean':>7} {'min':>7} "
          f"{'mean@1536':>10} {'min@1536':>9} {'MB/tok':>8}")
    print("-" * 72)
    by_shape: dict[tuple, dict] = {}
    for (_name, shape), s in MATS.items():
        d = by_shape.setdefault(shape, {"t": 0.0, "n": 0, "min": 1e9})
        d["t"] += s["t"]
        d["n"] += s["n"]
        d["min"] = min(d["min"], s["min"])
    for shape, d in sorted(by_shape.items(), key=lambda kv: -kv[1]["t"]):
        M, K = shape
        mean_row = (d["t"] / d["n"]) / M * 1e9
        min_row = d["min"] / M * 1e9
        f = 1536.0 / K
        mb = d["n"] * M * ((K + 3) // 4) / args.tokens / 1e6
        print(f"{M:5d} x {K:<6d} {d['n'] / args.tokens:10.0f} "
              f"{mean_row:7.1f} {min_row:7.1f} {mean_row * f:10.1f} "
              f"{min_row * f:9.1f} {mb:8.1f}")
    print("probe reference for K=1536: 21.8 ns/row warm, 28.0 ns/row DRAM-cold")

    # --- what T2 would buy -------------------------------------------------
    head = [r for r in rows if r[0].startswith("fp32")]
    if head:
        _, head_ms, head_mb, head_gbs = head[0]
        int8_mb = head_mb / 4.0
        # The head is bandwidth-bound (see stream_probe), so its time scales with
        # bytes at the same achieved GB/s. This is an upper bound: an int8 kernel
        # also has to dequantise, so it will not be exactly free.
        projected_ms = head_ms * (int8_mb / head_mb)
        saved = head_ms - projected_ms
        print(f"\nT2 projection (fp32 head -> int8, {head_mb:.1f} -> "
              f"{int8_mb:.1f} MB/token):")
        print(f"  head stage   {head_ms:.2f} ms -> ~{projected_ms:.2f} ms "
              f"(bandwidth-bound, {head_gbs:.2f} GB/s measured)")
        print(f"  token        {per_tok * 1e3:.2f} ms -> "
              f"~{per_tok * 1e3 - saved:.2f} ms  "
              f"({1 / (per_tok - saved / 1e3):.1f} tok/s, "
              f"+{100 * saved / (per_tok * 1e3):.0f}%)")
        print("  upper bound only: it ignores int8 quantisation error and the "
              "dequant work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
