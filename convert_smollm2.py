"""SmolLM2-135M → NanoLang .nanow Converter.

Reads real BF16 safetensors weights, ternarizes every linear projection
to BitNet b1.58 {-1, 0, +1}, keeps embedding and norm layers in FP32,
packs everything into a page-aligned .nanow container for zero-copy mmap.
"""

from __future__ import annotations
import json
import struct
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase1", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase0", "python"))
from export_to_nanow import NanoWExporter, pack_ternary_2bit, NANOW_MAGIC, NANOW_VERSION, PAGE_ALIGNMENT
from export_to_nanow import PACKING_2BIT, PACKING_RAW_INT8, DTYPE_TERNARY, DTYPE_FP32, DTYPE_FP16

# ---------------------------------------------------------------------------
# BF16 safetensors reader (no torch dependency)
# ---------------------------------------------------------------------------

def read_safetensors_index(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(header_len).decode())
    data_offset = 8 + header_len
    return meta, data_offset

def read_tensor_bf16(path: str, meta: dict, name: str, data_offset: int) -> np.ndarray:
    info = meta[name]
    start, end = info["data_offsets"]
    shape = info["shape"]
    with open(path, "rb") as f:
        f.seek(data_offset + start)
        raw = f.read(end - start)
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(shape)

# ---------------------------------------------------------------------------
# Ternarization
# ---------------------------------------------------------------------------

def ternarize(W: np.ndarray) -> tuple[np.ndarray, float]:
    gamma = float(np.mean(np.abs(W))) + 1e-7
    scaled = W / gamma
    Wt = np.clip(np.round(scaled), -1.0, 1.0).astype(np.int8)
    return Wt, gamma

# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert(safetensors_path: str, config_path: str, output_path: str):
    with open(config_path) as f:
        cfg = json.load(f)

    hidden = cfg["hidden_size"]          # 576
    layers = cfg["num_hidden_layers"]    # 30
    vocab  = cfg["vocab_size"]           # 49152
    inter  = cfg["intermediate_size"]    # 1536

    print("=" * 70)
    print(" SMOLLM2-135M → NANOLANG .nanow CONVERTER")
    print("=" * 70)
    print(f"  Hidden:       {hidden}")
    print(f"  Layers:       {layers}")
    print(f"  Vocab:        {vocab}")
    print(f"  Intermediate: {inter}")
    print()

    meta, data_off = read_safetensors_index(safetensors_path)
    tensor_keys = [k for k in meta if k != "__metadata__"]
    print(f"  Safetensors contains {len(tensor_keys)} tensors")

    # Custom exporter that handles mixed precision
    t_start = time.time()
    tensors_out: list[dict] = []
    total_params = 0
    total_ternary = 0
    total_fp32 = 0

    def add_tensor(name: str, data: np.ndarray, ternary: bool, scale: float = 1.0):
        nonlocal total_params, total_ternary, total_fp32
        total_params += data.size
        if ternary:
            total_ternary += data.size
            payload = pack_ternary_2bit(data)
            tensors_out.append({
                "name": name.encode()[:63],
                "dtype": DTYPE_TERNARY,
                "packing": PACKING_2BIT,
                "shape": list(data.shape) + [1] * (4 - len(data.shape)),
                "num_elements": data.size,
                "scale": scale,
                "payload": payload,
            })
        else:
            total_fp32 += data.size
            payload = data.astype(np.float32).tobytes()
            tensors_out.append({
                "name": name.encode()[:63],
                "dtype": DTYPE_FP32,
                "packing": PACKING_RAW_INT8,
                "shape": list(data.shape) + [1] * (4 - len(data.shape)),
                "num_elements": data.size,
                "scale": 1.0,
                "payload": payload,
            })

    # 1. Embedding (keep FP32 — critical for quality)
    print("  [1/4] Converting embedding table...", flush=True)
    emb = read_tensor_bf16(safetensors_path, meta, "model.embed_tokens.weight", data_off)
    add_tensor("embed_tokens", emb, ternary=False)

    # 2. Per-layer weights
    print(f"  [2/4] Converting {layers} transformer layers...", flush=True)
    for l in range(layers):
        prefix = f"model.layers.{l}"
        # Norms — keep FP32
        for norm_name in ["input_layernorm", "post_attention_layernorm"]:
            key = f"{prefix}.{norm_name}.weight"
            if key in meta:
                w = read_tensor_bf16(safetensors_path, meta, key, data_off)
                add_tensor(f"layers.{l}.{norm_name}", w, ternary=False)

        # Linear projections — ternarize
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                      "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            key = f"{prefix}.{proj}.weight"
            if key in meta:
                w = read_tensor_bf16(safetensors_path, meta, key, data_off)
                wt, gamma = ternarize(w)
                short_name = f"layers.{l}.{proj.replace('self_attn.', 'attn.').replace('mlp.', 'mlp.')}"
                add_tensor(short_name, wt, ternary=True, scale=gamma)

        if (l + 1) % 5 == 0:
            print(f"    ... layer {l+1}/{layers} done", flush=True)

    # 3. Final norm
    print("  [3/4] Converting final norm...", flush=True)
    if "model.norm.weight" in meta:
        w = read_tensor_bf16(safetensors_path, meta, "model.norm.weight", data_off)
        add_tensor("final_norm", w, ternary=False)

    # 4. LM Head
    print("  [4/4] Converting lm_head...", flush=True)
    if "lm_head.weight" in meta:
        w = read_tensor_bf16(safetensors_path, meta, "lm_head.weight", data_off)
        add_tensor("lm_head", w, ternary=False)
    else:
        print("    (lm_head not found — likely tied to embed_tokens)")

    # Write .nanow container
    print(f"\n  Serializing .nanow container...", flush=True)
    num_tensors = len(tensors_out)

    header_fmt = "<IIII IIII QQ I 3I"
    header_size = struct.calcsize(header_fmt)
    desc_fmt = "<64s III 4I Q Q Q f"
    desc_size = struct.calcsize(desc_fmt)

    directory_size = header_size + num_tensors * desc_size
    initial_pad = (PAGE_ALIGNMENT - (directory_size % PAGE_ALIGNMENT)) % PAGE_ALIGNMENT
    current_offset = directory_size + initial_pad

    for t in tensors_out:
        payload_len = len(t["payload"])
        t["offset"] = current_offset
        t["size"] = payload_len
        t_pad = (PAGE_ALIGNMENT - (payload_len % PAGE_ALIGNMENT)) % PAGE_ALIGNMENT
        t["pad"] = t_pad
        current_offset += payload_len + t_pad

    total_file_size = current_offset

    header_bytes = struct.pack(
        header_fmt,
        NANOW_MAGIC, NANOW_VERSION, 0, num_tensors,
        layers, hidden, 16, vocab,
        total_params, total_file_size, directory_size,
        0, 0, 0,
    )

    with open(output_path, "wb") as f:
        f.write(header_bytes)
        for t in tensors_out:
            desc = struct.pack(
                desc_fmt,
                t["name"].ljust(64, b"\x00"),
                t["dtype"], t["packing"],
                len([s for s in t["shape"] if s > 1]),
                *t["shape"],
                t["num_elements"],
                t["offset"],
                t["size"],
                t["scale"],
            )
            f.write(desc)
        if initial_pad:
            f.write(b"\x00" * initial_pad)
        for t in tensors_out:
            f.write(t["payload"])
            if t["pad"]:
                f.write(b"\x00" * t["pad"])

    elapsed = time.time() - t_start
    file_mb = os.path.getsize(output_path) / (1024 * 1024)
    orig_mb = os.path.getsize(safetensors_path) / (1024 * 1024)

    print()
    print("=" * 70)
    print(" CONVERSION COMPLETE")
    print("=" * 70)
    print(f"  Total Parameters:   {total_params:,}")
    print(f"  Ternary (1.58-bit): {total_ternary:,} ({100*total_ternary/total_params:.1f}%)")
    print(f"  FP32 (preserved):   {total_fp32:,} ({100*total_fp32/total_params:.1f}%)")
    print(f"  Original BF16:      {orig_mb:.1f} MB")
    print(f"  NanoLang .nanow:    {file_mb:.1f} MB")
    print(f"  Compression Ratio:  {orig_mb / file_mb:.2f}x")
    print(f"  Conversion Time:    {elapsed:.1f} seconds")
    print(f"  Output File:        {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    convert(
        safetensors_path="models/smollm2/model.safetensors",
        config_path="models/smollm2/config.json",
        output_path="models/smollm2/smollm2-135m.nanow",
    )
