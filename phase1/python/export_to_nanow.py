"""Python Model Exporter and Inspector for .nanow Container Format.

Allows exporting PyTorch/NumPy weights into page-aligned, bit-packed
.nanow binary files for zero-copy mmap loading in C/C++/Mobile engines.
"""

from __future__ import annotations
import mmap
import os
import struct
import numpy as np

NANOW_MAGIC = 0x4E414E4F  # 'N', 'A', 'N', 'O'
NANOW_VERSION = 1
PAGE_ALIGNMENT = 4096

# Packing enum
PACKING_RAW_INT8 = 0
PACKING_2BIT = 1
PACKING_RADIX3 = 2

# Dtype enum
DTYPE_TERNARY = 0
DTYPE_INT8 = 1
DTYPE_FP16 = 2
DTYPE_FP32 = 3


def pack_ternary_2bit(ternary_arr: np.ndarray) -> bytes:
    """Packs array of int8 {-1, 0, 1} into 2-bit bytes (4 weights per byte)."""
    flat = ternary_arr.flatten().astype(np.int8)
    # Mapping: 0 -> 0, +1 -> 1, -1 -> 3
    code = np.zeros_like(flat, dtype=np.uint8)
    code[flat == 1] = 0x01
    code[flat == -1] = 0x03

    pad_len = (4 - (len(code) % 4)) % 4
    if pad_len > 0:
        code = np.pad(code, (0, pad_len), constant_values=0)

    # Reshape to (N, 4) and pack
    grouped = code.reshape(-1, 4)
    packed = (
        grouped[:, 0]
        | (grouped[:, 1] << 2)
        | (grouped[:, 2] << 4)
        | (grouped[:, 3] << 6)
    )
    return packed.tobytes()


def unpack_ternary_2bit(packed_bytes: bytes, num_elements: int) -> np.ndarray:
    """Unpacks 2-bit bytes back into int8 array {-1, 0, 1}."""
    arr = np.frombuffer(packed_bytes, dtype=np.uint8)
    c0 = arr & 0x03
    c1 = (arr >> 2) & 0x03
    c2 = (arr >> 4) & 0x03
    c3 = (arr >> 6) & 0x03

    stacked = np.stack([c0, c1, c2, c3], axis=1).flatten()[:num_elements]
    unpacked = np.zeros_like(stacked, dtype=np.int8)
    unpacked[stacked == 1] = 1
    unpacked[stacked == 3] = -1
    return unpacked


class NanoWExporter:
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        state_dim: int = 16,
        vocab_size: int = 32000,
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.vocab_size = vocab_size
        self.tensors: list[dict] = []

    def add_tensor(
        self,
        name: str,
        data: np.ndarray,
        is_ternary: bool = True,
        scale: float = 1.0,
    ):
        """Adds a tensor to be exported."""
        if is_ternary:
            packed_payload = pack_ternary_2bit(data)
            packing = PACKING_2BIT
            dtype = DTYPE_TERNARY
        else:
            packed_payload = data.tobytes()
            packing = PACKING_RAW_INT8
            dtype = DTYPE_FP32

        self.tensors.append(
            {
                "name": name.encode("utf-8")[:63],
                "dtype": dtype,
                "packing": packing,
                "shape": list(data.shape) + [1] * (4 - len(data.shape)),
                "num_elements": data.size,
                "scale": scale,
                "payload": packed_payload,
            }
        )

    def export(self, output_path: str):
        """Writes the .nanow container with 4KB page alignment."""
        num_tensors = len(self.tensors)

        # Header format:
        # 4u: magic, version, flags, num_tensors
        # 4u: num_layers, hidden_dim, state_dim, vocab_size
        # 2Q: total_params, file_size
        # 1u: header_size
        # 3u: reserved
        header_fmt = "<IIII IIII QQ I 3I"
        header_size = struct.calcsize(header_fmt)

        # Tensor desc format:
        # 64s: name
        # 3I: dtype, packing, ndims
        # 4I: shape
        # 1Q: num_elements
        # 1Q: data_offset
        # 1Q: data_size
        # 1f: scale_factor
        desc_fmt = "<64s III 4I Q Q Q f"
        desc_size = struct.calcsize(desc_fmt)

        directory_size = header_size + num_tensors * desc_size
        initial_pad = (PAGE_ALIGNMENT - (directory_size % PAGE_ALIGNMENT)) % PAGE_ALIGNMENT
        data_offset = directory_size + initial_pad

        # Calculate offsets for each tensor
        processed_tensors = []
        total_params = 0
        for t in self.tensors:
            payload_len = len(t["payload"])
            t_pad = (PAGE_ALIGNMENT - (payload_len % PAGE_ALIGNMENT)) % PAGE_ALIGNMENT
            processed_tensors.append(
                {
                    **t,
                    "offset": data_offset,
                    "size": payload_len,
                    "pad": t_pad,
                }
            )
            data_offset += payload_len + t_pad
            total_params += t["num_elements"]

        total_file_size = data_offset

        # Build Header
        header_bytes = struct.pack(
            header_fmt,
            NANOW_MAGIC,
            NANOW_VERSION,
            0,
            num_tensors,
            self.num_layers,
            self.hidden_dim,
            self.state_dim,
            self.vocab_size,
            total_params,
            total_file_size,
            directory_size,
            0,
            0,
            0,
        )

        with open(output_path, "wb") as f:
            f.write(header_bytes)

            # Write Descriptors
            for pt in processed_tensors:
                desc_bytes = struct.pack(
                    desc_fmt,
                    pt["name"].ljust(64, b"\x00"),
                    pt["dtype"],
                    pt["packing"],
                    len([s for s in pt["shape"] if s > 1]),
                    *pt["shape"],
                    pt["num_elements"],
                    pt["offset"],
                    pt["size"],
                    pt["scale"],
                )
                f.write(desc_bytes)

            # Write initial alignment padding
            if initial_pad > 0:
                f.write(b"\x00" * initial_pad)

            # Write Tensors
            for pt in processed_tensors:
                f.write(pt["payload"])
                if pt["pad"] > 0:
                    f.write(b"\x00" * pt["pad"])


def inspect_nanow(file_path: str):
    """Inspects a .nanow file using memory mapping."""
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            header_fmt = "<IIII IIII QQ I 3I"
            header_size = struct.calcsize(header_fmt)
            (
                magic,
                ver,
                flags,
                num_tensors,
                layers,
                hdim,
                sdim,
                vocab,
                tot_params,
                file_sz,
                hdr_sz,
                _,
                _,
                _,
            ) = struct.unpack(header_fmt, mm[:header_size])

            assert magic == NANOW_MAGIC, "Invalid magic number!"
            print(f"File: {file_path}")
            print(f"  Version:      {ver}")
            print(f"  Tensors:      {num_tensors}")
            print(f"  Layers:       {layers}")
            print(f"  Hidden Dim:   {hdim}")
            print(f"  State Dim:    {sdim}")
            print(f"  Total Params: {tot_params} ({tot_params/1e6:.2f}M)")
            print(f"  File Size:    {file_sz / (1024*1024):.2f} MB")


if __name__ == "__main__":
    exporter = NanoWExporter(num_layers=4, hidden_dim=512, state_dim=16)
    W = np.random.choice([-1, 0, 1], size=(512, 512)).astype(np.int8)
    exporter.add_tensor("layer0.ssm_in", W, is_ternary=True, scale=0.02)

    sample_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "python_export.nanow")
    exporter.export(sample_file)
    print("Exported successfully.")
    inspect_nanow(sample_file)
