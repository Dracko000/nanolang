"""KV prefix cache stored as a `.nanow` container.

The cache began life as a `np.savez` file, which made one directory hold two
different artifact formats and dragged a zip reader (and numpy's pickle story)
into loading a file that is really just a few float arrays. It is now an ordinary
`.nanow` container with the `NANOW_FLAG_KV_CACHE` header bit set:

  * the same mmap reader, the same page alignment, the same descriptor table
  * one artifact extension for everything this project writes to disk
  * the flag lets a reader tell cache state from model weights, in both
    directions, instead of guessing from the filename

The payload is fp32 because that is the only non-ternary dtype the existing
Python reader decodes. Token ids and positions are meaningless if rounded, so
`write_cache` asserts they are exactly representable rather than trusting that
they are small.

This is deliberately a second *writer* of the documented layout, not a second
layout: the struct formats and the exporter are imported from
`phase1/python/export_to_nanow.py`, and the repo's tests read the result back
with both the C reader and `run_smollm2.load_nanow` so drift is caught.
"""

from __future__ import annotations

import mmap
import os
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE1_PYTHON = os.path.join(_HERE, "phase1", "python")
if _PHASE1_PYTHON not in sys.path:
    sys.path.insert(0, _PHASE1_PYTHON)

from export_to_nanow import (  # noqa: E402
    DESC_FMT,
    DTYPE_FP32,
    HEADER_FMT,
    NANOW_FLAG_KV_CACHE,
    NANOW_MAGIC,
    NANOW_VERSION,
    NanoWExporter,
)

# Every value is stored as fp32, so an integer is only safe below 2**24.
_MAX_EXACT_INT = 1 << 24


def write_cache(path: str, arrays: dict, num_layers: int = 0,
                integers: tuple = ()) -> None:
    """Write ``arrays`` (name -> ndarray) to ``path`` as a KV-cache container.

    ``integers`` names the arrays that hold counted things (token ids, the
    position). They are checked to be finite, integral and below 2**24 so a
    rounded id can never be written silently.

    The file is written next to ``path`` and then renamed, so an interrupted
    write cannot leave a half-written cache behind for the next run to load.
    """
    exporter = NanoWExporter(
        num_layers=num_layers,
        hidden_dim=0,
        state_dim=0,
        vocab_size=0,
        flags=NANOW_FLAG_KV_CACHE,
    )
    integers = set(integers)

    for name, array in arrays.items():
        data = np.ascontiguousarray(array, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError(f"cache array {name!r} holds non-finite values")
        if name in integers:
            if not np.all(data == np.floor(data)):
                raise ValueError(f"cache array {name!r} must hold whole numbers")
            if data.size and float(np.max(np.abs(data))) >= _MAX_EXACT_INT:
                raise ValueError(
                    f"cache array {name!r} holds a value too large to store exactly "
                    f"as fp32 (limit {_MAX_EXACT_INT})"
                )
        exporter.add_tensor(name, data, is_ternary=False, scale=1.0)

    tmp = f"{path}.tmp"
    try:
        exporter.export(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_cache(path: str):
    """Read a KV-cache container written by :func:`write_cache`.

    Returns ``(arrays, meta)``. Raises ``ValueError`` when the file exists but is
    not a KV cache — for example a model container someone pointed the cache
    directory at — and ``OSError`` when it cannot be read at all.
    """
    with open(path, "rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            header_size = struct.calcsize(HEADER_FMT)
            if len(mm) < header_size:
                raise ValueError(f"{path} is too short to hold a container header")

            (magic, version, flags, num_tensors, num_layers, _hidden, _state,
             _vocab, total_params, _file_size, _dir_size, _r0, _r1, _r2) = struct.unpack(
                HEADER_FMT, mm[:header_size]
            )

            if magic != NANOW_MAGIC:
                raise ValueError(f"{path} is not a .nanow container (magic {magic:#x})")
            if version != NANOW_VERSION:
                raise ValueError(
                    f"{path} has container version {version}, expected {NANOW_VERSION}"
                )
            if not flags & NANOW_FLAG_KV_CACHE:
                raise ValueError(
                    f"{path} is a .nanow container but not a KV cache "
                    f"(flags {flags:#x}); refusing to read it as one"
                )

            desc_size = struct.calcsize(DESC_FMT)
            arrays = {}
            for i in range(num_tensors):
                start = header_size + i * desc_size
                desc = struct.unpack(DESC_FMT, mm[start:start + desc_size])
                name = desc[0].split(b"\x00")[0].decode("utf-8", "replace")
                dtype, _packing, ndims = desc[1], desc[2], desc[3]
                shape = tuple(int(s) for s in desc[4:8][:ndims])
                num_elements = desc[8]
                data_offset, data_size = desc[9], desc[10]

                if dtype != DTYPE_FP32:
                    raise ValueError(
                        f"{path}:{name} has dtype {dtype}; a KV cache stores fp32 only"
                    )
                if num_elements * 4 != data_size:
                    raise ValueError(
                        f"{path}:{name} descriptor claims {num_elements} elements but "
                        f"{data_size} bytes"
                    )

                # Slice (a copy) rather than memoryview, which would keep the
                # mmap alive and raise BufferError when it is closed below.
                flat = np.frombuffer(mm[data_offset:data_offset + data_size],
                                     dtype="<f4")
                arrays[name] = flat.reshape(shape).copy() if shape else flat.copy()

            meta = {
                "num_layers": num_layers,
                "num_tensors": num_tensors,
                "total_params": total_params,
                "flags": flags,
            }
            return arrays, meta
