"""Reader for the scout_kv capture files.

The capture writes safetensors: an 8-byte little-endian header length, a JSON
header mapping each tensor name to dtype, shape and byte offsets, then the raw
bytes. That layout is fixed by the format, so it is read here directly rather
than through the safetensors package, which would be a new pinned dependency
for 30 lines of parsing. Only the dtypes the capture produces are accepted.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np

GLOBAL_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
_DTYPES = {"BF16": 2, "F16": 2, "F32": 4}


def h_path(capture_dir: Path, book: int, layer: int) -> Path:
    return Path(capture_dir) / f"doc{book}" / f"h_L{layer:02d}.safetensors"


def _header(f) -> tuple[dict, int]:
    (n,) = struct.unpack("<Q", f.read(8))
    return json.loads(f.read(n)), 8 + n


def read_safetensor(path: Path, name: str) -> np.ndarray:
    with open(path, "rb") as f:
        header, base = _header(f)
        if name not in header:
            raise KeyError(
                f"{name} not in {path}: has {sorted(k for k in header if k != '__metadata__')}"
            )
        meta = header[name]
        dtype, shape, (start, end) = meta["dtype"], meta["shape"], meta["data_offsets"]
        if dtype not in _DTYPES:
            raise ValueError(
                f"{path}:{name} has dtype {dtype}; only {sorted(_DTYPES)} are supported"
            )
        f.seek(base + start)
        raw = f.read(end - start)
    if dtype == "F32":
        out = np.frombuffer(
            raw, "<f4"
        ).copy()  # frombuffer is read-only; torch.from_numpy warns on it
    elif dtype == "F16":
        out = np.frombuffer(raw, "<f2").astype(np.float32)
    else:
        out = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
    return out.reshape(shape).astype(np.float32, copy=False)


def write_bf16_safetensor(path: Path, name: str, arr: np.ndarray) -> None:
    a = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    rounded = ((a + 0x7FFF + ((a >> 16) & 1)) >> 16).astype("<u2")
    header = json.dumps(
        {
            name: {
                "dtype": "BF16",
                "shape": list(arr.shape),
                "data_offsets": [0, rounded.nbytes],
            }
        }
    ).encode()
    Path(path).write_bytes(struct.pack("<Q", len(header)) + header + rounded.tobytes())


def load_kv_capture(
    capture_dir: Path, layer: int, books: Sequence[int], n_sink: int = 1
) -> np.ndarray:
    if layer not in GLOBAL_LAYERS:
        raise ValueError(f"layer {layer} is not a global layer {GLOBAL_LAYERS}")
    return np.vstack(
        [read_safetensor(h_path(capture_dir, b, layer), "h")[n_sink:] for b in books]
    )
