"""Shared tiny settings for the scout_kv v2 (book latent) tests."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from src.data.kv_capture import h_path, write_bf16_safetensor


def _write_multi(path: Path, tensors: dict) -> None:
    """F32 multi-tensor safetensors, the layout of capture/weights/layerXX."""
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        b = np.ascontiguousarray(arr, "<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [off, off + len(b)],
        }
        blobs.append(b)
        off += len(b)
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def write_tiny_capture(cap, books, layer=3, T=64, D=16, seed=0) -> None:
    """Write books to a capture dir: rows N(0,1) + book index, plus the layer's weights."""
    cap = Path(cap)
    rng = np.random.default_rng(seed)
    for b in books:
        p = h_path(cap, b, layer)
        p.parent.mkdir(parents=True, exist_ok=True)
        write_bf16_safetensor(p, "h", (rng.normal(size=(T, D)) + b).astype(np.float32))
    (cap / "weights").mkdir(parents=True, exist_ok=True)
    _write_multi(
        cap / "weights" / f"layer{layer:02d}.safetensors",
        {
            "k_proj": rng.normal(size=(2 * 128, D)).astype(np.float32),
            "q_proj": rng.normal(size=(2 * 5 * 128, D)).astype(np.float32),
        },
    )
