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


def tiny_config(tmp_path) -> dict:
    """The v2 config at test size: D 16, T 64, books 0 and 1 written under tmp_path."""
    tmp_path = Path(tmp_path)
    cap = tmp_path / "capture"
    write_tiny_capture(cap, [0, 1], layer=3, T=64, D=16)
    return {
        "seed": 42,
        "device": "cpu",
        "output_dir": str(tmp_path / "run"),
        "data": {
            "real_path": str(cap),
            "format": "kv_capture",
            "kv_layer": 3,
            "kv_books": [0, 1],
            "kv_n_sink": 1,
            "kv_n_tail": 16,
            "descriptor_dim": 16,
            "metric": "l2",
            "preprocess": {"center": True, "whiten": False, "l2_normalize": False},
        },
        "model": {
            "generator_type": "book_latent",
            "latent_dim": 8,
            "generator_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 16],
            "negative_slope": 0.2,
            "output_normalize": False,
            "n_flags": 2,
            "book_latent": {
                "channels": 4,
                "n_features": 2,
                "min_length": 2.0,
                "max_length": 64.0,
            },
            "window_critic": {
                "window": 8,
                "strides": [1, 4],
                "token_hidden_dims": [16, 8],
                "window_hidden_dims": [16, 8],
            },
        },
        "training": {
            "n_windows": 8,
            "tail_window_share": 0.25,
            "num_gen_steps": 4,
            "n_critic": 2,
            "lr_g": 1.0e-4,
            "lr_d": 1.0e-4,
            "betas": [0.0, 0.9],
            "lambda_gp": 10.0,
            "window_weight": 0.1,
            "ema_decay": 0.999,
            "log_every": 1,
            "save_every": 2,
        },
    }
