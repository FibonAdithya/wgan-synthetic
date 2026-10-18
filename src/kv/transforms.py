"""h -> per-head keys and queries for Llama 4 Scout's global (no-RoPE) layers.

k = W_k h and q = s(pos) W_q h reproduce the model to bf16 precision
(docs/results/kv-cache-profile/). s(pos) does not change any query's ranking
in flat, IVF or Xbox-HNSW search, so search code may ignore it; attention
error does not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.kv_capture import read_safetensor

HEAD_DIM = 128


def load_layer_weights(capture_dir: Path, layer: int) -> dict[str, torch.Tensor]:
    path = Path(capture_dir) / "weights" / f"layer{layer:02d}.safetensors"
    out = {n: torch.from_numpy(read_safetensor(path, n)) for n in ("k_proj", "q_proj")}
    try:
        out["v_proj"] = torch.from_numpy(read_safetensor(path, "v_proj"))
    except KeyError:
        pass
    return out


def n_rep(wq: torch.Tensor, wk: torch.Tensor) -> int:
    return wq.shape[0] // wk.shape[0]


def head_keys(h, wk, g):
    return h @ wk[g * HEAD_DIM : (g + 1) * HEAD_DIM].T


def head_queries(h, wq, g, n_rep):
    w = wq[g * n_rep * HEAD_DIM : (g + 1) * n_rep * HEAD_DIM]
    return (h @ w.T).reshape(-1, HEAD_DIM)


def pos_scale(pos, attn_scale, floor_scale):
    p = pos.to(torch.float64)
    return (1.0 + attn_scale * torch.log1p(torch.floor((p + 1.0) / floor_scale))).to(
        torch.float32
    )


def xbox(base: np.ndarray) -> np.ndarray:
    n2 = (base.astype(np.float64) ** 2).sum(1)
    extra = np.sqrt(np.maximum(n2.max() - n2, 0.0)).astype(np.float32)
    return np.hstack([base, extra[:, None]])


def pad0(q: np.ndarray) -> np.ndarray:
    return np.hstack([q, np.zeros((len(q), 1), np.float32)])
