"""How much of each head's key (and query) variance the leading principal components of h carry.

`s256` kept the top 256 components of h and shuffled the rest, and still removed the search
hardness at L11 and L15 while only partly removing it at L43. This measures, per layer and
head, the share of key variance (k = W_k h) and query variance (q = W_q h) in the top r
components of h. The key variance that `s256` disturbs is the share outside the top 256.
Books 12-19, non-sink rows, all 12 global layers; the PCA is `surgery.fit_pca`, as in the run.

Run from the repo root on the box:
    PYTHONPATH=. python docs/results/scout-kv/v3-surgery/key_projection.py \
        --capture-dir /workspace/kv/capture --out /workspace/kv/v3surgery/key_projection.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.data.kv_capture import GLOBAL_LAYERS
from src.eval import kv_gate
from src.kv.surgery import fit_pca, projected_variance_share
from src.kv.transforms import HEAD_DIM, load_layer_weights

BOOKS = range(12, 20)
RANKS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
FRACTIONS = (0.5, 0.9, 0.99)


def rank_for(cum: torch.Tensor, f: float) -> list[int]:
    """Smallest r with cumulative share >= f, per row."""
    return ((cum < f).sum(-1) + 1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    dev = kv_gate._device()
    out = []
    for layer in GLOBAL_LAYERS:
        t0 = time.time()
        books = [
            h.to(dev).float() for h in kv_gate.load_books(a.capture_dir, layer, BOOKS)
        ]
        pca = fit_pca(books, n_sink=1)
        del books
        w = load_layer_weights(a.capture_dir, layer)
        k = projected_variance_share(pca, w["k_proj"].to(dev).float(), HEAD_DIM)
        q = projected_variance_share(pca, w["q_proj"].to(dev).float(), HEAD_DIM)
        hv = pca.var.double()
        h = hv.cumsum(0) / hv.sum()
        out.append(
            {
                "layer": layer,
                "dim": int(len(hv)),
                "n_key_heads": int(k.shape[0]),
                "n_query_heads": int(q.shape[0]),
                "h_share": {r: float(h[r - 1]) for r in RANKS},
                "key_share": {r: k[:, r - 1].tolist() for r in RANKS},
                "query_share": {r: q[:, r - 1].tolist() for r in RANKS},
                "h_rank_for": {f: rank_for(h, f) for f in FRACTIONS},
                "key_rank_for": {f: rank_for(k, f) for f in FRACTIONS},
                "query_rank_for": {f: rank_for(q, f) for f in FRACTIONS},
                "seconds": round(time.time() - t0, 1),
            }
        )
        k256 = sorted(out[-1]["key_share"][256])
        print(
            f"L{layer}: h top-256 {out[-1]['h_share'][256]:.4f}, "
            f"key top-256 median {k256[len(k256) // 2]:.4f} "
            f"(min {k256[0]:.4f}), {out[-1]['seconds']} s",
            flush=True,
        )
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print("KEY_PROJECTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
