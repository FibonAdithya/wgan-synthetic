from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import load_descriptors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot distance CDF percentile curves (10/50/90) from random queries "
            "to sampled dataset points for real SIFT and synthetic data."
        )
    )
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument(
        "--real-format", type=str, default="auto", choices=["auto", "npy", "fvecs"]
    )
    parser.add_argument("--synthetic-path", type=str, required=True)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--num-targets", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


def l2_normalize(x: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def sampled_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k >= n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)


def query_cdf_quantiles(
    x: np.ndarray,
    num_queries: int,
    num_targets: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_idx = sampled_indices(x.shape[0], num_queries, rng)
    t_idx = sampled_indices(x.shape[0], num_targets, rng)
    queries = x[q_idx]
    targets = x[t_idx]

    # Distances shape: [Q, T]
    dists = np.linalg.norm(queries[:, None, :] - targets[None, :, :], axis=2)

    # If query points overlap with targets, remove exact self-distance where possible.
    dist_rows = []
    t_lookup = {int(v): i for i, v in enumerate(t_idx.tolist())}
    for row_id, q_global in enumerate(q_idx.tolist()):
        row = dists[row_id]
        if q_global in t_lookup:
            row = np.delete(row, t_lookup[q_global])
        row = np.sort(row)
        dist_rows.append(row)

    min_len = min(len(r) for r in dist_rows)
    stacked = np.stack([r[:min_len] for r in dist_rows], axis=0)
    y = np.linspace(0.0, 1.0, min_len, endpoint=False)
    q10 = np.quantile(stacked, 0.10, axis=0)
    q50 = np.quantile(stacked, 0.50, axis=0)
    q90 = np.quantile(stacked, 0.90, axis=0)
    return y, q10, q50, q90


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    real = load_descriptors(Path(args.real_path), file_format=args.real_format)
    synthetic = np.load(args.synthetic_path).astype(np.float32, copy=False)

    real = l2_normalize(real.astype(np.float32, copy=False))
    synthetic = l2_normalize(synthetic)

    y_r, r10, r50, r90 = query_cdf_quantiles(
        real, num_queries=args.num_queries, num_targets=args.num_targets, rng=rng
    )
    y_s, s10, s50, s90 = query_cdf_quantiles(
        synthetic, num_queries=args.num_queries, num_targets=args.num_targets, rng=rng
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(r10, y_r, linestyle="--", label="SIFT q10")
    ax.plot(r50, y_r, linewidth=2.0, label="SIFT q50")
    ax.plot(r90, y_r, linestyle="--", label="SIFT q90")

    ax.plot(s10, y_s, linestyle=":", label="Generated q10")
    ax.plot(s50, y_s, linewidth=2.0, label="Generated q50")
    ax.plot(s90, y_s, linestyle=":", label="Generated q90")

    ax.set_xlabel("Distance (L2)")
    ax.set_ylabel("CDF")
    ax.set_title("Distance CDF from Random Query to Dataset")
    ax.grid(alpha=0.25)
    ax.legend()

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
