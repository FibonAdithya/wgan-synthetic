from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.sift1m_dataset import load_descriptors, train_holdout_split
from src.eval.evaluate_distribution import (
    ann_proxy_recall,
    covariance_fro,
    knn_recall,
    mean_var_l2,
    mmd_rbf,
    pairwise_hist_l1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic file directly against real descriptor file."
    )
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument("--synthetic-path", type=str, required=True)
    parser.add_argument("--real-format", type=str, default="auto", choices=["auto", "npy", "fvecs"])
    parser.add_argument(
        "--synthetic-format", type=str, default="auto", choices=["auto", "npy", "fvecs"]
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-l2-normalize",
        action="store_true",
        help="Disable per-vector L2 normalization before evaluation.",
    )
    return parser.parse_args()


def l2_normalize(x: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def random_sample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n >= x.shape[0]:
        return x
    idx = rng.choice(x.shape[0], size=n, replace=False)
    return x[idx]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    real = load_descriptors(Path(args.real_path), file_format=args.real_format).astype(
        np.float32, copy=False
    )
    synthetic = load_descriptors(
        Path(args.synthetic_path), file_format=args.synthetic_format
    ).astype(np.float32, copy=False)

    if real.shape[1] != synthetic.shape[1]:
        raise ValueError(
            f"Dimension mismatch: real dim={real.shape[1]} vs synthetic dim={synthetic.shape[1]}"
        )

    if not args.skip_l2_normalize:
        real = l2_normalize(real)
        synthetic = l2_normalize(synthetic)

    real_train, real_holdout = train_holdout_split(
        real, holdout_fraction=args.holdout_fraction, seed=args.seed
    )
    n = min(args.num_samples, real_holdout.shape[0], synthetic.shape[0])
    real_eval = random_sample(real_holdout, n=n, rng=rng)
    fake_eval = random_sample(synthetic, n=n, rng=rng)

    metrics = {}
    metrics.update(mean_var_l2(real_eval, fake_eval))
    metrics["cov_fro"] = covariance_fro(real_eval, fake_eval)
    metrics["mmd_rbf"] = mmd_rbf(real_eval, fake_eval, gamma=float(args.gamma))
    metrics["pairwise_hist_l1"] = pairwise_hist_l1(real_eval, fake_eval)
    metrics["knn_recall"] = knn_recall(real_train, real_eval, fake_eval, k=10)
    metrics["ann_proxy_recall"] = ann_proxy_recall(real_train, fake_eval, real_eval, k=10)
    metrics["num_samples_used"] = int(n)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
