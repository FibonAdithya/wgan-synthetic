from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

from src.data.sift1m_dataset import (
    PreprocessState,
    apply_preprocess,
    load_descriptors,
    train_holdout_split,
)
from src.models.generator import build_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthetic vectors against real descriptors.")
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--format", type=str, default="auto", choices=["auto", "npy", "fvecs"])
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--gamma", type=float, default=1.0, help="RBF gamma for MMD.")
    return parser.parse_args()


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def load_generator(config: Dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    data_cfg = config["data"]
    model_cfg = config["model"]
    generator = build_generator(
        model_cfg, output_dim=int(data_cfg["descriptor_dim"])
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator


def sample_fake(
    generator: torch.nn.Module,
    latent_dim: int,
    n: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    out = []
    generated = 0
    with torch.no_grad():
        while generated < n:
            cur = min(batch_size, n - generated)
            z = torch.randn(cur, latent_dim, device=device)
            x = generator(z)
            x = x / torch.clamp(torch.linalg.vector_norm(x, dim=1, keepdim=True), min=1.0e-8)
            x = x.cpu().numpy().astype(np.float32, copy=False)
            out.append(x)
            generated += cur
    return np.concatenate(out, axis=0)


def covariance_fro(real: np.ndarray, fake: np.ndarray) -> float:
    cov_real = np.cov(real, rowvar=False)
    cov_fake = np.cov(fake, rowvar=False)
    return float(np.linalg.norm(cov_real - cov_fake, ord="fro"))


def mean_var_l2(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    out = {
        "mean_l2": float(np.linalg.norm(real.mean(axis=0) - fake.mean(axis=0))),
        "var_l2": float(np.linalg.norm(real.var(axis=0) - fake.var(axis=0))),
    }
    return out


def mmd_rbf(real: np.ndarray, fake: np.ndarray, gamma: float) -> float:
    max_samples = 5000
    if real.shape[0] > max_samples or fake.shape[0] > max_samples:
        rng = np.random.default_rng(0)
        idx_r = rng.choice(real.shape[0], size=min(max_samples, real.shape[0]), replace=False)
        idx_f = rng.choice(fake.shape[0], size=min(max_samples, fake.shape[0]), replace=False)
        real = real[idx_r]
        fake = fake[idx_f]
    k_xx = rbf_kernel(real, real, gamma=gamma)
    k_yy = rbf_kernel(fake, fake, gamma=gamma)
    k_xy = rbf_kernel(real, fake, gamma=gamma)
    mmd2 = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return float(mmd2)


def _hist_density_from_pairwise(
    a: np.ndarray,
    b: np.ndarray,
    edges: np.ndarray,
    block_size: int,
) -> np.ndarray:
    counts = np.zeros(len(edges) - 1, dtype=np.float64)
    for i in range(0, a.shape[0], block_size):
        ai = a[i : i + block_size]
        for j in range(0, b.shape[0], block_size):
            bj = b[j : j + block_size]
            # Chunked pairwise distances avoid allocating [N, N, D] for full inputs.
            d = np.linalg.norm(ai[:, None, :] - bj[None, :, :], axis=-1).ravel()
            c, _ = np.histogram(d, bins=edges)
            counts += c

    widths = np.diff(edges)
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts)
    return counts / (total * widths)


def pairwise_hist_l1(real: np.ndarray, fake: np.ndarray, bins: int = 50) -> float:
    rng = np.random.default_rng(0)
    n = min(5000, real.shape[0], fake.shape[0])
    idx_r = rng.choice(real.shape[0], size=n, replace=False)
    idx_f = rng.choice(fake.shape[0], size=n, replace=False)
    rr = real[idx_r]
    ff = fake[idx_f]

    # Dynamic max bound via triangle inequality; avoids expensive full-distance pre-pass.
    max_rr = float(2.0 * np.max(np.linalg.norm(rr, axis=1)))
    max_rf = float(np.max(np.linalg.norm(rr, axis=1)) + np.max(np.linalg.norm(ff, axis=1)))
    max_dist = max(max_rr, max_rf, 1.0e-6)
    edges = np.linspace(0.0, max_dist, bins + 1, dtype=np.float64)

    block_size = 256
    hist_rr = _hist_density_from_pairwise(rr, rr, edges=edges, block_size=block_size)
    hist_rf = _hist_density_from_pairwise(rr, ff, edges=edges, block_size=block_size)
    return float(np.abs(hist_rr - hist_rf).sum())


def knn_recall(real_train: np.ndarray, real_holdout: np.ndarray, fake: np.ndarray, k: int = 10) -> float:
    k = min(k, max(1, real_train.shape[0] - 1))
    nn_real = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=1).fit(real_train)
    nn_fake = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=1).fit(fake)
    d_real, _ = nn_real.kneighbors(real_holdout, return_distance=True)
    d_fake, _ = nn_fake.kneighbors(real_holdout, return_distance=True)
    threshold = d_real[:, -1]
    recall = (d_fake[:, 0] <= threshold).mean()
    return float(recall)


def ann_proxy_recall(real_train: np.ndarray, fake_train: np.ndarray, queries: np.ndarray, k: int = 10) -> float:
    k = min(k, max(1, real_train.shape[0] - 1), max(1, fake_train.shape[0] - 1))
    nn_real = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=1).fit(real_train)
    nn_fake = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=1).fit(fake_train)
    d_real, _ = nn_real.kneighbors(queries, return_distance=True)
    d_fake, _ = nn_fake.kneighbors(queries, return_distance=True)
    ratio = np.mean(np.clip(d_fake.mean(axis=1) / (d_real.mean(axis=1) + 1.0e-8), 0.0, 10.0))
    return float(np.exp(-abs(ratio - 1.0)))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = get_device(config["device"])
    checkpoint_path = Path(args.checkpoint)
    generator = load_generator(config=config, checkpoint_path=checkpoint_path, device=device)

    real = load_descriptors(Path(args.real_path), file_format=args.format)
    seed = int(config["seed"])
    holdout_fraction = float(config["data"]["holdout_fraction"])
    real_train_raw, real_holdout_raw = train_holdout_split(real, holdout_fraction=holdout_fraction, seed=seed)

    preprocess_payload = None
    meta_path = checkpoint_path.parent / "run_metadata.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            preprocess_payload = json.load(f).get("preprocess_state")
    if preprocess_payload is None:
        raise ValueError(
            "Could not load preprocess_state from run_metadata.json next to checkpoint."
        )
    preprocess_state = PreprocessState.from_serializable(preprocess_payload)

    real_train = apply_preprocess(real_train_raw, preprocess_state)
    real_holdout = apply_preprocess(real_holdout_raw, preprocess_state)

    n = min(args.num_samples, real_holdout.shape[0])
    real_eval = real_holdout[:n]
    fake_eval = sample_fake(
        generator=generator,
        latent_dim=int(config["model"]["latent_dim"]),
        n=n,
        batch_size=int(config["training"]["batch_size"]),
        device=device,
    )

    metrics = {}
    metrics.update(mean_var_l2(real_eval, fake_eval))
    metrics["cov_fro"] = covariance_fro(real_eval, fake_eval)
    metrics["mmd_rbf"] = mmd_rbf(real_eval, fake_eval, gamma=float(args.gamma))
    metrics["pairwise_hist_l1"] = pairwise_hist_l1(real_eval, fake_eval)
    metrics["knn_recall"] = knn_recall(real_train, real_eval, fake_eval, k=10)
    metrics["ann_proxy_recall"] = ann_proxy_recall(real_train, fake_eval, real_eval, k=10)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
