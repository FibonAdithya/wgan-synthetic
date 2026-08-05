from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.data.dataset import load_descriptors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create t-SNE/UMAP embedding visualizations for real and synthetic datasets."
    )
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument(
        "--real-format", type=str, default="auto", choices=["auto", "npy", "fvecs"]
    )
    parser.add_argument("--synthetic-path", type=str, required=True)
    parser.add_argument(
        "--synthetic-format", type=str, default="auto", choices=["auto", "npy", "fvecs"]
    )
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"])
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def l2_normalize(x: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def sample_rows(
    x: np.ndarray, sample_size: int, rng: np.random.Generator
) -> np.ndarray:
    if sample_size >= x.shape[0]:
        return x
    idx = rng.choice(x.shape[0], size=sample_size, replace=False)
    return x[idx]


def compute_embedding(
    x: np.ndarray,
    method: str,
    seed: int,
    perplexity: float,
) -> np.ndarray:
    n_components = min(50, x.shape[1], x.shape[0] - 1)
    if n_components >= 2:
        x = PCA(n_components=n_components, random_state=seed).fit_transform(x)

    if method == "tsne":
        effective_perplexity = max(5.0, min(perplexity, (x.shape[0] - 1) / 3.0))
        model = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
            max_iter=1000,
            verbose=0,
        )
        return model.fit_transform(x).astype(np.float32, copy=False)

    if method == "umap":
        try:
            import umap.umap_ as umap
        except Exception as exc:
            raise RuntimeError(
                "UMAP requested but package is not installed. "
                "Install with: .venv/bin/pip install umap-learn"
            ) from exc

        model = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
        )
        return model.fit_transform(x).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported method: {method}")


def draw_scatter(
    points: np.ndarray,
    title: str,
    output_path: Path,
    color: tuple[int, int, int],
) -> None:
    width, height = 1200, 900
    margin = 70
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    x = points[:, 0]
    y = points[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_span = max(x_max - x_min, 1.0e-8)
    y_span = max(y_max - y_min, 1.0e-8)

    draw.rectangle(
        [(margin, margin), (width - margin, height - margin)],
        outline=(0, 0, 0),
        width=2,
    )

    for px, py in points:
        sx = margin + (float(px) - x_min) / x_span * (width - 2 * margin)
        sy = height - margin - (float(py) - y_min) / y_span * (height - 2 * margin)
        r = 2
        draw.ellipse([(sx - r, sy - r), (sx + r, sy + r)], fill=color, outline=color)

    draw.text((margin, 20), title, fill=(0, 0, 0))
    draw.text((margin, height - 40), "Embedding dim 1", fill=(0, 0, 0))
    draw.text((20, margin), "Embedding dim 2", fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real = load_descriptors(Path(args.real_path), file_format=args.real_format).astype(
        np.float32, copy=False
    )
    synthetic = load_descriptors(
        Path(args.synthetic_path), file_format=args.synthetic_format
    ).astype(np.float32, copy=False)

    real = l2_normalize(real)
    synthetic = l2_normalize(synthetic)

    real_sample = sample_rows(real, args.sample_size, rng)
    synth_sample = sample_rows(synthetic, args.sample_size, rng)

    real_emb = compute_embedding(
        real_sample, method=args.method, seed=args.seed, perplexity=args.perplexity
    )
    synth_emb = compute_embedding(
        synth_sample, method=args.method, seed=args.seed + 1, perplexity=args.perplexity
    )

    real_path = output_dir / f"{args.method}_sift_real.png"
    synth_path = output_dir / f"{args.method}_synthetic.png"

    draw_scatter(
        real_emb,
        f"{args.method.upper()} - SIFT Real ({real_sample.shape[0]} samples)",
        real_path,
        (31, 119, 180),
    )
    draw_scatter(
        synth_emb,
        f"{args.method.upper()} - Synthetic ({synth_sample.shape[0]} samples)",
        synth_path,
        (214, 39, 40),
    )

    print(f"Saved real embedding plot: {real_path}")
    print(f"Saved synthetic embedding plot: {synth_path}")


if __name__ == "__main__":
    main()
