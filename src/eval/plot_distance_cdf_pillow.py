from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw
import yaml

from src.data.sift1m_dataset import load_descriptors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot distance CDF percentile curves (10/50/90) for SIFT vs generated data."
    )
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument("--real-format", type=str, default="auto", choices=["auto", "npy", "fvecs"])
    parser.add_argument("--synthetic-path", type=str, required=True)
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        help="Optional YAML config path used to auto-label generator model settings.",
    )
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--num-targets", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config-label",
        type=str,
        default="",
        help="Optional label shown in plot subtitle (e.g. config or run name).",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="",
        help="Optional extra caption line shown under title.",
    )
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


def l2_normalize(x: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def sampled_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k >= n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)


def query_cdf_quantiles(
    x: np.ndarray,
    num_queries: int,
    num_targets: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_idx = sampled_indices(x.shape[0], num_queries, rng)
    t_idx = sampled_indices(x.shape[0], num_targets, rng)
    queries = x[q_idx]
    targets = x[t_idx]

    dists = np.linalg.norm(queries[:, None, :] - targets[None, :, :], axis=2)
    t_lookup = {int(v): i for i, v in enumerate(t_idx.tolist())}

    rows = []
    for i, q_global in enumerate(q_idx.tolist()):
        row = dists[i]
        if q_global in t_lookup:
            row = np.delete(row, t_lookup[q_global])
        rows.append(np.sort(row))

    min_len = min(len(r) for r in rows)
    stacked = np.stack([r[:min_len] for r in rows], axis=0)
    y = np.linspace(0.0, 1.0, min_len, endpoint=False)
    return (
        y,
        np.quantile(stacked, 0.10, axis=0),
        np.quantile(stacked, 0.50, axis=0),
        np.quantile(stacked, 0.90, axis=0),
    )


def data_to_pixels(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    width: int,
    height: int,
    margin_left: int,
    margin_right: int,
    margin_top: int,
    margin_bottom: int,
) -> np.ndarray:
    px_w = width - margin_left - margin_right
    px_h = height - margin_top - margin_bottom
    x_norm = (x - x_min) / max(x_max - x_min, 1.0e-12)
    y_norm = y
    px = margin_left + x_norm * px_w
    py = height - margin_bottom - y_norm * px_h
    return np.stack([px, py], axis=1)


def draw_curve(draw: ImageDraw.ImageDraw, pts: np.ndarray, color: Tuple[int, int, int], width: int = 2) -> None:
    if len(pts) < 2:
        return
    draw.line([tuple(p.tolist()) for p in pts], fill=color, width=width)


def generator_model_label_from_config(config_path: str) -> str:
    if not config_path:
        return ""
    path = Path(config_path)
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    model = cfg.get("model", {})
    latent_dim = model.get("latent_dim")
    g_hidden = model.get("generator_hidden_dims")
    if latent_dim is None and g_hidden is None:
        return ""
    return f"G(latent={latent_dim}, hidden={g_hidden})"


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    real = load_descriptors(Path(args.real_path), file_format=args.real_format)
    synthetic = np.load(args.synthetic_path).astype(np.float32, copy=False)
    real = l2_normalize(real.astype(np.float32, copy=False))
    synthetic = l2_normalize(synthetic)

    y_r, r10, r50, r90 = query_cdf_quantiles(real, args.num_queries, args.num_targets, rng)
    y_s, s10, s50, s90 = query_cdf_quantiles(synthetic, args.num_queries, args.num_targets, rng)

    x_min = float(min(r10.min(), r50.min(), r90.min(), s10.min(), s50.min(), s90.min()))
    x_max = float(max(r10.max(), r50.max(), r90.max(), s10.max(), s50.max(), s90.max()))

    width, height = 1200, 800
    ml, mr, mt, mb = 100, 60, 60, 90
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Axes
    draw.line([(ml, mt), (ml, height - mb)], fill=(0, 0, 0), width=2)
    draw.line([(ml, height - mb), (width - mr, height - mb)], fill=(0, 0, 0), width=2)

    # Grid lines
    for t in np.linspace(0.0, 1.0, 6):
        y = int(height - mb - t * (height - mt - mb))
        draw.line([(ml, y), (width - mr, y)], fill=(230, 230, 230), width=1)
    for t in np.linspace(0.0, 1.0, 6):
        x = int(ml + t * (width - ml - mr))
        draw.line([(x, mt), (x, height - mb)], fill=(235, 235, 235), width=1)

    colors: Dict[str, Tuple[int, int, int]] = {
        "r10": (31, 119, 180),
        "r50": (31, 119, 180),
        "r90": (31, 119, 180),
        "s10": (214, 39, 40),
        "s50": (214, 39, 40),
        "s90": (214, 39, 40),
    }

    draw_curve(
        draw,
        data_to_pixels(r10, y_r, x_min, x_max, width, height, ml, mr, mt, mb),
        colors["r10"],
        width=2,
    )
    draw_curve(
        draw,
        data_to_pixels(r50, y_r, x_min, x_max, width, height, ml, mr, mt, mb),
        colors["r50"],
        width=4,
    )
    draw_curve(
        draw,
        data_to_pixels(r90, y_r, x_min, x_max, width, height, ml, mr, mt, mb),
        colors["r90"],
        width=2,
    )
    draw_curve(
        draw,
        data_to_pixels(s10, y_s, x_min, x_max, width, height, ml, mr, mt, mb),
        colors["s10"],
        width=2,
    )
    draw_curve(
        draw,
        data_to_pixels(s50, y_s, x_min, x_max, width, height, ml, mr, mt, mb),
        colors["s50"],
        width=4,
    )
    draw_curve(
        draw,
        data_to_pixels(s90, y_s, x_min, x_max, width, height, ml, mr, mt, mb),
        colors["s90"],
        width=2,
    )

    # Labels and simple legend
    draw.text((width // 2 - 110, 18), "Distance CDF from Random Query to Dataset", fill=(0, 0, 0))
    subtitle_parts = []
    model_label = generator_model_label_from_config(args.config_path)
    if model_label:
        subtitle_parts.append(model_label)
    if args.config_label:
        subtitle_parts.append(f"config={args.config_label}")
    subtitle_parts.append(f"queries={args.num_queries}")
    subtitle_parts.append(f"targets={args.num_targets}")
    subtitle = " | ".join(subtitle_parts)
    draw.text((width // 2 - 180, 36), subtitle, fill=(0, 0, 0))
    if args.caption:
        draw.text((width // 2 - 220, 52), args.caption, fill=(0, 0, 0))
    draw.text((width // 2 - 45, height - 35), "Distance (L2)", fill=(0, 0, 0))
    draw.text((20, height // 2), "CDF", fill=(0, 0, 0))

    legend_y = mt + 10
    draw.rectangle([(ml + 20, legend_y), (ml + 40, legend_y + 20)], fill=(31, 119, 180))
    draw.text((ml + 48, legend_y + 2), "SIFT q10/q50/q90", fill=(0, 0, 0))
    draw.rectangle([(ml + 220, legend_y), (ml + 240, legend_y + 20)], fill=(214, 39, 40))
    draw.text((ml + 248, legend_y + 2), "Generated q10/q50/q90", fill=(0, 0, 0))

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"Saved plot to {out}")
    print(
        f"x_range=[{x_min:.6f}, {x_max:.6f}] curves: SIFT(q10/q50/q90)=blue, Generated(q10/q50/q90)=red"
    )


if __name__ == "__main__":
    main()
