"""Overlay every trained deep variant on real DEEP in one EDA report.

Usage:

    python -m src.deep.report \
        --real-path data/deep96_1m.npy \
        --output-dir runs/eda_deep

The variant table and the sampling call are the only things that differ from
src/eval/compare_variants.py. Variant, resolve_variants and variant_seed are
imported from there rather than copied: they are dataset-agnostic. Sampling
cannot be shared, because deep samples must be returned to original
coordinates by src/deep/sample.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.deep.sample import sample_variant
from src.eval import eda_report
from src.eval.compare_variants import Variant, resolve_variants, variant_seed

# Ordered so the report legend reads as a progression. Each entry is one
# config delta from the previous.
DEEP_VARIANTS: Tuple[Variant, ...] = (
    Variant("v0", "configs/deep_gan_v0.yaml", "runs/deep_gan_v0"),
    Variant("v1", "configs/deep_gan_v1.yaml", "runs/deep_gan_v1"),
    Variant("v2", "configs/deep_gan_v2.yaml", "runs/deep_gan_v2"),
)


def generate_samples(
    variant: Variant,
    root: Path,
    num_samples: int,
    batch_size: int,
    out_dir: Path,
    seed: int,
) -> Path:
    """Sample a deep variant to an .npy in original coordinates."""
    x = sample_variant(
        root / variant.run_dir,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=variant_seed(seed, variant.name),
    )
    out_path = out_dir / f"{variant.name}.npy"
    np.save(out_path, x)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument(
        "--real-format", type=str, default="auto", choices=["auto", "npy", "fvecs"]
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-vectors", type=int, default=50000)
    parser.add_argument("--num-pairs", type=int, default=200000)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--ann-k", type=int, default=eda_report.ANN_K_DEFAULT)
    parser.add_argument("--ann-hub-k", type=int, default=eda_report.ANN_HUB_K_DEFAULT)
    parser.add_argument(
        "--ann-max-rows", type=int, default=eda_report.ANN_MAX_ROWS_DEFAULT
    )
    parser.add_argument("--ivf-nlist", type=int, default=eda_report.IVF_NLIST_DEFAULT)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--top-divergent", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument(
        "--plotlyjs", type=str, default="inline", choices=["inline", "cdn", "directory"]
    )
    return parser.parse_args()


def build_report_args(
    args: argparse.Namespace, specs: List[str]
) -> argparse.Namespace:
    """Build the Namespace eda_report.run expects from our own parsed args.

    preprocess="l2" because both sides are already unit-norm: real DEEP ships
    that way, and every variant's samples come off normalize_l2.
    """
    return argparse.Namespace(
        real_path=args.real_path,
        real_format=args.real_format,
        synthetic_path=specs,
        synthetic_format="npy",
        output_dir=args.output_dir,
        preprocess="l2",
        max_vectors=args.max_vectors,
        num_pairs=args.num_pairs,
        knn=args.knn,
        ann_k=args.ann_k,
        ann_hub_k=args.ann_hub_k,
        ann_max_rows=args.ann_max_rows,
        ivf_nlist=args.ivf_nlist,
        bins=args.bins,
        top_divergent=args.top_divergent,
        seed=args.seed,
        no_png=args.no_png,
        plotlyjs=args.plotlyjs,
    )


def main() -> None:
    args = parse_args()
    num_samples = args.num_samples if args.num_samples is not None else args.max_vectors
    root = Path(args.root)
    out_dir = Path(args.output_dir)

    found, skipped = resolve_variants(DEEP_VARIANTS, root)
    for variant, reason in skipped:
        print(f"skipping {variant.name}: {reason}")
    if not found:
        raise SystemExit(
            "No deep variant has both a checkpoint and a run config on this "
            "machine. Copy them from the training box, or pass --root at the "
            "tree holding them."
        )

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    specs = []
    for variant in found:
        print(f"sampling {variant.name} from {variant.run_dir}")
        path = generate_samples(
            variant, root, num_samples, args.batch_size, samples_dir, seed=args.seed
        )
        specs.append(f"{variant.name}={path}")

    report_path = eda_report.run(build_report_args(args, specs))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
