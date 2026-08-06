"""Render real and generated SIFT descriptors as a grid of orientation glyphs.

Every other panel in `eda_report` is an aggregate over tens of thousands of
vectors. All of them can look healthy while the generator produces
descriptors that are structurally wrong, because a matched marginal says
nothing about whether the 128 numbers form a plausible gradient histogram.
This figure shows individual descriptors instead.

Two rows of real descriptors are drawn, not one. Without a sense of how much
two real descriptors differ from each other, a variant row below them is just
a vibe; the real-a/real-b gap is the baseline the rest is read against.

Samples are drawn at random under a fixed seed, never selected.

Example:
    python -m src.eval.plot_descriptor_grid \
        --real-path data/sift_base.npy \
        --output-dir runs/descriptor_grid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import yaml

from src.data.dataset import load_descriptors
from src.eval import compare_variants as cv
from src.eval import eda_report
from src.eval.descriptor_glyph import DESCRIPTOR_DIM
from src.eval.eda import series as eda_series
from src.eval.evaluate_distribution import get_device, load_generator
from src.train.train_wgan_gp import sample_generator

# Geometry, the negative-ray treatment and the figure itself come from
# `eda_report`, which draws the same panel from materialised arrays.
REAL_COLORS = eda_report.GLYPH_REAL_COLORS
VARIANT_COLORS = ("#ff7f0e", "#2ca02c", "#9467bd", "#8c564b")
# Any variant not in `cv.VARIANTS`. Deliberately outside the palette above:
# wrapping the index into it instead handed an unknown name v0's orange, and
# two identically-coloured rows are unreadable in a figure whose whole job is
# row-to-row comparison. Grey also reads as "unplaced", which is accurate.
UNKNOWN_VARIANT_COLOR = "#7f7f7f"

REPORT_NAME = "descriptor_grid.html"


def check_finite(arr: np.ndarray, source: str) -> None:
    """Refuse a descriptor array holding NaN or inf.

    `glyph_segments` routes a NaN bin into the negative-ray arrays (`value >
    0` is False for NaN) without tripping its zero-length guard (`NaN <= 0.0`
    is also False), so a single NaN bin invents a red "negative" trace for an
    otherwise well-behaved row. `inf` is worse: it survives the `> 0.0`
    filter in `shared_scale`, and `np.percentile` returns NaN, blanking the
    whole figure. Refusing loudly here matches this module's stance on
    centering/whitening and a wrong width -- all three are silent-lie risks,
    not something to paper over by dropping a few rows.
    """
    bad = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
    if bad:
        raise ValueError(
            f"{source} contains {bad} non-finite value(s) (NaN or inf); "
            "refusing to plot, since a non-finite bin would draw a "
            "spurious or blank glyph"
        )


def pick_real_rows(
    real: np.ndarray, num_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw two disjoint random rows of real descriptors."""
    needed = 2 * num_samples
    if real.shape[0] < needed:
        raise ValueError(
            f"need at least {needed} real vectors for two rows of "
            f"{num_samples}, got {real.shape[0]}"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(real.shape[0], size=needed, replace=False)
    return real[idx[:num_samples]], real[idx[num_samples:]]


def build_figure(rows: list[tuple[str, np.ndarray, str]]) -> go.Figure:
    """Assemble the glyph grid.

    The figure itself lives in `eda_report`, which draws the same panel from
    already-materialised arrays; this module's job is getting rows out of
    generator checkpoints. One implementation, so the two cannot drift.
    """
    return eda_report.fig_descriptor_glyphs(rows)


def write_report(
    fig: go.Figure, out_dir: Path, plotlyjs_mode: str, write_png: bool
) -> Path:
    """Write the HTML report, and optionally a static PNG beside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    head = eda_report.plotlyjs_head(plotlyjs_mode, out_dir)
    body = fig.to_html(full_html=False, include_plotlyjs=False)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>SIFT descriptor glyph grid</title>"
        f"{head}</head><body>{body}</body></html>"
    )
    path = out_dir / REPORT_NAME
    path.write_text(html, encoding="utf-8")
    if write_png:
        try:
            eda_report.export_pngs([("descriptor grid", "", fig)], out_dir)
        except Exception as exc:  # kaleido needs a Chrome binary
            print(f"skipping PNG export: {exc}")
    return path


def check_preprocess(config: dict, name: str) -> None:
    """Refuse a run whose preprocessing destroys the bin-to-dimension map.

    `center: false, whiten: false` across all four current variant configs is
    the only reason dimension k still means "cell i, orientation bin j".
    Centering shifts by a constant and whitening applies a dense linear mix;
    under either, the glyph becomes a picture of mixed bins that still looks
    entirely plausible. Better to refuse than to draw a silent lie.

    A missing preprocess block means the `PreprocessConfig` defaults, both of
    which are False.
    """
    preprocess = (config.get("data") or {}).get("preprocess") or {}
    enabled = [key for key in ("center", "whiten") if bool(preprocess.get(key, False))]
    if enabled:
        raise ValueError(
            f"variant {name} was trained with {' and '.join(enabled)} enabled, so "
            "its dimensions no longer map to (cell, orientation bin) and the "
            "glyph would be meaningless. Refusing to plot it."
        )


def variant_color(name: str) -> str:
    """Colour a variant by its identity in `cv.VARIANTS`, not by position in
    the machine-dependent resolved list.

    `variant_rows` only sees the subset of `cv.VARIANTS` whose checkpoint is
    on this machine, so indexing `found` directly would make v1's colour
    depend on whether v0's checkpoint happens to be present -- the same
    machine-dependence `cv.variant_seed` exists to prevent for sampling. A
    name absent from `cv.VARIANTS` gets `UNKNOWN_VARIANT_COLOR` rather than
    raising, and rather than wrapping back onto a known variant's colour.
    """
    names = [variant.name for variant in cv.VARIANTS]
    if name not in names:
        return UNKNOWN_VARIANT_COLOR
    return VARIANT_COLORS[names.index(name) % len(VARIANT_COLORS)]


def variant_rows(
    root: Path, num_samples: int, seed: int
) -> list[tuple[str, np.ndarray, str]]:
    """Sample every resolvable variant checkpoint into one row each.

    A variant whose artifacts are not on this machine is skipped with a
    message rather than aborting: checkpoints usually live on the training
    box, and a partial poster is still worth reading.
    """
    found, skipped = cv.resolve_variants(cv.VARIANTS, root)
    for variant, reason in skipped:
        # check_preprocess even though this variant is skipped. `resolve_variants`
        # also drops a centered/whitened run whose fitted transform cannot be
        # recovered, and such a variant would otherwise slip past the refusal
        # below and go silently absent from the poster instead of loudly
        # refused. The objection is to the training config, not to the run
        # being unsamplable.
        #
        # Only for a variant that is otherwise plottable, though: one skipped
        # because its checkpoint is not on this machine cannot be drawn under
        # any preprocessing, and raising for it would make a missing checkpoint
        # fatal -- the machine-dependent abort this skip list exists to avoid.
        run_dir = root / variant.run_dir
        run_config = run_dir / cv.RUN_CONFIG_NAME
        if (run_dir / cv.CHECKPOINT_NAME).exists() and run_config.exists():
            check_preprocess(
                yaml.safe_load(run_config.read_text(encoding="utf-8")), variant.name
            )
        print(f"skipping {variant.name}: {reason}")
    if not found:
        print("no variant checkpoints resolved; rendering the real rows only")

    rows: list[tuple[str, np.ndarray, str]] = []
    for variant in found:
        run_dir = root / variant.run_dir
        config = yaml.safe_load(
            (run_dir / cv.RUN_CONFIG_NAME).read_text(encoding="utf-8")
        )
        check_preprocess(config, variant.name)
        device = get_device(config["device"])
        generator = load_generator(config, run_dir / cv.CHECKPOINT_NAME, device)
        # GatedGenerator samples its gate in eval() mode too, so the seed is
        # what makes a row reproducible. Keying off the variant name keeps a
        # row identical whether or not other checkpoints are on this machine.
        torch.manual_seed(cv.variant_seed(seed, variant.name))
        samples = sample_generator(
            generator,
            num_samples=num_samples,
            latent_dim=int(config["model"]["latent_dim"]),
            batch_size=num_samples,
            device=device,
        )
        if samples.shape[1] != DESCRIPTOR_DIM:
            raise ValueError(
                f"variant {variant.name} generates {samples.shape[1]}-dimensional "
                f"vectors; the glyph mapping needs {DESCRIPTOR_DIM}"
            )
        check_finite(samples, f"variant {variant.name}")
        rows.append((variant.name, samples, variant_color(variant.name)))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument(
        "--real-format", type=str, default="auto", choices=["auto", "npy", "fvecs"]
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repo root that variant config and run paths resolve against.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Descriptors per row. Each row is an independent random draw.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument(
        "--plotlyjs", type=str, default="inline", choices=["inline", "cdn", "directory"]
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    real = load_descriptors(Path(args.real_path), args.real_format)
    if real.shape[1] != DESCRIPTOR_DIM:
        raise ValueError(
            f"the glyph mapping is only defined for {DESCRIPTOR_DIM}-dimensional "
            f"descriptors; {args.real_path} holds {real.shape[1]}-dimensional ones"
        )
    # Select the handful of rows we plot before normalising or checking them,
    # not after -- selection is purely index-based, so touching the other ~1M
    # rows we never look at is wasted work, and refusing the whole poster over
    # a non-finite value in one of them buys nothing. The guard is about what
    # gets drawn.
    row_a, row_b = pick_real_rows(real, args.num_samples, args.seed)
    check_finite(row_a, str(args.real_path))
    check_finite(row_b, str(args.real_path))
    # `eda.series`'s normaliser rather than a local one: this has to match the
    # training preprocessing, and a third copy of that rule is a third place
    # for it to drift out of step with `dataset.apply_preprocess`.
    row_a = eda_series.maybe_l2_normalize(row_a, "l2")
    row_b = eda_series.maybe_l2_normalize(row_b, "l2")

    rows: list[tuple[str, np.ndarray, str]] = [
        ("real-a", row_a, REAL_COLORS[0]),
        ("real-b", row_b, REAL_COLORS[1]),
    ]

    # Real rows are normalised just above; generated rows arrive already
    # normalised from `sample_generator`. Both have to hold, because the ray
    # scale is one percentile over the two pooled together -- if either side
    # drifts off unit norm the figure silently rescales and stops being a
    # comparison. Pinned by
    # `test_generated_rows_arrive_unit_norm_like_the_real_rows`.
    rows.extend(variant_rows(Path(args.root), args.num_samples, args.seed))

    fig = build_figure(rows)
    return write_report(fig, Path(args.output_dir), args.plotlyjs, not args.no_png)


def main() -> None:
    print(run(parse_args()))


if __name__ == "__main__":
    main()
