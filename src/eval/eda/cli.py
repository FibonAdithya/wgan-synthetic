# This docstring is the CLI's --help text: parse_args passes it as
# description=__doc__. Editing it for tidiness changes the CLI contract.
"""Exploratory data analysis for SIFT1M descriptors, with optional synthetic overlays.

Complements the metric-driven scripts in this package (evaluate_distribution,
plot_distance_cdf, plot_embedding_clusters) by showing raw distributional shape
instead of scalar summaries. The point is to be able to reject a generator by
eye even when the critic cannot separate the two sets -- a weak critic produces
good-looking Wasserstein estimates over samples whose marginals are obviously
wrong.

Any number of synthetic sets can be overlaid on the real data at once, so a
baseline and one or more improved variants can be read against SIFT and against
each other in a single figure.

Emits one self-contained HTML report plus per-figure PNGs and a summary.json.

Example:
    python -m src.eval.eda_report \
        --real-path data/sift_base.npy \
        --synthetic-path baseline=runs/long_baseline/samples.npy \
        --synthetic-path improved=runs/long_improved/samples.npy \
        --output-dir runs/eda_compare
"""

from __future__ import annotations

import argparse

from src.eval.eda.config import (
    ANN_HUB_K_DEFAULT,
    ANN_K_DEFAULT,
    ANN_MAX_ROWS_DEFAULT,
    GLYPH_SAMPLES_DEFAULT,
    IVF_NLIST_DEFAULT,
    KNN_MAX_ROWS_DEFAULT,
    MAX_PANEL_DIM_DEFAULT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument(
        "--real-format", type=str, default="auto", choices=["auto", "npy", "fvecs"]
    )
    parser.add_argument(
        "--synthetic-path",
        type=str,
        action="append",
        default=None,
        metavar="[LABEL=]PATH",
        help=(
            "Optional, repeatable. Each occurrence adds one synthetic set overlaid "
            "on the real data. Prefix with 'LABEL=' to name it in the legend, "
            "otherwise the file stem is used. Pass several times to compare a "
            "baseline and improved variants against SIFT in one report."
        ),
    )
    parser.add_argument(
        "--synthetic-format", type=str, default="auto", choices=["auto", "npy", "fvecs"]
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--preprocess",
        type=str,
        default="l2",
        choices=["none", "l2"],
        help=(
            "Match the training contract before comparing. Generator output is "
            "unit-norm, so raw SIFT must be L2-normalized for the overlay to mean "
            "anything. Use 'none' to inspect raw integer-valued SIFT."
        ),
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=50000,
        help="Subsample each set to at most this many rows (0 = use all).",
    )
    parser.add_argument("--num-pairs", type=int, default=200000)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument(
        "--ann-k",
        type=int,
        default=ANN_K_DEFAULT,
        help="Neighbours per query for the LID and relative-contrast panels.",
    )
    parser.add_argument(
        "--ann-hub-k",
        type=int,
        default=ANN_HUB_K_DEFAULT,
        help="Neighbour depth for the k-occurrence count behind the hubness panel.",
    )
    parser.add_argument(
        "--ann-max-rows",
        type=int,
        default=ANN_MAX_ROWS_DEFAULT,
        help=(
            "Equal-N truncation for every ANN-difficulty metric. LID, "
            "contrast and hubness all drift with sample count, so every set "
            "must be cut to the same size."
        ),
    )
    parser.add_argument(
        "--knn-max-rows",
        type=int,
        default=KNN_MAX_ROWS_DEFAULT,
        help=(
            "Equal-N truncation for the within-set k-NN distance panel, "
            "which is not an ANN-difficulty panel. k-NN distance shrinks as "
            "sample count grows, so every set must be cut to the same size."
        ),
    )
    parser.add_argument(
        "--ivf-nlist",
        type=int,
        default=IVF_NLIST_DEFAULT,
        help="Cluster count for the IVF cell-balance panel.",
    )
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument(
        "--top-divergent",
        type=int,
        default=16,
        help="How many worst-matching dimensions to call out in the report.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip static PNG export (kaleido needs a Chrome install).",
    )
    parser.add_argument(
        "--glyph-samples",
        type=int,
        default=GLYPH_SAMPLES_DEFAULT,
        help=(
            "Descriptors drawn per row in the descriptor glyph panel. 0 turns "
            "the panel off. The panel is skipped automatically unless every "
            "series is 128-dimensional, since the (cell, orientation bin) "
            "mapping it draws exists only for SIFT descriptors."
        ),
    )
    parser.add_argument(
        "--max-panel-dim",
        type=int,
        default=MAX_PANEL_DIM_DEFAULT,
        help=(
            "Drop the per-dimension marginals and correlation panels above "
            "this width. Both scale with the square of the dimension and stop "
            "being readable well before they stop being cheap, so the default "
            "keeps them for sift, deep, glove and nytimes and drops them for "
            "gist and openai. Raise it to force them back on."
        ),
    )
    parser.add_argument(
        "--plotlyjs",
        type=str,
        default="inline",
        choices=["inline", "cdn", "directory"],
        help=(
            "How to ship plotly.js. 'inline' (default) bundles it so the report "
            "opens offline, costing ~4.5MB on top of the figure data. 'cdn' drops "
            "that but needs internet to view -- roughly 4x smaller, which matters "
            "over a slow link. 'directory' writes plotly.min.js once beside the "
            "report, so several reports in one directory share a single copy."
        ),
    )
    return parser.parse_args()
