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
import json
from pathlib import Path

from src.eval import ann_difficulty
from src.eval.eda import html, metrics, panels
from src.eval.eda.config import (
    ANN_HUB_K_DEFAULT,
    ANN_K_DEFAULT,
    ANN_MAX_ROWS_DEFAULT,
    GLYPH_SAMPLES_DEFAULT,
    IVF_NLIST_DEFAULT,
    KNN_MAX_ROWS_DEFAULT,
    EdaConfig,
)
from src.eval.eda.series import load_series


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


# --------------------------------------------------------------------------
# data prep
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EdaConfig.from_args(args)
    series = load_series(cfg)
    has_synth = len(series) > 1
    ann_metrics = {
        s.name: ann_difficulty.compute(
            s.x,
            k=args.ann_k,
            k_hub=args.ann_hub_k,
            nlist=args.ivf_nlist,
            max_rows=args.ann_max_rows,
            seed=args.seed,
        )
        for s in series
    }
    stats = [
        metrics.summary_stats(
            s,
            args.knn,
            args.num_pairs,
            args.seed,
            args.knn_max_rows,
            ann_metrics[s.name],
        )
        for s in series
    ]

    ctx = panels.Context(
        config=cfg,
        series=series,
        ann_metrics=ann_metrics,
        divergence=(
            metrics.dimension_divergence(series, cfg.top_divergent)
            if len(series) > 1
            else None
        ),
    )
    sections = [
        (panel.resolve_title(ctx), panel.resolve_note(ctx), fig)
        for panel in panels.PANELS
        if (fig := panel.build(ctx)) is not None
    ]
    worst_dims = ctx.divergence.worst if ctx.divergence else {}

    synth_desc = (
        " &middot; ".join(f"{s.name}" for s in series if not s.is_real)
        if has_synth
        else "no synthetic overlay"
    )
    meta_html = (
        f'<div class="meta">real: <code>{args.real_path}</code>'
        f" &middot; overlays: {synth_desc}"
        f" &middot; preprocess: <code>{args.preprocess}</code>"
        f" &middot; subsampled to {series[0].x.shape[0]} rows</div>"
        + html.stats_table_html(stats)
    )

    report_html = html.build_report(
        sections, meta_html, html.plotlyjs_head(args.plotlyjs, out_dir)
    )
    report_path = out_dir / "eda_report.html"
    report_path.write_text(report_html, encoding="utf-8")

    png_paths: list[str] = []
    png_error = None
    if not args.no_png:
        try:
            png_paths = html.export_pngs(sections, out_dir)
        except Exception as exc:  # kaleido needs a Chrome binary
            png_error = str(exc)

    summary = {
        "real_path": args.real_path,
        "synthetic_paths": args.synthetic_path or [],
        "preprocess": args.preprocess,
        "seed": args.seed,
        "ann_settings": {
            "k": args.ann_k,
            "k_hub": args.ann_hub_k,
            "max_rows": args.ann_max_rows,
            "nlist": args.ivf_nlist,
        },
        "knn_settings": {"k": args.knn, "max_rows": args.knn_max_rows},
        "stats": stats,
        "worst_dimensions": worst_dims,
        "report_html": str(report_path),
        "png_files": png_paths,
    }
    if png_error:
        summary["png_error"] = png_error
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Wrote {report_path}")
    if png_paths:
        print(f"Wrote {len(png_paths)} PNGs to {out_dir / 'png'}")
    elif png_error:
        print(f"PNG export skipped: {png_error}")
    print(f"Wrote {out_dir / 'summary.json'}")

    return report_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
