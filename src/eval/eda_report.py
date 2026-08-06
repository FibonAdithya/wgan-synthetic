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
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from src.eval import ann_difficulty
from src.eval.eda import figures, glyphs, metrics
from src.eval.eda.config import (
    ANN_HUB_K_DEFAULT,
    ANN_K_DEFAULT,
    ANN_MAX_ROWS_DEFAULT,
    GLYPH_SAMPLES_DEFAULT,
    IVF_NLIST_DEFAULT,
    KNN_MAX_ROWS_DEFAULT,
    EdaConfig,
)
from src.eval.eda.series import Series, load_series


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


def format_stat(value: float | int | None) -> str:
    """Render one statistics-table cell.

    Counts stay counts: `format(1200000, '.6g')` is `1.2e+06`, which reads as
    a measurement rather than a tally, so integer-valued statistics (row
    counts, discarded queries, the measured k and nlist) are formatted as
    plain integers and only the genuinely continuous ones get `.6g`.
    """
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return format(int(value), "d")
    return format(value, ".6g")


def stats_table_html(stats: list[dict]) -> str:
    keys = [k for k in stats[0] if k != "name"]
    header = "".join(f"<th>{s['name']}</th>" for s in stats)
    rows = []
    for k in keys:
        cells = "".join(f"<td>{format_stat(s[k])}</td>" for s in stats)
        rows.append(f"<tr><th>{k}</th>{cells}</tr>")
    return (
        "<table><thead><tr><th>statistic</th>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


REPORT_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
       max-width: 1180px; padding: 24px; color: #1a202c; }
h1 { margin-bottom: 4px; } h2 { margin-top: 36px; border-bottom: 1px solid #e2e8f0;
       padding-bottom: 6px; }
.meta { color: #4a5568; font-size: 14px; margin-bottom: 24px; }
.note { background: #f7fafc; border-left: 3px solid #2b6cb0; padding: 10px 14px;
        margin: 12px 0; font-size: 14px; }
table { border-collapse: collapse; font-size: 14px; margin: 12px 0; }
th, td { border: 1px solid #e2e8f0; padding: 6px 12px; text-align: right; }
thead th { background: #f7fafc; } tbody th { text-align: left; }
"""

CDN_SRC = "https://cdn.plot.ly/plotly-3.7.0.min.js"


def plotlyjs_head(mode: str, out_dir: Path) -> str:
    """Return the <script> tag(s) that make Plotly available to the page."""
    import plotly.offline as pyo

    if mode == "cdn":
        return f'<script charset="utf-8" src="{CDN_SRC}"></script>'
    if mode == "directory":
        asset = out_dir / "plotly.min.js"
        if not asset.exists():
            asset.write_text(pyo.get_plotlyjs(), encoding="utf-8")
        return '<script charset="utf-8" src="plotly.min.js"></script>'
    return f"<script>{pyo.get_plotlyjs()}</script>"


def build_report(
    sections: list[tuple[str, str, go.Figure]],
    meta_html: str,
    head_script: str,
) -> str:
    body = []
    for title, note, fig in sections:
        body.append(f"<h2>{title}</h2>")
        if note:
            body.append(f'<div class="note">{note}</div>')
        body.append(fig.to_html(full_html=False, include_plotlyjs=False))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>SIFT descriptor EDA</title>"
        f"<style>{REPORT_CSS}</style>"
        f"{head_script}"
        "</head><body>"
        "<h1>SIFT descriptor EDA</h1>"
        f"{meta_html}" + "".join(body) + "</body></html>"
    )


def export_pngs(sections: list[tuple[str, str, go.Figure]], out_dir: Path) -> list[str]:
    """Static export is best-effort: kaleido v1 shells out to Chrome."""
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for idx, (title, _, fig) in enumerate(sections, start=1):
        slug = title.lower().replace(" ", "_").replace("/", "-")
        path = png_dir / f"{idx:02d}_{slug}.png"
        fig.write_image(str(path), width=1200, height=fig.layout.height or 460, scale=2)
        written.append(str(path))
    return written


def ann_condition_note(
    series: Sequence[Series],
    ann_metrics: dict[str, ann_difficulty.AnnMetrics],
    attrs: tuple[tuple[str, str], ...],
) -> str:
    """State the actual per-series ANN measurement conditions for `attrs`.

    `attrs` is a sequence of (AnnMetrics field name, display label) pairs,
    e.g. (("num_rows", "rows"), ("k", "k")). When every series in this run
    was measured under the same conditions, one summary sentence is enough.
    When they differ -- e.g. a series with fewer rows than --ann-max-rows
    gets num_rows, k or nlist clamped -- a reader must not be able to mistake
    one series' numbers for all of them, so each series' actual values are
    spelled out instead.
    """
    per_series = {
        s.name: tuple(getattr(ann_metrics[s.name], field) for field, _ in attrs)
        for s in series
    }
    if len(set(per_series.values())) == 1:
        values = next(iter(per_series.values()))
        parts = ", ".join(f"{label}={v}" for (_, label), v in zip(attrs, values))
        return f" Measured with {parts} for every series."
    per_series_text = "; ".join(
        f"{name} ("
        + ", ".join(f"{label}={v}" for (_, label), v in zip(attrs, values))
        + ")"
        for name, values in per_series.items()
    )
    return (
        " Measurement conditions differ across series (a series with fewer "
        f"rows than requested has k and/or nlist clamped): {per_series_text}."
    )


def ann_discarded_note(
    series: Sequence[Series],
    ann_metrics: dict[str, ann_difficulty.AnnMetrics],
) -> str:
    """Call out any series that contributed no queries at all, and why.

    `summary` returns None for `lid_median` and `relative_contrast_median`
    when every query was discarded, and the panel simply has no trace for
    that series. That is the honest answer, but on its own it renders as a
    silent `n/a`. The two ways to get there are a set of exact duplicates
    (every query has r_1 == 0) and `k == 1` -- either passed via `--ann-k 1`
    or clamped there by `knn` for a two-row series -- where r_1 and r_k are
    the same column, so `survivor_mask`'s r_1 < r_k can never hold.

    Only the LID/contrast panels need this: hubness and IVF balance are
    computed over every row regardless of which queries survived.
    """
    affected = []
    for s in series:
        m = ann_metrics[s.name]
        if m.num_rows == 0 or m.discarded_queries != m.num_rows:
            continue
        # k < 2 is checked first: at k == 1 the mask cannot pass whatever the
        # data looks like, so it explains the whole series on its own.
        reason = (
            "measured at k=1, where the nearest and the k-th neighbour are "
            "the same point, so no query can pass the estimator's r_1 < r_k "
            "test"
            if m.k < 2
            else "every query sits on an exact duplicate"
        )
        affected.append(f"{s.name} ({reason})")
    if not affected:
        return ""
    return (
        f" <b>No surviving queries for {'; '.join(affected)}</b>. Both panels "
        "report n/a for those series rather than a number, and draw no trace "
        "for them."
    )


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

    sections: list[tuple[str, str, go.Figure]] = []

    # First, because it frames everything after it: every other panel is an
    # aggregate that can look healthy while individual descriptors are
    # structurally wrong.
    rows = glyphs.glyph_rows(series, cfg.glyph_samples, cfg.seed)
    if rows:
        sections.append(
            (
                glyphs.GLYPH_SECTION_TITLE,
                "Individual descriptors, not an aggregate. Each 128-value vector "
                "is drawn as a 4x4 grid of spatial cells, each cell an 8-ray star, "
                "one ray per orientation bin. Real SIFT is sparse and spiky, with "
                "most cells dominated by one or two directions; even, bushy stars "
                "mean a generator matched the marginals without the structure. "
                "<b>Red rays are negative bins, impossible for a gradient "
                "histogram</b>, and are drawn at a minimum length so they stay "
                "visible -- read their presence and count, not their size. The "
                "<code>real-a</code>/<code>real-b</code> pair is the baseline for "
                "how much natural variation to expect before judging a synthetic "
                "row. Ray length is otherwise a shared percentile scale across "
                "every descriptor here, so rows stay comparable. This panel "
                "assumes the vectors are raw descriptors: a set that was centered "
                "or whitened before being written out no longer maps dimension to "
                "(cell, orientation bin), and would be drawn as a plausible-looking "
                "lie.",
                glyphs.fig_descriptor_glyphs(rows),
            )
        )

    ann_note_suffix = (
        " Compare against the <code>real</code> series in this report only. "
        "These numbers come from a self-queried subsample, so they are not "
        "comparable with published SIFT1M figures."
    )
    sections.append(
        (
            "Local intrinsic dimensionality",
            "How locally high-dimensional the neighbourhood of a typical query "
            "is, and the strongest single predictor of how hard an index will "
            "find this data. A synthetic set landing well below real is easier "
            "to search and would understate any index's difficulty; well above "
            "and it overstates it. Relative contrast sits alongside: values "
            "near 1 mean the nearest neighbour is barely closer than an "
            "arbitrary point, leaving an index little to exploit."
            + ann_condition_note(
                series, ann_metrics, (("num_rows", "rows"), ("k", "k"))
            )
            + ann_discarded_note(series, ann_metrics)
            + ann_note_suffix,
            figures.fig_ann_profile(series, ann_metrics, args.bins),
        )
    )
    sections.append(
        (
            "Hubness",
            "How often each point turns up in other points' neighbour lists. A "
            "long right tail means a few hubs dominate, which is what stalls "
            "graph indexes like HNSW. A generator gets no direct training "
            "pressure to reproduce this, so matching it is genuine evidence "
            "rather than a fitted artefact."
            + ann_condition_note(series, ann_metrics, (("num_rows", "rows"),))
            + ann_note_suffix,
            figures.overlay_hist_fig(
                [
                    (
                        s.name,
                        ann_metrics[s.name].k_occurrence.astype(np.float64),
                        s.color,
                    )
                    for s in series
                ],
                args.bins,
                f"k-occurrence at k={args.ann_hub_k} (log density)",
                "times appearing in a neighbour list",
                log_y=True,
            ),
        )
    )
    sections.append(
        (
            "IVF cell balance",
            "How evenly k-means would partition each set, which drives how many "
            "cells an IVF query has to probe. Each set is clustered on its own, "
            "because an index would be built on whichever set you shipped."
            + ann_condition_note(series, ann_metrics, (("nlist", "nlist"),))
            + ann_note_suffix,
            figures.fig_ivf_balance(series, ann_metrics),
        )
    )

    sections.append(
        (
            "Pooled value distribution",
            "Every coordinate of every vector flattened into one histogram -- the "
            "equal-weight mixture of all per-dimension marginals. Raw SIFT is "
            "quantized with heavy mass at exactly zero, so a smooth unimodal blob "
            "here is wrong regardless of what the critic score says.",
            figures.fig_value_distribution(series, args.bins),
        )
    )
    sections.append(
        (
            "Per-dimension marginals",
            "The same histogram split by dimension; use the dropdown to page "
            "through. Aggregate overlap can hide compensating per-dimension error.",
            figures.fig_per_dim_marginals(series, args.bins),
        )
    )
    sections.append(
        (
            "Per-dimension profiles",
            "Mean, spread and exact-zero rate across dimensions. SIFT's zero rate "
            "varies strongly by dimension -- corner cells of the 4x4 grid are "
            "emptier than central ones -- so a flat profile means the generator "
            "learned an average rather than the descriptor layout.",
            figures.fig_dim_profiles(series),
        )
    )
    sections.append(
        (
            "Pairwise distances",
            "Distances between random pairs: the global geometry, and what "
            "downstream ANN benchmarking depends on.",
            figures.overlay_hist_fig(
                [
                    (
                        s.name,
                        metrics.pairwise_distance_sample(
                            s.x, args.num_pairs, args.seed
                        ),
                        s.color,
                    )
                    for s in series
                ],
                args.bins,
                "Pairwise Euclidean distance",
                "distance",
            ),
        )
    )
    sections.append(
        (
            f"Within-set {args.knn}-NN distances",
            "Each set measured against itself, not against real. Local packing "
            "rather than global spread, and the clearest mode-collapse tell: a "
            "collapsed generator crowds its samples, pushing this left of real. "
            "All sets are cut to equal N first, since k-NN distance shrinks as "
            "sample count grows.",
            figures.overlay_hist_fig(
                [
                    (
                        s.name,
                        metrics.nn_distances(
                            s.x, args.knn, args.seed, args.knn_max_rows
                        ),
                        s.color,
                    )
                    for s in series
                ],
                args.bins,
                f"Distance to {args.knn}-th nearest neighbour within set",
                "distance",
            ),
        )
    )

    if args.preprocess == "none":
        sections.append(
            (
                "Vector norms",
                "Only informative without L2 normalization.",
                figures.overlay_hist_fig(
                    [(s.name, np.linalg.norm(s.x, axis=1), s.color) for s in series],
                    args.bins,
                    "L2 norm",
                    "norm",
                ),
            )
        )

    sections.append(
        (
            "PCA spectrum",
            "A generator covering fewer effective directions than the data falls "
            "off more steeply and saturates earlier in the cumulative curve.",
            figures.fig_pca_spectrum(series),
        )
    )
    sections.append(
        (
            "Correlation structure",
            "SIFT is 4x4 spatial cells x 8 orientation bins, which produces "
            "visible block structure. Each synthetic is shown as a difference "
            "against real, which is what localizes the error.",
            figures.fig_correlation(series),
        )
    )

    worst_dims: dict[str, list[dict]] = {}
    if has_synth:
        divergence = metrics.dimension_divergence(series, args.top_divergent)
        worst_dims = divergence.worst
        div_fig = figures.fig_dim_divergence(divergence, series)
        sections.append(
            (
                "Per-dimension mismatch",
                "Dimensions ordered by the worst mismatch across all synthetic "
                "sets, so bars line up across series. Cross-reference the leaders "
                "against the marginals dropdown above.",
                div_fig,
            )
        )

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
        + stats_table_html(stats)
    )

    html = build_report(sections, meta_html, plotlyjs_head(args.plotlyjs, out_dir))
    report_path = out_dir / "eda_report.html"
    report_path.write_text(html, encoding="utf-8")

    png_paths: list[str] = []
    png_error = None
    if not args.no_png:
        try:
            png_paths = export_pngs(sections, out_dir)
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
