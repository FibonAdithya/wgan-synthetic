"""Exploratory data analysis for SIFT1M descriptors, with optional synthetic overlay.

Complements the metric-driven scripts in this package (evaluate_distribution,
plot_distance_cdf, plot_embedding_clusters) by showing raw distributional shape
instead of scalar summaries. The point is to be able to reject a generator by
eye even when the critic cannot separate the two sets -- a weak critic produces
good-looking Wasserstein estimates over samples whose marginals are obviously
wrong.

Emits one self-contained HTML report plus per-figure PNGs and a summary.json.

Example:
    python -m src.eval.eda_report \
        --real-path data/sift_base.npy \
        --synthetic-path runs/bench_improved/samples.npy \
        --output-dir runs/bench_improved/eda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.data.sift1m_dataset import load_descriptors

REAL_COLOR = "#2b6cb0"
SYNTH_COLOR = "#dd6b20"
REAL_NAME = "real"
SYNTH_NAME = "synthetic"


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
        default=None,
        help="Optional. When given, every panel overlays synthetic on real.",
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


def subsample(x: np.ndarray, max_vectors: int, seed: int) -> np.ndarray:
    if max_vectors <= 0 or x.shape[0] <= max_vectors:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_vectors, replace=False)
    return x[np.sort(idx)]


def maybe_l2_normalize(x: np.ndarray, mode: str, eps: float = 1.0e-8) -> np.ndarray:
    if mode == "none":
        return x
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norm, eps, None)).astype(np.float32, copy=False)


def pairwise_distance_sample(
    x: np.ndarray, num_pairs: int, seed: int
) -> np.ndarray:
    """Euclidean distances over randomly drawn distinct pairs."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    return np.linalg.norm(x[i] - x[j], axis=1)


def nn_distances(x: np.ndarray, k: int, seed: int, max_rows: int = 20000) -> np.ndarray:
    """Distance to the k-th nearest *other* point within the same set.

    Collapsed generators put mass on a few modes, which shows up as a
    within-set NN distance distribution shifted far below the real one.
    """
    sub = subsample(x, max_rows, seed)
    nn = NearestNeighbors(n_neighbors=min(k + 1, sub.shape[0]))
    nn.fit(sub)
    dist, _ = nn.kneighbors(sub)
    return dist[:, -1]


def wasserstein1(a: np.ndarray, b: np.ndarray, num_quantiles: int = 512) -> float:
    """1-D Wasserstein-1 via quantile functions; avoids a scipy dependency."""
    q = np.linspace(0.0, 1.0, num_quantiles)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


# --------------------------------------------------------------------------
# plotting helpers
# --------------------------------------------------------------------------


def shared_hist(
    real: np.ndarray,
    synth: Optional[np.ndarray],
    bins: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Density-normalized histograms of real/synthetic over shared bin edges."""
    if synth is None:
        lo, hi = float(real.min()), float(real.max())
    else:
        lo = float(min(real.min(), synth.min()))
        hi = float(max(real.max(), synth.max()))
    if hi <= lo:
        hi = lo + 1.0e-6
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    real_h, _ = np.histogram(real, bins=edges, density=True)
    synth_h = None if synth is None else np.histogram(synth, bins=edges, density=True)[0]
    return centers, real_h, synth_h


def overlay_hist_fig(
    real: np.ndarray,
    synth: Optional[np.ndarray],
    bins: int,
    title: str,
    xaxis_title: str,
    log_y: bool = False,
) -> go.Figure:
    centers, real_h, synth_h = shared_hist(real, synth, bins)
    fig = go.Figure()
    fig.add_bar(
        x=centers, y=real_h, name=REAL_NAME, marker_color=REAL_COLOR, opacity=0.65
    )
    if synth_h is not None:
        fig.add_bar(
            x=centers, y=synth_h, name=SYNTH_NAME, marker_color=SYNTH_COLOR, opacity=0.65
        )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="density",
        barmode="overlay",
        bargap=0.0,
        template="plotly_white",
        height=420,
    )
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def fig_value_distribution(
    real: np.ndarray, synth: Optional[np.ndarray], bins: int
) -> go.Figure:
    """All coordinates pooled. SIFT's quantized, zero-heavy shape lives here."""
    fig = overlay_hist_fig(
        real.ravel(),
        None if synth is None else synth.ravel(),
        bins,
        title="Pooled coordinate values (log density)",
        xaxis_title="coordinate value",
        log_y=True,
    )
    return fig


def fig_per_dim_marginals(
    real: np.ndarray, synth: Optional[np.ndarray], bins: int
) -> go.Figure:
    """One overlaid histogram per dimension, selectable from a dropdown."""
    dim = real.shape[1]
    fig = go.Figure()
    traces_per_dim = 1 if synth is None else 2

    for d in range(dim):
        centers, real_h, synth_h = shared_hist(
            real[:, d], None if synth is None else synth[:, d], bins
        )
        fig.add_bar(
            x=centers,
            y=real_h,
            name=REAL_NAME,
            marker_color=REAL_COLOR,
            opacity=0.65,
            visible=(d == 0),
        )
        if synth_h is not None:
            fig.add_bar(
                x=centers,
                y=synth_h,
                name=SYNTH_NAME,
                marker_color=SYNTH_COLOR,
                opacity=0.65,
                visible=(d == 0),
            )

    buttons = []
    for d in range(dim):
        visible = [False] * (dim * traces_per_dim)
        for t in range(traces_per_dim):
            visible[d * traces_per_dim + t] = True
        buttons.append(
            dict(
                label=f"dim {d}",
                method="update",
                args=[{"visible": visible}, {"title": f"Marginal, dimension {d}"}],
            )
        )

    fig.update_layout(
        title="Marginal, dimension 0",
        xaxis_title="coordinate value",
        yaxis_title="density",
        barmode="overlay",
        bargap=0.0,
        template="plotly_white",
        height=460,
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.0,
                xanchor="right",
                y=1.18,
                yanchor="top",
            )
        ],
    )
    return fig


def fig_dim_profiles(real: np.ndarray, synth: Optional[np.ndarray]) -> go.Figure:
    """Per-dimension mean, std and exact-zero fraction across all 128 dims."""
    dims = np.arange(real.shape[1])
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("per-dim mean", "per-dim std", "per-dim fraction of exact zeros"),
        vertical_spacing=0.07,
    )
    series = [(REAL_NAME, real, REAL_COLOR)]
    if synth is not None:
        series.append((SYNTH_NAME, synth, SYNTH_COLOR))

    for name, arr, color in series:
        stats = [
            arr.mean(axis=0),
            arr.std(axis=0),
            (arr == 0.0).mean(axis=0),
        ]
        for row, values in enumerate(stats, start=1):
            fig.add_scatter(
                x=dims,
                y=values,
                name=name,
                legendgroup=name,
                showlegend=(row == 1),
                line=dict(color=color),
                row=row,
                col=1,
            )

    fig.update_xaxes(title_text="dimension", row=3, col=1)
    fig.update_layout(
        title="Per-dimension profiles",
        template="plotly_white",
        height=760,
    )
    return fig


def fig_pca_spectrum(real: np.ndarray, synth: Optional[np.ndarray]) -> go.Figure:
    """Explained-variance spectrum. A collapsed generator is rank-deficient."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("explained variance ratio (log)", "cumulative explained variance"),
    )
    series = [(REAL_NAME, real, REAL_COLOR)]
    if synth is not None:
        series.append((SYNTH_NAME, synth, SYNTH_COLOR))

    for name, arr, color in series:
        pca = PCA(n_components=min(arr.shape[1], arr.shape[0]))
        pca.fit(arr)
        ratio = pca.explained_variance_ratio_
        comps = np.arange(1, ratio.size + 1)
        fig.add_scatter(
            x=comps,
            y=ratio,
            name=name,
            legendgroup=name,
            line=dict(color=color),
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=comps,
            y=np.cumsum(ratio),
            name=name,
            legendgroup=name,
            showlegend=False,
            line=dict(color=color),
            row=1,
            col=2,
        )

    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_xaxes(title_text="component", row=1, col=1)
    fig.update_xaxes(title_text="component", row=1, col=2)
    fig.update_layout(title="PCA spectrum", template="plotly_white", height=440)
    return fig


def fig_correlation(real: np.ndarray, synth: Optional[np.ndarray]) -> go.Figure:
    """Dimension-by-dimension correlation. SIFT has strong block structure from
    its 4x4 spatial cells x 8 orientation bins layout; a generator that misses
    it produces a visibly flatter matrix."""
    real_corr = np.corrcoef(real, rowvar=False)
    panels = [("real", real_corr, "RdBu")]
    if synth is not None:
        synth_corr = np.corrcoef(synth, rowvar=False)
        panels.append(("synthetic", synth_corr, "RdBu"))
        panels.append(("synthetic - real", synth_corr - real_corr, "RdBu"))

    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels])
    for col, (name, mat, scale) in enumerate(panels, start=1):
        fig.add_heatmap(
            z=mat,
            colorscale=scale,
            zmid=0.0,
            showscale=(col == len(panels)),
            row=1,
            col=col,
        )
    fig.update_layout(
        title="Per-dimension correlation structure",
        template="plotly_white",
        height=420,
    )
    return fig


def fig_dim_divergence(
    real: np.ndarray, synth: np.ndarray, top_k: int
) -> Tuple[go.Figure, List[Dict]]:
    """Rank dimensions by 1-D Wasserstein distance between marginals."""
    dists = np.array(
        [wasserstein1(real[:, d], synth[:, d]) for d in range(real.shape[1])]
    )
    order = np.argsort(dists)[::-1]
    fig = go.Figure()
    fig.add_bar(
        x=[f"dim {d}" for d in order],
        y=dists[order],
        marker_color=SYNTH_COLOR,
    )
    fig.update_layout(
        title="Per-dimension marginal mismatch (Wasserstein-1, worst first)",
        xaxis_title="dimension",
        yaxis_title="W1(real, synthetic)",
        template="plotly_white",
        height=420,
    )
    worst = [
        {"dim": int(d), "wasserstein1": float(dists[d])} for d in order[:top_k]
    ]
    return fig, worst


# --------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------


def summary_stats(name: str, x: np.ndarray) -> Dict:
    norms = np.linalg.norm(x, axis=1)
    return {
        "name": name,
        "num_vectors": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "value_mean": float(x.mean()),
        "value_std": float(x.std()),
        "value_min": float(x.min()),
        "value_max": float(x.max()),
        "exact_zero_fraction": float((x == 0.0).mean()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "duplicate_row_fraction": float(
            1.0 - np.unique(x, axis=0).shape[0] / x.shape[0]
        ),
    }


def stats_table_html(stats: List[Dict]) -> str:
    keys = [k for k in stats[0] if k != "name"]
    header = "".join(f"<th>{s['name']}</th>" for s in stats)
    rows = []
    for k in keys:
        cells = "".join(f"<td>{s[k]:.6g}</td>" for s in stats)
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
    sections: List[Tuple[str, str, go.Figure]],
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
        f"{meta_html}"
        + "".join(body)
        + "</body></html>"
    )


def export_pngs(sections: List[Tuple[str, str, go.Figure]], out_dir: Path) -> List[str]:
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = load_descriptors(Path(args.real_path), file_format=args.real_format)
    real = subsample(real, args.max_vectors, args.seed)
    real = maybe_l2_normalize(real, args.preprocess)

    synth = None
    if args.synthetic_path is not None:
        synth = load_descriptors(
            Path(args.synthetic_path), file_format=args.synthetic_format
        )
        if synth.shape[1] != real.shape[1]:
            raise ValueError(
                f"Dimension mismatch: real has {real.shape[1]}, "
                f"synthetic has {synth.shape[1]}"
            )
        synth = subsample(synth, args.max_vectors, args.seed)
        synth = maybe_l2_normalize(synth, args.preprocess)

    stats = [summary_stats(REAL_NAME, real)]
    if synth is not None:
        stats.append(summary_stats(SYNTH_NAME, synth))

    sections: List[Tuple[str, str, go.Figure]] = []

    sections.append(
        (
            "Pooled value distribution",
            "Raw SIFT coordinates are quantized integers with heavy mass at zero. "
            "A generator emitting a smooth unimodal blob here is wrong regardless "
            "of what the critic score says.",
            fig_value_distribution(real, synth, args.bins),
        )
    )
    sections.append(
        (
            "Per-dimension marginals",
            "Use the dropdown to page through all dimensions. Aggregate overlap "
            "can hide per-dimension mismatch.",
            fig_per_dim_marginals(real, synth, args.bins),
        )
    )
    sections.append(
        (
            "Per-dimension profiles",
            "Mean, spread and exact-zero rate across dimensions. SIFT's zero rate "
            "varies strongly by dimension; a generator with a flat profile has not "
            "learned the descriptor layout.",
            fig_dim_profiles(real, synth),
        )
    )

    real_pairs = pairwise_distance_sample(real, args.num_pairs, args.seed)
    synth_pairs = (
        None if synth is None else pairwise_distance_sample(synth, args.num_pairs, args.seed)
    )
    sections.append(
        (
            "Pairwise distances",
            "Distances between random pairs. This is what downstream ANN "
            "benchmarking actually depends on.",
            overlay_hist_fig(
                real_pairs,
                synth_pairs,
                args.bins,
                "Pairwise Euclidean distance",
                "distance",
            ),
        )
    )

    real_nn = nn_distances(real, args.knn, args.seed)
    synth_nn = None if synth is None else nn_distances(synth, args.knn, args.seed)
    sections.append(
        (
            f"Within-set {args.knn}-NN distances",
            "The clearest mode-collapse tell: a collapsed generator packs samples "
            "together, pushing this distribution far below the real one.",
            overlay_hist_fig(
                real_nn,
                synth_nn,
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
                overlay_hist_fig(
                    np.linalg.norm(real, axis=1),
                    None if synth is None else np.linalg.norm(synth, axis=1),
                    args.bins,
                    "L2 norm",
                    "norm",
                ),
            )
        )

    sections.append(
        (
            "PCA spectrum",
            "A generator covering fewer effective directions than the data shows a "
            "steeper falloff and a cumulative curve that saturates early.",
            fig_pca_spectrum(real, synth),
        )
    )
    sections.append(
        (
            "Correlation structure",
            "SIFT is 4x4 spatial cells x 8 orientation bins, which produces visible "
            "block structure. The difference panel highlights what the generator missed.",
            fig_correlation(real, synth),
        )
    )

    worst_dims: List[Dict] = []
    if synth is not None:
        div_fig, worst_dims = fig_dim_divergence(real, synth, args.top_divergent)
        sections.append(
            (
                "Per-dimension mismatch",
                "Worst dimensions first. Cross-reference the leaders against the "
                "marginals dropdown above.",
                div_fig,
            )
        )

    meta_html = (
        f'<div class="meta">real: <code>{args.real_path}</code>'
        + (
            f' &middot; synthetic: <code>{args.synthetic_path}</code>'
            if args.synthetic_path
            else " &middot; no synthetic overlay"
        )
        + f" &middot; preprocess: <code>{args.preprocess}</code>"
        + f" &middot; subsampled to {real.shape[0]} rows</div>"
        + stats_table_html(stats)
    )

    html = build_report(sections, meta_html, plotlyjs_head(args.plotlyjs, out_dir))
    report_path = out_dir / "eda_report.html"
    report_path.write_text(html, encoding="utf-8")

    png_paths: List[str] = []
    png_error = None
    if not args.no_png:
        try:
            png_paths = export_pngs(sections, out_dir)
        except Exception as exc:  # kaleido needs a Chrome binary
            png_error = str(exc)

    summary = {
        "real_path": args.real_path,
        "synthetic_path": args.synthetic_path,
        "preprocess": args.preprocess,
        "seed": args.seed,
        "stats": stats,
        "worst_dimensions": worst_dims,
        "report_html": str(report_path),
        "png_files": png_paths,
    }
    if png_error:
        summary["png_error"] = png_error
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {report_path}")
    if png_paths:
        print(f"Wrote {len(png_paths)} PNGs to {out_dir / 'png'}")
    elif png_error:
        print(f"PNG export skipped: {png_error}")
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
