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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.data.sift1m_dataset import load_descriptors
from src.eval import ann_difficulty

REAL_NAME = "real"
REAL_COLOR = "#2b6cb0"
# Colors for synthetic sets, in order. Deliberately distinct from REAL_COLOR so
# the reference curve stays identifiable when several overlays are present.
SYNTH_PALETTE = [
    "#dd6b20",
    "#38a169",
    "#805ad5",
    "#d53f8c",
    "#00897b",
    "#a0522d",
]

# Single source of truth for the ANN-difficulty flag defaults, shared with
# compare_variants.py so its hand-built Namespace cannot silently drift from
# what this module's own --ann-* / --ivf-nlist flags default to.
ANN_K_DEFAULT = 100
ANN_HUB_K_DEFAULT = 10
ANN_MAX_ROWS_DEFAULT = 20000
IVF_NLIST_DEFAULT = 256


@dataclass
class Series:
    """One dataset to plot, already subsampled and preprocessed."""

    name: str
    x: np.ndarray
    color: str

    @property
    def is_real(self) -> bool:
        return self.name == REAL_NAME


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
            "Equal-N truncation for every difficulty metric, and for the "
            "within-set k-NN panel. LID, contrast and hubness all drift with "
            "sample count, so every set must be cut to the same size."
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


def parse_synthetic_spec(spec: str) -> Tuple[str, Path]:
    """Split a '[LABEL=]PATH' argument into its label and path.

    Only the first '=' separates, so paths containing '=' still work when a
    label is supplied. A bare path falls back to the file stem as its label.
    """
    if "=" in spec:
        label, _, raw = spec.partition("=")
        label = label.strip()
        if label:
            return label, Path(raw)
    path = Path(spec)
    return path.stem, path


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


def pairwise_distance_sample(x: np.ndarray, num_pairs: int, seed: int) -> np.ndarray:
    """Euclidean distances over randomly drawn distinct pairs."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    return np.linalg.norm(x[i] - x[j], axis=1)


def nn_distances(x: np.ndarray, k: int, seed: int, max_rows: int) -> np.ndarray:
    """Distance to the k-th nearest *other* point within the same set.

    Collapsed generators put mass on a few modes, which shows up as a
    within-set NN distance distribution shifted far below the real one. All
    sets are cut to the same max_rows first: k-NN distance shrinks as sample
    count grows, so unequal N would make the comparison meaningless.
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


def shared_edges(arrays: Sequence[np.ndarray], bins: int) -> np.ndarray:
    """Common bin edges spanning every array, so histograms are comparable."""
    lo = float(min(float(a.min()) for a in arrays))
    hi = float(max(float(a.max()) for a in arrays))
    if hi <= lo:
        hi = lo + 1.0e-6
    return np.linspace(lo, hi, bins + 1)


def overlay_hist_fig(
    named_values: Sequence[Tuple[str, np.ndarray, str]],
    bins: int,
    title: str,
    xaxis_title: str,
    log_y: bool = False,
) -> go.Figure:
    edges = shared_edges([v for _, v, _ in named_values], bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig = go.Figure()
    for name, values, color in named_values:
        hist, _ = np.histogram(values, bins=edges, density=True)
        fig.add_bar(x=centers, y=hist, name=name, marker_color=color, opacity=0.55)
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="density",
        barmode="overlay",
        bargap=0.0,
        template="plotly_white",
        height=440,
    )
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def fig_value_distribution(series: Sequence[Series], bins: int) -> go.Figure:
    """All coordinates pooled. SIFT's quantized, zero-heavy shape lives here."""
    return overlay_hist_fig(
        [(s.name, s.x.ravel(), s.color) for s in series],
        bins,
        title="Pooled coordinate values (log density)",
        xaxis_title="coordinate value",
        log_y=True,
    )


def fig_per_dim_marginals(series: Sequence[Series], bins: int) -> go.Figure:
    """One overlaid histogram per dimension, selectable from a dropdown."""
    dim = series[0].x.shape[1]
    per_dim = len(series)
    fig = go.Figure()

    for d in range(dim):
        edges = shared_edges([s.x[:, d] for s in series], bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for s in series:
            hist, _ = np.histogram(s.x[:, d], bins=edges, density=True)
            fig.add_bar(
                x=centers,
                y=hist,
                name=s.name,
                marker_color=s.color,
                opacity=0.55,
                visible=(d == 0),
            )

    buttons = []
    for d in range(dim):
        visible = [False] * (dim * per_dim)
        for t in range(per_dim):
            visible[d * per_dim + t] = True
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


def fig_dim_profiles(series: Sequence[Series]) -> go.Figure:
    """Per-dimension mean, std and exact-zero fraction across all dims."""
    dims = np.arange(series[0].x.shape[1])
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("per-dim mean", "per-dim std", "per-dim fraction of exact zeros"),
        vertical_spacing=0.07,
    )
    for s in series:
        stats = [s.x.mean(axis=0), s.x.std(axis=0), (s.x == 0.0).mean(axis=0)]
        for row, values in enumerate(stats, start=1):
            fig.add_scatter(
                x=dims,
                y=values,
                name=s.name,
                legendgroup=s.name,
                showlegend=(row == 1),
                line=dict(color=s.color),
                row=row,
                col=1,
            )
    fig.update_xaxes(title_text="dimension", row=3, col=1)
    fig.update_layout(title="Per-dimension profiles", template="plotly_white", height=760)
    return fig


def fig_pca_spectrum(series: Sequence[Series]) -> go.Figure:
    """Explained-variance spectrum. A collapsed generator is rank-deficient."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("explained variance ratio (log)", "cumulative explained variance"),
    )
    for s in series:
        pca = PCA(n_components=min(s.x.shape[1], s.x.shape[0]))
        pca.fit(s.x)
        ratio = pca.explained_variance_ratio_
        comps = np.arange(1, ratio.size + 1)
        fig.add_scatter(
            x=comps, y=ratio, name=s.name, legendgroup=s.name,
            line=dict(color=s.color), row=1, col=1,
        )
        fig.add_scatter(
            x=comps, y=np.cumsum(ratio), name=s.name, legendgroup=s.name,
            showlegend=False, line=dict(color=s.color), row=1, col=2,
        )
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_xaxes(title_text="component", row=1, col=1)
    fig.update_xaxes(title_text="component", row=1, col=2)
    fig.update_layout(title="PCA spectrum", template="plotly_white", height=440)
    return fig


def fig_correlation(series: Sequence[Series]) -> go.Figure:
    """Dimension-by-dimension correlation. SIFT has block structure from its
    4x4 spatial cells x 8 orientation bins layout; a generator that misses it
    produces a visibly flatter matrix. With overlays, each synthetic gets a
    difference panel against real rather than its own raw matrix, since the
    difference is what localizes the error."""
    real = next(s for s in series if s.is_real)
    real_corr = np.corrcoef(real.x, rowvar=False)
    panels: List[Tuple[str, np.ndarray]] = [("real", real_corr)]
    for s in series:
        if s.is_real:
            continue
        panels.append((f"{s.name} - real", np.corrcoef(s.x, rowvar=False) - real_corr))

    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels])
    for col, (_, mat) in enumerate(panels, start=1):
        fig.add_heatmap(
            z=mat, colorscale="RdBu", zmid=0.0,
            showscale=(col == len(panels)), row=1, col=col,
        )
    fig.update_layout(
        title="Per-dimension correlation structure",
        template="plotly_white",
        height=400,
    )
    return fig


def fig_ann_profile(
    series: Sequence[Series], metrics: Dict[str, "ann_difficulty.AnnMetrics"], bins: int
) -> go.Figure:
    """LID and relative contrast side by side, overlaid across sets.

    Both read off the same surviving queries, so a set that shifts left on
    LID and right on contrast is unambiguously easier to search than real --
    not an artefact of different query subsets.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "local intrinsic dimensionality",
            "relative contrast",
        ),
    )
    for col, attr in ((1, "lid"), (2, "relative_contrast")):
        values = [getattr(metrics[s.name], attr) for s in series]
        populated = [v for v in values if v.size]
        if not populated:
            continue
        edges = shared_edges(populated, bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for s in series:
            v = getattr(metrics[s.name], attr)
            if not v.size:
                continue
            hist, _ = np.histogram(v, bins=edges, density=True)
            fig.add_bar(
                x=centers,
                y=hist,
                name=s.name,
                legendgroup=s.name,
                showlegend=(col == 1),
                marker_color=s.color,
                opacity=0.55,
                row=1,
                col=col,
            )
    fig.update_layout(
        title="ANN difficulty profile",
        barmode="overlay",
        bargap=0.0,
        template="plotly_white",
        height=440,
    )
    return fig


def fig_ivf_balance(
    series: Sequence[Series], metrics: Dict[str, "ann_difficulty.AnnMetrics"]
) -> go.Figure:
    """Lorenz curve of cluster occupancy: how lopsided an IVF partition is.

    The diagonal is a perfectly even split. Bowing below it means a few
    cells hold most of the points, so a query has to probe more of them to
    reach the same recall.
    """
    fig = go.Figure()
    fig.add_scatter(
        x=[0.0, 1.0],
        y=[0.0, 1.0],
        name="perfect balance",
        line=dict(color="#a0aec0", dash="dash"),
    )
    for s in series:
        occupancy = metrics[s.name].cell_occupancy
        fig.add_scatter(
            x=np.arange(1, occupancy.size + 1) / occupancy.size,
            y=np.cumsum(occupancy) / occupancy.sum(),
            name=s.name,
            line=dict(color=s.color),
        )
    fig.update_layout(
        title="IVF cell balance",
        xaxis_title="fraction of cells (emptiest first)",
        yaxis_title="cumulative fraction of points",
        template="plotly_white",
        height=440,
    )
    return fig


def fig_dim_divergence(
    series: Sequence[Series], top_k: int
) -> Tuple[go.Figure, Dict[str, List[Dict]]]:
    """Rank dimensions by 1-D Wasserstein distance from real, per synthetic set.

    Dimensions are ordered by the worst mismatch across all synthetics, so the
    same x-axis ordering applies to every series and they stay comparable.
    """
    real = next(s for s in series if s.is_real)
    synths = [s for s in series if not s.is_real]
    dim = real.x.shape[1]

    dists = {
        s.name: np.array(
            [wasserstein1(real.x[:, d], s.x[:, d]) for d in range(dim)]
        )
        for s in synths
    }
    worst_overall = np.max(np.stack(list(dists.values())), axis=0)
    order = np.argsort(worst_overall)[::-1]

    fig = go.Figure()
    for s in synths:
        fig.add_bar(
            x=[f"dim {d}" for d in order],
            y=dists[s.name][order],
            name=s.name,
            marker_color=s.color,
        )
    fig.update_layout(
        title="Per-dimension marginal mismatch vs real (Wasserstein-1, worst first)",
        xaxis_title="dimension",
        yaxis_title="W1(real, synthetic)",
        barmode="group",
        template="plotly_white",
        height=440,
    )
    worst = {
        name: [{"dim": int(d), "wasserstein1": float(v[d])} for d in order[:top_k]]
        for name, v in dists.items()
    }
    return fig, worst


# --------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------


def effective_rank(x: np.ndarray) -> float:
    """exp(Shannon entropy of the explained-variance spectrum).

    Reads as "how many directions meaningfully carry variance": equals the
    dimension count when variance is spread evenly and 1 when it all sits on a
    single direction. Note this uses variance ratios, not the normalized
    singular values of Roy & Vetterli, so absolute values are not comparable
    with that definition -- only across sets measured here.
    """
    ratio = PCA(n_components=min(x.shape[1], x.shape[0])).fit(x).explained_variance_ratio_
    return float(np.exp(-np.sum(ratio * np.log(ratio + 1.0e-12))))


def summary_stats(
    s: Series,
    knn: int,
    num_pairs: int,
    seed: int,
    max_rows: int,
    metrics: ann_difficulty.AnnMetrics,
) -> Dict:
    norms = np.linalg.norm(s.x, axis=1)
    stats = {
        "name": s.name,
        "num_vectors": int(s.x.shape[0]),
        "dim": int(s.x.shape[1]),
        "value_mean": float(s.x.mean()),
        "value_std": float(s.x.std()),
        "value_min": float(s.x.min()),
        "value_max": float(s.x.max()),
        "exact_zero_fraction": float((s.x == 0.0).mean()),
        "negative_fraction": float((s.x < 0.0).mean()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "duplicate_row_fraction": float(
            1.0 - np.unique(s.x, axis=0).shape[0] / s.x.shape[0]
        ),
        "median_pairwise_distance": float(
            np.median(pairwise_distance_sample(s.x, num_pairs, seed))
        ),
        f"median_{knn}nn_distance": float(
            np.median(nn_distances(s.x, knn, seed, max_rows))
        ),
        "effective_rank": effective_rank(s.x),
    }
    stats.update(ann_difficulty.summary(metrics))
    # Actual (post-clamp) measurement conditions, not the requested ones: a
    # series with fewer rows than --ann-max-rows gets its k and nlist clamped
    # inside knn()/cell_occupancy(), and its num_vectors above is the
    # PRE-truncation count. Without these, nothing records what a series was
    # actually measured under, and the report's section notes cannot tell a
    # reader when conditions diverge across series.
    stats["ann_measured_rows"] = metrics.num_rows
    stats["ann_measured_k"] = metrics.k
    stats["ann_measured_nlist"] = metrics.nlist
    return stats


def stats_table_html(stats: List[Dict]) -> str:
    keys = [k for k in stats[0] if k != "name"]
    header = "".join(f"<th>{s['name']}</th>" for s in stats)
    rows = []
    for k in keys:
        cells = "".join(
            f"<td>{'n/a' if s[k] is None else format(s[k], '.6g')}</td>" for s in stats
        )
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


def load_series(args: argparse.Namespace) -> List[Series]:
    real_x = load_descriptors(Path(args.real_path), file_format=args.real_format)
    real_x = maybe_l2_normalize(subsample(real_x, args.max_vectors, args.seed), args.preprocess)
    series = [Series(REAL_NAME, real_x, REAL_COLOR)]

    seen = {REAL_NAME}
    for i, spec in enumerate(args.synthetic_path or []):
        label, path = parse_synthetic_spec(spec)
        if label in seen:
            raise ValueError(f"Duplicate series label {label!r}; use LABEL=PATH to rename")
        seen.add(label)
        x = load_descriptors(path, file_format=args.synthetic_format)
        if x.shape[1] != real_x.shape[1]:
            raise ValueError(
                f"Dimension mismatch for {label!r}: real has {real_x.shape[1]}, "
                f"got {x.shape[1]}"
            )
        x = maybe_l2_normalize(subsample(x, args.max_vectors, args.seed), args.preprocess)
        series.append(Series(label, x, SYNTH_PALETTE[i % len(SYNTH_PALETTE)]))
    return series


def ann_condition_note(
    series: Sequence[Series],
    ann_metrics: Dict[str, "ann_difficulty.AnnMetrics"],
    attrs: Tuple[Tuple[str, str], ...],
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


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = load_series(args)
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
        summary_stats(
            s, args.knn, args.num_pairs, args.seed, args.ann_max_rows, ann_metrics[s.name]
        )
        for s in series
    ]

    sections: List[Tuple[str, str, go.Figure]] = []

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
            + ann_condition_note(series, ann_metrics, (("num_rows", "rows"), ("k", "k")))
            + ann_note_suffix,
            fig_ann_profile(series, ann_metrics, args.bins),
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
            overlay_hist_fig(
                [
                    (s.name, ann_metrics[s.name].k_occurrence.astype(np.float64), s.color)
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
            fig_ivf_balance(series, ann_metrics),
        )
    )

    sections.append(
        (
            "Pooled value distribution",
            "Every coordinate of every vector flattened into one histogram -- the "
            "equal-weight mixture of all per-dimension marginals. Raw SIFT is "
            "quantized with heavy mass at exactly zero, so a smooth unimodal blob "
            "here is wrong regardless of what the critic score says.",
            fig_value_distribution(series, args.bins),
        )
    )
    sections.append(
        (
            "Per-dimension marginals",
            "The same histogram split by dimension; use the dropdown to page "
            "through. Aggregate overlap can hide compensating per-dimension error.",
            fig_per_dim_marginals(series, args.bins),
        )
    )
    sections.append(
        (
            "Per-dimension profiles",
            "Mean, spread and exact-zero rate across dimensions. SIFT's zero rate "
            "varies strongly by dimension -- corner cells of the 4x4 grid are "
            "emptier than central ones -- so a flat profile means the generator "
            "learned an average rather than the descriptor layout.",
            fig_dim_profiles(series),
        )
    )
    sections.append(
        (
            "Pairwise distances",
            "Distances between random pairs: the global geometry, and what "
            "downstream ANN benchmarking depends on.",
            overlay_hist_fig(
                [
                    (s.name, pairwise_distance_sample(s.x, args.num_pairs, args.seed), s.color)
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
            overlay_hist_fig(
                [
                    (s.name, nn_distances(s.x, args.knn, args.seed, args.ann_max_rows), s.color)
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
                overlay_hist_fig(
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
            fig_pca_spectrum(series),
        )
    )
    sections.append(
        (
            "Correlation structure",
            "SIFT is 4x4 spatial cells x 8 orientation bins, which produces "
            "visible block structure. Each synthetic is shown as a difference "
            "against real, which is what localizes the error.",
            fig_correlation(series),
        )
    )

    worst_dims: Dict[str, List[Dict]] = {}
    if has_synth:
        div_fig, worst_dims = fig_dim_divergence(series, args.top_divergent)
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

    png_paths: List[str] = []
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

    return report_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
