"""Plotly figures for the report's aggregate panels.

Each builder takes already-computed inputs and returns a `go.Figure`. None of
them read settings, write files, or decide whether their panel belongs in the
report -- that is `panels.py`'s job.

The descriptor glyph panel lives in `glyphs.py` instead, because
`plot_descriptor_grid` draws it from generator checkpoints and should not
have to import the aggregate figures to do so.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from src.eval import ann_difficulty
from src.eval.eda import metrics
from src.eval.eda.series import REAL_NAME, Series


def shared_edges(arrays: Sequence[np.ndarray], bins: int) -> np.ndarray:
    """Common bin edges spanning every array, so histograms are comparable."""
    lo = float(min(float(a.min()) for a in arrays))
    hi = float(max(float(a.max()) for a in arrays))
    if hi <= lo:
        hi = lo + 1.0e-6
    return np.linspace(lo, hi, bins + 1)


# Relative contrast is a ratio with the nearest-neighbour distance in its
# denominator, so it has no bounded upper tail: on a corpus with near-duplicate
# pairs a few queries reach 1e7 and up. The contrast subplot bins to this
# quantile and states what it left out. Not applied to LID, whose Hill estimate
# is already bounded in practice.
CONTRAST_UPPER_QUANTILE = 99.5


def clipped_edges(
    arrays: Sequence[np.ndarray], bins: int, upper_quantile: float
) -> tuple[np.ndarray, int]:
    """Bin edges bounded above at a quantile, plus how many values fall past it.

    Relative contrast divides by the nearest-neighbour distance, so a corpus
    containing near-coincident pairs -- SIFT's quantized lattice produces them --
    sends a handful of queries to enormous values. On min-to-max edges those few
    points set the axis and every other bin collapses against the left edge, so
    the panel renders as one spike and reads as a rendering failure.

    Bounding the upper edge at a quantile makes the bulk visible; returning the
    count past it lets the caller say what is not shown, so the clip is a stated
    choice rather than a silent truncation.
    """
    lo = float(min(float(a.min()) for a in arrays))
    hi = float(max(float(np.quantile(a, upper_quantile / 100.0)) for a in arrays))
    if hi <= lo:
        hi = lo + 1.0e-6
    dropped = int(sum(int((a > hi).sum()) for a in arrays))
    return np.linspace(lo, hi, bins + 1), dropped


def _rgba(color: str, alpha: float) -> str | None:
    """'#rgb' or '#rrggbb' -> 'rgba(r, g, b, alpha)', for a fill under a line.

    Returns None for anything else -- a CSS colour name, an existing rgb()
    string -- so the caller drops the fill rather than raising. A panel is worth
    more without its fill than not at all.
    """
    h = color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return None
    try:
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None
    return f"rgba({r}, {g}, {b}, {alpha})"


def density_trace(
    centers: np.ndarray,
    hist: np.ndarray,
    name: str,
    color: str,
    *,
    is_real: bool = False,
    log_y: bool = False,
    fill: bool = True,
    **kwargs: object,
) -> go.Scatter:
    """One binned-density curve.

    These panels overlay every set on one axis, and overlaid translucent bars
    stop being readable at three or four series: the fills multiply together
    into mud and no single set can be followed across the range. The density is
    binned exactly as before -- same edges, same `density=True` normalization --
    and only the mark changes, so the curve reads at any series count.

    `real` is drawn heavier, and is the only curve given a fill: it is the
    reference every other curve is compared against, not one series among
    equals. Filling all of them reinstates the mud the bars produced -- four
    translucent fills over each other are unreadable whatever the mark.

    On a log y-axis the fill is dropped -- there is no zero to fill down to --
    and empty bins become breaks in the line rather than plunges to the axis
    floor, which would otherwise read as data.
    """
    y = np.asarray(hist, dtype=np.float64)
    if log_y:
        y = np.where(y > 0.0, y, np.nan)
    trace = dict(
        x=centers,
        y=y,
        name=name,
        mode="lines",
        line=dict(color=color, width=2.6 if is_real else 1.8),
        **kwargs,
    )
    if fill and is_real and not log_y:
        fillcolor = _rgba(color, 0.16)
        if fillcolor is not None:
            trace["fill"] = "tozeroy"
            trace["fillcolor"] = fillcolor
    return go.Scatter(**trace)


def overlay_hist_fig(
    named_values: Sequence[tuple[str, np.ndarray, str]],
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
        fig.add_trace(
            density_trace(
                centers,
                hist,
                name,
                color,
                is_real=(name == REAL_NAME),
                log_y=log_y,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="density",
        template="plotly_white",
        height=440,
        hovermode="x unified",
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
            fig.add_trace(
                density_trace(
                    centers,
                    hist,
                    s.name,
                    s.color,
                    is_real=s.is_real,
                    visible=(d == 0),
                )
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
        template="plotly_white",
        height=460,
        hovermode="x unified",
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
        subplot_titles=(
            "per-dim mean",
            "per-dim std",
            "per-dim fraction of exact zeros",
        ),
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
    fig.update_layout(
        title="Per-dimension profiles", template="plotly_white", height=760
    )
    return fig


def fig_pca_spectrum(series: Sequence[Series]) -> go.Figure:
    """Explained-variance spectrum. A collapsed generator is rank-deficient."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "explained variance ratio (log)",
            "cumulative explained variance",
        ),
    )
    for s in series:
        pca = PCA(n_components=min(s.x.shape[1], s.x.shape[0]))
        pca.fit(s.x)
        ratio = pca.explained_variance_ratio_
        comps = np.arange(1, ratio.size + 1)
        fig.add_scatter(
            x=comps,
            y=ratio,
            name=s.name,
            legendgroup=s.name,
            line=dict(color=s.color),
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=comps,
            y=np.cumsum(ratio),
            name=s.name,
            legendgroup=s.name,
            showlegend=False,
            line=dict(color=s.color),
            row=1,
            col=2,
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
    panels: list[tuple[str, np.ndarray]] = [("real", real_corr)]
    for s in series:
        if s.is_real:
            continue
        panels.append((f"{s.name} - real", np.corrcoef(s.x, rowvar=False) - real_corr))

    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels])
    for col, (_, mat) in enumerate(panels, start=1):
        fig.add_heatmap(
            z=mat,
            colorscale="RdBu",
            zmid=0.0,
            showscale=(col == len(panels)),
            row=1,
            col=col,
        )
    fig.update_layout(
        title="Per-dimension correlation structure",
        template="plotly_white",
        height=400,
    )
    return fig


def fig_ann_profile(
    series: Sequence[Series], metrics: dict[str, ann_difficulty.AnnMetrics], bins: int
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
            # Every series was fully degenerate, so there is nothing to bin.
            # Say so in the subplot rather than leaving an empty pair of axes
            # that reads as a rendering failure.
            fig.add_annotation(
                text="no surviving queries",
                showarrow=False,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                font=dict(color="#718096"),
                row=1,
                col=col,
            )
            continue
        if attr == "relative_contrast":
            edges, dropped = clipped_edges(populated, bins, CONTRAST_UPPER_QUANTILE)
            if dropped:
                total = sum(int(v.size) for v in populated)
                fig.add_annotation(
                    text=(
                        f"{dropped} of {total} queries above {edges[-1]:.2f} not shown"
                    ),
                    showarrow=False,
                    xref="x domain",
                    yref="y domain",
                    x=0.98,
                    xanchor="right",
                    # Inside the axes, not above them: y=1.0 with yanchor
                    # "bottom" lands on the subplot title.
                    y=0.98,
                    yanchor="top",
                    font=dict(color="#718096", size=11),
                    row=1,
                    col=col,
                )
        else:
            edges = shared_edges(populated, bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for s in series:
            v = getattr(metrics[s.name], attr)
            if not v.size:
                continue
            hist, _ = np.histogram(v, bins=edges, density=True)
            fig.add_trace(
                density_trace(
                    centers,
                    hist,
                    s.name,
                    s.color,
                    is_real=s.is_real,
                    legendgroup=s.name,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
    fig.update_layout(
        title="ANN difficulty profile",
        template="plotly_white",
        height=440,
        hovermode="x unified",
    )
    return fig


def fig_ivf_balance(
    series: Sequence[Series], metrics: dict[str, ann_difficulty.AnnMetrics]
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
    divergence: metrics.DimDivergence, series: Sequence[Series]
) -> go.Figure:
    """Draw the per-dimension mismatch bars in the shared worst-first order."""
    fig = go.Figure()
    for s in series:
        if s.is_real:
            continue
        fig.add_bar(
            x=[f"dim {d}" for d in divergence.order],
            y=divergence.distances[s.name][divergence.order],
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
    return fig
