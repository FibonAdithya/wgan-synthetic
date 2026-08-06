"""The report's panels, as data.

Each panel is a title, a prose note explaining what a reader should conclude
from it, and a builder that returns its figure. `build` returning None means
"this panel does not apply to this run" -- the report simply omits it. That
one mechanism covers all three cases: the glyph panel needs 128-dimensional
data, the norms panel needs unnormalized vectors, and the mismatch panel
needs something to compare against.

To add a panel: append a Panel here and add its builder to figures.py. To
change what a panel claims, edit its note here. Nothing else in the package
needs to know the report has thirteen sections rather than twelve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from src.eval import ann_difficulty
from src.eval.eda import figures, glyphs, metrics, notes
from src.eval.eda.config import EdaConfig
from src.eval.eda.series import Series


@dataclass(frozen=True)
class Context:
    """Everything a panel is allowed to see.

    Deliberately does not carry the summary statistics: no panel reads them,
    they feed the header table and summary.json, and keeping them out of here
    keeps the panel contract to what a panel actually draws from.
    """

    config: EdaConfig
    series: list[Series]
    ann_metrics: dict[str, ann_difficulty.AnnMetrics]
    divergence: metrics.DimDivergence | None

    @property
    def has_synthetic(self) -> bool:
        return len(self.series) > 1


Text = str | Callable[[Context], str]


@dataclass(frozen=True)
class Panel:
    """One report section.

    `title` and `note` accept a callable because some are computed: the k-NN
    title embeds --knn, and the ANN notes embed the conditions each series was
    actually measured under.
    """

    title: Text
    note: Text
    build: Callable[[Context], go.Figure | None]

    def resolve_title(self, ctx: Context) -> str:
        return self.title(ctx) if callable(self.title) else self.title

    def resolve_note(self, ctx: Context) -> str:
        return self.note(ctx) if callable(self.note) else self.note


# --------------------------------------------------------------------------
# computed prose
# --------------------------------------------------------------------------

LID_NOTE = (
    "How locally high-dimensional the neighbourhood of a typical query "
    "is, and the strongest single predictor of how hard an index will "
    "find this data. A synthetic set landing well below real is easier "
    "to search and would understate any index's difficulty; well above "
    "and it overstates it. Relative contrast sits alongside: values "
    "near 1 mean the nearest neighbour is barely closer than an "
    "arbitrary point, leaving an index little to exploit."
)

HUBNESS_NOTE = (
    "How often each point turns up in other points' neighbour lists. A "
    "long right tail means a few hubs dominate, which is what stalls "
    "graph indexes like HNSW. A generator gets no direct training "
    "pressure to reproduce this, so matching it is genuine evidence "
    "rather than a fitted artefact."
)

IVF_NOTE = (
    "How evenly k-means would partition each set, which drives how many "
    "cells an IVF query has to probe. Each set is clustered on its own, "
    "because an index would be built on whichever set you shipped."
)


def _knn_title(ctx: Context) -> str:
    return f"Within-set {ctx.config.knn}-NN distances"


def _lid_note(ctx: Context) -> str:
    return (
        LID_NOTE
        + notes.ann_condition_note(
            ctx.series, ctx.ann_metrics, (("num_rows", "rows"), ("k", "k"))
        )
        + notes.ann_discarded_note(ctx.series, ctx.ann_metrics)
        + notes.ANN_NOTE_SUFFIX
    )


def _hubness_note(ctx: Context) -> str:
    return (
        HUBNESS_NOTE
        + notes.ann_condition_note(ctx.series, ctx.ann_metrics, (("num_rows", "rows"),))
        + notes.ANN_NOTE_SUFFIX
    )


def _ivf_note(ctx: Context) -> str:
    return (
        IVF_NOTE
        + notes.ann_condition_note(ctx.series, ctx.ann_metrics, (("nlist", "nlist"),))
        + notes.ANN_NOTE_SUFFIX
    )


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def _build_glyphs(ctx: Context) -> go.Figure | None:
    rows = glyphs.glyph_rows(ctx.series, ctx.config.glyph_samples, ctx.config.seed)
    # glyph_rows returns [] when the mapping does not apply -- a width other
    # than 128, or a series too small for its rows. The applicability test and
    # the row choice are the same work, which is why this is one call and not
    # a separate predicate.
    return glyphs.fig_descriptor_glyphs(rows) if rows else None


def _build_ann_profile(ctx: Context) -> go.Figure:
    return figures.fig_ann_profile(ctx.series, ctx.ann_metrics, ctx.config.bins)


def _build_hubness(ctx: Context) -> go.Figure:
    return figures.overlay_hist_fig(
        [
            (
                s.name,
                ctx.ann_metrics[s.name].k_occurrence.astype(np.float64),
                s.color,
            )
            for s in ctx.series
        ],
        ctx.config.bins,
        f"k-occurrence at k={ctx.config.ann_hub_k} (log density)",
        "times appearing in a neighbour list",
        log_y=True,
    )


def _build_ivf_balance(ctx: Context) -> go.Figure:
    return figures.fig_ivf_balance(ctx.series, ctx.ann_metrics)


def _build_value_distribution(ctx: Context) -> go.Figure:
    return figures.fig_value_distribution(ctx.series, ctx.config.bins)


def _build_per_dim_marginals(ctx: Context) -> go.Figure:
    return figures.fig_per_dim_marginals(ctx.series, ctx.config.bins)


def _build_dim_profiles(ctx: Context) -> go.Figure:
    return figures.fig_dim_profiles(ctx.series)


def _build_pairwise(ctx: Context) -> go.Figure:
    return figures.overlay_hist_fig(
        [
            (
                s.name,
                metrics.pairwise_distance_sample(
                    s.x, ctx.config.num_pairs, ctx.config.seed
                ),
                s.color,
            )
            for s in ctx.series
        ],
        ctx.config.bins,
        "Pairwise Euclidean distance",
        "distance",
    )


def _build_knn(ctx: Context) -> go.Figure:
    return figures.overlay_hist_fig(
        [
            (
                s.name,
                metrics.nn_distances(
                    s.x, ctx.config.knn, ctx.config.seed, ctx.config.knn_max_rows
                ),
                s.color,
            )
            for s in ctx.series
        ],
        ctx.config.bins,
        f"Distance to {ctx.config.knn}-th nearest neighbour within set",
        "distance",
    )


def _build_norms(ctx: Context) -> go.Figure | None:
    if ctx.config.preprocess != "none":
        return None
    return figures.overlay_hist_fig(
        [(s.name, np.linalg.norm(s.x, axis=1), s.color) for s in ctx.series],
        ctx.config.bins,
        "L2 norm",
        "norm",
    )


def _build_pca(ctx: Context) -> go.Figure:
    return figures.fig_pca_spectrum(ctx.series)


def _build_correlation(ctx: Context) -> go.Figure:
    return figures.fig_correlation(ctx.series)


def _build_mismatch(ctx: Context) -> go.Figure | None:
    if ctx.divergence is None:
        return None
    return figures.fig_dim_divergence(ctx.divergence, ctx.series)


# --------------------------------------------------------------------------
# the registry -- list order is report order
# --------------------------------------------------------------------------

PANELS: list[Panel] = [
    # First, because it frames everything after it: every other panel is an
    # aggregate that can look healthy while individual descriptors are
    # structurally wrong.
    Panel(
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
        _build_glyphs,
    ),
    Panel("Local intrinsic dimensionality", _lid_note, _build_ann_profile),
    Panel("Hubness", _hubness_note, _build_hubness),
    Panel("IVF cell balance", _ivf_note, _build_ivf_balance),
    Panel(
        "Pooled value distribution",
        "Every coordinate of every vector flattened into one histogram -- the "
        "equal-weight mixture of all per-dimension marginals. Raw SIFT is "
        "quantized with heavy mass at exactly zero, so a smooth unimodal blob "
        "here is wrong regardless of what the critic score says.",
        _build_value_distribution,
    ),
    Panel(
        "Per-dimension marginals",
        "The same histogram split by dimension; use the dropdown to page "
        "through. Aggregate overlap can hide compensating per-dimension error.",
        _build_per_dim_marginals,
    ),
    Panel(
        "Per-dimension profiles",
        "Mean, spread and exact-zero rate across dimensions. SIFT's zero rate "
        "varies strongly by dimension -- corner cells of the 4x4 grid are "
        "emptier than central ones -- so a flat profile means the generator "
        "learned an average rather than the descriptor layout.",
        _build_dim_profiles,
    ),
    Panel(
        "Pairwise distances",
        "Distances between random pairs: the global geometry, and what "
        "downstream ANN benchmarking depends on.",
        _build_pairwise,
    ),
    Panel(
        _knn_title,
        "Each set measured against itself, not against real. Local packing "
        "rather than global spread, and the clearest mode-collapse tell: a "
        "collapsed generator crowds its samples, pushing this left of real. "
        "All sets are cut to equal N first, since k-NN distance shrinks as "
        "sample count grows.",
        _build_knn,
    ),
    Panel(
        "Vector norms",
        "Only informative without L2 normalization.",
        _build_norms,
    ),
    Panel(
        "PCA spectrum",
        "A generator covering fewer effective directions than the data falls "
        "off more steeply and saturates earlier in the cumulative curve.",
        _build_pca,
    ),
    Panel(
        "Correlation structure",
        "SIFT is 4x4 spatial cells x 8 orientation bins, which produces "
        "visible block structure. Each synthetic is shown as a difference "
        "against real, which is what localizes the error.",
        _build_correlation,
    ),
    Panel(
        "Per-dimension mismatch",
        "Dimensions ordered by the worst mismatch across all synthetic "
        "sets, so bars line up across series. Cross-reference the leaders "
        "against the marginals dropdown above.",
        _build_mismatch,
    ),
]
