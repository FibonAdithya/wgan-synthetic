"""The descriptor glyph panel.

Every other panel in the report is an aggregate over tens of thousands of
vectors; this one draws a handful of individual descriptors, because a
matched marginal says nothing about whether the 128 numbers form a plausible
gradient histogram.

Kept apart from `figures.py` because `plot_descriptor_grid` renders the same
panel straight from generator checkpoints, and this module is the whole of
what it needs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from src.eval.descriptor_glyph import (
    DESCRIPTOR_DIM,
    NEGATIVE_RAY_FLOOR,
    descriptor_to_cells,
    glyph_segments,
    shared_scale,
)
from src.eval.eda.series import Series

# Descriptor glyph panel. Every other section here is an aggregate over tens of
# thousands of vectors; this one draws a handful of individual descriptors,
# because a matched marginal says nothing about whether the 128 numbers form a
# plausible gradient histogram.
GLYPH_SECTION_TITLE = "Descriptor glyphs"
GLYPH_CELL_PITCH = 1.0
# Roughly one glyph width (4 * GLYPH_CELL_PITCH) plus a gutter, so rows read as
# discrete descriptors rather than one continuous texture.
GLYPH_PITCH = 5.0
# Two rows are drawn for the real series: without a sense of how much two real
# descriptors differ from each other, a variant row below them is just a vibe.
GLYPH_REAL_COLORS = ("#2b6cb0", "#17becf")
GLYPH_NEGATIVE_COLOR = "#d62728"


def fig_descriptor_glyphs(rows: Sequence[tuple[str, np.ndarray, str]]) -> go.Figure:
    """Assemble the glyph grid from `(label, vectors, color)` rows.

    One positive-ray trace per row so each gets its own colour and legend
    entry, plus a single shared trace for negative rays across all rows --
    those mark impossible values and should read as one alarming category,
    not as a per-row detail.

    Shared by `plot_descriptor_grid`, which renders the same figure straight
    from generator checkpoints rather than from materialised arrays.
    """
    scale = shared_scale(np.concatenate([vecs for _, vecs, _ in rows], axis=0))

    fig = go.Figure()
    neg_x: list[np.ndarray] = []
    neg_y: list[np.ndarray] = []

    for row_index, (label, vecs, color) in enumerate(rows):
        pos_x: list[np.ndarray] = []
        pos_y: list[np.ndarray] = []
        for col_index in range(vecs.shape[0]):
            cells = descriptor_to_cells(vecs[col_index])
            origin = (col_index * GLYPH_PITCH, -row_index * GLYPH_PITCH)
            gx, gy, nx, ny = glyph_segments(cells, origin, GLYPH_CELL_PITCH, scale)
            pos_x.append(gx)
            pos_y.append(gy)
            neg_x.append(nx)
            neg_y.append(ny)
        fig.add_scatter(
            x=np.concatenate(pos_x) if pos_x else np.array([]),
            y=np.concatenate(pos_y) if pos_y else np.array([]),
            mode="lines",
            name=label,
            line=dict(color=color, width=1.4),
            hoverinfo="skip",
        )
        fig.add_annotation(
            x=-GLYPH_PITCH * 0.7,
            y=-row_index * GLYPH_PITCH,
            text=label,
            showarrow=False,
            xanchor="right",
            font=dict(size=13),
        )

    stacked_neg_x = np.concatenate(neg_x) if neg_x else np.array([])
    title = "SIFT descriptor glyphs: real vs generated"
    if stacked_neg_x.size:
        fig.add_scatter(
            x=stacked_neg_x,
            y=np.concatenate(neg_y),
            mode="lines",
            name="negative",
            line=dict(color=GLYPH_NEGATIVE_COLOR, width=1.8),
            hoverinfo="skip",
        )
        # Length encodes magnitude for every ray except these, so the figure
        # has to say so rather than let a reader size them by eye.
        title += (
            "<br><sub>Red rays are negative bins, impossible for a gradient "
            f"histogram, drawn at a minimum length of "
            f"{NEGATIVE_RAY_FLOOR:.0%} of the half-cell -- read their presence "
            "and count, not their size.</sub>"
        )

    # Rule separating the two real rows from the variant rows below them.
    if len(rows) > 2:
        fig.add_hline(
            y=-1.5 * GLYPH_PITCH, line=dict(color="#999999", width=1, dash="dot")
        )

    axis = dict(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(
        title=title,
        xaxis=axis,
        yaxis=dict(**axis, scaleanchor="x", scaleratio=1),
        height=180 * len(rows) + 120,
        plot_bgcolor="white",
        margin=dict(l=90, r=20, t=60, b=20),
    )
    return fig


def glyph_rows(
    series: Sequence[Series], num_samples: int, seed: int
) -> list[tuple[str, np.ndarray, str]]:
    """Pick the descriptors the glyph panel draws, or nothing if it cannot.

    The real series contributes two disjoint rows and each synthetic series
    one. Returns an empty list -- meaning "skip the panel" -- when the glyph
    mapping does not apply: a width other than 128 has no
    (cell, orientation bin) interpretation, and a series too small for its
    rows cannot be drawn. Every other panel in this report is
    dimension-agnostic, so this one drops out rather than failing the run.

    Samples are drawn at random under a fixed seed, never selected.
    """
    if num_samples <= 0:
        return []
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, np.ndarray, str]] = []
    for s in series:
        needed = 2 * num_samples if s.is_real else num_samples
        if s.x.shape[1] != DESCRIPTOR_DIM or s.x.shape[0] < needed:
            return []
        idx = rng.choice(s.x.shape[0], size=needed, replace=False)
        if s.is_real:
            rows.append(("real-a", s.x[idx[:num_samples]], GLYPH_REAL_COLORS[0]))
            rows.append(("real-b", s.x[idx[num_samples:]], GLYPH_REAL_COLORS[1]))
        else:
            rows.append((s.name, s.x[idx], s.color))
    return rows
