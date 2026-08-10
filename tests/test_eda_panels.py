"""The panel registry: every panel declares its own title, note and builder."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from src.eval import ann_difficulty
from src.eval.eda import config, metrics, panels, series

# List order is report order (pipeline.py walks PANELS directly). This pins
# the report's whole composition -- count, order and titles -- now that the
# out-of-repo golden harness that used to catch a reorder or deletion is gone.
EXPECTED_PANEL_TITLES = [
    "Descriptor glyphs",
    "Local intrinsic dimensionality",
    "Hubness",
    "IVF cell balance",
    "Pooled value distribution",
    "Per-dimension marginals",
    "Per-dimension profiles",
    "Pairwise distances",
    "Within-set 5-NN distances",  # 5 is _namespace()'s --knn default
    "Vector norms",
    "PCA spectrum",
    "Correlation structure",
    "Per-dimension mismatch",
]


def _namespace(preprocess: str = "l2") -> argparse.Namespace:
    """A fully populated Namespace, as argparse would hand one to run().

    Local to this file rather than shared: tests/conftest.py's make_args
    writes .npy files and needs a tmp_path, and no panel test loads from disk.
    """
    return argparse.Namespace(
        real_path="r.npy",
        real_format="auto",
        synthetic_path=["a=a.npy"],
        synthetic_format="auto",
        output_dir="out",
        preprocess=preprocess,
        max_vectors=50000,
        num_pairs=500,
        knn=5,
        ann_k=10,
        ann_hub_k=5,
        ann_max_rows=200,
        knn_max_rows=200,
        ivf_nlist=8,
        bins=16,
        top_divergent=4,
        seed=0,
        no_png=True,
        glyph_samples=config.GLYPH_SAMPLES_DEFAULT,
        plotlyjs="cdn",
    )


def _by_title(
    all_panels: Sequence[panels.Panel], title: str, ctx: panels.Context
) -> panels.Panel:
    """The single panel whose resolved title is `title`."""
    found = [p for p in all_panels if p.resolve_title(ctx) == title]
    if len(found) != 1:
        raise AssertionError(f"expected exactly one {title!r} panel, got {len(found)}")
    return found[0]


def _context(dim: int = 128, num_synth: int = 2, preprocess: str = "l2"):
    rng = np.random.default_rng(0)
    sets = [series.Series("real", rng.random((300, dim), dtype=np.float32), "#000")]
    for i in range(num_synth):
        sets.append(
            series.Series(f"s{i}", rng.random((300, dim), dtype=np.float32), "#111")
        )
    cfg = config.EdaConfig.from_args(_namespace(preprocess=preprocess))
    ann_metrics = {
        s.name: ann_difficulty.compute(
            s.x, k=10, k_hub=5, nlist=8, max_rows=200, seed=0
        )
        for s in sets
    }
    return panels.Context(
        config=cfg,
        series=sets,
        ann_metrics=ann_metrics,
        # Computed rather than hardcoded to None: _build_mismatch skips on
        # `divergence is None`, so a fixture that always passes None makes the
        # skip test pass whatever num_synth is. run() derives it the same way.
        divergence=(
            metrics.dimension_divergence(sets, cfg.top_divergent)
            if len(sets) > 1
            else None
        ),
    )


def test_panels_are_registered_in_report_order():
    """List order is report order; nothing else pins it once the golden harness is gone."""
    ctx = _context()
    assert [p.resolve_title(ctx) for p in panels.PANELS] == EXPECTED_PANEL_TITLES


def test_every_panel_declares_a_title_and_a_note():
    """A panel with an empty title renders as an unlabelled <h2>."""
    ctx = _context()

    for panel in panels.PANELS:
        assert panel.resolve_title(ctx).strip()
        assert panel.resolve_note(ctx).strip()


def test_panel_titles_are_unique():
    """export_pngs slugs the title into a filename; duplicates collide."""
    ctx = _context()
    titles = [p.resolve_title(ctx) for p in panels.PANELS]

    assert len(titles) == len(set(titles))


def test_vector_norms_panel_is_omitted_under_l2_normalization():
    """Norms are all 1.0 after L2, so the panel would say nothing."""
    ctx = _context(preprocess="l2")
    panel = _by_title(panels.PANELS, "Vector norms", ctx)

    assert panel.build(ctx) is None


def test_vector_norms_panel_is_built_without_normalization():
    ctx = _context(preprocess="none")
    panel = _by_title(panels.PANELS, "Vector norms", ctx)

    assert isinstance(panel.build(ctx), go.Figure)


def test_mismatch_panel_is_omitted_without_a_synthetic_overlay():
    ctx = _context(num_synth=0)
    panel = _by_title(panels.PANELS, "Per-dimension mismatch", ctx)

    assert panel.build(ctx) is None


def test_glyph_panel_is_omitted_for_non_128_dimensional_data():
    ctx = _context(dim=64)
    panel = _by_title(panels.PANELS, "Descriptor glyphs", ctx)

    assert panel.build(ctx) is None
