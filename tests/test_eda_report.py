import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.eval import eda_report


def make_args(tmp_path, real, synthetic):
    real_path = tmp_path / "real.npy"
    np.save(real_path, real)
    specs = []
    for label, arr in synthetic.items():
        p = tmp_path / f"{label}.npy"
        np.save(p, arr)
        specs.append(f"{label}={p}")
    return argparse.Namespace(
        real_path=str(real_path),
        real_format="npy",
        synthetic_path=specs,
        synthetic_format="npy",
        output_dir=str(tmp_path / "out"),
        preprocess="l2",
        max_vectors=200,
        num_pairs=500,
        knn=3,
        ann_k=eda_report.ANN_K_DEFAULT,
        ann_hub_k=eda_report.ANN_HUB_K_DEFAULT,
        ann_max_rows=eda_report.ANN_MAX_ROWS_DEFAULT,
        ivf_nlist=eda_report.IVF_NLIST_DEFAULT,
        bins=16,
        top_divergent=4,
        seed=42,
        glyph_samples=eda_report.GLYPH_SAMPLES_DEFAULT,
        no_png=True,
        plotlyjs="cdn",
    )


def test_run_returns_written_report_path(tmp_path):
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 8)).astype(np.float32)
    synth = {"v0": rng.normal(size=(200, 8)).astype(np.float32)}

    out = eda_report.run(make_args(tmp_path, real, synth))

    assert isinstance(out, Path)
    assert out.exists()
    assert out.suffix == ".html"
    assert "v0" in out.read_text()


def test_run_accepts_several_synthetic_sets(tmp_path):
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 8)).astype(np.float32)
    synth = {
        "v0": rng.normal(size=(200, 8)).astype(np.float32),
        "v1": rng.normal(size=(200, 8)).astype(np.float32),
        "v2": rng.normal(size=(200, 8)).astype(np.float32),
    }

    html = eda_report.run(make_args(tmp_path, real, synth)).read_text()

    for label in ("v0", "v1", "v2"):
        assert label in html


def _sparse_sift_like(rows, seed, dim=128):
    """Sparse non-negative rows, standing in for real SIFT descriptors."""
    rng = np.random.default_rng(seed)
    x = rng.random((rows, dim)).astype(np.float32)
    x[x < 0.8] = 0.0
    x[:, 0] = 1.0  # no all-zero row
    return x


def test_glyph_section_is_included_for_128_dimensional_data(tmp_path):
    real = _sparse_sift_like(200, seed=0)
    synth = {"v0": _sparse_sift_like(200, seed=1)}

    html = eda_report.run(make_args(tmp_path, real, synth)).read_text()

    assert eda_report.GLYPH_SECTION_TITLE in html


def test_glyph_section_draws_two_real_rows_as_the_variation_baseline(tmp_path):
    """One real row gives no sense of how much two real descriptors differ,
    which is what a variant row has to be judged against."""
    real = _sparse_sift_like(200, seed=0)
    synth = {"v0": _sparse_sift_like(200, seed=1)}

    html = eda_report.run(make_args(tmp_path, real, synth)).read_text()

    assert "real-a" in html and "real-b" in html


def test_glyph_section_is_skipped_for_other_dimensions(tmp_path):
    """The (cell, orientation bin) mapping only exists at 128 dimensions, and
    the rest of the report is dimension-agnostic, so the panel drops out
    rather than taking the whole report down with it."""
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 8)).astype(np.float32)
    synth = {"v0": rng.normal(size=(200, 8)).astype(np.float32)}

    html = eda_report.run(make_args(tmp_path, real, synth)).read_text()

    assert eda_report.GLYPH_SECTION_TITLE not in html


def test_glyph_section_is_skipped_when_a_series_is_too_small(tmp_path):
    """Two disjoint real rows need 2 * glyph_samples vectors. Dense rather
    than sparse here: with this few rows a sparse column can end up constant,
    and the correlation panel warns on the zero standard deviation."""
    rng = np.random.default_rng(3)
    real = rng.random((6, 128)).astype(np.float32)
    synth = {"v0": rng.random((6, 128)).astype(np.float32)}

    args = make_args(tmp_path, real, synth)
    args.knn = 2
    args.glyph_samples = 8
    html = eda_report.run(args).read_text()

    assert eda_report.GLYPH_SECTION_TITLE not in html


def test_glyph_samples_zero_disables_the_section(tmp_path):
    real = _sparse_sift_like(200, seed=0)
    synth = {"v0": _sparse_sift_like(200, seed=1)}

    args = make_args(tmp_path, real, synth)
    args.glyph_samples = 0
    html = eda_report.run(args).read_text()

    assert eda_report.GLYPH_SECTION_TITLE not in html


def _write_set(path, rows, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    np.save(path, x)


def test_report_writes_html_and_summary_with_ann_sections(tmp_path, monkeypatch):
    real = tmp_path / "real.npy"
    fake = tmp_path / "fake.npy"
    _write_set(real, 400, seed=0)
    _write_set(fake, 400, seed=1)
    out = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eda_report",
            "--real-path",
            str(real),
            "--synthetic-path",
            f"fake={fake}",
            "--output-dir",
            str(out),
            "--ann-max-rows",
            "300",
            "--ann-k",
            "20",
            "--ann-hub-k",
            "5",
            "--ivf-nlist",
            "8",
            "--max-vectors",
            "400",
            "--num-pairs",
            "2000",
            "--no-png",
            "--plotlyjs",
            "cdn",
        ],
    )
    eda_report.main()

    html = (out / "eda_report.html").read_text(encoding="utf-8")
    assert "Local intrinsic dimensionality" in html
    assert "Hubness" in html
    assert "IVF cell balance" in html

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["ann_settings"]["k"] == 20
    for row in summary["stats"]:
        assert row["lid_median"] > 0
        assert "hubness_skew" in row
        assert "ivf_gini" in row


def test_maybe_l2_normalize_gives_unit_rows():
    x = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    out = eda_report.maybe_l2_normalize(x, "l2")
    assert np.linalg.norm(out, axis=1) == pytest.approx([1.0, 1.0])


def test_maybe_l2_normalize_leaves_a_zero_row_finite():
    """The eps clamp, not a divide by zero. `plot_descriptor_grid` hands this
    raw SIFT rows, and an all-zero descriptor is rare but legal."""
    out = eda_report.maybe_l2_normalize(np.zeros((1, 4), dtype=np.float32), "l2")
    assert np.all(np.isfinite(out))


def test_maybe_l2_normalize_none_mode_passes_rows_through():
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    assert np.array_equal(eda_report.maybe_l2_normalize(x, "none"), x)
