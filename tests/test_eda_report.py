import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.eval import ann_difficulty, eda_report


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
        knn_max_rows=eda_report.KNN_MAX_ROWS_DEFAULT,
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


def _stub_series(name: str) -> eda_report.Series:
    """A Series carrying the smallest array the note helpers never look at.

    `ann_condition_note` and `ann_discarded_note` read only `.name` and the
    metrics keyed by it, so building real vectors here would only slow the
    test down without exercising anything more.
    """
    return eda_report.Series(name, np.zeros((1, 2), dtype=np.float32), "#000000")


def _stub_metrics(
    *, num_rows: int, k: int, nlist: int = 8, discarded: int = 0
) -> ann_difficulty.AnnMetrics:
    """An AnnMetrics with only the scalar fields the note helpers read set."""
    empty = np.empty(0, dtype=np.float64)
    return ann_difficulty.AnnMetrics(
        lid=empty,
        relative_contrast=empty,
        k_occurrence=np.zeros(num_rows, dtype=np.int64),
        cell_occupancy=np.zeros(nlist, dtype=np.int64),
        num_rows=num_rows,
        k=k,
        nlist=nlist,
        discarded_queries=discarded,
    )


def test_ann_condition_note_states_one_condition_when_every_series_matches():
    series = [_stub_series("real"), _stub_series("fake")]
    metrics = {
        "real": _stub_metrics(num_rows=300, k=20),
        "fake": _stub_metrics(num_rows=300, k=20),
    }
    note = eda_report.ann_condition_note(
        series, metrics, (("num_rows", "rows"), ("k", "k"))
    )
    assert note == " Measured with rows=300, k=20 for every series."


def test_ann_condition_note_spells_out_every_series_when_conditions_diverge():
    """The clamped branch: a short series gets its own k, so one summary
    sentence would let a reader read the majority's k as everyone's."""
    series = [_stub_series("real"), _stub_series("tiny")]
    metrics = {
        "real": _stub_metrics(num_rows=300, k=20),
        "tiny": _stub_metrics(num_rows=15, k=14),
    }
    note = eda_report.ann_condition_note(
        series, metrics, (("num_rows", "rows"), ("k", "k"))
    )
    assert "differ across series" in note
    assert "real (rows=300, k=20)" in note, note
    assert "tiny (rows=15, k=14)" in note, note


def test_ann_condition_note_diverges_on_a_single_attribute_too():
    series = [_stub_series("real"), _stub_series("tiny")]
    metrics = {
        "real": _stub_metrics(num_rows=300, k=20),
        "tiny": _stub_metrics(num_rows=15, k=20),
    }
    note = eda_report.ann_condition_note(series, metrics, (("num_rows", "rows"),))
    assert "real (rows=300)" in note, note
    assert "tiny (rows=15)" in note, note


def test_ann_discarded_note_is_empty_when_every_series_kept_some_queries():
    series = [_stub_series("real")]
    metrics = {"real": _stub_metrics(num_rows=300, k=20, discarded=12)}
    assert eda_report.ann_discarded_note(series, metrics) == ""


def test_ann_discarded_note_names_a_series_whose_queries_were_all_discarded():
    series = [_stub_series("real"), _stub_series("dupes")]
    metrics = {
        "real": _stub_metrics(num_rows=300, k=20, discarded=0),
        "dupes": _stub_metrics(num_rows=300, k=20, discarded=300),
    }
    note = eda_report.ann_discarded_note(series, metrics)
    assert "dupes" in note
    assert "real" not in note, "a series with survivors must not be called out"
    assert "exact duplicate" in note


def test_ann_discarded_note_blames_k_equals_one_when_that_is_the_cause():
    """k_eff == 1 makes survivor_mask's r_1 < r_k unsatisfiable, so every
    query is dropped for a reason that has nothing to do with duplicates."""
    series = [_stub_series("real")]
    metrics = {"real": _stub_metrics(num_rows=2, k=1, nlist=2, discarded=2)}
    note = eda_report.ann_discarded_note(series, metrics)
    assert "k=1" in note, note


def test_format_stat_renders_counts_as_integers_not_scientific_notation():
    assert eda_report.format_stat(1200000) == "1200000"
    assert eda_report.format_stat(np.int64(1200000)) == "1200000"
    assert eda_report.format_stat(None) == "n/a"
    assert eda_report.format_stat(0.5) == "0.5"


def test_stats_table_renders_a_large_discarded_count_as_a_tally():
    html = eda_report.stats_table_html(
        [{"name": "real", "lid_discarded_queries": 1200000, "lid_median": 12.5}]
    )
    assert "1200000" in html
    assert "1.2e+06" not in html


def test_fig_ann_profile_annotates_a_panel_with_no_surviving_queries():
    """Every series fully degenerate: without the annotation the subplot is a
    bare pair of axes that reads as a rendering failure."""
    series = [_stub_series("real")]
    metrics = {"real": _stub_metrics(num_rows=5, k=1, nlist=2, discarded=5)}
    fig = eda_report.fig_ann_profile(series, metrics, bins=8)
    texts = [a.text for a in fig.layout.annotations]
    assert texts.count("no surviving queries") == 2, texts


def test_knn_max_rows_is_a_separate_flag_from_ann_max_rows(monkeypatch, tmp_path):
    """The within-set k-NN panel is not an ANN-difficulty panel, so tuning the
    cost of the difficulty metrics must not move it."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eda_report",
            "--real-path",
            "real.npy",
            "--output-dir",
            str(tmp_path / "out"),
            "--ann-max-rows",
            "500",
        ],
    )
    args = eda_report.parse_args()
    assert args.ann_max_rows == 500
    assert args.knn_max_rows == eda_report.KNN_MAX_ROWS_DEFAULT


def test_knn_max_rows_defaults_to_the_same_value_as_ann_max_rows():
    """Same default, so no existing invocation changes what it measures."""
    assert eda_report.KNN_MAX_ROWS_DEFAULT == eda_report.ANN_MAX_ROWS_DEFAULT
