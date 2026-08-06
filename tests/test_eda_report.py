import json
import sys
from pathlib import Path

import numpy as np
from tests.conftest import make_args

from src.eval import ann_difficulty, eda_report
from src.eval.eda import figures


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
    fig = figures.fig_ann_profile(series, metrics, bins=8)
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
