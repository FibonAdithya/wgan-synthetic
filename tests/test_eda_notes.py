import numpy as np

from src.eval import ann_difficulty
from src.eval.eda import notes
from src.eval.eda.series import Series


def _stub_series(name: str) -> Series:
    """A Series carrying the smallest array the note helpers never look at.

    `ann_condition_note` and `ann_discarded_note` read only `.name` and the
    metrics keyed by it, so building real vectors here would only slow the
    test down without exercising anything more.
    """
    return Series(name, np.zeros((1, 2), dtype=np.float32), "#000000")


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
    note = notes.ann_condition_note(series, metrics, (("num_rows", "rows"), ("k", "k")))
    assert note == " Measured with rows=300, k=20 for every series."


def test_ann_condition_note_spells_out_every_series_when_conditions_diverge():
    """The clamped branch: a short series gets its own k, so one summary
    sentence would let a reader read the majority's k as everyone's."""
    series = [_stub_series("real"), _stub_series("tiny")]
    metrics = {
        "real": _stub_metrics(num_rows=300, k=20),
        "tiny": _stub_metrics(num_rows=15, k=14),
    }
    note = notes.ann_condition_note(series, metrics, (("num_rows", "rows"), ("k", "k")))
    assert "differ across series" in note
    assert "real (rows=300, k=20)" in note, note
    assert "tiny (rows=15, k=14)" in note, note


def test_ann_condition_note_diverges_on_a_single_attribute_too():
    series = [_stub_series("real"), _stub_series("tiny")]
    metrics = {
        "real": _stub_metrics(num_rows=300, k=20),
        "tiny": _stub_metrics(num_rows=15, k=20),
    }
    note = notes.ann_condition_note(series, metrics, (("num_rows", "rows"),))
    assert "real (rows=300)" in note, note
    assert "tiny (rows=15)" in note, note


def test_ann_discarded_note_is_empty_when_every_series_kept_some_queries():
    series = [_stub_series("real")]
    metrics = {"real": _stub_metrics(num_rows=300, k=20, discarded=12)}
    assert notes.ann_discarded_note(series, metrics) == ""


def test_ann_discarded_note_names_a_series_whose_queries_were_all_discarded():
    series = [_stub_series("real"), _stub_series("dupes")]
    metrics = {
        "real": _stub_metrics(num_rows=300, k=20, discarded=0),
        "dupes": _stub_metrics(num_rows=300, k=20, discarded=300),
    }
    note = notes.ann_discarded_note(series, metrics)
    assert "dupes" in note
    assert "real" not in note, "a series with survivors must not be called out"
    assert "exact duplicate" in note


def test_ann_discarded_note_blames_k_equals_one_when_that_is_the_cause():
    """k_eff == 1 makes survivor_mask's r_1 < r_k unsatisfiable, so every
    query is dropped for a reason that has nothing to do with duplicates."""
    series = [_stub_series("real")]
    metrics = {"real": _stub_metrics(num_rows=2, k=1, nlist=2, discarded=2)}
    note = notes.ann_discarded_note(series, metrics)
    assert "k=1" in note, note
