"""Tests for the benchmark grid loop.

Driven entirely by fake adapters. The runner must never name a cuVS type, and
these tests are what holds that line: they run on a CPU-only box.
"""

import json

import numpy as np
import pytest

from src.eval.ann_benchmark import corpora, indexes, runner


@pytest.fixture
def tiny_corpus(tmp_path):
    """A four-point corpus with two queries and exact ground truth on disk."""
    vectors = np.eye(4, dtype=np.float32)
    queries = np.eye(4, dtype=np.float32)[:2]
    corpus_dir = tmp_path / "tiny"
    corpus_dir.mkdir()
    np.save(corpus_dir / "vectors.npy", vectors)
    np.save(corpus_dir / "queries.npy", queries)

    adapter = indexes.NumpyFlatAdapter()
    built = adapter.build(vectors)
    dist, ids = adapter.search(built, queries, k=2, param=None)
    np.save(corpus_dir / "truth_distances.npy", dist)
    np.save(corpus_dir / "truth_ids.npy", ids)

    return corpora.Corpus(
        name="tiny",
        vectors_path=corpus_dir / "vectors.npy",
        queries_path=corpus_dir / "queries.npy",
        truth_distances_path=corpus_dir / "truth_distances.npy",
        truth_ids_path=corpus_dir / "truth_ids.npy",
        num_vectors=4,
        num_queries=2,
        dim=4,
    )


class SweepingAdapter(indexes.NumpyFlatAdapter):
    """Exact search that pretends to have a swept knob."""

    name = "sweeping"
    param_name = "n_probes"

    def sweep_params(self):
        return (1, 2)


class ExplodingAdapter(indexes.NumpyFlatAdapter):
    name = "exploding"

    def build(self, vectors):
        raise RuntimeError("out of memory")


class ExplodingSearchAdapter(indexes.NumpyFlatAdapter):
    name = "exploding_search"
    param_name = "n_probes"

    def sweep_params(self):
        return (1, 2)

    def search(self, built, queries, k, param):
        if param == 2:
            raise RuntimeError("search blew up")
        return super().search(built, queries, k, param)


class LyingDistancesAdapter(indexes.NumpyFlatAdapter):
    """Exact search (correct ids) that reports wildly wrong distances.

    Stands in for IVF-PQ's asymmetric distance computation: an index whose
    reported distances are not in the same space as the stored vectors.
    `run_grid` must score recall from distances it recomputes from the
    corpus vectors and the returned ids -- never from what `search()` itself
    hands back -- so this adapter's bogus distances must have zero effect on
    the recorded recall.
    """

    name = "lying_distances"

    def search(self, built, queries, k, param):
        _, ids = super().search(built, queries, k, param)
        # If these were used for scoring, every point would compare as a
        # miss (a real ground-truth distance can never be this large), and
        # recall would come out near 0.0 instead of the true 1.0.
        bogus_distances = np.full((queries.shape[0], k), 1.0e9, dtype=np.float32)
        return bogus_distances, ids


class CountingAdapter(indexes.NumpyFlatAdapter):
    """Exact search that counts every `search()` call it receives.

    Used to pin the warmup discipline: the runner must issue one untimed,
    discarded warmup search before a cell's timed repeats, so `repeats`
    timed calls must show up here as `repeats + 1`.
    """

    name = "counting"

    def __init__(self):
        super().__init__()
        self.search_calls = 0

    def search(self, built, queries, k, param):
        self.search_calls += 1
        return super().search(built, queries, k, param)


def test_exact_adapter_scores_perfect_recall(tiny_corpus, tmp_path):
    builds, searches = runner.run_grid(
        [tiny_corpus],
        [indexes.NumpyFlatAdapter()],
        k=2,
        repeats=2,
        records_path=tmp_path / "records.json",
    )
    assert len(builds) == 1
    assert len(searches) == 1
    assert searches[0].recall == pytest.approx(1.0)
    assert searches[0].failed is None


def test_one_record_per_swept_parameter(tiny_corpus, tmp_path):
    _, searches = runner.run_grid(
        [tiny_corpus],
        [SweepingAdapter()],
        k=2,
        repeats=1,
        records_path=tmp_path / "records.json",
    )
    assert [s.param_value for s in searches] == [1, 2]
    assert all(s.param_name == "n_probes" for s in searches)


def test_qps_summary_has_all_three_figures(tiny_corpus, tmp_path):
    _, searches = runner.run_grid(
        [tiny_corpus],
        [indexes.NumpyFlatAdapter()],
        k=2,
        repeats=3,
        records_path=tmp_path / "records.json",
    )
    record = searches[0]
    assert record.qps_min > 0.0
    assert record.qps_median >= record.qps_min
    assert record.qps_p95 >= record.qps_min


def test_a_failed_build_is_recorded_and_the_grid_continues(tiny_corpus, tmp_path):
    builds, searches = runner.run_grid(
        [tiny_corpus],
        [ExplodingAdapter(), indexes.NumpyFlatAdapter()],
        k=2,
        repeats=1,
        records_path=tmp_path / "records.json",
    )
    assert builds[0].failed is not None
    assert "out of memory" in builds[0].failed
    # No search records for the index that never built, and the next adapter
    # still ran: one bad cell must not cost the rest of the grid.
    assert [s.index for s in searches] == ["numpy_flat"]


def test_a_failed_search_leaves_its_siblings_intact(tiny_corpus, tmp_path):
    _, searches = runner.run_grid(
        [tiny_corpus],
        [ExplodingSearchAdapter()],
        k=2,
        repeats=1,
        records_path=tmp_path / "records.json",
    )
    by_param = {s.param_value: s for s in searches}
    assert by_param[1].failed is None
    assert by_param[2].failed is not None
    assert by_param[2].recall is None


def test_records_are_written_incrementally(tiny_corpus, tmp_path):
    records_path = tmp_path / "records.json"
    runner.run_grid(
        [tiny_corpus],
        [SweepingAdapter()],
        k=2,
        repeats=1,
        records_path=records_path,
    )
    payload = json.loads(records_path.read_text())
    assert len(payload["searches"]) == 2
    assert len(payload["builds"]) == 1
    assert payload["searches"][0]["corpus"] == "tiny"


def test_build_record_carries_the_fixed_parameters(tiny_corpus, tmp_path):
    builds, _ = runner.run_grid(
        [tiny_corpus],
        [indexes.NumpyFlatAdapter()],
        k=2,
        repeats=1,
        records_path=tmp_path / "records.json",
    )
    assert builds[0].params == {"metric": "sqeuclidean"}
    assert builds[0].index_bytes_estimated == 64


def test_warmup_precedes_every_cells_timed_repeats(tiny_corpus, tmp_path):
    # One swept parameter means one cell; a discarded warmup plus `repeats`
    # timed calls must show up as exactly `repeats + 1` calls to `search()`.
    # No warmup, or a warmup that also gets timed, would fail this.
    adapter = CountingAdapter()
    runner.run_grid(
        [tiny_corpus],
        [adapter],
        k=2,
        repeats=4,
        records_path=tmp_path / "records.json",
    )
    assert adapter.search_calls == 5


def test_recall_is_scored_from_recomputed_exact_distances_not_the_adapters_own(
    tiny_corpus, tmp_path
):
    # CRITICAL 2 regression test: an adapter reporting distances in the
    # wrong space (IVF-PQ's asymmetric distance computation, stood in for
    # here by deliberately bogus values) must not be able to inflate -- or
    # in this case, wreck -- its own recall. `run_grid` must recompute
    # exact distances from the corpus vectors and the returned ids and score
    # against those, so the recall recorded here is the true 1.0 (the ids
    # are exact), even though every reported distance was 1e9.
    _, searches = runner.run_grid(
        [tiny_corpus],
        [LyingDistancesAdapter()],
        k=2,
        repeats=1,
        records_path=tmp_path / "records.json",
    )
    assert searches[0].failed is None
    assert searches[0].recall == pytest.approx(1.0)
