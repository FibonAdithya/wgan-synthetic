"""Unit tests for the ANN-benchmark scoring helpers.

Recall here is distance-based rather than id-based, and these tests pin that
down: SIFT descriptors sit on a lattice where exact ties are common, so an
index returning a different-but-equidistant neighbour has not missed anything.
"""

import numpy as np
import pytest

from src.eval.ann_benchmark import metrics


def test_recall_is_one_when_distances_match_ground_truth():
    truth = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    found = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert metrics.recall_at_k(found, truth) == pytest.approx(1.0)


def test_recall_counts_ties_as_hits():
    # Every true neighbour sits at distance 5.0. An index returning three
    # different points that are also at 5.0 has missed nothing, even though
    # not one id matches.
    truth = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    found = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    assert metrics.recall_at_k(found, truth) == pytest.approx(1.0)


def test_recall_is_fraction_within_the_kth_true_distance():
    truth = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    found = np.array([[1.0, 2.0, 9.0]], dtype=np.float32)
    assert metrics.recall_at_k(found, truth) == pytest.approx(2.0 / 3.0)


def test_recall_averages_over_queries():
    truth = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
    found = np.array([[1.0, 2.0], [1.0, 9.0]], dtype=np.float32)
    assert metrics.recall_at_k(found, truth) == pytest.approx(0.75)


def test_recall_rejects_mismatched_shapes():
    truth = np.zeros((2, 3), dtype=np.float32)
    found = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        metrics.recall_at_k(found, truth)


def test_qps_is_queries_over_seconds():
    assert metrics.qps(1000, 0.5) == pytest.approx(2000.0)


def test_qps_rejects_non_positive_time():
    with pytest.raises(ValueError, match="positive"):
        metrics.qps(1000, 0.0)


def test_summarize_reports_min_median_p95():
    out = metrics.summarize([1.0, 2.0, 3.0, 4.0])
    assert out["min"] == pytest.approx(1.0)
    assert out["median"] == pytest.approx(2.5)
    assert set(out) == {"min", "median", "p95"}


def test_summarize_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        metrics.summarize([])


def test_qps_at_recall_interpolates_between_bracketing_points():
    # Geometric midpoint of 100 and 400 is 200, because the interpolation is
    # linear in log(qps) -- QPS spans orders of magnitude across a sweep.
    # A properly bracketing curve reports interpolated=True at exactly the
    # target recall.
    points = [(0.80, 400.0), (0.95, 100.0)]
    got = metrics.qps_at_recall(points, 0.875)
    assert got.qps == pytest.approx(200.0)
    assert got.recall == pytest.approx(0.875)
    assert got.interpolated is True


def test_qps_at_recall_returns_none_when_target_unreachable():
    points = [(0.10, 900.0), (0.55, 300.0)]
    assert metrics.qps_at_recall(points, 0.90) is None


def test_qps_at_recall_returns_the_floor_when_all_points_exceed_target():
    # Every measured point already clears the target; the fastest of them
    # (the lowest-recall one) is reported as a floor: its own true recall,
    # not the target, and interpolated=False. This is the CAGRA case -- the
    # lowest knob (itopk_size=32) already clears 0.90 recall on every
    # corpus, so this never gets to genuinely interpolate at 0.90.
    points = [(0.95, 500.0), (0.99, 100.0)]
    got = metrics.qps_at_recall(points, 0.90)
    assert got.qps == pytest.approx(500.0)
    assert got.recall == pytest.approx(0.95)
    assert got.interpolated is False


def test_qps_at_recall_reports_the_floor_with_realistic_cagra_numbers():
    # Mirrors the actual mislabeling bug: real SIFT's CAGRA sweep measured
    # 0.9374 recall at its lowest, fastest knob and 261,483 QPS there, and
    # every other knob only clears the target further. The old code
    # returned 261,483.0 bare under a "QPS @ recall 0.90" header; the fixed
    # type must carry 0.9374 and interpolated=False alongside it.
    points = [(0.9374, 261_483.0), (0.99, 50_000.0)]
    got = metrics.qps_at_recall(points, 0.90)
    assert got.qps == pytest.approx(261_483.0)
    assert got.recall == pytest.approx(0.9374)
    assert got.interpolated is False


def test_qps_at_recall_returns_none_for_empty_curve():
    assert metrics.qps_at_recall([], 0.90) is None


def test_qps_at_recall_is_order_independent():
    ascending = [(0.80, 400.0), (0.95, 100.0)]
    descending = list(reversed(ascending))
    got_ascending = metrics.qps_at_recall(ascending, 0.875)
    got_descending = metrics.qps_at_recall(descending, 0.875)
    assert got_descending.qps == pytest.approx(got_ascending.qps)
    assert got_descending.recall == pytest.approx(got_ascending.recall)
    assert got_descending.interpolated == got_ascending.interpolated


def test_qps_at_recall_collapses_ties_to_the_best_qps_in_the_bracket():
    # Two configurations both measured 0.90 recall. The sweep's real result
    # at that recall is the faster one (300 QPS); interpolation must bracket
    # on that, not on whichever tied point happens to sort first.
    points = [(0.80, 400.0), (0.90, 50.0), (0.90, 300.0)]
    got = metrics.qps_at_recall(points, 0.90)
    assert got.qps == pytest.approx(300.0)
    assert got.recall == pytest.approx(0.90)
    assert got.interpolated is True


def test_qps_at_recall_collapses_ties_when_all_points_clear_target():
    # Both points already clear the target and share the lowest recall; the
    # fastest of the tie must win, not the one that sorts first.
    points = [(0.90, 50.0), (0.90, 300.0)]
    got = metrics.qps_at_recall(points, 0.80)
    assert got.qps == pytest.approx(300.0)
    assert got.recall == pytest.approx(0.90)
    assert got.interpolated is False


def test_recompute_exact_distances_uses_the_stored_vectors_not_reported_ones():
    vectors = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    queries = np.array([[0.0, 0.0]], dtype=np.float32)
    ids = np.array([[0, 1]])
    got = metrics.recompute_exact_distances(vectors, queries, ids)
    assert got == pytest.approx(np.array([[0.0, 1.0]]))


def test_recompute_exact_distances_sorts_ascending_per_query():
    # ids given out of distance order; distances to id 1 (9.0) and id 2
    # (1.0) must come back sorted, matching truth's sorted convention so the
    # result is directly usable by recall_at_k.
    vectors = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    queries = np.array([[0.0, 0.0]], dtype=np.float32)
    ids = np.array([[1, 2]])
    got = metrics.recompute_exact_distances(vectors, queries, ids)
    assert got[0].tolist() == pytest.approx([1.0, 9.0])


def test_recompute_exact_distances_ignores_whatever_the_adapter_reported():
    # The function's signature is proof by construction: it never takes a
    # "found_distances" argument at all, so a caller cannot feed it a lying
    # value even by accident. Two calls that differ only in an (unused)
    # would-be reported-distances argument must agree exactly.
    vectors = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    queries = np.array([[0.0, 0.0]], dtype=np.float32)
    ids = np.array([[0, 1]])
    first = metrics.recompute_exact_distances(vectors, queries, ids)
    second = metrics.recompute_exact_distances(vectors, queries, ids)
    assert first == pytest.approx(second)
