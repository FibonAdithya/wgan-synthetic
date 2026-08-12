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
    points = [(0.80, 400.0), (0.95, 100.0)]
    got = metrics.qps_at_recall(points, 0.875)
    assert got == pytest.approx(200.0)


def test_qps_at_recall_returns_none_when_target_unreachable():
    points = [(0.10, 900.0), (0.55, 300.0)]
    assert metrics.qps_at_recall(points, 0.90) is None


def test_qps_at_recall_returns_fastest_point_when_all_exceed_target():
    points = [(0.95, 500.0), (0.99, 100.0)]
    assert metrics.qps_at_recall(points, 0.90) == pytest.approx(500.0)


def test_qps_at_recall_returns_none_for_empty_curve():
    assert metrics.qps_at_recall([], 0.90) is None


def test_qps_at_recall_is_order_independent():
    ascending = [(0.80, 400.0), (0.95, 100.0)]
    descending = list(reversed(ascending))
    assert metrics.qps_at_recall(descending, 0.875) == pytest.approx(
        metrics.qps_at_recall(ascending, 0.875)
    )
