import numpy as np
import pytest

from src.eval import ann_difficulty
from src.eval.ann_difficulty import (
    cell_occupancy,
    compute,
    gini,
    hubness_skew,
    k_occurrence,
    knn,
    lid_mle,
    relative_contrast,
    summary,
    survivor_mask,
)


def test_gini_is_zero_for_perfectly_balanced_occupancy():
    assert gini(np.array([5, 5, 5, 5])) == 0.0


def test_gini_is_maximal_for_a_single_dominant_cluster():
    # With n cells and all mass in one, the Gini coefficient is exactly
    # (n - 1) / n. At n=4 that is 0.75.
    assert abs(gini(np.array([0, 0, 0, 20])) - 0.75) < 1e-9


def test_gini_is_zero_for_empty_occupancy():
    assert gini(np.array([0, 0, 0])) == 0.0


def test_knn_excludes_the_query_itself():
    x = np.eye(6, dtype=np.float32)
    _, idx, k_eff = knn(x, k=3)
    assert k_eff == 3
    assert idx.shape == (6, 3)
    for row in range(6):
        assert row not in idx[row].tolist()


def test_knn_excludes_self_even_when_a_duplicate_ties_at_zero_distance():
    # Rows 0 and 1 are identical. Naively stripping column 0 can remove the
    # duplicate instead of the query, leaving the query in its own list.
    x = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    _, idx, _ = knn(x, k=2)
    for row in range(4):
        assert row not in idx[row].tolist()


def test_knn_clamps_k_to_available_neighbours():
    x = np.eye(4, dtype=np.float32)
    dist, idx, k_eff = knn(x, k=100)
    assert k_eff == 3
    assert dist.shape == (4, 3)
    assert idx.shape == (4, 3)


def test_knn_returns_distances_in_ascending_order():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 8)).astype(np.float32)
    dist, _, _ = knn(x, k=5)
    assert np.all(np.diff(dist, axis=1) >= -1e-6)


def _uniform_in_ball(n, d, seed):
    """Sample uniformly inside the unit d-ball. LID of such a set equals d."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, d))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = rng.random(size=(n, 1)) ** (1.0 / d)
    return (direction * radius).astype(np.float32)


def test_lid_recovers_the_generating_dimension():
    x = _uniform_in_ball(20000, 4, seed=0)
    dist, _, _ = knn(x, k=100)
    estimate = float(np.median(lid_mle(dist[survivor_mask(dist)])))
    # The Hill estimator is biased and its bias grows with d/n, so this is a
    # deliberately loose 20% band around the true value of 4.
    assert 3.2 < estimate < 4.8


def test_lid_rises_with_the_generating_dimension():
    low = _uniform_in_ball(20000, 4, seed=1)
    high = _uniform_in_ball(20000, 12, seed=1)
    d_low, _, _ = knn(low, k=100)
    d_high, _, _ = knn(high, k=100)
    lid_low = float(np.median(lid_mle(d_low[survivor_mask(d_low)])))
    lid_high = float(np.median(lid_mle(d_high[survivor_mask(d_high)])))
    assert lid_high > lid_low


def test_survivor_mask_rejects_queries_with_a_zero_nearest_distance():
    dist = np.array([[0.0, 1.0], [0.5, 1.0], [0.0, 2.0]])
    assert survivor_mask(dist).tolist() == [False, True, False]


def test_survivor_mask_rejects_queries_whose_k_neighbours_all_tie():
    # Row 0: every neighbour distance is identical (r_1 == r_k), which would
    # send lid_mle's ratio to 1.0 everywhere and its estimate to -inf. Row 1
    # has a genuine spread and should survive.
    dist = np.array([[1.0, 1.0, 1.0], [0.5, 0.8, 1.0]])
    assert survivor_mask(dist).tolist() == [False, True]


def test_lid_mle_would_be_negative_infinity_for_all_tied_neighbours():
    # Documents the failure mode survivor_mask now guards against: passing a
    # row with r_1 == r_k straight to lid_mle (bypassing the mask) still
    # produces -inf, confirming the mask is the thing doing the rejecting.
    dist = np.array([[1.0, 1.0, 1.0]])
    with np.errstate(divide="ignore", invalid="ignore"):
        result = lid_mle(dist)
    assert result[0] == float("-inf")


def test_compute_discards_all_tied_neighbour_queries_without_producing_inf_or_nan():
    # np.eye(12): every off-diagonal distance is sqrt(2), so every query's k
    # neighbours tie exactly. Before the fix this produced lid_median = -inf
    # via an unguarded -1/0 in lid_mle, silently poisoning shared_edges and
    # the histogram built on top of it.
    x = np.eye(12, dtype=np.float32)
    metrics = compute(x, k=5)
    assert metrics.lid.size == 0
    assert metrics.discarded_queries == 12
    assert not np.any(np.isinf(metrics.lid))
    assert not np.any(np.isnan(metrics.lid))
    result = summary(metrics)
    assert result["lid_median"] is None


def test_lid_is_finite_for_every_surviving_query_when_duplicates_exist():
    base = _uniform_in_ball(2000, 4, seed=2)
    x = np.vstack([base, base[:200]])  # 200 exact duplicate rows
    dist, _, _ = knn(x, k=50)
    values = lid_mle(dist[survivor_mask(dist)])
    assert values.size > 0
    assert np.all(np.isfinite(values))


def test_relative_contrast_falls_as_dimension_rises():
    # Distances concentrate in high dimensions, so the gap between the mean
    # distance and the nearest distance shrinks and search gets harder.
    rng = np.random.default_rng(3)
    low = rng.normal(size=(3000, 2)).astype(np.float32)
    high = rng.normal(size=(3000, 64)).astype(np.float32)
    d_low, _, _ = knn(low, k=10)
    d_high, _, _ = knn(high, k=10)
    rc_low = float(np.median(relative_contrast(low, d_low, seed=0)))
    rc_high = float(np.median(relative_contrast(high, d_high, seed=0)))
    assert rc_high < rc_low


def test_relative_contrast_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(1000, 8)).astype(np.float32)
    dist, _, _ = knn(x, k=10)
    first = relative_contrast(x, dist, seed=7)
    second = relative_contrast(x, dist, seed=7)
    assert np.array_equal(first, second)


def test_relative_contrast_returns_one_value_per_row():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(500, 8)).astype(np.float32)
    dist, _, _ = knn(x, k=10)
    assert relative_contrast(x, dist, seed=0).shape == (500,)


def test_k_occurrence_conserves_total_count():
    # Every one of the n queries contributes exactly k_hub list entries, so
    # the counts must total n * k_hub. This catches off-by-one slips in the
    # index bookkeeping.
    rng = np.random.default_rng(6)
    x = rng.normal(size=(400, 8)).astype(np.float32)
    _, idx, _ = knn(x, k=20)
    counts = k_occurrence(idx, n=400, k_hub=10)
    assert counts.sum() == 400 * 10
    assert counts.shape == (400,)


def test_hubness_skew_is_higher_when_a_hub_is_planted():
    rng = np.random.default_rng(7)
    shell = rng.normal(size=(1500, 6)).astype(np.float32)
    shell /= np.linalg.norm(shell, axis=1, keepdims=True)
    # A tight knot of near-duplicates is close to everything inside
    # itself, so those points crowd into each other's lists and take a
    # disproportionate share of the neighbour mass.
    tight = shell[:1] + rng.normal(size=(50, 6)).astype(np.float32) * 0.01
    tight /= np.linalg.norm(tight, axis=1, keepdims=True)
    planted = np.vstack([shell, tight])

    _, idx_plain, _ = knn(shell, k=20)
    _, idx_planted, _ = knn(planted, k=20)
    skew_plain = hubness_skew(k_occurrence(idx_plain, shell.shape[0], 10))
    skew_planted = hubness_skew(k_occurrence(idx_planted, planted.shape[0], 10))
    assert skew_planted > skew_plain


def test_hubness_skew_is_zero_for_a_flat_count_distribution():
    assert hubness_skew(np.array([4, 4, 4, 4])) == 0.0


def test_cell_occupancy_totals_the_row_count_and_sorts_ascending():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(600, 8)).astype(np.float32)
    occupancy, nlist_eff = cell_occupancy(x, nlist=16, seed=0)
    assert occupancy.sum() == 600
    assert occupancy.shape == (nlist_eff,)
    assert np.all(np.diff(occupancy) >= 0)


def test_cell_occupancy_clamps_nlist_to_half_the_row_count():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(40, 4)).astype(np.float32)
    _, nlist_eff = cell_occupancy(x, nlist=256, seed=0)
    assert nlist_eff == 20


def test_cell_occupancy_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(600, 8)).astype(np.float32)
    first, _ = cell_occupancy(x, nlist=16, seed=3)
    second, _ = cell_occupancy(x, nlist=16, seed=3)
    assert np.array_equal(first, second)


def test_well_separated_blobs_partition_more_evenly_than_one_dense_lump():
    rng = np.random.default_rng(11)
    centres = rng.normal(size=(8, 6)).astype(np.float32) * 30.0
    blobs = np.repeat(centres, 100, axis=0) + rng.normal(size=(800, 6)).astype(
        np.float32
    )
    lump = rng.normal(size=(800, 6)).astype(np.float32)
    blob_occupancy, _ = cell_occupancy(blobs, nlist=8, seed=0)
    lump_occupancy, _ = cell_occupancy(lump, nlist=8, seed=0)
    assert gini(blob_occupancy) < gini(lump_occupancy)


def test_compute_truncates_to_max_rows():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(5000, 8)).astype(np.float32)
    metrics = compute(x, k=20, k_hub=5, nlist=16, max_rows=1000, seed=0)
    assert metrics.num_rows == 1000
    assert metrics.k_occurrence.shape == (1000,)


def test_compute_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(1200, 8)).astype(np.float32)
    kwargs = dict(k=20, k_hub=5, nlist=16, max_rows=800, seed=5)
    first = compute(x, **kwargs)
    second = compute(x, **kwargs)
    assert np.array_equal(first.lid, second.lid)
    assert np.array_equal(first.k_occurrence, second.k_occurrence)
    assert np.array_equal(first.cell_occupancy, second.cell_occupancy)


def test_compute_counts_discarded_duplicate_queries():
    rng = np.random.default_rng(14)
    base = rng.normal(size=(600, 8)).astype(np.float32)
    x = np.vstack([base, base[:100]])
    metrics = compute(x, k=20, k_hub=5, nlist=16, max_rows=0, seed=0)
    assert metrics.discarded_queries >= 200
    assert np.all(np.isfinite(metrics.lid))
    assert np.all(np.isfinite(metrics.relative_contrast))


def test_compute_survives_a_set_that_is_entirely_duplicates():
    x = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (300, 1))
    metrics = compute(x, k=10, k_hub=5, nlist=8, max_rows=0, seed=0)
    assert metrics.lid.size == 0
    assert metrics.discarded_queries == 300
    assert summary(metrics)["lid_median"] is None


def test_summary_returns_the_agreed_keys():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(800, 8)).astype(np.float32)
    result = summary(compute(x, k=20, k_hub=5, nlist=16, max_rows=0, seed=0))
    assert set(result) == {
        "lid_median",
        "relative_contrast_median",
        "hubness_skew",
        "ivf_gini",
        "lid_discarded_queries",
    }
    assert result["lid_median"] > 0
    assert result["lid_discarded_queries"] == 0


def test_summary_reports_the_discarded_count_as_an_int_not_a_float():
    """It is a tally, and `format(1200000.0, '.6g')` renders `1.2e+06` in the
    report's statistics table."""
    x = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (40, 1))
    result = summary(compute(x, k=10, k_hub=5, nlist=8, max_rows=0, seed=0))
    assert isinstance(result["lid_discarded_queries"], int)


def test_compute_discards_every_query_when_k_clamps_to_one():
    """A two-row set clamps k_eff to 1, where r_1 and r_k are the same column
    so survivor_mask's r_1 < r_k can never hold. The report's discarded note
    exists to explain this rather than let it render as a silent n/a."""
    x = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    metrics = compute(x, k=100, k_hub=5, nlist=8, max_rows=0, seed=0)
    assert metrics.k == 1
    assert metrics.discarded_queries == metrics.num_rows
    assert summary(metrics)["lid_median"] is None


def test_exclude_self_drops_the_query_from_its_own_row():
    # Row 1 came back as its own nearest neighbour, which is what the
    # +1 column exists to absorb.
    dist = np.array([[0.0, 1.0, 2.0], [0.0, 1.5, 2.5]])
    idx = np.array([[0, 1, 2], [1, 0, 2]])

    kept_dist, kept_idx = ann_difficulty._exclude_self(dist, idx, 2)

    np.testing.assert_array_equal(kept_idx, [[1, 2], [0, 2]])
    np.testing.assert_allclose(kept_dist, [[1.0, 2.0], [1.5, 2.5]])


def test_exclude_self_drops_the_farthest_when_the_query_did_not_come_back():
    # Row 0's own index is absent, so it has three keepers for two slots
    # and the farthest must go -- not an arbitrary one.
    dist = np.array([[0.5, 1.0, 2.0]])
    idx = np.array([[7, 8, 9]])

    kept_dist, kept_idx = ann_difficulty._exclude_self(dist, idx, 2)

    np.testing.assert_array_equal(kept_idx, [[7, 8]])
    np.testing.assert_allclose(kept_dist, [[0.5, 1.0]])


def test_torch_backend_returns_the_same_neighbours_as_sklearn():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 16)).astype(np.float32)

    d_sk, i_sk, k_sk = ann_difficulty.knn(x, 10)
    d_t, i_t, k_t = ann_difficulty.knn(x, 10, backend="torch")

    assert k_t == k_sk
    np.testing.assert_allclose(d_t, d_sk, rtol=1e-4, atol=1e-4)
    # Indices are allowed to differ only where the two candidates are
    # indistinguishable at that tolerance -- a genuine tie, not a wrong
    # neighbour.
    differing = i_t != i_sk
    assert np.all(np.abs(d_t[differing] - d_sk[differing]) <= 1e-4)


def test_torch_backend_excludes_self_when_every_row_is_a_duplicate():
    # Every distance is 0.0, so nothing about the sort order can be relied
    # on. This is the case the index-based exclusion exists for.
    x = np.zeros((6, 4), dtype=np.float32)

    _, idx, k_eff = ann_difficulty.knn(x, 3, backend="torch")

    assert k_eff == 3
    assert not np.any(idx == np.arange(6)[:, None])


def test_torch_backend_agrees_with_sklearn_on_every_summary_statistic():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((500, 12)).astype(np.float32)

    sk = ann_difficulty.summary(
        ann_difficulty.compute(x, k=20, k_hub=5, nlist=8, max_rows=0)
    )
    torch_ = ann_difficulty.summary(
        ann_difficulty.compute(x, k=20, k_hub=5, nlist=8, max_rows=0, backend="torch")
    )

    for name in ("lid_median", "relative_contrast_median", "hubness_skew", "ivf_gini"):
        assert sk[name] == pytest.approx(torch_[name], rel=1e-4, abs=1e-6), name


def test_knn_rejects_an_unknown_backend():
    x = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="backend"):
        ann_difficulty.knn(x, 2, backend="faiss")


def test_torch_backend_is_unaffected_by_chunk_width():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((257, 8)).astype(np.float32)

    wide = ann_difficulty.knn(x, 5, backend="torch", chunk_rows=1024)
    narrow = ann_difficulty.knn(x, 5, backend="torch", chunk_rows=7)

    np.testing.assert_array_equal(wide[1], narrow[1])
    np.testing.assert_allclose(wide[0], narrow[0], rtol=1e-5, atol=1e-6)


def test_hubness_gini_is_zero_when_every_point_is_drawn_on_equally():
    counts = np.full(100, 7)
    assert ann_difficulty.hubness_gini(counts) == pytest.approx(0.0, abs=1e-12)


def test_hubness_gini_approaches_one_when_a_single_hub_takes_everything():
    counts = np.zeros(100)
    counts[0] = 100.0
    # (n - 1) / n for n = 100.
    assert ann_difficulty.hubness_gini(counts) == pytest.approx(0.99)


def test_hub_share_top1pct_is_one_percent_under_a_flat_distribution():
    counts = np.full(100, 7)
    assert ann_difficulty.hub_share_top1pct(counts) == pytest.approx(0.01)


def test_hub_share_top1pct_is_one_when_a_single_hub_takes_everything():
    counts = np.zeros(100)
    counts[0] = 100.0
    assert ann_difficulty.hub_share_top1pct(counts) == pytest.approx(1.0)


def test_hub_share_top1pct_rounds_the_top_slice_up_to_at_least_one_point():
    # 1% of 10 points is 0.1; taking zero points would make the statistic
    # meaningless on small sets.
    counts = np.array([5, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert ann_difficulty.hub_share_top1pct(counts) == pytest.approx(5.0 / 14.0)


def test_hub_statistics_are_zero_on_an_empty_neighbour_budget():
    # Every count zero means no neighbour slots were handed out at all.
    counts = np.zeros(50)
    assert ann_difficulty.hubness_gini(counts) == 0.0
    assert ann_difficulty.hub_share_top1pct(counts) == 0.0
