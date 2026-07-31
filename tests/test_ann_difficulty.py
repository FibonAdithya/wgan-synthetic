import numpy as np

from src.eval.ann_difficulty import gini


def test_gini_is_zero_for_perfectly_balanced_occupancy():
    assert gini(np.array([5, 5, 5, 5])) == 0.0


def test_gini_is_maximal_for_a_single_dominant_cluster():
    # With n cells and all mass in one, the Gini coefficient is exactly
    # (n - 1) / n. At n=4 that is 0.75.
    assert abs(gini(np.array([0, 0, 0, 20])) - 0.75) < 1e-9


def test_gini_is_zero_for_empty_occupancy():
    assert gini(np.array([0, 0, 0])) == 0.0


from src.eval.ann_difficulty import knn


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


from src.eval.ann_difficulty import lid_mle, survivor_mask


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


def test_lid_is_finite_for_every_surviving_query_when_duplicates_exist():
    base = _uniform_in_ball(2000, 4, seed=2)
    x = np.vstack([base, base[:200]])  # 200 exact duplicate rows
    dist, _, _ = knn(x, k=50)
    values = lid_mle(dist[survivor_mask(dist)])
    assert values.size > 0
    assert np.all(np.isfinite(values))
