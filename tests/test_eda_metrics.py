import numpy as np

from src.eval.eda import metrics, series


def _series(name: str, x: np.ndarray, color: str = "#000000") -> series.Series:
    return series.Series(name, x.astype(np.float32), color)


def test_dimension_divergence_orders_dimensions_by_worst_mismatch():
    """Dimension 1 is the one the synthetics get wrong, so it must rank first."""
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 3))
    bad = real.copy()
    bad[:, 1] += 5.0

    div = metrics.dimension_divergence(
        [_series("real", real), _series("a", bad)], top_k=2
    )

    assert div.order[0] == 1
    assert div.worst["a"][0]["dim"] == 1
    assert div.worst["a"][0]["wasserstein1"] > div.worst["a"][1]["wasserstein1"]


def test_dimension_divergence_orders_by_the_worst_across_all_synthetics():
    """One shared x-axis ordering, driven by the worst offender on each dim.

    Series 'a' is wrong on dim 0 and 'b' on dim 2, so both must outrank the
    dimension neither one misses.
    """
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 3))
    a, b = real.copy(), real.copy()
    a[:, 0] += 9.0
    b[:, 2] += 5.0

    div = metrics.dimension_divergence(
        [_series("real", real), _series("a", a), _series("b", b)], top_k=3
    )

    assert list(div.order[:2]) == [0, 2]


def test_dimension_divergence_reports_top_k_dimensions_per_series():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(120, 5))

    div = metrics.dimension_divergence(
        [_series("real", real), _series("a", real + 1.0)], top_k=2
    )

    assert set(div.worst) == {"a"}
    assert len(div.worst["a"]) == 2
    assert all(isinstance(e["dim"], int) for e in div.worst["a"])
    assert all(isinstance(e["wasserstein1"], float) for e in div.worst["a"])
