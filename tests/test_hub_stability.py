import numpy as np
import pytest

from src.eval import hub_stability


def test_draws_are_disjoint_when_the_pool_can_afford_it():
    draws, disjoint = hub_stability.allocate_draws(1000, 100, 10, seed=42)

    assert disjoint is True
    assert len(draws) == 10
    assert all(d.shape == (100,) for d in draws)
    combined = np.concatenate(draws)
    assert combined.size == np.unique(combined).size


def test_draws_overlap_and_say_so_when_the_pool_cannot():
    draws, disjoint = hub_stability.allocate_draws(1000, 400, 10, seed=42)

    assert disjoint is False
    assert len(draws) == 10
    # Each draw is still internally without replacement.
    assert all(np.unique(d).size == 400 for d in draws)


def test_the_exact_boundary_where_the_pool_is_used_up_is_still_disjoint():
    _, disjoint = hub_stability.allocate_draws(1000, 100, 10, seed=42)
    assert disjoint is True
    _, one_more = hub_stability.allocate_draws(999, 100, 10, seed=42)
    assert one_more is False


def test_draw_indices_are_sorted_so_the_subsample_preserves_corpus_order():
    draws, _ = hub_stability.allocate_draws(1000, 100, 3, seed=7)
    for d in draws:
        np.testing.assert_array_equal(d, np.sort(d))


def test_allocation_is_reproducible_under_the_same_seed():
    first, _ = hub_stability.allocate_draws(1000, 100, 4, seed=11)
    second, _ = hub_stability.allocate_draws(1000, 100, 4, seed=11)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_a_draw_larger_than_the_pool_is_an_error():
    with pytest.raises(hub_stability.HubStabilityError, match="pool"):
        hub_stability.allocate_draws(50, 100, 2, seed=42)


def _draw(rows: int = 300, dim: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, dim)).astype(np.float32)


def test_measure_draw_returns_every_statistic_as_a_finite_number():
    values = hub_stability.measure_draw(
        _draw(), k=10, k_hub=5, nlist=4, seed=42, backend="sklearn", chunk_rows=1024
    )

    assert sorted(values) == sorted(hub_stability.STATISTICS)
    assert all(np.isfinite(v) for v in values.values())


def test_measure_draw_measures_every_row_it_is_given():
    # max_rows must be disabled inside: the caller has already drawn the
    # rows, and a second subsample would silently shrink the draw.
    big = hub_stability.measure_draw(
        _draw(rows=400), k=10, k_hub=5, nlist=4, seed=42,
        backend="sklearn", chunk_rows=1024,
    )
    small = hub_stability.measure_draw(
        _draw(rows=400)[:200], k=10, k_hub=5, nlist=4, seed=42,
        backend="sklearn", chunk_rows=1024,
    )
    assert big["lid_median"] != small["lid_median"]


def test_measure_draw_is_deterministic():
    kwargs = dict(k=10, k_hub=5, nlist=4, seed=42, backend="sklearn", chunk_rows=1024)
    first = hub_stability.measure_draw(_draw(), **kwargs)
    second = hub_stability.measure_draw(_draw(), **kwargs)
    assert first == second


def test_measure_draw_agrees_between_backends():
    kwargs = dict(k=10, k_hub=5, nlist=4, seed=42, chunk_rows=1024)
    sk = hub_stability.measure_draw(_draw(), backend="sklearn", **kwargs)
    torch_ = hub_stability.measure_draw(_draw(), backend="torch", **kwargs)
    for name in hub_stability.STATISTICS:
        assert sk[name] == pytest.approx(torch_[name], rel=1e-4, abs=1e-6), name
