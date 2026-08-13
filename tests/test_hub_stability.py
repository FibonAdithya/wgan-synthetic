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
