"""The within-batch neighbourhood features and the critic that consumes them.

Every test names the mutation it catches in its docstring. The per-vector
`Critic` is untouched and keeps its own tests in test_critic.py.
"""

import math

import numpy as np
import pytest
import torch

from src.models.critic import (
    DEFAULT_DISTANCE_FLOOR,
    DEFAULT_K,
    neighbourhood_distances,
    profile_features,
)


def _unit_batch(seed: int, n: int, dim: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return torch.from_numpy(x)


def test_defaults_are_the_spec_values():
    """Catches a changed default that would silently alter every neighbourhood
    config."""
    assert DEFAULT_K == 20
    assert DEFAULT_DISTANCE_FLOOR == 0.01


def test_distances_have_shape_n_by_k_and_are_sorted_ascending():
    """Catches a wrong topk axis or an unsorted topk."""
    x = _unit_batch(0, n=12, dim=5)
    r = neighbourhood_distances(x, k=4, floor=0.01)
    assert r.shape == (12, 4)
    assert (r[:, 1:] >= r[:, :-1]).all()


def test_distances_match_brute_force_on_a_batch_without_ties():
    """Catches an expanded-square sign error or a lost self-mask column."""
    x = _unit_batch(1, n=9, dim=6)
    r = neighbourhood_distances(x, k=3, floor=1e-6)
    full = torch.cdist(x, x)
    full.fill_diagonal_(float("inf"))
    expected, _ = torch.topk(full, 3, dim=1, largest=False, sorted=True)
    torch.testing.assert_close(r, expected, atol=1e-5, rtol=1e-5)


def test_self_is_excluded_by_index():
    """Catches a missing diagonal mask: every row's own distance is 0, which
    would read as the floor."""
    x = 2.0 * torch.eye(6)  # pairwise distance 2*sqrt(2), well above 0.5
    r = neighbourhood_distances(x, k=2, floor=0.01)
    assert r.min() > 0.5


def _batch_with_an_exact_copy(seed: int, n: int, dim: int) -> torch.Tensor:
    """Rows 0 and 1 are the same one-hot vector. One-hot, not a random unit
    row: the expanded-square distance of a random copy rounds to ~1e-7, not
    0, so a test built on it would pass even without the clamp. For a
    one-hot pair sq + sq - 2*dot is exactly 1 + 1 - 2 = 0."""
    x = _unit_batch(seed, n, dim)
    x[0] = 0.0
    x[0, 0] = 1.0
    x[1] = x[0]
    return x


def test_exact_copies_read_as_the_floor_not_zero_or_nan():
    """Catches a lost clamp (would be 0, then log gives -inf) and a clamp
    applied after sqrt (gradient at sqrt(0) is infinite)."""
    x = _batch_with_an_exact_copy(2, n=8, dim=5)
    r = neighbourhood_distances(x, k=2, floor=0.01)
    assert r[0, 0].item() == pytest.approx(0.01, abs=1e-8)
    assert r[1, 0].item() == pytest.approx(0.01, abs=1e-8)
    assert torch.isfinite(profile_features(r)).all()


def test_floor_gradient_is_finite_through_a_copy():
    """Catches sqrt-before-clamp: d sqrt(0)/dx is inf and poisons the batch."""
    x = _batch_with_an_exact_copy(3, n=8, dim=5).requires_grad_(True)
    r = neighbourhood_distances(x, k=2, floor=0.01)
    r.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_floor_does_not_touch_distances_above_it():
    """Catches a clamp on the wrong side (min vs max)."""
    x = _unit_batch(4, n=10, dim=8)
    r_low = neighbourhood_distances(x, k=3, floor=1e-6)
    r_high = neighbourhood_distances(x, k=3, floor=0.01)
    torch.testing.assert_close(r_low, r_high)


def test_within_batch_requires_more_than_k_rows():
    """Catches silent truncation of k, which would change the feature width."""
    x = _unit_batch(5, n=4, dim=3)
    with pytest.raises(ValueError, match="k"):
        neighbourhood_distances(x, k=4, floor=0.01)


def test_bank_neighbours_come_from_the_bank_not_the_batch():
    """Catches a bank argument that is accepted and ignored."""
    x = _unit_batch(6, n=3, dim=4)
    bank = _unit_batch(7, n=20, dim=4)
    r = neighbourhood_distances(x, k=5, floor=1e-6, bank=bank)
    expected, _ = torch.topk(torch.cdist(x, bank), 5, dim=1, largest=False, sorted=True)
    assert r.shape == (3, 5)
    torch.testing.assert_close(r, expected, atol=1e-5, rtol=1e-5)


def test_self_index_excludes_a_row_from_a_bank_that_contains_it():
    """Catches self_index accepted and ignored: with the bank equal to the
    batch, every row would see itself at distance 0."""
    x = 2.0 * torch.eye(6)
    r = neighbourhood_distances(x, k=2, floor=0.01, bank=x, self_index=torch.arange(6))
    assert r.min() > 0.5


def test_self_index_minus_one_means_not_in_bank():
    """Catches a mask applied at index -1 (which numpy/torch read as the last
    row) instead of being skipped."""
    x = 2.0 * torch.eye(6)
    bank = x.clone()
    self_index = torch.full((6,), -1, dtype=torch.long)
    r = neighbourhood_distances(x, k=1, floor=0.01, bank=bank, self_index=self_index)
    # No exclusion: each row finds its own copy in the bank at the floor.
    assert r.max().item() == pytest.approx(0.01, abs=1e-8)


def test_self_index_exclusions_count_toward_candidate_validation():
    """Catches a candidate count that ignores self_index: inf would leak into
    the profile."""
    x = torch.randn(4, 5)
    bank = torch.randn(3, 5)
    bank[0] = x[0]
    self_index = torch.tensor([0, -1, -1, -1])
    with pytest.raises(ValueError, match="k"):
        neighbourhood_distances(x, k=3, floor=0.01, bank=bank, self_index=self_index)


def test_profile_is_log_ratios_then_log_scale():
    """Catches a wrong divisor (r_1 instead of r_k) or a dropped scale entry."""
    r = torch.tensor([[1.0, 2.0, 4.0]])
    phi = profile_features(r)
    expected = torch.tensor([[math.log(0.25), math.log(0.5), math.log(4.0)]])
    torch.testing.assert_close(phi, expected)
    assert phi.shape == (1, 3)


def test_profile_runs_in_float32_under_autocast():
    """Catches the profile inheriting fp16 from an enclosing autocast region,
    where the expanded-square form cancels catastrophically."""
    x = _unit_batch(8, n=16, dim=8)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        r = neighbourhood_distances(x, k=3, floor=0.01)
    assert r.dtype == torch.float32
