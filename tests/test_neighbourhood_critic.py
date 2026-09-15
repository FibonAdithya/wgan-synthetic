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
    NeighbourhoodCritic,
    neighbourhood_distances,
    profile_features,
)
from src.train.train_wgan_gp import gradient_penalty


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


def test_critic_features_and_scores_are_float32_safe_under_autocast():
    """Catches the neighbour maths running under an enclosing autocast (the
    bf16 cross term is promoted back to float32, so dtype alone cannot see
    it) and a forward that rounds x before profiling: both move the features
    by ~1e-3 against a float64 oracle; the clean path is within 1e-6."""
    torch.manual_seed(0)
    critic = NeighbourhoodCritic(input_dim=8, hidden_dims=[6], k=3)
    x = _unit_batch(8, n=16, dim=8)
    # Independent float64 oracle: cdist is fine here, it is the test, not the implementation.
    full = torch.cdist(x.double(), x.double())
    full.fill_diagonal_(float("inf"))
    r64, _ = torch.topk(full, 3, dim=1, largest=False, sorted=True)
    oracle = profile_features(r64.clamp(min=critic.distance_floor)).float()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        assert neighbourhood_distances(x, k=3, floor=0.01).dtype == torch.float32
        phi = critic.features(x)
        with torch.no_grad():
            scored = critic(x)
    assert phi.dtype == torch.float32
    torch.testing.assert_close(phi, oracle, atol=1e-5, rtol=0.0)
    assert torch.isfinite(scored).all()


def test_critic_emits_one_score_per_row():
    """Catches a wrong squeeze or reduction axis."""
    critic = NeighbourhoodCritic(input_dim=5, hidden_dims=[8, 4], k=3)
    assert critic(_unit_batch(10, n=7, dim=5)).shape == (7,)


def test_critic_input_width_is_dim_plus_k():
    """Catches the profile computed but not concatenated."""
    critic = NeighbourhoodCritic(input_dim=5, hidden_dims=[8], k=3)
    first = next(m for m in critic.mlp.net if isinstance(m, torch.nn.Linear))
    assert first.in_features == 5 + 3


def test_features_have_width_k():
    """Catches a features() that returns the distances' shape with an extra
    column."""
    critic = NeighbourhoodCritic(input_dim=5, hidden_dims=[8], k=4)
    assert critic.features(_unit_batch(11, n=9, dim=5)).shape == (9, 4)


def test_scores_depend_on_the_rest_of_the_batch():
    """The mirror image of test_critic.py's independence test. Catches the
    class silently degenerating to per-vector (features not wired).

    k = n - 1 so every row's profile uses every other row; moving one row
    then changes every other row's r_k or a ratio."""
    torch.manual_seed(0)
    critic = NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=7)
    x = _unit_batch(12, n=8, dim=4)
    base = critic(x)
    moved = x.clone()
    moved[7] = moved[7] + 1.0
    assert (critic(moved)[:7] - base[:7]).abs().max() > 1e-6


def test_permuting_the_batch_permutes_the_scores():
    """Catches a topk/gather index bug pairing a row with another's profile."""
    torch.manual_seed(0)
    critic = NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=3)
    x = _unit_batch(13, n=10, dim=4)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(critic(x[perm]), critic(x)[perm], atol=1e-5, rtol=1e-5)


def test_gradient_penalty_is_finite_and_double_backward_works():
    """Catches a non-differentiable op in the profile path: the penalty
    needs d(grad norm)/d(params), i.e. create_graph=True through topk,
    clamp, sqrt and log."""
    torch.manual_seed(0)
    critic = NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=3)
    real = _unit_batch(14, n=8, dim=4)
    fake = _unit_batch(15, n=8, dim=4)
    gp = gradient_penalty(critic, real, fake, device=torch.device("cpu"))
    assert torch.isfinite(gp)
    gp.backward()
    # The head's bias never gets gradient from the penalty (it does not
    # affect d score / d input), on any critic. Check the first layer, which
    # the profile feeds, and that whatever grads exist are finite.
    first = next(m for m in critic.mlp.net if isinstance(m, torch.nn.Linear))
    assert first.weight.grad is not None and torch.isfinite(first.weight.grad).all()
    assert first.weight.grad[:, 4:].abs().sum() > 0  # the profile columns
    assert all(
        torch.isfinite(p.grad).all() for p in critic.parameters() if p.grad is not None
    )


def test_gradient_penalty_reaches_neighbour_rows():
    """With a batch-dependent critic the per-row gradient is of the summed
    batch score, so it includes how row i moves other rows' features. Catches
    a forward that detaches the profile."""
    torch.manual_seed(0)
    critic = NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=7)
    x = _unit_batch(16, n=8, dim=4).requires_grad_(True)
    critic(x)[0].backward()  # only row 0's score...
    # ...yet other rows get gradient, because they are row 0's neighbours.
    assert x.grad[1:].abs().sum() > 0


def test_rejects_k_below_two_and_nonpositive_floor():
    """k=1 leaves no ratio entries; the profile would be scale only."""
    with pytest.raises(ValueError, match="k"):
        NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=1)
    with pytest.raises(ValueError, match="floor"):
        NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=3, distance_floor=0.0)


def test_forward_refuses_a_batch_no_larger_than_k():
    """Catches silent truncation of k in forward."""
    critic = NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=5)
    with pytest.raises(ValueError, match="k"):
        critic(_unit_batch(17, n=5, dim=4))
