import numpy as np
import pytest
import torch

from src.eval.ann_difficulty import knn, survivor_mask
from src.train.log_ratio import batch_log_ratio_profile


def _blob(n=128, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def test_profile_has_one_entry_per_neighbour_below_k():
    profile = batch_log_ratio_profile(_blob(), k=10)
    assert profile.shape == (9,)


def test_profile_entries_are_non_positive_and_rise_toward_zero():
    # r_i <= r_k by construction, so every log-ratio is <= 0, and the ratio
    # approaches 1 as i approaches k.
    profile = batch_log_ratio_profile(_blob(n=512), k=10)
    assert (profile <= 1e-6).all()
    assert (profile.diff() >= -1e-6).all()
    assert profile[-1].abs() < profile[0].abs()


def test_profile_matches_the_numpy_reference():
    # The eval module is the reference implementation. If these drift, LID
    # measured after training stops corresponding to what training optimised.
    x = _blob(n=256, d=6, seed=3)
    profile = batch_log_ratio_profile(x, k=12, max_points=0)

    dist, _, _ = knn(x.numpy().astype(np.float32), k=12)
    kept = dist[survivor_mask(dist)]
    reference = np.log(
        np.clip(kept[:, :-1] / kept[:, -1:], 1e-12, 1.0)
    ).mean(axis=0)

    assert np.allclose(profile.numpy(), reference, atol=1e-4)


def test_queries_on_a_duplicate_are_dropped_not_clamped():
    # r_1 == 0 makes log(r_1/r_k) = -inf. Dropping declines to answer;
    # clamping would invent a number.
    base = _blob(n=64, d=4, seed=5)
    x = torch.cat([base, base[:8]])
    profile = batch_log_ratio_profile(x, k=6)
    assert profile is not None
    assert torch.isfinite(profile).all()


def test_all_duplicate_batch_returns_none():
    x = torch.ones(32, 4)
    assert batch_log_ratio_profile(x, k=5) is None


def test_all_tied_batch_returns_none():
    # Rows of the identity are all sqrt(2) apart, so r_1 == r_k for every query.
    assert batch_log_ratio_profile(torch.eye(16), k=5) is None


def test_k_is_clamped_to_the_batch():
    profile = batch_log_ratio_profile(_blob(n=6, d=3), k=100)
    assert profile.shape == (4,)  # k_eff = n - 1 = 5, profile is k_eff - 1


def test_batch_too_small_returns_none():
    assert batch_log_ratio_profile(_blob(n=2, d=3), k=5) is None


def test_max_points_subsamples_without_changing_the_shape():
    torch.manual_seed(0)
    profile = batch_log_ratio_profile(_blob(n=512), k=10, max_points=64)
    assert profile.shape == (9,)


def test_gradient_flows_to_the_input():
    x = _blob(n=64, d=4, seed=7).requires_grad_(True)
    batch_log_ratio_profile(x, k=8).sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_gradient_is_finite_when_duplicates_are_present():
    # The trap: torch.cdist has an undefined gradient at distance zero, so a
    # collapsed generator would poison the whole backward pass. The expanded
    # -square form with a clamped sqrt keeps it finite.
    base = _blob(n=32, d=4, seed=11)
    x = torch.cat([base, base[:4]]).requires_grad_(True)
    batch_log_ratio_profile(x, k=6).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_profile_dtype_follows_the_input():
    profile = batch_log_ratio_profile(_blob().to(torch.float64), k=8)
    assert profile.dtype == torch.float64
