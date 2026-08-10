import numpy as np
import torch

from src.eval.ann_difficulty import knn, survivor_mask
from src.train.log_ratio import (
    LogRatioTarget,
    batch_log_ratio_profile,
    log_ratio_penalty,
)


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
    reference = np.log(np.clip(kept[:, :-1] / kept[:, -1:], 1e-12, 1.0)).mean(axis=0)

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


def test_max_points_subsamples_and_clamps_k_eff():
    # Property 1: max_points actually changes the computation. Same seed, same
    # input, but subsampling picks a different subset of rows, so the profile
    # must differ from the unsubsampled one -- an implementation that ignored
    # max_points would produce the identical profile here and fail this
    # assertion.
    torch.manual_seed(0)
    full = batch_log_ratio_profile(_blob(n=512), k=10, max_points=0)
    torch.manual_seed(0)
    subsampled = batch_log_ratio_profile(_blob(n=512), k=10, max_points=64)
    assert subsampled.shape == (9,)
    assert not torch.allclose(subsampled, full)

    # Property 2: max_points clamps k_eff, not just n. With max_points=6 the
    # subsample has only 6 rows, so k_eff = min(k, 6 - 1) = 5 and the profile
    # has 4 entries -- an implementation that ignored max_points would instead
    # clamp against the original n=512 and produce shape (9,).
    clamped = batch_log_ratio_profile(_blob(n=512), k=10, max_points=6)
    assert clamped.shape == (4,)


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


def test_target_initialises_to_the_first_profile():
    target = LogRatioTarget(decay=0.9)
    first = torch.tensor([-1.0, -0.5])
    assert torch.equal(target.update(first), first)


def test_target_moves_toward_later_profiles():
    target = LogRatioTarget(decay=0.5)
    target.update(torch.tensor([0.0, 0.0]))
    updated = target.update(torch.tensor([-1.0, -1.0]))
    assert torch.allclose(updated, torch.tensor([-0.5, -0.5]))


def test_target_is_detached_from_the_graph():
    # The target is a fixed reference, not something the generator can move by
    # gradient. Keeping it attached would let the penalty be minimised by
    # dragging the target instead of the samples.
    target = LogRatioTarget()
    profile = torch.tensor([-1.0, -0.5], requires_grad=True)
    assert not target.update(profile).requires_grad


def test_target_resets_when_the_profile_length_changes():
    # k_eff shrinks on a short final batch; a stale target of the wrong length
    # would otherwise raise or broadcast silently.
    target = LogRatioTarget()
    target.update(torch.tensor([-1.0, -0.5, -0.2]))
    assert target.update(torch.tensor([-1.0, -0.5])).shape == (2,)


def test_penalty_is_near_zero_for_samples_from_one_distribution():
    torch.manual_seed(0)
    a, b = torch.randn(256, 8), torch.randn(256, 8)
    penalty = log_ratio_penalty(a, b, k=10, max_points=0, target=LogRatioTarget())
    assert penalty.item() < 0.05


def test_penalty_grows_when_local_structure_differs():
    # A 2-D blob and a 32-D blob have very different local geometry; the
    # penalty must see what mean pairwise distance alone cannot.
    torch.manual_seed(1)
    real = torch.randn(256, 32)
    same = log_ratio_penalty(
        torch.randn(256, 32), real, k=10, max_points=0, target=LogRatioTarget()
    )
    low_dim = torch.randn(256, 2)
    low_dim = torch.cat([low_dim, torch.zeros(256, 30)], dim=1)
    different = log_ratio_penalty(
        low_dim, real, k=10, max_points=0, target=LogRatioTarget()
    )
    assert different.item() > same.item() * 3.0


def test_penalty_is_zero_when_either_side_is_degenerate():
    torch.manual_seed(2)
    real = torch.randn(64, 8)
    collapsed = torch.ones(64, 8)
    penalty = log_ratio_penalty(
        collapsed, real, k=6, max_points=0, target=LogRatioTarget()
    )
    assert penalty.item() == 0.0
    assert torch.isfinite(penalty)


def test_penalty_gradient_reaches_the_fake_batch():
    torch.manual_seed(3)
    fake = torch.randn(128, 8, requires_grad=True)
    real = torch.randn(128, 8)
    log_ratio_penalty(fake, real, k=8, max_points=0, target=LogRatioTarget()).backward()
    assert fake.grad is not None
    assert fake.grad.abs().sum() > 0
