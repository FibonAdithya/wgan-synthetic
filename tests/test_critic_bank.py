"""Tests for the bank-neighbourhood critic (NYTimes v2b).

Kept apart from `tests/test_critic.py` and `tests/test_neighbourhood_critic.py`
so the v2b branch does not edit files the v2 branch owns. Every test names
the mutation it catches; the plan records the mutation checks run after
implementation.
"""

import math

import numpy as np
import pytest
import torch

from src.models.critic import (
    BankNeighbourhoodCritic,
    Critic,
    NeighbourhoodCritic,
    build_critic,
    draw_real_bank_indices,
    neighbourhood_distances,
    profile_features,
    score_population,
)
from src.train.train_wgan_gp import gradient_penalty


def _rows(seed: int, n: int, dim: int) -> torch.Tensor:
    """`n` random unit vectors, the geometry every NYTimes batch has."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return torch.from_numpy(x)


def _brute_force(x, bank, k, floor, self_index=None):
    d = torch.cdist(x.double(), bank.double())
    if self_index is not None:
        for i, slot in enumerate(self_index.tolist()):
            if slot >= 0:
                d[i, slot] = math.inf
    r, _ = torch.topk(d, k, dim=1, largest=False, sorted=True)
    return r.clamp(min=floor).float()


# --- the landed distance function, one case its own tests do not pin ---------


def test_self_index_masks_row_then_slot_not_the_transpose():
    """Catches: the mask applied at `[slot, row]` instead of `[row, slot]`.
    The landed tests use a bank equal to the batch with `arange`, where the
    two are the same cells. Here the bank is the batch rolled by one, so
    row i's copy sits at slot (i + 1) % 3, never at slot i."""
    x = _rows(3, 3, 6)
    bank = torch.roll(x, shifts=1, dims=0)
    self_index = torch.tensor([1, -1, 0])

    r = neighbourhood_distances(x, 2, 0.01, bank=bank, self_index=self_index)

    assert r[0, 0] > 0.01, "row 0's copy at slot 1 must be masked"
    assert r[2, 0] > 0.01, "row 2's copy at slot 0 must be masked"
    assert r[1, 0] == pytest.approx(0.01), "row 1 is not excluded; it reads its copy"
    torch.testing.assert_close(
        r, _brute_force(x, bank, 2, 0.01, self_index), atol=1e-5, rtol=1e-4
    )


# --- draw_real_bank_indices --------------------------------------------------


def test_bank_draw_is_deterministic_in_the_seed_and_distinct_across_seeds():
    """Catches: the draw reading the global RNG (which the trainer has already
    consumed by an amount that depends on everything built before the critic)."""
    a = draw_real_bank_indices(100, 10, seed=3)
    b = draw_real_bank_indices(100, 10, seed=3)
    c = draw_real_bank_indices(100, 10, seed=4)

    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.dtype == torch.long and a.shape == (10,)
    assert len(set(a.tolist())) == 10, "bank rows must be distinct"
    assert a.min() >= 0 and a.max() < 100


def test_bank_draw_rejects_a_bank_larger_than_the_split():
    with pytest.raises(ValueError, match="critic_bank_size"):
        draw_real_bank_indices(10, 11, seed=0)


# --- BankNeighbourhoodCritic -------------------------------------------------


def _bank_critic(seed=0, k=3, bank_size=20, hidden=(8, 4)):
    torch.manual_seed(seed)
    return BankNeighbourhoodCritic(
        input_dim=6,
        hidden_dims=list(hidden),
        k=k,
        distance_floor=0.01,
        bank_size=bank_size,
    )


def _filled(critic, seed=10):
    critic.write_fake(_rows(seed, critic.bank_size, 6))
    assert critic.fake_bank_full
    return critic


def test_bank_critic_rejects_a_bank_no_larger_than_k():
    with pytest.raises(ValueError, match="critic_bank_size"):
        BankNeighbourhoodCritic(input_dim=6, hidden_dims=[8], k=5, bank_size=5)


def test_bank_critic_scores_one_per_row_in_every_population():
    """Catches: a wrong squeeze or reduction axis on any of the three paths."""
    critic = _filled(_bank_critic())
    critic.set_real_bank(_rows(1, 40, 6), draw_real_bank_indices(40, 20, seed=0))
    x = _rows(2, 7, 6)

    for population in ("real", "fake", "mixed"):
        assert critic(x, population=population).shape == (7,), population
    assert critic(x).shape == (7,), "the bare call is the mixed population"


def test_bank_critic_rejects_an_unknown_population_and_an_unset_real_bank():
    """Catches: the `mixed` path silently profiling against an all-zero real
    bank once the ring is full (it needs no real bank only while it falls
    back to within-batch)."""
    critic = _bank_critic()
    x = _rows(3, 5, 6)

    with pytest.raises(ValueError, match="population"):
        critic(x, population="interpolated")
    with pytest.raises(RuntimeError, match="set_real_bank"):
        critic(x, population="real")
    assert critic(x, population="mixed").shape == (5,), (
        "within-batch fallback needs no bank"
    )

    _filled(critic)
    with pytest.raises(RuntimeError, match="set_real_bank"):
        critic(x, population="mixed")


def test_real_population_reads_the_real_bank():
    """Catches: the real path reading the fake bank, the union, or the batch.
    The expected score is recomputed by hand from the real bank alone."""
    critic = _filled(_bank_critic())
    x_train = _rows(4, 40, 6)
    critic.set_real_bank(x_train, torch.arange(20))
    x = _rows(5, 5, 6)
    r = neighbourhood_distances(x, 3, 0.01, bank=x_train[:20])
    expected = critic.mlp(torch.cat([x, profile_features(r)], dim=1))

    torch.testing.assert_close(critic(x, population="real"), expected)


def test_mixed_population_reads_the_union_of_both_banks():
    """Catches: the mixed path reading one bank instead of the union.
    The expected score is recomputed by hand from the real and fake banks
    concatenated; the union must also differ from either bank scored alone."""
    critic = _filled(_bank_critic())
    critic.set_real_bank(_rows(4, 40, 6), draw_real_bank_indices(40, 20, seed=0))
    x = _rows(5, 5, 6)
    r = neighbourhood_distances(
        x, 3, 0.01, bank=torch.cat([critic.real_bank, critic.fake_bank], dim=0)
    )
    expected = critic.mlp(torch.cat([x, profile_features(r)], dim=1))

    torch.testing.assert_close(critic(x, population="mixed"), expected)
    assert not torch.allclose(
        critic(x, population="mixed"), critic(x, population="fake")
    ), "the union must differ from the fake bank alone"
    assert not torch.allclose(
        critic(x, population="mixed"), critic(x, population="real")
    ), "the union must differ from the real bank alone"


def test_a_real_row_in_the_bank_does_not_see_its_own_copy():
    """Catches: `row_to_slot` built as the identity instead of through the
    drawn indices (the bank is a permutation, so slot != row), or `row_ids`
    ignored. Spec: 'with the bank equal to the batch, the smallest distance
    read is above the floor'."""
    x_train = _rows(6, 30, 6)
    critic = _bank_critic(bank_size=30)
    critic.set_real_bank(x_train, draw_real_bank_indices(30, 30, seed=1))
    batch, ids = x_train[:5], torch.arange(5)

    self_index = critic.row_to_slot[ids]
    r = neighbourhood_distances(
        batch, 3, 0.01, bank=critic.real_bank, self_index=self_index
    )
    assert r.min() > 0.1, "with its own slot masked no row may read the floor"
    assert torch.all(critic.real_bank[self_index] == batch), (
        "slot must point at the copy"
    )

    with_ids = critic(batch, population="real", row_ids=ids)
    without = critic(batch, population="real")
    assert not torch.allclose(with_ids, without), (
        "row_ids must change what a bank row sees"
    )


def test_before_the_fake_bank_fills_fake_and_mixed_scores_equal_the_within_batch_critic():
    """Spec's fallback test. Catches: the fallback wired to a bank path, or a
    feature layout that differs from approach 1's."""
    torch.manual_seed(0)
    plain = NeighbourhoodCritic(
        input_dim=6, hidden_dims=[8, 4], k=3, distance_floor=0.01
    )
    bank = _bank_critic(seed=1, bank_size=16)
    bank.load_state_dict(plain.state_dict(), strict=False)
    x = _rows(7, 6, 6)

    torch.testing.assert_close(bank(x, population="fake"), plain(x))
    torch.testing.assert_close(bank(x, population="mixed"), plain(x))

    _filled(bank)
    assert not torch.allclose(bank(x, population="fake"), plain(x)), (
        "once full, fake rows must be profiled against the bank, not the batch"
    )


def test_bank_mode_scores_do_not_depend_on_batch_mates_but_fallback_scores_do():
    """Mirror of the spec's dependence test. In bank mode each row's profile
    comes from the bank alone, so replacing row 0 must leave every other score
    unchanged; catches the bank path concatenating the batch into the bank.
    Before the fill, the within-batch fallback must show the dependence."""
    critic = _bank_critic(bank_size=16)
    x = _rows(8, 6, 6)
    y = x.clone()
    y[0] = _rows(9, 1, 6)[0]

    assert not torch.allclose(
        critic(x, population="fake")[1:], critic(y, population="fake")[1:]
    )

    _filled(critic)
    torch.testing.assert_close(
        critic(x, population="fake")[1:], critic(y, population="fake")[1:]
    )


def test_fake_bank_is_a_ring_that_overwrites_the_oldest_rows():
    """Spec's ring test. Catches: a pointer that never wraps, or writes that
    append past the end."""
    critic = _bank_critic(bank_size=8)
    marker = torch.full((4, 6), 7.0)

    critic.write_fake(marker)
    critic.write_fake(_rows(10, 4, 6))
    assert critic.fake_bank_full
    assert torch.equal(critic.fake_bank[:4], marker)

    critic.write_fake(_rows(11, 4, 6))
    assert not (critic.fake_bank == 7.0).any(), "the marker rows must be gone"
    assert critic.fake_rows_written == 12


def test_fake_bank_write_wraps_a_batch_across_the_end():
    """Catches: an off-by-one at the wrap (slot 8 written, slot 0 skipped)."""
    critic = _bank_critic(bank_size=8)
    critic.write_fake(_rows(12, 6, 6))
    tail = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    critic.write_fake(tail)

    assert torch.equal(critic.fake_bank[6:8], tail[:2])
    assert torch.equal(critic.fake_bank[0:2], tail[2:])
    assert critic.fake_rows_written == 10


def test_fake_bank_write_rejects_a_batch_larger_than_the_bank():
    with pytest.raises(ValueError, match="bank"):
        _bank_critic(bank_size=8).write_fake(_rows(13, 9, 6))


def test_fake_bank_write_detaches_and_does_not_hold_the_graph():
    """Catches: dropping both the `@torch.no_grad()` decorator and the
    `.detach()` (either alone suffices, so removing only one is not
    observable here), which would attach the ring to the generator's graph
    and keep every past generator step alive on the device."""
    critic = _bank_critic(bank_size=8)
    rows = _rows(14, 4, 6).requires_grad_(True)

    critic.write_fake(rows * 2.0)

    assert not critic.fake_bank.requires_grad


def test_real_bank_indices_survive_a_state_dict_round_trip_and_the_rows_do_not():
    """Spec's checkpoint test. Catches: indices registered non-persistent (a
    resume would redraw them), or the 16 MB banks leaking into every checkpoint."""
    x_train = _rows(15, 40, 6)
    a = _bank_critic(bank_size=10)
    a.set_real_bank(x_train, draw_real_bank_indices(40, 10, seed=3))
    b = _bank_critic(seed=5, bank_size=10)

    b.load_state_dict(a.state_dict())

    assert torch.equal(b.real_bank_indices, a.real_bank_indices)
    assert "real_bank" not in a.state_dict()
    assert "fake_bank" not in a.state_dict()


def test_set_real_bank_rejects_indices_of_the_wrong_shape():
    with pytest.raises(ValueError, match="bank_size"):
        _bank_critic(bank_size=10).set_real_bank(_rows(16, 40, 6), torch.arange(9))


def test_gradient_penalty_is_finite_through_the_union_bank_and_reaches_the_weights():
    """Spec's double-backward test for this class. Catches: a non-differentiable
    op on the mixed path, or `torch.cat` of the banks breaking `create_graph`."""
    critic = _filled(_bank_critic())
    critic.set_real_bank(_rows(17, 40, 6), draw_real_bank_indices(40, 20, seed=0))
    real, fake = _rows(18, 6, 6), _rows(19, 6, 6)

    gp = gradient_penalty(critic, real, fake, device=torch.device("cpu"))
    gp.backward()

    assert torch.isfinite(gp)
    grads = [p.grad for p in critic.mlp.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# --- score_population and build_critic --------------------------------------


def test_score_population_routes_only_the_bank_critic():
    """Catches: the dispatcher passing keywords a plain critic cannot take, or
    dropping them for the bank critic."""
    x = _rows(20, 5, 6)
    plain = Critic(input_dim=6, hidden_dims=[8])
    torch.testing.assert_close(score_population(plain, x, "real"), plain(x))

    bank = _filled(_bank_critic())
    bank.set_real_bank(_rows(21, 40, 6), draw_real_bank_indices(40, 20, seed=0))
    torch.testing.assert_close(
        score_population(bank, x, "fake"), bank(x, population="fake")
    )
    assert not torch.allclose(
        score_population(bank, x, "fake"), bank(x, population="real")
    )


def test_build_critic_dispatches_neighbourhood_bank_with_its_size():
    """Catches: the factory branch missing (a config typo trains the old
    critic), or `critic_bank_size` not read."""
    cfg = {
        "critic_type": "neighbourhood_bank",
        "critic_hidden_dims": [8],
        "negative_slope": 0.2,
        "critic_k": 3,
        "critic_distance_floor": 0.02,
        "critic_bank_size": 64,
    }

    critic = build_critic(cfg, input_dim=6)

    assert isinstance(critic, BankNeighbourhoodCritic)
    assert critic.bank_size == 64 and critic.k == 3 and critic.distance_floor == 0.02
    assert critic.real_bank.shape == (64, 6)


def test_build_critic_bank_size_defaults_to_16384():
    cfg = {
        "critic_type": "neighbourhood_bank",
        "critic_hidden_dims": [8],
        "negative_slope": 0.2,
    }

    assert build_critic(cfg, input_dim=6).bank_size == 16384
