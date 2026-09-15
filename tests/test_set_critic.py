"""The learned set critic: one EdgeConv layer over within-batch neighbour
differences. Each test names the mutation it catches."""

import numpy as np
import pytest
import torch

from src.models.critic import SetNeighbourhoodCritic, neighbourhood_indices
from src.train.train_wgan_gp import gradient_penalty


def _unit(seed, n, dim):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return torch.from_numpy(x)


def test_indices_match_brute_force_and_exclude_self():
    """Catches a lost diagonal mask (self would be index 0 of every row).
    torch.cdist is the brute-force oracle here, as in
    test_neighbourhood_critic.py; the no-cdist rule is for src/."""
    x = _unit(0, n=9, dim=6)
    idx = neighbourhood_indices(x, k=3)
    full = torch.cdist(x, x)
    full.fill_diagonal_(float("inf"))
    _, expected = torch.topk(full, 3, dim=1, largest=False, sorted=True)
    assert idx.shape == (9, 3) and idx.dtype == torch.long
    assert torch.equal(idx, expected)
    assert (idx != torch.arange(9)[:, None]).all()


def test_indices_keep_an_exact_copy_as_a_neighbour():
    """Catches self-exclusion by dropping the nearest column, which would
    drop a copy in the row's place."""
    x = _unit(1, n=8, dim=5)
    x[3] = x[0]
    idx = neighbourhood_indices(x, k=2)
    assert idx[0, 0] == 3 and idx[3, 0] == 0


def test_indices_require_more_than_k_rows():
    with pytest.raises(ValueError, match="k"):
        neighbourhood_indices(_unit(2, n=4, dim=3), k=4)


def _set_critic(k=3, dim=4, edge_dim=8, pool="max"):
    torch.manual_seed(0)
    return SetNeighbourhoodCritic(
        input_dim=dim, hidden_dims=[6], k=k, edge_dim=edge_dim, edge_pool=pool
    )


def test_one_score_per_row():
    assert _set_critic()(_unit(3, 7, 4)).shape == (7,)


def test_layer_widths():
    """Catches the edge MLP fed x_j alone (width D) or the head not
    receiving the pooled features."""
    c = _set_critic(dim=4, edge_dim=8)
    first_edge = next(m for m in c.edge if isinstance(m, torch.nn.Linear))
    first_head = next(m for m in c.mlp.net if isinstance(m, torch.nn.Linear))
    assert first_edge.in_features == 2 * 4
    assert first_head.in_features == 4 + 8


def test_scores_depend_on_the_rest_of_the_batch():
    """Mirror of test_critic.py's independence test."""
    c = _set_critic(k=7)
    x = _unit(4, 8, 4)
    base = c(x)
    moved = x.clone()
    moved[7] = moved[7] + 1.0
    assert (c(moved)[:7] - base[:7]).abs().max() > 1e-6


def test_permuting_the_batch_permutes_the_scores():
    c = _set_critic(k=3)
    x = _unit(5, 10, 4)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(c(x[perm]), c(x)[perm], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("pool", ["max", "mean"])
def test_pooling_is_invariant_to_neighbour_order(pool):
    """Catches a flatten or a positional treatment of the k neighbours."""
    c = _set_critic(k=4, pool=pool)
    x = _unit(6, 9, 4)
    idx = neighbourhood_indices(x, k=4)
    shuffled = idx[:, torch.tensor([2, 0, 3, 1])]
    torch.testing.assert_close(c.pooled(x, idx), c.pooled(x, shuffled))


def test_edge_features_use_the_difference_not_the_absolute_neighbour():
    """With the x_i columns of BOTH first layers zeroed, the only input left
    is x_j - x_i, which a translation cannot change. Catches edge features
    built from absolute x_j."""
    c = _set_critic(k=3, dim=4)
    first_edge = next(m for m in c.edge if isinstance(m, torch.nn.Linear))
    first_head = next(m for m in c.mlp.net if isinstance(m, torch.nn.Linear))
    with torch.no_grad():
        first_edge.weight[:, :4].zero_()
        first_head.weight[:, :4].zero_()
    x = _unit(7, 9, 4)
    shift = torch.full((1, 4), 0.7)
    torch.testing.assert_close(c(x + shift), c(x), atol=1e-5, rtol=1e-5)


def test_max_and_mean_pools_both_build_and_differ():
    """Catches the pool key ignored."""
    x = _unit(8, 9, 4)
    a = _set_critic(pool="max")
    b = _set_critic(pool="mean")
    b.load_state_dict(a.state_dict())
    assert not torch.allclose(a(x), b(x))


def test_unknown_pool_raises():
    with pytest.raises(ValueError, match="edge_pool"):
        SetNeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=3, edge_pool="sum")


def test_gradient_penalty_is_finite_and_double_backward_works():
    torch.manual_seed(0)
    c = _set_critic(k=3)
    gp = gradient_penalty(
        c, _unit(9, 8, 4), _unit(10, 8, 4), device=torch.device("cpu")
    )
    assert torch.isfinite(gp)
    gp.backward()
    # The head's bias never gets gradient from the penalty (it does not
    # affect d score / d input), on any critic. The edge MLP's first weight
    # must, since the penalty reaches neighbours through it.
    first_edge = next(m for m in c.edge if isinstance(m, torch.nn.Linear))
    assert (
        first_edge.weight.grad is not None
        and torch.isfinite(first_edge.weight.grad).all()
    )
    assert all(
        torch.isfinite(p.grad).all() for p in c.parameters() if p.grad is not None
    )


def test_gradient_reaches_neighbour_rows_through_the_differences():
    """The summed-score penalty is doing real work here: row 0's score must
    have gradient with respect to its neighbours' coordinates."""
    c = _set_critic(k=7)
    x = _unit(11, 8, 4).requires_grad_(True)
    c(x)[0].backward()
    assert x.grad[1:].abs().sum() > 0


def test_exact_copies_give_finite_scores_and_gradients():
    """A copy gives a zero difference vector; nothing here divides by it."""
    x = _unit(12, 8, 4)
    x[1] = x[0]
    x.requires_grad_(True)
    s = _set_critic(k=3)(x)
    assert torch.isfinite(s).all()
    s.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_forward_refuses_a_batch_no_larger_than_k():
    with pytest.raises(ValueError, match="k"):
        _set_critic(k=5)(_unit(13, 5, 4))
