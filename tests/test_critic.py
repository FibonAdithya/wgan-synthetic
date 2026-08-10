"""Direct tests for the WGAN-GP critic.

`tests/test_train_smoke.py` only exercises the critic transitively through a
training step, which cannot distinguish "the critic is wired correctly" from
"the training loop happens not to notice". These tests pin the shape contract,
the architecture the constructor arguments are supposed to produce, and the
double-backward path the gradient penalty depends on.
"""

import numpy as np
import pytest
import torch
from torch import nn

from src.models.critic import Critic


def _batch(rng: np.random.Generator, n: int, dim: int) -> torch.Tensor:
    return torch.from_numpy(rng.normal(size=(n, dim)).astype(np.float32))


def test_critic_emits_exactly_one_score_per_input_vector():
    critic = Critic(input_dim=5, hidden_dims=[8, 4])
    x = _batch(np.random.default_rng(0), n=7, dim=5)

    scores = critic(x)

    assert scores.shape == (7,), (
        f"critic must return a 1-D score per row, got shape {tuple(scores.shape)}"
    )


def test_critic_scores_each_row_independently_of_the_rest_of_the_batch():
    """A per-sample score must not depend on batch composition.

    The WGAN loss is a mean over the batch, so a critic that mixed rows would
    still train to a plausible-looking number while scoring individual vectors
    meaninglessly.
    """
    critic = Critic(input_dim=4, hidden_dims=[6])
    x = _batch(np.random.default_rng(1), n=5, dim=4)

    batched = critic(x)
    one_at_a_time = torch.cat([critic(row.unsqueeze(0)) for row in x])

    torch.testing.assert_close(batched, one_at_a_time)


def test_critic_builds_one_linear_per_hidden_dim_plus_a_scalar_head():
    critic = Critic(input_dim=16, hidden_dims=[12, 8, 4])

    linears = [m for m in critic.net if isinstance(m, nn.Linear)]
    shapes = [(m.in_features, m.out_features) for m in linears]

    assert shapes == [(16, 12), (12, 8), (8, 4), (4, 1)], (
        f"hidden_dims must define the interior widths, got {shapes}"
    )


def test_critic_with_no_hidden_dims_is_a_single_linear_projection_to_a_scalar():
    critic = Critic(input_dim=6, hidden_dims=[])

    modules = list(critic.net)

    assert len(modules) == 1, f"expected a bare projection, got {modules}"
    assert isinstance(modules[0], nn.Linear)
    assert (modules[0].in_features, modules[0].out_features) == (6, 1)
    assert critic(_batch(np.random.default_rng(2), n=3, dim=6)).shape == (3,)


def test_critic_accepts_hidden_dims_as_any_iterable_not_only_a_list():
    """`hidden_dims` is typed `Iterable[int]`, so a one-shot generator must work."""
    critic = Critic(input_dim=4, hidden_dims=(d for d in (7, 3)))

    shapes = [
        (m.in_features, m.out_features) for m in critic.net if isinstance(m, nn.Linear)
    ]

    assert shapes == [(4, 7), (7, 3), (3, 1)]


def test_critic_puts_a_leaky_relu_after_every_hidden_linear_but_not_the_head():
    critic = Critic(input_dim=4, hidden_dims=[5, 5])

    kinds = [type(m) for m in critic.net]

    assert kinds == [nn.Linear, nn.LeakyReLU, nn.Linear, nn.LeakyReLU, nn.Linear], (
        f"the scalar head must stay unactivated, got {kinds}"
    )


def test_critic_applies_the_configured_negative_slope_to_negative_activations():
    """Recompute the forward pass by hand to prove the slope is the one asked for."""
    slope = 0.35
    critic = Critic(input_dim=3, hidden_dims=[4], negative_slope=slope)
    x = _batch(np.random.default_rng(3), n=6, dim=3)

    first, _, head = critic.net
    with torch.no_grad():
        hidden = first(x)
        expected = head(torch.where(hidden >= 0, hidden, slope * hidden)).squeeze(-1)
        actual = critic(x)

    assert (hidden < 0).any(), "test input must reach the negative branch to be useful"
    torch.testing.assert_close(actual, expected)


def test_critic_negative_slope_defaults_to_zero_point_two():
    critic = Critic(input_dim=3, hidden_dims=[4])

    slopes = [m.negative_slope for m in critic.net if isinstance(m, nn.LeakyReLU)]

    assert slopes == [0.2]


def test_critic_negative_slope_changes_the_scores_it_produces():
    x = _batch(np.random.default_rng(4), n=8, dim=3)
    shallow = Critic(input_dim=3, hidden_dims=[5], negative_slope=0.01)
    steep = Critic(input_dim=3, hidden_dims=[5], negative_slope=0.9)
    steep.load_state_dict(shallow.state_dict())

    with torch.no_grad():
        assert not torch.allclose(shallow(x), steep(x)), (
            "identical weights under different slopes must not score identically"
        )


def test_critic_is_differentiable_with_respect_to_its_input():
    """The gradient penalty differentiates the score w.r.t. the interpolated input."""
    critic = Critic(input_dim=4, hidden_dims=[8, 8])
    x = _batch(np.random.default_rng(5), n=5, dim=4).requires_grad_(True)

    (grad,) = torch.autograd.grad(critic(x).sum(), x)

    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0.0, "critic gradient w.r.t. its input must not vanish"


def test_critic_supports_the_double_backward_the_gradient_penalty_needs():
    """`create_graph=True` then `.backward()` is exactly the WGAN-GP update.

    The hidden LeakyReLUs are constructed with `inplace=True`, which is the kind
    of detail that silently breaks second-order autograd, so pin it here.
    """
    critic = Critic(input_dim=4, hidden_dims=[8])
    x = _batch(np.random.default_rng(6), n=5, dim=4).requires_grad_(True)

    (grad,) = torch.autograd.grad(critic(x).sum(), x, create_graph=True)
    penalty = ((grad.norm(dim=1) - 1.0) ** 2).mean()
    penalty.backward()

    weight_grads = [p.grad for p in critic.parameters() if p.grad is not None]
    assert weight_grads, "gradient penalty must reach the critic parameters"
    assert any(g.abs().sum() > 0.0 for g in weight_grads)


def test_critic_forward_does_not_mutate_the_tensor_it_was_given():
    """`inplace=True` must only touch the critic's own intermediates."""
    critic = Critic(input_dim=4, hidden_dims=[6])
    x = _batch(np.random.default_rng(7), n=5, dim=4)
    before = x.clone()

    critic(x)

    torch.testing.assert_close(x, before)


def test_critic_rejects_inputs_whose_width_is_not_the_configured_input_dim():
    critic = Critic(input_dim=4, hidden_dims=[6])
    wrong = _batch(np.random.default_rng(8), n=3, dim=5)

    with pytest.raises(RuntimeError):
        critic(wrong)
