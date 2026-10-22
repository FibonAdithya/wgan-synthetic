import pytest
import torch

from src.models.conditional import BODY, TAIL
from src.models.critic import Critic, build_critic, score_population
from src.models.generator import Generator, build_generator

MLP_CFG = {
    "latent_dim": 8,
    "generator_hidden_dims": [16, 16],
    "critic_hidden_dims": [16, 16],
    "negative_slope": 0.2,
}


def test_flags_change_generator_output():
    """Catches the one-hot being dropped or zeroed: same z, different flag, different h."""
    g = build_generator({**MLP_CFG, "n_flags": 2}, output_dim=16)
    z = torch.randn(4, 8)
    a = g(z, torch.full((4,), BODY))
    b = g(z, torch.full((4,), TAIL))
    assert not torch.allclose(a, b)


def test_conditioned_generator_refuses_a_call_without_flags():
    """Catches a default flag that would let an unconditioned sampler read a v1 checkpoint."""
    g = build_generator({**MLP_CFG, "n_flags": 2}, output_dim=16)
    with pytest.raises(TypeError):
        g(torch.randn(4, 8))


def test_out_of_range_flag_raises():
    """Catches one_hot silently accepting flag 2 (would need n_flags 3)."""
    g = build_generator({**MLP_CFG, "n_flags": 2}, output_dim=16)
    with pytest.raises(RuntimeError):
        g(torch.randn(2, 8), torch.tensor([0, 2]))


@pytest.mark.parametrize(
    "bad", [{"n_flags": 3}, {"n_flags": 2, "generator_type": "spherical"}]
)
def test_build_generator_rejects_unsupported_flag_configs(bad):
    """Catches a builder that wraps a non-mlp generator or accepts n_flags != 2."""
    with pytest.raises(ValueError):
        build_generator({**MLP_CFG, **bad}, output_dim=16)


@pytest.mark.parametrize(
    "bad", [{"n_flags": 1}, {"n_flags": 2, "critic_type": "neighbourhood"}]
)
def test_build_critic_rejects_unsupported_flag_configs(bad):
    """Catches a builder that wraps a non-per_vector critic or accepts n_flags != 2."""
    with pytest.raises(ValueError):
        build_critic({**MLP_CFG, **bad}, 16)


def test_unflagged_builders_are_unchanged():
    """Catches the wrapper leaking into v0: no n_flags -> plain Generator/Critic, v0 state-dict keys."""
    assert type(build_generator(MLP_CFG, 16)) is Generator
    assert type(build_critic(MLP_CFG, 16)) is Critic
    assert "net.0.weight" in build_generator(MLP_CFG, 16).state_dict()
    assert "net.0.weight" in build_critic(MLP_CFG, 16).state_dict()


def test_score_population_requires_flags_for_a_flagged_critic():
    """Catches a flagged critic scored without flags (would crash in forward or skip the flag)."""
    c = build_critic({**MLP_CFG, "n_flags": 2}, 16)
    with pytest.raises(ValueError):
        score_population(c, torch.randn(4, 16), "real")


def test_score_population_passes_flags_to_a_flagged_critic():
    """Catches score_population dropping or ignoring flags for a flagged critic."""
    c = build_critic({**MLP_CFG, "n_flags": 2}, 16)
    x = torch.randn(4, 16)
    flags = torch.tensor([0, 1, 0, 1])
    assert torch.equal(score_population(c, x, "real", flags=flags), c(x, flags))


def test_score_population_rejects_flags_for_an_unflagged_critic():
    """Catches flags being silently ignored by a critic that cannot use them."""
    c = build_critic(MLP_CFG, 16)
    with pytest.raises(ValueError):
        score_population(
            c, torch.randn(4, 16), "real", flags=torch.zeros(4, dtype=torch.long)
        )


def test_critic_input_layer_has_two_extra_columns():
    """Catches the critic ignoring the flag (built with input_dim instead of input_dim + 2)."""
    c = build_critic({**MLP_CFG, "n_flags": 2}, 16)
    assert c.base.net[0].in_features == 18


def test_flags_change_critic_output():
    """Catches the critic's one-hot being dropped or zeroed in forward (the in_features
    test above passes against that mutation): same x, different flag, different score."""
    c = build_critic({**MLP_CFG, "n_flags": 2}, 16)
    x = torch.randn(4, 16)
    assert not torch.allclose(
        c(x, torch.full((4,), BODY)), c(x, torch.full((4,), TAIL))
    )
