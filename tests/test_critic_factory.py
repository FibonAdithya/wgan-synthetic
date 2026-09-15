"""`build_critic` mirrors `build_generator`: the config's `critic_type`
picks the class, and an unknown string fails loudly rather than silently
training the per-vector critic."""

import pytest
import torch

from src.models.critic import (
    CRITIC_TYPES,
    Critic,
    NeighbourhoodCritic,
    SetNeighbourhoodCritic,
    build_critic,
)

BASE_CFG = {"critic_hidden_dims": [16, 8], "negative_slope": 0.2}


def test_the_documented_types():
    """Catches a type added to or dropped from the tuple without the docs table changing"""
    assert CRITIC_TYPES == (
        "per_vector",
        "neighbourhood",
        "neighbourhood_bank",
        "neighbourhood_set",
    )


def test_missing_critic_type_defaults_to_the_per_vector_critic():
    """Catches a factory that dispatches to NeighbourhoodCritic by default"""
    critic = build_critic(dict(BASE_CFG), input_dim=12)
    assert type(critic) is Critic
    first = next(m for m in critic.net if isinstance(m, torch.nn.Linear))
    assert first.in_features == 12


def test_explicit_per_vector():
    """Catches per_vector no longer building the plain Critic class"""
    assert (
        type(build_critic(dict(BASE_CFG, critic_type="per_vector"), input_dim=12))
        is Critic
    )


def test_neighbourhood_reads_k_and_floor_from_the_config():
    """Catches k/floor read from the wrong config key"""
    cfg = dict(
        BASE_CFG, critic_type="neighbourhood", critic_k=7, critic_distance_floor=0.02
    )
    critic = build_critic(cfg, input_dim=12)
    assert isinstance(critic, NeighbourhoodCritic)
    assert critic.k == 7
    assert critic.distance_floor == 0.02


def test_neighbourhood_defaults_k_20_and_floor_0_01():
    """Catches the neighbourhood defaults drifting from DEFAULT_K/DEFAULT_DISTANCE_FLOOR"""
    critic = build_critic(dict(BASE_CFG, critic_type="neighbourhood"), input_dim=12)
    assert (critic.k, critic.distance_floor) == (20, 0.01)


def test_negative_slope_reaches_every_class():
    """Catches negative_slope dropped on one branch"""
    for kind in CRITIC_TYPES:
        critic = build_critic(
            dict(BASE_CFG, critic_type=kind, negative_slope=0.31), input_dim=6
        )
        net = critic.net if isinstance(critic, Critic) else critic.mlp.net
        modules = list(net)
        if isinstance(critic, SetNeighbourhoodCritic):
            modules += list(critic.edge)
        slopes = {
            m.negative_slope for m in modules if isinstance(m, torch.nn.LeakyReLU)
        }
        assert slopes == {0.31}, kind


def test_unknown_type_raises_naming_the_value():
    """Catches an unknown type silently building the per-vector critic"""
    with pytest.raises(ValueError, match="vibes"):
        build_critic(dict(BASE_CFG, critic_type="vibes"), input_dim=12)


def test_state_dicts_do_not_cross_load():
    """A resume must not silently load per-vector weights into the
    neighbourhood critic or vice versa, and likewise for the set critic."""
    a = build_critic(dict(BASE_CFG, critic_type="per_vector"), input_dim=12)
    b = build_critic(
        dict(BASE_CFG, critic_type="neighbourhood", critic_k=4), input_dim=12
    )
    c = build_critic(
        dict(
            BASE_CFG,
            critic_type="neighbourhood_set",
            critic_k=4,
            critic_edge_dim=8,
        ),
        input_dim=12,
    )
    with pytest.raises(RuntimeError):
        a.load_state_dict(b.state_dict())
    with pytest.raises(RuntimeError):
        b.load_state_dict(a.state_dict())
    with pytest.raises(RuntimeError):
        a.load_state_dict(c.state_dict())
    with pytest.raises(RuntimeError):
        c.load_state_dict(a.state_dict())
    with pytest.raises(RuntimeError):
        b.load_state_dict(c.state_dict())
    with pytest.raises(RuntimeError):
        c.load_state_dict(b.state_dict())


def test_neighbourhood_set_reads_edge_dim_and_pool_from_the_config():
    cfg = dict(
        BASE_CFG,
        critic_type="neighbourhood_set",
        critic_k=4,
        critic_edge_dim=32,
        critic_edge_pool="mean",
    )
    critic = build_critic(cfg, input_dim=12)
    assert isinstance(critic, SetNeighbourhoodCritic)
    assert (critic.k, critic.edge_dim, critic.edge_pool) == (4, 32, "mean")


def test_neighbourhood_set_defaults_edge_dim_128_and_max_pool():
    critic = build_critic(dict(BASE_CFG, critic_type="neighbourhood_set"), input_dim=12)
    assert (critic.edge_dim, critic.edge_pool) == (128, "max")
