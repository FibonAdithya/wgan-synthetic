from types import MappingProxyType

import pytest

from src.models.generator import Generator, SparseGenerator, build_generator

BASE_CFG = {
    "latent_dim": 16,
    "generator_hidden_dims": [32, 32],
    "negative_slope": 0.2,
}


def test_missing_generator_type_defaults_to_mlp():
    assert isinstance(build_generator(dict(BASE_CFG), output_dim=128), Generator)


def test_explicit_mlp():
    cfg = dict(BASE_CFG, generator_type="mlp")
    assert isinstance(build_generator(cfg, output_dim=128), Generator)


def test_sparse():
    cfg = dict(BASE_CFG, generator_type="sparse")
    generator = build_generator(cfg, output_dim=128)
    assert isinstance(generator, SparseGenerator)
    assert generator.gate_temperature == 0.5
    assert generator.logit_clamp == 10.0


def test_sparse_honours_overrides():
    cfg = dict(BASE_CFG, generator_type="sparse", gate_temperature=0.25, logit_clamp=4.0)
    generator = build_generator(cfg, output_dim=128)
    assert generator.gate_temperature == 0.25
    assert generator.logit_clamp == 4.0


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="nope"):
        build_generator(dict(BASE_CFG, generator_type="nope"), output_dim=128)


def test_output_dim_is_respected():
    generator = build_generator(dict(BASE_CFG, generator_type="sparse"), output_dim=64)
    assert generator.magnitude_head.out_features == 64
    assert generator.gate_head.out_features == 64


def test_read_only_mapping_config_is_accepted():
    cfg = MappingProxyType(dict(BASE_CFG, generator_type="sparse"))
    assert isinstance(build_generator(cfg, output_dim=128), SparseGenerator)


def test_checkpoint_mismatch_fails_loudly():
    sparse = build_generator(dict(BASE_CFG, generator_type="sparse"), output_dim=128)
    mlp = build_generator(dict(BASE_CFG, generator_type="mlp"), output_dim=128)
    with pytest.raises(RuntimeError):
        mlp.load_state_dict(sparse.state_dict())
    with pytest.raises(RuntimeError):
        sparse.load_state_dict(mlp.state_dict())
