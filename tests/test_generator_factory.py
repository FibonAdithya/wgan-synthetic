from types import MappingProxyType

import pytest

from src.models.generator import (
    Generator,
    GatedGenerator,
    StructuredGateGenerator,
    build_generator,
)

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


def test_gated():
    cfg = dict(BASE_CFG, generator_type="gated")
    generator = build_generator(cfg, output_dim=128)
    assert isinstance(generator, GatedGenerator)
    assert generator.gate_temperature == 0.5
    assert generator.logit_clamp == 10.0


def test_gated_honours_overrides():
    cfg = dict(BASE_CFG, generator_type="gated", gate_temperature=0.25, logit_clamp=4.0)
    generator = build_generator(cfg, output_dim=128)
    assert generator.gate_temperature == 0.25
    assert generator.logit_clamp == 4.0


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="nope"):
        build_generator(dict(BASE_CFG, generator_type="nope"), output_dim=128)


def test_sparse_is_no_longer_accepted():
    with pytest.raises(ValueError, match="Unknown generator_type"):
        build_generator(dict(BASE_CFG, generator_type="sparse"), output_dim=128)


def test_output_dim_is_respected():
    generator = build_generator(dict(BASE_CFG, generator_type="gated"), output_dim=64)
    assert generator.magnitude_head.out_features == 64
    assert generator.gate_head.out_features == 64


def test_read_only_mapping_config_is_accepted():
    cfg = MappingProxyType(dict(BASE_CFG, generator_type="gated"))
    assert isinstance(build_generator(cfg, output_dim=128), GatedGenerator)


def test_checkpoint_mismatch_fails_loudly():
    gated = build_generator(dict(BASE_CFG, generator_type="gated"), output_dim=128)
    mlp = build_generator(dict(BASE_CFG, generator_type="mlp"), output_dim=128)
    with pytest.raises(RuntimeError):
        mlp.load_state_dict(gated.state_dict())
    with pytest.raises(RuntimeError):
        gated.load_state_dict(mlp.state_dict())


def test_structured_gated():
    cfg = dict(BASE_CFG, generator_type="structured_gated")
    generator = build_generator(cfg, output_dim=128)
    assert isinstance(generator, StructuredGateGenerator)
    assert generator.layout == (4, 4, 8)
    assert generator.gate_kernel == 3
    assert generator.gate_temperature == 0.5
    assert generator.logit_clamp == 10.0


def test_structured_gated_honours_overrides():
    cfg = dict(
        BASE_CFG,
        generator_type="structured_gated",
        layout=[2, 4, 8],
        gate_kernel=1,
        noise_kernel_sigma=1.5,
        logit_clamp=4.0,
    )
    generator = build_generator(cfg, output_dim=64)
    assert generator.layout == (2, 4, 8)
    assert generator.gate_kernel == 1
    assert generator.noise_kernel_sigma == 1.5
    assert generator.logit_clamp == 4.0


def test_structured_gated_rejects_a_layout_that_does_not_match_output_dim():
    cfg = dict(BASE_CFG, generator_type="structured_gated", layout=[4, 4, 8])
    with pytest.raises(ValueError, match="layout"):
        build_generator(cfg, output_dim=64)


def test_structured_and_gated_checkpoints_do_not_interchange():
    structured = build_generator(
        dict(BASE_CFG, generator_type="structured_gated"), output_dim=128
    )
    gated = build_generator(dict(BASE_CFG, generator_type="gated"), output_dim=128)
    with pytest.raises(RuntimeError):
        gated.load_state_dict(structured.state_dict())
    with pytest.raises(RuntimeError):
        structured.load_state_dict(gated.state_dict())
