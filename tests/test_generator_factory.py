from types import MappingProxyType

import pytest
import torch

from src.models.generator import (
    GatedGenerator,
    Generator,
    LinearSkipGenerator,
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


# `sparse` was renamed to `gated` in b29e317, but run configs already written
# still say `sparse`. A run config is the historical record of a run that
# happened, and invariant 4 in AGENTS.md rebuilds the architecture from it at
# load time -- so the rename must not make those checkpoints unloadable.
def test_sparse_is_a_deprecated_alias_for_gated():
    cfg = dict(BASE_CFG, generator_type="sparse")
    generator = build_generator(cfg, output_dim=128)
    assert isinstance(generator, GatedGenerator)
    assert generator.gate_temperature == 0.5
    assert generator.logit_clamp == 10.0


def test_sparse_alias_honours_the_same_overrides_as_gated():
    cfg = dict(
        BASE_CFG, generator_type="sparse", gate_temperature=0.25, logit_clamp=4.0
    )
    generator = build_generator(cfg, output_dim=128)
    assert generator.gate_temperature == 0.25
    assert generator.logit_clamp == 4.0


def test_sparse_and_gated_build_the_same_architecture():
    """The alias must be load-compatible, not merely the same class."""
    sparse = build_generator(dict(BASE_CFG, generator_type="sparse"), output_dim=128)
    gated = build_generator(dict(BASE_CFG, generator_type="gated"), output_dim=128)
    sparse.load_state_dict(gated.state_dict())


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
    # The learned structure is what separates them, not the fixed noise kernel:
    # that is deliberately non-persistent, so it cannot be what raises here.
    keys = structured.state_dict().keys()
    assert "noise_kernel" not in keys
    assert {"sparsity_head.weight", "gate_coupling.weight"} <= set(keys)
    with pytest.raises(RuntimeError):
        gated.load_state_dict(structured.state_dict())
    with pytest.raises(RuntimeError):
        structured.load_state_dict(gated.state_dict())


def test_linear_skip_defaults_skip_dim_to_output_dim():
    cfg = dict(BASE_CFG, generator_type="linear_skip", latent_dim=16 + 128)
    generator = build_generator(cfg, output_dim=128)
    assert isinstance(generator, LinearSkipGenerator)
    assert generator.skip_dim == 128
    assert generator.trunk_latent_dim == 16


def test_linear_skip_honours_overrides():
    cfg = dict(
        BASE_CFG,
        generator_type="linear_skip",
        latent_dim=16 + 64,
        skip_dim=64,
        skip_init="orthogonal",
        skip_init_gain=0.5,
    )
    generator = build_generator(cfg, output_dim=128)
    assert generator.skip_dim == 64
    w = generator.skip.weight
    assert torch.allclose(w.T @ w, 0.25 * torch.eye(64), atol=1e-5)


def test_linear_skip_rejects_skip_dim_at_or_above_latent_dim():
    cfg = dict(BASE_CFG, generator_type="linear_skip", latent_dim=16, skip_dim=16)
    with pytest.raises(ValueError, match="skip_dim"):
        build_generator(cfg, output_dim=128)


def test_linear_skip_rejects_identity_init_of_the_wrong_width():
    cfg = dict(
        BASE_CFG,
        generator_type="linear_skip",
        latent_dim=16 + 64,
        skip_dim=64,
        skip_init="identity",
    )
    with pytest.raises(ValueError, match="identity"):
        build_generator(cfg, output_dim=128)
