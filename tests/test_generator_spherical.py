"""SphericalGenerator: unit-norm by construction, constant residual angle
across samples, residual shape conditioned on the trunk.

Each test names the mutation it catches. See
docs/ai/specs/2026-09-15-spherical-generator-design.md, "Tests".
"""

import math

import pytest
import torch

from src.models.generator import SphericalGenerator

LATENT, OUT, SKIP, HID, TANGENT = 8 + 32, 32, 32, [16, 16], 64


def make(**overrides):
    kwargs = dict(
        latent_dim=LATENT,
        output_dim=OUT,
        hidden_dims=HID,
        skip_dim=SKIP,
        tangent_hidden_dim=TANGENT,
    )
    kwargs.update(overrides)
    return SphericalGenerator(**kwargs)


def test_output_shape_and_split():
    torch.manual_seed(0)
    gen = make()
    out = gen(torch.randn(64, LATENT))
    assert out.shape == (64, OUT)
    assert gen.trunk_latent_dim == 8 and gen.skip_dim == 32


def test_output_is_unit_norm_by_construction():
    """Catches dropping the normalisation of `t` or of `u`."""
    torch.manual_seed(0)
    gen = make()
    out = gen(torch.randn(512, LATENT))
    norms = torch.linalg.vector_norm(out, dim=1)
    assert torch.allclose(norms, torch.ones(512), atol=1e-5)


def test_tangent_is_orthogonal_to_direction():
    """Catches removing the projection."""
    torch.manual_seed(0)
    gen = make()
    u, t = gen.components(torch.randn(512, LATENT))
    assert torch.allclose((u * t).sum(dim=1), torch.zeros(512), atol=1e-5)
    assert torch.allclose(
        torch.linalg.vector_norm(u, dim=1), torch.ones(512), atol=1e-5
    )
    assert torch.allclose(
        torch.linalg.vector_norm(t, dim=1), torch.ones(512), atol=1e-5
    )


def test_angle_from_direction_is_the_radius_on_every_row():
    """The hub guard. Catches a per-sample radius and a wrong sigmoid
    mapping: every row sits at exactly `radius_init` from `u`."""
    torch.manual_seed(0)
    gen = make(radius_init=0.95)
    z = torch.randn(512, LATENT)
    u, _ = gen.components(z)
    x = gen(z)
    angles = torch.acos((x * u).sum(dim=1).clamp(-1.0, 1.0))
    assert torch.allclose(angles, torch.full((512,), 0.95), atol=1e-5)
    assert float(angles.std()) < 1e-5


def test_radius_starts_at_radius_init():
    gen = make(radius_init=0.7, radius_min=0.1, radius_max=1.2)
    assert abs(float(gen.radius) - 0.7) < 1e-6


def test_radius_stays_in_the_band():
    """Catches an unclamped radius. sigmoid(+-50) is exactly 0 or 1 in
    float32, so the extremes land on the closed band's edges; +-5 stays
    strictly inside."""
    gen = make(radius_min=0.2, radius_max=1.5)
    for raw, strict in ((50.0, False), (-50.0, False), (5.0, True), (-5.0, True)):
        with torch.no_grad():
            gen.radius_raw.fill_(raw)
        r = float(gen.radius)
        assert 0.2 <= r <= 1.5
        if strict:
            assert 0.2 < r < 1.5


def test_radius_receives_a_gradient():
    """Catches a radius cut from the graph (a detached radius_raw): r is one
    learned scalar, so a loss on the output must reach it. The band and init
    tests above still pass with `self.radius_raw.detach()`."""
    torch.manual_seed(0)
    gen = make()
    gen(torch.randn(64, LATENT)).sum().backward()
    assert gen.radius_raw.grad is not None
    assert float(gen.radius_raw.grad.abs()) > 0.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(skip_dim=0), "skip_dim"),
        (dict(skip_dim=LATENT), "skip_dim"),
        (dict(hidden_dims=[]), "hidden_dims"),
        (dict(tangent_hidden_dim=0), "tangent_hidden_dim"),
        (dict(radius_min=0.0), "radius_min"),
        (dict(radius_min=0.95), "radius_min"),
        (dict(radius_init=1.5), "radius_init"),
        (dict(radius_max=math.pi / 2), "radius_max"),
    ],
)
def test_rejects_bad_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make(**kwargs)
