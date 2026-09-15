"""SphericalGenerator: unit-norm by construction, constant residual angle
across samples, residual shape conditioned on the trunk.

Each test names the mutation it catches. See
docs/ai/specs/2026-09-15-spherical-generator-design.md, "Tests".
"""

import math

import pytest
import torch

from src.eval.eda.metrics import effective_rank
from src.models.generator import SphericalGenerator, _effective_rank

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
    with torch.no_grad():
        assert float(angles.std()) < 1e-5


def test_radius_starts_at_radius_init():
    gen = make(radius_init=0.7, radius_min=0.1, radius_max=1.2)
    assert abs(float(gen.radius.detach()) - 0.7) < 1e-6


def test_radius_stays_in_the_band():
    """Catches an unclamped radius. sigmoid(+-50) is exactly 0 or 1 in
    float32, so the extremes land on the closed band's edges; +-5 stays
    strictly inside."""
    gen = make(radius_min=0.2, radius_max=1.5)
    for raw, strict in ((50.0, False), (-50.0, False), (5.0, True), (-5.0, True)):
        with torch.no_grad():
            gen.radius_raw.fill_(raw)
        r = float(gen.radius.detach())
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


def _jacobian_wrt_skip(gen, z_trunk, z_skip):
    def f(zs):
        return gen(torch.cat([z_trunk, zs[None, :]], dim=1))[0]

    return torch.autograd.functional.jacobian(f, z_skip)


@pytest.mark.parametrize("skip_dim, expected_rank", [(32, OUT - 2), (8, 8)])
def test_local_rank_wrt_skip_block_is_full(skip_dim, expected_rank):
    """The Jacobian of one output row wrt the skip block has rank
    min(skip_dim, output_dim - 2): t is a unit vector in the (output_dim - 1)
    dimensional tangent space, so its Jacobian loses one more dimension.
    Catches zeroing the tangent head's skip weight."""
    torch.manual_seed(0)
    gen = make(latent_dim=8 + skip_dim, skip_dim=skip_dim)
    z_trunk = torch.randn(1, 8)
    z_skip = torch.randn(skip_dim)
    jac = _jacobian_wrt_skip(gen, z_trunk, z_skip)
    assert jac.shape == (OUT, skip_dim)
    assert int(torch.linalg.matrix_rank(jac, rtol=1e-4)) >= expected_rank


def test_tangent_head_depends_on_the_trunk_through_the_modulation():
    """Catches a modulation that is wired but inert. With gamma and beta
    live, the pre-projection tangent output for a fixed skip draw changes
    with the trunk latent; with both frozen to constants it does not."""
    torch.manual_seed(0)
    gen = make()
    z_skip = torch.randn(4, SKIP)
    h1 = gen.trunk(torch.randn(4, 8))
    h2 = gen.trunk(torch.randn(4, 8))
    assert not torch.allclose(gen.gamma(h1), gen.gamma(h2))
    assert not torch.allclose(gen.tangent_raw(h1, z_skip), gen.tangent_raw(h2, z_skip))
    with torch.no_grad():
        for layer in (gen.gamma, gen.beta):
            layer.weight.zero_()
            layer.bias.zero_()
    assert torch.allclose(
        gen.tangent_raw(h1, z_skip), gen.tangent_raw(h2, z_skip), atol=1e-6
    )


def test_diagnostics_report_radius_and_direction_rank():
    """Catches the rank computed on the wrong tensor: with the direction
    head pinned to one vector, u is constant and its effective rank is 1
    whatever t does."""
    torch.manual_seed(0)
    gen = make(radius_init=0.8)
    z = torch.randn(2048, LATENT)
    d = gen.diagnostics(z)
    assert set(d) == {"radius", "direction_effective_rank"}
    assert abs(d["radius"] - 0.8) < 1e-6
    assert 1.0 < d["direction_effective_rank"] <= OUT
    with torch.no_grad():
        gen.direction.weight.zero_()
        gen.direction.weight[0, 0] = 1.0
    assert gen.diagnostics(z)["direction_effective_rank"] < 1.5


def test_effective_rank_centres_before_the_eigendecomposition():
    """Catches dropping the mean-centring in `_effective_rank`: an offset
    added uniformly to every row changes the raw second moment but not the
    true spread around the mean, so an uncentred computation disagrees with
    `src.eval.eda.metrics.effective_rank` (which centres via PCA) once the
    input carries a large constant offset."""
    torch.manual_seed(0)
    scales = torch.arange(1, 17, dtype=torch.float32)
    x = torch.randn(2048, 16) * scales + 100.0
    got = _effective_rank(x)
    want = effective_rank(x.numpy())
    assert abs(got - want) / abs(want) < 1e-4
