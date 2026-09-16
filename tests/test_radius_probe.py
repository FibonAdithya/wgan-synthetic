"""The radius probe's recombination must agree with the generator it replaces.

The probe decodes `components(z)` once and rebuilds the output at many angles
instead of calling `forward`. If that rebuild is wrong -- cos and sin swapped,
the components transposed, a stray renormalisation -- every row of the probe's
table is wrong in the same direction and nothing in the output looks odd. The
first test below pins the rebuild against `forward` itself, which is the only
reference that cannot drift with it.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from src.models.generator import SphericalGenerator

_SPEC = importlib.util.spec_from_file_location(
    "radius_probe",
    Path(__file__).resolve().parents[1] / "tools" / "probes" / "radius_probe.py",
)
radius_probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(radius_probe)


def _generator(seed: int = 0) -> SphericalGenerator:
    torch.manual_seed(seed)
    return SphericalGenerator(
        latent_dim=16,
        output_dim=12,
        hidden_dims=[24, 24],
        skip_dim=8,
        tangent_hidden_dim=20,
        radius_init=0.95,
        radius_min=0.2,
        radius_max=1.5,
    ).eval()


def _draw(gen: SphericalGenerator, n: int = 64) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(7)
    return torch.randn(n, 16, generator=g)


def test_decode_at_learned_radius_reproduces_forward():
    """Catches: cos/sin swapped, components returned in the other order, a
    renormalisation that changes the mixture."""
    gen = _generator()
    z = _draw(gen)
    with torch.no_grad():
        u, t = gen.components(z)
        expected = gen(z).numpy()
        learned_r = float(gen.radius)
    got = radius_probe.decode(u.float(), t.float(), learned_r)
    assert got == pytest.approx(expected, abs=1.0e-6)


def test_decode_at_band_ends_returns_each_component():
    """r = 0 is the trunk direction alone and r = pi/2 the tangent alone. A
    swapped pair passes the forward test only if cos and sin are swapped too;
    this pins each end independently."""
    gen = _generator()
    z = _draw(gen)
    with torch.no_grad():
        u, t = gen.components(z)
    u = u.float()
    t = t.float()
    assert radius_probe.decode(u, t, 0.0) == pytest.approx(u.numpy(), abs=1.0e-6)
    assert radius_probe.decode(u, t, math.pi / 2) == pytest.approx(
        t.numpy(), abs=1.0e-6
    )


def test_decode_changes_the_mixture_with_r():
    """A `decode` that ignored r -- returning forward's output whatever the
    angle -- would pass nothing here."""
    gen = _generator()
    z = _draw(gen)
    with torch.no_grad():
        u, t = gen.components(z)
    a = radius_probe.decode(u.float(), t.float(), 1.10)
    b = radius_probe.decode(u.float(), t.float(), 1.40)
    assert abs(a - b).max() > 1.0e-3


def test_check_components_rejects_non_orthogonal():
    """The invariant check must be discriminating: feeding it the raw tangent
    before projection has to fail, or it is decorative."""
    u = torch.nn.functional.normalize(torch.randn(32, 12), dim=1)
    # Unit norm, but deliberately not orthogonal to u.
    bad = torch.nn.functional.normalize(u + 0.3 * torch.randn(32, 12), dim=1)
    with pytest.raises(SystemExit, match="not orthogonal"):
        radius_probe.check_components("bad", u, bad)


def test_check_components_rejects_non_unit():
    u = torch.nn.functional.normalize(torch.randn(32, 12), dim=1)
    t = torch.nn.functional.normalize(torch.randn(32, 12), dim=1)
    t = t - (t * u).sum(dim=1, keepdim=True) * u
    t = torch.nn.functional.normalize(t, dim=1)
    with pytest.raises(SystemExit, match="not unit norm"):
        radius_probe.check_components("bad", u * 1.5, t)


def test_check_components_accepts_the_real_thing():
    gen = _generator()
    z = _draw(gen)
    with torch.no_grad():
        u, t = gen.components(z)
    stats = radius_probe.check_components("ok", u.float(), t.float())
    assert stats["max_abs_dot"] < radius_probe.ORTHO_TOL
