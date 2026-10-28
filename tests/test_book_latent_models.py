from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

from src.models.book_latent import (
    BookLatentGenerator,
    DriftLatent,
    WindowCritic,
    book_latent_settings,
    build_book_latent,
)
from src.models.conditional import BODY, TAIL, FlagConditionedCritic
from src.models.generator import build_generator


def full_config() -> dict:
    """The full-size v2 config, read from configs/scout_kv/v2/L19.yaml."""
    return yaml.safe_load(
        (
            Path(__file__).resolve().parent.parent / "configs/scout_kv/v2/L19.yaml"
        ).read_text()
    )


def small_config() -> dict:
    cfg = full_config()
    cfg["model"].update(
        latent_dim=8,
        generator_hidden_dims=[16, 16],
        critic_hidden_dims=[16, 16],
    )
    cfg["model"]["book_latent"] = {
        "channels": 4,
        "n_features": 2,
        "min_length": 2.0,
        "max_length": 64.0,
    }
    cfg["model"]["window_critic"] = {
        "window": 8,
        "strides": [1, 4],
        "token_hidden_dims": [16, 8],
        "window_hidden_dims": [16, 8],
    }
    cfg["data"]["descriptor_dim"] = 12
    return cfg


def test_omega_matches_the_spec_formula():
    """Catches wrong length scales (linspace for logspace, or the range not 2-16,384)."""
    lat = DriftLatent(42, 64, 8, 2.0, 16384.0)
    g = torch.Generator().manual_seed(42)
    ell = torch.logspace(math.log10(2.0), math.log10(16384.0), 64)
    assert torch.equal(lat.omega, torch.randn(64, 8, generator=g) / ell[:, None])


def test_latent_value_matches_the_formula():
    """Catches a missing 1/sqrt(M), a swapped cos/sin, or a and b read from the wrong slot,
    against an inline evaluation of u_c(t) at 3 positions for one book."""
    lat = DriftLatent(7, 5, 3, 2.0, 100.0)
    coef = lat.sample_coefficients(1, "cpu", torch.Generator().manual_seed(1))
    pos = torch.tensor([[0, 5, 123]])
    got = lat(coef, pos)
    assert got.shape == (1, 3, 5)
    for li, t in enumerate([0, 5, 123]):
        for c in range(5):
            want = 0.0
            for m in range(3):
                w = float(lat.omega[c, m])
                a, b = float(coef[0, 0, c, m]), float(coef[0, 1, c, m])
                want += a * math.cos(w * t) + b * math.sin(w * t)
            want /= math.sqrt(3)
            assert got[0, li, c].item() == pytest.approx(want, abs=1e-4)


def test_latent_is_shared_within_a_book_and_differs_between_books():
    """Catches every token drawing its own latent, or every book the same (spec test 1):
    one book's u at the same position from two calls is equal; two books' u differ."""
    lat = DriftLatent(3, 4, 2, 2.0, 64.0)
    coef = lat.sample_coefficients(2, "cpu", torch.Generator().manual_seed(0))
    pos = torch.tensor([[10, 20, 10], [10, 20, 10]])
    u = lat(coef, pos)
    assert torch.equal(u[0, 0], u[0, 2])
    assert not torch.allclose(u[0, 0], u[1, 0])
    again = lat(coef[:1], pos[:1])
    assert torch.equal(again[0], u[0])


def test_slow_channels_drift_slowly():
    """Catches omega not divided by the length scale: the last channel (l = 16,384) changes by
    less than 0.05 between positions 0 and 64, averaged over 256 books; the first (l = 2) by more than 0.5."""
    lat = DriftLatent(42, 64, 8, 2.0, 16384.0)
    coef = lat.sample_coefficients(256, "cpu", torch.Generator().manual_seed(0))
    pos = torch.tensor([[0, 64]]).expand(256, 2)
    u = lat(coef, pos)
    change = (u[:, 1] - u[:, 0]).abs().mean(0)
    assert change[-1] < 0.05
    assert change[0] > 0.5


def test_flag_and_latent_change_the_generator_output():
    """Catches a generator that ignores the flag or the latent input."""
    gen, _, _ = build_book_latent(small_config())
    z = torch.randn(3, 8)
    u = torch.randn(3, 4)
    base = gen(z, u, torch.full((3,), BODY))
    assert base.shape == (3, 12)
    assert not torch.allclose(base, gen(z, u, torch.full((3,), TAIL)))
    assert not torch.allclose(base, gen(z, u + 1.0, torch.full((3,), BODY)))


def test_generator_accepts_leading_dims():
    """Catches a forward that only handles 2-D input: (n, L) tokens in, (n, L, D) out,
    identical to the flattened call."""
    gen, _, _ = build_book_latent(small_config())
    z, u = torch.randn(2, 5, 8), torch.randn(2, 5, 4)
    flags = torch.randint(0, 2, (2, 5))
    out = gen(z, u, flags)
    assert out.shape == (2, 5, 12)
    flat = gen(z.reshape(10, 8), u.reshape(10, 4), flags.reshape(10))
    assert torch.allclose(out.reshape(10, 12), flat, atol=1e-6)


def test_generator_refuses_a_call_without_flags():
    """Catches a default for flags that lets an unconditioned call through."""
    gen, _, _ = build_book_latent(small_config())
    with pytest.raises(TypeError):
        gen(torch.randn(2, 8), torch.randn(2, 4))


def test_generator_input_width_is_latent_plus_channels_plus_flags():
    """Catches the latent or the flag missing from the input."""
    gen, _, _ = build_book_latent(full_config())
    assert isinstance(gen, BookLatentGenerator)
    assert gen.net.net[0].in_features == 128 + 64 + 2
    assert gen.net.net[-1].out_features == 5120


def _window_inputs(n=3, w=8, d=12):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, w, d, generator=g)
    flags = torch.zeros(n, w, dtype=torch.long)
    return x, flags, torch.zeros(n, dtype=torch.long)


def test_window_critic_reads_order_and_stride():
    """Catches the window head being skipped (score = token mean only): reversing the token
    order of a window or changing its stride index changes the score."""
    _, _, crit = build_book_latent(small_config())
    x, flags, s = _window_inputs()
    base = crit(x, flags, s)
    assert base.shape == (3,)
    assert not torch.allclose(base, crit(x.flip(1), flags, s))
    assert not torch.allclose(base, crit(x, flags, s + 1))


def test_window_critic_reads_the_flags():
    """Catches the window critic's one-hot being dropped or zeroed (spec: all three networks take
    the same flag encoding): same x and stride, flags all BODY vs all TAIL -> different scores;
    and its first layer has input_dim + 2 inputs."""
    _, _, crit = build_book_latent(small_config())
    x, flags, s = _window_inputs()
    assert not torch.allclose(
        crit(x, flags, s), crit(x, torch.full_like(flags, TAIL), s)
    )
    assert crit.tok[0].in_features == 12 + 2
    _, _, full = build_book_latent(full_config())
    assert full.tok[0].in_features == 5120 + 2
    assert full.win[0].in_features == 8 * 512 + 6


def test_window_critic_token_head_is_a_mean_over_the_window():
    """Catches a token head that sums or takes the last token instead of averaging: with the
    window head's last layer zeroed the score equals the mean of per-token head outputs."""
    _, _, crit = build_book_latent(small_config())
    with torch.no_grad():
        crit.win[-1].weight.zero_()
        crit.win[-1].bias.zero_()
    x, flags, s = _window_inputs()
    out = crit(x, flags, s)
    single = WindowCritic(
        input_dim=12,
        window=1,
        n_strides=2,
        token_hidden_dims=[16, 8],
        window_hidden_dims=[16, 8],
        negative_slope=0.2,
        n_flags=2,
    )
    single.tok.load_state_dict(crit.tok.state_dict())
    single.tok_head.load_state_dict(crit.tok_head.state_dict())
    with torch.no_grad():
        single.win[-1].weight.zero_()
        single.win[-1].bias.zero_()
        per_token = torch.stack(
            [single(x[:, i : i + 1], flags[:, i : i + 1], s) for i in range(8)], dim=1
        )
    assert torch.allclose(out, per_token.mean(1), atol=1e-6)
    assert not torch.allclose(out, per_token.sum(1), atol=1e-3)


def test_token_critic_is_v1s_flag_critic():
    """Catches a new token critic drifting from v1's: same class, same layer widths."""
    _, tok, _ = build_book_latent(full_config())
    assert type(tok) is FlagConditionedCritic and tok.base.net[0].in_features == 5122


def test_omega_is_in_the_state_dict():
    """Catches omega being a plain attribute: a checkpoint would not carry it."""
    gen, _, _ = build_book_latent(small_config())
    assert "latent.omega" in gen.state_dict()


def test_build_generator_refuses_book_latent():
    """Catches build_generator silently building an mlp for a book_latent config."""
    with pytest.raises(ValueError, match="src.models.book_latent") as exc:
        build_generator(full_config()["model"], 5120)
    assert "src.sample.sample_books" in str(exc.value)


def test_build_book_latent_rejects_other_types():
    """Catches build_book_latent accepting a v0/v1 config or a missing flag count."""
    cfg = small_config()
    cfg["model"]["generator_type"] = "mlp"
    with pytest.raises(ValueError):
        build_book_latent(cfg)
    cfg = small_config()
    cfg["model"]["n_flags"] = None
    with pytest.raises(ValueError):
        build_book_latent(cfg)


def test_settings_are_plain_and_survive_a_save_load_round_trip(tmp_path):
    """Catches tuples or tensors in the record: a torch.save/load round trip must compare equal."""
    cfg = full_config()
    cfg["model"]["book_latent"]["min_length"] = 2
    settings = book_latent_settings(cfg)
    assert settings["kv_books"] == list(range(12))
    assert settings["window_critic"]["strides"] == [1, 4, 16, 64, 256, 1024]
    assert settings["kv_n_tail"] == 1024 and settings["kv_layer"] == 19
    torch.save({"book_latent": settings}, tmp_path / "c.pt")
    back = torch.load(tmp_path / "c.pt", weights_only=False)["book_latent"]
    assert back == book_latent_settings(copy.deepcopy(cfg))


def test_same_seed_gives_same_omega_and_other_seed_differs():
    """Catches omega ignoring config['seed']."""
    a, _, _ = build_book_latent(small_config())
    b, _, _ = build_book_latent(small_config())
    other = small_config()
    other["seed"] = 43
    c, _, _ = build_book_latent(other)
    assert torch.equal(a.latent.omega, b.latent.omega)
    assert not torch.equal(a.latent.omega, c.latent.omega)
