"""Unit tests for the generator-EMA machinery and the collapse monitor."""

import numpy as np
import pytest
import torch
from torch import nn

from src.train.train_wgan_gp import (
    collapse_stats,
    ema_bias_correction,
    ema_update,
    ema_weights,
    init_ema_params,
    load_ema_into_model,
    save_checkpoint,
)


class TinyModel(nn.Module):
    """Two-parameter model with deterministic, easily hand-checked weights."""

    def __init__(self, w: float = 1.0, b: float = -2.0) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 1)
        with torch.no_grad():
            self.lin.weight.fill_(w)
            self.lin.bias.fill_(b)

    def set_weight(self, w: float, b: float) -> None:
        with torch.no_grad():
            self.lin.weight.fill_(w)
            self.lin.bias.fill_(b)


def params(model: nn.Module) -> dict:
    return {name: p.data.clone() for name, p in model.named_parameters()}


# --------------------------------------------------------------------------
# ema_update
# --------------------------------------------------------------------------


def test_init_ema_params_starts_at_zero():
    model = TinyModel(w=3.0, b=7.0)
    ema = init_ema_params(model)
    assert set(ema) == {"lin.weight", "lin.bias"}
    for tensor in ema.values():
        assert torch.equal(tensor, torch.zeros_like(tensor))


def test_ema_update_matches_hand_computed_value_over_several_steps():
    decay = 0.8
    model = TinyModel()
    ema = init_ema_params(model)

    weight_sequence = [1.0, 2.0, 5.0, -3.0]
    expected = 0.0
    for w in weight_sequence:
        model.set_weight(w, b=0.0)
        ema_update(ema, model, decay)
        expected = decay * expected + (1.0 - decay) * w

    assert ema["lin.weight"].shape == model.lin.weight.shape
    assert ema["lin.weight"].allclose(torch.full_like(ema["lin.weight"], expected))


def test_ema_update_is_in_place():
    model = TinyModel()
    ema = init_ema_params(model)
    handle = ema["lin.weight"]
    ema_update(ema, model, 0.5)
    assert ema["lin.weight"] is handle
    assert not torch.equal(handle, torch.zeros_like(handle))


def test_ema_update_does_not_touch_live_weights():
    model = TinyModel(w=4.0, b=1.5)
    before = params(model)
    ema = init_ema_params(model)
    ema_update(ema, model, 0.9)
    after = params(model)
    for name in before:
        assert torch.equal(before[name], after[name])


# --------------------------------------------------------------------------
# bias correction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("decay", [0.5, 0.9, 0.999])
def test_one_update_bias_corrected_ema_equals_the_parameter(decay):
    """Standard bias-correction property: after one update the corrected EMA
    reproduces the observed parameter exactly (no initialisation bleed)."""
    model = TinyModel(w=2.5, b=-0.75)
    ema = init_ema_params(model)
    ema_update(ema, model, decay)

    target = TinyModel(w=0.0, b=0.0)
    load_ema_into_model(ema, target, decay=decay, ema_step=1)

    assert torch.allclose(target.lin.weight, model.lin.weight, atol=1e-6)
    assert torch.allclose(target.lin.bias, model.lin.bias, atol=1e-6)


def test_bias_corrected_ema_of_a_constant_parameter_is_that_constant():
    """With the parameter held fixed, the corrected EMA equals it at every step
    -- the property that makes short runs usable."""
    decay = 0.999
    model = TinyModel(w=1.25, b=-4.0)
    ema = init_ema_params(model)
    target = TinyModel(w=0.0, b=0.0)

    for step in range(1, 6):
        ema_update(ema, model, decay)
        load_ema_into_model(ema, target, decay=decay, ema_step=step)
        assert torch.allclose(target.lin.weight, model.lin.weight, atol=1e-6)
        assert torch.allclose(target.lin.bias, model.lin.bias, atol=1e-6)


def test_bias_correction_matters_for_short_runs_at_high_decay():
    """Regression guard for the reviewed bug: 200 steps at 0.999 leaves the raw
    accumulator at ~18% of its target."""
    decay, steps = 0.999, 200
    model = TinyModel(w=1.0, b=1.0)
    ema = init_ema_params(model)
    for _ in range(steps):
        ema_update(ema, model, decay)

    raw = float(ema["lin.weight"].detach().flatten()[0])
    assert raw == pytest.approx(1.0 - decay**steps, rel=1e-4)
    assert raw < 0.2  # uncorrected: mostly still the zero init

    corrected = TinyModel(w=0.0, b=0.0)
    load_ema_into_model(ema, corrected, decay=decay, ema_step=steps)
    assert float(corrected.lin.weight.detach().flatten()[0]) == pytest.approx(1.0, rel=1e-4)


def test_load_ema_leaves_the_accumulator_uncorrected():
    decay, steps = 0.9, 3
    model = TinyModel(w=1.0, b=1.0)
    ema = init_ema_params(model)
    for _ in range(steps):
        ema_update(ema, model, decay)
    snapshot = {k: v.clone() for k, v in ema.items()}

    load_ema_into_model(ema, TinyModel(), decay=decay, ema_step=steps)
    for name, tensor in ema.items():
        assert torch.equal(tensor, snapshot[name])


@pytest.mark.parametrize(
    "decay, step, expected",
    [
        (0.0, 5, 1.0),  # EMA disabled
        (1.0, 5, 1.0),  # degenerate decay -> denominator is exactly zero
        (0.9, 0, 1.0),  # no updates yet
        (0.9, -1, 1.0),  # nonsensical step count
        (0.5, 1, 2.0),
    ],
)
def test_ema_bias_correction_guards(decay, step, expected):
    assert ema_bias_correction(decay, step) == pytest.approx(expected)


def test_ema_bias_correction_is_finite_for_tiny_denominators():
    # decay astronomically close to 1: 1 - decay**1 underflows toward zero.
    assert np.isfinite(ema_bias_correction(1.0 - 1e-18, 1))


def test_load_ema_without_correction_args_copies_raw_values():
    """Default args keep the historical (uncorrected) copy semantics."""
    ema = {"lin.weight": torch.full((1, 2), 0.25), "lin.bias": torch.full((1,), -0.5)}
    model = TinyModel()
    load_ema_into_model(ema, model)
    assert torch.equal(model.lin.weight.data, ema["lin.weight"])
    assert torch.equal(model.lin.bias.data, ema["lin.bias"])


# --------------------------------------------------------------------------
# swap / restore
# --------------------------------------------------------------------------


def test_ema_weights_swaps_in_ema_and_restores_bitwise():
    decay = 0.9
    model = TinyModel(w=0.3, b=0.7)
    ema = init_ema_params(model)
    ema_update(ema, model, decay)

    model.set_weight(9.0, -9.0)
    live = params(model)

    with ema_weights(model, ema, decay=decay, ema_step=1):
        # Inside: the bias-corrected EMA, i.e. the weights at update time.
        assert torch.allclose(model.lin.weight, torch.full_like(model.lin.weight, 0.3))
        assert torch.allclose(model.lin.bias, torch.full_like(model.lin.bias, 0.7))

    for name, p in model.named_parameters():
        assert torch.equal(p.data, live[name]), f"{name} not restored bitwise"


def test_ema_weights_restores_when_the_body_raises():
    decay = 0.9
    model = TinyModel(w=1.0, b=2.0)
    ema = init_ema_params(model)
    ema_update(ema, model, decay)
    model.set_weight(-5.0, 6.0)
    live = params(model)

    class EvalBlewUp(RuntimeError):
        pass

    with pytest.raises(EvalBlewUp):
        with ema_weights(model, ema, decay=decay, ema_step=1):
            raise EvalBlewUp("sampling failed")

    for name, p in model.named_parameters():
        assert torch.equal(p.data, live[name]), f"{name} stranded on EMA weights"


def test_ema_weights_is_a_noop_when_ema_disabled():
    model = TinyModel(w=1.1, b=2.2)
    live = params(model)
    with ema_weights(model, {}, decay=0.0, ema_step=0):
        for name, p in model.named_parameters():
            assert torch.equal(p.data, live[name])
    for name, p in model.named_parameters():
        assert torch.equal(p.data, live[name])


def test_ema_weights_restores_even_if_a_key_is_missing_from_ema():
    """Partial EMA coverage must not corrupt the untracked parameter."""
    model = TinyModel(w=1.0, b=2.0)
    ema = {"lin.weight": torch.full((1, 2), 5.0)}
    live = params(model)

    with ema_weights(model, ema, decay=0.5, ema_step=1):
        assert torch.equal(model.lin.bias.data, live["lin.bias"])
    for name, p in model.named_parameters():
        assert torch.equal(p.data, live[name])


# --------------------------------------------------------------------------
# checkpoint provenance
# --------------------------------------------------------------------------


def test_save_checkpoint_records_which_generator_weights_it_holds(tmp_path):
    gen, critic = TinyModel(), TinyModel()
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-3)

    save_checkpoint(gen, critic, opt_g, opt_d, tmp_path, step=1, generator_weights="live")
    save_checkpoint(
        gen, critic, opt_g, opt_d, tmp_path, step=2, best=True, generator_weights="ema"
    )

    live = torch.load(tmp_path / "checkpoint_step_1.pt", weights_only=False)
    best = torch.load(tmp_path / "best_generator.pt", weights_only=False)
    assert live["generator_weights"] == "live"
    assert best["generator_weights"] == "ema"


def test_save_checkpoint_defaults_to_live(tmp_path):
    gen, critic = TinyModel(), TinyModel()
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-3)
    save_checkpoint(gen, critic, opt_g, opt_d, tmp_path, step=1)
    ckpt = torch.load(tmp_path / "checkpoint_step_1.pt", weights_only=False)
    assert ckpt["generator_weights"] == "live"


def test_save_checkpoint_rejects_unknown_provenance(tmp_path):
    gen, critic = TinyModel(), TinyModel()
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        save_checkpoint(
            gen, critic, opt_g, opt_d, tmp_path, step=1, generator_weights="maybe"
        )


# --------------------------------------------------------------------------
# collapse_stats
# --------------------------------------------------------------------------


def test_collapse_stats_flags_a_collapsed_batch():
    fake = np.tile(np.linspace(0.0, 1.0, 8, dtype=np.float32), (64, 1))
    stats = collapse_stats(fake)
    assert stats["fake_std"] == pytest.approx(0.0, abs=1e-6)
    assert stats["fake_min_pdist"] == pytest.approx(0.0, abs=1e-6)
    assert stats["fake_mean_pdist"] == pytest.approx(0.0, abs=1e-6)


def test_collapse_stats_does_not_flag_a_spread_batch():
    rng = np.random.default_rng(1234)
    fake = rng.normal(size=(64, 8)).astype(np.float32)
    stats = collapse_stats(fake)
    assert stats["fake_std"] > 0.5
    assert stats["fake_min_pdist"] > 0.0
    assert stats["fake_mean_pdist"] > 1.0


def test_collapse_stats_returns_json_serializable_floats():
    import json

    rng = np.random.default_rng(0)
    stats = collapse_stats(rng.normal(size=(32, 4)).astype(np.float32))
    assert set(stats) == {"fake_std", "fake_min_pdist", "fake_mean_pdist"}
    for value in stats.values():
        assert type(value) is float  # not np.float32/np.float64
    json.dumps(stats)  # must not raise


def test_collapse_stats_subsamples_deterministically():
    """The default_rng(0) reseed is deliberate: identical indices every call,
    so eval numbers are comparable across steps."""
    rng = np.random.default_rng(7)
    fake = rng.normal(size=(500, 4)).astype(np.float32)
    first = collapse_stats(fake, max_points=64)
    second = collapse_stats(fake, max_points=64)
    assert first == second
    # And it really did subsample rather than use everything.
    assert first != collapse_stats(fake, max_points=500)


def test_collapse_stats_handles_a_single_row():
    stats = collapse_stats(np.ones((1, 4), dtype=np.float32))
    assert stats["fake_min_pdist"] == 0.0
    assert stats["fake_mean_pdist"] == 0.0


def test_ema_decay_of_one_is_rejected(tmp_path):
    """decay >= 1 never accumulates, leaving the zero-init shadow at zero.

    Bias correction cannot rescue it, so eval and best_generator.pt would come
    from an all-zero generator. It must fail loudly at config-read time.
    """
    from src.train.train_wgan_gp import train
    from tests.test_train_smoke import make_config

    config = make_config(tmp_path, "sparse")
    config["training"]["ema_decay"] = 1.0
    with pytest.raises(ValueError, match="ema_decay"):
        train(config)
