"""The scout_kv v2 (book latent) trainer, on the tiny capture of book_latent_fixtures."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from tests.book_latent_fixtures import tiny_config
from torch import nn

import src.train.train_book_latent as tbl
from src.data.kv_capture import h_path, load_kv_books, write_bf16_safetensor
from src.models.book_latent import book_latent_settings
from src.train import train_wgan_gp
from src.train.book_windows import RealBooks, data_mean
from src.train.train_book_latent import (
    generator_loss,
    train,
    validate_config,
    window_gradient_penalty,
)

CKPT_KEYS = {
    "step",
    "generator_type",
    "book_latent",
    "T",
    "generator_state_dict",
    "ema_params",
    "ema_step",
    "ema_decay",
    "token_critic_state_dict",
    "window_critic_state_dict",
    "optim_g_state_dict",
    "optim_token_state_dict",
    "optim_window_state_dict",
    "data_mean",
}
LOG_KEYS = {
    "step",
    "w_dist_token",
    "gp_token",
    "w_dist_window",
    "gp_window",
    "loss_g",
    "grad_token",
    "grad_window",
    "seconds",
}


def _load(path) -> dict:
    return torch.load(path, weights_only=False)


def _log(run_dir) -> list[dict]:
    lines = (Path(run_dir) / "train_log.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _spy(monkeypatch, owner, name, calls):
    """Replace owner.name with a wrapper that records its bound arguments."""
    real = getattr(owner, name)
    sig = inspect.signature(real)

    def wrapper(*args, **kwargs):
        calls.append(sig.bind(*args, **kwargs).arguments)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)


def test_tiny_run_writes_final_checkpoint_and_record(tmp_path):
    """Catches a missing record field: the sampler and resume both depend on these keys."""
    cfg = tiny_config(tmp_path)
    final = train(cfg)
    run = Path(cfg["output_dir"])
    assert final == run / "final.pt" and not (run / "last.pt").exists()
    ckpt = _load(final)
    assert CKPT_KEYS <= set(ckpt)
    assert ckpt["generator_type"] == "book_latent" and ckpt["step"] == 4
    assert ckpt["book_latent"] == book_latent_settings(tiny_config(tmp_path))
    assert ckpt["T"] == 64 and ckpt["ema_decay"] == 0.999
    assert "latent.omega" in ckpt["generator_state_dict"] and ckpt["ema_step"] == 4
    cap = Path(cfg["data"]["real_path"])
    assert ckpt["data_mean"].dtype == torch.float32
    assert ckpt["data_mean"].device.type == "cpu"
    assert torch.allclose(
        ckpt["data_mean"], data_mean(load_kv_books(cap, 3, [0, 1]), 1)
    )
    assert yaml.safe_load((run / "run_config.yaml").read_text()) == cfg
    meta = json.loads((run / "run_metadata.json").read_text())
    assert {"seed", "device", "gpu", "T", "D", "n_books", "seconds"} <= set(meta)
    assert (meta["seed"], meta["T"], meta["D"], meta["n_books"]) == (42, 64, 16, 2)


def test_log_records_both_critics_and_both_gradients(tmp_path):
    """Catches a log missing one critic's distance or penalty, or a grad_* left out or NaN."""
    cfg = tiny_config(tmp_path)
    train(cfg)
    recs = _log(cfg["output_dir"])
    assert [r["step"] for r in recs] == [1, 2, 3, 4]
    for r in recs:
        assert set(r) == LOG_KEYS
        assert all(math.isfinite(v) for v in r.values())
        assert r["grad_token"] > 0 and r["grad_window"] > 0


def test_generator_loss_weights_only_the_window_term():
    """Catches the 0.1 being dropped or applied to the token term (spec test 3)."""
    assert generator_loss(
        torch.tensor([2.0]), torch.tensor([10.0]), 0.1
    ) == pytest.approx(-3.0)
    assert generator_loss(
        torch.tensor([2.0]), torch.tensor([0.0]), 0.1
    ) == pytest.approx(-2.0)


def test_trainer_passes_the_configured_window_weight(tmp_path, monkeypatch):
    """Catches train() calling generator_loss with a literal instead of training.window_weight:
    spy on src.train.train_book_latent.generator_loss; set window_weight 0.37; every call gets 0.37."""
    cfg = tiny_config(tmp_path)
    cfg["training"]["window_weight"] = 0.37
    calls: list = []
    _spy(monkeypatch, tbl, "generator_loss", calls)
    train(cfg)
    assert len(calls) == 4
    assert all(c["window_weight"] == 0.37 for c in calls)


def test_tail_windows_per_batch(tmp_path, monkeypatch):
    """Catches tail_window_share ignored: spy on window_ends; every call asks for 2 of 8 tail windows."""
    cfg = tiny_config(tmp_path)
    calls: list = []
    _spy(monkeypatch, tbl, "window_ends", calls)
    train(cfg)
    # n_critic 2 critic batches plus one generator batch per step, 4 steps.
    assert len(calls) == 12
    assert all((c["n_windows"], c["n_tail_windows"]) == (8, 2) for c in calls)


def test_critics_see_real_and_fake_at_the_same_positions(tmp_path, monkeypatch):
    """Catches real windows drawn at other positions than the fakes: spy on RealBooks.windows and
    fake_windows within one critic iteration; the pos arguments are equal."""
    cfg = tiny_config(tmp_path)
    events: list = []
    real_windows, real_fake = RealBooks.windows, tbl.fake_windows

    def windows(self, pos, generator=None):
        events.append(("real", pos.clone()))
        return real_windows(self, pos, generator)

    def fake(gen, pos, T, n_tail):
        events.append(("fake", pos.clone()))
        return real_fake(gen, pos, T, n_tail)

    monkeypatch.setattr(RealBooks, "windows", windows)
    monkeypatch.setattr(tbl, "fake_windows", fake)
    train(cfg)
    reals = [i for i, (kind, _) in enumerate(events) if kind == "real"]
    assert len(reals) == 8  # n_critic 2 x 4 steps
    for i in reals:
        assert events[i - 1][0] == "fake"
        assert torch.equal(events[i - 1][1], events[i][1])


class _SumCritic(nn.Module):
    def forward(self, x, flags, stride_idx):
        return x.sum((1, 2))


def test_window_gp_is_per_window():
    """Catches a per-token GP on the window critic: a critic x.sum((1, 2)) has gradient all-ones,
    so the per-window norm is sqrt(W*D) and the penalty (sqrt(W*D) - 1)^2; per token it would be
    (sqrt(D) - 1)^2."""
    n, W, D = 3, 8, 16
    real, fake = torch.randn(n, W, D), torch.randn(n, W, D)
    flags = torch.zeros(n, W, dtype=torch.long)
    stride_idx = torch.zeros(n, dtype=torch.long)
    gp = window_gradient_penalty(_SumCritic(), real, fake, flags, stride_idx)
    assert float(gp) == pytest.approx((math.sqrt(W * D) - 1.0) ** 2)


@pytest.mark.parametrize("share", [0.0, 1.0, 0.001])
def test_share_that_empties_a_class_is_refused(tmp_path, share):
    """Catches a tail_window_share that rounds to no tail windows or no body windows (0.001 x 64
    rounds to 0) being trained on instead of refused."""
    cfg = tiny_config(tmp_path)
    cfg["training"]["n_windows"] = 64
    cfg["training"]["tail_window_share"] = share
    with pytest.raises(ValueError, match="tail_window_share"):
        validate_config(cfg)


@pytest.mark.parametrize("key,value", [("output_normalize", True), ("n_flags", 3)])
def test_validate_config_refuses(tmp_path, key, value):
    """Catches validate_config skipping the output_normalize or n_flags check."""
    cfg = tiny_config(tmp_path)
    cfg["model"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


def test_validate_config_refuses_whitening(tmp_path):
    """Catches a preprocess other than centring-only being accepted (the mean is the only
    transform the sampler undoes)."""
    cfg = tiny_config(tmp_path)
    cfg["data"]["preprocess"]["whiten"] = True
    with pytest.raises(ValueError, match="preprocess"):
        validate_config(cfg)


def test_missing_v2_key_is_a_key_error(tmp_path):
    """Catches a silent default for a v2 key: dropping training.window_weight must raise."""
    cfg = tiny_config(tmp_path)
    del cfg["training"]["window_weight"]
    with pytest.raises(KeyError, match="window_weight"):
        validate_config(cfg)


def _train_a(tmp_path) -> Path:
    cfg = tiny_config(tmp_path)
    cfg["output_dir"] = str(tmp_path / "A")
    cfg["training"]["num_gen_steps"] = 2
    return train(cfg)


def _config_b(tmp_path) -> dict:
    cfg = tiny_config(tmp_path)
    cfg["output_dir"] = str(tmp_path / "B")
    cfg["training"]["num_gen_steps"] = 4
    return cfg


@pytest.mark.parametrize(
    "change,message",
    [
        ("channels", "channels"),
        ("strides", "strides"),
        ("seed", "seed"),
        ("kv_books", "kv_books"),
        ("capture", "data mean"),
        ("checkpoint_type", "generator_type"),
    ],
)
def test_resume_that_changes_the_model_is_refused(
    tmp_path, monkeypatch, change, message
):
    """Catches a resume that half-loads, silently re-seeds omega, or trains on other data (spec
    item 2). Train run A with num_gen_steps 2 (final.pt at step 2). Run B: a new output_dir,
    num_gen_steps 4, one thing changed, resume=A/final.pt -> ValueError whose message names the
    change, raised before load_state_dict (spy on torch.nn.Module.load_state_dict: never called).
    Changes: book_latent.channels; window_critic.strides; seed (message names "seed"); kv_books
    [0, 1] -> [0] (names "kv_books"); capture = rewrite book 1 with offset 5
    between A and B, config unchanged (names "data mean"); checkpoint_type = save A's checkpoint
    with "generator_type" removed, as a v1 checkpoint has none (names "generator_type"). Each
    case reaches its own check: the settings comparison cannot fire for "seed" or "capture",
    and only the checkpoint-type check fires for "checkpoint_type"."""
    resume = _train_a(tmp_path)
    cfg = _config_b(tmp_path)
    if change == "channels":
        cfg["model"]["book_latent"]["channels"] = 5
    elif change == "strides":
        cfg["model"]["window_critic"]["strides"] = [1, 2]
    elif change == "seed":
        cfg["seed"] = 43
    elif change == "kv_books":
        cfg["data"]["kv_books"] = [0]
    elif change == "capture":
        p = h_path(Path(cfg["data"]["real_path"]), 1, 3)
        h = load_kv_books(Path(cfg["data"]["real_path"]), 3, [1])[0].float().numpy()
        write_bf16_safetensor(p, "h", (h + 5.0).astype(np.float32))
    elif change == "checkpoint_type":
        ckpt = _load(resume)
        del ckpt["generator_type"]
        resume = tmp_path / "v1_like.pt"
        torch.save(ckpt, resume)
    calls: list = []
    monkeypatch.setattr(
        torch.nn.Module, "load_state_dict", lambda *a, **k: calls.append(a)
    )
    with pytest.raises(ValueError, match=message):
        train(cfg, resume=str(resume))
    assert calls == []


def test_resume_continues_from_the_saved_step(tmp_path, monkeypatch):
    """Catches a resume that restarts the EMA or the step count: as above with nothing changed,
    B's final.pt has step 4 and ema_step 4, B's log starts at step 3, and B's first ema_update
    receives A's saved ema_params (fails if the ema_params restore line is dropped, which leaves
    ema_step at 4 but the accumulator at zero)."""
    resume = _train_a(tmp_path)
    saved = _load(resume)["ema_params"]
    cfg = _config_b(tmp_path)
    seen: list = []
    real = tbl.ema_update

    def spy(ema_params, model, decay):
        seen.append({k: v.detach().clone() for k, v in ema_params.items()})
        return real(ema_params, model, decay)

    monkeypatch.setattr(tbl, "ema_update", spy)
    ckpt = _load(train(cfg, resume=str(resume)))
    assert ckpt["step"] == 4 and ckpt["ema_step"] == 4
    assert [r["step"] for r in _log(cfg["output_dir"])] == [3, 4]
    assert len(seen) == 2 and seen[0].keys() == saved.keys()
    assert all(torch.equal(seen[0][k], saved[k]) for k in saved)


def test_ema_accumulates_every_step(tmp_path, monkeypatch):
    """Catches an EMA that never updates (ema_update a no-op or its call dropped): final.pt's
    ema_params equal the zero-initialised accumulation ema <- d*ema + (1-d)*p of the generator
    parameters seen at each of the 4 steps, and are not the zero init."""
    cfg = tiny_config(tmp_path)
    params: list = []
    real = tbl.ema_update

    def spy(ema_params, model, decay):
        params.append({n: p.detach().clone() for n, p in model.named_parameters()})
        return real(ema_params, model, decay)

    monkeypatch.setattr(tbl, "ema_update", spy)
    ckpt = _load(train(cfg))
    assert len(params) == 4
    d = ckpt["ema_decay"]
    expected = {n: torch.zeros_like(p) for n, p in params[0].items()}
    for step_params in params:
        for n, p in step_params.items():
            expected[n].mul_(d).add_(p, alpha=1.0 - d)
    ema = ckpt["ema_params"]
    assert ema.keys() == expected.keys()
    assert all(torch.allclose(ema[n], expected[n], rtol=0, atol=1e-7) for n in ema)
    assert any(bool(v.abs().sum() > 0) for v in ema.values())


class _Scale(nn.Module):
    """d(x) = w * x with one weight, w = 1."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))

    def forward(self, x):
        return self.w * x


def test_critic_step_minimises_fake_minus_real_plus_penalty():
    """Catches a flipped critic loss (d_real - d_fake + λ·gp) or λ·gp left out: with
    d = w·x, real 2, fake 0, gp = w², λ 0.5 and SGD lr 0.1 from w = 1, the loss
    mean(fake) - mean(real) + λ·gp has gradient -2 + 1 = -1, so w becomes 1.1 (flipped
    sign: 0.7; no penalty: 1.2)."""
    critic = _Scale()
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    real, fake = torch.tensor([2.0]), torch.tensor([0.0])
    w_dist, gp = tbl._critic_step(optim, critic(real), critic(fake), critic.w**2, 0.5)
    assert float(critic.w.detach()) == pytest.approx(1.1)
    assert float(w_dist) == pytest.approx(2.0) and float(gp) == pytest.approx(1.0)


def test_token_critic_gets_a_flagged_penalty_in_its_loss(tmp_path, monkeypatch):
    """Catches the token critic's gradient penalty dropped, zeroed, called without flags or on
    other rows than the n_windows*W tokens: spy on the trainer's gradient_penalty; it is called
    once per critic iteration (2 x 4 steps), with flags of 64 rows, the gradient reaching its
    output in backward is λ_gp = 10 (so λ·gp is in the loss), and the logged gp_token is the
    value of each step's last call."""
    cfg = tiny_config(tmp_path)
    calls: list = []
    real = tbl.gradient_penalty

    def spy(critic, real_x, fake_x, device, flags=None):
        out = real(critic, real_x, fake_x, device, flags=flags)
        rec = {
            "rows": real_x.shape[0],
            "flags": flags,
            "value": float(out.detach()),
            "grad": [],
        }
        out.register_hook(lambda g: rec["grad"].append(float(g)))
        calls.append(rec)
        return out

    monkeypatch.setattr(tbl, "gradient_penalty", spy)
    train(cfg)
    assert len(calls) == 8
    for c in calls:
        assert c["rows"] == 8 * 8
        assert c["flags"] is not None and c["flags"].shape == (64,)
        assert c["grad"] == [pytest.approx(10.0)]
    logged = [r["gp_token"] for r in _log(cfg["output_dir"])]
    assert logged == pytest.approx([calls[2 * i + 1]["value"] for i in range(4)])


def test_window_critic_gets_a_penalty_in_its_loss(tmp_path, monkeypatch):
    """Catches the window critic's gradient penalty dropped or multiplied by 0.0, called on other
    windows than the n_windows batch, or drawn with one alpha per token instead of one per
    window: spy on tbl.window_gradient_penalty (called 2 x 4 times, on (8, 8, 16) windows), the
    gradient reaching its output in backward is lambda_gp = 10, torch.rand inside it is drawn
    with shape (8, 1, 1), and the logged gp_window is each step's last call. The critic is the
    real nonlinear WindowCritic, so the penalty value depends on the alpha draw."""
    cfg = tiny_config(tmp_path)
    calls: list = []
    real = tbl.window_gradient_penalty
    rand = torch.rand

    def spy(critic, real_x, fake_x, flags, stride_idx):
        shapes: list = []

        def rec_rand(*size, **kw):
            shapes.append(tuple(size))
            return rand(*size, **kw)

        with monkeypatch.context() as m:
            m.setattr(torch, "rand", rec_rand)
            out = real(critic, real_x, fake_x, flags, stride_idx)
        rec = {
            "shape": tuple(real_x.shape),
            "fake_shape": tuple(fake_x.shape),
            "rand": shapes,
            "value": float(out.detach()),
            "grad": [],
        }
        out.register_hook(lambda g: rec["grad"].append(float(g)))
        calls.append(rec)
        return out

    monkeypatch.setattr(tbl, "window_gradient_penalty", spy)
    train(cfg)
    assert len(calls) == 8
    for c in calls:
        assert c["shape"] == c["fake_shape"] == (8, 8, 16)
        assert c["rand"] == [(8, 1, 1)]
        assert c["grad"] == [pytest.approx(10.0)]
    logged = [r["gp_window"] for r in _log(cfg["output_dir"])]
    assert logged == pytest.approx([calls[2 * i + 1]["value"] for i in range(4)])


def test_each_critic_is_stepped_with_its_own_real_and_fake_scores(
    tmp_path, monkeypatch
):
    """Catches d_real and d_fake swapped for either critic, a critic scored on the other
    critic's inputs, and the window critic stepped by the token critic's optimizer (or the
    reverse): spy on tbl._critic_step; at each call recompute the critic's output on the tensor
    RealBooks.windows returned and on the fake_windows output, and require d_real and d_fake to
    equal them (and to differ from each other), and the optimizer to own that critic's
    parameters."""
    cfg = tiny_config(tmp_path)
    latest: dict = {}
    critics: dict = {}
    steps: list = []
    real_windows, real_fake, real_build = (
        RealBooks.windows,
        tbl.fake_windows,
        tbl.build_book_latent,
    )
    real_step = tbl._critic_step

    def windows(self, pos, generator=None):
        latest["real"] = real_windows(self, pos, generator)
        return latest["real"]

    def fake(gen, pos, T, n_tail):
        latest["fake"], latest["flags"] = real_fake(gen, pos, T, n_tail)
        return latest["fake"], latest["flags"]

    def build(config):
        gen, token, window = real_build(config)
        critics["token"], critics["window"] = token, window
        return gen, token, window

    def score(name, x):
        flags = latest["flags"]
        with torch.no_grad():
            if name == "token":
                return critics[name](x.reshape(-1, x.shape[-1]), flags.reshape(-1))
            return critics[name](x, flags, tbl_stride[0])

    tbl_stride: list = []
    real_sample = tbl.sample_stride

    def sample(*a, **k):
        tbl_stride.clear()
        tbl_stride.append(real_sample(*a, **k))
        return tbl_stride[0]

    def step(optim, d_real, d_fake, gp, lambda_gp):
        owner = [
            n
            for n, c in critics.items()
            if {id(p) for p in c.parameters()}
            == {id(p) for g in optim.param_groups for p in g["params"]}
        ]
        assert len(owner) == 1
        name = owner[0]
        steps.append(name)
        assert torch.equal(d_real, score(name, latest["real"]))
        assert torch.equal(d_fake, score(name, latest["fake"]))
        assert not torch.allclose(d_real, d_fake)
        return real_step(optim, d_real, d_fake, gp, lambda_gp)

    monkeypatch.setattr(RealBooks, "windows", windows)
    monkeypatch.setattr(tbl, "fake_windows", fake)
    monkeypatch.setattr(tbl, "build_book_latent", build)
    monkeypatch.setattr(tbl, "sample_stride", sample)
    monkeypatch.setattr(tbl, "_critic_step", step)
    train(cfg)
    assert steps == ["token", "window"] * 8


def test_fresh_run_into_a_crashed_run_dir_restarts_the_log(tmp_path):
    """Catches the train log opened for append on a fresh run: a run whose final.pt is missing
    (crashed) and is retrained into the same dir would log steps [1, 2, 3, 4, 1, 2, 3, 4];
    it must log [1, 2, 3, 4] once."""
    cfg = tiny_config(tmp_path)
    final = train(cfg)
    assert [r["step"] for r in _log(cfg["output_dir"])] == [1, 2, 3, 4]
    final.unlink()
    train(cfg)
    assert [r["step"] for r in _log(cfg["output_dir"])] == [1, 2, 3, 4]


def test_resume_into_the_same_dir_appends_to_the_log(tmp_path):
    """Catches a resume that truncates the log it is continuing: a 4-step run is resumed from its
    final.pt, in the same dir, to 6 steps; the log must hold steps 1-6, not 5, 6."""
    cfg = tiny_config(tmp_path)
    final = train(cfg)
    assert _load(final)["step"] == 4
    cfg["training"]["num_gen_steps"] = 6
    final.rename(final.with_name("resume_from.pt"))
    train(cfg, resume=str(final.with_name("resume_from.pt")))
    assert [r["step"] for r in _log(cfg["output_dir"])] == [1, 2, 3, 4, 5, 6]


def test_resume_at_the_budget_is_refused(tmp_path):
    """Catches a resume from a checkpoint already at num_gen_steps running zero steps and
    renaming nothing: it must be refused, naming num_gen_steps."""
    resume = _train_a(tmp_path)
    cfg = _config_b(tmp_path)
    cfg["training"]["num_gen_steps"] = 2
    with pytest.raises(ValueError, match="num_gen_steps"):
        train(cfg, resume=str(resume))


def test_existing_final_checkpoint_is_refused(tmp_path):
    """Catches a second train() into a finished run dir overwriting its final.pt."""
    cfg = tiny_config(tmp_path)
    train(cfg)
    with pytest.raises(FileExistsError, match="final.pt"):
        train(cfg)


def test_train_wgan_gp_refuses_a_book_latent_config(tmp_path, monkeypatch):
    """Catches the v0/v1 trainer loading 4 GB of data before refusing: spy on build_training_data,
    never called."""
    calls: list = []
    monkeypatch.setattr(
        train_wgan_gp, "build_training_data", lambda *a, **k: calls.append(a)
    )
    with pytest.raises(ValueError, match="src.train.train_book_latent"):
        train_wgan_gp.train(tiny_config(tmp_path))
    assert calls == []


def test_same_seed_same_run(tmp_path):
    """Catches an unseeded draw (window ends, strides, latent coefficients, book choice): two
    tiny CPU runs with one seed in two dirs end with equal generator weights."""
    finals = []
    for name in ("a", "b"):
        cfg = tiny_config(tmp_path)
        cfg["output_dir"] = str(tmp_path / name)
        finals.append(_load(train(cfg)))
    a, b = (f["generator_state_dict"] for f in finals)
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert all(
        torch.equal(finals[0]["ema_params"][k], finals[1]["ema_params"][k])
        for k in finals[0]["ema_params"]
    )
