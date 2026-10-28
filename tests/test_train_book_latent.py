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


def test_resume_continues_from_the_saved_step(tmp_path):
    """Catches a resume that restarts the EMA or the step count: as above with nothing changed,
    B's final.pt has step 4 and ema_step 4, and B's log starts at step 3."""
    resume = _train_a(tmp_path)
    cfg = _config_b(tmp_path)
    ckpt = _load(train(cfg, resume=str(resume)))
    assert ckpt["step"] == 4 and ckpt["ema_step"] == 4
    assert [r["step"] for r in _log(cfg["output_dir"])] == [3, 4]


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
