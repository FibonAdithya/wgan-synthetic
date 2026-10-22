"""train() with a body/tail flag (model.n_flags, data.kv_n_tail).

The fixture is a tiny capture: 2 books x T=64 x 16 dims at layer 3, with
kv_n_sink 1, kv_n_tail 8 and kv_tail_holdout 4. That gives 16 tail rows (12
train, 4 holdout) and 110 body rows (88 train, 22 holdout). A batch of 16 at
tail_batch_share 0.25 holds 12 body rows then 4 tail rows.
"""

import math

import numpy as np
import pytest
import torch
from tests.test_train_smoke import make_config

import src.train.train_wgan_gp as train_mod
from src.data.kv_capture import h_path, write_bf16_safetensor
from src.models.conditional import BODY, TAIL, FlagConditionedGenerator
from src.train.train_wgan_gp import train


def _write_capture(root):
    rng = np.random.default_rng(0)
    for book in (0, 1):
        p = h_path(root, book, 3)
        p.parent.mkdir(parents=True, exist_ok=True)
        write_bf16_safetensor(p, "h", rng.normal(size=(64, 16)).astype(np.float32))


def flagged_config(tmp_path, name="flagged"):
    root = tmp_path / "capture"
    if not root.exists():
        _write_capture(root)
    cfg = make_config(tmp_path, "mlp")
    cfg["output_dir"] = str(tmp_path / name)
    cfg["data"] = {
        "real_path": str(root),
        "format": "kv_capture",
        "descriptor_dim": 16,
        "holdout_fraction": 0.2,
        "synthetic_if_missing": False,
        "synthetic_num_vectors": 256,
        "preprocess": {"center": True, "whiten": False, "l2_normalize": False},
        "kv_layer": 3,
        "kv_books": [0, 1],
        "kv_n_sink": 1,
        "kv_n_tail": 8,
        "kv_tail_holdout": 4,
    }
    cfg["model"]["n_flags"] = 2
    cfg["model"]["output_normalize"] = False
    cfg["training"].update(
        tail_batch_share=0.25, batch_size=16, num_gen_steps=4, select_on="gate"
    )
    return cfg


def unflagged_config(tmp_path, name="unflagged"):
    """The same capture and training settings with the flag removed."""
    cfg = flagged_config(tmp_path, name)
    del cfg["data"]["kv_n_tail"], cfg["data"]["kv_tail_holdout"]
    del cfg["model"]["n_flags"], cfg["training"]["tail_batch_share"]
    return cfg


def test_flagged_run_trains_and_records_both_halves(tmp_path):
    """Catches the tail never reaching eval/selection: selection_score computed from
    the body alone fails the sum check, and a missing tail half drops its keys."""
    ckpt, meta = train(flagged_config(tmp_path))
    ev = meta["eval"][-1]
    assert {"selection_score_body", "selection_score_tail", "tail_cov_fro"} <= ev.keys()
    assert {"tail_gate_fake_lid_median", "tail_gate_real_lid_median"} <= ev.keys()
    assert math.isfinite(ev["selection_score_tail"])
    assert ev["selection_score"] == pytest.approx(
        ev["selection_score_body"] + ev["selection_score_tail"]
    )
    assert torch.load(ckpt, weights_only=False)["n_flags"] == 2
    assert meta["data"]["num_holdout_tail"] == 4
    assert meta["data"]["num_train_tail"] == 12
    assert meta["data"]["num_holdout"] == 22 + 4


def test_flagged_run_logs_per_flag_wasserstein(tmp_path):
    """Catches the per-flag gaps missing from a log line, computed on the whole batch
    (body == tail), or on swapped masks. The batch mean is linear in the rows, so the
    overall gap is 12/16 of the body gap plus 4/16 of the tail gap."""
    _, meta = train(flagged_config(tmp_path))
    assert len(meta["metrics"]) == 4
    for line in meta["metrics"]:
        wb, wt = line["wasserstein_body"], line["wasserstein_tail"]
        assert math.isfinite(wb) and math.isfinite(wt)
        assert wb != wt
        assert line["wasserstein"] == pytest.approx(
            0.75 * wb + 0.25 * wt, rel=1e-4, abs=1e-6
        )


@pytest.mark.parametrize("half", ["kv_n_tail_only", "n_flags_only"])
def test_half_configured_flag_is_refused(tmp_path, half):
    """Catches a dead gate: kv_n_tail without n_flags (or the reverse) silently trains
    unconditioned, or fails later with an unpacking error that names neither key."""
    cfg = flagged_config(tmp_path)
    if half == "kv_n_tail_only":
        del cfg["model"]["n_flags"]
    else:
        del cfg["data"]["kv_n_tail"], cfg["data"]["kv_tail_holdout"]
    with pytest.raises(
        ValueError, match=r"(?s)(kv_n_tail.*n_flags|n_flags.*kv_n_tail)"
    ):
        train(cfg)


def test_flagged_run_without_tail_batch_share_is_refused(tmp_path):
    """Catches a default tail share being invented when the key is missing."""
    cfg = flagged_config(tmp_path)
    del cfg["training"]["tail_batch_share"]
    with pytest.raises(ValueError, match="tail_batch_share"):
        train(cfg)


def test_flags_with_cov_fro_selection_is_refused(tmp_path):
    """Catches a flagged run accepting select_on cov_fro, which would select on the
    body covariance alone and ignore the tail."""
    cfg = flagged_config(tmp_path)
    cfg["training"]["select_on"] = "cov_fro"
    with pytest.raises(ValueError, match="cov_fro"):
        train(cfg)


@pytest.mark.parametrize("direction", ["v0_into_v1", "v1_into_v0"])
def test_resume_across_conditioning_is_refused(tmp_path, direction):
    """Catches a resume that would fail deep in load_state_dict or, worse, half-load.
    The resuming config must have num_gen_steps above the checkpoint's step and the same
    select_on, or an earlier refusal fires first and the n_flags check is never reached.
    The v0 checkpoint has its n_flags key deleted, as the real v0 checkpoints lack it,
    so this also catches a missing key read as anything other than unconditioned."""
    first, second = (
        (unflagged_config, flagged_config)
        if direction == "v0_into_v1"
        else (flagged_config, unflagged_config)
    )
    cfg = first(tmp_path, "first")
    cfg["training"]["save_every"] = 2
    train(cfg)
    live = tmp_path / "first" / "checkpoint_step_2.pt"
    if direction == "v0_into_v1":
        ckpt = torch.load(live, weights_only=False)
        del ckpt["n_flags"]
        torch.save(ckpt, live)
    cfg2 = second(tmp_path, "second")
    cfg2["training"]["num_gen_steps"] = 8
    with pytest.raises(ValueError, match="n_flags"):
        train(cfg2, resume=str(live))


def test_resume_without_n_flags_key_into_unflagged_config_works(tmp_path):
    """Catches a missing n_flags key being refused outright: the v0 checkpoints on the
    box predate the key and must stay resumable under a v0 config."""
    cfg = unflagged_config(tmp_path, "first")
    cfg["training"]["save_every"] = 2
    train(cfg)
    live = tmp_path / "first" / "checkpoint_step_2.pt"
    ckpt = torch.load(live, weights_only=False)
    del ckpt["n_flags"]
    torch.save(ckpt, live)
    train(unflagged_config(tmp_path, "second"), resume=str(live))


def test_flagged_resume_continues(tmp_path):
    """Catches the n_flags check refusing a matching flagged checkpoint."""
    cfg = flagged_config(tmp_path, "first")
    cfg["training"]["save_every"] = 2
    train(cfg)
    cfg2 = flagged_config(tmp_path, "second")
    _, meta = train(cfg2, resume=str(tmp_path / "first" / "checkpoint_step_2.pt"))
    assert meta["resumed_from_step"] == 2
    assert [m["step"] for m in meta["metrics"]] == [3, 4]


def test_critic_sees_matching_flags_for_real_and_fake(tmp_path, monkeypatch):
    """Catches fakes generated under a different flag vector than the real batch.
    Spy on three things: FlagConditionedGenerator.forward (the flags each fake batch was
    generated with), src.train.train_wgan_gp.score_population (flags per 'real'/'fake'
    call) and src.train.train_wgan_gp.gradient_penalty (its flags). Within each critic
    step, the generator's flags, the 'real' call's flags, the 'fake' call's flags and the
    GP's flags are all equal. Spying on the critic alone would not catch
    generator(z, zeros_like(flags)): the fakes would still be *scored* with the real flags."""
    cfg = flagged_config(tmp_path)
    events = _record_flag_events(monkeypatch, cfg)

    real_idx = [i for i, (kind, _) in enumerate(events) if kind == "real"]
    assert (
        len(real_idx) == cfg["training"]["num_gen_steps"] * cfg["training"]["n_critic"]
    )
    for i in real_idx:
        kinds = [kind for kind, _ in events[i - 1 : i + 3]]
        assert kinds == ["gen", "real", "fake", "gp"]
        flags = events[i][1]
        assert (flags == BODY).sum() == 12 and (flags == TAIL).sum() == 4
        for _, other in events[i - 1 : i + 3]:
            assert torch.equal(other, flags)


def test_generator_step_generates_under_the_batch_flags(tmp_path, monkeypatch):
    """Catches the generator step generating under other flags than it scores with,
    e.g. generator(z, zeros_like(gen_flags)). A generator step is a 'gen' call followed
    directly by a 'fake' score with no 'real' between; eval sampling has no score."""
    cfg = flagged_config(tmp_path)
    events = _record_flag_events(monkeypatch, cfg)
    gen_steps = [
        i
        for i in range(1, len(events))
        if events[i][0] == "fake" and events[i - 1][0] == "gen"
    ]
    assert len(gen_steps) == cfg["training"]["num_gen_steps"]
    for i in gen_steps:
        flags = events[i][1]
        assert (flags == BODY).sum() == 12 and (flags == TAIL).sum() == 4
        assert torch.equal(events[i - 1][1], flags)


def _record_flag_events(monkeypatch, cfg):
    """Train `cfg` and return `(kind, flags)` for every generator forward, critic score
    and gradient penalty, in call order. kind is 'gen', 'real', 'fake' or 'gp'."""
    events = []
    real_forward = FlagConditionedGenerator.forward
    real_score = train_mod.score_population
    real_gp = train_mod.gradient_penalty

    def gen_spy(self, z, flags):
        events.append(("gen", flags.detach().cpu().clone()))
        return real_forward(self, z, flags)

    def score_spy(critic, x, population, row_ids=None, flags=None):
        events.append((population, None if flags is None else flags.cpu().clone()))
        return real_score(critic, x, population, row_ids=row_ids, flags=flags)

    def gp_spy(critic, real, fake, device, flags=None):
        events.append(("gp", None if flags is None else flags.cpu().clone()))
        return real_gp(critic, real, fake, device, flags=flags)

    monkeypatch.setattr(FlagConditionedGenerator, "forward", gen_spy)
    monkeypatch.setattr(train_mod, "score_population", score_spy)
    monkeypatch.setattr(train_mod, "gradient_penalty", gp_spy)
    train(cfg)
    return events


@pytest.mark.parametrize("key", ["kv_tail_holdout", "tail_batch_share"])
def test_flag_only_keys_without_n_flags_are_refused(tmp_path, key):
    """Catches a dead config key: kv_tail_holdout or tail_batch_share left in an
    unflagged config would be ignored and the run would train unconditioned."""
    cfg = unflagged_config(tmp_path)
    section = "data" if key == "kv_tail_holdout" else "training"
    cfg[section][key] = 4 if key == "kv_tail_holdout" else 0.25
    with pytest.raises(ValueError, match=key):
        train(cfg)


def test_tail_fake_gate_failure_is_contained(tmp_path, monkeypatch):
    """Catches an unguarded tail fake-side gate call taking the run down, or its error
    landing under the body's key. gate_statistics is called for the real body, the real
    tail, then the body and tail fakes of each eval; the 4th call (first eval's tail
    fake) fails."""
    real_gate_statistics = train_mod.gate_statistics
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("boom")
        return real_gate_statistics(*args, **kwargs)

    monkeypatch.setattr(train_mod, "gate_statistics", flaky)
    _, meta = train(flagged_config(tmp_path))
    first, second = meta["eval"]
    assert first["tail_gate_error"] == "RuntimeError: boom"
    assert "gate_error" not in first
    assert first["tail_gate_fake_lid_median"] is None
    assert first["selection_score_tail"] == math.inf
    assert math.isfinite(first["selection_score_body"])
    assert first["selection_score"] == math.inf
    assert "tail_gate_error" not in second
    assert math.isfinite(second["selection_score"])
