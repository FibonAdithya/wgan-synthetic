"""Sampling scout_kv v2 (book latent) runs into capture-format books."""

from __future__ import annotations

import json

import pytest
import torch
import yaml
from tests.book_latent_fixtures import tiny_config, write_tiny_capture
from torch import nn

from src.data.kv_capture import h_path
from src.models.book_latent import DriftLatent
from src.sample.sample_books import (
    load_book_latent_run,
    main,
    sample_books,
    write_books,
)
from src.train.train_book_latent import train
from src.train.train_wgan_gp import ema_bias_correction


def _run(tmp_path):
    cfg = tiny_config(tmp_path)
    final = train(cfg)
    return cfg, final.parent


def _sample(tmp_path, **kw):
    cfg, run = _run(tmp_path)
    gen, mean, _ = load_book_latent_run(run)
    args = dict(n_books=3, T=64, n_sink=1, n_tail=16, seed=0)
    args.update(kw)
    return gen, mean, sample_books(gen, mean, **args)


def test_sampled_sink_is_the_mean(tmp_path):
    """Catches position 0 being left at zero (or generated) instead of the mean."""
    _, mean, h = _sample(tmp_path)
    assert h.shape == (3, 64, 16)
    for b in range(3):
        assert torch.equal(h[b, 0], mean)


def test_books_differ_and_seed_repeats(tmp_path):
    """Catches all books sharing one latent draw, or an unseeded sampler: books 0 and 1
    differ; a second call with the same seed is identical; seed 1 differs."""
    gen, mean, h = _sample(tmp_path)
    kw = dict(n_books=3, T=64, n_sink=1, n_tail=16)
    assert not torch.allclose(h[0, 1:], h[1, 1:])
    assert torch.equal(h, sample_books(gen, mean, seed=0, **kw))
    assert not torch.allclose(h, sample_books(gen, mean, seed=1, **kw))


def test_coefficients_are_drawn_once_per_call(tmp_path, monkeypatch):
    """Catches a book's latent being redrawn per chunk, which would break the book at
    every chunk edge: T 64, chunk 8 -> sample_coefficients called once, with n = n_books."""
    cfg, run = _run(tmp_path)
    gen, mean, _ = load_book_latent_run(run)
    calls = []
    orig = DriftLatent.sample_coefficients

    def spy(self, n, device, generator=None):
        calls.append(n)
        return orig(self, n, device, generator)

    monkeypatch.setattr(DriftLatent, "sample_coefficients", spy)
    sample_books(gen, mean, n_books=3, T=64, n_sink=1, n_tail=16, seed=0, chunk=8)
    assert calls == [3]


class _PosLatent(nn.Module):
    def sample_coefficients(self, n, device, generator=None):
        return torch.zeros(n, 1)

    def forward(self, coef, pos):
        return pos.float()[..., None]


class _StubGen(nn.Module):
    latent_dim = 2

    def __init__(self, D):
        super().__init__()
        self.D = D
        self.latent = _PosLatent()

    def forward(self, z, u, flags):
        return u.expand(*u.shape[:-1], self.D)


def test_chunking_covers_every_position_once():
    """Catches an off-by-one at chunk edges (a repeated or skipped position): T 64,
    chunk 10, a stub whose latent returns each token's position; h[:, n_sink:] - mean
    equals positions 1..63 exactly."""
    mean = torch.arange(4, dtype=torch.float32)
    h = sample_books(
        _StubGen(4), mean, n_books=2, T=64, n_sink=1, n_tail=16, seed=0, chunk=10
    )
    want = torch.arange(1, 64, dtype=torch.float32)[None, :, None].expand(2, 63, 4)
    assert torch.equal(h[:, 1:] - mean, want)


def test_flags_follow_the_tail_boundary():
    """Catches the sampler flagging tokens by chunk instead of position, or with the
    wrong boundary: the stub echoes its flags, so the tail starts exactly at T - n_tail."""

    class Echo(_StubGen):
        def forward(self, z, u, flags):
            return flags.float()[..., None].expand(*flags.shape, self.D)

    h = sample_books(
        Echo(2), torch.zeros(2), n_books=1, T=64, n_sink=1, n_tail=16, seed=0, chunk=7
    )
    flags = h[0, :, 0]
    assert flags[1:48].sum() == 0 and flags[48:].min() == 1


def test_write_books_refuses_an_existing_dir(tmp_path):
    """Catches overwriting or appending to an existing capture dir."""
    cap = tmp_path / "cap"
    write_tiny_capture(cap, [0], T=8)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        write_books(torch.zeros(1, 8, 16), out, 3, [12], cap)


def test_sampler_uses_ema_weights(tmp_path):
    """Catches sampling from the live weights (v2 is judged on the final EMA weights):
    every parameter equals ema_params[name] * ema_bias_correction(decay, ema_step), at
    least one differs from the live state dict, and latent.omega is the checkpoint's."""
    cfg, run = _run(tmp_path)
    ckpt = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    gen, mean, _ = load_book_latent_run(run)
    corr = ema_bias_correction(ckpt["ema_decay"], ckpt["ema_step"])
    differs = False
    for name, p in gen.named_parameters():
        want = ckpt["ema_params"][name] * corr
        assert torch.allclose(p, want, atol=1e-6), name
        differs |= not torch.allclose(p, ckpt["generator_state_dict"][name])
    assert differs
    assert torch.equal(gen.latent.omega, ckpt["generator_state_dict"]["latent.omega"])
    assert torch.equal(mean, ckpt["data_mean"])


def test_load_refuses_a_non_book_latent_run(tmp_path):
    """Catches loading a v0/v1 run dir as a book-latent one."""
    cfg, run = _run(tmp_path)
    rc = yaml.safe_load((run / "run_config.yaml").read_text())
    rc["model"]["generator_type"] = "mlp"
    (run / "run_config.yaml").write_text(yaml.safe_dump(rc))
    with pytest.raises(ValueError, match="book_latent"):
        load_book_latent_run(run)


def test_load_refuses_a_checkpoint_from_another_config(tmp_path):
    """Catches a checkpoint paired with an edited run_config (shape or seed drift)."""
    cfg, run = _run(tmp_path)
    rc = yaml.safe_load((run / "run_config.yaml").read_text())
    rc["model"]["book_latent"]["channels"] = 6
    (run / "run_config.yaml").write_text(yaml.safe_dump(rc))
    with pytest.raises(ValueError, match="book_latent"):
        load_book_latent_run(run)


def test_end_to_end_tiny_train_sample_measure(tmp_path):
    """Catches a capture the gate cannot read (wrong path layout, missing weights link,
    wrong T). T 400 because profile_corpus takes k=128 ground truth: train, sample 2
    books as ids 12-13, measure-real them: 2 records (one per KV head), source v2,
    n_keys == 2 * ((400 - 16 - 1) // 2) == 382."""
    from src.eval import kv_gate

    cfg = tiny_config(tmp_path)
    cap = tmp_path / "cap400"
    write_tiny_capture(cap, [0, 1], layer=3, T=400, D=16)
    cfg["data"]["real_path"] = str(cap)
    run = train(cfg).parent
    out = run / "capture"
    rc = main(
        [
            "--run-dir", str(run),
            "--capture-dir", str(cap),
            "--out-dir", str(out),
            "--book-ids", "12-13",
        ]
    )  # fmt: skip
    assert rc == 0
    assert h_path(out, 12, 3).exists() and (out / "weights").is_symlink()
    recs_path = tmp_path / "recs.jsonl"
    kv_gate.measure_real(
        out, books=[12, 13], layers=[3], seed=0, out=recs_path, n_sink=1, n_tail=16,
        nlist=4, hnsw_m=8, ef_construction=16, threads=1, source="v2",
    )  # fmt: skip
    recs = [json.loads(line) for line in recs_path.read_text().splitlines()]
    assert len(recs) == 2
    assert {r["source"] for r in recs} == {"v2"}
    assert all(r["n_keys"] == 382 for r in recs)
