"""Flag plumbing in the trainer's helpers and the body/tail batch stream."""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.conditional import BODY, TAIL, FlagConditionedGenerator
from src.train.flag_batches import FlagBatchStream
from src.train.train_wgan_gp import (
    gradient_penalty,
    sample_generator,
    save_checkpoint,
)

CPU = torch.device("cpu")


def _data(n_body=200, n_tail=40, dim=4):
    """x[:, 0] encodes the row's flag, so a mismatch is visible."""
    flags = np.array([BODY] * n_body + [TAIL] * n_tail, dtype=np.int64)
    x = np.random.default_rng(0).normal(size=(len(flags), dim)).astype(np.float32)
    x[:, 0] = flags
    return x, flags


def _stream(x, f, **kw):
    args = dict(batch_size=32, tail_share=0.25, seed=0, num_workers=0, pin_memory=False)
    args.update(kw)
    return FlagBatchStream(x, f, **args)


def test_flag_batches_imports_on_its_own():
    """Catches a module-level import cycle with train_wgan_gp in either order."""
    for first in ("src.train.flag_batches", "src.train.train_wgan_gp"):
        code = (
            f"import {first}; import src.train.flag_batches; "
            "import src.train.train_wgan_gp"
        )
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[1],
        )
        assert r.returncode == 0, r.stderr


def test_stream_batches_have_the_configured_mix():
    """Catches the tail share being ignored: batch 32, share 0.25 -> 24 body then 8 tail."""
    x, f = _data()
    rows, flags = _stream(x, f).next()
    assert rows.shape[0] == 32
    assert flags.tolist() == [BODY] * 24 + [TAIL] * 8
    assert rows.dtype == torch.float32 and flags.dtype == torch.long


def test_stream_rows_carry_their_own_flags():
    """Catches rows and flags drawn from different permutations: x[:, 0] encodes the flag."""
    x, f = _data()
    s = _stream(x, f)
    for _ in range(10):
        rows, flags = s.next()
        assert torch.equal(rows[:, 0].long(), flags)


def test_stream_survives_epoch_end():
    """Catches StopIteration escaping: 20 next() calls on 40 tail rows at 8 per batch."""
    x, f = _data(n_body=400, n_tail=40)
    s = _stream(x, f)
    for _ in range(20):
        rows, flags = s.next()
        assert rows.shape[0] == 32
        assert int((flags == TAIL).sum()) == 8


def test_stream_is_reproducible_and_seed_sensitive():
    """Catches a loader ignoring the seed (or both loaders sharing one seed)."""
    x, f = _data()
    a = _stream(x, f, seed=3).next()[0]
    b = _stream(x, f, seed=3).next()[0]
    c = _stream(x, f, seed=4).next()[0]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize("share", [0.0, 1.0, 0.001])
def test_stream_rejects_a_share_that_empties_a_class(share):
    """Catches a share that rounds to zero tail or zero body rows being accepted."""
    x, f = _data()
    with pytest.raises(ValueError):
        _stream(x, f, tail_share=share)


def test_stream_rejects_a_class_smaller_than_its_batch():
    """Catches a class with fewer rows than its batch size (drop_last would yield nothing)."""
    x, f = _data(n_body=200, n_tail=5)
    with pytest.raises(ValueError):
        _stream(x, f)
    x, f = _data(n_body=20, n_tail=40)
    with pytest.raises(ValueError):
        _stream(x, f)


def test_gradient_penalty_passes_flags_to_the_critic():
    """Catches the GP scoring interpolates unconditioned: the critic is
    x.sum * 1[TAIL], so the gradient norm is sqrt(d) for TAIL rows and 0 for BODY
    rows: penalty (sqrt(d) - 1)^2 for all-TAIL, 1 for all-BODY. The output stays a
    function of x for BODY rows (a constant 0 makes autograd.grad raise)."""
    d = 9

    def critic(x, flags):
        return x.sum(1, keepdim=True) * (flags == TAIL).unsqueeze(1).to(x.dtype)

    real, fake = torch.randn(16, d), torch.randn(16, d)
    tail = gradient_penalty(critic, real, fake, CPU, flags=torch.full((16,), TAIL))
    body = gradient_penalty(critic, real, fake, CPU, flags=torch.full((16,), BODY))
    assert tail.item() == pytest.approx((math.sqrt(d) - 1) ** 2, rel=1e-5)
    assert body.item() == pytest.approx(1.0, rel=1e-5)


def test_gradient_penalty_without_flags_is_unchanged():
    """Catches a flagless call being routed through the two-argument critic call."""
    real, fake = torch.randn(8, 4), torch.randn(8, 4)
    pen = gradient_penalty(lambda x: x.sum(1, keepdim=True), real, fake, CPU)
    assert pen.item() == pytest.approx((math.sqrt(4) - 1) ** 2, rel=1e-5)


class _Lin(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = torch.nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.lin(x)


def _flagged_gen():
    torch.manual_seed(0)
    return FlagConditionedGenerator(_Lin(4 + 2, 3), 2)


def test_sample_generator_uses_the_requested_flag():
    """Catches the flag being ignored: same seed, flag BODY vs TAIL -> different
    samples; flag BODY twice -> identical."""
    g = _flagged_gen()

    def run(flag):
        torch.manual_seed(5)
        return sample_generator(g, 10, 4, 4, CPU, normalize=False, flag=flag)

    assert not np.allclose(run(BODY), run(TAIL))
    np.testing.assert_array_equal(run(BODY), run(BODY))


def test_sample_generator_requires_flag_for_a_flagged_generator():
    """Catches a flagged generator being sampled with no flag."""
    with pytest.raises(ValueError):
        sample_generator(_flagged_gen(), 4, 4, 4, CPU)


def test_sample_generator_rejects_flag_for_an_unflagged_generator():
    """Catches a flag silently ignored by an unflagged generator."""
    with pytest.raises(ValueError):
        sample_generator(_Lin(4, 3), 4, 4, 4, CPU, flag=BODY)


def test_sample_generator_unflagged_randn_sequence_is_unchanged():
    """Catches an extra torch.randn call on the unflagged path."""
    g = _Lin(4, 3)
    torch.manual_seed(1)
    got = sample_generator(g, 6, 4, 3, CPU, normalize=False)
    torch.manual_seed(1)
    with torch.no_grad():
        z = torch.cat([torch.randn(3, 4), torch.randn(3, 4)])
        want = g(z).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_checkpoint_records_n_flags(tmp_path):
    """Catches n_flags not being written (or defaulting to something other than None)."""
    g, c = _Lin(2, 2), _Lin(2, 2)
    og = torch.optim.Adam(g.parameters())
    od = torch.optim.Adam(c.parameters())
    save_checkpoint(g, c, og, od, tmp_path, step=1, n_flags=2)
    save_checkpoint(g, c, og, od, tmp_path, step=2)
    assert (
        torch.load(tmp_path / "checkpoint_step_1.pt", weights_only=False)["n_flags"]
        == 2
    )
    assert (
        torch.load(tmp_path / "checkpoint_step_2.pt", weights_only=False)["n_flags"]
        is None
    )
