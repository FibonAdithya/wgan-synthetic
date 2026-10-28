"""Whole-book data and window positions for the scout_kv v2 trainer.

Random numbers are drawn on the CPU (from `generator` when given, else the global CPU
RNG) and moved to the target device, because a CPU `torch.Generator` cannot draw into
cuda tensors.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.models.conditional import BODY, TAIL


def data_mean(h: torch.Tensor, n_sink: int) -> torch.Tensor:
    """Float32 (D,) mean over books and positions n_sink ... T-1, accumulated book by book."""
    total = torch.zeros(h.shape[-1], dtype=torch.float32, device=h.device)
    for book in h:
        total += book[n_sink:].float().mean(0)
    return total / h.shape[0]


def tail_flags(pos: torch.Tensor, T: int, n_tail: int) -> torch.Tensor:
    """Long flags: TAIL where pos >= T - n_tail, else BODY."""
    return torch.where(pos >= T - n_tail, TAIL, BODY).long()


def _uniform_int(lo: int, hi: int, n: int, generator) -> torch.Tensor:
    """n integers uniform on [lo, hi] inclusive, drawn on the CPU."""
    return torch.randint(lo, hi + 1, (n,), generator=generator)


def window_ends(
    n_windows: int,
    n_tail_windows: int,
    T: int,
    n_sink: int,
    n_tail: int,
    W: int,
    device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(t_end, lo) per window: body windows first, then n_tail_windows tail windows.

    Body: t_end uniform in [n_sink + W - 1, T - n_tail - 1], lo = n_sink.
    Tail: t_end uniform in [T - n_tail + W - 1, T - 1], lo = T - n_tail.
    """
    body = (n_sink + W - 1, T - n_tail - 1)
    tail = (T - n_tail + W - 1, T - 1)
    if body[0] > body[1]:
        raise ValueError(
            f"book of {T} positions is too short for n_sink={n_sink}, n_tail={n_tail}, W={W}"
        )
    if tail[0] > tail[1]:
        raise ValueError(f"n_tail={n_tail} is shorter than a window of W={W} tokens")
    n_body = n_windows - n_tail_windows
    t_end = torch.cat(
        [
            _uniform_int(*body, n_body, generator),
            _uniform_int(*tail, n_tail_windows, generator),
        ]
    )
    lo = torch.cat(
        [
            torch.full((n_body,), n_sink, dtype=torch.long),
            torch.full((n_tail_windows,), T - n_tail, dtype=torch.long),
        ]
    )
    return t_end.to(device), lo.to(device)


def sample_stride(
    t_end: torch.Tensor,
    lo: torch.Tensor,
    strides: Sequence[int],
    W: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Stride index per window, uniform over the strides s with t_end - (W-1)*s >= lo.

    Strides are strictly increasing, so a stride that fits implies every smaller one
    fits and the r-th fitting stride is index r.
    """
    s = list(strides)
    if any(b <= a for a, b in zip(s, s[1:], strict=False)):
        raise ValueError(f"strides must be strictly increasing, got {s}")
    st = torch.tensor(s, dtype=torch.long)
    t, floor = t_end.cpu(), lo.cpu()
    n_ok = ((t[:, None] - (W - 1) * st[None, :]) >= floor[:, None]).sum(1)
    if (n_ok == 0).any():
        raise ValueError("a window admits no stride")
    r = (torch.rand(t.shape[0], generator=generator) * n_ok).long()
    return torch.minimum(r, n_ok - 1).to(t_end.device)


def window_positions(t_end: torch.Tensor, stride: torch.Tensor, W: int) -> torch.Tensor:
    """(n, W) positions, oldest first: t_end - stride*(W-1) ... t_end."""
    steps = torch.arange(W - 1, -1, -1, device=t_end.device)
    return t_end[:, None] - stride[:, None] * steps[None, :]


class RealBooks:
    """Training books on their target device, with the data mean for centring."""

    def __init__(self, h: torch.Tensor, n_sink: int):
        self.h = h
        self.n_books, self.T, self.D = h.shape
        self.mean = data_mean(h, n_sink)

    def windows(
        self, pos: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """(n, W, D) float32 `h[b, pos] - mean`, one uniformly drawn book b per window."""
        b = torch.randint(self.n_books, (pos.shape[0],), generator=generator).to(
            pos.device
        )
        return self.h[b[:, None], pos].float() - self.mean
