"""Book loading and window positions for scout_kv v2."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tests.book_latent_fixtures import write_tiny_capture

from src.data.kv_capture import h_path, load_kv_books, write_bf16_safetensor
from src.models.conditional import BODY, TAIL
from src.train.book_windows import (
    RealBooks,
    data_mean,
    sample_stride,
    tail_flags,
    window_ends,
    window_positions,
)

STRIDES = [1, 4, 16, 64, 256, 1024]


def test_load_kv_books_keeps_every_position_and_book_order(tmp_path):
    """Catches the sink being dropped (load_kv_capture drops it) or books stacked out of order:
    book b's rows are N(0,1)+b, so h[i].float().mean() is near books[i]."""
    write_tiny_capture(tmp_path, [0, 1, 2])
    h = load_kv_books(tmp_path, 3, [2, 0, 1])
    assert h.shape == (3, 64, 16) and h.dtype == torch.bfloat16
    assert [round(float(x.float().mean())) for x in h] == [2, 0, 1]


def test_load_kv_books_rejects_unequal_lengths(tmp_path):
    """Catches books of different lengths being stacked or truncated silently."""
    write_tiny_capture(tmp_path, [0], T=64)
    write_tiny_capture(tmp_path, [1], T=48)
    with pytest.raises(ValueError, match="different lengths"):
        load_kv_books(tmp_path, 3, [0, 1])


def test_load_kv_books_rejects_a_local_layer(tmp_path):
    """Catches the global-layer check being dropped."""
    write_tiny_capture(tmp_path, [0], layer=4)
    with pytest.raises(ValueError, match="global layer"):
        load_kv_books(tmp_path, 4, [0])


def test_data_mean_excludes_the_sink():
    """Catches the sink row entering the mean: position 0 set to 1e4, others 0 -> mean 0."""
    h = torch.zeros(2, 10, 4, dtype=torch.bfloat16)
    h[:, 0] = 1e4
    m = data_mean(h, 1)
    assert m.dtype == torch.float32 and m.shape == (4,)
    assert torch.equal(m, torch.zeros(4))


def test_tail_flags_boundary():
    """Catches an off-by-one at the body/tail boundary (spec: tail = 31,744-32,767)."""
    f = tail_flags(torch.tensor([31743, 31744, 32767]), 32768, 1024)
    assert f.tolist() == [BODY, TAIL, TAIL]
    assert f.dtype == torch.long


def test_windows_stay_in_their_part_at_full_scale():
    """Catches a body window touching position 0 or the tail, a tail window leaving the tail,
    or a stride that does not fit (spec Review Focus 3). 20 seeded draws of 64 windows at
    T 32,768, n_tail 1,024, W 8, strides 1..1024, 16 tail windows."""
    for s in range(20):
        g = torch.Generator().manual_seed(s)
        t_end, lo = window_ends(64, 16, 32768, 1, 1024, 8, "cpu", g)
        idx = sample_stride(t_end, lo, STRIDES, 8, g)
        pos = window_positions(t_end, torch.tensor(STRIDES)[idx], 8)
        body, tail = pos[:48], pos[48:]
        assert body.min() >= 1 and body.max() <= 31743
        assert tail.min() >= 31744 and tail.max() <= 32767
        assert (pos[:, 1:] - pos[:, :-1] > 0).all()


def test_lowest_t_end_still_gets_a_fitting_stride():
    """Catches a range that starts at t_end = W - 1 (window would reach position 0) or a
    stride draw that ignores lo: t_end = n_sink + W - 1 = 8 admits only stride 1."""
    idx = sample_stride(torch.tensor([8] * 50), torch.tensor([1] * 50), STRIDES, 8)
    assert idx.tolist() == [0] * 50


def test_stride_draw_is_uniform_over_fitting_strides():
    """Catches the r-th-fitting-stride shortcut breaking (a biased or clamped draw): t_end 31,743
    in the body admits all 6 strides; over 6,000 seeded draws each index appears 800-1,200 times.
    In the tail (lo 31,744, t_end 32,767) only strides 1, 4, 16, 64 fit: indices 4, 5 never appear."""
    g = torch.Generator().manual_seed(0)
    n = 6000
    body = sample_stride(torch.full((n,), 31743), torch.full((n,), 1), STRIDES, 8, g)
    counts = torch.bincount(body, minlength=6)
    assert ((counts >= 800) & (counts <= 1200)).all(), counts
    tail = sample_stride(
        torch.full((n,), 32767), torch.full((n,), 31744), STRIDES, 8, g
    )
    counts = torch.bincount(tail, minlength=6)
    assert counts[4] == 0 and counts[5] == 0
    assert ((counts[:4] >= 1300) & (counts[:4] <= 1700)).all(), counts


def test_sample_stride_rejects_unsorted_strides():
    """Catches non-increasing strides being accepted (the r-th-fitting shortcut needs order)."""
    with pytest.raises(ValueError, match="increasing"):
        sample_stride(torch.tensor([100]), torch.tensor([1]), [1, 4, 4], 8)
    with pytest.raises(ValueError, match="increasing"):
        sample_stride(torch.tensor([100]), torch.tensor([1]), [4, 1], 8)


def test_window_end_ranges_are_attained_exactly():
    """Catches an off-by-one at either end of either range (randint's exclusive upper bound, or a
    body range starting at W - 1). The full-scale test above draws each value with p ~ 3e-5 and
    would miss these. Small book: T 64, n_sink 1, n_tail 16, W 8; 2,000 seeded windows of each
    kind: body t_end min == 8 and max == 47; tail t_end min == 55 and max == 63."""
    g = torch.Generator().manual_seed(0)
    t_end, lo = window_ends(4000, 2000, 64, 1, 16, 8, "cpu", g)
    body, tail = t_end[:2000], t_end[2000:]
    assert (int(body.min()), int(body.max())) == (8, 47)
    assert (int(tail.min()), int(tail.max())) == (55, 63)
    assert (lo[:2000] == 1).all() and (lo[2000:] == 48).all()


def test_exact_tail_window_count():
    """Catches the tail count being random (a Bernoulli draw) or off by one."""
    _, lo = window_ends(64, 16, 64, 1, 16, 8, "cpu", torch.Generator().manual_seed(3))
    assert int((lo == 64 - 16).sum()) == 16 and int((lo == 1).sum()) == 48


@pytest.mark.parametrize("n_tail_windows", [0, 64])
def test_window_ends_allows_all_body_or_all_tail(n_tail_windows):
    """Catches a guard that rejects 0 or all tail windows here (the trainer rejects them, Task 3)."""
    _, lo = window_ends(64, n_tail_windows, 64, 1, 16, 8, "cpu")
    assert int((lo == 48).sum()) == n_tail_windows


def test_window_ends_shortest_book():
    """Catches the length check being off by one: the body needs T >= n_sink + n_tail + W.
    T = 1 + 16 + 8 = 25 works (body t_end is exactly 8); T = 24 raises ValueError.
    n_tail = W - 1 = 7 raises ValueError (no tail window fits)."""
    t_end, _ = window_ends(10, 0, 25, 1, 16, 8, "cpu")
    assert (t_end == 8).all()
    with pytest.raises(ValueError):
        window_ends(10, 0, 24, 1, 16, 8, "cpu")
    with pytest.raises(ValueError):
        window_ends(10, 2, 64, 1, 7, 8, "cpu")


def test_real_windows_come_from_one_book_each(tmp_path):
    """Catches tokens of one window drawn from different books (each window's 8 rows are equal,
    because book b holds the constant b in every entry), and one book drawn for the whole batch
    (across 64 windows from 4 books, at least 2 distinct books appear)."""
    T, D = 32, 4
    for b in range(4):
        p = h_path(tmp_path, b, 3)
        p.parent.mkdir(parents=True)
        write_bf16_safetensor(p, "h", np.full((T, D), float(b), dtype=np.float32))
    h = load_kv_books(tmp_path, 3, [0, 1, 2, 3])
    rb = RealBooks(h, 1)
    assert (rb.T, rb.D, rb.n_books) == (T, D, 4)
    pos = torch.arange(8).repeat(64, 1) + 5
    w = rb.windows(pos, torch.Generator().manual_seed(0))
    assert w.shape == (64, 8, D) and w.dtype == torch.float32
    v = w + rb.mean
    assert (v == v[:, :1]).all()
    assert len(torch.unique(v[:, 0, 0])) >= 2
