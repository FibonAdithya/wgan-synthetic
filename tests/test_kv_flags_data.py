import numpy as np
import pytest

from src.data.dataset import PreprocessConfig, build_training_data
from src.data.kv_capture import (
    h_path,
    kv_flags,
    load_kv_capture,
    write_bf16_safetensor,
)
from src.models.conditional import BODY, TAIL


def _write(root, book, layer, arr):
    p = h_path(root, book, layer)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_bf16_safetensor(p, "h", arr)


def _position_book(T, d=4):
    """Row value = position, so a loaded row's first column names its position."""
    return np.repeat(np.arange(T, dtype=np.float32)[:, None], d, axis=1)


def _build(root, books=(0, 1), drop_zero=False, **kw):
    return build_training_data(
        str(root),
        "kv_capture",
        4,
        0.2,
        PreprocessConfig(l2_normalize=False, drop_zero_rows=drop_zero),
        0,
        kv_layer=3,
        kv_books=list(books),
        kv_n_sink=1,
        **kw,
    )


def test_kv_flags_marks_exactly_the_last_n_tail_positions():
    """Catches an off-by-one at the body/tail boundary (spec: tail = 31,744-32,767)."""
    f = kv_flags(32768, 1, 1024)
    assert len(f) == 32767 and f.sum() == 1024
    assert f.dtype == np.int64
    assert f[31742] == BODY and f[31743] == TAIL  # row i is position i + 1


def test_kv_flags_rejects_a_book_too_short():
    """Catches accepting a book with no body rows (T = n_sink + n_tail)."""
    with pytest.raises(ValueError):
        kv_flags(1025, 1, 1024)
    assert len(kv_flags(1026, 1, 1024)) == 1025


def test_load_kv_capture_without_n_tail_is_unchanged(tmp_path):
    """Catches the v0 return type changing (would break every v0 caller)."""
    for b in (0, 1):
        _write(tmp_path, b, 3, _position_book(20))
    assert isinstance(load_kv_capture(tmp_path, 3, [0, 1], 1), np.ndarray)
    out = load_kv_capture(tmp_path, 3, [0, 1], 1, n_tail=4)
    assert isinstance(out, tuple) and len(out) == 2


def test_load_kv_capture_flags_align_with_rows(tmp_path):
    """Catches flags built with np.repeat instead of per-book concatenation, or computed
    over positions 0..T-1 without the sink offset: each row's value is its position, so
    flags must equal (position >= T - n_tail) for every row, in both directions."""
    T, n_tail = 40, 6
    for b in (0, 1):
        _write(tmp_path, b, 3, _position_book(T))
    x, flags = load_kv_capture(tmp_path, 3, [0, 1], 1, n_tail=n_tail)
    assert x.shape[0] == flags.shape[0] == 2 * (T - 1)
    np.testing.assert_array_equal(flags == TAIL, x[:, 0] >= T - n_tail)
    assert (flags == TAIL).sum() == 2 * n_tail


def test_load_kv_capture_unequal_book_lengths_raise(tmp_path):
    """Catches silently stacking books of different lengths with mismatched flags."""
    _write(tmp_path, 0, 3, _position_book(20))
    _write(tmp_path, 1, 3, _position_book(24))
    with pytest.raises(ValueError):
        load_kv_capture(tmp_path, 3, [0, 1], 1, n_tail=4)


def test_holdout_has_exactly_kv_tail_holdout_tail_rows(tmp_path):
    """Catches a random (unstratified) holdout: 2 books x T=64, n_tail 8, tail holdout 4."""
    for b in (0, 1):
        _write(tmp_path, b, 3, _position_book(64))
    xt, xh, _, ft, fh = _build(tmp_path, kv_n_tail=8, kv_tail_holdout=4)
    assert (fh == TAIL).sum() == 4 and (ft == TAIL).sum() == 2 * 8 - 4
    assert (fh == BODY).sum() == round(0.2 * 2 * (64 - 1 - 8))
    assert len(xt) == len(ft) and len(xh) == len(fh)


def test_flags_travel_with_their_rows_through_the_split(tmp_path):
    """Catches flags permuted independently of rows: with l2/centering off the row value is
    still its position, so every TAIL-flagged row must have position >= 56."""
    for b in (0, 1):
        _write(tmp_path, b, 3, _position_book(64))
    xt, xh, _, ft, fh = _build(tmp_path, kv_n_tail=8, kv_tail_holdout=4)
    np.testing.assert_array_equal(ft == TAIL, xt[:, 0] >= 56)
    np.testing.assert_array_equal(fh == TAIL, xh[:, 0] >= 56)


def test_drop_zero_rows_drops_their_flags_too(tmp_path):
    """Catches flags shifting after a dropped row: zero one tail row; tail count drops by 1
    and every remaining TAIL row is still a tail position."""
    T = 64
    for b in (0, 1):
        arr = _position_book(T) + 1.0  # no zero rows by default
        if b == 0:
            arr[T - 2] = 0.0  # a tail position
        _write(tmp_path, b, 3, arr)
    xt, xh, _, ft, fh = _build(tmp_path, drop_zero=True, kv_n_tail=8, kv_tail_holdout=4)
    assert (ft == TAIL).sum() + (fh == TAIL).sum() == 2 * 8 - 1
    assert (fh == TAIL).sum() == 4
    np.testing.assert_array_equal(ft == TAIL, xt[:, 0] >= 56 + 1)
    np.testing.assert_array_equal(fh == TAIL, xh[:, 0] >= 56 + 1)
    assert len(xt) + len(xh) == 2 * (T - 1) - 1


@pytest.mark.parametrize("bad", [0, 16])
def test_kv_tail_holdout_out_of_range_raises(tmp_path, bad):
    """Catches accepting a holdout with no tail rows or all of them (16 tail rows here)."""
    for b in (0, 1):
        _write(tmp_path, b, 3, _position_book(64))
    with pytest.raises(ValueError):
        _build(tmp_path, kv_n_tail=8, kv_tail_holdout=bad)


def test_kv_tail_holdout_is_required_with_kv_n_tail(tmp_path):
    """Catches defaulting a missing tail holdout instead of failing."""
    for b in (0, 1):
        _write(tmp_path, b, 3, _position_book(64))
    with pytest.raises(ValueError):
        _build(tmp_path, kv_n_tail=8)


def test_kv_n_tail_with_another_format_raises(tmp_path):
    """Catches silently ignoring kv_n_tail for a non-kv_capture format. The path is a real
    .npy file, so only the format check can raise (deleting it returns a 3-tuple)."""
    path = tmp_path / "x.npy"
    np.save(path, np.ones((50, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="kv_n_tail"):
        build_training_data(
            str(path),
            "npy",
            4,
            0.2,
            PreprocessConfig(l2_normalize=False),
            0,
            kv_n_tail=8,
            kv_tail_holdout=4,
        )


def test_kv_n_tail_without_a_path_raises(tmp_path):
    """Catches kv_n_tail being accepted with descriptor_path=None and kv_capture: with
    synthetic_if_missing the synthetic branch would otherwise return a flagless 3-tuple."""
    with pytest.raises(ValueError, match="kv_n_tail"):
        build_training_data(
            None,
            "kv_capture",
            4,
            0.2,
            PreprocessConfig(l2_normalize=False),
            0,
            synthetic_if_missing=True,
            kv_layer=3,
            kv_books=[0],
            kv_n_tail=8,
            kv_tail_holdout=4,
        )


def test_default_path_split_is_unchanged(tmp_path):
    """Catches the v0 split changing: kv_n_tail=None must reproduce train_holdout_split."""
    from src.data.dataset import train_holdout_split

    for b in (0, 1):
        _write(tmp_path, b, 3, _position_book(64))
    xt, xh, _ = _build(tmp_path)
    raw = load_kv_capture(tmp_path, 3, [0, 1], 1)
    et, eh = train_holdout_split(raw, 0.2, 0)
    np.testing.assert_array_equal(xt, et)
    np.testing.assert_array_equal(xh, eh)
