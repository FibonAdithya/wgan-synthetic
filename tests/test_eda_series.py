import numpy as np
import pytest

from src.eval.eda import series


def test_maybe_l2_normalize_gives_unit_rows():
    x = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    out = series.maybe_l2_normalize(x, "l2")
    assert np.linalg.norm(out, axis=1) == pytest.approx([1.0, 1.0])


def test_maybe_l2_normalize_leaves_a_zero_row_finite():
    """The eps clamp, not a divide by zero. `plot_descriptor_grid` hands this
    raw SIFT rows, and an all-zero descriptor is rare but legal."""
    out = series.maybe_l2_normalize(np.zeros((1, 4), dtype=np.float32), "l2")
    assert np.all(np.isfinite(out))


def test_maybe_l2_normalize_none_mode_passes_rows_through():
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    assert np.array_equal(series.maybe_l2_normalize(x, "none"), x)
