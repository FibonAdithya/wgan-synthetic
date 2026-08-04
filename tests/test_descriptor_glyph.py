import numpy as np
import pytest

from src.eval import descriptor_glyph as dg


def test_constants_multiply_to_the_descriptor_dim():
    assert dg.CELL_ROWS * dg.CELL_COLS * dg.ORIENTATION_BINS == dg.DESCRIPTOR_DIM
    assert dg.DESCRIPTOR_DIM == 128


def test_index_convention_is_cell_major():
    """index = (row * 4 + col) * 8 + bin, pinned exactly."""
    cells = dg.descriptor_to_cells(np.arange(128, dtype=np.float64))
    assert cells.shape == (4, 4, 8)
    for row in range(4):
        for col in range(4):
            for bin_ in range(8):
                expected = (row * 4 + col) * 8 + bin_
                assert cells[row, col, bin_] == expected


@pytest.mark.parametrize("length", [0, 127, 129, 256])
def test_wrong_length_raises(length):
    with pytest.raises(ValueError, match="128"):
        dg.descriptor_to_cells(np.zeros(length))


def test_two_dimensional_input_raises():
    with pytest.raises(ValueError, match="128"):
        dg.descriptor_to_cells(np.zeros((2, 128)))


def test_shared_scale_maps_the_percentile_value_to_one():
    """A bin at the reference percentile draws a full half-cell ray."""
    data = np.full((10, 128), 0.5)
    scale = dg.shared_scale(data)
    assert 0.5 * scale == pytest.approx(1.0)


def test_shared_scale_ignores_a_single_outlier():
    """One huge spike must not shrink every other ray into invisibility."""
    typical = np.full((100, 128), 0.5)
    with_spike = typical.copy()
    with_spike[0, 0] = 1000.0
    assert dg.shared_scale(with_spike) == pytest.approx(
        dg.shared_scale(typical), rel=0.05
    )


def test_shared_scale_uses_magnitude_of_negatives():
    positive = np.full((4, 128), 0.5)
    negative = np.full((4, 128), -0.5)
    assert dg.shared_scale(negative) == pytest.approx(dg.shared_scale(positive))


def test_shared_scale_ignores_zeros_on_sparse_input():
    """Real SIFT is sparse; a percentile over the raw values would land in
    the run of zeros and scale every ray to nothing."""
    sparse = np.zeros((4, 128))
    sparse[:, :3] = 0.5
    assert 0.5 * dg.shared_scale(sparse) == pytest.approx(1.0)


def test_shared_scale_of_all_zeros_is_zero():
    assert dg.shared_scale(np.zeros((4, 128))) == 0.0


def test_shared_scale_of_empty_input_is_zero():
    assert dg.shared_scale(np.zeros((0, 128))) == 0.0


def _one_hot(row, col, bin_, value=1.0):
    vec = np.zeros(128)
    vec[(row * 4 + col) * 8 + bin_] = value
    return dg.descriptor_to_cells(vec)


def _segments(xs, ys):
    """Split NaN-separated coordinates into a list of (start, end) pairs."""
    pairs = []
    for i in range(0, len(xs), 3):
        assert np.isnan(xs[i + 2]) and np.isnan(ys[i + 2])
        pairs.append(((xs[i], ys[i]), (xs[i + 1], ys[i + 1])))
    return pairs


def test_one_hot_bin_draws_exactly_one_ray():
    cells = _one_hot(row=1, col=2, bin_=0)
    px, py, nx, ny = dg.glyph_segments(cells, origin=(0.0, 0.0), pitch=2.0, scale=1.0)
    assert len(nx) == 0 and len(ny) == 0
    pairs = _segments(px, py)
    assert len(pairs) == 1
    (start, end) = pairs[0]
    # cell (1, 2): x = (2 - 1.5) * 2 = 1.0, y = (1.5 - 1) * 2 = 1.0
    assert start == pytest.approx((1.0, 1.0))
    # bin 0 points along +x; length = min(1.0 * 1.0, 1.0) * 2.0 / 2 = 1.0
    assert end == pytest.approx((2.0, 1.0))


@pytest.mark.parametrize(
    "bin_, unit",
    [(0, (1.0, 0.0)), (1, (0.7071, 0.7071)), (2, (0.0, 1.0)), (6, (0.0, -1.0))],
)
def test_bin_angles_are_45_degrees_counter_clockwise_from_x(bin_, unit):
    cells = _one_hot(row=0, col=0, bin_=bin_)
    px, py, _, _ = dg.glyph_segments(cells, origin=(0.0, 0.0), pitch=2.0, scale=1.0)
    (start, end) = _segments(px, py)[0]
    assert (end[0] - start[0], end[1] - start[1]) == pytest.approx(unit, abs=1e-3)


def test_row_zero_is_at_the_top():
    top = dg.glyph_segments(_one_hot(0, 0, 0), (0.0, 0.0), 2.0, 1.0)[1][0]
    bottom = dg.glyph_segments(_one_hot(3, 0, 0), (0.0, 0.0), 2.0, 1.0)[1][0]
    assert top > bottom


def test_origin_is_the_glyph_centre():
    """Cells straddle the origin symmetrically: (0,0) and (3,3) mirror."""
    a = dg.glyph_segments(_one_hot(0, 0, 0), (0.0, 0.0), 2.0, 1.0)
    b = dg.glyph_segments(_one_hot(3, 3, 0), (0.0, 0.0), 2.0, 1.0)
    assert a[0][0] == pytest.approx(-b[0][0])
    assert a[1][0] == pytest.approx(-b[1][0])


def test_origin_offset_translates_every_coordinate():
    at_zero = dg.glyph_segments(_one_hot(1, 2, 0), (0.0, 0.0), 2.0, 1.0)
    shifted = dg.glyph_segments(_one_hot(1, 2, 0), (10.0, -5.0), 2.0, 1.0)
    assert shifted[0][0] == pytest.approx(at_zero[0][0] + 10.0)
    assert shifted[1][0] == pytest.approx(at_zero[1][0] - 5.0)


def test_negative_bin_goes_to_the_negative_arrays_at_full_magnitude():
    """Negatives are impossible for a gradient histogram, so they are shown,
    not clamped -- this is what distinguishes the MLP variants from v2."""
    cells = _one_hot(row=1, col=2, bin_=0, value=-1.0)
    px, py, nx, ny = dg.glyph_segments(cells, (0.0, 0.0), pitch=2.0, scale=1.0)
    assert len(px) == 0 and len(py) == 0
    (start, end) = _segments(nx, ny)[0]
    assert start == pytest.approx((1.0, 1.0))
    assert end == pytest.approx((2.0, 1.0))


def test_mixed_signs_are_split_across_both_outputs():
    vec = np.zeros(128)
    vec[(1 * 4 + 2) * 8 + 0] = 1.0
    vec[(1 * 4 + 2) * 8 + 4] = -1.0
    px, py, nx, ny = dg.glyph_segments(
        dg.descriptor_to_cells(vec), (0.0, 0.0), 2.0, 1.0
    )
    assert len(_segments(px, py)) == 1
    assert len(_segments(nx, ny)) == 1


def test_ray_length_is_linear_in_scale_below_the_clip():
    cells = _one_hot(0, 0, 0, value=0.25)
    short = dg.glyph_segments(cells, (0.0, 0.0), 2.0, scale=1.0)[0]
    long = dg.glyph_segments(cells, (0.0, 0.0), 2.0, scale=2.0)[0]
    assert (long[1] - long[0]) == pytest.approx(2.0 * (short[1] - short[0]))


def test_ray_is_clipped_to_half_a_cell():
    """A bin far above the percentile must not bleed into its neighbour."""
    cells = _one_hot(0, 0, 0, value=1000.0)
    px, _, _, _ = dg.glyph_segments(cells, (0.0, 0.0), pitch=2.0, scale=1.0)
    assert (px[1] - px[0]) == pytest.approx(1.0)  # pitch / 2


def test_zero_vector_draws_nothing():
    cells = dg.descriptor_to_cells(np.zeros(128))
    px, py, nx, ny = dg.glyph_segments(cells, (0.0, 0.0), 2.0, 1.0)
    assert len(px) == len(py) == len(nx) == len(ny) == 0


def test_zero_scale_draws_nothing():
    cells = dg.descriptor_to_cells(np.ones(128))
    px, _, nx, _ = dg.glyph_segments(cells, (0.0, 0.0), 2.0, scale=0.0)
    assert len(px) == 0 and len(nx) == 0


def test_wrong_cell_shape_raises():
    with pytest.raises(ValueError, match="4, 4, 8"):
        dg.glyph_segments(np.zeros((4, 4)), (0.0, 0.0), 2.0, 1.0)
