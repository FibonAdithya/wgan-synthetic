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
