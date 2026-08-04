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
