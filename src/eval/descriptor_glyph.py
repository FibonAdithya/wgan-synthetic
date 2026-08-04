"""Geometry for drawing one SIFT descriptor as a 4x4 grid of 8-ray stars.

A SIFT descriptor is a 4x4 grid of spatial cells, each holding an 8-bin
histogram of gradient orientations (Lowe 2004). Real descriptors are sparse
and spiky because most image patches contain edges; a generator that matches
the marginals but produces even, bushy stars is failing in a way no aggregate
panel in `eda_report` will show.

The glyph is a diagram of the descriptor, not of the image patch it came
from -- SIFT is lossy and orientation-normalised, and inverting it is out of
scope.

Convention caveat: the row/col scan order and the zero direction of the
orientation bins differ between SIFT implementations (VLFeat, OpenCV, Lowe's
original binary), and we do not know which extractor produced SIFT1M. The
glyph may be rotated or transposed relative to true patch geometry. That does
not affect the comparison this module exists for: the same convention is
applied to real and generated vectors alike. It would only matter for a claim
about underlying image content, which we do not make.

Pure NumPy by design: no I/O and no Plotly import, so the index and angle
conventions can be pinned by tests on a machine holding neither the dataset
nor any checkpoint.
"""

from __future__ import annotations

import numpy as np

CELL_ROWS = 4
CELL_COLS = 4
ORIENTATION_BINS = 8
DESCRIPTOR_DIM = CELL_ROWS * CELL_COLS * ORIENTATION_BINS  # 128


def descriptor_to_cells(vec: np.ndarray) -> np.ndarray:
    """Reshape a descriptor to (4, 4, 8) indexed [row][col][orientation_bin].

    The flat layout is `index = (row * 4 + col) * 8 + bin`, which is exactly
    C-order for this shape, so the reshape is the convention.
    """
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape != (DESCRIPTOR_DIM,):
        raise ValueError(
            f"expected a flat descriptor of {DESCRIPTOR_DIM} values, "
            f"got array of shape {arr.shape}"
        )
    return arr.reshape(CELL_ROWS, CELL_COLS, ORIENTATION_BINS)
