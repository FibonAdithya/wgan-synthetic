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


def shared_scale(descriptors: np.ndarray, percentile: float = 99.0) -> float:
    """Ray-length factor shared by every glyph in a figure.

    Computed from the given percentile of `|value|` across every descriptor
    that will be plotted -- real and generated together -- so rows stay
    honestly comparable. A value at the percentile maps to `1.0`, i.e. a ray
    filling the half-cell; larger values are clipped by `glyph_segments`.

    Zeros are excluded before the percentile is taken. Real SIFT descriptors
    are sparse, and over the raw values the percentile can fall inside the run
    of zeros, returning 0.0 and drawing an empty figure. Excluding them also
    makes the reference mean "a typical meaningful bin", which is what a ray
    length should be relative to.

    Returns 0.0 for empty or all-zero input, which draws no rays rather than
    dividing by zero.
    """
    arr = np.abs(np.asarray(descriptors, dtype=np.float64))
    nonzero = arr[arr > 0.0]
    if nonzero.size == 0:
        return 0.0
    reference = float(np.percentile(nonzero, percentile))
    if reference <= 0.0:
        return 0.0
    return 1.0 / reference
