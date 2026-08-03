"""The inverse of the preprocessing transform in src/data/sift1m_dataset.py.

DEEP's v2 rung trains in PCA-whitened space, so its samples have to be mapped
back to the original coordinates before anything compares them against real
DEEP. The forward transform is applied in the order center -> whiten ->
l2_normalize; the inverse undoes whiten then center.

L2 normalization is deliberately NOT inverted: it discards each vector's norm,
so the information needed to undo it is gone. This is not a limitation in
practice -- DEEP vectors are unit-norm to begin with, and the comparison is
angular.
"""
from __future__ import annotations

import numpy as np

from src.data.sift1m_dataset import PreprocessState


def invert_preprocess(x: np.ndarray, state: PreprocessState) -> np.ndarray:
    """Undo centering and whitening, in reverse order of application.

    The whitening matrix is symmetric (u @ diag(1/sqrt(s)) @ u.T over a
    symmetric covariance), so its inverse is likewise symmetric and is
    obtained with a plain matrix inverse. pinv rather than inv, because the
    eps-regularized covariance can still be near-singular on the tail
    dimensions of a PCA-derived set like DEEP.
    """
    out = np.asarray(x, dtype=np.float32)
    if state.whitening_matrix is not None:
        out = out @ np.linalg.pinv(state.whitening_matrix).astype(np.float32)
    if state.mean is not None:
        out = out + state.mean
    return np.ascontiguousarray(out, dtype=np.float32)
