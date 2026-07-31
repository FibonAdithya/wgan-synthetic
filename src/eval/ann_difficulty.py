"""ANN-difficulty metrics for real-vs-synthetic descriptor comparison.

The EDA report next door compares distributional shape -- whether a synthetic
set *looks* like SIFT. These metrics ask the other question: whether it would
*behave* like SIFT under nearest-neighbour search. A generator can match mean,
variance, pairwise-distance median and effective rank while producing a set
that is far easier or harder to search than the real thing, which makes it
useless as a benchmark stand-in.

Everything here is computed from the vectors alone; no index is built. The
metrics are the published predictors of ANN hardness: local intrinsic
dimensionality, relative contrast, hubness, and partition balance.

All values are relative instruments. They are comparable across the sets
measured in one run and are *not* comparable with published figures for
SIFT1M, which are measured on the full 1M set against the real query set
rather than on a self-queried subsample.

This module deliberately does not import from eda_report: it must stay usable
and testable without plotly or argparse.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def gini(occupancy: np.ndarray) -> float:
    """Gini coefficient of a cluster-occupancy vector.

    0.0 means every cell holds the same number of points; (n-1)/n means one
    cell holds everything. Reads as how lopsided an IVF partition would be,
    which drives how many cells a query has to probe.
    """
    x = np.sort(np.asarray(occupancy, dtype=np.float64))
    total = x.sum()
    if total <= 0.0:
        return 0.0
    n = x.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float(2.0 * np.sum(index * x) / (n * total) - (n + 1.0) / n)


def knn(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Nearest neighbours of every row among the *other* rows.

    Returns (distances, indices, k_eff). k is clamped to n-1 when the set is
    smaller than requested, and k_eff reports what was actually used.

    Self-exclusion is by index, not by dropping the first column. Exact
    duplicate rows tie with the query at distance 0 and sklearn does not
    promise the query sorts first, so column-dropping can silently leave a
    point in its own neighbour list -- which would drag its k-occurrence up
    and its LID down.
    """
    n = x.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 rows to compute neighbours, got {n}")
    k_eff = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="brute").fit(x)
    dist, idx = nn.kneighbors(x)

    rows = np.arange(n)[:, None]
    keep = idx != rows
    # A row whose own index did not come back has k_eff+1 keepers; drop its
    # farthest so every row yields exactly k_eff.
    surplus = keep.sum(axis=1) > k_eff
    if np.any(surplus):
        last_true = (keep.shape[1] - 1) - np.argmax(keep[:, ::-1], axis=1)
        keep[surplus, last_true[surplus]] = False

    selected = np.where(keep)
    return (
        dist[selected].reshape(n, k_eff),
        idx[selected].reshape(n, k_eff),
        k_eff,
    )


def survivor_mask(dist: np.ndarray) -> np.ndarray:
    """Queries whose nearest neighbour is strictly farther than zero.

    A query sitting on an exact duplicate has r_1 = 0, which sends the LID
    estimator to a degenerate value. Those queries are dropped rather than
    clamped: clamping invents a number, dropping just declines to answer.
    The count is reported so the bias stays visible -- duplicates are exactly
    the low-LID region, so discarding them nudges the estimate upward.
    """
    return dist[:, 0] > 0.0


def lid_mle(dist: np.ndarray) -> np.ndarray:
    """Hill / Amsaleg maximum-likelihood local intrinsic dimensionality.

        LID(q) = -[ (1/k) * sum_i log(r_i / r_k) ]^-1

    Pass only rows selected by survivor_mask. The i=k term contributes zero
    and is kept so the divisor is k, matching the standard MLE form.

    Higher means locally higher-dimensional, which means harder to search:
    this is the strongest single published predictor of ANN difficulty.
    """
    if dist.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    r_k = dist[:, -1:]
    ratio = np.clip(dist / r_k, 1.0e-12, 1.0)
    return -1.0 / np.mean(np.log(ratio), axis=1)
