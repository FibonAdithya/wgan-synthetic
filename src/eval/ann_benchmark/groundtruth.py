"""Exact k-nearest neighbours, used as the recall reference.

Computed on GPU with the same brute-force index the benchmark reports as its
recall-1.0 row, so this costs nothing beyond a row that was being measured
anyway.

Queries are *not* excluded from the corpus by index here, unlike
`src.eval.ann_difficulty.knn`. That module self-queries a corpus against
itself, where a point is its own nearest neighbour and must be dropped. Here
the query set is disjoint from the corpus by construction -- a fresh
generator draw, or SIFT's own held-out query set -- so there is no self to
exclude, and dropping a column would silently discard a real neighbour.
"""

from __future__ import annotations

import numpy as np

from src.eval.ann_benchmark.indexes import FlatAdapter, IndexAdapter


def exact_neighbours(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    *,
    adapter: IndexAdapter | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact `(distances, ids)` for every query, in squared L2.

    `adapter` defaults to the GPU brute-force index; tests pass
    `NumpyFlatAdapter` to run the same code path without a device.
    """
    if vectors.ndim != 2 or queries.ndim != 2:
        raise ValueError(
            f"expected 2-D arrays, got {vectors.shape} and {queries.shape}"
        )
    if vectors.shape[1] != queries.shape[1]:
        raise ValueError(
            f"dimension mismatch: corpus is {vectors.shape[1]}-d, queries are "
            f"{queries.shape[1]}-d"
        )
    if k > vectors.shape[0]:
        raise ValueError(
            f"k={k} exceeds the corpus size {vectors.shape[0]}; there are not "
            "that many neighbours to find"
        )

    index = adapter if adapter is not None else FlatAdapter()
    built = index.build(vectors)
    index.sync()
    distances, ids = index.search(built, queries, k, None)
    index.sync()
    return distances, ids
