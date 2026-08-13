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

This module deliberately does not import from `src.eval.eda`: it must stay
usable and testable without plotly or argparse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans
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


def _exclude_self(
    dist: np.ndarray, idx: np.ndarray, k_eff: int
) -> tuple[np.ndarray, np.ndarray]:
    """Drop each row's own index from its neighbour list.

    Self-exclusion is by index, not by dropping the first column. Exact
    duplicate rows tie with the query at distance 0 and neither backend
    promises the query sorts first, so column-dropping can silently leave a
    point in its own neighbour list -- which would drag its k-occurrence up
    and its LID down.

    Callers pass the raw (n, k_eff + 1) arrays. A row whose own index did not
    come back has k_eff + 1 keepers; its farthest is dropped so every row
    yields exactly k_eff.
    """
    n = idx.shape[0]
    rows = np.arange(n)[:, None]
    keep = idx != rows
    surplus = keep.sum(axis=1) > k_eff
    if np.any(surplus):
        last_true = (keep.shape[1] - 1) - np.argmax(keep[:, ::-1], axis=1)
        keep[surplus, last_true[surplus]] = False

    selected = np.where(keep)
    return dist[selected].reshape(n, k_eff), idx[selected].reshape(n, k_eff)


def _knn_torch(
    x: np.ndarray, want: int, chunk_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force neighbours on the GPU when there is one, else on CPU torch.

    Chunked over query rows because the distance block is the memory cost:
    chunk_rows x n floats. At the default 1024 and n = 250,000 that is 1 GB,
    which sits comfortably beside a 100 MB corpus on an 8 GB card.

    Returns the raw (n, want) arrays including each row's own index; the
    caller drops it via `_exclude_self`.

    torch is imported here rather than at module scope so the default
    sklearn path -- what every figure under docs/datasets/ was measured
    with -- does not pay torch's import cost to have this backend exist.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
    n = corpus.shape[0]

    dist = np.empty((n, want), dtype=np.float64)
    idx = np.empty((n, want), dtype=np.int64)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        block = torch.cdist(corpus[start:stop], corpus)
        values, columns = torch.topk(block, want, dim=1, largest=False, sorted=True)
        dist[start:stop] = values.to(torch.float64).cpu().numpy()
        idx[start:stop] = columns.cpu().numpy()
    return dist, idx


def knn(
    x: np.ndarray, k: int, *, backend: str = "sklearn", chunk_rows: int = 1024
) -> tuple[np.ndarray, np.ndarray, int]:
    """Nearest neighbours of every row among the *other* rows.

    Returns (distances, indices, k_eff). k is clamped to n-1 when the set is
    smaller than requested, and k_eff reports what was actually used.

    `backend` selects the neighbour search. "sklearn" is the default and is
    what every figure committed under docs/datasets/ was measured with;
    "torch" runs the same search on the GPU when one is available. The two
    agree to the tolerance asserted in tests/test_ann_difficulty.py --
    sklearn's brute euclidean uses the ||x||^2 + ||y||^2 - 2xy expansion,
    which is the numerically worse form, so the difference is not all in
    torch's column.

    Self-exclusion is deferred to `_exclude_self`.
    """
    n = x.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 rows to compute neighbours, got {n}")
    k_eff = min(k, n - 1)

    if backend == "sklearn":
        nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="brute").fit(x)
        dist, idx = nn.kneighbors(x)
    elif backend == "torch":
        dist, idx = _knn_torch(x, k_eff + 1, chunk_rows)
    else:
        raise ValueError(f"unknown knn backend: {backend!r}")

    dist, idx = _exclude_self(dist, idx, k_eff)
    return dist, idx, k_eff


def survivor_mask(dist: np.ndarray) -> np.ndarray:
    """Queries fit for the LID/relative-contrast estimators.

    Two degenerate cases are dropped rather than clamped: clamping invents a
    number, dropping just declines to answer.

    - r_1 == 0: the query sits on an exact duplicate.
    - r_1 >= r_k: every one of the k neighbours ties at the same distance
      (only possible when r_1 == r_k, since distances are sorted ascending).
      lid_mle's ratio r_i/r_k is then 1.0 for every i, so mean(log(ratio)) is
      0.0 and the estimator divides by zero, returning -inf. One -inf among
      thousands of finite values is enough to poison every downstream
      consumer that takes a min/max over the array (histogram bin edges,
      summary statistics), so it must not reach lid_mle at all.

    The count of dropped queries is reported so the bias stays visible --
    duplicates are exactly the low-LID region, so discarding them nudges the
    estimate upward.
    """
    return (dist[:, 0] > 0.0) & (dist[:, 0] < dist[:, -1])


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


def relative_contrast(
    x: np.ndarray,
    dist: np.ndarray,
    seed: int,
    num_targets: int = 2000,
) -> np.ndarray:
    """Mean distance to a fixed target sample, divided by nearest distance.

    The classic Indyk-Motwani hardness measure. A value near 1 means the
    nearest neighbour is barely closer than an arbitrary point, so an index
    has almost nothing to exploit.

    One target sample is drawn per set and shared by every query, so the
    numerator is measured against a fixed reference rather than a per-query
    one. Apply survivor_mask to the result before summarising; rows whose
    nearest distance is zero divide to infinity here.
    """
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    targets = x[np.sort(rng.choice(n, size=min(num_targets, n), replace=False))]

    # Expanded-square distances in chunks. The broadcast form would allocate
    # chunk x targets x dim floats, which is gigabytes at these sizes.
    target_sq = np.einsum("ij,ij->i", targets, targets)
    mean_distance = np.empty(n, dtype=np.float64)
    chunk = 2048
    for start in range(0, n, chunk):
        block = x[start : start + chunk]
        block_sq = np.einsum("ij,ij->i", block, block)
        d2 = block_sq[:, None] + target_sq[None, :] - 2.0 * (block @ targets.T)
        mean_distance[start : start + chunk] = np.sqrt(np.maximum(d2, 0.0)).mean(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        return mean_distance / dist[:, 0]


def k_occurrence(idx: np.ndarray, n: int, k_hub: int) -> np.ndarray:
    """How often each point appears in other points' neighbour lists.

    Reuses the leading columns of the k-NN cache, so k_hub must not exceed
    the k the cache was built with. Ten is the convention in the hubness
    literature and is what the report passes.
    """
    if k_hub > idx.shape[1]:
        raise ValueError(f"k_hub={k_hub} exceeds cached neighbours {idx.shape[1]}")
    return np.bincount(idx[:, :k_hub].ravel(), minlength=n)


def hubness_skew(counts: np.ndarray) -> float:
    """Skewness of the k-occurrence distribution.

    Zero means every point is drawn on about equally. Large positive values
    mean a few hubs dominate the neighbour lists, which is what degrades
    graph indexes like HNSW -- searches funnel into the hubs and stall. A
    generator has no direct training pressure to reproduce this, so it is a
    property worth checking explicitly.
    """
    x = np.asarray(counts, dtype=np.float64)
    spread = x.std()
    if spread <= 0.0:
        return 0.0
    return float(np.mean(((x - x.mean()) / spread) ** 3))


def hubness_gini(counts: np.ndarray) -> float:
    """Gini coefficient of the k-occurrence distribution.

    A candidate replacement for `hubness_skew`, which is a third moment and
    is therefore set by whichever handful of tail hubs happened to land in
    the draw -- 108% of its own mean across eight draws of real GloVe. The
    Gini reads the whole distribution rather than its tail, so no small set
    of points can move it far.

    0.0 means every point is drawn on equally; (n - 1) / n means one point
    takes every neighbour slot. Same helper the IVF cell balance uses, which
    is deliberate: two lopsidedness measures should not disagree about what
    lopsided means.
    """
    return gini(np.asarray(counts, dtype=np.float64))


def hub_share_top1pct(counts: np.ndarray) -> float:
    """Fraction of all neighbour slots taken by the top 1% of points.

    The other candidate replacement for `hubness_skew`, and the one that
    reads most directly as the thing that hurts a graph index: how much of
    the neighbour traffic funnels into hubs. Bounded in [0.01, 1] -- 0.01
    when every point is drawn on equally, 1.0 when the top 1% take
    everything.

    The top slice rounds up, so it is never empty on a small set.
    """
    x = np.asarray(counts, dtype=np.float64)
    total = x.sum()
    if total <= 0.0:
        return 0.0
    top = max(1, math.ceil(0.01 * x.size))
    return float(np.sort(x)[-top:].sum() / total)


def cell_occupancy(x: np.ndarray, nlist: int, seed: int) -> tuple[np.ndarray, int]:
    """Points per cluster under a k-means partition, sorted ascending.

    Stands in for how an IVF index would carve up the set. Each set is
    clustered independently, because an index would be built on whichever
    set was actually shipped -- clustering real and reusing its centroids
    would measure coverage instead of balance.

    nlist is clamped to n // 2 so a small set cannot ask for more cells than
    it can meaningfully fill. Returns the clamped value alongside the counts
    so the caller can report it.
    """
    n = x.shape[0]
    nlist_eff = max(2, min(nlist, n // 2))
    kmeans = MiniBatchKMeans(
        n_clusters=nlist_eff,
        random_state=seed,
        n_init=3,
        batch_size=1024,
    )
    labels = kmeans.fit_predict(x)
    return np.sort(np.bincount(labels, minlength=nlist_eff)), nlist_eff


@dataclass
class AnnMetrics:
    """Everything one set contributes to the difficulty panels.

    lid and relative_contrast are aligned: both carry one entry per query
    that survived survivor_mask, in the same order.
    """

    lid: np.ndarray
    relative_contrast: np.ndarray
    k_occurrence: np.ndarray
    cell_occupancy: np.ndarray
    num_rows: int
    k: int
    nlist: int
    discarded_queries: int


def _subsample(x: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Cut to max_rows (0 = keep all).

    Deliberately duplicated from `eda.series.subsample` rather than imported:
    this module must not depend on the report. It is five lines and the
    dependency direction is worth more than the sharing.
    """
    if max_rows <= 0 or x.shape[0] <= max_rows:
        return x
    rng = np.random.default_rng(seed)
    return x[np.sort(rng.choice(x.shape[0], size=max_rows, replace=False))]


def compute(
    x: np.ndarray,
    *,
    k: int = 100,
    k_hub: int = 10,
    nlist: int = 256,
    max_rows: int = 20000,
    seed: int = 42,
    backend: str = "sklearn",
    chunk_rows: int = 1024,
) -> AnnMetrics:
    """Measure every difficulty metric for one set off a single k-NN pass.

    Callers must pass the same max_rows for every set they intend to compare:
    LID, relative contrast and hubness all drift with sample count, so
    unequal N makes the overlay meaningless.
    """
    x = np.ascontiguousarray(_subsample(x, max_rows, seed), dtype=np.float32)
    n = x.shape[0]

    dist, idx, k_eff = knn(x, k, backend=backend, chunk_rows=chunk_rows)
    survivors = survivor_mask(dist)

    lid = lid_mle(dist[survivors])
    contrast = relative_contrast(x, dist, seed)[survivors]
    counts = k_occurrence(idx, n, min(k_hub, k_eff))
    occupancy, nlist_eff = cell_occupancy(x, nlist, seed)

    return AnnMetrics(
        lid=lid,
        relative_contrast=contrast,
        k_occurrence=counts,
        cell_occupancy=occupancy,
        num_rows=n,
        k=k_eff,
        nlist=nlist_eff,
        discarded_queries=int((~survivors).sum()),
    )


def summary(m: AnnMetrics) -> dict[str, float | int | None]:
    """Scalars for the report's statistics table and summary.json.

    lid_median and relative_contrast_median are None when every query was
    discarded, which happens only for a fully degenerate set. Callers must
    render None rather than assuming a float.

    lid_discarded_queries is a count, so it is an int and not a float: at a
    million rows `format(1200000.0, '.6g')` renders `1.2e+06`, which reads as
    a measurement rather than a tally.
    """
    has_queries = m.lid.size > 0
    return {
        "lid_median": float(np.median(m.lid)) if has_queries else None,
        "relative_contrast_median": (
            float(np.median(m.relative_contrast)) if has_queries else None
        ),
        "hubness_skew": hubness_skew(m.k_occurrence),
        "ivf_gini": gini(m.cell_occupancy),
        "lid_discarded_queries": int(m.discarded_queries),
    }
