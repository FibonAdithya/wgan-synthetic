"""The numbers behind the report, with no plotly in sight.

Kept separate from `figures` so a statistic can be tested without rendering
anything, and so a reader can see what is measured without reading how it is
drawn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.eval import ann_difficulty
from src.eval.eda.series import Series, subsample


def pairwise_distance_sample(x: np.ndarray, num_pairs: int, seed: int) -> np.ndarray:
    """Euclidean distances over randomly drawn distinct pairs."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    return np.linalg.norm(x[i] - x[j], axis=1)


def nn_distances(x: np.ndarray, k: int, seed: int, max_rows: int) -> np.ndarray:
    """Distance to the k-th nearest *other* point within the same set.

    Collapsed generators put mass on a few modes, which shows up as a
    within-set NN distance distribution shifted far below the real one. All
    sets are cut to the same max_rows first: k-NN distance shrinks as sample
    count grows, so unequal N would make the comparison meaningless.
    """
    sub = subsample(x, max_rows, seed)
    nn = NearestNeighbors(n_neighbors=min(k + 1, sub.shape[0]))
    nn.fit(sub)
    dist, _ = nn.kneighbors(sub)
    return dist[:, -1]


def wasserstein1(a: np.ndarray, b: np.ndarray, num_quantiles: int = 512) -> float:
    """1-D Wasserstein-1 via quantile functions; avoids a scipy dependency."""
    q = np.linspace(0.0, 1.0, num_quantiles)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


@dataclass(frozen=True)
class DimDivergence:
    """Per-dimension W1 against real, plus the shared plotting order.

    `distances` is one array per synthetic series; `order` is the single
    dimension ordering every series is drawn in, worst first, so bars line
    up across series; `worst` is the top-k slice that reaches summary.json.
    """

    distances: dict[str, np.ndarray]
    order: np.ndarray
    worst: dict[str, list[dict]]


def dimension_divergence(series: Sequence[Series], top_k: int) -> DimDivergence:
    """Rank dimensions by 1-D Wasserstein distance from real, per synthetic set.

    Dimensions are ordered by the worst mismatch across all synthetics, so the
    same x-axis ordering applies to every series and they stay comparable.

    Requires one series with `is_real` True and at least one without: raises
    `StopIteration` if no series is real, `ValueError` if none are synthetic.
    """
    real = next(s for s in series if s.is_real)
    synths = [s for s in series if not s.is_real]
    dim = real.x.shape[1]

    distances = {
        s.name: np.array([wasserstein1(real.x[:, d], s.x[:, d]) for d in range(dim)])
        for s in synths
    }
    worst_overall = np.max(np.stack(list(distances.values())), axis=0)
    order = np.argsort(worst_overall)[::-1]
    worst = {
        name: [{"dim": int(d), "wasserstein1": float(v[d])} for d in order[:top_k]]
        for name, v in distances.items()
    }
    return DimDivergence(distances=distances, order=order, worst=worst)


def effective_rank(x: np.ndarray) -> float:
    """exp(Shannon entropy of the explained-variance spectrum).

    Reads as "how many directions meaningfully carry variance": equals the
    dimension count when variance is spread evenly and 1 when it all sits on a
    single direction. Note this uses variance ratios, not the normalized
    singular values of Roy & Vetterli, so absolute values are not comparable
    with that definition -- only across sets measured here.
    """
    ratio = (
        PCA(n_components=min(x.shape[1], x.shape[0])).fit(x).explained_variance_ratio_
    )
    return float(np.exp(-np.sum(ratio * np.log(ratio + 1.0e-12))))


def summary_stats(
    s: Series,
    knn: int,
    num_pairs: int,
    seed: int,
    knn_max_rows: int,
    metrics: ann_difficulty.AnnMetrics,
) -> dict:
    norms = np.linalg.norm(s.x, axis=1)
    stats = {
        "name": s.name,
        "num_vectors": int(s.x.shape[0]),
        "dim": int(s.x.shape[1]),
        "value_mean": float(s.x.mean()),
        "value_std": float(s.x.std()),
        "value_min": float(s.x.min()),
        "value_max": float(s.x.max()),
        "exact_zero_fraction": float((s.x == 0.0).mean()),
        "negative_fraction": float((s.x < 0.0).mean()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "duplicate_row_fraction": float(
            1.0 - np.unique(s.x, axis=0).shape[0] / s.x.shape[0]
        ),
        "median_pairwise_distance": float(
            np.median(pairwise_distance_sample(s.x, num_pairs, seed))
        ),
        f"median_{knn}nn_distance": float(
            np.median(nn_distances(s.x, knn, seed, knn_max_rows))
        ),
        "effective_rank": effective_rank(s.x),
    }
    stats.update(ann_difficulty.summary(metrics))
    # Actual (post-clamp) measurement conditions, not the requested ones: a
    # series with fewer rows than --ann-max-rows gets its k and nlist clamped
    # inside knn()/cell_occupancy(), and its num_vectors above is the
    # PRE-truncation count. Without these, nothing records what a series was
    # actually measured under, and the report's section notes cannot tell a
    # reader when conditions diverge across series.
    stats["ann_measured_rows"] = metrics.num_rows
    stats["ann_measured_k"] = metrics.k
    stats["ann_measured_nlist"] = metrics.nlist
    return stats
