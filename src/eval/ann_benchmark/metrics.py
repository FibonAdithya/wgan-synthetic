"""Scoring for the ANN benchmark: recall, throughput, and curve interpolation.

Pure numpy. Nothing here touches a device, a file, or an index, which is what
makes the whole scoring path testable on a CPU-only box with no cuVS.

Recall is measured against ground-truth *distances*, not ids. SIFT descriptors
are quantized onto a lattice, so exact ties and true duplicates are common and
dominate the top of any neighbour list -- see `docs/datasets/sift.md`. Under
id-based recall an index that returns a different but exactly equidistant
point is scored as a miss, which would understate every corpus in this
benchmark and understate the real one most of all.

Every distance `recall_at_k` sees must be in the same space as the ground
truth: exact squared L2 to the stored vectors. Some indexes (IVF-PQ's
asymmetric distance computation, most notably) report distances computed
against a compressed or otherwise approximate representation, not against the
stored vectors -- measured on the box, that inflates IVF-PQ's recall by up to
13.7 points at high probe counts. `recompute_exact_distances` is what closes
that gap: it throws away whatever distances an adapter reported and
recomputes the true ones from the corpus vectors, using only the ids the
adapter returned. It is applied uniformly to every adapter, not just IVF-PQ,
so every row in the benchmark is scored the same way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def recall_at_k(
    found_distances: np.ndarray,
    truth_distances: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Fraction of returned neighbours that are as close as the true k-th.

    Both arrays are `(num_queries, k)`, sorted ascending, and must be in the
    same distance space -- squared-L2 throughout this package. `eps` is a
    relative tolerance absorbing float error in the tie comparison; it is not
    an accuracy knob.
    """
    found = np.asarray(found_distances, dtype=np.float64)
    truth = np.asarray(truth_distances, dtype=np.float64)
    if found.shape != truth.shape:
        raise ValueError(
            f"found and truth distances must have the same shape, got "
            f"{found.shape} and {truth.shape}"
        )
    if found.ndim != 2 or found.shape[1] == 0:
        raise ValueError(f"expected a (num_queries, k) array, got {found.shape}")

    threshold = truth[:, -1:] * (1.0 + eps)
    return float(np.mean(found <= threshold))


def recompute_exact_distances(
    vectors: np.ndarray, queries: np.ndarray, ids: np.ndarray
) -> np.ndarray:
    """Squared-L2 distance from each query to the vectors an index returned.

    Takes only the *ids* an adapter's search produced -- never the distances
    it reported alongside them -- and recomputes distances directly from the
    stored vectors. This is what makes `recall_at_k` comparable across
    adapters: a search implementation that scores candidates in a compressed
    or quantized space (IVF-PQ's asymmetric distance computation) would
    otherwise get to grade its own homework.

    `ids` is `(num_queries, k)`; the result is the same shape, sorted
    ascending per query so it can be passed to `recall_at_k` directly. This
    is scoring, not search: it belongs outside every timed region, and at
    10k queries x 10 ids x 128 dims the cost is trivial in plain numpy.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    ids = np.asarray(ids)
    if queries.ndim != 2 or ids.ndim != 2:
        raise ValueError(
            f"expected queries (num_queries, dim) and ids (num_queries, k), "
            f"got {queries.shape} and {ids.shape}"
        )
    if queries.shape[0] != ids.shape[0]:
        raise ValueError(
            f"queries and ids must agree on num_queries, got {queries.shape[0]} "
            f"and {ids.shape[0]}"
        )
    selected = vectors[ids]  # (num_queries, k, dim)
    diff = selected - queries[:, None, :]
    distances = np.einsum("qkd,qkd->qk", diff, diff)
    return np.sort(distances, axis=1).astype(np.float32)


def qps(num_queries: int, seconds: float) -> float:
    """Queries per second."""
    if seconds <= 0.0:
        raise ValueError(f"elapsed time must be positive, got {seconds}")
    return float(num_queries) / float(seconds)


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Min, median and p95 of repeated timings.

    Same three figures `src/sample/benchmark.py` reports, for the same reason:
    min is the machine's ceiling, median the typical case, p95 what a deadline
    should be budgeted against.
    """
    if len(values) == 0:
        raise ValueError("summarize needs at least one value")
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


@dataclass(frozen=True)
class RecallPoint:
    """One QPS figure plus the recall it was actually evaluated at.

    `interpolated=True` means `recall == target`, exactly: `qps` was found by
    interpolating between two measured points that bracket the target.
    `interpolated=False` means every measured point already cleared the
    target and `qps`/`recall` are the fastest (lowest-recall) point actually
    run -- a floor, not a match. A reader must not be able to tell those two
    cases apart by looking at `qps` alone, which is why `recall` and
    `interpolated` travel with it rather than being reconstructed later.
    """

    qps: float
    recall: float
    interpolated: bool


def qps_at_recall(
    points: Sequence[tuple[float, float]], target: float
) -> RecallPoint | None:
    """QPS where a (recall, qps) curve crosses `target` recall.

    Returns None when the curve never reaches the target. That is a real
    result -- an index that cannot hit 0.90 on a corpus is the most
    interesting thing the table can say -- so it is reported rather than
    replaced with the nearest point or an extrapolation.

    When every measured point already clears the target, the fastest of them
    (the lowest-recall one) is returned as a floor -- `RecallPoint.recall` is
    that point's true recall, not `target`, and `interpolated` is False.
    Extrapolating past it would invent a configuration that was never run;
    every CAGRA sweep in this benchmark hits this branch (CAGRA's cheapest
    knob, `itopk_size=32`, already clears 0.90 recall on every corpus), so
    silently mislabeling the floor as a match at `target` is the exact bug
    this type exists to make impossible.

    Points sharing a recall are first collapsed to their Pareto-best (the
    max QPS at that recall) -- standard ann-benchmarks practice. Several
    configurations can land on the same measured recall, and only the
    fastest of them is what the sweep actually achieved there; keeping a
    slower duplicate around would let it shadow the faster one, whether it
    lands inside the interpolation bracket or as the fastest already-passing
    point.

    Interpolation is linear in log(qps) because QPS spans orders of magnitude
    across a sweep while recall does not.
    """
    if not points:
        return None

    best_qps_by_recall: dict[float, float] = {}
    for r, q in points:
        r, q = float(r), float(q)
        if r not in best_qps_by_recall or q > best_qps_by_recall[r]:
            best_qps_by_recall[r] = q
    ordered = sorted(best_qps_by_recall.items())

    if ordered[-1][0] < target:
        return None
    if ordered[0][0] >= target:
        # Every measured point already clears the target; the fastest of them
        # is the lowest-recall one. Extrapolating past it would invent a
        # configuration that was never run -- report it as the floor it is,
        # at the recall it was actually measured at.
        r0, q0 = ordered[0]
        return RecallPoint(qps=q0, recall=r0, interpolated=False)

    for (r0, q0), (r1, q1) in zip(ordered, ordered[1:]):
        if r0 < target <= r1:
            # `ordered` holds one entry per distinct recall (deduplicated
            # above), so r1 == r0 can't happen here.
            frac = (target - r0) / (r1 - r0)
            if q0 <= 0.0 or q1 <= 0.0:
                interpolated_qps = float(q0 + frac * (q1 - q0))
            else:
                interpolated_qps = float(
                    np.exp(np.log(q0) + frac * (np.log(q1) - np.log(q0)))
                )
            return RecallPoint(qps=interpolated_qps, recall=target, interpolated=True)
    return None
