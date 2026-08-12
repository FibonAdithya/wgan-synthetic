"""Scoring for the ANN benchmark: recall, throughput, and curve interpolation.

Pure numpy. Nothing here touches a device, a file, or an index, which is what
makes the whole scoring path testable on a CPU-only box with no cuVS.

Recall is measured against ground-truth *distances*, not ids. SIFT descriptors
are quantized onto a lattice, so exact ties and true duplicates are common and
dominate the top of any neighbour list -- see `docs/datasets/sift.md`. Under
id-based recall an index that returns a different but exactly equidistant
point is scored as a miss, which would understate every corpus in this
benchmark and understate the real one most of all.
"""

from __future__ import annotations

from collections.abc import Sequence

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


def qps_at_recall(points: Sequence[tuple[float, float]], target: float) -> float | None:
    """QPS where a (recall, qps) curve crosses `target` recall.

    Returns None when the curve never reaches the target. That is a real
    result -- an index that cannot hit 0.90 on a corpus is the most
    interesting thing the table can say -- so it is reported rather than
    replaced with the nearest point or an extrapolation.

    Interpolation is linear in log(qps) because QPS spans orders of magnitude
    across a sweep while recall does not.
    """
    ordered = sorted((float(r), float(q)) for r, q in points)
    if not ordered:
        return None
    if ordered[-1][0] < target:
        return None
    if ordered[0][0] >= target:
        # Every measured point already clears the target; the fastest of them
        # is the lowest-recall one. Extrapolating past it would invent a
        # configuration that was never run.
        return ordered[0][1]

    for (r0, q0), (r1, q1) in zip(ordered, ordered[1:]):
        if r0 < target <= r1:
            if r1 == r0:
                return q1
            frac = (target - r0) / (r1 - r0)
            if q0 <= 0.0 or q1 <= 0.0:
                return float(q0 + frac * (q1 - q0))
            return float(np.exp(np.log(q0) + frac * (np.log(q1) - np.log(q0))))
    return None
