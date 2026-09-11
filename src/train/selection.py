"""Checkpoint selection for the trainer.

`best_generator.pt` used to be chosen by the smallest covariance Frobenius
gap (`cov_fro`). On NYTimes that chose step 1,000 of a 100,000-step run,
because the covariance gap is smallest for a tight cone, while the
statistics the project actually gates on (src/eval/ann_difficulty.py) got
worse for the rest of training. A generator built to fix local dimension
needs a selector that can see local dimension.

`gate` scores a checkpoint on the normalised gaps of the two statistics that
read local dimension -- LID median and relative contrast -- between the
fake holdout sample and the real holdout. Hubness skew and IVF Gini are
measured and logged but not scored: their noise floors are wider, and on a
near-isotropic corpus Gini mostly reflects k-means behaviour.

The holdout is smaller than the canonical N on a dataset page, so these
numbers rank checkpoints of one run against each other; they are not the
family's profile and must not be copied into a dataset page.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from src.eval import ann_difficulty

SELECTORS = ("cov_fro", "gate")

# The gate's canonical k / k_hub / nlist (gates/*.yaml). max_rows=0 means
# "all rows": the caller passes the holdout, and the holdout is the sample.
GATE_K = 100
GATE_K_HUB = 10
GATE_NLIST = 256

SCORED = ("lid_median", "relative_contrast_median")
LOGGED = (
    "lid_median",
    "relative_contrast_median",
    "hubness_skew",
    "ivf_gini",
    "lid_discarded_queries",
)


def gate_statistics(
    x: np.ndarray, *, metric: str, seed: int
) -> dict[str, float | int | None]:
    """The four ANN-difficulty statistics of `x`, plus the LID discard count."""
    m = ann_difficulty.compute(
        np.ascontiguousarray(x, dtype=np.float32),
        k=GATE_K,
        k_hub=GATE_K_HUB,
        nlist=GATE_NLIST,
        max_rows=0,
        seed=seed,
        metric=metric,
    )
    s = ann_difficulty.summary(m)
    return {k: s[k] for k in LOGGED}


def selection_score(
    fake: Mapping[str, float | int | None], real: Mapping[str, float | int | None]
) -> float:
    """Sum over LID and contrast of |fake - real| / real. Lower is better.

    Infinite when a statistic is missing or None on either side (every query
    discarded), or when the real value is zero: an unmeasurable checkpoint
    must never win by accident.
    """
    total = 0.0
    for key in SCORED:
        f, r = fake.get(key), real.get(key)
        if f is None or r is None or r == 0:
            return math.inf
        total += abs(float(f) - float(r)) / abs(float(r))
    return total


def prefixed(
    stats: Mapping[str, float | int | None], prefix: str
) -> dict[str, float | int | None]:
    return {prefix + k: v for k, v in stats.items()}
