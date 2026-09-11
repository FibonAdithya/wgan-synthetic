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
    "zero_rows",
    "measured_rows",
)

# A checkpoint whose surviving queries are a minority is unmeasurable, the
# same principle as the None guard in selection_score below.
MAX_DISCARD_FRACTION = 0.5


def gate_statistics(
    x: np.ndarray, *, metric: str, seed: int
) -> dict[str, float | int | None]:
    """The four ANN-difficulty statistics of `x`, plus the LID discard count.

    Rows with an exact-zero L2 norm are dropped before measuring, on
    whichever side has them: the real NYTimes holdout carries exact-zero
    rows (`docs/datasets/nytimes.md`), and a zero row sits at the origin
    rather than on the sphere, which corrupts every one of these statistics
    -- on the shipped 12,500-row holdout it roughly halves the reference
    LID. The fake side structurally has none, because the trainer's
    `normalize_l2` clamps its divisor rather than dividing by zero. Exact
    duplicate rows are left alone; only exact zeros are a preprocessing
    artefact, not a legitimate query. `zero_rows` and `measured_rows` are
    logged so a run's metadata says how much of the holdout the statistics
    were actually measured on.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)
    zero_mask = norms == 0.0
    zero_rows = int(np.count_nonzero(zero_mask))
    measured = x[~zero_mask] if zero_rows else x
    m = ann_difficulty.compute(
        measured,
        k=GATE_K,
        k_hub=GATE_K_HUB,
        nlist=GATE_NLIST,
        max_rows=0,
        seed=seed,
        metric=metric,
    )
    s = ann_difficulty.summary(m)
    s["zero_rows"] = zero_rows
    s["measured_rows"] = int(measured.shape[0])
    return {k: s[k] for k in LOGGED}


def selection_score(
    fake: Mapping[str, float | int | None], real: Mapping[str, float | int | None]
) -> float:
    """Sum over LID and contrast of |fake - real| / real. Lower is better.

    Infinite when a statistic is missing or None on either side (every query
    discarded), or when the real value is zero: an unmeasurable checkpoint
    must never win by accident.

    Also infinite when the fake side carries both `lid_discarded_queries`
    and `measured_rows`, and either there were zero measured rows or more
    than `MAX_DISCARD_FRACTION` of them were discarded from the LID/contrast
    estimators: a checkpoint whose surviving queries are a minority is
    unmeasurable, the same principle as the None guard above. Either key
    absent skips the check, so hand-built stats dicts in existing tests stay
    valid.
    """
    discarded = fake.get("lid_discarded_queries")
    measured = fake.get("measured_rows")
    if discarded is not None and measured is not None:
        if measured == 0 or discarded / measured > MAX_DISCARD_FRACTION:
            return math.inf
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
