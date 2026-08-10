"""Seed-to-seed noise floor for a family's ANN-difficulty statistics.

`eda_report` measures a set of series under one set of conditions and writes
`summary.json`. When those series are the *same* configuration trained at
different seeds, the spread between them is the floor: a band tighter than it
is unenforceable, and a ladder rung whose improvement is smaller than it is
indistinguishable from a reseed.

Both noise floors this repo has published were computed by scripts that are
not in the tree, so neither can be reproduced from a pinned commit. This
module exists so the third one can be.

Like `check_gate`, it deliberately does not import from `src.eval.eda`: the
arithmetic must run anywhere `summary.json` can be copied to, without plotly
and without loading any vectors.
"""

from __future__ import annotations

import statistics
from typing import Sequence

# Named exactly as `src/eval/ann_difficulty.py::summary()` returns them, which
# is how they reach `stats` in summary.json.
GATE_STATISTICS = (
    "lid_median",
    "relative_contrast_median",
    "hubness_skew",
    "ivf_gini",
)

REAL_NAME = "real"


class NoiseFloorError(Exception):
    """The floor could not be computed at all -- bad summary or too few seeds.

    Distinct from a wide floor, which is a result.
    """


def summarize_spread(values: Sequence[float]) -> dict[str, float]:
    """Spread of one statistic across seeds.

    Key names and conventions match docs/datasets/glove_noise_floor.json so
    the real-side and synthetic-side floors can be read against each other
    without a translation step: sample standard deviation (ddof=1), and a
    range expressed as a percentage of the mean.
    """
    if len(values) < 2:
        raise NoiseFloorError(
            f"spread needs at least two values, got {len(values)}"
        )
    mean = statistics.fmean(values)
    low, high = min(values), max(values)
    std = statistics.stdev(values)
    return {
        "mean": mean,
        "std": std,
        "min": low,
        "max": high,
        "range_pct_of_mean": (high - low) / mean * 100.0,
        "cv_pct": std / mean * 100.0,
    }
