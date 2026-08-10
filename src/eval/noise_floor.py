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


# summary.json key -> the name this module reports it under. Mirrors
# check_gate.CONDITION_KEYS: k_hub is absent because eda_report records no
# post-clamp actual for the hubness depth.
CONDITION_KEYS = {
    "ann_measured_rows": "n",
    "ann_measured_k": "k",
    "ann_measured_nlist": "nlist",
}


def _entry(summary: dict, name: str) -> dict:
    for entry in summary.get("stats", []):
        if entry.get("name") == name:
            return entry
    raise NoiseFloorError(f"no series named {name!r} in summary.json")


def _value(entry: dict, statistic: str) -> float:
    if statistic not in entry:
        raise NoiseFloorError(
            f"series {entry.get('name')!r} has no {statistic!r}"
        )
    value = entry[statistic]
    if value is None:
        # ann_difficulty writes null when every query was discarded. That is a
        # measurement that did not happen, and averaging it as 0.0 would put a
        # number nobody measured into a committed floor.
        raise NoiseFloorError(
            f"series {entry.get('name')!r} has {statistic!r} = null; "
            "the statistic was not measurable on that run"
        )
    return float(value)


def compute_floor(
    summary: dict,
    series_names: Sequence[str],
    *,
    real_name: str = REAL_NAME,
) -> dict:
    """Spread across seeds, and how far real sits from them in units of it.

    The last of those is the column that decides whether a future ladder rung
    could be told from a reseed at all.
    """
    if len(series_names) < 2:
        raise NoiseFloorError(
            f"a floor needs at least two series, got {len(series_names)}"
        )

    real_entry = _entry(summary, real_name)
    entries = [_entry(summary, name) for name in series_names]

    real: dict[str, float] = {}
    per_seed: list[dict[str, float]] = [{} for _ in entries]
    spread: dict[str, dict[str, float]] = {}
    distance_from_real: dict[str, float] = {}
    distance_in_spreads: dict[str, float | None] = {}

    for statistic in GATE_STATISTICS:
        values = [_value(entry, statistic) for entry in entries]
        for row, value in zip(per_seed, values):
            row[statistic] = value

        real_value = _value(real_entry, statistic)
        real[statistic] = real_value

        summarized = summarize_spread(values)
        spread[statistic] = summarized

        gap = summarized["mean"] - real_value
        distance_from_real[statistic] = gap

        spread_range = summarized["max"] - summarized["min"]
        distance_in_spreads[statistic] = (
            None if spread_range == 0.0 else abs(gap) / spread_range
        )

    return {
        "series": list(series_names),
        "real_name": real_name,
        "conditions": {
            reported: real_entry[key]
            for key, reported in CONDITION_KEYS.items()
            if key in real_entry
        },
        "real": real,
        "per_seed": per_seed,
        "spread": spread,
        "distance_from_real": distance_from_real,
        "distance_in_spreads": distance_in_spreads,
    }
