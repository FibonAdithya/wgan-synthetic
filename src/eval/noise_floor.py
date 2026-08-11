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

import argparse
import json
import statistics
import sys
from pathlib import Path
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


def _check_conditions(real_name: str, real_entry: dict, name: str, entry: dict) -> None:
    """Raise unless `entry` was measured under the same conditions as `real_entry`.

    `src/eval/eda/metrics.py` records post-clamp actuals precisely because a
    series with fewer rows than requested gets its k and nlist clamped, and
    `src/eval/check_gate.py` already guards a single run against the gate's
    canonical conditions for the same reason. A floor spread across series
    measured at different `ann_measured_rows` / `ann_measured_k` /
    `ann_measured_nlist` would be recorded as if it were one measurement, and
    the underlying statistics are not comparable across those conditions.
    """
    for key in CONDITION_KEYS:
        expected = real_entry.get(key)
        actual = entry.get(key)
        if actual != expected:
            raise NoiseFloorError(
                f"series {name!r} was measured at {key}={actual!r}, but "
                f"{real_name!r} was measured at {key}={expected!r}; a floor "
                "needs every series measured under the same conditions"
            )


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
    # bool is a subclass of int, so isinstance(value, (int, float)) alone
    # would let True/False through as 1.0/0.0. And float() on its own accepts
    # more than a summary.json should ever contain: it silently turns a
    # numeric string like "1.5" into 1.5, and raises a bare ValueError or
    # TypeError -- not a NoiseFloorError -- on "abc" or a list, which main()
    # below would otherwise let escape as a traceback instead of the clean
    # stderr line every other bad-input path here produces.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NoiseFloorError(
            f"series {entry.get('name')!r} has {statistic!r} = {value!r}, "
            f"a {type(value).__name__}, not a number"
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
    for name, entry in zip(series_names, entries):
        _check_conditions(real_name, real_entry, name, entry)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="summary.json written by eda_report, holding every series.",
    )
    parser.add_argument(
        "--series",
        type=str,
        action="append",
        required=True,
        metavar="NAME",
        help=(
            "Repeatable. One --series per seeded run, named as it was labelled "
            "in eda_report's --synthetic-path LABEL=PATH."
        ),
    )
    parser.add_argument(
        "--real-name",
        type=str,
        default=REAL_NAME,
        help=f"Series to measure the distance against. Default {REAL_NAME!r}.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Also write the JSON floor to this path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        floor = compute_floor(summary, args.series, real_name=args.real_name)
    except (NoiseFloorError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        # stderr, so stdout stays parseable as JSON or empty, never half a
        # report.
        print(f"noise_floor: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(floor, indent=2), encoding="utf-8")

    print(json.dumps(floor, indent=2))


if __name__ == "__main__":
    main()
