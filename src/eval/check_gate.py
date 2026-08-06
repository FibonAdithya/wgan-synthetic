"""Machine-readable gate check for a finished run.

`eda_report` writes an HTML report a human reads and a `summary.json` a
program can read. This module turns the second one into a verdict: it takes a
run directory and a `gates/<dataset>.yaml` band file, compares the four
ANN-difficulty statistics against their bands, prints JSON, and exits non-zero
when the run does not pass. An agent that trains a rung can then tell whether
it succeeded without a human opening a browser.

Like `ann_difficulty`, this deliberately does not import from `src.eval.eda`:
the check must run anywhere `summary.json` can be copied to, without plotly
and without loading any vectors.

Exit codes
----------
0   verdict "pass" -- every statistic sits inside its band. Also returned for
    verdict "unset" when --allow-unset is passed.
1   verdict "fail" -- a statistic is outside its band, is missing from
    summary.json, is None (every query was discarded), or the run was measured
    under conditions the gate file does not consider comparable.
2   verdict "unset" -- no statistic failed, but at least one band is null, so
    the gate cannot be enforced. This is the current state of every family: a
    gate nobody has calibrated must not exit 0 pretending it passed.

Usage
-----
    python -m src.eval.check_gate --dataset sift --run-dir runs/sift/profile
    python -m src.eval.check_gate --gate-file gates/sift.yaml \
        --run-dir runs/sift/profile --stats-name v2 --allow-unset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Named exactly as ann_difficulty.summary() returns them. Order is the order
# they are reported in.
GATE_STATISTICS = (
    "lid_median",
    "relative_contrast_median",
    "hubness_skew",
    "ivf_gini",
)

# gate key -> key recording the post-clamp actual in a summary.json stats
# entry. k_hub is absent on purpose: eda_report records no post-clamp actual
# for the hubness depth, so there is nothing to compare it against.
CONDITION_KEYS = {
    "n": "ann_measured_rows",
    "k": "ann_measured_k",
    "nlist": "ann_measured_nlist",
}

REAL_NAME = "real"

GATES_DIR = Path(__file__).resolve().parents[2] / "gates"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_UNSET = 2


class GateError(Exception):
    """The check could not be run at all -- bad gate file or bad run dir.

    Distinct from a failing verdict: a run that fails the gate is a result,
    while a gate file that does not parse is a bug in the caller's inputs.
    """


def load_gate(path: Path) -> dict[str, Any]:
    """Parse and validate a gates/<dataset>.yaml file.

    Validating here rather than at use means a typo in a statistic name is an
    error instead of a silently skipped check.
    """
    if not path.exists():
        raise GateError(f"gate file not found: {path}")
    try:
        gate = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise GateError(f"could not parse {path}: {exc}") from exc
    if not isinstance(gate, dict):
        raise GateError(f"gate file is not a mapping: {path}")

    canonical = gate.get("canonical")
    if not isinstance(canonical, dict):
        raise GateError(f"gate file has no 'canonical' mapping: {path}")
    missing_conditions = sorted(set(CONDITION_KEYS) - set(canonical))
    if missing_conditions:
        raise GateError(
            f"gate file {path} is missing canonical conditions: "
            f"{', '.join(missing_conditions)}"
        )

    statistics = gate.get("statistics")
    if not isinstance(statistics, dict):
        raise GateError(f"gate file has no 'statistics' mapping: {path}")
    missing = [name for name in GATE_STATISTICS if name not in statistics]
    if missing:
        raise GateError(f"gate file {path} is missing bands for: {', '.join(missing)}")
    unknown = sorted(set(statistics) - set(GATE_STATISTICS))
    if unknown:
        raise GateError(
            f"gate file {path} has bands for unknown statistics: {', '.join(unknown)}"
        )
    for name in GATE_STATISTICS:
        band = statistics[name]
        if band is None:
            continue
        if not isinstance(band, dict):
            raise GateError(f"band for {name} in {path} is not a mapping or null")
        stray = sorted(set(band) - {"min", "max"})
        if stray:
            raise GateError(
                f"band for {name} in {path} has unknown keys: {', '.join(stray)}"
            )
        for bound in ("min", "max"):
            value = band.get(bound)
            if value is None or isinstance(value, bool):
                if isinstance(value, bool):
                    raise GateError(
                        f"band {bound} for {name} in {path} is a boolean, not a number"
                    )
                continue
            if not isinstance(value, (int, float)):
                raise GateError(
                    f"band {bound} for {name} in {path} is not a number: {value!r}"
                )
        # An inverted band can never be satisfied, so it would read as a
        # statistic that is permanently broken rather than as the typo it is.
        low, high = band.get("min"), band.get("max")
        if low is not None and high is not None and float(low) > float(high):
            raise GateError(
                f"band for {name} in {path} is inverted: min {low!r} > max {high!r}"
            )
    return gate


def band_bounds(band: Any) -> tuple[float | None, float | None]:
    """Normalise a band entry to a (min, max) pair, either bound optional.

    A band is unset when both bounds are None, which a null entry and an
    explicit `{min: null, max: null}` both mean.
    """
    if not isinstance(band, dict):
        return (None, None)
    low = band.get("min")
    high = band.get("max")
    return (
        None if low is None else float(low),
        None if high is None else float(high),
    )


def load_stats_entry(run_dir: Path, stats_name: str) -> dict[str, Any]:
    """Pull one named entry out of `<run_dir>/summary.json`'s `stats` list.

    The gate statistics live here and nowhere else -- not in
    run_metadata.json, not in eval/metrics.json -- because eda_report merges
    ann_difficulty.summary() into the per-series stats row.
    """
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise GateError(
            f"no summary.json in {run_dir}; the gate statistics are written by "
            "src.eval.eda_report, so run it against this run first"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateError(f"could not parse {summary_path}: {exc}") from exc
    stats = summary.get("stats")
    if not isinstance(stats, list):
        raise GateError(f"{summary_path} has no 'stats' list")

    for entry in stats:
        if isinstance(entry, dict) and entry.get("name") == stats_name:
            return entry
    available = (
        ", ".join(str(e.get("name")) for e in stats if isinstance(e, dict)) or "(none)"
    )
    raise GateError(
        f"{summary_path} has no stats entry named {stats_name!r}; available: {available}"
    )


def check_statistic(name: str, entry: dict[str, Any], band: Any) -> dict[str, Any]:
    """Compare one statistic against one band.

    Missing and None values fail rather than being skipped, and they fail even
    when the band is unset: summary() returns None for lid_median and
    relative_contrast_median when every query was discarded, which is a
    degenerate set, and a degenerate set must never come out of this checker
    looking clean.
    """
    low, high = band_bounds(band)
    result: dict[str, Any] = {
        "statistic": name,
        "value": None,
        "band": {"min": low, "max": high},
    }

    if name not in entry:
        result["status"] = "fail"
        result["reason"] = f"{name} is absent from the run's summary.json stats entry"
        return result

    value = entry[name]
    if value is None:
        result["status"] = "fail"
        result["reason"] = (
            f"{name} is null, which ann_difficulty.summary() returns only when "
            "every query was discarded"
        )
        return result

    try:
        result["value"] = float(value)
    except (TypeError, ValueError):
        # Fail rather than raise: a garbled statistic is a property of the run
        # being checked, so it belongs in the verdict alongside the others.
        result["status"] = "fail"
        result["reason"] = f"{name} is not a number: {value!r}"
        return result

    if low is None and high is None:
        result["status"] = "unset"
        result["reason"] = "no band is set for this statistic yet"
        return result

    if low is not None and result["value"] < low:
        result["status"] = "fail"
        result["reason"] = f"{result['value']!r} is below the band minimum {low!r}"
        return result
    if high is not None and result["value"] > high:
        result["status"] = "fail"
        result["reason"] = f"{result['value']!r} is above the band maximum {high!r}"
        return result

    result["status"] = "pass"
    return result


def check_conditions(
    canonical: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any]:
    """Compare the run's post-clamp measurement conditions to the canonical ones.

    A set truncated below --ann-max-rows gets its k and nlist clamped, so the
    actuals here are what the statistics were really measured under. Comparing
    the requested settings instead would call a clamped run comparable when it
    is not.
    """
    checks = []
    for gate_key, stats_key in CONDITION_KEYS.items():
        expected = canonical[gate_key]
        actual = entry.get(stats_key)
        checks.append(
            {
                "condition": gate_key,
                "summary_key": stats_key,
                "expected": expected,
                "actual": actual,
                "match": actual is not None and actual == expected,
            }
        )
    mismatched = [c["condition"] for c in checks if not c["match"]]
    return {
        "match": not mismatched,
        "mismatched": mismatched,
        "checks": checks,
        "note": (
            "k_hub is not checked: eda_report records no post-clamp actual for "
            "the hubness neighbour depth."
        ),
    }


def evaluate(
    gate: dict[str, Any],
    entry: dict[str, Any],
    *,
    run_dir: Path,
    gate_file: Path,
    stats_name: str,
    allow_unset: bool,
    allow_condition_mismatch: bool,
) -> dict[str, Any]:
    """Build the full report and its verdict. Pure -- no I/O, no exiting."""
    statistics = [
        check_statistic(name, entry, gate["statistics"][name])
        for name in GATE_STATISTICS
    ]
    conditions = check_conditions(gate["canonical"], entry)

    statuses = {s["status"] for s in statistics}
    failures = [s["statistic"] for s in statistics if s["status"] == "fail"]
    unset = [s["statistic"] for s in statistics if s["status"] == "unset"]

    conditions_fail = not conditions["match"] and not allow_condition_mismatch

    if failures or conditions_fail:
        verdict = "fail"
    elif "unset" in statuses:
        verdict = "unset"
    else:
        verdict = "pass"

    if verdict == "fail":
        exit_code = EXIT_FAIL
    elif verdict == "unset":
        exit_code = EXIT_PASS if allow_unset else EXIT_UNSET
    else:
        exit_code = EXIT_PASS

    reasons = []
    if failures:
        reasons.append(f"outside band or unusable: {', '.join(failures)}")
    if not conditions["match"]:
        detail = ", ".join(conditions["mismatched"])
        reasons.append(
            f"measured under non-canonical conditions: {detail}"
            + (" (allowed)" if allow_condition_mismatch else "")
        )
    if unset:
        reasons.append(
            f"no band set for: {', '.join(unset)}"
            + (" (allowed)" if allow_unset else "")
        )

    return {
        "verdict": verdict,
        "exit_code": exit_code,
        "dataset": gate.get("dataset"),
        "run_dir": str(run_dir),
        "gate_file": str(gate_file),
        "stats_name": stats_name,
        "allow_unset": allow_unset,
        "allow_condition_mismatch": allow_condition_mismatch,
        "statistics": statistics,
        "conditions": conditions,
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset family name; reads the band file at gates/<dataset>.yaml.",
    )
    source.add_argument(
        "--gate-file",
        type=str,
        default=None,
        help="Path to a band file, for a gate that does not live under gates/.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help=(
            "Run directory holding summary.json, as written by "
            "src.eval.eda_report --output-dir."
        ),
    )
    parser.add_argument(
        "--stats-name",
        type=str,
        default=REAL_NAME,
        help=(
            "Which entry of summary.json's stats list to check: 'real' for the "
            "reference profile, or a variant label for a synthetic set."
        ),
    )
    parser.add_argument(
        "--allow-unset",
        action="store_true",
        help=(
            "Exit 0 on verdict 'unset'. Without it an uncalibrated gate exits "
            "non-zero, so a null band can never read as a pass."
        ),
    )
    parser.add_argument(
        "--allow-condition-mismatch",
        action="store_true",
        help=(
            "Report non-canonical measurement conditions without failing on "
            "them. For exploratory runs only; the numbers are not comparable."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Also write the JSON report to this path.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Do the check and return the report. Callers decide what to do with it."""
    gate_file = (
        Path(args.gate_file)
        if args.gate_file is not None
        else GATES_DIR / f"{args.dataset}.yaml"
    )
    run_dir = Path(args.run_dir)

    gate = load_gate(gate_file)
    entry = load_stats_entry(run_dir, args.stats_name)
    report = evaluate(
        gate,
        entry,
        run_dir=run_dir,
        gate_file=gate_file,
        stats_name=args.stats_name,
        allow_unset=args.allow_unset,
        allow_condition_mismatch=args.allow_condition_mismatch,
    )

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> None:
    try:
        report = run(parse_args())
    except GateError as exc:
        # Not a verdict: the check never ran. Goes to stderr so stdout stays
        # parseable as JSON or empty, never half a report.
        print(f"check_gate: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_FAIL) from exc
    print(json.dumps(report, indent=2))
    raise SystemExit(report["exit_code"])


if __name__ == "__main__":
    main()
