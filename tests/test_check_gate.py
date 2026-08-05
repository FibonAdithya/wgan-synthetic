"""Tests for the executable gate.

The point of `check_gate` is the exit code, so most of these assert on it
rather than only on the report body: a checker that prints "fail" and exits 0
is worse than no checker, because CI would go green on it.
"""

import json
from pathlib import Path

import pytest
import yaml

from src.eval import check_gate

CANONICAL = {"n": 20000, "k": 100, "k_hub": 10, "nlist": 256}

# Values inside the bands written by `write_gate` below. Not measured from
# anything -- these tests are about the comparison, not about SIFT.
PASSING_STATS = {
    "lid_median": 12.0,
    "relative_contrast_median": 1.8,
    "hubness_skew": 2.0,
    "ivf_gini": 0.35,
}

SET_BANDS = {
    "lid_median": {"min": 8.0, "max": 16.0},
    "relative_contrast_median": {"min": 1.5, "max": 2.5},
    "hubness_skew": {"min": 1.0, "max": 3.0},
    "ivf_gini": {"min": 0.2, "max": 0.5},
}

NULL_BANDS = {name: {"min": None, "max": None} for name in check_gate.GATE_STATISTICS}


@pytest.fixture
def write_gate():
    """Write a gate file with the given bands, defaulting to a calibrated one."""

    def _write(tmp_path: Path, bands=None, canonical=None, name="sift") -> Path:
        path = tmp_path / "gates" / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                {
                    "dataset": name,
                    "canonical": dict(CANONICAL if canonical is None else canonical),
                    "statistics": dict(SET_BANDS if bands is None else bands),
                }
            )
        )
        return path

    return _write


@pytest.fixture
def write_run():
    """Write a run dir holding a summary.json shaped like eda_report's."""

    def _write(tmp_path: Path, stats=None, conditions=None, name="real") -> Path:
        run_dir = tmp_path / "runs" / "profile"
        run_dir.mkdir(parents=True, exist_ok=True)
        measured = {
            "ann_measured_rows": CANONICAL["n"],
            "ann_measured_k": CANONICAL["k"],
            "ann_measured_nlist": CANONICAL["nlist"],
        }
        measured.update(conditions or {})
        entry = {"name": name, "num_vectors": 250000}
        entry.update(PASSING_STATS if stats is None else stats)
        entry.update(measured)
        summary = {"stats": [entry], "seed": 42}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return run_dir

    return _write


def make_args(gate_file: Path, run_dir: Path, **overrides):
    """Build the namespace `run` expects, so tests exercise the real entry point."""
    import argparse

    defaults = {
        "dataset": None,
        "gate_file": str(gate_file),
        "run_dir": str(run_dir),
        "stats_name": "real",
        "allow_unset": False,
        "allow_condition_mismatch": False,
        "output": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def statuses(report):
    return {s["statistic"]: s["status"] for s in report["statistics"]}


def test_check_gate_passes_and_exits_zero_when_every_statistic_is_inside_its_band(
    tmp_path: Path, write_gate, write_run
):
    report = check_gate.run(make_args(write_gate(tmp_path), write_run(tmp_path)))

    assert report["verdict"] == "pass", report["reasons"]
    assert report["exit_code"] == check_gate.EXIT_PASS
    assert set(statuses(report).values()) == {"pass"}


def test_check_gate_exits_non_zero_when_a_statistic_falls_outside_its_band(
    tmp_path: Path, write_gate, write_run
):
    stats = dict(PASSING_STATS, ivf_gini=0.9)

    report = check_gate.run(
        make_args(write_gate(tmp_path), write_run(tmp_path, stats=stats))
    )

    assert report["verdict"] == "fail"
    assert report["exit_code"] == check_gate.EXIT_FAIL
    assert statuses(report)["ivf_gini"] == "fail"
    assert statuses(report)["lid_median"] == "pass", (
        "one failing statistic must not contaminate the others -- the bands are "
        "per statistic precisely so the report says which one moved"
    )


def test_check_gate_fails_a_statistic_below_the_band_minimum(
    tmp_path: Path, write_gate, write_run
):
    stats = dict(PASSING_STATS, lid_median=1.0)

    report = check_gate.run(
        make_args(write_gate(tmp_path), write_run(tmp_path, stats=stats))
    )

    assert statuses(report)["lid_median"] == "fail"
    assert report["exit_code"] == check_gate.EXIT_FAIL


def test_check_gate_treats_the_band_bounds_as_inclusive(
    tmp_path: Path, write_gate, write_run
):
    stats = dict(
        PASSING_STATS,
        lid_median=SET_BANDS["lid_median"]["min"],
        ivf_gini=SET_BANDS["ivf_gini"]["max"],
    )

    report = check_gate.run(
        make_args(write_gate(tmp_path), write_run(tmp_path, stats=stats))
    )

    assert report["verdict"] == "pass", report["reasons"]


def test_check_gate_accepts_a_one_sided_band_as_set(
    tmp_path: Path, write_gate, write_run
):
    bands = dict(SET_BANDS, ivf_gini={"min": None, "max": 0.5})

    report = check_gate.run(
        make_args(write_gate(tmp_path, bands=bands), write_run(tmp_path))
    )

    assert statuses(report)["ivf_gini"] == "pass"
    assert report["verdict"] == "pass"


def test_check_gate_reports_unset_and_exits_non_zero_when_every_band_is_null(
    tmp_path: Path, write_gate, write_run
):
    report = check_gate.run(
        make_args(write_gate(tmp_path, bands=NULL_BANDS), write_run(tmp_path))
    )

    assert report["verdict"] == "unset"
    assert report["exit_code"] == check_gate.EXIT_UNSET
    assert set(statuses(report).values()) == {"unset"}
    assert report["exit_code"] != check_gate.EXIT_PASS, (
        "an uncalibrated gate must never exit 0: that is indistinguishable "
        "from a run that actually passed"
    )


def test_check_gate_treats_a_wholly_null_band_entry_the_same_as_null_bounds(
    tmp_path: Path, write_gate, write_run
):
    bands = {name: None for name in check_gate.GATE_STATISTICS}

    report = check_gate.run(
        make_args(write_gate(tmp_path, bands=bands), write_run(tmp_path))
    )

    assert set(statuses(report).values()) == {"unset"}
    assert report["verdict"] == "unset"


def test_check_gate_exits_zero_on_an_unset_gate_only_when_allow_unset_is_passed(
    tmp_path: Path, write_gate, write_run
):
    report = check_gate.run(
        make_args(
            write_gate(tmp_path, bands=NULL_BANDS),
            write_run(tmp_path),
            allow_unset=True,
        )
    )

    assert report["verdict"] == "unset"
    assert report["exit_code"] == check_gate.EXIT_PASS


def test_check_gate_fails_a_none_statistic_even_when_its_band_is_unset(
    tmp_path: Path, write_gate, write_run
):
    # summary() returns None for these two when every query was discarded.
    stats = dict(PASSING_STATS, lid_median=None, relative_contrast_median=None)

    report = check_gate.run(
        make_args(
            write_gate(tmp_path, bands=NULL_BANDS),
            write_run(tmp_path, stats=stats),
            allow_unset=True,
        )
    )

    assert statuses(report)["lid_median"] == "fail"
    assert statuses(report)["relative_contrast_median"] == "fail"
    assert report["verdict"] == "fail"
    assert report["exit_code"] == check_gate.EXIT_FAIL, (
        "--allow-unset waives the missing bands, not a degenerate set; a null "
        "statistic means every query was discarded"
    )


def test_check_gate_fails_when_a_statistic_is_absent_from_summary_json(
    tmp_path: Path, write_gate, write_run
):
    stats = {k: v for k, v in PASSING_STATS.items() if k != "hubness_skew"}

    report = check_gate.run(
        make_args(write_gate(tmp_path), write_run(tmp_path, stats=stats))
    )

    assert statuses(report)["hubness_skew"] == "fail"
    assert "absent" in report["statistics"][2]["reason"]
    assert report["exit_code"] == check_gate.EXIT_FAIL


def test_check_gate_fails_a_run_measured_under_non_canonical_conditions(
    tmp_path: Path, write_gate, write_run
):
    run_dir = write_run(tmp_path, conditions={"ann_measured_rows": 5000})

    report = check_gate.run(make_args(write_gate(tmp_path), run_dir))

    assert report["conditions"]["match"] is False
    assert report["conditions"]["mismatched"] == ["n"]
    assert report["verdict"] == "fail"
    assert report["exit_code"] == check_gate.EXIT_FAIL
    assert set(statuses(report).values()) == {"pass"}, (
        "the statistics themselves are inside their bands -- the run fails "
        "because they were not measured under comparable conditions"
    )


def test_check_gate_flags_clamped_k_and_nlist_separately(
    tmp_path: Path, write_gate, write_run
):
    run_dir = write_run(
        tmp_path, conditions={"ann_measured_k": 50, "ann_measured_nlist": 64}
    )

    report = check_gate.run(make_args(write_gate(tmp_path), run_dir))

    assert report["conditions"]["mismatched"] == ["k", "nlist"]


def test_check_gate_reports_a_condition_mismatch_without_failing_when_allowed(
    tmp_path: Path, write_gate, write_run
):
    run_dir = write_run(tmp_path, conditions={"ann_measured_rows": 5000})

    report = check_gate.run(
        make_args(write_gate(tmp_path), run_dir, allow_condition_mismatch=True)
    )

    assert report["conditions"]["match"] is False
    assert report["verdict"] == "pass"
    assert report["exit_code"] == check_gate.EXIT_PASS
    assert any("non-canonical" in r for r in report["reasons"])


def test_check_gate_fails_when_the_measurement_conditions_are_not_recorded(
    tmp_path: Path, write_gate, write_run
):
    run_dir = write_run(tmp_path)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    del summary["stats"][0]["ann_measured_nlist"]
    summary_path.write_text(json.dumps(summary))

    report = check_gate.run(make_args(write_gate(tmp_path), run_dir))

    assert report["conditions"]["mismatched"] == ["nlist"]
    assert report["exit_code"] == check_gate.EXIT_FAIL


def test_check_gate_fails_a_statistic_that_is_not_a_number(
    tmp_path: Path, write_gate, write_run
):
    stats = dict(PASSING_STATS, hubness_skew="nan-ish")

    report = check_gate.run(
        make_args(write_gate(tmp_path), write_run(tmp_path, stats=stats))
    )

    assert statuses(report)["hubness_skew"] == "fail"
    assert report["exit_code"] == check_gate.EXIT_FAIL


def test_check_gate_raises_on_an_inverted_band(tmp_path: Path, write_gate, write_run):
    bands = dict(SET_BANDS, ivf_gini={"min": 0.5, "max": 0.2})

    with pytest.raises(check_gate.GateError, match="inverted"):
        check_gate.run(
            make_args(write_gate(tmp_path, bands=bands), write_run(tmp_path))
        )


def test_check_gate_raises_on_a_non_numeric_band_bound(
    tmp_path: Path, write_gate, write_run
):
    bands = dict(SET_BANDS, lid_median={"min": "eight", "max": None})

    with pytest.raises(check_gate.GateError, match="is not a number"):
        check_gate.run(
            make_args(write_gate(tmp_path, bands=bands), write_run(tmp_path))
        )


def test_check_gate_raises_on_a_gate_file_that_is_not_valid_yaml(
    tmp_path: Path, write_run
):
    path = tmp_path / "broken.yaml"
    path.write_text("statistics: [unclosed\n")

    with pytest.raises(check_gate.GateError, match="could not parse"):
        check_gate.run(make_args(path, write_run(tmp_path)))


def test_check_gate_raises_on_a_summary_json_that_is_not_valid_json(
    tmp_path: Path, write_gate, write_run
):
    run_dir = write_run(tmp_path)
    (run_dir / "summary.json").write_text("{not json")

    with pytest.raises(check_gate.GateError, match="could not parse"):
        check_gate.run(make_args(write_gate(tmp_path), run_dir))


def test_check_gate_checks_the_named_stats_entry_rather_than_the_first_one(
    tmp_path: Path, write_gate, write_run
):
    run_dir = write_run(tmp_path)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    variant = dict(summary["stats"][0], name="v2", ivf_gini=0.95)
    summary["stats"].append(variant)
    summary_path.write_text(json.dumps(summary))

    report = check_gate.run(make_args(write_gate(tmp_path), run_dir, stats_name="v2"))

    assert report["stats_name"] == "v2"
    assert statuses(report)["ivf_gini"] == "fail"


def test_check_gate_raises_when_the_named_stats_entry_does_not_exist(
    tmp_path: Path, write_gate, write_run
):
    with pytest.raises(check_gate.GateError, match="no stats entry named"):
        check_gate.run(
            make_args(write_gate(tmp_path), write_run(tmp_path), stats_name="v9")
        )


def test_check_gate_raises_when_the_run_dir_has_no_summary_json(
    tmp_path: Path, write_gate
):
    empty = tmp_path / "runs" / "empty"
    empty.mkdir(parents=True)

    with pytest.raises(check_gate.GateError, match="no summary.json"):
        check_gate.run(make_args(write_gate(tmp_path), empty))


def test_check_gate_raises_on_a_gate_file_missing_a_statistic(
    tmp_path: Path, write_gate, write_run
):
    bands = {k: v for k, v in SET_BANDS.items() if k != "hubness_skew"}

    with pytest.raises(check_gate.GateError, match="missing bands for: hubness_skew"):
        check_gate.run(
            make_args(write_gate(tmp_path, bands=bands), write_run(tmp_path))
        )


def test_check_gate_raises_on_a_gate_file_naming_a_statistic_that_does_not_exist(
    tmp_path: Path, write_gate, write_run
):
    bands = dict(SET_BANDS, lid_mean={"min": 1.0, "max": 2.0})

    with pytest.raises(check_gate.GateError, match="unknown statistics: lid_mean"):
        check_gate.run(
            make_args(write_gate(tmp_path, bands=bands), write_run(tmp_path))
        )


def test_check_gate_raises_on_a_gate_file_missing_canonical_conditions(
    tmp_path: Path, write_gate, write_run
):
    canonical = {"n": 20000}

    with pytest.raises(check_gate.GateError, match="missing canonical conditions"):
        check_gate.run(
            make_args(write_gate(tmp_path, canonical=canonical), write_run(tmp_path))
        )


def test_check_gate_writes_the_report_to_the_output_path_when_asked(
    tmp_path: Path, write_gate, write_run
):
    out = tmp_path / "verdicts" / "gate.json"

    report = check_gate.run(
        make_args(write_gate(tmp_path), write_run(tmp_path), output=str(out))
    )

    assert json.loads(out.read_text()) == report


def test_check_gate_resolves_dataset_to_the_repo_gate_file(tmp_path: Path, write_run):
    args = make_args(Path("unused"), write_run(tmp_path))
    args.gate_file = None
    args.dataset = "sift"
    args.allow_unset = True

    report = check_gate.run(args)

    assert report["dataset"] == "sift"
    assert report["gate_file"].endswith("gates/sift.yaml")


@pytest.mark.parametrize(
    "dataset", ["sift", "deep", "gist", "glove", "nytimes", "openai"]
)
def test_every_shipped_gate_file_parses_and_declares_all_four_bands_unset(dataset: str):
    gate = check_gate.load_gate(check_gate.GATES_DIR / f"{dataset}.yaml")

    assert gate["dataset"] == dataset
    assert gate["canonical"] == CANONICAL
    for name in check_gate.GATE_STATISTICS:
        assert check_gate.band_bounds(gate["statistics"][name]) == (None, None), (
            f"{dataset}.{name} has a band; these are unset until a trained "
            "ladder shows what is achievable"
        )


def test_check_gate_main_exits_non_zero_on_an_unset_shipped_gate(
    tmp_path: Path, write_run, monkeypatch
):
    run_dir = write_run(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["check_gate", "--dataset", "sift", "--run-dir", str(run_dir)],
    )

    with pytest.raises(SystemExit) as excinfo:
        check_gate.main()

    assert excinfo.value.code == check_gate.EXIT_UNSET


def test_check_gate_main_exits_one_when_the_check_cannot_run(
    tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "sys.argv",
        ["check_gate", "--dataset", "sift", "--run-dir", str(tmp_path / "nope")],
    )

    with pytest.raises(SystemExit) as excinfo:
        check_gate.main()

    assert excinfo.value.code == check_gate.EXIT_FAIL
    captured = capsys.readouterr()
    assert captured.out == "", "a failed check must not print half a JSON report"
    assert "no summary.json" in captured.err
