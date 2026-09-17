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


# Which families have had bands calibrated from a measured seed sweep. A band
# may only appear here once run-to-run noise has been measured on that family,
# because a band narrower than the noise fails honest runs at random. Listing
# them explicitly means adding one is a deliberate edit to this table rather
# than something that slips in with a config change.
CALIBRATED_BANDS = {
    # DEEP, 2026-08-10: three-seed sweep, docs/datasets/deep_seed_sweep_summary.json.
    # hubness_skew and ivf_gini stay unset because the ladder already matches
    # real within noise (0.04 and 0.27 pooled sd), not because they are noisy.
    "deep": {"lid_median", "relative_contrast_median"},
    # GloVe, 2026-09-17: a tolerance around real's mean, not DEEP's regression
    # guard. Real side is the eight-draw floor, docs/datasets/glove_noise_floor.json;
    # the five-seed v1 and v0 sweeps it was checked against are
    # docs/datasets/glove_v1_noise_floor.json and glove_v0_*noise_floor.json.
    "glove": {
        "lid_median",
        "relative_contrast_median",
        "hubness_skew",
        "ivf_gini",
    },
    # NYTimes, 2026-09-17: a tolerance around the cleaned corpus's mean, set so
    # v3's gate-selected checkpoint passes. Real side is the ten-draw floor,
    # docs/datasets/nytimes_noise_floor.json (zero_and_duplicate_rows_removed).
    # There is no seed sweep behind it: v3 is one seed,
    # docs/results/nytimes-v3/, re-measured in nytimes-v3-seed42-100k/.
    "nytimes": {
        "lid_median",
        "relative_contrast_median",
        "hubness_skew",
        "ivf_gini",
    },
}


@pytest.mark.parametrize(
    "dataset", ["sift", "deep", "gist", "glove", "nytimes", "openai"]
)
def test_every_shipped_gate_file_parses_with_all_four_statistics(dataset: str):
    gate = check_gate.load_gate(check_gate.GATES_DIR / f"{dataset}.yaml")

    assert gate["dataset"] == dataset
    assert gate["canonical"] == CANONICAL
    assert set(gate["statistics"]) == set(check_gate.GATE_STATISTICS)


@pytest.mark.parametrize(
    "dataset", ["sift", "deep", "gist", "glove", "nytimes", "openai"]
)
def test_bands_are_set_only_where_a_seed_sweep_has_calibrated_them(dataset: str):
    """A band that appears without an entry above is a band fitted to noise."""
    gate = check_gate.load_gate(check_gate.GATES_DIR / f"{dataset}.yaml")
    expected = CALIBRATED_BANDS.get(dataset, set())

    for name in check_gate.GATE_STATISTICS:
        bounds = check_gate.band_bounds(gate["statistics"][name])
        if name in expected:
            assert bounds != (None, None), (
                f"{dataset}.{name} is listed as calibrated but ships no band"
            )
        else:
            assert bounds == (None, None), (
                f"{dataset}.{name} has a band but no measured noise floor "
                "justifies one; add it to CALIBRATED_BANDS with the sweep it "
                "came from, or remove the band"
            )


def test_the_calibrated_deep_bands_admit_every_cell_of_the_sweep_they_came_from():
    """The bands must not reject the runs they were calibrated from.

    Guards the arithmetic behind `best rung mean +/- 3 pooled sd`: a band that
    excludes one of its own nine cells was mis-derived.
    """
    summary = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "datasets"
            / "deep_seed_sweep_summary.json"
        ).read_text(encoding="utf-8")
    )
    by_name = {entry["name"]: entry for entry in summary["stats"]}
    gate = check_gate.load_gate(check_gate.GATES_DIR / "deep.yaml")
    cells = [f"{rung}_s{seed}" for rung in ("v0", "v1", "v2") for seed in (42, 43, 44)]

    for name in CALIBRATED_BANDS["deep"]:
        low, high = check_gate.band_bounds(gate["statistics"][name])
        for cell in cells:
            value = by_name[cell][name]
            assert low <= value <= high, (
                f"deep.{name} band [{low}, {high}] excludes {cell} ({value}), "
                "which it was calibrated from"
            )


def _glove_per_seed(filename: str) -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "docs" / "datasets" / filename
    return json.loads(path.read_text(encoding="utf-8"))["per_seed"]


def test_the_glove_bands_admit_every_real_draw_and_every_v1_seed():
    """The bands say how close to real counts as close enough, and v1 was
    judged close enough. Real draws must pass too: these bands are centred on
    real, unlike DEEP's, so a real draw outside them means a mis-derived band.
    """
    gate = check_gate.load_gate(check_gate.GATES_DIR / "glove.yaml")
    cells = {
        **{
            f"real_draw{i}": row
            for i, row in enumerate(_glove_per_seed("glove_noise_floor.json"))
        },
        **{
            f"v1_seed{42 + i}": row
            for i, row in enumerate(_glove_per_seed("glove_v1_noise_floor.json"))
        },
    }

    for name in check_gate.GATE_STATISTICS:
        low, high = check_gate.band_bounds(gate["statistics"][name])
        for cell, row in cells.items():
            assert low <= row[name] <= high, (
                f"glove.{name} band [{low}, {high}] excludes {cell} ({row[name]})"
            )


@pytest.mark.parametrize(
    "filename", ["glove_v0_noise_floor.json", "glove_v0_new_box_noise_floor.json"]
)
def test_every_glove_band_rejects_every_v0_seed(filename: str):
    """Each statistic on its own must reject v0, on both boxes it was measured
    on. Checked per statistic so that loosening any one band until v0 fits it
    fails here, even though v0 would still fail the gate on the other three.
    """
    gate = check_gate.load_gate(check_gate.GATES_DIR / "glove.yaml")

    for name in check_gate.GATE_STATISTICS:
        low, high = check_gate.band_bounds(gate["statistics"][name])
        for i, row in enumerate(_glove_per_seed(filename)):
            assert not low <= row[name] <= high, (
                f"glove.{name} band [{low}, {high}] admits v0 seed {42 + i} "
                f"from {filename} ({row[name]})"
            )


NYTIMES_RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results"


def _nytimes_summary_entries() -> dict[str, dict]:
    """Every stats entry in every committed NYTimes canonical summary, keyed
    `<results dir>:<entry name>`."""
    entries = {}
    for path in sorted(NYTIMES_RESULTS.glob("nytimes-*/eda_clean*summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        for entry in summary["stats"]:
            entries[f"{path.parent.name}:{entry['name']}"] = entry
    return entries


def test_the_nytimes_bands_admit_every_real_draw_and_v3_best():
    """The owner judged v3's selected checkpoint close enough, and the bands
    record that. Real must pass too: these bands are centred on the cleaned
    real corpus, so a real draw outside them means a mis-derived band."""
    gate = check_gate.load_gate(check_gate.GATES_DIR / "nytimes.yaml")
    floor = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "datasets"
            / "nytimes_noise_floor.json"
        ).read_text(encoding="utf-8")
    )["zero_and_duplicate_rows_removed"]["per_seed"]
    entries = _nytimes_summary_entries()
    cells = {
        **{f"real_draw{i}": row for i, row in enumerate(floor)},
        **{k: v for k, v in entries.items() if k.endswith(":real")},
        "nytimes-v3:v3_best": entries["nytimes-v3:v3_best"],
        "nytimes-v3-seed42-100k:v3_best": entries["nytimes-v3-seed42-100k:v3_best"],
    }
    assert len(cells) > len(floor) + 2, "no committed real rows were found"

    for name in check_gate.GATE_STATISTICS:
        low, high = check_gate.band_bounds(gate["statistics"][name])
        for cell, row in cells.items():
            assert low <= row[name] <= high, (
                f"nytimes.{name} band [{low}, {high}] excludes {cell} ({row[name]})"
            )


def test_every_nytimes_band_rejects_v0():
    """Each statistic on its own must reject v0, so that loosening any one band
    until v0 fits it fails here even though v0 would still fail the other
    three."""
    gate = check_gate.load_gate(check_gate.GATES_DIR / "nytimes.yaml")
    v0 = _nytimes_summary_entries()["nytimes-v0-seed42:v0"]

    for name in check_gate.GATE_STATISTICS:
        low, high = check_gate.band_bounds(gate["statistics"][name])
        assert not low <= v0[name] <= high, (
            f"nytimes.{name} band [{low}, {high}] admits v0 ({v0[name]})"
        )


def test_v3_best_is_the_only_committed_nytimes_checkpoint_the_gate_passes():
    """Through the real verdict, conditions included. A band loose enough to
    pass a second checkpoint -- v2c's endpoint, a collapsed step -- was not
    what the owner accepted."""
    gate_file = check_gate.GATES_DIR / "nytimes.yaml"
    gate = check_gate.load_gate(gate_file)
    passed = set()
    for key, entry in _nytimes_summary_entries().items():
        if key.endswith(":real"):
            continue
        report = check_gate.evaluate(
            gate,
            entry,
            run_dir=NYTIMES_RESULTS,
            gate_file=gate_file,
            stats_name=entry["name"],
            allow_unset=False,
            allow_condition_mismatch=False,
        )
        if report["verdict"] == "pass":
            passed.add(entry["name"])

    assert passed == {"v3_best"}


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
