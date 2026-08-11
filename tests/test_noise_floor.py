"""Tests for the seed-to-seed noise floor.

The numbers here are hand-computed, not measured from anything: these tests
are about the arithmetic, not about GloVe.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval import noise_floor

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_summarize_spread_reports_hand_computed_values():
    # mean 2.0; sample std (ddof=1) 1.0; range 2.0 -> 100% of mean; cv 50%.
    result = noise_floor.summarize_spread([1.0, 2.0, 3.0])
    assert result["mean"] == pytest.approx(2.0)
    assert result["std"] == pytest.approx(1.0)
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(3.0)
    assert result["range_pct_of_mean"] == pytest.approx(100.0)
    assert result["cv_pct"] == pytest.approx(50.0)


def test_summarize_spread_uses_sample_not_population_std():
    """Pins ddof=1, the convention docs/datasets/glove_noise_floor.json used.

    Population std of these values is 0.5; sample std is 0.5773502692. A file
    written under one convention and read under the other would silently
    understate the floor.
    """
    result = noise_floor.summarize_spread([1.0, 2.0])
    assert result["std"] == pytest.approx(0.7071067811865476)


def test_summarize_spread_rejects_a_single_value():
    """One draw has no spread; reporting 0.0 would read as 'perfectly stable'."""
    with pytest.raises(noise_floor.NoiseFloorError, match="at least two"):
        noise_floor.summarize_spread([1.0])


def _summary(real, series):
    """A summary.json-shaped dict: one 'real' entry plus one entry per series."""
    conditions = {
        "ann_measured_rows": 20000,
        "ann_measured_k": 100,
        "ann_measured_nlist": 256,
    }
    stats = [{"name": noise_floor.REAL_NAME, **real, **conditions}]
    stats.extend(
        {"name": name, **values, **conditions} for name, values in series.items()
    )
    return {"stats": stats}


BASE = {
    "lid_median": 10.0,
    "relative_contrast_median": 1.5,
    "hubness_skew": 2.0,
    "ivf_gini": 0.3,
}


def test_compute_floor_reports_spread_and_distance_in_spreads():
    summary = _summary(
        BASE,
        {
            "s42": {**BASE, "lid_median": 12.0},
            "s43": {**BASE, "lid_median": 14.0},
            "s44": {**BASE, "lid_median": 13.0},
        },
    )
    floor = noise_floor.compute_floor(summary, ["s42", "s43", "s44"])

    # mean 13.0, range 14.0 - 12.0 = 2.0, real 10.0 -> gap 3.0 -> 1.5 spreads.
    assert floor["spread"]["lid_median"]["mean"] == pytest.approx(13.0)
    assert floor["real"]["lid_median"] == pytest.approx(10.0)
    assert floor["distance_from_real"]["lid_median"] == pytest.approx(3.0)
    assert floor["distance_in_spreads"]["lid_median"] == pytest.approx(1.5)
    assert floor["series"] == ["s42", "s43", "s44"]
    assert floor["conditions"] == {"n": 20000, "k": 100, "nlist": 256}


def test_distance_from_real_keeps_its_sign():
    """A generator below real and one above are different failures."""
    summary = _summary(
        BASE,
        {"s42": {**BASE, "lid_median": 8.0}, "s43": {**BASE, "lid_median": 6.0}},
    )
    floor = noise_floor.compute_floor(summary, ["s42", "s43"])
    assert floor["distance_from_real"]["lid_median"] == pytest.approx(-3.0)
    assert floor["distance_in_spreads"]["lid_median"] == pytest.approx(1.5)


def test_zero_spread_reports_none_not_infinity():
    """Identical seeds mean the separation is unmeasurable, not infinite.

    JSON has no infinity, and a reader who meets `inf` here reads "infinitely
    well separated" when the truth is the opposite.
    """
    summary = _summary(
        BASE,
        {"s42": {**BASE, "lid_median": 12.0}, "s43": {**BASE, "lid_median": 12.0}},
    )
    floor = noise_floor.compute_floor(summary, ["s42", "s43"])
    assert floor["spread"]["lid_median"]["max"] == pytest.approx(12.0)
    assert floor["distance_in_spreads"]["lid_median"] is None


def test_missing_series_is_an_error_not_a_silent_skip():
    summary = _summary(BASE, {"s42": BASE, "s43": BASE})
    with pytest.raises(noise_floor.NoiseFloorError, match="s99"):
        noise_floor.compute_floor(summary, ["s42", "s99"])


def test_missing_real_series_is_an_error():
    summary = _summary(BASE, {"s42": BASE, "s43": BASE})
    summary["stats"] = [
        e for e in summary["stats"] if e["name"] != noise_floor.REAL_NAME
    ]
    with pytest.raises(noise_floor.NoiseFloorError, match="real"):
        noise_floor.compute_floor(summary, ["s42", "s43"])


def test_compute_floor_rejects_fewer_than_two_series():
    """The `"a floor needs at least two series"` guard is otherwise unexercised.

    Deleting it would still raise -- summarize_spread's own "at least two
    values" guard fires one level down -- so nothing would notice the
    regression without a test pinning this exact message.
    """
    summary = _summary(BASE, {"s42": BASE})
    with pytest.raises(noise_floor.NoiseFloorError, match="at least two series"):
        noise_floor.compute_floor(summary, ["s42"])


def test_compute_floor_rejects_a_series_measured_at_different_conditions():
    """A floor spread across mismatched N/k/nlist is not one measurement.

    src/eval/eda/metrics.py warns that post-clamp conditions can diverge
    between series in the same eda_report run; src/eval/check_gate.py guards
    a single run against the gate's canonical conditions for exactly that
    reason. compute_floor needs the same guard across every series it pools.
    """
    summary = _summary(BASE, {"s42": BASE, "s43": BASE})
    for entry in summary["stats"]:
        if entry["name"] == "s43":
            entry["ann_measured_rows"] = 10000
    with pytest.raises(noise_floor.NoiseFloorError, match="s43"):
        noise_floor.compute_floor(summary, ["s42", "s43"])


def test_compute_floor_names_the_disagreeing_condition_key():
    summary = _summary(BASE, {"s42": BASE, "s43": BASE})
    for entry in summary["stats"]:
        if entry["name"] == "s43":
            entry["ann_measured_nlist"] = 128
    with pytest.raises(noise_floor.NoiseFloorError, match="ann_measured_nlist"):
        noise_floor.compute_floor(summary, ["s42", "s43"])


def test_a_bool_statistic_is_an_error():
    """bool is a subclass of int; float(True) silently becoming 1.0 is wrong."""
    summary = _summary(BASE, {"s42": {**BASE, "hubness_skew": True}, "s43": BASE})
    with pytest.raises(noise_floor.NoiseFloorError, match="hubness_skew"):
        noise_floor.compute_floor(summary, ["s42", "s43"])


def test_a_numeric_string_statistic_is_an_error():
    """float("1.5") silently succeeding would let a JSON-encoding bug through."""
    summary = _summary(BASE, {"s42": {**BASE, "ivf_gini": "0.3"}, "s43": BASE})
    with pytest.raises(noise_floor.NoiseFloorError, match="ivf_gini"):
        noise_floor.compute_floor(summary, ["s42", "s43"])


def test_a_non_numeric_statistic_is_an_error_not_a_bare_traceback():
    """float(["abc"]) raises TypeError; float("abc") raises ValueError.

    Neither is a NoiseFloorError, so before this both escaped compute_floor
    and main()'s except clause as a bare traceback instead of the clean
    `noise_floor: ...` stderr line every other bad-input path here produces.
    """
    summary = _summary(BASE, {"s42": {**BASE, "lid_median": [1.0, 2.0]}, "s43": BASE})
    with pytest.raises(noise_floor.NoiseFloorError, match="lid_median"):
        noise_floor.compute_floor(summary, ["s42", "s43"])


def test_a_none_statistic_is_an_error():
    """ann_difficulty writes null when every query was discarded.

    Treating that as 0.0 would put a fabricated number in a committed floor.
    """
    summary = _summary(
        BASE,
        {"s42": {**BASE, "lid_median": None}, "s43": BASE},
    )
    with pytest.raises(noise_floor.NoiseFloorError, match="lid_median"):
        noise_floor.compute_floor(summary, ["s42", "s43"])


def test_cli_writes_the_floor_to_a_file(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            _summary(
                BASE,
                {
                    "s42": {**BASE, "lid_median": 12.0},
                    "s43": {**BASE, "lid_median": 14.0},
                },
            )
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "floor.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.eval.noise_floor",
            "--summary",
            str(summary_path),
            "--series",
            "s42",
            "--series",
            "s43",
            "--output",
            str(out_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["series"] == ["s42", "s43"]
    assert written["spread"]["lid_median"]["mean"] == pytest.approx(13.0)
    # stdout stays parseable as JSON on its own.
    assert json.loads(result.stdout)["series"] == ["s42", "s43"]


def test_cli_exits_nonzero_on_a_missing_series(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(_summary(BASE, {"s42": BASE, "s43": BASE})), encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.eval.noise_floor",
            "--summary",
            str(summary_path),
            "--series",
            "s42",
            "--series",
            "s99",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "s99" in result.stderr
    assert result.stdout == ""


def test_cli_exits_nonzero_not_a_traceback_on_a_non_numeric_statistic(tmp_path):
    """A bad summary.json must produce the clean stderr line, never a traceback."""
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(_summary(BASE, {"s42": {**BASE, "lid_median": "abc"}, "s43": BASE})),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.eval.noise_floor",
            "--summary",
            str(summary_path),
            "--series",
            "s42",
            "--series",
            "s43",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "lid_median" in result.stderr
    assert result.stdout == ""
