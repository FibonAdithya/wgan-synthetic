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
    stats.extend({"name": name, **values, **conditions} for name, values in series.items())
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
    summary["stats"] = [e for e in summary["stats"] if e["name"] != noise_floor.REAL_NAME]
    with pytest.raises(noise_floor.NoiseFloorError, match="real"):
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
            sys.executable, "-m", "src.eval.noise_floor",
            "--summary", str(summary_path),
            "--series", "s42",
            "--series", "s43",
            "--output", str(out_path),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["series"] == ["s42", "s43"]
    assert written["spread"]["lid_median"]["mean"] == pytest.approx(13.0)
    # stdout stays parseable as JSON on its own.
    assert json.loads(result.stdout)["series"] == ["s42", "s43"]


def test_cli_exits_nonzero_on_a_missing_series(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary(BASE, {"s42": BASE, "s43": BASE})), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable, "-m", "src.eval.noise_floor",
            "--summary", str(summary_path),
            "--series", "s42",
            "--series", "s99",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )

    assert result.returncode == 1
    assert "s99" in result.stderr
    assert result.stdout == ""
