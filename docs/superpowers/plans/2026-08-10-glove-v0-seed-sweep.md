# GloVe v0 Seed Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train GloVe `v0` five times varying only the training seed, and report which of the four ANN-difficulty statistics survive that variation well enough to be gated on.

**Architecture:** Five "instrument" configs differ from the `v0` rung only in seed, `output_dir` and an absolute `real_path`. Each trains through `gpuq` on the shared box. All five checkpoints are sampled at a *fixed* seed and measured in one `eda_report` invocation, producing a single `summary.json` holding six series. A new pure-Python module, `src/eval/noise_floor.py`, turns that file into a committed floor JSON and the table that goes on the dataset page.

**Tech Stack:** Python 3.12, PyTorch (training only), PyYAML, pytest. No new dependencies.

## Global Constraints

- **Design authority:** `docs/superpowers/specs/2026-08-10-glove-v0-seed-sweep-design.md`. Where this plan and the spec disagree, the spec wins.
- **`configs/glove/v0.yaml` is never edited and never run.** It defines the rung.
- **`gates/glove.yaml` bands stay null.** Setting a band is reserved for a human with a full ladder (`AGENTS.md` invariant 1).
- **Canonical measurement conditions, locked:** `n = 20000`, `k = 100`, `k_hub = 10`, `nlist = 256`, `preprocess: l2`, measurement `seed: 42`.
- **Training seeds:** 42, 43, 44, 45, 46.
- **Sampling seed is fixed at 42 for every run**, so only the training seed varies.
- **Sample count is 50,000 vectors per run**, matching `num_vectors` in the committed real profile.
- **Spread convention, matching `docs/datasets/glove_noise_floor.json`:** sample standard deviation (`statistics.stdev`, ddof=1); `range_pct_of_mean = (max - min) / mean * 100`; `cv_pct = std / mean * 100`.
- **"Spread" in the units-of-spread column means the range** (`max - min`), which is what SIFT's n=2 floor used and what the real-side file reports.
- **Python interpreter for local test runs:** `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python` — worktrees have no `.venv` of their own.
- **Never pass a `runs/...` path to `gpuq --artifact`.** `runs/` is gitignored, the runner collects with `git add`, and the job is marked failed with `exit_code: 0` — training succeeds and the output is thrown away. Declare no artifacts; copy outputs to `/workspace/` inside the job command.
- **Each `gpuq` job runs in a fresh detached worktree with no gitignored files.** The corpus must be linked in by the job command.
- **The runner fetches from `origin`,** so the branch must be pushed before a job pinned to its commit can run.
- **`gpuq` lives at `/venv/main/bin/gpuq` on the box and is not on the ssh `PATH`.** Invoke it by absolute path. There is no local `gpuq`.
- **`ssh tig-gpu` from Bash requires `dangerouslyDisableSandbox: true`.**

## File Structure

| Path | Responsibility |
|---|---|
| `src/eval/noise_floor.py` | Create. Spread arithmetic, floor computation, CLI. Pure functions over a loaded `summary.json` dict — no GPU, no vectors, no plotly. |
| `tests/test_noise_floor.py` | Create. Unit tests for the above against hand-computed values. |
| `configs/glove/v0_seed{42,43,44,45,46}.yaml` | Create. Five measurement instruments. |
| `tests/test_glove_configs.py` | Create. Pins that the five differ from `v0.yaml` in exactly the three declared keys, and that `latent_dim` stays 128. |
| `docs/datasets/glove_v0_noise_floor.json` | Create. The committed floor, written by the CLI. |
| `docs/datasets/glove.md` | Modify. Fill the `v0` ladder row; add a `## Noise floor` section. |
| `gates/glove.yaml` | Modify, comments only. A warning beside any statistic the sweep shows to be noise-dominated. Bands stay null. |

`noise_floor.py` deliberately does not import from `src.eval.eda`, following the precedent set at the top of `src/eval/check_gate.py`: the arithmetic must run anywhere a `summary.json` can be copied, without plotly and without loading any vectors.

---

### Task 1: Spread arithmetic

**Files:**
- Create: `src/eval/noise_floor.py`
- Test: `tests/test_noise_floor.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `GATE_STATISTICS: tuple[str, ...]`, `REAL_NAME: str`, `NoiseFloorError(Exception)`, `summarize_spread(values: Sequence[float]) -> dict[str, float]` returning keys `mean`, `std`, `min`, `max`, `range_pct_of_mean`, `cv_pct`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the seed-to-seed noise floor.

The numbers here are hand-computed, not measured from anything: these tests
are about the arithmetic, not about GloVe.
"""

import pytest

from src.eval import noise_floor


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_noise_floor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.noise_floor'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_noise_floor.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add src/eval/noise_floor.py tests/test_noise_floor.py
git commit -m "feat(eval): spread arithmetic for a seed-to-seed noise floor"
```

---

### Task 2: The floor, with distance from real in units of spread

**Files:**
- Modify: `src/eval/noise_floor.py`
- Test: `tests/test_noise_floor.py`

**Interfaces:**
- Consumes: `summarize_spread`, `GATE_STATISTICS`, `REAL_NAME`, `NoiseFloorError` from Task 1.
- Produces: `compute_floor(summary: dict, series_names: Sequence[str], *, real_name: str = REAL_NAME) -> dict`. Returned keys: `series`, `real_name`, `conditions`, `real`, `per_seed`, `spread`, `distance_from_real`, `distance_in_spreads`.

`distance_from_real` is **signed** (`mean_of_seeds - real`), because SIFT's floor found two runs that disagreed on the sign of the contrast gap and that fact mattered. `distance_in_spreads` uses the absolute value over the range, and is `None` when the range is zero.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_noise_floor.py -v`
Expected: FAIL — `AttributeError: module 'src.eval.noise_floor' has no attribute 'compute_floor'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/eval/noise_floor.py`, after `summarize_spread`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_noise_floor.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add src/eval/noise_floor.py tests/test_noise_floor.py
git commit -m "feat(eval): report seed spread and the distance from real in units of it"
```

---

### Task 3: CLI

**Files:**
- Modify: `src/eval/noise_floor.py`
- Test: `tests/test_noise_floor.py`

**Interfaces:**
- Consumes: `compute_floor` from Task 2.
- Produces: `python -m src.eval.noise_floor --summary <path> --series <name> [--series <name> ...] [--real-name <name>] [--output <path>]`, printing the floor as JSON to stdout. Exit 1 with a stderr message on `NoiseFloorError`.

- [ ] **Step 1: Write the failing test**

```python
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_noise_floor.py -k cli -v`
Expected: FAIL — the module has no `__main__` entry point, non-zero exit with "No module named src.eval.noise_floor.__main__"

- [ ] **Step 3: Write minimal implementation**

Add the imports `argparse`, `json`, `sys` and `from pathlib import Path` at the top of `src/eval/noise_floor.py`, then append:

```python
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
    except (NoiseFloorError, OSError, json.JSONDecodeError) as exc:
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
```

- [ ] **Step 4: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -p no:warnings -q`
Expected: PASS, exit 0. The docs-reference tests must stay green.

- [ ] **Step 5: Commit**

```bash
git add src/eval/noise_floor.py tests/test_noise_floor.py
git commit -m "feat(eval): CLI writing a committed noise-floor JSON"
```

---

### Task 4: The five instrument configs

**Files:**
- Create: `configs/glove/v0_seed42.yaml`, `v0_seed43.yaml`, `v0_seed44.yaml`, `v0_seed45.yaml`, `v0_seed46.yaml`
- Create: `tests/test_glove_configs.py`

**Interfaces:**
- Consumes: nothing.
- Produces: five config paths the Task 5 job commands name, and run directories `runs/glove/v0_seed<N>`.

- [ ] **Step 1: Write the failing test**

```python
"""The GloVe v0 seed sweep is only a measurement if one thing varies.

These tests pin that: five configs identical to the rung and to each other
except for the seed, the output directory and an absolute corpus path.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "glove"
SEEDS = [42, 43, 44, 45, 46]
INSTRUMENTS = [f"v0_seed{seed}" for seed in SEEDS]

# The three keys an instrument is allowed to differ from the rung on. Anything
# else differing means the sweep is measuring more than the seed.
ALLOWED_DELTAS = {"seed", "output_dir", "data.real_path"}


def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


@pytest.mark.parametrize("name", INSTRUMENTS)
def test_instrument_differs_from_the_rung_only_where_allowed(name: str):
    rung, instrument = _flatten(_load("v0")), _flatten(_load(name))
    differing = {k for k in rung.keys() | instrument.keys() if rung.get(k) != instrument.get(k)}
    assert differing <= ALLOWED_DELTAS


@pytest.mark.parametrize("seed", SEEDS)
def test_instrument_carries_its_own_seed_and_output_dir(seed: int):
    config = _load(f"v0_seed{seed}")
    assert config["seed"] == seed
    assert config["output_dir"] == f"runs/glove/v0_seed{seed}"


@pytest.mark.parametrize("name", INSTRUMENTS)
def test_instrument_names_an_absolute_corpus_path(name: str):
    """gpuq runs each job in a fresh worktree where data/ does not exist."""
    assert Path(_load(name)["data"]["real_path"]).is_absolute()


def test_the_seeds_are_distinct():
    """A repeated seed would be a duplicate run masquerading as a draw."""
    seeds = [_load(name)["seed"] for name in INSTRUMENTS]
    assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize("name", ["v0", *INSTRUMENTS])
def test_latent_dim_stays_128_over_a_100_dim_corpus(name: str):
    """Deliberate, and not the sift-inherited value deep corrected away from.

    GloVe's measured effective rank is 94.6 of 100, so a latent at or below
    the corpus rank would impose a bottleneck the corpus does not have. Deep's
    correction to descriptor_dim was driven by its own rank of 65 of 96 and
    does not transfer. Without this test the 128 reads as an oversight.
    """
    config = _load(name)
    assert config["model"]["latent_dim"] == 128
    assert config["data"]["descriptor_dim"] == 100


def test_the_rung_still_points_at_the_repo_relative_corpus():
    """v0.yaml is the rung and must stay box-independent."""
    assert _load("v0")["data"]["real_path"] == "data/glove_250k.npy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_glove_configs.py -v`
Expected: FAIL — `FileNotFoundError` for `configs/glove/v0_seed42.yaml`

- [ ] **Step 3: Create the five configs**

Create `configs/glove/v0_seed42.yaml` with the body below. Then create the other four by copying it and changing **only** the seed in the header comment, `seed:`, and `output_dir:`.

```yaml
# GloVe v0 seed sweep, run 42. Identical to configs/glove/v0.yaml in every
# training hyperparameter -- that is the point of the exercise. Three things
# differ, and each is listed here so a reader can confirm nothing else moved:
#
#   1. seed        42 -> 42   (unchanged; 42 is the rung's own seed)
#   2. output_dir  runs/glove/v0 -> runs/glove/v0_seed42
#   3. real_path   data/glove_250k.npy -> an absolute path (see below)
#
# Why this file exists rather than an edit to v0.yaml: v0 is a ladder rung,
# and a rung is a historical record of a run that happened. Changing what it
# points at would redefine what GloVe v0 reproduces, which AGENTS.md reserves
# for a human decision. This config is a measurement instrument, not a rung.
#
# Why an absolute real_path: the gpuq runner executes each job in a fresh
# detached git worktree cut from the pinned commit, and data/ is gitignored,
# so a relative data/... path does not exist at run time. That makes this
# config box-specific by construction, which is acceptable for an instrument
# and would not be for a rung.
#
# Why five of these: GloVe v0 has never been trained, and a single run would
# give the ladder a baseline with no variance estimate -- the defect
# docs/datasets/deep.md documents in its own ladder, where a reseed moved a
# gap tenfold. docs/datasets/sift.md puts the number at three to five seeds.
seed: 42
device: auto
output_dir: runs/glove/v0_seed42

data:
  real_path: /workspace/data-cache/glove_250k.npy
  format: npy
  metric: angular
  descriptor_dim: 100
  holdout_fraction: 0.05
  synthetic_if_missing: false
  synthetic_num_vectors: 100000
  preprocess:
    center: false
    whiten: false
    l2_normalize: true

model:
  # 128 over a 100-dimensional corpus is deliberate. GloVe's measured
  # effective rank is 94.6 of 100, so there is no bottleneck to discover and a
  # latent at or below the corpus rank would impose one. configs/deep/v0.yaml
  # corrected the same sift-inherited 128 down to its descriptor_dim, but that
  # was driven by deep's effective rank of 65 of 96 and does not transfer.
  latent_dim: 128
  generator_hidden_dims: [512, 1024, 1024]
  critic_hidden_dims: [1024, 512, 256]
  negative_slope: 0.2
  generator_type: mlp

training:
  batch_size: 512
  num_gen_steps: 30000
  n_critic: 3
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  betas: [0.0, 0.9]
  lambda_gp: 5.0
  distance_reg_alpha: 0.0
  distance_reg_max_points: 256
  num_workers: 0
  amp: false
  log_every: 250
  eval_every: 1000
  save_every: 2000
```

For the other four, the header's line 1 and item 1 read (43 shown):

```
# GloVe v0 seed sweep, run 43. Identical to configs/glove/v0.yaml in every
...
#   1. seed        42 -> 43   (the only difference from run 42)
#   2. output_dir  runs/glove/v0 -> runs/glove/v0_seed43
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_glove_configs.py -v`
Expected: PASS, 23 tests

- [ ] **Step 5: Commit**

```bash
git add configs/glove/v0_seed4*.yaml tests/test_glove_configs.py
git commit -m "feat(glove): five v0 instrument configs varying only the seed"
```

---

### Task 5: Train the five runs

**Files:**
- No repo files change. This task produces `/workspace/glove-sweep/v0_seed<N>/` on the box.

**Interfaces:**
- Consumes: the five configs from Task 4, at a commit pushed to `origin`.
- Produces: `/workspace/glove-sweep/v0_seed<N>/best_generator.pt` and `run_config.yaml` for each of the five seeds — the inputs Task 6 samples.

**All `ssh` calls in this task need `dangerouslyDisableSandbox: true`.**

- [ ] **Step 1: Push the branch**

The runner fetches from `origin` and checks out the pinned commit, so an unpushed commit cannot be run.

```bash
git push -u origin glove-gan-v1
```

- [ ] **Step 2: Preflight the staging on the cheap CPU lane**

The two traps — a gitignored `--artifact` path, and a fresh worktree with no corpus — both cost a full run's GPU time to discover. Prove the staging works for about a minute of CPU first.

```bash
COMMIT=$(git rev-parse HEAD)
ssh tig-gpu "/venv/main/bin/gpuq submit --project wgan-synthetic \
  --commit $COMMIT --branch glove-gan-v1 --lane cpu \
  --dedupe-key glove-sweep-preflight \
  -- bash -c 'ln -sf /workspace/data-cache/glove_250k.npy data/glove_250k.npy && \
     ls -l data/glove_250k.npy && \
     /venv/main/bin/python -c \"import yaml; c=yaml.safe_load(open(\\\"configs/glove/v0_seed42.yaml\\\")); print(c[\\\"seed\\\"], c[\\\"data\\\"][\\\"real_path\\\"])\"'"
```

Then `ssh tig-gpu "/venv/main/bin/gpuq wait <id>"`. Expected: exit 0. If it fails, read `ssh tig-gpu "/venv/main/bin/gpuq show <id>"` — the stderr tail is in `error`.

- [ ] **Step 3: Submit the five training jobs**

Declare **no** `--artifact`; copy the outputs to `/workspace/` from inside the job command instead, because `runs/` is gitignored and the runner's `git add` would fail the job with `exit_code: 0`.

```bash
COMMIT=$(git rev-parse HEAD)
for SEED in 42 43 44 45 46; do
  ssh tig-gpu "/venv/main/bin/gpuq submit --project wgan-synthetic \
    --commit $COMMIT --branch glove-gan-v1 --lane gpu \
    --dedupe-key glove-v0-seed$SEED \
    -- bash -c 'mkdir -p data && \
       ln -sf /workspace/data-cache/glove_250k.npy data/glove_250k.npy && \
       /venv/main/bin/python -m src.train.train_wgan_gp --config configs/glove/v0_seed$SEED.yaml && \
       mkdir -p /workspace/glove-sweep/v0_seed$SEED && \
       cp runs/glove/v0_seed$SEED/best_generator.pt runs/glove/v0_seed$SEED/run_config.yaml /workspace/glove-sweep/v0_seed$SEED/'"
done
```

Record the five job ids. `--dedupe-key` makes a repeated submit return the existing job rather than duplicating it.

- [ ] **Step 4: Wait**

The GPU lane is serialized box-wide on one RTX 4060, so these run one at a time behind each other and behind anyone else's training — roughly three hours for the five, not 35 minutes.

```bash
ssh tig-gpu "/venv/main/bin/gpuq wait <id>"   # per job; 0 done, 1 failed
```

`wait` returns immediately if the job already finished, so there is no penalty for waiting late. Do other work between checks.

- [ ] **Step 5: Verify all five produced a checkpoint**

```bash
ssh tig-gpu "ls -l /workspace/glove-sweep/v0_seed*/best_generator.pt"
```

Expected: five files. A missing one means that job failed despite a green `wait`; check `gpuq show <id>`.

No commit — this task changes no repo files.

---

### Task 6: Sample and measure

**Files:**
- No repo files change. Produces `/workspace/glove-sweep/summary.json` on the box.

**Interfaces:**
- Consumes: the five checkpoints from Task 5.
- Produces: a `summary.json` holding six series — `real` plus `v0_seed42..46` — which Task 7 feeds to `noise_floor`.

- [ ] **Step 1: Sample each checkpoint at a fixed seed**

`--seed 42` on every one. Only the *training* seed varies; letting the sampling seed move too would fold a second source of variation into the floor and make it unattributable.

```bash
ssh tig-gpu "for SEED in 42 43 44 45 46; do \
  /venv/main/bin/python -m src.sample.generate \
    --checkpoint /workspace/glove-sweep/v0_seed\$SEED/best_generator.pt \
    --config /workspace/glove-sweep/v0_seed\$SEED/run_config.yaml \
    --num-samples 50000 --seed 42 \
    --output-path /workspace/glove-sweep/samples/v0_seed\$SEED.npy || exit 1; \
done"
```

Run this from a checkout on the box that has the code — not from a runner-owned queue worktree, which must not have git run in it. Use a `--lane cpu` `gpuq` job if no such checkout is free.

- [ ] **Step 2: Measure all five in one `eda_report` invocation**

One invocation, not five: it makes the conditions identical by construction and puts all six series in one `summary.json`.

```bash
ssh tig-gpu "/venv/main/bin/python -m src.eval.eda_report \
  --real-path /workspace/data-cache/glove_250k.npy \
  --synthetic-path v0_seed42=/workspace/glove-sweep/samples/v0_seed42.npy \
  --synthetic-path v0_seed43=/workspace/glove-sweep/samples/v0_seed43.npy \
  --synthetic-path v0_seed44=/workspace/glove-sweep/samples/v0_seed44.npy \
  --synthetic-path v0_seed45=/workspace/glove-sweep/samples/v0_seed45.npy \
  --synthetic-path v0_seed46=/workspace/glove-sweep/samples/v0_seed46.npy \
  --output-dir /workspace/glove-sweep/report \
  --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --no-png"
```

- [ ] **Step 3: Confirm the conditions came back canonical**

```bash
ssh tig-gpu "/venv/main/bin/python -c \"import json; s=json.load(open('/workspace/glove-sweep/report/summary.json')); print([(e['name'], e['ann_measured_rows'], e['ann_measured_k'], e['ann_measured_nlist']) for e in s['stats']])\""
```

Expected: six rows, each `20000, 100, 256`. A different value means the run is not comparable with the committed real profile and the measurement must be repeated, not reinterpreted.

- [ ] **Step 4: Copy the summary back**

```bash
scp tig-gpu:/workspace/glove-sweep/report/summary.json /tmp/glove_sweep_summary.json
```

No commit — this task changes no repo files.

---

### Task 7: Compute the floor and write it up

**Files:**
- Create: `docs/datasets/glove_v0_noise_floor.json`
- Modify: `docs/datasets/glove.md`
- Modify: `gates/glove.yaml` (comments only)

**Interfaces:**
- Consumes: `/tmp/glove_sweep_summary.json` from Task 6 and `python -m src.eval.noise_floor` from Task 3.

- [ ] **Step 1: Generate the committed floor**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m src.eval.noise_floor \
  --summary /tmp/glove_sweep_summary.json \
  --series v0_seed42 --series v0_seed43 --series v0_seed44 \
  --series v0_seed45 --series v0_seed46 \
  --output docs/datasets/glove_v0_noise_floor.json
```

- [ ] **Step 2: Fill the `v0` ladder row in `docs/datasets/glove.md`**

Replace the `Run` and `Status` cells of the `v0` row. Report the **mean across the five seeds and the min–max range**, not a single number — range rather than standard deviation, matching what `glove_noise_floor.json` reports on the real side so the two tables read together. Fill the "Synthetic (best variant)" column of the "Measured profile" table with the same means.

- [ ] **Step 3: Add a `## Noise floor` section**

Mirror `docs/datasets/sift.md`'s section. It must carry the units-of-spread column and state, for each of the four statistics, whether the sweep leaves it gateable. Say plainly that this is `n=5` on the synthetic side and that the real-side floor in `glove_noise_floor.json` is a separate measurement of a different thing.

Reproduce commands go in this section, naming `configs/glove/v0_seed{42..46}.yaml`, the fixed sampling seed, and the single `eda_report` invocation.

- [ ] **Step 4: Update `gates/glove.yaml` comments**

For any statistic the sweep shows to be noise-dominated, add a warning beside its band in the same voice as the existing `hubness_skew` one, citing the new `## Noise floor` section. **Every band stays null.**

- [ ] **Step 5: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -p no:warnings -q`
Expected: exit 0. `tests/test_docs_references.py` checks that every path and anchor the docs cite resolves — the new JSON filename and the new section anchor must both be real.

- [ ] **Step 6: Commit**

```bash
git add docs/datasets/glove_v0_noise_floor.json docs/datasets/glove.md gates/glove.yaml
git commit -m "docs(glove): record the v0 baseline as a five-seed sweep"
```

---

## Self-Review

**Spec coverage.** Five instrument configs → Task 4. Fixed sampling seed and single `eda_report` → Task 6. `noise_floor.py` with matching schema, the units-of-spread column, and the zero-spread `null` → Tasks 1–3. Ladder row as mean and range, `## Noise floor` section, committed JSON → Task 7. Bands stay null → Task 7 step 4 and the Global Constraints. The spec's "result this may produce" is carried by Task 7 step 3, which requires a verdict per statistic rather than only a table.

**Non-goals honored.** No task picks or trains a `v1` delta, edits `configs/glove/v0.yaml`, sets a band, touches phase (b) or (c), or backfills SIFT's floor.

**Known gap, deliberate.** Task 6 runs on the box outside `gpuq` for a CPU-cheap step. If no free checkout exists there, it becomes a `--lane cpu` job; the step says so rather than assuming one is free.
