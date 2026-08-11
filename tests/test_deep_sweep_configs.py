"""The DEEP sweep is only interpretable if each cell varies what it claims to.

A cell that quietly differs from its base rung in a fourth key would produce a
number nobody could attribute, after 35 minutes of GPU time. These tests are
cheap and run before the jobs are submitted.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "deep"
SWEEP_DIR = CONFIG_DIR / "sweep"

SEED_CELLS = [
    (f"{rung}_s{seed}", rung, seed)
    for rung in ("v0", "v1", "v2")
    for seed in (42, 43, 44)
]
ALPHA_CELLS = [("v1_alpha1_s42", 1.0), ("v1_alpha5_s42", 5.0)]
ALL_CELLS = [name for name, _, _ in SEED_CELLS] + [name for name, _ in ALPHA_CELLS]


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _cell(name: str) -> dict[str, Any]:
    return _load(SWEEP_DIR / f"{name}.yaml")


def _rung(name: str) -> dict[str, Any]:
    return _load(CONFIG_DIR / f"{name}.yaml")


def test_the_sweep_has_exactly_the_expected_cells():
    """A stray file here is 35 minutes of GPU time spent on an unplanned run."""
    on_disk = {p.stem for p in SWEEP_DIR.glob("*.yaml")}
    assert on_disk == set(ALL_CELLS)


@pytest.mark.parametrize("name,rung,seed", SEED_CELLS)
def test_seed_cell_differs_from_its_rung_only_in_seed(name: str, rung: str, seed: int):
    """The rungs ship at seed 42, so the s42 cells are exact replicas of them.

    Re-running them rather than reusing the published numbers is deliberate:
    it makes all three seeds of a rung come out of one commit on one card, so
    the spread between them measures seed and nothing else.
    """
    a, b = _flatten(_rung(rung)), _flatten(_cell(name))
    differing = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    expected = {"seed"} if seed != a["seed"] else set()
    assert differing - {"output_dir"} == expected
    assert b["seed"] == seed


@pytest.mark.parametrize("name,alpha", ALPHA_CELLS)
def test_alpha_probe_differs_from_v1_only_in_alpha(name: str, alpha: float):
    """The probes hold seed at 42 so they compare against the shipped v1 run."""
    a, b = _flatten(_rung("v1")), _flatten(_cell(name))
    differing = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert differing - {"output_dir"} == {"training.spectrum_reg_alpha"}
    assert b["training.spectrum_reg_alpha"] == alpha
    assert b["seed"] == 42


@pytest.mark.parametrize("name", ALL_CELLS)
def test_every_cell_writes_to_its_own_run_directory(name: str):
    """Two cells sharing an output_dir would overwrite each other's checkpoint."""
    assert _cell(name)["output_dir"] == f"runs/deep/sweep/{name}"


def test_no_two_cells_share_an_output_directory():
    dirs = [_cell(name)["output_dir"] for name in ALL_CELLS]
    assert len(set(dirs)) == len(dirs)


@pytest.mark.parametrize("name", ALL_CELLS)
def test_every_cell_reads_the_corpus_the_fetcher_writes(name: str):
    """The runner stages this exact path into a fresh worktree per job."""
    assert _cell(name)["data"]["real_path"] == "data/deep_1m.npy"


@pytest.mark.parametrize("name", ALL_CELLS)
def test_every_cell_keeps_the_ladder_step_budget(name: str):
    """A cell at a different step count is not comparable with the ladder."""
    assert _cell(name)["training"]["num_gen_steps"] == 30000
