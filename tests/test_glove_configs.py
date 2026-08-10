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
