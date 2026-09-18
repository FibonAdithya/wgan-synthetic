"""The GloVe seed sweeps are only measurements if one thing varies.

These tests pin that: each rung's five instruments are identical to the rung
and to each other except for the seed, the output directory, an absolute
corpus path and (for v1) the device; v1 is v0 plus exactly its delta; and the
lid_reg probe is v0 plus lid_reg, seed for seed.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "glove"
SEEDS = [42, 43, 44, 45, 46]
INSTRUMENTS = [f"v0_seed{seed}" for seed in SEEDS]

# v1 is v0 plus the covariance-spectrum regularizer. Its five seed instruments
# were trained under the name probe_spectrum_seed<N> (seed 42 at 5774227,
# seeds 43-46 at b17bd5f) and renamed afterwards; only the file name and
# output_dir changed.
V1_DELTA = {"training.spectrum_reg_alpha"}
V1_INSTRUMENTS = [f"v1_seed{seed}" for seed in SEEDS]

# The lid_reg probe was measured alongside v1 and not chosen as a rung.
# probe_lidreg_seed<N> is v0_seed<N> plus lid_reg.
LIDREG_KEYS = {
    "training.lid_reg_alpha",
    "training.lid_reg_k",
    "training.lid_reg_max_points",
}
LIDREG_PROBES = [f"probe_lidreg_seed{seed}" for seed in SEEDS]

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
    differing = {
        k for k in rung.keys() | instrument.keys() if rung.get(k) != instrument.get(k)
    }
    assert differing <= ALLOWED_DELTAS


@pytest.mark.parametrize("seed", SEEDS)
def test_instrument_carries_its_own_seed_and_output_dir(seed: int):
    config = _load(f"v0_seed{seed}")
    assert config["seed"] == seed
    assert config["output_dir"] == f"runs/glove/v0_seed{seed}"


@pytest.mark.parametrize("name", INSTRUMENTS + V1_INSTRUMENTS + LIDREG_PROBES)
def test_instrument_names_an_absolute_corpus_path(name: str):
    """gpuq runs each job in a fresh worktree where data/ does not exist."""
    assert Path(_load(name)["data"]["real_path"]).is_absolute()


def test_instruments_agree_on_the_real_path():
    """Absoluteness alone doesn't catch one instrument pointing at a different corpus.

    `data.real_path` is one of the three allowed deltas, so nothing above
    checks the five agree on *which* absolute path they name. One instrument
    pointed at a different corpus file would pass every test above and
    silently invalidate the whole sweep -- the exact failure this file exists
    to prevent.
    """
    paths = {_load(name)["data"]["real_path"] for name in INSTRUMENTS}
    assert len(paths) == 1


def test_the_seeds_are_distinct():
    """A repeated seed would be a duplicate run masquerading as a draw."""
    seeds = [_load(name)["seed"] for name in INSTRUMENTS]
    assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize("name", ["v0", "v1", *INSTRUMENTS, *V1_INSTRUMENTS])
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


def _differing(a: dict[str, Any], b: dict[str, Any]) -> set[str]:
    fa, fb = _flatten(a), _flatten(b)
    return {k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)}


def test_v1_differs_from_v0_by_exactly_its_delta():
    """Equality, not subset: a v1 that lost its alpha would otherwise pass as v0.

    output_dir is set aside because each rung writes to its own run directory.
    """
    assert _differing(_load("v0"), _load("v1")) - {"output_dir"} == V1_DELTA


def test_v1_uses_the_alpha_deep_found_binding():
    """DEEP's sweep moved effective_rank at 5.0 and not at 0.1 or 1.0."""
    assert _load("v1")["training"]["spectrum_reg_alpha"] == 5.0


@pytest.mark.parametrize("name", V1_INSTRUMENTS)
def test_v1_instrument_differs_from_the_rung_only_where_allowed(name: str):
    """device is allowed as well: the sweep ran with cuda:0 under the runner's pin."""
    assert _differing(_load("v1"), _load(name)) <= ALLOWED_DELTAS | {"device"}


@pytest.mark.parametrize("seed", SEEDS)
def test_v1_instrument_carries_its_own_seed_and_output_dir(seed: int):
    config = _load(f"v1_seed{seed}")
    assert config["seed"] == seed
    assert config["output_dir"] == f"runs/glove/v1_seed{seed}"


@pytest.mark.parametrize("seed", SEEDS)
def test_v1_instrument_is_its_v0_instrument_plus_the_rung_delta(seed: int):
    """Pairs the two sweeps seed by seed, so a v1-vs-v0 gap is attributable to the delta."""
    differing = _differing(_load(f"v0_seed{seed}"), _load(f"v1_seed{seed}"))
    assert differing <= V1_DELTA | {"output_dir", "device"}
    assert V1_DELTA <= differing


def test_v0_and_v1_instruments_agree_on_the_real_path():
    paths = {_load(name)["data"]["real_path"] for name in INSTRUMENTS + V1_INSTRUMENTS}
    assert len(paths) == 1


@pytest.mark.parametrize("seed", SEEDS)
def test_lidreg_probe_differs_from_its_v0_seed_only_by_lid_reg(seed: int):
    differing = _differing(_load(f"v0_seed{seed}"), _load(f"probe_lidreg_seed{seed}"))
    assert differing <= LIDREG_KEYS | {"output_dir", "device"}


@pytest.mark.parametrize("seed", SEEDS)
def test_lidreg_probe_is_switched_on_and_names_itself(seed: int):
    """A probe whose alpha is zero trains v0 again under a different name."""
    config = _load(f"probe_lidreg_seed{seed}")
    assert config["training"]["lid_reg_alpha"] > 0.0
    assert config["seed"] == seed
    assert config["output_dir"] == f"runs/glove/probe_lidreg_seed{seed}"


def test_lidreg_probe_seeds_agree_on_every_lid_reg_key():
    """The per-seed test allows these keys to differ from v0, so a seed carrying
    a different alpha would pass it unnoticed."""
    for key in LIDREG_KEYS:
        values = {_flatten(_load(name)).get(key) for name in LIDREG_PROBES}
        assert len(values) == 1, (key, values)


def test_lidreg_probe_keeps_the_values_the_scale_probe_sized():
    """Agreement across seeds is not enough: all five moving to k=10 together
    would pass it, but alpha was sized from the gap measured at k=20."""
    values = {
        key: {_flatten(_load(name)).get(key) for name in LIDREG_PROBES}
        for key in LIDREG_KEYS
    }
    assert values == {
        "training.lid_reg_alpha": {0.01858},
        "training.lid_reg_k": {20},
        "training.lid_reg_max_points": {256},
    }


@pytest.mark.parametrize("name", ["v0", "v1"])
def test_the_rungs_point_at_the_repo_relative_corpus(name: str):
    """Rungs must stay box-independent; only instruments name an absolute path."""
    assert _load(name)["data"]["real_path"] == "data/glove_250k.npy"
