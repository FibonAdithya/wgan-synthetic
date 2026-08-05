from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

CONFIG_DIR = Path("configs/deep")
LADDER = ["v0", "v1", "v2"]


def _load(name: str) -> Dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


@pytest.mark.parametrize("name", LADDER)
def test_every_rung_targets_96_dimensions(name: str):
    assert _load(name)["data"]["descriptor_dim"] == 96


@pytest.mark.parametrize("name", LADDER)
def test_every_rung_records_the_angular_search_metric(name: str):
    """DEEP is searched angularly; the ladder must say so on every rung."""
    assert _load(name)["data"]["metric"] == "angular"


@pytest.mark.parametrize("name", LADDER)
def test_every_rung_reads_what_the_fetcher_writes(name: str):
    """`python -m src.data.fetch deep` produces exactly this filename."""
    assert _load(name)["data"]["real_path"] == "data/deep_1m.npy"


@pytest.mark.parametrize("name", LADDER)
def test_every_rung_shares_the_fixed_hyperparameters(name: str):
    config = _load(name)
    assert config["seed"] == 42
    assert config["model"]["latent_dim"] == 128
    assert config["model"]["generator_hidden_dims"] == [512, 1024, 1024]
    assert config["model"]["critic_hidden_dims"] == [1024, 512, 256]
    assert config["model"]["generator_type"] == "mlp"
    assert config["training"]["batch_size"] == 512
    assert config["training"]["n_critic"] == 3
    assert config["training"]["lambda_gp"] == 5.0
    assert config["training"]["ema_decay"] == 0.999
    assert config["training"]["distance_reg_alpha"] == 0.0
    assert config["training"]["num_gen_steps"] == 30000


@pytest.mark.parametrize(
    "lower,upper,expected_delta",
    [
        ("v0", "v1", "training.spectrum_reg_alpha"),
        ("v1", "v2", "data.preprocess.whiten"),
    ],
)
def test_each_rung_is_exactly_one_change_from_the_previous(
    lower: str, upper: str, expected_delta: str
):
    """The ladder is only interpretable if one thing varies at a time."""
    a, b = _flatten(_load(lower)), _flatten(_load(upper))
    differing = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert differing - {"output_dir"} == {expected_delta}


def test_v0_disables_the_spectrum_regularizer():
    assert _load("v0")["training"]["spectrum_reg_alpha"] == 0.0


def test_v1_enables_the_spectrum_regularizer():
    assert _load("v1")["training"]["spectrum_reg_alpha"] == 0.1


def test_only_v2_whitens():
    assert _load("v0")["data"]["preprocess"]["whiten"] is False
    assert _load("v1")["data"]["preprocess"]["whiten"] is False
    assert _load("v2")["data"]["preprocess"]["whiten"] is True


@pytest.mark.parametrize("name", LADDER)
def test_no_rung_centers(name: str):
    """Centering plus l2_normalize is refused by compare_variants.invert_samples.

    The whitened rung could not be sampled at all with centering on, and the
    unwhitened rungs keep it off so the ladder stays a single-delta chain.
    """
    assert _load(name)["data"]["preprocess"]["center"] is False
