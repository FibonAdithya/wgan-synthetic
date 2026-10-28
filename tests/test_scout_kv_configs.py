from pathlib import Path

import pytest
import yaml

from src.data.kv_capture import GLOBAL_LAYERS
from src.train.train_book_latent import validate_config
from src.train.train_wgan_gp import flag_settings

CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "scout_kv"
VERSIONS = ["v0", "v1", "v2"]


def _load(version, L):
    return yaml.safe_load((CONFIGS / version / f"L{L:02d}.yaml").read_text())


def _get(config, path):
    for key in path:
        config = config[key]
    return config


@pytest.mark.parametrize("version", VERSIONS)
def test_one_config_per_global_layer_and_nothing_else(version):
    """Catches a missing, extra or misnamed per-layer config file."""
    assert sorted(p.name for p in (CONFIGS / version).glob("*.yaml")) == [
        f"L{L:02d}.yaml" for L in GLOBAL_LAYERS
    ]


@pytest.mark.parametrize("version", VERSIONS)
def test_configs_differ_only_in_layer_and_output_dir(version):
    """Catches one layer's config drifting (a silent per-layer hyperparameter change)."""
    base = _load(version, 43)
    for L in GLOBAL_LAYERS:
        c = _load(version, L)
        assert (
            c["data"]["kv_layer"] == L
            and c["output_dir"] == f"runs/scout_kv/{version}/L{L:02d}"
        )
        for d in (c, base):
            d["data"].pop("kv_layer")
            d.pop("output_dir")
        assert c == base
        base = _load(version, 43)


@pytest.mark.parametrize("version", VERSIONS)
def test_train_books_exclude_reference_books(version):
    """Catches training on books 12-19, which the gate's reference uses."""
    assert set(_load(version, 43)["data"]["kv_books"]) == set(range(12))


@pytest.mark.parametrize("version", VERSIONS)
def test_norms_are_preserved(version):
    """Catches any normalisation sneaking back in."""
    c = _load(version, 43)
    assert c["data"]["preprocess"]["l2_normalize"] is False
    assert c["model"]["output_normalize"] is False


def test_v1_differs_from_v0_only_by_the_flag_keys():
    """Catches a second, unannounced change riding along with the flag (spec: flag is the only difference)."""
    for L in GLOBAL_LAYERS:
        v0, v1 = _load("v0", L), _load("v1", L)
        assert v1["data"].pop("kv_n_tail") == 1024
        assert v1["data"].pop("kv_tail_holdout") == 1024
        assert v1["model"].pop("n_flags") == 2
        assert v1["training"].pop("tail_batch_share") == 0.25
        assert v1.pop("output_dir") == f"runs/scout_kv/v1/L{L:02d}"
        v0.pop("output_dir")
        assert v1 == v0


@pytest.mark.parametrize("L", GLOBAL_LAYERS)
def test_v1_config_passes_the_trainer_flag_validation(L):
    """Catches a v1 config the trainer would refuse (a missing or half-set flag key)."""
    assert flag_settings(_load("v1", L)) == (2, 1024, 1024, 0.25)


def test_v0_configs_are_unflagged():
    """Catches a flag key leaking into v0, which must stay unconditioned."""
    for L in GLOBAL_LAYERS:
        assert flag_settings(_load("v0", L)) == (None, None, None, 0.0)


def test_v2_keeps_v1s_shared_settings():
    """Catches a second, unannounced change riding along with the rung's delta: every setting
    v1 and v2 both have, other than output_dir and lambda_gp, is equal."""
    shared = [
        ("seed",),
        ("device",),
        ("data", "real_path"),
        ("data", "metric"),
        ("data", "kv_books"),
        ("data", "kv_n_sink"),
        ("data", "kv_n_tail"),
        ("data", "descriptor_dim"),
        ("data", "preprocess"),
        ("model", "latent_dim"),
        ("model", "generator_hidden_dims"),
        ("model", "critic_hidden_dims"),
        ("model", "negative_slope"),
        ("model", "output_normalize"),
        ("model", "n_flags"),
        ("training", "num_gen_steps"),
        ("training", "n_critic"),
        ("training", "lr_g"),
        ("training", "lr_d"),
        ("training", "betas"),
        ("training", "ema_decay"),
    ]
    for L in GLOBAL_LAYERS:
        v1, v2 = _load("v1", L), _load("v2", L)
        for path in shared:
            assert _get(v2, path) == _get(v1, path), (L, path)
        assert (
            v1["training"]["lambda_gp"] == 5.0 and v2["training"]["lambda_gp"] == 10.0
        )


def test_v2_batch_matches_v1s_token_count():
    """Catches the batch size drifting from v1's 512 tokens or the tail share from 0.25."""
    c = _load("v2", 43)
    assert c["training"]["n_windows"] * c["model"]["window_critic"]["window"] == 512
    assert (
        c["training"]["tail_window_share"]
        == _load("v1", 43)["training"]["tail_batch_share"]
        == 0.25
    )


@pytest.mark.parametrize("L", GLOBAL_LAYERS)
def test_v2_config_passes_the_trainer_validation(L):
    """Catches a v2 config the book-latent trainer would refuse (missing or invalid key)."""
    assert validate_config(_load("v2", L)) is None


def test_v2_latent_matches_the_spec():
    """Catches drift in the latent (channels, features, length range), the critic strides or
    the window-critic weight from the design spec."""
    c = _load("v2", 43)
    assert c["model"]["generator_type"] == "book_latent"
    assert c["model"]["book_latent"] == {
        "channels": 64,
        "n_features": 8,
        "min_length": 2.0,
        "max_length": 16384.0,
    }
    assert c["model"]["window_critic"]["strides"] == [1, 4, 16, 64, 256, 1024]
    assert c["training"]["window_weight"] == 0.1
