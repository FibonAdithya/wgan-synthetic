import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.sift1m_dataset import (
    PreprocessConfig,
    _fit_preprocess_state,
)
from src.deep.sample import load_preprocess_state, sample_variant


def _write_run(tmp_path: Path, *, whiten: bool) -> Path:
    """Build a run directory shaped exactly like one train() writes."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    dim, latent = 96, 16

    config = {
        "device": "cpu",
        "model": {
            "latent_dim": latent,
            "generator_hidden_dims": [32],
            "critic_hidden_dims": [32],
            "negative_slope": 0.2,
            "generator_type": "mlp",
        },
        "data": {"descriptor_dim": dim},
    }
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    from src.models.generator import build_generator

    # Seed the generator's weight init explicitly. Left unseeded, an untrained
    # MLP's random output-layer weights impose their own per-dimension
    # variance pattern (amplified by the latent_dim=16 -> hidden=32 bottleneck
    # above 96 output dims), which can outweigh the anisotropy signal this
    # module is meant to restore and makes
    # test_whitened_run_output_is_not_left_in_whitened_space flaky across
    # process runs. Fixing the seed makes the whole test deterministic, in
    # keeping with the already-fixed numpy rng used for the synthetic data.
    torch.manual_seed(2)
    generator = build_generator(config["model"], output_dim=dim)
    torch.save({"generator_state_dict": generator.state_dict()},
               run_dir / "best_generator.pt")

    rng = np.random.default_rng(0)
    scale = np.linspace(1.0, 0.05, dim).astype(np.float32)
    x = (rng.normal(size=(400, dim)) * scale).astype(np.float32)
    cfg = PreprocessConfig(center=True, whiten=whiten, l2_normalize=True)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=dim, cfg=cfg)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"preprocess_state": state.to_serializable()}), encoding="utf-8"
    )
    return run_dir


def test_load_preprocess_state_round_trips_from_run_metadata(tmp_path: Path):
    state = load_preprocess_state(_write_run(tmp_path, whiten=True))
    assert state.descriptor_dim == 96
    assert state.mean is not None
    assert state.whitening_matrix is not None


def test_sample_variant_returns_the_requested_shape_and_dtype(tmp_path: Path):
    x = sample_variant(_write_run(tmp_path, whiten=False), num_samples=64, batch_size=32)
    assert x.shape == (64, 96)
    assert x.dtype == np.float32


def test_sample_variant_is_deterministic_under_the_same_seed(tmp_path: Path):
    run_dir = _write_run(tmp_path, whiten=False)
    a = sample_variant(run_dir, num_samples=32, seed=7)
    b = sample_variant(run_dir, num_samples=32, seed=7)
    np.testing.assert_array_equal(a, b)


def test_whitened_run_output_is_not_left_in_whitened_space(tmp_path: Path):
    """The whole reason this module exists.

    A whitened run's raw generator output has near-flat per-dimension variance.
    After inversion the anisotropy of the fitted state must be restored, so the
    spread of per-dimension variance is visibly larger.
    """
    inverted = sample_variant(_write_run(tmp_path, whiten=True), num_samples=512)
    unwhitened = sample_variant(_write_run(tmp_path, whiten=False), num_samples=512)
    assert np.ptp(inverted.var(axis=0)) > np.ptp(unwhitened.var(axis=0))


def test_sample_variant_errors_clearly_when_run_metadata_is_missing(tmp_path: Path):
    run_dir = _write_run(tmp_path, whiten=False)
    (run_dir / "run_metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="run_metadata.json"):
        sample_variant(run_dir, num_samples=8)
