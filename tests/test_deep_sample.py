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


def _write_run(tmp_path: Path, *, whiten: bool, center: bool = False) -> Path:
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

    generator = build_generator(config["model"], output_dim=dim)
    torch.save({"generator_state_dict": generator.state_dict()},
               run_dir / "best_generator.pt")

    rng = np.random.default_rng(0)
    scale = np.linspace(1.0, 0.05, dim).astype(np.float32)
    x = (rng.normal(size=(400, dim)) * scale).astype(np.float32)
    # center defaults to False, matching every shipped deep_gan_* config:
    # sample_variant refuses center=True combined with l2_normalize=True
    # (finding 1), so a helper representative of real runs must default to
    # no centering. Pass center=True to build the guard-rejection case.
    cfg = PreprocessConfig(center=center, whiten=whiten, l2_normalize=True)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=dim, cfg=cfg)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"preprocess_state": state.to_serializable()}), encoding="utf-8"
    )
    return run_dir


def _write_isotropic_run(
    tmp_path: Path, *, whiten: bool, suffix: str, center: bool = False
) -> Path:
    """Build a run whose raw (pre-inversion) generator output is EXACTLY
    flat per-dimension, by construction, for the one test that needs that
    guarantee rather than a statistical likelihood of it.

    Routed through an ordinarily-initialized generator (as `_write_run`
    does), an untrained network's raw per-dimension output variance is
    itself uneven purely from random weight init -- and with only
    latent_dim=16 feeding hidden_dims=[32] into 96 outputs, that unevenness
    can rival or exceed the anisotropy `invert_preprocess` is supposed to
    restore. Empirically, under the *correct* inverse, comparing against
    such a generator passed for only 54/200 weight-init seeds: the
    assertion was a coin flip on RNG entropy, not a property of the code
    under test.

    The fix is to remove that confound at the source: build a generator
    with `generator_hidden_dims=[]` (a single `Linear(latent_dim, dim)`,
    with no LeakyReLU sitting between it and the latent noise) and hand-set
    its weight rows to unit L2 norm. For z ~ N(0, I_latent),
    Var(output_i) = ||W_i||^2 * Var(z) = 1 for every output dimension i --
    exactly, analytically, independent of any RNG seed. That gives an
    unambiguous "flat" starting point, so a large spread appearing after
    inversion can only be explained by `invert_preprocess` correctly
    restoring the covariance structure `_fit_preprocess_state` fit -- not by
    luck in an untrained network's random weights.
    """
    run_dir = tmp_path / f"run_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    dim, latent = 96, 16

    config = {
        "device": "cpu",
        "model": {
            "latent_dim": latent,
            "generator_hidden_dims": [],
            "critic_hidden_dims": [32],
            "negative_slope": 0.2,
            "generator_type": "mlp",
        },
        "data": {"descriptor_dim": dim},
    }
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    from src.models.generator import build_generator

    generator = build_generator(config["model"], output_dim=dim)
    projection_rng = np.random.default_rng(1)
    unit_rows = projection_rng.normal(size=(dim, latent)).astype(np.float32)
    unit_rows /= np.linalg.norm(unit_rows, axis=1, keepdims=True)
    with torch.no_grad():
        generator.net[0].weight.copy_(torch.from_numpy(unit_rows))
        generator.net[0].bias.zero_()
    torch.save({"generator_state_dict": generator.state_dict()},
               run_dir / "best_generator.pt")

    rng = np.random.default_rng(0)
    scale = np.linspace(1.0, 0.05, dim).astype(np.float32)
    x = (rng.normal(size=(400, dim)) * scale).astype(np.float32)
    cfg = PreprocessConfig(center=center, whiten=whiten, l2_normalize=True)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=dim, cfg=cfg)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"preprocess_state": state.to_serializable()}), encoding="utf-8"
    )
    return run_dir


def test_load_preprocess_state_round_trips_from_run_metadata(tmp_path: Path):
    state = load_preprocess_state(_write_run(tmp_path, whiten=True))
    assert state.descriptor_dim == 96
    assert state.mean is None
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

    Uses `_write_isotropic_run`, not `_write_run`: the raw (pre-inversion)
    generator output there is flat per-dimension by construction (see that
    helper's docstring), not merely plausibly flat. That alone would let a
    "spread out variance in the wrong direction" bug slip through though --
    e.g. applying the forward whitening matrix instead of its inverse also
    inflates the per-dimension spread, just anti-correlated with the real
    pattern (verified in a scratch copy: corr -0.40, vs. +0.98 for the
    correct inverse; a no-op scores ~0.03, near the unwhitened baseline).
    So the assertion checks *direction*, not just magnitude: the inverted
    output's per-dimension variance must be strongly, positively correlated
    with the descriptor_dim-indexed `scale` pattern `_write_isotropic_run`
    fits `_fit_preprocess_state` on (`np.linspace(1.0, 0.05, dim) ** 2`).
    A no-op, a wrong-direction inverse, or any other subtly broken one
    cannot produce that specific positive correlation by chance.
    """
    inverted = sample_variant(
        _write_isotropic_run(tmp_path, whiten=True, suffix="w"), num_samples=2000
    )
    unwhitened = sample_variant(
        _write_isotropic_run(tmp_path, whiten=False, suffix="u"), num_samples=2000
    )

    dim = 96
    expected_pattern = np.linspace(1.0, 0.05, dim).astype(np.float32) ** 2
    inverted_correlation = np.corrcoef(inverted.var(axis=0), expected_pattern)[0, 1]
    unwhitened_correlation = np.corrcoef(unwhitened.var(axis=0), expected_pattern)[0, 1]

    assert inverted_correlation > 0.9
    assert abs(unwhitened_correlation) < 0.3


def test_sample_variant_errors_clearly_when_run_metadata_is_missing(tmp_path: Path):
    run_dir = _write_run(tmp_path, whiten=False)
    (run_dir / "run_metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="run_metadata.json"):
        sample_variant(run_dir, num_samples=8)


def test_sample_variant_refuses_centered_and_l2_normalized_state(tmp_path: Path):
    """Pins the guard against finding 1: centering + l2_normalize together
    breaks the exactness argument invert_preprocess relies on, silently.
    Every shipped deep_gan_* config uses center=False, so this state is
    fitted explicitly with center=True to exercise the unsupported
    combination; sample_variant must raise before it ever touches the
    checkpoint.
    """
    run_dir = _write_run(tmp_path, whiten=False, center=True)
    with pytest.raises(ValueError, match="center.*l2_normalize|l2_normalize.*center"):
        sample_variant(run_dir, num_samples=8)
