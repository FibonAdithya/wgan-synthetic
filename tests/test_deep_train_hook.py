import numpy as np
import pytest
import torch

from src.train.train_wgan_gp import train


def _config(tmp_path, alpha: float) -> dict:
    """A tiny but real training run: 96-D, a handful of steps, CPU."""
    return {
        "seed": 42,
        "device": "cpu",
        "output_dir": str(tmp_path / "run"),
        "data": {
            "real_path": None,
            "format": "npy",
            "descriptor_dim": 96,
            "holdout_fraction": 0.2,
            "synthetic_if_missing": True,
            "synthetic_num_vectors": 512,
            "preprocess": {"center": False, "whiten": False, "l2_normalize": True},
        },
        "model": {
            "latent_dim": 16,
            "generator_hidden_dims": [32],
            "critic_hidden_dims": [32],
            "negative_slope": 0.2,
            "generator_type": "mlp",
        },
        "training": {
            "batch_size": 32,
            "num_gen_steps": 3,
            "n_critic": 1,
            "lr_g": 1.0e-4,
            "lr_d": 1.0e-4,
            "betas": [0.0, 0.9],
            "lambda_gp": 5.0,
            "ema_decay": 0.0,
            "distance_reg_alpha": 0.0,
            "spectrum_reg_alpha": alpha,
            "num_workers": 0,
            "amp": False,
            "log_every": 1,
            "eval_every": 100,
            "save_every": 100,
        },
    }


def test_spectrum_reg_is_logged_as_zero_when_disabled(tmp_path):
    _, meta = train(_config(tmp_path, alpha=0.0))
    assert all(m["spectrum_reg"] == 0.0 for m in meta["metrics"])


def test_spectrum_reg_is_positive_when_enabled(tmp_path):
    _, meta = train(_config(tmp_path, alpha=0.1))
    assert any(m["spectrum_reg"] > 0.0 for m in meta["metrics"])


def test_missing_spectrum_reg_alpha_defaults_to_disabled(tmp_path):
    """Every existing SIFT config omits this key and must keep working."""
    config = _config(tmp_path, alpha=0.0)
    del config["training"]["spectrum_reg_alpha"]
    _, meta = train(config)
    assert all(m["spectrum_reg"] == 0.0 for m in meta["metrics"])


def test_enabling_the_regularizer_changes_the_generator(tmp_path):
    """Proves the term reaches the generator's gradients, not just the log."""
    off_path, _ = train(_config(tmp_path / "off", alpha=0.0))
    on_path, _ = train(_config(tmp_path / "on", alpha=5.0))
    off = torch.load(off_path, map_location="cpu")["generator_state_dict"]
    on = torch.load(on_path, map_location="cpu")["generator_state_dict"]
    assert any(not torch.equal(off[k], on[k]) for k in off)
