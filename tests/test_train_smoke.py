import math

import pytest

from src.train.train_wgan_gp import train


def make_config(tmp_path, generator_type):
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / (generator_type or "default")),
        "data": {
            "real_path": None,
            "format": "npy",
            "descriptor_dim": 16,
            "holdout_fraction": 0.2,
            "synthetic_if_missing": True,
            "synthetic_num_vectors": 256,
            "preprocess": {"center": False, "whiten": False, "l2_normalize": True},
        },
        "model": {
            "latent_dim": 8,
            "generator_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 16],
            "negative_slope": 0.2,
        },
        "training": {
            "batch_size": 32,
            "num_gen_steps": 4,
            "n_critic": 2,
            "lr_g": 1.0e-4,
            "lr_d": 1.0e-4,
            "betas": [0.0, 0.9],
            "lambda_gp": 5.0,
            "num_workers": 0,
            "distance_reg_alpha": 0.1,
            "distance_reg_max_points": 16,
            "amp": False,
            "log_every": 1,
            "eval_every": 2,
            "save_every": 4,
        },
    }
    if generator_type is not None:
        cfg["model"]["generator_type"] = generator_type
    return cfg


@pytest.mark.parametrize("generator_type", ["mlp", "sparse"])
def test_training_loop_runs(tmp_path, generator_type):
    ckpt_path, meta = train(make_config(tmp_path, generator_type))
    assert ckpt_path.exists()
    assert meta["metrics"]
    for entry in meta["metrics"]:
        assert math.isfinite(entry["g_loss"])
        assert math.isfinite(entry["d_loss"])


def test_sparse_eval_reports_zero_negatives(tmp_path):
    _, meta = train(make_config(tmp_path, "sparse"))
    assert meta["eval"]
    assert all(entry["negative_fraction"] == 0.0 for entry in meta["eval"])


def test_mlp_config_without_generator_type_still_trains(tmp_path):
    ckpt_path, _ = train(make_config(tmp_path, None))
    assert ckpt_path.exists()
