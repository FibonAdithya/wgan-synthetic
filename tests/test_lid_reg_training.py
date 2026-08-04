import pytest

from src.train.train_wgan_gp import train


def _config(tmp_path, **training):
    cfg = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "run"),
        "data": {
            "real_path": None,
            "format": "npy",
            "descriptor_dim": 8,
            "holdout_fraction": 0.2,
            "synthetic_if_missing": True,
            "synthetic_num_vectors": 256,
            "preprocess": {"center": False, "whiten": False, "l2_normalize": True},
        },
        "model": {
            "latent_dim": 4,
            "generator_hidden_dims": [8],
            "critic_hidden_dims": [8],
            "negative_slope": 0.2,
        },
        "training": {
            "batch_size": 32, "num_gen_steps": 3, "n_critic": 1,
            "lr_g": 1e-4, "lr_d": 1e-4, "betas": [0.0, 0.9], "lambda_gp": 5.0,
            "ema_decay": 0.0, "num_workers": 0, "distance_reg_alpha": 0.0,
            "distance_reg_max_points": 16, "amp": False,
            "log_every": 1, "eval_every": 100, "save_every": 3,
        },
    }
    cfg["training"].update(training)
    return cfg


def _metrics(meta):
    return [m for m in meta["metrics"] if "g_loss" in m]


def test_absent_keys_leave_the_generator_loss_unchanged(tmp_path):
    # The backward-compatibility guarantee: with lid_reg_alpha unset, g_loss
    # is exactly adv_loss, so v0-v3 are bit-identical to before this change.
    _, meta = train(_config(tmp_path))
    rows = _metrics(meta)
    assert rows
    for row in rows:
        assert row["lid_reg"] == 0.0
        assert row["g_loss"] == pytest.approx(row["adv_loss"], abs=0.0)


def test_explicit_zero_alpha_behaves_the_same(tmp_path):
    _, meta = train(_config(tmp_path, lid_reg_alpha=0.0))
    for row in _metrics(meta):
        assert row["lid_reg"] == 0.0
        assert row["g_loss"] == pytest.approx(row["adv_loss"], abs=0.0)


def test_enabled_regulariser_contributes_to_the_generator_loss(tmp_path):
    _, meta = train(
        _config(tmp_path, lid_reg_alpha=0.5, lid_reg_k=8, lid_reg_max_points=32)
    )
    rows = _metrics(meta)
    assert rows
    assert any(row["lid_reg"] > 0.0 for row in rows)
    for row in rows:
        assert row["g_loss"] == pytest.approx(
            row["adv_loss"] + 0.5 * row["lid_reg"], rel=1e-5
        )


def test_training_completes_with_the_regulariser_enabled(tmp_path):
    ckpt, meta = train(_config(tmp_path, lid_reg_alpha=0.1, lid_reg_k=8))
    assert ckpt.exists()
    assert all(row["lid_reg"] >= 0.0 for row in _metrics(meta))
