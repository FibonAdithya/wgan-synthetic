import pytest
import torch

from src.models.critic import Critic
from src.models.generator import Generator
from src.train.train_wgan_gp import save_checkpoint, train


def _tiny_setup():
    generator = Generator(latent_dim=4, output_dim=6, hidden_dims=[8])
    critic = Critic(input_dim=6, hidden_dims=[8])
    optim_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optim_d = torch.optim.Adam(critic.parameters(), lr=1e-4)
    return generator, critic, optim_g, optim_d


def test_checkpoint_carries_the_ema_shadow_and_best_cov(tmp_path):
    generator, critic, optim_g, optim_d = _tiny_setup()
    ema = {name: p.detach().clone() for name, p in generator.named_parameters()}

    save_checkpoint(
        generator, critic, optim_g, optim_d, tmp_path, step=500,
        ema_params=ema, ema_step=500, best_cov=0.25,
    )

    ckpt = torch.load(tmp_path / "checkpoint_step_500.pt", weights_only=False)
    assert ckpt["ema_step"] == 500
    assert ckpt["best_cov"] == 0.25
    assert set(ckpt["ema_params"]) == set(ema)
    for name, tensor in ema.items():
        assert torch.allclose(ckpt["ema_params"][name], tensor)


def test_checkpoint_without_ema_records_an_empty_shadow(tmp_path):
    generator, critic, optim_g, optim_d = _tiny_setup()
    save_checkpoint(generator, critic, optim_g, optim_d, tmp_path, step=10)
    ckpt = torch.load(tmp_path / "checkpoint_step_10.pt", weights_only=False)
    assert ckpt["ema_params"] == {}
    assert ckpt["ema_step"] == 0


def _smoke_config(tmp_path, num_gen_steps, save_every):
    return {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "run"),
        "data": {
            # None plus synthetic_if_missing is how tests/test_train_smoke.py
            # drives training without the 512MB dataset.
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
            "batch_size": 16, "num_gen_steps": num_gen_steps, "n_critic": 1,
            "lr_g": 1e-4, "lr_d": 1e-4, "betas": [0.0, 0.9], "lambda_gp": 5.0,
            "ema_decay": 0.9, "num_workers": 0, "distance_reg_alpha": 0.0,
            "distance_reg_max_points": 16, "amp": False,
            "log_every": 100, "eval_every": 100, "save_every": save_every,
        },
    }


def test_resuming_continues_from_the_saved_step(tmp_path):
    cfg = _smoke_config(tmp_path, num_gen_steps=4, save_every=2)
    train(cfg)
    ckpt_path = tmp_path / "run" / "checkpoint_step_4.pt"
    assert ckpt_path.exists()

    cfg_more = _smoke_config(tmp_path, num_gen_steps=6, save_every=2)
    _, meta = train(cfg_more, resume=str(ckpt_path))

    # A resumed run must not redo the first four steps.
    assert meta["resumed_from_step"] == 4
    assert (tmp_path / "run" / "checkpoint_step_6.pt").exists()


def test_resuming_restores_the_ema_shadow(tmp_path):
    cfg = _smoke_config(tmp_path, num_gen_steps=4, save_every=2)
    train(cfg)
    ckpt = torch.load(tmp_path / "run" / "checkpoint_step_4.pt", weights_only=False)
    assert ckpt["ema_step"] == 4
    assert ckpt["ema_params"], "EMA shadow must be non-empty at ema_decay 0.9"


def test_resuming_past_the_step_budget_is_rejected(tmp_path):
    cfg = _smoke_config(tmp_path, num_gen_steps=4, save_every=2)
    train(cfg)
    ckpt_path = tmp_path / "run" / "checkpoint_step_4.pt"
    # Asking to resume into a budget already exhausted is a config mistake,
    # not a no-op: silently doing nothing would look like a successful run.
    with pytest.raises(ValueError, match="already at or past"):
        train(_smoke_config(tmp_path, num_gen_steps=4, save_every=2),
              resume=str(ckpt_path))
