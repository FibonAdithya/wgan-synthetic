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
        generator,
        critic,
        optim_g,
        optim_d,
        tmp_path,
        step=500,
        ema_params=ema,
        ema_step=500,
        best_cov=0.25,
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
            "batch_size": 16,
            "num_gen_steps": num_gen_steps,
            "n_critic": 1,
            "lr_g": 1e-4,
            "lr_d": 1e-4,
            "betas": [0.0, 0.9],
            "lambda_gp": 5.0,
            "ema_decay": 0.9,
            "num_workers": 0,
            "distance_reg_alpha": 0.0,
            "distance_reg_max_points": 16,
            "amp": False,
            "log_every": 100,
            "eval_every": 100,
            "save_every": save_every,
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


def test_training_writes_a_populated_ema_shadow_into_the_checkpoint(tmp_path):
    # Not a resume test: just confirms train() writes what a resume needs.
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
        train(
            _smoke_config(tmp_path, num_gen_steps=4, save_every=2),
            resume=str(ckpt_path),
        )


def test_resuming_does_not_redo_steps(tmp_path):
    cfg = _smoke_config(tmp_path, num_gen_steps=4, save_every=2)
    train(cfg)
    ckpt_path = tmp_path / "run" / "checkpoint_step_4.pt"

    cfg_more = _smoke_config(tmp_path, num_gen_steps=6, save_every=2)
    _, meta = train(cfg_more, resume=str(ckpt_path))

    # A resumed run must not redo the first four steps: exactly two more
    # EMA updates should have happened (step 5, step 6), not three.
    assert meta["resumed_from_step"] == 4
    ckpt6 = torch.load(tmp_path / "run" / "checkpoint_step_6.pt", weights_only=False)
    assert ckpt6["ema_step"] == 6


def test_resuming_carries_the_generator_and_critic_weights_forward(tmp_path):
    cfg = _smoke_config(tmp_path, num_gen_steps=4, save_every=1)
    train(cfg)
    ckpt_path = tmp_path / "run" / "checkpoint_step_4.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)

    # Give the generator and critic state distinctive, unmistakable values --
    # far from anything a fresh init would produce -- so the test does not
    # depend on how much one training step happens to move parameters (which,
    # at lr=1e-4 and with both training calls sharing the same config seed,
    # can coincidentally look similar to a fresh init too).
    distinctive_g = {
        name: torch.full_like(tensor, 1000.0)
        for name, tensor in ckpt["generator_state_dict"].items()
    }
    distinctive_c = {
        name: torch.full_like(tensor, -1000.0)
        for name, tensor in ckpt["critic_state_dict"].items()
    }
    ckpt["generator_state_dict"] = distinctive_g
    ckpt["critic_state_dict"] = distinctive_c
    edited_path = tmp_path / "edited_generator.pt"
    torch.save(ckpt, edited_path)

    cfg_more = _smoke_config(tmp_path, num_gen_steps=5, save_every=1)
    train(cfg_more, resume=str(edited_path))
    ckpt5 = torch.load(tmp_path / "run" / "checkpoint_step_5.pt", weights_only=False)

    for name, tensor in ckpt5["generator_state_dict"].items():
        # One step at lr=1e-4 barely moves a parameter that started at 1000;
        # a generator that was never loaded from the checkpoint would instead
        # sit near its normal random init, nowhere close to 1000.
        assert torch.allclose(tensor, distinctive_g[name], atol=1.0), name
    for name, tensor in ckpt5["critic_state_dict"].items():
        assert torch.allclose(tensor, distinctive_c[name], atol=1.0), name

    # Adam's own step counter is a clean signal that optim_g's state (not
    # just the model weights) was actually restored: a fresh optimizer would
    # start this counter at 1 for the single resumed step, not continue from
    # the 4 already recorded in the checkpoint.
    optim_g_step = next(iter(ckpt5["optim_g_state_dict"]["state"].values()))["step"]
    assert int(optim_g_step) == 5


def test_resuming_restores_the_ema_shadow_not_reinitialized(tmp_path):
    cfg = _smoke_config(tmp_path, num_gen_steps=4, save_every=1)
    train(cfg)
    ckpt4 = torch.load(tmp_path / "run" / "checkpoint_step_4.pt", weights_only=False)

    cfg_more = _smoke_config(tmp_path, num_gen_steps=5, save_every=1)
    train(cfg_more, resume=str(tmp_path / "run" / "checkpoint_step_4.pt"))
    ckpt5 = torch.load(tmp_path / "run" / "checkpoint_step_5.pt", weights_only=False)

    def flat(state_dict):
        return torch.cat([v.flatten().float() for v in state_dict.values()])

    norm4 = flat(ckpt4["ema_params"]).norm()
    norm5 = flat(ckpt5["ema_params"]).norm()
    # A correctly-restored shadow decays by only 10% (decay=0.9) for one more
    # update, so norm5 stays close to (and can exceed) norm4. A shadow that
    # was reinitialised to zero on resume instead starts the accumulation
    # over and collapses to roughly (1 - decay) times a single parameter
    # draw -- far smaller than four steps of accumulation.
    assert norm5 > 0.6 * norm4


def test_resuming_carries_best_cov_forward(tmp_path):
    cfg = _smoke_config(tmp_path, num_gen_steps=4, save_every=2)
    train(cfg)
    best_path = tmp_path / "run" / "best_generator.pt"
    assert best_path.exists()
    before_step = torch.load(best_path, weights_only=False)["step"]

    # Hand-edit best_cov to a value no later eval could ever beat.
    ckpt_path = tmp_path / "run" / "checkpoint_step_4.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    ckpt["best_cov"] = -1.0
    edited_path = tmp_path / "edited_step_4.pt"
    torch.save(ckpt, edited_path)

    cfg_more = _smoke_config(tmp_path, num_gen_steps=6, save_every=2)
    train(cfg_more, resume=str(edited_path))

    after_step = torch.load(best_path, weights_only=False)["step"]
    # If best_cov had not been carried (reset to inf), the step-6 eval would
    # beat it and overwrite best_generator.pt.
    assert after_step == before_step


def test_resume_refuses_a_checkpoint_without_an_ema_shadow(tmp_path):
    generator, critic, optim_g, optim_d = _tiny_setup()
    out_dir = tmp_path / "run"
    save_checkpoint(
        generator,
        critic,
        optim_g,
        optim_d,
        out_dir,
        step=2,
        ema_params={},
        ema_step=0,
    )
    cfg = _smoke_config(tmp_path, num_gen_steps=4, save_every=2)
    with pytest.raises(ValueError, match="no EMA shadow"):
        train(cfg, resume=str(out_dir / "checkpoint_step_2.pt"))


def test_resume_refuses_a_checkpoint_holding_ema_generator_weights(tmp_path):
    generator, critic, optim_g, optim_d = _tiny_setup()
    out_dir = tmp_path / "run"
    ema = {name: p.detach().clone() for name, p in generator.named_parameters()}
    save_checkpoint(
        generator,
        critic,
        optim_g,
        optim_d,
        out_dir,
        step=2,
        generator_weights="ema",
        ema_params=ema,
        ema_step=2,
    )
    cfg = _smoke_config(tmp_path, num_gen_steps=4, save_every=2)
    with pytest.raises(ValueError, match="'live'"):
        train(cfg, resume=str(out_dir / "checkpoint_step_2.pt"))
