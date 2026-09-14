"""One training step end-to-end with `critic_type: neighbourhood_bank`.

`tests/test_train_smoke.py` is left untouched: it is the guard that plain
critics still receive plain tensor batches. This file covers what the bank
critic adds to the loop: the row-index dataset, the bank draw, the ring
write, and the resume path that must rebuild the same bank.
"""

import math

import torch
from tests.test_train_smoke import make_config

from src.models.critic import (
    BankNeighbourhoodCritic,
    build_critic,
    draw_real_bank_indices,
)
from src.train.train_wgan_gp import train

BANK_SIZE = 64  # batch 32, so the ring is full after exactly two generator steps


def make_bank_config(tmp_path, name="bank", seed=0):
    cfg = make_config(tmp_path, "mlp")
    cfg["seed"] = seed
    cfg["output_dir"] = str(tmp_path / name)
    cfg["model"].update(
        {
            "critic_type": "neighbourhood_bank",
            "critic_k": 5,
            "critic_distance_floor": 0.01,
            "critic_bank_size": BANK_SIZE,
        }
    )
    return cfg


def test_bank_critic_smoke_run_writes_a_checkpoint_the_factory_can_load(tmp_path):
    """Spec's end-to-end test. Catches: the trainer not using the factory, a
    state-dict key mismatch, indices never drawn (still -1), or the ring
    written once per critic step instead of once per generator step."""
    cfg = make_bank_config(tmp_path)
    ckpt_path, meta = train(cfg)

    assert ckpt_path.exists()
    ckpt = torch.load(tmp_path / "bank" / "checkpoint_step_4.pt", weights_only=False)
    critic = build_critic(cfg["model"], input_dim=16)
    assert isinstance(critic, BankNeighbourhoodCritic)
    critic.load_state_dict(ckpt["critic_state_dict"])

    indices = critic.real_bank_indices
    assert indices.shape == (BANK_SIZE,)
    assert indices.min() >= 0 and indices.max() < meta["data"]["num_train"]
    assert len(set(indices.tolist())) == BANK_SIZE
    assert "real_bank" not in ckpt["critic_state_dict"]
    assert meta["critic_bank_size"] == BANK_SIZE
    expected_fill = math.ceil(BANK_SIZE / cfg["training"]["batch_size"])
    assert meta["fake_bank_filled_step"] == expected_fill
    for entry in meta["metrics"]:
        assert math.isfinite(entry["g_loss"]) and math.isfinite(entry["d_loss"])


def test_resume_rebuilds_the_real_bank_from_the_checkpoint_not_from_the_seed(tmp_path):
    """Spec's round-trip test at trainer level. The resumed config carries a
    different seed, so a trainer that redraws on resume produces different
    indices; catches exactly that redraw. (The different seed also changes
    the train/holdout split, so the rows behind the indices differ; resuming
    under another seed is already unsupported for the data order, and this
    test compares the indices, which is what the checkpoint carries.)"""
    first = make_bank_config(tmp_path, name="first", seed=0)
    _, meta_first = train(first)
    step4 = torch.load(tmp_path / "first" / "checkpoint_step_4.pt", weights_only=False)
    drawn_at_seed_0 = step4["critic_state_dict"]["real_bank_indices"]

    second = make_bank_config(tmp_path, name="second", seed=1)
    second["training"]["num_gen_steps"] = 8
    train(second, resume=str(tmp_path / "first" / "checkpoint_step_4.pt"))
    step8 = torch.load(tmp_path / "second" / "checkpoint_step_8.pt", weights_only=False)

    assert torch.equal(step8["critic_state_dict"]["real_bank_indices"], drawn_at_seed_0)
    n_train = meta_first["data"]["num_train"]
    redraw = draw_real_bank_indices(n_train, BANK_SIZE, seed=1)
    assert not torch.equal(drawn_at_seed_0, redraw), "the mutation must be observable"
