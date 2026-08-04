"""Fixtures shared by the eval test modules.

`write_tiny_gated_run` builds a real checkpoint rather than a stub because
`load_generator` rebuilds the architecture from `run_config.yaml` and then
loads a state dict into it -- a fake file would only exercise the path
lookup, not the round trip.
"""

import pytest
import torch
import yaml

from src.eval import compare_variants as cv
from src.models.critic import Critic
from src.models.generator import build_generator
from src.train.train_wgan_gp import save_checkpoint


@pytest.fixture
def make_run_dir():
    """Create a run directory with placeholder artifacts for resolve tests."""

    def _make(root, name, with_checkpoint=True, with_config=True):
        d = root / name
        d.mkdir(parents=True)
        if with_config:
            (d / "run_config.yaml").write_text("model: {}\n")
        if with_checkpoint:
            (d / "best_generator.pt").write_bytes(b"")
        return d

    return _make


@pytest.fixture
def write_tiny_gated_run():
    """Write a real save_checkpoint + run_config pair for a tiny gated model."""

    def _write(tmp_path, name="tiny_gated", descriptor_dim=8):
        model_cfg = {
            "latent_dim": 4,
            "generator_hidden_dims": [6],
            "negative_slope": 0.2,
            "generator_type": "gated",
            "gate_temperature": 0.5,
            "logit_clamp": 4.0,
        }

        generator = build_generator(model_cfg, output_dim=descriptor_dim)
        critic = Critic(
            input_dim=descriptor_dim, hidden_dims=[6], negative_slope=0.2
        )
        optim_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
        optim_d = torch.optim.Adam(critic.parameters(), lr=1e-4)

        run_dir = tmp_path / "runs" / name
        save_checkpoint(
            generator,
            critic,
            optim_g,
            optim_d,
            out_dir=run_dir,
            step=1,
            best=True,
            generator_weights="live",
        )

        run_config = {
            "device": "cpu",
            "model": model_cfg,
            "data": {"descriptor_dim": descriptor_dim},
        }
        (run_dir / "run_config.yaml").write_text(yaml.safe_dump(run_config))

        variant = cv.Variant(name, "configs/sift_gan_v2.yaml", f"runs/{name}")
        return variant, descriptor_dim

    return _write
