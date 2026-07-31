from pathlib import Path

import torch

from src.eval.evaluate_distribution import load_generator
from src.models.generator import SparseGenerator, build_generator


def test_load_generator_uses_configured_generator_factory(tmp_path: Path):
    config = {
        "data": {"descriptor_dim": 8},
        "model": {
            "latent_dim": 4,
            "generator_hidden_dims": [6],
            "negative_slope": 0.2,
            "generator_type": "sparse",
            "gate_temperature": 0.25,
            "logit_clamp": 4.0,
        },
    }
    expected = build_generator(config["model"], output_dim=8)
    checkpoint_path = tmp_path / "generator.pt"
    torch.save({"generator_state_dict": expected.state_dict()}, checkpoint_path)

    loaded = load_generator(config, checkpoint_path, torch.device("cpu"))

    assert isinstance(loaded, SparseGenerator)
    assert loaded.gate_temperature == 0.25
    assert loaded.logit_clamp == 4.0
    assert not loaded.training
