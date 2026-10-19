"""output_normalize: false must reach every place generator output is drawn.

Each test catches one call site that still normalises unconditionally."""

import sys

import numpy as np
import torch
import yaml
from tests.test_train_smoke import make_config

from src.eval.evaluate_distribution import load_generator, sample_fake
from src.sample import generate
from src.train import train_wgan_gp
from src.train.train_wgan_gp import output_normalize, sample_generator, train


class Scaled(torch.nn.Module):
    """Emits rows of norm 3, so any normalisation is visible."""

    def forward(self, z):
        return 3.0 * z / torch.linalg.vector_norm(z, dim=1, keepdim=True)


def _tiny_unnormalised_run(tmp_path):
    cfg = make_config(tmp_path, "mlp")
    cfg["data"]["preprocess"]["l2_normalize"] = False
    cfg["model"]["output_normalize"] = False
    cfg["training"]["num_gen_steps"] = 2
    ckpt, _ = train(cfg)
    return ckpt, ckpt.parent / "run_config.yaml"


def test_sample_generator_default_still_normalises():
    """Catches the default flipping to unnormalised (would change every family)."""
    x = sample_generator(Scaled(), 8, 4, 4, torch.device("cpu"))
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, rtol=1e-5)


def test_sample_generator_normalize_false_keeps_norms():
    """Catches sample_generator ignoring normalize=False."""
    x = sample_generator(Scaled(), 8, 4, 4, torch.device("cpu"), normalize=False)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 3.0, rtol=1e-5)


def test_output_normalize_reads_model_cfg_with_true_default():
    """Catches a default of False, or reading the key from the wrong section."""
    assert output_normalize({}) is True
    assert output_normalize({"output_normalize": False}) is False


def test_training_with_output_normalize_false_learns_scale(tmp_path):
    """Catches the two training-loop sites still normalising: real rows have
    norm ~4 (unnormalised Gaussian in 16 dims), so a normalised generator's
    eval samples would sit at exactly 1.0."""
    cfg = make_config(tmp_path, "mlp")
    cfg["data"]["preprocess"]["l2_normalize"] = False
    cfg["model"]["output_normalize"] = False
    cfg["training"]["num_gen_steps"] = 2
    ckpt, _ = train(cfg)
    run_cfg = yaml.safe_load((ckpt.parent / "run_config.yaml").read_text())
    assert run_cfg["model"]["output_normalize"] is False

    gen = load_generator(run_cfg, ckpt, torch.device("cpu"))
    x = sample_generator(gen, 64, 8, 64, torch.device("cpu"), normalize=False)
    assert not np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-3)


def test_generate_cli_honours_output_normalize(tmp_path, monkeypatch):
    """Catches generate.py's own hard-coded division surviving."""
    ckpt, run_cfg = _tiny_unnormalised_run(tmp_path)
    out = tmp_path / "s.npy"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--checkpoint",
            str(ckpt),
            "--config",
            str(run_cfg),
            "--num-samples",
            "32",
            "--output-path",
            str(out),
        ],
    )
    generate.main()
    assert not np.allclose(np.linalg.norm(np.load(out), axis=1), 1.0, atol=1e-3)


def test_evaluate_distribution_honours_output_normalize(tmp_path):
    """Catches sample_fake's hard-coded division surviving."""
    ckpt, run_cfg = _tiny_unnormalised_run(tmp_path)
    gen = load_generator(yaml.safe_load(run_cfg.read_text()), ckpt, torch.device("cpu"))
    x = sample_fake(gen, 8, 32, 32, torch.device("cpu"), normalize=False)
    assert not np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-3)


def test_training_eval_samples_honour_output_normalize(tmp_path, monkeypatch):
    """Catches the in-training eval call not passing normalize=norm_out.
    A spy wraps the real sample_generator, so deleting it still fails the test."""
    seen = []
    real = train_wgan_gp.sample_generator

    def spy(*args, **kwargs):
        seen.append(kwargs.get("normalize", True))
        return real(*args, **kwargs)

    monkeypatch.setattr(train_wgan_gp, "sample_generator", spy)
    _tiny_unnormalised_run(tmp_path)
    assert seen and not any(seen)
