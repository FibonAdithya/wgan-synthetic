"""Fixtures shared by the eval test modules.

`write_tiny_gated_run` builds a real checkpoint rather than a stub because
`load_generator` rebuilds the architecture from `run_config.yaml` and then
loads a state dict into it -- a fake file would only exercise the path
lookup, not the round trip.
"""

import argparse

import numpy as np
import pytest
import torch
import yaml

from src.eval import compare_variants as cv
from src.eval.eda import config as eda_config
from src.models.critic import Critic
from src.models.generator import build_generator
from src.train.train_wgan_gp import save_checkpoint


def make_args(tmp_path, real, synthetic):
    real_path = tmp_path / "real.npy"
    np.save(real_path, real)
    specs = []
    for label, arr in synthetic.items():
        p = tmp_path / f"{label}.npy"
        np.save(p, arr)
        specs.append(f"{label}={p}")
    return argparse.Namespace(
        real_path=str(real_path),
        real_format="npy",
        synthetic_path=specs,
        synthetic_format="npy",
        output_dir=str(tmp_path / "out"),
        preprocess="l2",
        metric=eda_config.METRIC_DEFAULT,
        max_vectors=200,
        num_pairs=500,
        knn=3,
        ann_k=eda_config.ANN_K_DEFAULT,
        ann_hub_k=eda_config.ANN_HUB_K_DEFAULT,
        ann_max_rows=eda_config.ANN_MAX_ROWS_DEFAULT,
        knn_max_rows=eda_config.KNN_MAX_ROWS_DEFAULT,
        ivf_nlist=eda_config.IVF_NLIST_DEFAULT,
        bins=16,
        top_divergent=4,
        seed=42,
        glyph_samples=eda_config.GLYPH_SAMPLES_DEFAULT,
        no_png=True,
        plotlyjs="cdn",
    )


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


def _write_run(tmp_path, name, descriptor_dim, model_cfg, config_path):
    generator = build_generator(model_cfg, output_dim=descriptor_dim)
    critic = Critic(input_dim=descriptor_dim, hidden_dims=[6], negative_slope=0.2)
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

    # `family_metric` resolves config paths against --root, which these tests
    # point at tmp_path. Write the config the Variant names so the fixture is
    # a complete tree rather than one that only happens to work.
    config_full = tmp_path / config_path
    config_full.parent.mkdir(parents=True, exist_ok=True)
    config_full.write_text(
        yaml.safe_dump({"data": {"metric": "l2", "descriptor_dim": descriptor_dim}})
    )

    return cv.Variant(name, config_path, f"runs/{name}"), descriptor_dim


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
        return _write_run(
            tmp_path, name, descriptor_dim, model_cfg, "configs/sift/v2.yaml"
        )

    return _write


@pytest.fixture
def write_tiny_mlp_run():
    """Same, for the plain MLP generator behind v0/v1/v1_5.

    Distinct from the gated fixture because `GatedGenerator` normalises its
    own output, which makes it blind to whether a caller normalises. The MLP
    does not, so it is the only fixture that can observe the unit-norm
    contract `sample_generator` is responsible for -- and it is also the
    generator that emits the negative bins this figure exists to expose.
    """

    def _write(tmp_path, name="tiny_mlp", descriptor_dim=8):
        model_cfg = {
            "latent_dim": 4,
            "generator_hidden_dims": [6],
            "negative_slope": 0.2,
            "generator_type": "mlp",
        }
        return _write_run(
            tmp_path, name, descriptor_dim, model_cfg, "configs/sift/v0.yaml"
        )

    return _write
