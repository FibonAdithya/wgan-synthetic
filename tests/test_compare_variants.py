import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.eval import compare_variants as cv
from src.eval import eda_report
from src.models.critic import Critic
from src.models.generator import build_generator
from src.train.train_wgan_gp import save_checkpoint


def test_variants_are_the_four_named_ones():
    assert [v.name for v in cv.VARIANTS] == ["v0", "v1", "v1_5", "v2"]


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_variant_config_exists():
    for v in cv.VARIANTS:
        assert (REPO_ROOT / v.config_path).exists(), f"missing config for {v.name}"


def _make_run_dir(root, name, with_checkpoint=True, with_config=True):
    d = root / name
    d.mkdir(parents=True)
    if with_config:
        (d / "run_config.yaml").write_text("model: {}\n")
    if with_checkpoint:
        (d / "best_generator.pt").write_bytes(b"")
    return d


def test_resolve_skips_variants_with_no_checkpoint(tmp_path):
    variants = (
        cv.Variant("v0", "configs/sift_gan_v0.yaml", "runs/a"),
        cv.Variant("v1", "configs/sift_gan_v1.yaml", "runs/b"),
    )
    _make_run_dir(tmp_path / "runs", "a")
    _make_run_dir(tmp_path / "runs", "b", with_checkpoint=False)

    found, skipped = cv.resolve_variants(variants, root=tmp_path)

    assert [v.name for v in found] == ["v0"]
    assert [v.name for v, _ in skipped] == ["v1"]
    assert "best_generator.pt" in skipped[0][1]


def test_resolve_skips_variants_with_no_run_config(tmp_path):
    variants = (cv.Variant("v0", "configs/sift_gan_v0.yaml", "runs/a"),)
    _make_run_dir(tmp_path / "runs", "a", with_config=False)

    found, skipped = cv.resolve_variants(variants, root=tmp_path)

    assert found == []
    assert "run_config.yaml" in skipped[0][1]


def test_resolve_reports_a_missing_run_dir(tmp_path):
    variants = (cv.Variant("v0", "configs/sift_gan_v0.yaml", "runs/nope"),)

    found, skipped = cv.resolve_variants(variants, root=tmp_path)

    assert found == []
    assert [v.name for v, _ in skipped] == ["v0"]


def test_resolve_finds_everything_when_present(tmp_path):
    variants = (
        cv.Variant("v0", "configs/sift_gan_v0.yaml", "runs/a"),
        cv.Variant("v2", "configs/sift_gan_v2.yaml", "runs/b"),
    )
    _make_run_dir(tmp_path / "runs", "a")
    _make_run_dir(tmp_path / "runs", "b")

    found, skipped = cv.resolve_variants(variants, root=tmp_path)

    assert [v.name for v in found] == ["v0", "v2"]
    assert skipped == []


def test_generate_samples_round_trips_a_real_gated_checkpoint(tmp_path):
    """Exercise the full generate_samples seam end to end.

    This is the only test in the branch that proves a checkpoint written by
    the real `save_checkpoint` (as `train_wgan_gp.train` writes them) can be
    loaded back by `generate_samples` and sampled -- and, since it uses a
    gated generator, that the sparse->gated rename did not break checkpoint
    loadability.
    """
    model_cfg = {
        "latent_dim": 4,
        "generator_hidden_dims": [6],
        "negative_slope": 0.2,
        "generator_type": "gated",
        "gate_temperature": 0.5,
        "logit_clamp": 4.0,
    }
    descriptor_dim = 8

    generator = build_generator(model_cfg, output_dim=descriptor_dim)
    critic = Critic(input_dim=descriptor_dim, hidden_dims=[6], negative_slope=0.2)
    optim_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optim_d = torch.optim.Adam(critic.parameters(), lr=1e-4)

    run_dir = tmp_path / "runs" / "tiny_gated"
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

    variant = cv.Variant("tiny", "configs/sift_gan_v2.yaml", "runs/tiny_gated")
    out_dir = tmp_path / "samples"
    out_dir.mkdir()

    num_samples = 20
    path = cv.generate_samples(
        variant, root=tmp_path, num_samples=num_samples, batch_size=8, out_dir=out_dir
    )

    assert path.exists()
    x = np.load(path)
    assert x.shape == (num_samples, descriptor_dim)

    norms = (x ** 2).sum(axis=1) ** 0.5
    assert (abs(norms - 1.0) < 1e-4).all(), "gated generator output must be unit-norm"
    assert (x == 0.0).any(), "gated generator should produce exact zeros"


def test_report_args_match_eda_report_fields(monkeypatch, tmp_path):
    """Parity check for Finding 4: compare_variants's Namespace vs eda_report's.

    `build_report_args` hand-builds the Namespace `eda_report.run` consumes.
    If `eda_report.parse_args` gains a required field and this helper is not
    updated, `compare_variants` breaks at runtime only after sampling. Assert
    the field sets stay identical so drift is caught at test time instead.
    """
    args = argparse.Namespace(
        real_path="real.npy",
        real_format="npy",
        output_dir=str(tmp_path / "out"),
        max_vectors=100,
        num_pairs=200,
        knn=3,
        bins=8,
        top_divergent=4,
        seed=42,
        no_png=True,
        plotlyjs="cdn",
    )
    report_args = cv.build_report_args(args, specs=["v0=a.npy"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eda_report.py",
            "--real-path",
            "real.npy",
            "--output-dir",
            str(tmp_path / "out2"),
        ],
    )
    eda_args = eda_report.parse_args()

    assert set(vars(report_args)) == set(vars(eda_args))
