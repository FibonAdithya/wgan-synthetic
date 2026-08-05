import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.eval import compare_variants as cv
from src.eval import eda_report


def test_variants_are_the_four_named_ones():
    assert [v.name for v in cv.VARIANTS] == ["v0", "v1", "v1_5", "v2"]


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_variant_config_exists():
    for v in cv.VARIANTS:
        assert (REPO_ROOT / v.config_path).exists(), f"missing config for {v.name}"


def test_resolve_skips_variants_with_no_checkpoint(tmp_path, make_run_dir):
    variants = (
        cv.Variant("v0", "configs/sift/v0.yaml", "runs/a"),
        cv.Variant("v1", "configs/sift/v1.yaml", "runs/b"),
    )
    make_run_dir(tmp_path / "runs", "a")
    make_run_dir(tmp_path / "runs", "b", with_checkpoint=False)

    found, skipped = cv.resolve_variants(variants, root=tmp_path)

    assert [v.name for v in found] == ["v0"]
    assert [v.name for v, _ in skipped] == ["v1"]
    assert "best_generator.pt" in skipped[0][1]


def test_resolve_skips_variants_with_no_run_config(tmp_path, make_run_dir):
    variants = (cv.Variant("v0", "configs/sift/v0.yaml", "runs/a"),)
    make_run_dir(tmp_path / "runs", "a", with_config=False)

    found, skipped = cv.resolve_variants(variants, root=tmp_path)

    assert found == []
    assert "run_config.yaml" in skipped[0][1]


def test_resolve_reports_a_missing_run_dir(tmp_path):
    variants = (cv.Variant("v0", "configs/sift/v0.yaml", "runs/nope"),)

    found, skipped = cv.resolve_variants(variants, root=tmp_path)

    assert found == []
    assert [v.name for v, _ in skipped] == ["v0"]


def test_resolve_finds_everything_when_present(tmp_path, make_run_dir):
    variants = (
        cv.Variant("v0", "configs/sift/v0.yaml", "runs/a"),
        cv.Variant("v2", "configs/sift/v2.yaml", "runs/b"),
    )
    make_run_dir(tmp_path / "runs", "a")
    make_run_dir(tmp_path / "runs", "b")

    found, skipped = cv.resolve_variants(variants, root=tmp_path)

    assert [v.name for v in found] == ["v0", "v2"]
    assert skipped == []


def test_generate_samples_round_trips_a_real_gated_checkpoint(
    tmp_path, write_tiny_gated_run
):
    """Exercise the full generate_samples seam end to end.

    This is the only test in the branch that proves a checkpoint written by
    the real `save_checkpoint` (as `train_wgan_gp.train` writes them) can be
    loaded back by `generate_samples` and sampled -- and, since it uses a
    gated generator, that the sparse->gated rename did not break checkpoint
    loadability.
    """
    variant, descriptor_dim = write_tiny_gated_run(tmp_path)
    out_dir = tmp_path / "samples"
    out_dir.mkdir()

    num_samples = 20
    path = cv.generate_samples(
        variant,
        root=tmp_path,
        num_samples=num_samples,
        batch_size=8,
        out_dir=out_dir,
        seed=42,
    )

    assert path.exists()
    x = np.load(path)
    assert x.shape == (num_samples, descriptor_dim)

    norms = (x ** 2).sum(axis=1) ** 0.5
    assert (abs(norms - 1.0) < 1e-4).all(), "gated generator output must be unit-norm"
    assert (x == 0.0).any(), "gated generator should produce exact zeros"


def test_variant_seed_differs_per_variant_and_is_stable():
    seeds = {v: cv.variant_seed(42, v) for v in ("v0", "v1", "v1_5", "v2")}
    assert len(set(seeds.values())) == 4, "each variant must get its own latents"
    assert cv.variant_seed(42, "v2") == seeds["v2"], "seeding must be deterministic"
    assert cv.variant_seed(7, "v2") != seeds["v2"], "--seed must still move it"


def test_generate_samples_does_not_depend_on_preceding_variants(
    tmp_path, write_tiny_gated_run
):
    """A skipped variant must not change the samples of the ones that survive.

    Variants are skipped whenever their checkpoint is not on this machine, so
    a single seed for the whole loop would make v2's samples depend on how
    many earlier checkpoints happened to be present.
    """
    variant, _ = write_tiny_gated_run(tmp_path)
    out_dir = tmp_path / "samples"
    out_dir.mkdir()

    def draw(sub_dir):
        sub_dir.mkdir()
        return np.load(
            cv.generate_samples(
                variant,
                root=tmp_path,
                num_samples=20,
                batch_size=8,
                out_dir=sub_dir,
                seed=42,
            )
        )

    first = draw(out_dir / "a")
    # Stand in for the RNG an earlier variant would have consumed.
    torch.randn(1000)
    second = draw(out_dir / "b")

    np.testing.assert_array_equal(first, second)


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
        ann_k=eda_report.ANN_K_DEFAULT,
        ann_hub_k=eda_report.ANN_HUB_K_DEFAULT,
        ann_max_rows=eda_report.ANN_MAX_ROWS_DEFAULT,
        ivf_nlist=eda_report.IVF_NLIST_DEFAULT,
        bins=8,
        top_divergent=4,
        seed=42,
        glyph_samples=eda_report.GLYPH_SAMPLES_DEFAULT,
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
