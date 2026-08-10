import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.eval import compare_variants as cv
from src.eval.eda import cli
from src.eval.eda import config as eda_config
from src.models.generator import build_generator


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

    norms = (x**2).sum(axis=1) ** 0.5
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
    """Parity check for Finding 4: compare_variants's Namespace vs the CLI's.

    `build_report_args` hand-builds the Namespace `eda.pipeline.run` consumes.
    If `eda.cli.parse_args` gains a required field and this helper is not
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
        ann_k=eda_config.ANN_K_DEFAULT,
        ann_hub_k=eda_config.ANN_HUB_K_DEFAULT,
        ann_max_rows=eda_config.ANN_MAX_ROWS_DEFAULT,
        knn_max_rows=eda_config.KNN_MAX_ROWS_DEFAULT,
        ivf_nlist=eda_config.IVF_NLIST_DEFAULT,
        bins=8,
        top_divergent=4,
        seed=42,
        glyph_samples=eda_config.GLYPH_SAMPLES_DEFAULT,
        no_png=True,
        plotlyjs="cdn",
        max_panel_dim=eda_config.MAX_PANEL_DIM_DEFAULT,
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
    eda_args = cli.parse_args()

    assert set(vars(report_args)) == set(vars(eda_args))


# --- Preprocess inversion -------------------------------------------------
#
# A run trained with `whiten: true` produces samples in the whitened space.
# These pin that `generate_samples` maps them back, that a run which cannot be
# mapped back is skipped before any sampling happens, and that the one
# silently-wrong configuration is refused outright.

from src.data.dataset import PreprocessConfig, _fit_preprocess_state  # noqa: E402


def _fit_state(dim=96, whiten=False, center=False, l2_normalize=True, seed=0):
    """Fit a transform on deliberately anisotropic data.

    The decaying per-dimension scale is what makes whitening a non-trivial
    transform, and is the pattern the inversion test looks for coming back.
    """
    rng = np.random.default_rng(seed)
    scale = np.linspace(1.0, 0.05, dim).astype(np.float32)
    x = (rng.normal(size=(400, dim)) * scale).astype(np.float32)
    cfg = PreprocessConfig(center=center, whiten=whiten, l2_normalize=l2_normalize)
    return _fit_preprocess_state(x_train=x, descriptor_dim=dim, cfg=cfg)


def _write_flat_run(tmp_path, name, *, whiten, center=False):
    """A run whose raw (pre-inversion) output is EXACTLY flat per-dimension.

    Not merely plausibly flat. Routed through an ordinarily-initialized
    generator, an untrained network's per-dimension output variance is itself
    uneven purely from weight init, and that unevenness can rival the
    anisotropy the inversion is supposed to restore -- making the assertion
    below a coin flip on RNG entropy rather than a property of the code.

    So: `generator_hidden_dims: []` gives a single `Linear(latent_dim, dim)`
    with no activation between it and the latent noise, and its weight rows
    are set to unit L2 norm. For z ~ N(0, I), Var(output_i) = ||W_i||^2 = 1
    for every i, exactly and independent of any seed. A spread appearing
    after inversion can then only come from the inversion itself.
    """
    dim, latent = 96, 16
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = {
        "latent_dim": latent,
        "generator_hidden_dims": [],
        "critic_hidden_dims": [32],
        "negative_slope": 0.2,
        "generator_type": "mlp",
    }
    run_config = {
        "device": "cpu",
        "model": model_cfg,
        "data": {
            "descriptor_dim": dim,
            "preprocess": {
                "center": center,
                "whiten": whiten,
                "l2_normalize": True,
            },
        },
    }
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(run_config))

    generator = build_generator(model_cfg, output_dim=dim)
    rows = np.random.default_rng(1).normal(size=(dim, latent)).astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    with torch.no_grad():
        generator.net[0].weight.copy_(torch.from_numpy(rows))
        generator.net[0].bias.zero_()
    torch.save(
        {"generator_state_dict": generator.state_dict()},
        run_dir / "best_generator.pt",
    )

    state = _fit_state(dim=dim, whiten=whiten, center=center)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"preprocess_state": state.to_serializable()}), encoding="utf-8"
    )
    return cv.Variant(name, "configs/deep/v2.yaml", f"runs/{name}")


def test_generate_samples_returns_a_whitened_run_to_original_coordinates(tmp_path):
    """The reason inversion lives in the sampling path at all.

    Checking magnitude alone would let a wrong-direction bug through: applying
    the forward whitening matrix instead of its inverse also inflates the
    per-dimension spread, just anti-correlated with the real pattern. So this
    asserts *direction* -- the inverted output's per-dimension variance must
    correlate strongly and positively with the `scale ** 2` pattern the
    transform was fitted on. A no-op, a wrong-direction inverse, or any other
    subtly broken one cannot produce that correlation by chance.
    """
    out_dir = tmp_path / "samples"
    out_dir.mkdir()

    def draw(variant, sub):
        (out_dir / sub).mkdir()
        return np.load(
            cv.generate_samples(variant, tmp_path, 2000, 512, out_dir / sub, seed=42)
        )

    inverted = draw(_write_flat_run(tmp_path, "w", whiten=True), "w")
    unwhitened = draw(_write_flat_run(tmp_path, "u", whiten=False), "u")

    expected = np.linspace(1.0, 0.05, 96).astype(np.float32) ** 2
    assert np.corrcoef(inverted.var(axis=0), expected)[0, 1] > 0.9
    assert abs(np.corrcoef(unwhitened.var(axis=0), expected)[0, 1]) < 0.3


def test_load_preprocess_state_returns_none_when_no_metadata_was_written(tmp_path):
    """Every SIFT run predates run_metadata.json; they must still sample."""
    run_dir = tmp_path / "runs" / "a"
    run_dir.mkdir(parents=True)
    assert cv.load_preprocess_state(run_dir) is None


def test_invert_samples_is_a_no_op_without_a_fitted_transform():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(cv.invert_samples(x, None), x)
    state = _fit_state(dim=4, whiten=False, center=False)
    np.testing.assert_array_equal(cv.invert_samples(x, state), x)


def test_invert_samples_refuses_centering_combined_with_l2_normalization():
    """Guards the one configuration that is wrong without being detectable.

    sample_generator L2-normalizes its output, and invert_preprocess only
    recovers directions exactly when no mean was subtracted. With both on the
    result is systematically wrong and nothing downstream would flag it, so
    this must raise rather than return.
    """
    state = _fit_state(whiten=True, center=True, l2_normalize=True)
    with pytest.raises(ValueError, match="center.*l2_normalize|l2_normalize.*center"):
        cv.invert_samples(np.zeros((4, 96), dtype=np.float32), state)


def test_resolve_skips_a_whitened_run_missing_its_metadata(tmp_path):
    """Reported as a skip before sampling starts, not as a mid-loop failure.

    Left unchecked this surfaces only after earlier variants have already
    generated hundreds of thousands of vectors.
    """
    variant = _write_flat_run(tmp_path, "w", whiten=True)
    (tmp_path / "runs" / "w" / "run_metadata.json").unlink()

    found, skipped = cv.resolve_variants((variant,), root=tmp_path)

    assert found == []
    assert "run_metadata.json" in skipped[0][1]


def test_resolve_skips_a_whitened_run_whose_metadata_omits_the_transform(tmp_path):
    """A present-but-useless metadata file must skip, exactly like an absent one.

    Checking only that run_metadata.json exists leaves the hole the check was
    added to close: `load_preprocess_state` returns None for a payload without
    a `preprocess_state` key, `invert_samples` reads that None as "nothing was
    fitted" and passes the samples straight through, and the run is written out
    in whitened coordinates with no error anywhere.
    """
    variant = _write_flat_run(tmp_path, "w", whiten=True)
    (tmp_path / "runs" / "w" / "run_metadata.json").write_text(
        json.dumps({"seed": 42}), encoding="utf-8"
    )

    found, skipped = cv.resolve_variants((variant,), root=tmp_path)

    assert found == []
    assert "run_metadata.json" in skipped[0][1]


def test_resolve_skips_a_whitened_run_whose_metadata_is_unparseable(tmp_path):
    """Truncated JSON is a skip, not a JSONDecodeError raised mid-sampling."""
    variant = _write_flat_run(tmp_path, "w", whiten=True)
    (tmp_path / "runs" / "w" / "run_metadata.json").write_text(
        '{"preprocess_state": ', encoding="utf-8"
    )

    found, skipped = cv.resolve_variants((variant,), root=tmp_path)

    assert found == []
    assert "run_metadata.json" in skipped[0][1]


def test_resolve_skips_a_centered_run_rather_than_failing_at_sample_time(tmp_path):
    """The center+l2 refusal belongs with the other skips, not mid-loop.

    `invert_samples` already refuses this combination, but it does so inside
    `generate_samples` -- after every earlier rung in the ladder has drawn its
    samples. `resolve_variants` already parses the run config to decide whether
    inversion is needed, so it can reject the combination there for the same
    reason the missing-metadata case is rejected there.
    """
    variant = _write_flat_run(tmp_path, "c", whiten=True, center=True)

    found, skipped = cv.resolve_variants((variant,), root=tmp_path)

    assert found == []
    assert "l2_normalize" in skipped[0][1]


def test_resolve_does_not_require_metadata_for_an_untransformed_run(tmp_path):
    variant = _write_flat_run(tmp_path, "u", whiten=False)
    (tmp_path / "runs" / "u" / "run_metadata.json").unlink()

    found, skipped = cv.resolve_variants((variant,), root=tmp_path)

    assert [v.name for v in found] == ["u"]
    assert skipped == []


# --- Manifest registry ----------------------------------------------------
#
# These replace the old `LADDERS` tests: the ladders are manifests on disk
# now, so what has to hold is that every family `--dataset` offers resolves to
# a readable manifest naming configs that exist.


def test_every_offered_dataset_has_a_manifest_whose_configs_exist():
    for dataset in cv.known_datasets():
        variants = cv.load_variants(cv.manifest_for_dataset(dataset))
        for variant in variants:
            assert (REPO_ROOT / variant.config_path).exists(), (
                f"{dataset}/{variant.name}"
            )


def test_both_shipped_families_are_offered():
    """DEEP arrived after SIFT; dropping it from `--dataset` is the regression."""
    assert {"sift", "deep"} <= set(cv.known_datasets())


def test_the_deep_manifest_covers_the_three_rungs():
    variants = cv.load_variants(cv.manifest_for_dataset("deep"))
    assert [v.name for v in variants] == ["v0", "v1", "v2"]


def test_families_do_not_share_run_directories():
    """A deep run must never be read out of a SIFT run directory."""
    sift = {v.run_dir for v in cv.load_variants(cv.manifest_for_dataset("sift"))}
    deep = {v.run_dir for v in cv.load_variants(cv.manifest_for_dataset("deep"))}
    assert not sift & deep


def write_manifest(path: Path, entries: list[dict]) -> Path:
    """Write a variant manifest, the shape `load_variants` reads."""
    path.write_text(yaml.safe_dump({"variants": entries}), encoding="utf-8")
    return path


def test_the_default_manifest_still_holds_the_four_historical_runs():
    """The manifest is the source of `VARIANTS`, so it pins the same values.

    Moving the list out of source must not quietly change what the headline
    comparison compares -- anyone who does have those run directories should
    see no difference.
    """
    variants = cv.load_variants(REPO_ROOT / cv.DEFAULT_MANIFEST)

    assert [(v.name, v.config_path, v.run_dir) for v in variants] == [
        ("v0", "configs/sift/v0.yaml", "runs/long_baseline"),
        ("v1", "configs/sift/v1.yaml", "runs/x100k_ema_only"),
        ("v1_5", "configs/sift/v1_5.yaml", "runs/x100k_improved"),
        ("v2", "configs/sift/v2.yaml", "runs/x100k_sparse_clamp4"),
    ]
    assert variants == cv.VARIANTS, "VARIANTS must be exactly the default manifest"


def test_load_variants_reads_an_alternative_manifest(tmp_path):
    manifest = write_manifest(
        tmp_path / "variants.yaml",
        [{"name": "mine", "config": "configs/sift/v0.yaml", "run_dir": "runs/mine"}],
    )

    variants = cv.load_variants(manifest)

    assert variants == (cv.Variant("mine", "configs/sift/v0.yaml", "runs/mine"),)


def test_load_variants_names_the_manifest_path_when_it_does_not_exist(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        cv.load_variants(tmp_path / "nope.yaml")

    assert "nope.yaml" in str(excinfo.value)


def test_load_variants_rejects_an_entry_missing_a_required_field(tmp_path):
    manifest = write_manifest(
        tmp_path / "variants.yaml", [{"name": "v0", "config": "configs/sift/v0.yaml"}]
    )

    with pytest.raises(SystemExit) as excinfo:
        cv.load_variants(manifest)

    assert "run_dir" in str(excinfo.value), "the missing field must be named"


def test_load_variants_rejects_duplicate_names(tmp_path):
    """Two variants sharing a name would overwrite each other's samples file."""
    entry = {"name": "v0", "config": "configs/sift/v0.yaml", "run_dir": "runs/a"}
    manifest = write_manifest(
        tmp_path / "variants.yaml", [entry, {**entry, "run_dir": "runs/b"}]
    )

    with pytest.raises(SystemExit) as excinfo:
        cv.load_variants(manifest)

    assert "v0" in str(excinfo.value)


def test_load_variants_rejects_a_manifest_with_no_variants_list(tmp_path):
    manifest = tmp_path / "variants.yaml"
    manifest.write_text("just a string\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        cv.load_variants(manifest)

    assert "variants" in str(excinfo.value)


def test_load_variants_rejects_an_empty_variants_list(tmp_path):
    manifest = write_manifest(tmp_path / "variants.yaml", [])

    with pytest.raises(SystemExit) as excinfo:
        cv.load_variants(manifest)

    assert str(manifest) in str(excinfo.value)


def test_describe_missing_names_the_path_and_what_would_produce_it(tmp_path):
    variant = cv.Variant("v0", "configs/sift/v0.yaml", "runs/long_baseline")
    _, skipped = cv.resolve_variants((variant,), root=tmp_path)

    message = cv.describe_missing(skipped, manifest=tmp_path / "m.yaml", root=tmp_path)

    assert str(tmp_path / "runs/long_baseline") in message, "name the missing path"
    assert "src.train.train_wgan_gp --config configs/sift/v0.yaml" in message
    assert str(tmp_path / "m.yaml") in message, "say which manifest asked for it"


def test_main_stops_before_sampling_when_a_run_directory_is_missing(
    monkeypatch, tmp_path
):
    """A fresh clone has no `runs/`, so this is its first experience of the tool.

    It must fail here -- naming the path and the command that produces it --
    rather than deep inside plotting, and must not leave a half-built output
    tree behind.
    """
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_variants.py",
            "--real-path",
            str(tmp_path / "real.npy"),
            "--output-dir",
            str(out_dir),
            "--root",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cv.main()

    message = str(excinfo.value)
    assert "runs/long_baseline" in message
    assert "src.train.train_wgan_gp" in message
    assert "--allow-missing" in message, "the escape hatch must be discoverable"
    assert not out_dir.exists(), "an aborted run must leave no output tree"


def test_main_reports_on_the_variants_a_custom_manifest_resolves(
    monkeypatch, tmp_path, write_tiny_gated_run
):
    """The manifest is what makes this reproducible off the training box.

    One entry points at a run that exists here, one at a run that does not;
    with --allow-missing the present one is sampled and handed to the report.
    """
    variant, _ = write_tiny_gated_run(tmp_path)
    manifest = write_manifest(
        tmp_path / "variants.yaml",
        [
            {
                "name": variant.name,
                "config": variant.config_path,
                "run_dir": variant.run_dir,
            },
            {
                "name": "absent",
                "config": "configs/sift/v0.yaml",
                "run_dir": "runs/nope",
            },
        ],
    )

    seen = {}

    def fake_run(args):
        seen["specs"] = args.synthetic_path
        return Path(args.output_dir) / "report.html"

    monkeypatch.setattr(cv.pipeline, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_variants.py",
            "--real-path",
            str(tmp_path / "real.npy"),
            "--output-dir",
            str(tmp_path / "out"),
            "--root",
            str(tmp_path),
            "--variants-manifest",
            str(manifest),
            "--allow-missing",
            "--num-samples",
            "20",
            "--batch-size",
            "8",
        ],
    )

    cv.main()

    assert len(seen["specs"]) == 1, "only the resolvable variant is sampled"
    label, _, path = seen["specs"][0].partition("=")
    assert label == variant.name
    assert np.load(path).shape[0] == 20
