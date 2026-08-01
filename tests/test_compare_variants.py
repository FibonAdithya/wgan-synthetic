from pathlib import Path

from src.eval import compare_variants as cv


def test_variants_are_the_four_named_ones():
    assert [v.name for v in cv.VARIANTS] == ["v0", "v1", "v1_5", "v2"]


def test_every_variant_config_exists():
    for v in cv.VARIANTS:
        assert Path(v.config_path).exists(), f"missing config for {v.name}"


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
