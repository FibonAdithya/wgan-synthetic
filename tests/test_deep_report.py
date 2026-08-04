import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

from src.eval import compare_variants, eda_report
from src.eval.compare_variants import Variant
from src.deep.report import DEEP_VARIANTS, build_report_args, filter_missing_run_metadata


def test_deep_variants_cover_the_whole_ladder():
    assert [v.name for v in DEEP_VARIANTS] == ["v0", "v1", "v2"]


def test_deep_variant_configs_all_exist():
    for variant in DEEP_VARIANTS:
        assert Path(variant.config_path).exists(), variant.config_path


def test_deep_variants_do_not_collide_with_sift_run_dirs():
    """A deep run must never be read out of a SIFT run directory."""
    sift_dirs = {v.run_dir for v in compare_variants.VARIANTS}
    assert not sift_dirs & {v.run_dir for v in DEEP_VARIANTS}


def test_report_args_match_eda_report_fields(monkeypatch, tmp_path):
    """Field-for-field parity with eda_report.parse_args is load-bearing.

    If eda_report gains a required argument and this Namespace is not updated,
    sampling hundreds of thousands of vectors succeeds before the mismatch
    surfaces as a runtime AttributeError. Rather than scraping parse_args's
    source (fragile against multi-line add_argument calls), invoke the real
    parse_args and compare its actual field set against build_report_args's
    output, both ways. Mirrors the SIFT test of the same name in
    tests/test_compare_variants.py.
    """
    args = argparse.Namespace(
        real_path="real.npy",
        real_format="npy",
        output_dir="out",
        max_vectors=100,
        num_pairs=100,
        knn=5,
        ann_k=eda_report.ANN_K_DEFAULT,
        ann_hub_k=eda_report.ANN_HUB_K_DEFAULT,
        ann_max_rows=eda_report.ANN_MAX_ROWS_DEFAULT,
        ivf_nlist=eda_report.IVF_NLIST_DEFAULT,
        bins=80,
        top_divergent=16,
        seed=42,
        no_png=True,
        plotlyjs="inline",
    )
    produced = build_report_args(args, ["v0=samples/v0.npy"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eda_report.py",
            "--real-path",
            "real.npy",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    eda_args = eda_report.parse_args()

    assert set(vars(produced)) == set(vars(eda_args))


def test_filter_missing_run_metadata_skips_variants_without_it(tmp_path: Path):
    """Pins finding 4: resolve_variants only checks best_generator.pt and
    run_config.yaml, but src/deep/sample.py also needs run_metadata.json to
    invert the fitted preprocess transform. A run dir with the first two but
    not the third must be moved from `found` to `skipped` before any
    sampling happens, not fail mid-loop after earlier variants already ran.
    """
    has_metadata = Variant("has_metadata", "configs/x.yaml", "run_a")
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a" / "run_metadata.json").write_text("{}", encoding="utf-8")

    missing_metadata = Variant("missing_metadata", "configs/y.yaml", "run_b")
    (tmp_path / "run_b").mkdir()

    found, skipped = filter_missing_run_metadata(
        [has_metadata, missing_metadata], tmp_path
    )

    assert found == [has_metadata]
    assert len(skipped) == 1
    assert skipped[0][0] is missing_metadata
    assert "run_metadata.json" in skipped[0][1]


def test_filter_missing_run_metadata_is_a_no_op_when_all_have_it(tmp_path: Path):
    variant = Variant("v", "configs/x.yaml", "run_a")
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a" / "run_metadata.json").write_text("{}", encoding="utf-8")

    found, skipped = filter_missing_run_metadata([variant], tmp_path)

    assert found == [variant]
    assert skipped == []


def test_report_args_pass_through_the_ann_settings():
    args = argparse.Namespace(
        real_path="real.npy", real_format="npy", output_dir="out",
        max_vectors=100, num_pairs=100, knn=5,
        ann_k=17, ann_hub_k=3, ann_max_rows=999, ivf_nlist=8,
        bins=80, top_divergent=16, seed=42, no_png=True, plotlyjs="inline",
    )
    produced = build_report_args(args, ["v0=samples/v0.npy"])
    assert produced.ann_k == 17
    assert produced.ann_hub_k == 3
    assert produced.ann_max_rows == 999
    assert produced.ivf_nlist == 8
    assert produced.synthetic_path == ["v0=samples/v0.npy"]
