import argparse
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.eval import compare_variants, eda_report
from src.deep.report import DEEP_VARIANTS, build_report_args


def test_deep_variants_cover_the_whole_ladder():
    assert [v.name for v in DEEP_VARIANTS] == ["v0", "v1", "v2"]


def test_deep_variant_configs_all_exist():
    for variant in DEEP_VARIANTS:
        assert Path(variant.config_path).exists(), variant.config_path


def test_deep_variants_do_not_collide_with_sift_run_dirs():
    """A deep run must never be read out of a SIFT run directory."""
    sift_dirs = {v.run_dir for v in compare_variants.VARIANTS}
    assert not sift_dirs & {v.run_dir for v in DEEP_VARIANTS}


def test_report_args_match_eda_report_fields():
    """Field-for-field parity with eda_report.parse_args is load-bearing.

    If eda_report gains a required argument and this Namespace is not updated,
    sampling hundreds of thousands of vectors succeeds before the mismatch
    surfaces as a runtime AttributeError. Mirrors the SIFT test of the same
    name in tests/test_compare_variants.py.
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
    produced = vars(build_report_args(args, ["v0=samples/v0.npy"]))

    source = inspect.getsource(eda_report.parse_args)
    expected = {
        line.split('"')[1].lstrip("-").replace("-", "_")
        for line in source.splitlines()
        if "add_argument(" in line and '"--' in line
    }
    assert expected - set(produced) == set()


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
