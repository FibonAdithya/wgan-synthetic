import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.eval import eda_report


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
        max_vectors=200,
        num_pairs=500,
        knn=3,
        ann_k=eda_report.ANN_K_DEFAULT,
        ann_hub_k=eda_report.ANN_HUB_K_DEFAULT,
        ann_max_rows=eda_report.ANN_MAX_ROWS_DEFAULT,
        ivf_nlist=eda_report.IVF_NLIST_DEFAULT,
        bins=16,
        top_divergent=4,
        seed=42,
        no_png=True,
        plotlyjs="cdn",
    )


def test_run_returns_written_report_path(tmp_path):
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 8)).astype(np.float32)
    synth = {"v0": rng.normal(size=(200, 8)).astype(np.float32)}

    out = eda_report.run(make_args(tmp_path, real, synth))

    assert isinstance(out, Path)
    assert out.exists()
    assert out.suffix == ".html"
    assert "v0" in out.read_text()


def test_run_accepts_several_synthetic_sets(tmp_path):
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 8)).astype(np.float32)
    synth = {
        "v0": rng.normal(size=(200, 8)).astype(np.float32),
        "v1": rng.normal(size=(200, 8)).astype(np.float32),
        "v2": rng.normal(size=(200, 8)).astype(np.float32),
    }

    html = eda_report.run(make_args(tmp_path, real, synth)).read_text()

    for label in ("v0", "v1", "v2"):
        assert label in html


def _write_set(path, rows, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    np.save(path, x)


def test_report_writes_html_and_summary_with_ann_sections(tmp_path, monkeypatch):
    real = tmp_path / "real.npy"
    fake = tmp_path / "fake.npy"
    _write_set(real, 400, seed=0)
    _write_set(fake, 400, seed=1)
    out = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eda_report",
            "--real-path", str(real),
            "--synthetic-path", f"fake={fake}",
            "--output-dir", str(out),
            "--ann-max-rows", "300",
            "--ann-k", "20",
            "--ann-hub-k", "5",
            "--ivf-nlist", "8",
            "--max-vectors", "400",
            "--num-pairs", "2000",
            "--no-png",
            "--plotlyjs", "cdn",
        ],
    )
    eda_report.main()

    html = (out / "eda_report.html").read_text(encoding="utf-8")
    assert "Local intrinsic dimensionality" in html
    assert "Hubness" in html
    assert "IVF cell balance" in html

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["ann_settings"]["k"] == 20
    for row in summary["stats"]:
        assert row["lid_median"] > 0
        assert "hubness_skew" in row
        assert "ivf_gini" in row
