import argparse
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
