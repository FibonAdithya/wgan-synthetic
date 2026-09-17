"""The SIFT v4 retrain and real noise-floor jobs.

Both run for hours on a shared card, so pin the parts that would waste the run
if they drifted: the draws must be disjoint, and the job must train the
unchanged 100k config against the hash-checked corpus.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SHA = "4921953796bc066d3a4be1c6b26e32a27b8931080f21afe30824a63e65cb768d"


def _floor_module():
    spec = importlib.util.spec_from_file_location(
        "sift_real_noise_floor", ROOT / "scripts" / "sift_real_noise_floor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_draws_are_disjoint_full_size_and_seeded():
    """Catches overlapping slices, a short last draw, and an unseeded permutation."""
    floor = _floor_module()
    x = np.arange(1000, dtype=np.float32).reshape(-1, 1)
    first = floor.draws(x, count=4, n=250, seed=42)
    assert [d.shape[0] for d in first] == [250] * 4
    rows = np.concatenate([d[:, 0] for d in first])
    assert np.unique(rows).size == 1000
    again = floor.draws(x, count=4, n=250, seed=42)
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))


def test_draws_refuse_to_overlap_when_the_corpus_is_too_small():
    """Catches silently wrapping or truncating instead of raising."""
    floor = _floor_module()
    with pytest.raises(ValueError):
        floor.draws(np.zeros((999, 1), dtype=np.float32), count=4, n=250, seed=42)


def test_retrain_job_runs_the_unchanged_100k_config_on_the_checked_corpus():
    """Catches a config swap, a lost hash check, or a --vram-mb that breaks the lock."""
    script = (ROOT / "scripts" / "sift_v4_x100k_job.sh").read_text()
    assert "--config configs/sift/v4_sift1m_x100k.yaml" in script
    assert f"EXPECT={SHA}" in script
    assert "sha256sum -c -" in script
    assert "checkpoint_step_100000.pt" in script
    assert "ln -sfn /workspace/sift-v4 runs/sift" in script
    assert "--vram-mb" not in script.split("Do NOT pass --vram-mb")[1]


def test_data_job_checks_the_corpus_hash_before_measuring():
    """Catches measuring a corpus that is not the one earlier results used."""
    script = (ROOT / "scripts" / "sift_data_job.sh").read_text()
    assert f"EXPECT={SHA}" in script
    assert script.index("sha256sum -c -") < script.index("sift_real_noise_floor.py")
