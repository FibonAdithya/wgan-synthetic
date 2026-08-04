import argparse

import numpy as np
import pytest

from src.eval import plot_descriptor_grid as pdg


def _write_real(tmp_path, n=64, dim=128, seed=0):
    """Sparse non-negative vectors, standing in for real SIFT descriptors."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, dim)).astype(np.float32)
    x[x < 0.8] = 0.0
    x[:, 0] = 1.0  # guarantee no all-zero row
    # Filename varies with dim so a dim-64 fixture and the dim-128 default
    # written inside _args() below don't collide and overwrite each other.
    path = tmp_path / f"real_{dim}.npy"
    np.save(path, x)
    return path


def _args(tmp_path, **overrides):
    base = dict(
        real_path=str(_write_real(tmp_path)),
        real_format="auto",
        output_dir=str(tmp_path / "out"),
        root=str(tmp_path),
        num_samples=4,
        seed=42,
        no_png=True,
        plotlyjs="cdn",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_l2_normalize_gives_unit_rows():
    x = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    out = pdg.l2_normalize(x)
    assert np.linalg.norm(out, axis=1) == pytest.approx([1.0, 1.0])


def test_l2_normalize_leaves_a_zero_row_finite():
    out = pdg.l2_normalize(np.zeros((1, 4), dtype=np.float32))
    assert np.all(np.isfinite(out))


def test_pick_real_rows_returns_two_disjoint_rows():
    real = np.arange(40 * 128, dtype=np.float32).reshape(40, 128)
    row_a, row_b = pdg.pick_real_rows(real, num_samples=5, seed=1)
    assert row_a.shape == (5, 128) and row_b.shape == (5, 128)
    seen = {tuple(v) for v in row_a} | {tuple(v) for v in row_b}
    assert len(seen) == 10


def test_pick_real_rows_is_seed_reproducible():
    real = np.arange(40 * 128, dtype=np.float32).reshape(40, 128)
    first = pdg.pick_real_rows(real, 5, seed=7)[0]
    second = pdg.pick_real_rows(real, 5, seed=7)[0]
    assert np.array_equal(first, second)


def test_pick_real_rows_rejects_too_few_vectors():
    real = np.zeros((9, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="at least 10"):
        pdg.pick_real_rows(real, num_samples=5, seed=1)


def test_run_writes_html_with_the_two_real_rows(tmp_path):
    out = pdg.run(_args(tmp_path))
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "real-a" in text and "real-b" in text


def test_run_rejects_a_non_128_dimensional_dataset(tmp_path):
    path = _write_real(tmp_path, dim=64)
    with pytest.raises(ValueError, match="128"):
        pdg.run(_args(tmp_path, real_path=str(path)))


def test_run_rejects_a_missing_real_path(tmp_path):
    """Nothing to compare against, so this is a hard error, not a skip."""
    with pytest.raises((FileNotFoundError, ValueError)):
        pdg.run(_args(tmp_path, real_path=str(tmp_path / "absent.npy")))


def test_build_figure_puts_negative_rays_in_their_own_trace():
    vecs = np.zeros((1, 128), dtype=np.float32)
    vecs[0, 0] = 1.0
    vecs[0, 8] = -1.0
    fig = pdg.build_figure([("row", vecs, "#000000")])
    names = [t.name for t in fig.data]
    assert "negative" in names
    negative = next(t for t in fig.data if t.name == "negative")
    assert len(negative.x) == 3  # one ray: centre, tip, NaN


def test_build_figure_omits_the_negative_trace_when_all_bins_are_positive():
    vecs = np.abs(np.random.default_rng(0).random((2, 128))).astype(np.float32)
    fig = pdg.build_figure([("row", vecs, "#000000")])
    assert "negative" not in [t.name for t in fig.data]
