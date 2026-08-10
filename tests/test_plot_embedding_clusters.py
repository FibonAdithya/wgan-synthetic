"""Smoke tests for the t-SNE/UMAP embedding scatter plots.

Low blast radius -- these are pictures, nothing downstream consumes them. So
the bar is: it runs end to end on tiny input, both PNGs land where the printed
paths say they do, and obviously bad input raises instead of writing a corrupt
or empty image. t-SNE output itself is not asserted on beyond its shape and
finiteness; the coordinates are not stable across BLAS builds and pinning them
would produce a test that fails for reasons unrelated to this module.

`main` is the only entry point (no `run(args)` seam), so the CLI is exercised
by driving `sys.argv`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval import plot_embedding_clusters as pec


def _write_npy(tmp_path: Path, name: str, n: int = 40, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, dim)).astype(np.float32)
    x[:, 0] = 1.0  # no all-zero row, so l2_normalize is well defined
    path = tmp_path / name
    np.save(path, x)
    return path


def _argv(tmp_path: Path, **overrides) -> list[str]:
    base = {
        "--real-path": str(_write_npy(tmp_path, "real.npy", seed=1)),
        "--synthetic-path": str(_write_npy(tmp_path, "synth.npy", seed=2)),
        "--method": "tsne",
        "--sample-size": "30",
        "--seed": "42",
        "--output-dir": str(tmp_path / "out"),
    }
    base.update(overrides)
    return ["plot_embedding_clusters", *[s for pair in base.items() for s in pair]]


def test_main_writes_both_scatter_pngs_named_after_the_method(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    pec.main()

    real = tmp_path / "out" / "tsne_sift_real.png"
    synth = tmp_path / "out" / "tsne_synthetic.png"
    for path in (real, synth):
        assert path.exists() and path.suffix == ".png"
        assert path.stat().st_size > 0
        with Image.open(path) as img:
            assert img.format == "PNG"
            assert img.size == (1200, 900)


def test_main_prints_the_paths_it_wrote(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    pec.main()

    out = capsys.readouterr().out
    assert str(tmp_path / "out" / "tsne_sift_real.png") in out
    assert str(tmp_path / "out" / "tsne_synthetic.png") in out


def test_main_refuses_a_missing_real_path(tmp_path: Path, monkeypatch):
    argv = _argv(tmp_path, **{"--real-path": str(tmp_path / "absent.npy")})
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError):
        pec.main()


def test_main_refuses_a_synthetic_file_whose_extension_it_cannot_read(
    tmp_path: Path, monkeypatch
):
    bogus = tmp_path / "synth.txt"
    bogus.write_text("not descriptors")
    argv = _argv(tmp_path, **{"--synthetic-path": str(bogus)})
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="extension"):
        pec.main()


def test_main_refuses_a_one_dimensional_input_without_leaving_a_png_behind(
    tmp_path: Path, monkeypatch
):
    flat = tmp_path / "flat.npy"
    np.save(flat, np.random.default_rng(3).random(16).astype(np.float32))
    argv = _argv(tmp_path, **{"--real-path": str(flat)})
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError):
        pec.main()

    assert not (tmp_path / "out" / "tsne_sift_real.png").exists()


def test_sample_rows_returns_the_array_untouched_when_it_is_already_small_enough():
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert np.array_equal(pec.sample_rows(x, 10, np.random.default_rng(4)), x)


def test_sample_rows_draws_the_requested_number_of_distinct_rows():
    x = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
    sampled = pec.sample_rows(x, 10, np.random.default_rng(5))
    assert sampled.shape == (10, 3)
    assert len({tuple(row) for row in sampled}) == 10


def test_compute_embedding_returns_two_finite_coordinates_per_row():
    x = np.random.default_rng(6).random((30, 16)).astype(np.float32)
    emb = pec.compute_embedding(x, method="tsne", seed=0, perplexity=30.0)
    assert emb.shape == (30, 2)
    assert emb.dtype == np.float32
    assert np.all(np.isfinite(emb))


def test_compute_embedding_clamps_perplexity_below_the_sample_count():
    """t-SNE rejects a perplexity that is not smaller than n_samples. The
    default is 30, so any run on fewer than ~90 rows depends on this clamp --
    without it the tiny-input path raises instead of drawing."""
    x = np.random.default_rng(7).random((20, 8)).astype(np.float32)
    emb = pec.compute_embedding(x, method="tsne", seed=0, perplexity=30.0)
    assert emb.shape == (20, 2)


def test_compute_embedding_rejects_an_unknown_method():
    x = np.random.default_rng(8).random((20, 8)).astype(np.float32)
    with pytest.raises(ValueError, match="Unsupported method"):
        pec.compute_embedding(x, method="pca", seed=0, perplexity=30.0)


def test_umap_is_refused_with_an_install_hint_when_the_package_is_absent(monkeypatch):
    """umap-learn is not in requirements.txt, so `--method umap` on a clean
    install must say how to fix that rather than surfacing a bare ImportError
    from a transitive import."""
    # `None` in sys.modules is the documented way to make an import fail, and
    # it keeps the test honest on a box where umap-learn happens to be present.
    monkeypatch.setitem(sys.modules, "umap", None)
    monkeypatch.setitem(sys.modules, "umap.umap_", None)
    x = np.random.default_rng(9).random((20, 8)).astype(np.float32)
    with pytest.raises(RuntimeError, match="umap-learn"):
        pec.compute_embedding(x, method="umap", seed=0, perplexity=30.0)


def test_draw_scatter_writes_a_png_into_a_directory_that_does_not_exist_yet(
    tmp_path: Path,
):
    points = np.random.default_rng(10).random((25, 2)).astype(np.float32)
    out = tmp_path / "nested" / "scatter.png"

    pec.draw_scatter(points, "title", out, (10, 20, 30))

    with Image.open(out) as img:
        assert img.size == (1200, 900)


def test_draw_scatter_survives_a_cloud_with_no_spread(tmp_path: Path):
    """A collapsed generator embeds every point on top of every other one. The
    span floor is what keeps that from dividing by zero and handing Pillow NaN
    coordinates, and a collapsed run is exactly when someone plots this."""
    points = np.full((10, 2), 3.0, dtype=np.float32)
    out = tmp_path / "degenerate.png"

    pec.draw_scatter(points, "collapsed", out, (10, 20, 30))

    assert out.stat().st_size > 0


# The normalisation tests that lived here moved to
# tests/test_normalisation_is_shared.py when this module stopped carrying its
# own copy of the rule.
