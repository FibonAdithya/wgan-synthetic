"""Smoke tests for the matplotlib distance-CDF plot.

The review calls this module low blast radius -- visual output, no training or
sampling depends on it. So these tests are deliberately smoke-level: they check
that the script runs end to end on tiny input, writes the file it says it
wrote, and fails loudly rather than emitting a corrupt image. They do not
inspect pixels.

The module exposes only `parse_args` + `main`, with no `run(args)` seam like
`plot_descriptor_grid` has, so the CLI surface is exercised the only way it can
be from a test: by driving `sys.argv` and calling `main()`.

Everything below skips on a clean install, and that is the point rather than an
oversight. `plot_distance_cdf` imports matplotlib, which is deliberately not in
`requirements.txt` (see the "Do not add dependencies" constraint in
docs/superpowers/plans/2026-08-04-sift-descriptor-glyph-grid.md). So the module
cannot be imported from a clean checkout at all, and adding the dependency here
to make these tests run would quietly overturn a decision this file has no
standing to overturn. The skip keeps that visible in the pytest summary. If the
module is retired in favour of `plot_distance_cdf_pillow`, which draws the same
figure without matplotlib, delete this file with it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "matplotlib",
    reason="matplotlib is deliberately absent from requirements.txt; "
    "src/eval/plot_distance_cdf.py cannot be imported without it.",
)

from src.eval import plot_distance_cdf as pdc  # noqa: E402


@pytest.fixture(autouse=True)
def _headless_backend():
    """`main` builds a figure at import-time-chosen backend; on a CI box with
    no display a GUI backend would either fail or block. Agg writes files and
    needs nothing else."""
    pdc.plt.switch_backend("Agg")
    yield
    pdc.plt.close("all")


def _write_npy(tmp_path: Path, name: str, n: int = 60, dim: int = 16, seed: int = 0):
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
        "--num-queries": "8",
        "--num-targets": "20",
        "--seed": "42",
        "--output-path": str(tmp_path / "out" / "cdf.png"),
    }
    base.update(overrides)
    return ["plot_distance_cdf", *[s for pair in base.items() for s in pair]]


def test_main_writes_a_non_empty_png_at_the_requested_path(tmp_path: Path, monkeypatch):
    out = tmp_path / "out" / "cdf.png"
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    pdc.main()

    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 0


def test_main_creates_the_output_directory_it_was_pointed_at(
    tmp_path: Path, monkeypatch
):
    """`--output-path` is the only way to say where the figure goes, so a path
    into a directory that does not exist yet must not be an error."""
    out = tmp_path / "deeply" / "nested" / "cdf.png"
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, **{"--output-path": str(out)}))

    pdc.main()

    assert out.exists()


def test_main_announces_where_it_wrote_the_figure(tmp_path: Path, monkeypatch, capsys):
    out = tmp_path / "out" / "cdf.png"
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    pdc.main()

    assert str(out) in capsys.readouterr().out


def test_main_refuses_a_missing_real_path(tmp_path: Path, monkeypatch):
    argv = _argv(tmp_path, **{"--real-path": str(tmp_path / "absent.npy")})
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError):
        pdc.main()


def test_main_refuses_a_missing_synthetic_path(tmp_path: Path, monkeypatch):
    argv = _argv(tmp_path, **{"--synthetic-path": str(tmp_path / "absent.npy")})
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError):
        pdc.main()


def test_main_refuses_a_real_file_whose_extension_it_cannot_read(
    tmp_path: Path, monkeypatch
):
    bogus = tmp_path / "real.txt"
    bogus.write_text("not descriptors")
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, **{"--real-path": str(bogus)}))
    with pytest.raises(ValueError, match="extension"):
        pdc.main()


def test_main_refuses_a_one_dimensional_synthetic_array(tmp_path: Path, monkeypatch):
    """A 1-D array is a plausible mistake -- a single flattened descriptor. It
    must raise rather than silently produce a figure of nothing, and no output
    file may be left behind for a reader to trust."""
    flat = tmp_path / "flat.npy"
    np.save(flat, np.random.default_rng(3).random(16).astype(np.float32))
    out = tmp_path / "out" / "cdf.png"
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, **{"--synthetic-path": str(flat)}))

    with pytest.raises(ValueError):
        pdc.main()

    assert not out.exists()


def test_l2_normalize_gives_unit_rows():
    x = np.random.default_rng(4).random((5, 16)).astype(np.float32) + 0.1
    assert np.linalg.norm(pdc.l2_normalize(x), axis=1) == pytest.approx(1.0, rel=1e-6)


def test_l2_normalize_leaves_an_all_zero_row_finite():
    """Dividing by the raw norm would give NaN and poison every quantile."""
    x = np.zeros((2, 16), dtype=np.float32)
    assert np.all(np.isfinite(pdc.l2_normalize(x)))


def test_sampled_indices_returns_every_row_when_k_is_not_smaller_than_n():
    rng = np.random.default_rng(5)
    assert np.array_equal(pdc.sampled_indices(4, 10, rng), np.arange(4))


def test_sampled_indices_draws_k_distinct_rows():
    idx = pdc.sampled_indices(50, 10, np.random.default_rng(6))
    assert idx.shape == (10,)
    assert len(set(idx.tolist())) == 10


def test_quantile_curves_are_ordered_and_the_cdf_axis_spans_zero_to_one():
    x = pdc.l2_normalize(np.random.default_rng(7).random((40, 16)).astype(np.float32))
    y, q10, q50, q90 = pdc.query_cdf_quantiles(
        x, num_queries=8, num_targets=20, rng=np.random.default_rng(8)
    )
    assert y[0] == 0.0 and y[-1] < 1.0
    assert np.all(q10 <= q50) and np.all(q50 <= q90)
    assert q10.shape == q50.shape == q90.shape == y.shape


def test_a_query_is_not_counted_as_its_own_nearest_neighbour():
    """With queries and targets both spanning the whole array every query sits
    in the target set, and a retained self-distance of 0 would drag the whole
    left edge of the CDF down. The curve then reports a density the data does
    not have, which is the one thing this figure exists to show."""
    x = pdc.l2_normalize(np.random.default_rng(9).random((30, 16)).astype(np.float32))
    y, q10, _, _ = pdc.query_cdf_quantiles(
        x, num_queries=30, num_targets=30, rng=np.random.default_rng(10)
    )
    assert y.shape == (29,)  # one target dropped per query
    assert q10[0] > 0.0
