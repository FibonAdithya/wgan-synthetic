import numpy as np
from tests.conftest import make_args

from src.eval import eda_report
from src.eval.eda import glyphs


def _sparse_sift_like(rows, seed, dim=128):
    """Sparse non-negative rows, standing in for real SIFT descriptors."""
    rng = np.random.default_rng(seed)
    x = rng.random((rows, dim)).astype(np.float32)
    x[x < 0.8] = 0.0
    x[:, 0] = 1.0  # no all-zero row
    return x


def test_glyph_section_is_included_for_128_dimensional_data(tmp_path):
    real = _sparse_sift_like(200, seed=0)
    synth = {"v0": _sparse_sift_like(200, seed=1)}

    html = eda_report.run(make_args(tmp_path, real, synth)).read_text()

    assert glyphs.GLYPH_SECTION_TITLE in html


def test_glyph_section_draws_two_real_rows_as_the_variation_baseline(tmp_path):
    """One real row gives no sense of how much two real descriptors differ,
    which is what a variant row has to be judged against."""
    real = _sparse_sift_like(200, seed=0)
    synth = {"v0": _sparse_sift_like(200, seed=1)}

    html = eda_report.run(make_args(tmp_path, real, synth)).read_text()

    assert "real-a" in html and "real-b" in html


def test_glyph_section_is_skipped_for_other_dimensions(tmp_path):
    """The (cell, orientation bin) mapping only exists at 128 dimensions, and
    the rest of the report is dimension-agnostic, so the panel drops out
    rather than taking the whole report down with it."""
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 8)).astype(np.float32)
    synth = {"v0": rng.normal(size=(200, 8)).astype(np.float32)}

    html = eda_report.run(make_args(tmp_path, real, synth)).read_text()

    assert glyphs.GLYPH_SECTION_TITLE not in html


def test_glyph_section_is_skipped_when_a_series_is_too_small(tmp_path):
    """Two disjoint real rows need 2 * glyph_samples vectors. Dense rather
    than sparse here: with this few rows a sparse column can end up constant,
    and the correlation panel warns on the zero standard deviation."""
    rng = np.random.default_rng(3)
    real = rng.random((6, 128)).astype(np.float32)
    synth = {"v0": rng.random((6, 128)).astype(np.float32)}

    args = make_args(tmp_path, real, synth)
    args.knn = 2
    args.glyph_samples = 8
    html = eda_report.run(args).read_text()

    assert glyphs.GLYPH_SECTION_TITLE not in html


def test_glyph_samples_zero_disables_the_section(tmp_path):
    real = _sparse_sift_like(200, seed=0)
    synth = {"v0": _sparse_sift_like(200, seed=1)}

    args = make_args(tmp_path, real, synth)
    args.glyph_samples = 0
    html = eda_report.run(args).read_text()

    assert glyphs.GLYPH_SECTION_TITLE not in html
