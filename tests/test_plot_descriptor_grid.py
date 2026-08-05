import argparse

import numpy as np
import pytest
import yaml

from src.eval import compare_variants as cv
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


def test_figure_discloses_the_floor_when_negative_rays_are_drawn():
    """A floored ray's length is not its magnitude. Saying so on the figure
    matches this module's refusal to draw anything the reader would misread."""
    vecs = np.zeros((1, 128), dtype=np.float32)
    vecs[0, 0] = 1.0
    vecs[0, 8] = -1.0e-4
    fig = pdg.build_figure([("row", vecs, "#000000")])
    assert "minimum length" in fig.layout.title.text.lower()


def test_figure_omits_the_floor_note_when_there_are_no_negatives():
    """The caveat describes rays that exist; without them it is just noise."""
    vecs = np.abs(np.random.default_rng(0).random((2, 128))).astype(np.float32)
    fig = pdg.build_figure([("row", vecs, "#000000")])
    assert "minimum length" not in fig.layout.title.text.lower()


def test_build_figure_omits_the_negative_trace_when_all_bins_are_positive():
    vecs = np.abs(np.random.default_rng(0).random((2, 128))).astype(np.float32)
    fig = pdg.build_figure([("row", vecs, "#000000")])
    assert "negative" not in [t.name for t in fig.data]


def test_check_preprocess_accepts_the_current_config_shape():
    config = {"data": {"preprocess": {"center": False, "whiten": False,
                                      "l2_normalize": True}}}
    pdg.check_preprocess(config, "v2")  # must not raise


def test_check_preprocess_accepts_a_missing_preprocess_block():
    """Absent keys mean the dataclass defaults, which are both False."""
    pdg.check_preprocess({"data": {}}, "v2")
    pdg.check_preprocess({}, "v2")


@pytest.mark.parametrize("flag", ["center", "whiten"])
def test_check_preprocess_refuses_centering_or_whitening(flag):
    config = {"data": {"preprocess": {flag: True}}}
    with pytest.raises(ValueError, match=flag):
        pdg.check_preprocess(config, "v2")


def test_variant_row_renders_from_a_real_checkpoint(
    tmp_path, write_tiny_gated_run, monkeypatch
):
    variant, _ = write_tiny_gated_run(tmp_path, name="v2", descriptor_dim=128)
    monkeypatch.setattr(cv, "VARIANTS", (variant,))
    out = pdg.run(_args(tmp_path))
    text = out.read_text(encoding="utf-8")
    assert "real-a" in text and "v2" in text


def test_variant_row_is_seed_reproducible(tmp_path, write_tiny_gated_run, monkeypatch):
    """GatedGenerator samples gate noise in eval() too, so this needs a seed."""
    variant, _ = write_tiny_gated_run(tmp_path, name="v2", descriptor_dim=128)
    monkeypatch.setattr(cv, "VARIANTS", (variant,))
    first = pdg.variant_rows(tmp_path, num_samples=4, seed=42)[0][1]
    second = pdg.variant_rows(tmp_path, num_samples=4, seed=42)[0][1]
    assert np.array_equal(first, second)


def test_missing_checkpoint_is_skipped_and_the_poster_still_renders(
    tmp_path, capsys, monkeypatch
):
    absent = cv.Variant("ghost", "configs/sift/v2.yaml", "runs/ghost")
    monkeypatch.setattr(cv, "VARIANTS", (absent,))
    out = pdg.run(_args(tmp_path))
    assert out.exists()
    assert "ghost" in capsys.readouterr().out


def test_whitened_run_is_refused(tmp_path, write_tiny_gated_run, monkeypatch):
    variant, _ = write_tiny_gated_run(tmp_path, name="v2", descriptor_dim=128)
    run_config_path = tmp_path / "runs" / "v2" / "run_config.yaml"
    config = yaml.safe_load(run_config_path.read_text())
    config["data"]["preprocess"] = {"center": False, "whiten": True}
    run_config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(cv, "VARIANTS", (variant,))
    with pytest.raises(ValueError, match="whiten"):
        pdg.run(_args(tmp_path))


def test_variant_generating_the_wrong_width_is_refused(
    tmp_path, write_tiny_gated_run, monkeypatch
):
    variant, _ = write_tiny_gated_run(tmp_path, name="v2", descriptor_dim=8)
    monkeypatch.setattr(cv, "VARIANTS", (variant,))
    with pytest.raises(ValueError, match="128"):
        pdg.run(_args(tmp_path))


def test_write_report_survives_a_png_export_failure(tmp_path, monkeypatch, capsys):
    """kaleido needs a Chrome binary; on a headless box `export_pngs` raises.
    The HTML must still exist and the run must not raise -- mirrors
    `eda_report.run`'s try/except around the same call."""

    def _boom(*args, **kwargs):
        raise RuntimeError("Kaleido requires Google Chrome to be installed.")

    monkeypatch.setattr(pdg.eda_report, "export_pngs", _boom)
    vecs = np.abs(np.random.default_rng(0).random((2, 128))).astype(np.float32)
    fig = pdg.build_figure([("row", vecs, "#000000")])
    out = pdg.write_report(fig, tmp_path / "out", "cdn", write_png=True)
    assert out.exists()
    assert "Chrome" in capsys.readouterr().out


def test_check_finite_accepts_a_clean_array():
    pdg.check_finite(np.zeros((2, 128), dtype=np.float32), "source")  # no raise


def test_check_finite_refuses_nan():
    arr = np.zeros((2, 128), dtype=np.float32)
    arr[0, 5] = np.nan
    with pytest.raises(ValueError, match="1 non-finite"):
        pdg.check_finite(arr, "some-source")


def test_check_finite_refuses_inf_and_names_the_source():
    arr = np.zeros((2, 128), dtype=np.float32)
    arr[1, 3] = np.inf
    with pytest.raises(ValueError, match="some-source"):
        pdg.check_finite(arr, "some-source")


def test_run_refuses_real_data_containing_nan(tmp_path):
    # A distinct filename from _write_real's default: _args() below calls
    # _write_real(tmp_path) again to build its base namespace, which would
    # otherwise overwrite this file's NaN back to clean data at the same path.
    x = np.random.default_rng(0).random((64, 128)).astype(np.float32)
    x[x < 0.8] = 0.0
    x[:, 0] = 1.0
    x[0, 1] = np.nan
    path = tmp_path / "real_with_nan.npy"
    np.save(path, x)
    with pytest.raises(ValueError, match="non-finite"):
        pdg.run(_args(tmp_path, real_path=str(path)))


def test_variant_row_with_nan_output_is_refused(
    tmp_path, write_tiny_gated_run, monkeypatch
):
    """A generator that produces a non-finite value must not invent a
    spurious red 'negative' trace for the affected variant."""
    variant, _ = write_tiny_gated_run(tmp_path, name="v2", descriptor_dim=128)

    def _nan_sampler(generator, num_samples, latent_dim, batch_size, device):
        out = np.zeros((num_samples, 128), dtype=np.float32)
        out[0, 0] = np.nan
        return out

    monkeypatch.setattr(cv, "VARIANTS", (variant,))
    monkeypatch.setattr(pdg, "sample_generator", _nan_sampler)
    with pytest.raises(ValueError, match="non-finite"):
        pdg.run(_args(tmp_path))


def test_variant_colour_is_keyed_by_identity_not_resolved_position(
    tmp_path, write_tiny_gated_run, monkeypatch
):
    """v1 must keep its colour whether or not v0's checkpoint is present --
    the same machine-dependence `cv.variant_seed` guards against for
    sampling."""
    v0, _ = write_tiny_gated_run(tmp_path, name="v0", descriptor_dim=128)
    v1, _ = write_tiny_gated_run(tmp_path, name="v1", descriptor_dim=128)
    monkeypatch.setattr(cv, "VARIANTS", (v0, v1))

    both = pdg.variant_rows(tmp_path, num_samples=4, seed=42)
    v1_color_with_v0_present = next(c for n, _, c in both if n == "v1")

    # Remove v0's checkpoint so it is skipped, and confirm v1's colour holds.
    import shutil

    shutil.rmtree(tmp_path / "runs" / "v0")
    only_v1 = pdg.variant_rows(tmp_path, num_samples=4, seed=42)
    v1_color_alone = next(c for n, _, c in only_v1 if n == "v1")

    assert v1_color_with_v0_present == v1_color_alone


def test_variant_colour_fallback_for_a_name_absent_from_cv_variants():
    assert pdg.variant_color("mystery") == pdg.VARIANT_COLORS[
        len(cv.VARIANTS) % len(pdg.VARIANT_COLORS)
    ]


def test_ray_scale_is_shared_across_rows_not_computed_per_row():
    """A regression to per-glyph normalisation would pass the rest of the
    suite while silently destroying the real-vs-generated comparison the
    figure exists to make. Every bin in `big` is identical so the shared
    scale gives every ray in a row the same length; if `small` were
    normalised on its own (per-row) rather than against `big`'s scale, its
    rays would come out the same length as `big`'s instead of 10x shorter."""
    big = np.full((1, 128), 0.05, dtype=np.float32)
    small = (big * 0.1).astype(np.float32)

    fig = pdg.build_figure([("big", big, "#000000"), ("small", small, "#111111")])
    big_trace = next(t for t in fig.data if t.name == "big")
    small_trace = next(t for t in fig.data if t.name == "small")

    def _ray_length(trace):
        # First segment: (centre_x, centre_y), (tip_x, tip_y), NaN.
        dx = trace.x[1] - trace.x[0]
        dy = trace.y[1] - trace.y[0]
        return float(np.hypot(dx, dy))

    ratio = _ray_length(big_trace) / _ray_length(small_trace)
    assert ratio == pytest.approx(10.0, rel=1e-6)
