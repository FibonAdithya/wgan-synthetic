"""Smoke tests for the Pillow distance-CDF plot.

Same figure as `plot_distance_cdf`, drawn by hand into a bitmap instead of
through matplotlib. Because it does its own axis mapping there is one thing
worth checking beyond "it ran": that `data_to_pixels` puts the data inside the
plot box. Everything else here is smoke-level -- the file exists, opens as an
image, and bad input raises instead of producing a misleading picture.

Like its matplotlib sibling this module has no `run(args)` seam, so the CLI is
driven through `sys.argv` and `main()`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from src.eval import plot_distance_cdf_pillow as pdcp


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
    return ["plot_distance_cdf_pillow", *[s for pair in base.items() for s in pair]]


def test_main_writes_a_non_empty_png_that_opens_as_an_image(
    tmp_path: Path, monkeypatch
):
    out = tmp_path / "out" / "cdf.png"
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    pdcp.main()

    assert out.exists() and out.suffix == ".png"
    assert out.stat().st_size > 0
    with Image.open(out) as img:
        assert img.format == "PNG"
        assert img.size == (1200, 800)


def test_main_creates_the_output_directory_it_was_pointed_at(
    tmp_path: Path, monkeypatch
):
    out = tmp_path / "deeply" / "nested" / "cdf.png"
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, **{"--output-path": str(out)}))

    pdcp.main()

    assert out.exists()


def test_main_accepts_the_optional_label_and_caption_flags(tmp_path: Path, monkeypatch):
    """These three flags only feed text into the subtitle, but they are part of
    the CLI surface and a typo in the drawing code would raise here."""
    out = tmp_path / "labelled.png"
    argv = _argv(
        tmp_path,
        **{
            "--output-path": str(out),
            "--config-label": "v2",
            "--caption": "smoke run",
        },
    )
    monkeypatch.setattr(sys, "argv", argv)

    pdcp.main()

    assert out.stat().st_size > 0


def test_main_reports_the_distance_range_it_plotted(
    tmp_path: Path, monkeypatch, capsys
):
    """The x axis carries no tick labels, so this printed range is the only
    thing that tells a reader what the horizontal extent means."""
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    pdcp.main()

    assert "x_range=[" in capsys.readouterr().out


def test_main_refuses_a_missing_real_path(tmp_path: Path, monkeypatch):
    argv = _argv(tmp_path, **{"--real-path": str(tmp_path / "absent.npy")})
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError):
        pdcp.main()


def test_main_refuses_a_missing_synthetic_path(tmp_path: Path, monkeypatch):
    argv = _argv(tmp_path, **{"--synthetic-path": str(tmp_path / "absent.npy")})
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError):
        pdcp.main()


def test_main_refuses_a_one_dimensional_synthetic_array(tmp_path: Path, monkeypatch):
    flat = tmp_path / "flat.npy"
    np.save(flat, np.random.default_rng(3).random(16).astype(np.float32))
    out = tmp_path / "out" / "cdf.png"
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, **{"--synthetic-path": str(flat)}))

    with pytest.raises(ValueError):
        pdcp.main()

    assert not out.exists()


def test_quantile_curves_are_ordered_and_the_cdf_axis_starts_at_zero():
    x = pdcp.l2_normalize(np.random.default_rng(4).random((40, 16)).astype(np.float32))
    y, q10, q50, q90 = pdcp.query_cdf_quantiles(
        x, num_queries=8, num_targets=20, rng=np.random.default_rng(5)
    )
    assert y[0] == 0.0 and y[-1] < 1.0
    assert np.all(q10 <= q50) and np.all(q50 <= q90)


def test_data_to_pixels_maps_the_data_range_onto_the_plot_box():
    """x_min lands on the left margin and x_max on the right one; CDF 0 sits on
    the axis and CDF 1 at the top. Getting this backwards would draw a CDF that
    falls instead of rises, which reads as a plausible figure."""
    px = pdcp.data_to_pixels(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        x_min=0.0,
        x_max=1.0,
        width=1200,
        height=800,
        margin_left=100,
        margin_right=60,
        margin_top=60,
        margin_bottom=90,
    )
    assert px[0].tolist() == [100.0, 710.0]
    assert px[1].tolist() == [1140.0, 60.0]


def test_data_to_pixels_survives_a_degenerate_x_range():
    """Every distance identical is unlikely but the guard exists, and a
    division by zero here would write NaN coordinates into Pillow."""
    px = pdcp.data_to_pixels(
        np.array([2.0, 2.0]),
        np.array([0.0, 1.0]),
        x_min=2.0,
        x_max=2.0,
        width=1200,
        height=800,
        margin_left=100,
        margin_right=60,
        margin_top=60,
        margin_bottom=90,
    )
    assert np.all(np.isfinite(px))


def test_draw_curve_ignores_a_single_point(tmp_path: Path):
    """Pillow's `line` raises on a one-point sequence, so the guard is load
    bearing rather than defensive."""
    img = Image.new("RGB", (20, 20), (255, 255, 255))
    pdcp.draw_curve(ImageDraw.Draw(img), np.array([[1.0, 1.0]]), (0, 0, 0))
    assert img.getpixel((1, 1)) == (255, 255, 255)


def test_generator_model_label_is_empty_without_a_config_path():
    assert pdcp.generator_model_label_from_config("") == ""


def test_generator_model_label_is_empty_when_the_config_is_absent(tmp_path: Path):
    """A stale path in a shell script must degrade to an unlabelled figure,
    not abort a plot that is otherwise perfectly drawable."""
    assert pdcp.generator_model_label_from_config(str(tmp_path / "gone.yaml")) == ""


def test_generator_model_label_is_empty_when_the_config_has_no_model_block(
    tmp_path: Path,
):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"data": {"descriptor_dim": 128}}))
    assert pdcp.generator_model_label_from_config(str(path)) == ""


def test_generator_model_label_reports_latent_and_hidden_dims(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"model": {"latent_dim": 4, "generator_hidden_dims": [6, 8]}})
    )
    label = pdcp.generator_model_label_from_config(str(path))
    assert "latent=4" in label and "[6, 8]" in label
