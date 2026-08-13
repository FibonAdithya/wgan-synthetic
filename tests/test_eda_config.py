import argparse
import dataclasses

from src.eval.eda import config


def _full_namespace() -> argparse.Namespace:
    return argparse.Namespace(
        real_path="r.npy",
        real_format="auto",
        synthetic_path=["a=a.npy"],
        synthetic_format="auto",
        output_dir="out",
        preprocess="l2",
        metric="l2",
        max_vectors=50000,
        num_pairs=200000,
        knn=5,
        ann_k=100,
        ann_hub_k=10,
        ann_max_rows=20000,
        knn_max_rows=20000,
        ivf_nlist=256,
        bins=80,
        top_divergent=16,
        seed=42,
        no_png=False,
        glyph_samples=8,
        plotlyjs="inline",
        max_panel_dim=256,
    )


def test_from_args_carries_every_field_across():
    cfg = config.EdaConfig.from_args(_full_namespace())

    assert cfg.real_path == "r.npy"
    assert cfg.synthetic_path == ["a=a.npy"]
    assert cfg.preprocess == "l2"
    assert cfg.ann_max_rows == 20000
    assert cfg.plotlyjs == "inline"


def test_from_args_covers_every_field_the_parser_produces():
    """Parity guard: a new --flag must reach EdaConfig, not be dropped here."""
    ns = _full_namespace()
    cfg = config.EdaConfig.from_args(ns)

    assert set(vars(ns)) == {f.name for f in dataclasses.fields(cfg)}


def test_missing_glyph_samples_falls_back_to_the_default():
    """compare_variants has historically built Namespaces without this field.

    eda_report.run guarded it with getattr; that guard moves here.
    """
    ns = _full_namespace()
    del ns.glyph_samples

    assert config.EdaConfig.from_args(ns).glyph_samples == config.GLYPH_SAMPLES_DEFAULT


def test_missing_max_panel_dim_falls_back_to_the_default():
    """Namespaces are still built by hand outside the CLI, as in compare_variants.

    One built before this field existed must land on the default rather than
    raising an AttributeError deep inside a panel.
    """
    ns = _full_namespace()
    del ns.max_panel_dim

    assert config.EdaConfig.from_args(ns).max_panel_dim == config.MAX_PANEL_DIM_DEFAULT


def test_synthetic_path_none_becomes_an_empty_list():
    """argparse leaves a repeatable append-action flag at None when unused."""
    ns = _full_namespace()
    ns.synthetic_path = None

    assert config.EdaConfig.from_args(ns).synthetic_path == []


def test_from_args_carries_the_metric():
    assert config.EdaConfig.from_args(_full_namespace()).metric == "l2"
