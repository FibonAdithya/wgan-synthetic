from pathlib import Path

import yaml

from src.data.kv_capture import GLOBAL_LAYERS

ROOT = Path(__file__).resolve().parent.parent / "configs" / "scout_kv" / "v0"


def _load(L):
    return yaml.safe_load((ROOT / f"L{L:02d}.yaml").read_text())


def test_one_config_per_global_layer_and_nothing_else():
    assert sorted(p.name for p in ROOT.glob("*.yaml")) == [
        f"L{L:02d}.yaml" for L in GLOBAL_LAYERS
    ]


def test_configs_differ_only_in_layer_and_output_dir():
    """Catches one layer's config drifting (a silent per-layer hyperparameter change)."""
    base = _load(43)
    for L in GLOBAL_LAYERS:
        c = _load(L)
        assert (
            c["data"]["kv_layer"] == L
            and c["output_dir"] == f"runs/scout_kv/v0/L{L:02d}"
        )
        for d in (c, base):
            d["data"].pop("kv_layer")
            d.pop("output_dir")
        assert c == base
        base = _load(43)


def test_train_books_exclude_reference_books():
    """Catches training on books 12-19, which the gate's reference uses."""
    assert set(_load(43)["data"]["kv_books"]) == set(range(12))


def test_norms_are_preserved():
    """Catches any normalisation sneaking back in."""
    c = _load(43)
    assert c["data"]["preprocess"]["l2_normalize"] is False
    assert c["model"]["output_normalize"] is False
