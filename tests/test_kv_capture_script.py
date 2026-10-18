import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "capture_scout",
    Path(__file__).parent.parent / "scripts" / "kv" / "capture_scout.py",
)
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)  # must not import torch/transformers at module level


def test_select_books_skips_short_rows_and_counts_globally():
    """Catches counting short rows as books (shifts every later book index)."""
    counts = [40000, 100, 50000, 33000, 10, 32768, 70000]
    assert cs.select_books(counts, 32768, 0, 2) == [0, 2]
    assert cs.select_books(counts, 32768, 2, 3) == [3, 5, 6]


def test_select_books_too_few_raises():
    with pytest.raises(ValueError, match="only"):
        cs.select_books([40000, 10], 32768, 0, 2)


def test_existing_doc_dir_refused(tmp_path):
    """Catches a re-run silently overwriting book 4."""
    (tmp_path / "doc4").mkdir()
    with pytest.raises(FileExistsError, match="doc4"):
        cs.planned_dirs(tmp_path, [4, 5])


def test_planned_dirs_returns_paths(tmp_path):
    assert cs.planned_dirs(tmp_path, [4, 5]) == [tmp_path / "doc4", tmp_path / "doc5"]


def test_meta_compat():
    """Catches capturing new books under a different scale or layer set."""
    m = {
        "global_layers": [3, 7],
        "attn_scale": 0.1,
        "floor_scale": 8192,
        "gqa": "x",
        "docs": [],
    }
    cs.check_meta_compatible(m, dict(m, docs=[1]))
    with pytest.raises(ValueError, match="attn_scale"):
        cs.check_meta_compatible(m, dict(m, attn_scale=0.2))
    with pytest.raises(ValueError, match="global_layers"):
        cs.check_meta_compatible(m, dict(m, global_layers=[3]))
