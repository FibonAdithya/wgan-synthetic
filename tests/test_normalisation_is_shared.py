"""The L2 rule must live in exactly one place.

`src/data/dataset.py::apply_preprocess` decides how descriptors are normalised
for training, and `src/eval/eda/series.py::maybe_l2_normalize` is the eval-side
statement of the same rule. Every copy of that arithmetic is somewhere the two
can drift apart without a test noticing -- an eval that silently measures
differently-preprocessed vectors than training saw. These tests fail if a copy
comes back.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.eval.eda.series import maybe_l2_normalize

EVAL_DIR = Path(__file__).resolve().parents[1] / "src" / "eval"
CANONICAL = EVAL_DIR / "eda" / "series.py"


def eval_modules() -> list[Path]:
    return sorted(p for p in EVAL_DIR.rglob("*.py") if p.name != "__init__.py")


def function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def test_the_eval_tree_has_modules_to_check():
    """Guard against the glob silently matching nothing."""
    assert len(eval_modules()) > 5


@pytest.mark.parametrize("path", eval_modules(), ids=lambda p: p.name)
def test_no_module_defines_its_own_normaliser(path: Path):
    if path == CANONICAL:
        return
    defined = function_names(path)
    assert "l2_normalize" not in defined, (
        f"{path.name} defines its own l2_normalize; "
        f"import maybe_l2_normalize from src.eval.eda.series instead"
    )
    assert "maybe_l2_normalize" not in defined


def test_canonical_normaliser_is_where_the_others_must_import_from():
    assert "maybe_l2_normalize" in function_names(CANONICAL)


def test_l2_mode_gives_unit_norm_rows():
    x = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    out = maybe_l2_normalize(x, "l2")
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], rtol=1e-6)


def test_none_mode_is_a_passthrough():
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    assert maybe_l2_normalize(x, "none") is x


def test_rescales_every_row_to_unit_length_whatever_its_magnitude():
    """Ported from test_evaluate_file_to_file.py, whose local copy this replaces."""
    rng = np.random.default_rng(9)
    x = rng.normal(size=(25, 6)) * rng.uniform(0.1, 50.0, size=(25, 1))
    out = maybe_l2_normalize(x, "l2")
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)


def test_direction_is_preserved_while_magnitude_is_dropped():
    """Also ported: unit length alone would pass if directions were scrambled."""
    rng = np.random.default_rng(10)
    x = rng.normal(size=(15, 4))
    out = maybe_l2_normalize(x, "l2")
    cosines = (x * out).sum(axis=1) / np.linalg.norm(x, axis=1)
    np.testing.assert_allclose(cosines, 1.0, rtol=1e-5)


def test_zero_rows_do_not_divide_by_zero():
    """The eps clamp is the reason a shared copy matters: it is easy to drop."""
    out = maybe_l2_normalize(np.zeros((2, 4), dtype=np.float32), "l2")
    assert np.isfinite(out).all()
