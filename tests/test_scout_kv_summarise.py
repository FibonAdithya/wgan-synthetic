import importlib.util
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[1] / "docs/results/scout-kv/controls/summarise.py"
)
_TOL_STATS = (
    "hnsw_reach_k10",
    "hnsw_recall_k10_ef176",
    "ivf_recall_k128_np24",
    "hnsw_recall_k10_ef176_id",
    "ivf_ratio_k128",
)


@pytest.fixture(scope="module")
def summarise():
    spec = importlib.util.spec_from_file_location("summarise", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cmp(statuses, values=None):
    """A compare_layers-shaped dict: {stat: (status at L11, L19, L43)}."""
    values = values or {}
    return {
        "layers": {
            L: {
                "statistics": {
                    s: {"status": v[i], "value": values.get(s, (0.5, 0.5, 0.5))[i]}
                    for s, v in statuses.items()
                }
            }
            for i, L in enumerate((11, 19, 43))
        }
    }


def test_rules_counts_only_cells_in_v0_direction(summarise):
    """Catches counting any outside cell (reach 3, recall 2 -> 2 statistics),
    counting cells where v0 is inside or none, and a threshold of > 2."""
    v0 = _cmp(
        {
            "hnsw_reach_k10": ("above", "above", "above"),
            "hnsw_recall_k10_ef176": ("below", "inside", "below"),
            "ivf_ratio_k128": ("none", "above", "above"),
        }
    )
    ctl = _cmp(
        {
            "hnsw_reach_k10": ("above", "below", "above"),
            "hnsw_recall_k10_ef176": ("below", "below", "inside"),
            "ivf_ratio_k128": ("none", "above", "inside"),
        }
    )
    r = summarise.rules({"c": ctl}, v0)["c"]
    per = r["per_statistic"]
    assert per["hnsw_reach_k10"]["layers_in_v0_direction"] == 2
    assert per["hnsw_recall_k10_ef176"]["layers_in_v0_direction"] == 1
    assert per["ivf_ratio_k128"]["layers_in_v0_direction"] == 1
    assert r["statistics_with_2_of_3"] == 1


def _match_case(summarise, committed, remeasured, stat, status=("inside",) * 2):
    base = {s: (0.5, 0.5, 0.5) for s in _TOL_STATS}
    sts = {s: ("inside",) * 3 for s in _TOL_STATS}
    sts_b = {**sts, stat: (status[1], "inside", "inside")}
    sts_a = {**sts, stat: (status[0], "inside", "inside")}
    a = _cmp(sts_a, {**base, stat: (committed, 0.5, 0.5)})
    b = _cmp(sts_b, {**base, stat: (remeasured, 0.5, 0.5)})
    return summarise.v0_match(a, b)["match"]


def test_v0_match_tolerance_edges(summarise):
    """Catches strict < at an edge, abs vs rel swapped, and a status flip ignored."""
    assert _match_case(summarise, 0.5, 0.625, "hnsw_reach_k10")
    assert not _match_case(summarise, 0.5, 0.75, "hnsw_reach_k10")
    assert _match_case(summarise, 0.80, 0.81, "hnsw_recall_k10_ef176")
    assert not _match_case(summarise, 0.80, 0.8102, "hnsw_recall_k10_ef176")
    assert _match_case(summarise, 2.0, 2.1, "ivf_ratio_k128")
    assert not _match_case(summarise, 2.0, 2.12, "ivf_ratio_k128")
    assert not _match_case(
        summarise, 0.5, 0.5, "hnsw_recall_k10_ef176", ("inside", "above")
    )


def test_v0_match_handles_null_values(summarise):
    """Both null matches; one null does not, and neither raises TypeError."""
    assert _match_case(summarise, None, None, "ivf_ratio_k128", ("none", "none"))
    assert not _match_case(summarise, None, 2.0, "ivf_ratio_k128", ("none", "none"))
    assert not _match_case(summarise, 2.0, None, "ivf_ratio_k128", ("none", "none"))


def test_spearman_ties_nulls_and_degenerate(summarise):
    """Average ranks over ties give 4.5 / sqrt(4.5 * 5); catches ordinal ranks
    (0.8), nulls not dropped, and constant input not returning None."""
    rho, n = summarise.spearman([1, 2, 2, 3, None], [1, 3, 2, 4, 9])
    assert n == 4 and rho == pytest.approx(4.5 / (4.5 * 5) ** 0.5, abs=1e-12)
    assert summarise.spearman([1, 2, None], [1, 2, 3]) == (None, 2)
    assert summarise.spearman([1, 1, 1], [1, 2, 3]) == (None, 3)


def test_n_outside_counts_both_directions(summarise):
    c = _cmp(
        {
            "hnsw_reach_k10": ("above", "below", "inside"),
            "hnsw_recall_k10_ef176": ("inside", "inside", "below"),
            "ivf_ratio_k128": ("none", "above", "inside"),
        }
    )
    assert summarise.n_outside(c) == 2


def test_xbook_verdict_boundaries(summarise):
    same = [True, False, False, False]
    want = ["reachable", "reachable", "inconclusive", "own_book", "own_book"]
    for n, reading in enumerate(want):
        cross = [True] * n + [False] * (4 - n)
        v = summarise.xbook_verdict(same, cross)
        assert v == {
            "positive_control_valid": True,
            "n_cross_flatter": n,
            "reading": reading,
        }
    v = summarise.xbook_verdict([True, True, False, False], [False] * 4)
    assert v["positive_control_valid"] is False and v["reading"] is None
