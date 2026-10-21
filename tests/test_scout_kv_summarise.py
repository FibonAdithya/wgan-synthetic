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


# --- held-out plan, Task 3 ---


def test_heldout_verdict_boundaries(summarise):
    """Catches off-by-one in either threshold, the positive-control or path gate
    being ignored, the label not renamed, and n_compared not checked."""
    v = summarise.heldout_verdict
    want = ["reachable", "reachable", "inconclusive", "not_reachable", "not_reachable"]
    for n, reading in enumerate(want):
        flat = [True] * n + [False] * (4 - n)
        out = v(True, [True, False, False, False], flat)
        assert out["reading"] == reading and out["n_held_out_flatter"] == n
        assert out["positive_control_valid"] is True and out["path_ok"] is True
    out = v(True, [True, True, False, False], [False] * 4)
    assert out["positive_control_valid"] is False and out["reading"] is None
    out = v(False, [False] * 4, [False] * 4)
    assert out["path_ok"] is False and out["reading"] is None
    clean = {"n_compared": 24, "missing": [], "mismatched": [], "meta_mismatch": []}
    assert summarise.path_check_ok(clean) is True
    assert summarise.path_check_ok({**clean, "n_compared": 23}) is False
    for k in ("missing", "mismatched", "meta_mismatch"):
        assert summarise.path_check_ok({**clean, k: [[11, 0]]}) is False


def _heldout_recs(control, seed=5, qbooks=(0, 1, 2, 3, 4, 5, 6, 7)):
    q = list(range(12, 20)) if control == "rebuild" else list(qbooks)
    return [
        {
            "seed": seed,
            "control": control,
            "books": list(range(12, 20)),
            "query_books": q,
        }
        for _ in range(3)
    ]


def test_check_heldout_draw_refuses_mislabelled_records(summarise):
    """Catches draws not checked, and only the first record checked."""
    chk = summarise.check_heldout_draw
    qb = (0, 1, 2, 3, 4, 5, 6, 7)
    chk(_heldout_recs("held_out_tail"), 5, qb, "held_out_tail")
    chk(_heldout_recs("rebuild"), 5, qb, "rebuild")
    for field, bad in (
        ("seed", 6),
        ("query_books", [4, 5, 6, 7, 8, 9, 10, 11]),
        ("control", "rebuild"),
        ("books", list(range(12, 19))),
    ):
        recs = _heldout_recs("held_out_tail")
        recs[-1][field] = bad
        with pytest.raises(SystemExit):
            chk(recs, 5, qb, "held_out_tail")
    recs = _heldout_recs("rebuild")
    recs[-1]["query_books"] = list(qb)
    with pytest.raises(SystemExit):
        chk(recs, 5, qb, "rebuild")


def test_paired_diff_keeps_zero(summarise):
    """Catches a truthiness test where `is None` is meant: 0.0 is a real value."""
    st = {s: ("inside",) * 3 for s in summarise.RULE_STATS}
    a = _cmp(st, {"hnsw_reach_k10": (0.0, 0.5, None)})
    b = _cmp(st, {"hnsw_reach_k10": (0.125, 0.0, 0.25)})
    d = summarise.paired_diff(a, b)
    assert d[11]["hnsw_reach_k10"] == pytest.approx(-0.125)
    assert d[19]["hnsw_reach_k10"] == pytest.approx(0.5)
    assert d[43]["hnsw_reach_k10"] is None
