"""B8: apply the plan's fixed decision rules to the B7 outputs and write summary.json.

Run from the repo root: PYTHONPATH=. python docs/results/scout-kv/controls/summarise.py
Inputs (this directory): neighbours.jsonl, and when present train_books_real.jsonl,
pooled_real.jsonl, norm_equalised.jsonl, gaussian.jsonl, v0_best_subset.jsonl.
References: docs/results/scout-kv/reference/seed{0..4}.jsonl, v0/best.jsonl.
Rules are those in docs/ai/plans/2026-10-19-scout-kv-controls.md, "Decision rules".
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.eval import kv_gate

HERE = Path(__file__).resolve().parent
RES = HERE.parent
LAYERS = (11, 19, 43)
RULE_STATS = ("hnsw_reach_k10", "hnsw_recall_k10_ef176", "ivf_ratio_k128")
CONTROLS = (
    "train_books_real",
    "pooled_real",
    "norm_equalised",
    "gaussian",
    "pooled_real_seed1",
    "pooled_real_seed2",
)


def _read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):  # average ranks over ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    keep = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
    if len(keep) < 3:
        return None, len(keep)
    ra = _ranks([a[i] for i in keep])
    rb = _ranks([b[i] for i in keep])
    if ra.std() == 0 or rb.std() == 0:
        return None, len(keep)
    return float(np.corrcoef(ra, rb)[0, 1]), len(keep)


def _ivf_ratio(r):
    v = kv_gate.ivf_ratios([r])
    return v[0] if v else None


def r5(neighbours, seed0):
    ref = {(r["layer"], r["kv_head"]): r for r in seed0}
    keys = sorted((r["layer"], r["kv_head"]) for r in neighbours)
    nb = {(r["layer"], r["kv_head"]): r for r in neighbours}

    def lag_excess(o):
        s = o["share_lag_le_1024"]
        return None if s is None else s - o["base_share_lag_le_1024"]

    scalars = {
        "lag_le_1024_minus_base": lambda o: lag_excess(o),
        "pos_lt_16_minus_base": lambda o: o["share_pos_lt_16"]
        - o["base_share_pos_lt_16"],
        "hub_slot_share": lambda o: o["hub_slot_share"],
        "own_book_share": lambda o: o["own_book_share"],
    }
    hardness = {
        "hnsw_recall_k10_ef176": lambda r: r["plain_ood_k10"]["hnsw_recall_op"],
        "ivf_ratio_k128": _ivf_ratio,
    }
    rows = []
    for sn, sf in scalars.items():
        xs = [sf(nb[k]["ood"]) for k in keys]
        for hn, hf in hardness.items():
            ys = [hf(ref[k]) for k in keys]
            rho, n = spearman(xs, ys)
            rows.append(
                {
                    "scalar": sn,
                    "hardness": hn,
                    "rho": rho,
                    "n": n,
                    "associated": rho is not None and abs(rho) >= 0.5,
                }
            )
    # Enrichment per bin: per-corpus ratio, median over the 8 heads per layer,
    # bins with baseline share < 0.01 omitted; flag ratio >= 2 at >= 4 of 12 layers.
    edges = neighbours[0]["edges"]
    enrich = {}
    for kind, num, den in (
        ("lag", "nbr_lag_share", "base_lag_share"),
        ("pos", "nbr_pos_share", "key_pos_share"),
    ):
        per_layer = defaultdict(list)
        for k in keys:
            o = nb[k]["ood"]
            if o[num] is None:
                continue
            ratio = [
                (n_ / d_) if d_ >= 0.01 else None for n_, d_ in zip(o[num], o[den])
            ]
            per_layer[k[0]].append(ratio)
        table = {}
        for L, ratios in sorted(per_layer.items()):
            table[L] = [
                (
                    float(np.median([r[b] for r in ratios]))
                    if all(r[b] is not None for r in ratios)
                    else None
                )
                for b in range(len(edges) - 1)
            ]
        flagged = []
        for b in range(len(edges) - 1):
            n_layers = sum(1 for v in table.values() if v[b] is not None and v[b] >= 2)
            if n_layers >= 4:
                flagged.append(
                    {
                        "bin": [edges[b], edges[b + 1]],
                        "layers_with_ratio_ge_2": n_layers,
                    }
                )
        enrich[kind] = {"per_layer_median_ratio": table, "flagged_bins": flagged}
    return {"spearman": rows, "enrichment": enrich}


def per_layer_neighbours(neighbours):
    by = defaultdict(list)
    for r in neighbours:
        by[r["layer"]].append(r)

    def med(xs):
        # .get: n_distinct_keys is absent from records made before it existed.
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else None

    out = {}
    for L, rs in sorted(by.items()):
        out[L] = {
            q: {
                f: med([r[q].get(f) for r in rs])
                for f in (
                    "own_book_share",
                    "share_lag_le_1024",
                    "base_share_lag_le_1024",
                    "share_pos_lt_16",
                    "base_share_pos_lt_16",
                    "hub_slot_share",
                    "n_distinct_keys",
                    "hub_median_pos",
                    "key_median_pos",
                )
            }
            for q in ("ood", "id")
        }
    return out


def rules(compare, v0cmp):
    """R1/R2 counts: layers outside the real per-layer range in v0's direction."""
    out = {}
    for name, cmp_ in compare.items():
        per_stat = {}
        for s in RULE_STATS:
            n = 0
            cells = {}
            for L in LAYERS:
                v0s = v0cmp["layers"][L]["statistics"][s]["status"]
                cs = cmp_["layers"][L]["statistics"][s]["status"]
                same = v0s in ("above", "below") and cs == v0s
                cells[L] = {"control": cs, "v0": v0s, "same_direction": same}
                n += same
            per_stat[s] = {"layers_in_v0_direction": n, "cells": cells}
        n_stats = sum(1 for v in per_stat.values() if v["layers_in_v0_direction"] >= 2)
        out[name] = {"per_statistic": per_stat, "statistics_with_2_of_3": n_stats}
    return out


XBOOK_DRAWS = ("a_s0", "a_s1", "b_s0", "b_s1")
XBOOK_CONTROLS = ("same_book_tail", "cross_book_tail")


def n_outside(cmp_):
    """Rule statistics with >= 2 of the 3 layers outside the real range in either
    direction. A null statistic (status "none") counts as outside."""
    return sum(
        1
        for s in RULE_STATS
        if sum(cmp_["layers"][L]["statistics"][s]["status"] != "inside" for L in LAYERS)
        >= 2
    )


def xbook_verdict(outside_same, flatters_cross):
    """R6/R7 from the cross-book plan (docs/ai/plans/2026-10-21-scout-kv-cross-book.md)."""
    valid = sum(outside_same) <= 1
    n = sum(flatters_cross)
    if not valid:
        reading = None
    elif n <= 1:
        reading = "reachable"
    elif n >= 3:
        reading = "own_book"
    else:
        reading = "inconclusive"
    return {"positive_control_valid": valid, "n_cross_flatter": n, "reading": reading}


def xbook(refs, v0cmp):
    out = {"draws": {}}
    for d in XBOOK_DRAWS:
        out["draws"][d] = {}
        for c in XBOOK_CONTROLS:
            cmp_ = _compare(_read(HERE / "xbook" / d / f"{c}.jsonl"), refs)
            r = rules({c: cmp_}, v0cmp)[c]
            out["draws"][d][c] = {
                "compare_layers": cmp_,
                "rules": r,
                "flatters": r["statistics_with_2_of_3"] >= 2,
                "n_outside": n_outside(cmp_),
            }
    dr = out["draws"]
    out["verdict"] = xbook_verdict(
        [dr[d]["same_book_tail"]["n_outside"] >= 2 for d in XBOOK_DRAWS],
        [dr[d]["cross_book_tail"]["flatters"] for d in XBOOK_DRAWS],
    )
    return out


# Held-out-book plan (docs/ai/plans/2026-10-21-scout-kv-held-out.md): draw ->
# (seed, held-out query books), fixed before the run.
HELDOUT_DRAWS = {
    "s5": (5, (0, 1, 2, 3, 4, 5, 6, 7)),
    "s6": (6, (4, 5, 6, 7, 8, 9, 10, 11)),
    "s7": (7, (0, 1, 2, 3, 8, 9, 10, 11)),
    "s8": (8, (0, 1, 2, 4, 5, 6, 8, 10)),
}
HELDOUT_CONTROLS = ("rebuild", "held_out_tail")
REF_BOOKS = list(range(12, 20))


def path_check_ok(report):
    """R6a: the seed-0 rebuild paired with reference/seed0 on all 24 rule-layer corpora."""
    return (
        report["n_compared"] == 24
        and not report["missing"]
        and not report["mismatched"]
        and not report["meta_mismatch"]
    )


def heldout_verdict(path_ok, outside_rebuild, flatters_held_out):
    """R6a/R6/R7 from the held-out plan."""
    if not path_ok:
        return {
            "path_ok": False,
            "positive_control_valid": None,
            "n_held_out_flatter": None,
            "reading": None,
        }
    v = xbook_verdict(outside_rebuild, flatters_held_out)
    reading = "not_reachable" if v["reading"] == "own_book" else v["reading"]
    return {
        "path_ok": True,
        "positive_control_valid": v["positive_control_valid"],
        "n_held_out_flatter": v["n_cross_flatter"],
        "reading": reading,
    }


def check_heldout_draw(records, seed, query_books, control):
    """Refuse records that do not belong to this draw and control."""
    want_q = REF_BOOKS if control == "rebuild" else list(query_books)
    for r in records:
        got = (r["seed"], r["control"], r["books"], r["query_books"])
        if got != (seed, control, REF_BOOKS, want_q):
            raise SystemExit(
                f"record {got} does not match draw seed={seed} control={control} "
                f"query_books={want_q}"
            )


def paired_diff(cmp_a, cmp_b):
    """Per rule layer and statistic, a - b of the pooled value; None if either is None."""
    out = {}
    for L in LAYERS:
        out[L] = {}
        for s in RULE_STATS:
            a = cmp_a["layers"][L]["statistics"][s]["value"]
            b = cmp_b["layers"][L]["statistics"][s]["value"]
            out[L][s] = None if a is None or b is None else a - b
    return out


def heldout(refs, v0cmp, path_report):
    out = {"path_check": path_report, "draws": {}}
    for d, (seed, qb) in HELDOUT_DRAWS.items():
        out["draws"][d] = {"seed": seed, "query_books": list(qb)}
        for c in HELDOUT_CONTROLS:
            recs = _read(HERE / "heldout" / d / f"{c}.jsonl")
            check_heldout_draw(recs, seed, qb, c)
            cmp_ = _compare(recs, refs)
            r = rules({c: cmp_}, v0cmp)[c]
            out["draws"][d][c] = {
                "compare_layers": cmp_,
                "rules": r,
                "flatters": r["statistics_with_2_of_3"] >= 2,
                "n_outside": n_outside(cmp_),
            }
        dr = out["draws"][d]
        dr["held_out_minus_rebuild"] = paired_diff(
            dr["held_out_tail"]["compare_layers"], dr["rebuild"]["compare_layers"]
        )
    dr = out["draws"]
    out["verdict"] = heldout_verdict(
        path_check_ok(path_report),
        [dr[d]["rebuild"]["n_outside"] >= 2 for d in HELDOUT_DRAWS],
        [dr[d]["held_out_tail"]["flatters"] for d in HELDOUT_DRAWS],
    )
    return out


def v0_match(v0cmp, subcmp):
    tol = {
        "hnsw_reach_k10": ("abs", 1 / 8),
        "hnsw_recall_k10_ef176": ("abs", 0.01),
        "ivf_recall_k128_np24": ("abs", 0.01),
        "hnsw_recall_k10_ef176_id": ("abs", 0.01),
        "ivf_ratio_k128": ("rel", 0.05),
    }
    cells, ok = [], True
    for L in LAYERS:
        for s, (kind, t) in tol.items():
            a = v0cmp["layers"][L]["statistics"][s]
            b = subcmp["layers"][L]["statistics"][s]
            if a["value"] is None or b["value"] is None:
                # A null statistic matches only a null statistic.
                within = a["value"] is None and b["value"] is None
            else:
                d = abs(a["value"] - b["value"])
                lim = t * abs(a["value"]) if kind == "rel" else t
                within = d <= lim + 1e-12
            same = a["status"] == b["status"]
            ok &= within and same
            cells.append(
                {
                    "layer": L,
                    "statistic": s,
                    "committed": a["value"],
                    "remeasured": b["value"],
                    "within_tolerance": bool(within),
                    "same_status": same,
                }
            )
    return {"match": bool(ok), "cells": cells}


_CANONICAL = None


def _compare(records, refs):
    """compare_layers under the gate's canonical conditions; refuses records whose
    conditions differ, so no summary is built on mismatched measurements."""
    global _CANONICAL
    if _CANONICAL is None:
        _CANONICAL = kv_gate.load_kv_gate(kv_gate.GATES_FILE)["canonical"]
    out = kv_gate.compare_layers(records, refs, _CANONICAL)
    if out["condition_mismatches"]:
        raise SystemExit(f"condition mismatches: {out['condition_mismatches'][:3]}")
    del out["condition_mismatches"]
    return out


def main():
    refs = [_read(RES / f"reference/seed{i}.jsonl") for i in range(5)]
    nb = _read(HERE / "neighbours.jsonl")
    summary = {
        "neighbours_per_layer": per_layer_neighbours(nb),
        "r5": r5(nb, refs[0]),
    }
    real_per_layer = _compare(_read(RES / "v0/best.jsonl"), refs)
    summary["real_per_layer_hardness"] = {
        L: {
            s: {"min": v["min"], "max": v["max"], "mean": v["mean"]}
            for s, v in e["statistics"].items()
        }
        for L, e in real_per_layer["layers"].items()
    }
    v0_sub = [r for r in _read(RES / "v0/best.jsonl") if r["layer"] in LAYERS]
    v0cmp = _compare(v0_sub, refs)
    compare = {}
    for name in CONTROLS:
        p = HERE / f"{name}.jsonl"
        if p.exists():
            compare[name] = _compare(_read(p), refs)
    summary["compare_layers"] = {"v0_best": v0cmp, **compare}
    if compare:
        summary["rules"] = rules(compare, v0cmp)
        if "pooled_real" in summary["rules"]:
            summary["R1_pooled_real_flatters"] = (
                summary["rules"]["pooled_real"]["statistics_with_2_of_3"] >= 2
            )
        # One draw against five: R1 is applied to every pooled_real seed measured.
        summary["R1_per_seed"] = {
            n: summary["rules"][n]["statistics_with_2_of_3"] >= 2
            for n in ("pooled_real", "pooled_real_seed1", "pooled_real_seed2")
            if n in summary["rules"]
        }
        if "train_books_real" in summary["rules"]:
            summary["R2_train_books_easier"] = (
                summary["rules"]["train_books_real"]["statistics_with_2_of_3"] >= 2
            )
    p = HERE / "v0_best_subset.jsonl"
    if p.exists():
        summary["v0_remeasure"] = v0_match(v0cmp, _compare(_read(p), refs))
    if all(
        (HERE / "xbook" / d / f"{c}.jsonl").exists()
        for d in XBOOK_DRAWS
        for c in XBOOK_CONTROLS
    ):
        summary["xbook"] = xbook(refs, v0cmp)
        summary["R6_R7_xbook"] = summary["xbook"]["verdict"]
    pc = HERE / "heldout" / "pair_check.json"
    if pc.exists() and all(
        (HERE / "heldout" / d / f"{c}.jsonl").exists()
        for d in HELDOUT_DRAWS
        for c in HELDOUT_CONTROLS
    ):
        summary["heldout"] = heldout(refs, v0cmp, json.loads(pc.read_text()))
        summary["R6_R7_heldout"] = summary["heldout"]["verdict"]
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k.startswith("R")}, indent=1))


if __name__ == "__main__":
    main()
