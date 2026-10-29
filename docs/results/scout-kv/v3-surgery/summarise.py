"""Apply the README's decision rule to the surgery records and write summary.json.

    PYTHONPATH=. python docs/results/scout-kv/v3-surgery/summarise.py

Inputs, all committed: this directory's gate_L{11,15,43}.jsonl, intrinsic.jsonl and
v2_neighbours/L{LL}_s{S}.jsonl; ../reference/seed{0..4}.jsonl; ../v2/compare_layers_s{42,1}.json
(for v2's per-layer values); ../controls/neighbours.jsonl (real neighbour fields, seed 0).
A missing input stops the script.
"""

import json
import statistics as pystats
from pathlib import Path

from src.data.kv_capture import GLOBAL_LAYERS
from src.eval.kv_gate import compare_layers, pooled_statistics
from src.kv.surgery import NEIGHBOUR_FIELDS, SETS

HERE = Path(__file__).resolve().parent
SCOUT = HERE.parent
LAYERS = (11, 15, 43)
SEEDS = (0, 1)
V2_SEEDS = (42, 1)
STAT = "hnsw_recall_k10_ef176"
SANITY_TOL = 0.01
KEEP_MARGIN = 0.05


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def need(p):
    if not Path(p).is_file():
        raise SystemExit(f"missing input: {p}")
    return p


def median_or_none(vals):
    vals = [v for v in vals if v is not None]
    return float(pystats.median(vals)) if vals else None


def nbr_medians(recs, nested):
    """Median over heads of each neighbour field; `nested` reads rec['ood'][field]."""
    out = {}
    for f in NEIGHBOUR_FIELDS:
        vals = [r["ood"][f] if nested else r[f"nbr_{f}"] for r in recs]
        out[f] = median_or_none(vals)
    return out


def main():
    # The box's gate.jsonl, split by layer to keep each file under 1 MB;
    # `cat gate_L11.jsonl gate_L15.jsonl gate_L43.jsonl` matches its sha256 in SHA256SUMS.
    gate = [r for L in LAYERS for r in read_jsonl(need(HERE / f"gate_L{L}.jsonl"))]
    intr = read_jsonl(need(HERE / "intrinsic.jsonl"))
    refs = [read_jsonl(need(SCOUT / f"reference/seed{s}.jsonl")) for s in range(5)]
    v2cmp = {
        s: json.loads(need(SCOUT / f"v2/compare_layers_s{s}.json").read_text())[
            "layers"
        ]
        for s in V2_SEEDS
    }
    real_nbr = read_jsonl(need(SCOUT / "controls/neighbours.jsonl"))
    v2_nbr = {
        (L, s): read_jsonl(need(HERE / f"v2_neighbours/L{L:02d}_s{s}.jsonl"))
        for L in GLOBAL_LAYERS
        for s in V2_SEEDS
    }

    rule = {}
    for L in LAYERS:
        st = [v2cmp[s][str(L)]["statistics"][STAT] for s in V2_SEEDS]
        real_max = st[0]["max"]
        v2_mean = sum(x["value"] for x in st) / len(st)
        rule[L] = {
            "real_min": st[0]["min"],
            "real_max": real_max,
            "v2_mean": v2_mean,
            "v2": {s: x["value"] for s, x in zip(V2_SEEDS, st, strict=True)},
            "remove_at": real_max + 0.5 * (v2_mean - real_max),
            "keep_at": real_max + KEEP_MARGIN,
        }

    sets = {L: {} for L in LAYERS}
    for name in SETS:
        for seed in SEEDS:
            recs = [r for r in gate if r["source"] == name and r["seed"] == seed]
            if sorted((r["layer"], r["kv_head"]) for r in recs) != sorted(
                (L, g) for L in LAYERS for g in range(8)
            ):
                raise SystemExit(f"{name} seed {seed}: expected 8 heads at {LAYERS}")
            cmp = compare_layers(recs, refs)["layers"]
            for L in LAYERS:
                layer_recs = [r for r in recs if r["layer"] == L]
                sets[L].setdefault(name, {})[seed] = {
                    "stats": cmp[L]["statistics"],
                    "nbr": nbr_medians(layer_recs, nested=False),
                }

    sanity = {}
    for L in LAYERS:
        for seed in SEEDS:
            ref = pooled_statistics([r for r in refs[seed] if r["layer"] == L])[STAT]
            got = sets[L]["real"][seed]["stats"][STAT]["value"]
            sanity[f"L{L}_s{seed}"] = {
                "real": got,
                "reference": ref,
                "diff": got - ref,
                "ok": abs(got - ref) <= SANITY_TOL,
            }
    sanity_ok = all(v["ok"] for v in sanity.values())

    calls = {}
    for L in LAYERS:
        calls[L] = {}
        for name in SETS:
            vals = [sets[L][name][s]["stats"][STAT]["value"] for s in SEEDS]
            if all(v >= rule[L]["remove_at"] for v in vals):
                call = "removes"
            elif all(v <= rule[L]["keep_at"] for v in vals):
                call = "keeps"
            else:
                call = "partial"
            calls[L][name] = {"call": call, "values": vals}

    intrinsic = {}
    for r in intr:
        key = (r["layer"], r["source"], r["seed"], r["flag"])
        if key in intrinsic:
            raise SystemExit(f"duplicate intrinsic line {key}")
        intrinsic[key] = {
            k: r[k]
            for k in (
                "lid_median",
                "relative_contrast_median",
                "hubness_skew",
                "ivf_gini",
                "n_rows",
            )
        }
    missing = [
        (L, n, s, f)
        for L in LAYERS
        for n in SETS
        for s in SEEDS
        for f in ("body", "tail")
        if (L, n, s, f) not in intrinsic
    ]
    if missing:
        raise SystemExit(f"missing intrinsic lines: {missing}")

    neighbours = {}
    for L in GLOBAL_LAYERS:
        row = {"real": nbr_medians([r for r in real_nbr if r["layer"] == L], True)}
        for s in V2_SEEDS:
            recs = v2_nbr[(L, s)]
            if sorted(r["kv_head"] for r in recs) != list(range(8)):
                raise SystemExit(f"v2 neighbours L{L} s{s}: expected 8 heads")
            row[f"v2_s{s}"] = nbr_medians(recs, True)
        neighbours[L] = row

    out = {
        "statistic": STAT,
        "rule": rule,
        "sanity": sanity,
        "sanity_ok": sanity_ok,
        "calls": calls,
        "sets": sets,
        "intrinsic": [
            {"layer": k[0], "source": k[1], "seed": k[2], "flag": k[3], **v}
            for k, v in sorted(intrinsic.items())
        ],
        "neighbours": neighbours,
    }
    (HERE / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print("sanity", "ok" if sanity_ok else "FAILED")
    for k, v in sanity.items():
        print(
            f"  {k}: real {v['real']:.4f} ref {v['reference']:.4f} diff {v['diff']:+.4f}"
        )
    for L in LAYERS:
        r = rule[L]
        print(f"L{L}: removes >= {r['remove_at']:.4f}, keeps <= {r['keep_at']:.4f}")
        for name in SETS:
            c = calls[L][name]
            print(
                f"  {name:9s} {c['call']:8s} "
                + " ".join(f"{v:.4f}" for v in c["values"])
            )


if __name__ == "__main__":
    main()
