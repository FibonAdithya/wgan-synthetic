"""Build docs/results/scout-kv/v2/report.html, the v2 results report for scout_kv.

    PYTHONPATH=. python docs/results/scout-kv/v2/make_report.py [out_html]

Inputs, all committed: this directory's verdict.json, gate_s{42,1}.jsonl, check_s{42,1}.json,
pilot/port.json, runs/L{LL}_s{SEED}/{locality,intrinsic,run_metadata}, runs/logs/*.log; and from
the repository: locality/locality.json (real), v1/full/{v1_final_seed0.jsonl,
check_v1_final_seed0.json, intrinsic.jsonl}, v1-diagnostics/noise/measure/v1s1_final_L*.jsonl,
runs/v1_seed1_intrinsic.jsonl, v2-probe/runs/v1s1_intrinsic_L27.jsonl,
v2-probe/control/cross_book_lag.json, reference/seed{0..4}.jsonl and gates/scout_kv.yaml.

A missing input stops the script before anything is written. Figures are PNGs inlined as
base64, so the page opens on its own. Every number on the page is computed from the inputs.
"""

import base64
import html
import io
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.kv_capture import GLOBAL_LAYERS  # noqa: E402
from src.eval.kv_gate import STATISTICS, compare_layers, load_kv_gate  # noqa: E402
from src.kv import book_judge as bj  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "report.html"
SCOUT = HERE.parent
REPO = SCOUT.parent.parent.parent
RUNS = HERE / "runs"
SEEDS = (42, 1)
V1S1_LAYERS = (19, 27, 35, 43)
LABEL = {
    "hnsw_reach_k10": "HNSW reach, k=10",
    "hnsw_recall_k10_ef176": "HNSW recall@10, ef 176",
    "ivf_ratio_k128": "IVF scan ratio, k=128",
    "ivf_recall_k128_np24": "IVF recall@128, nprobe 24",
    "hnsw_recall_k10_ef176_id": "HNSW recall@10, ef 176, key-side",
}
COL = {
    "real": "black",
    "v1": "#7f7f7f",
    "v1s1": "#bdbdbd",
    "s42": "#d95f02",
    "s1": "#1b9e77",
}

FILES = {
    "verdict": HERE / "verdict.json",
    "port": HERE / "pilot/port.json",
    **{f"gate_s{s}": HERE / f"gate_s{s}.jsonl" for s in SEEDS},
    **{f"check_s{s}": HERE / f"check_s{s}.json" for s in SEEDS},
    "real_locality": SCOUT / "locality/locality.json",
    "v1_gate": SCOUT / "v1/full/v1_final_seed0.jsonl",
    "v1_check": SCOUT / "v1/full/check_v1_final_seed0.json",
    "v1_intrinsic": SCOUT / "v1/full/intrinsic.jsonl",
    "v1s1_intrinsic": RUNS / "v1_seed1_intrinsic.jsonl",
    "v1s1_intrinsic_L27": SCOUT / "v2-probe/runs/v1s1_intrinsic_L27.jsonl",
    "cross_book": SCOUT / "v2-probe/control/cross_book_lag.json",
    "pilot_log": RUNS / "logs/v2_b2_pilot.log",
    "full_log": RUNS / "logs/v2_b4_full.log",
    "gate_file": REPO / "gates/scout_kv.yaml",
    **{f"ref{s}": SCOUT / f"reference/seed{s}.jsonl" for s in range(5)},
    **{
        f"v1s1_gate_L{L}": SCOUT / f"v1-diagnostics/noise/measure/v1s1_final_L{L}.jsonl"
        for L in V1S1_LAYERS
    },
    **{
        f"{n}_L{L:02d}_s{s}": RUNS / f"L{L:02d}_s{s}/{n}"
        for L in GLOBAL_LAYERS
        for s in SEEDS
        for n in ("locality.json", "intrinsic.jsonl", "run_metadata.json")
    },
}


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def read_json(p):
    return json.loads(Path(p).read_text())


def loc_entry(path, layer):
    for e in read_json(path)["layers"]:
        if e["layer"] == layer:
            return e
    raise SystemExit(f"layer {layer} not in {path}")


def intrinsic_row(rows, layer, flag, source):
    hits = [
        r
        for r in rows
        if r["layer"] == layer and r["flag"] == flag and r["source"] == source
    ]
    if len(hits) != 1:
        raise SystemExit(
            f"expected one {source} {flag} row at L{layer}, got {len(hits)}"
        )
    return hits[0]


def v1_row(rows, layer, flag):
    """v1's final checkpoint: the `v1_final` row, else `v1_best` (best was step 30,000)."""
    has_final = any(r["layer"] == layer and r["source"] == "v1_final" for r in rows)
    return intrinsic_row(rows, layer, flag, "v1_final" if has_final else "v1_best")


def run_seconds(log):
    return {
        m.group(1): int(m.group(2))
        for m in re.finditer(
            r"^DONE /workspace/kv/v2/(L\d+_s\d+) seconds=(\d+)$", log.read_text(), re.M
        )
    }


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def collect():
    missing = [f"{k}: {p}" for k, p in FILES.items() if not p.is_file()]
    if missing:
        raise SystemExit(
            "missing input files, not writing a report:\n  " + "\n  ".join(missing)
        )
    d = {"verdict": read_json(FILES["verdict"]), "port": read_json(FILES["port"])}
    canonical = load_kv_gate(FILES["gate_file"])["canonical"]
    refs = [read_jsonl(FILES[f"ref{s}"]) for s in range(5)]
    d["bands"] = {
        s["statistic"]: s["band"] for s in read_json(FILES["check_s42"])["statistics"]
    }
    d["check"] = {s: read_json(FILES[f"check_s{s}"]) for s in SEEDS}
    d["check"]["v1"] = read_json(FILES["v1_check"])
    d["per_layer"] = {
        s: compare_layers(read_jsonl(FILES[f"gate_s{s}"]), refs, canonical)["layers"]
        for s in SEEDS
    }
    d["per_layer"]["v1"] = compare_layers(
        read_jsonl(FILES["v1_gate"]), refs, canonical
    )["layers"]
    v1s1 = [r for L in V1S1_LAYERS for r in read_jsonl(FILES[f"v1s1_gate_L{L}"])]
    d["per_layer"]["v1s1"] = compare_layers(v1s1, refs, canonical)["layers"]

    real_loc = {L: loc_entry(FILES["real_locality"], L) for L in GLOBAL_LAYERS}
    v1i = read_jsonl(FILES["v1_intrinsic"])
    v1s1i = read_jsonl(FILES["v1s1_intrinsic"]) + read_jsonl(
        FILES["v1s1_intrinsic_L27"]
    )
    cross = {e["layer"]: e for e in read_json(FILES["cross_book"])["layers"]}
    rows = {}
    for L in GLOBAL_LAYERS:
        r = {
            "real_sb": real_loc[L]["h_l2"]["same_book"],
            "real_key_sb": real_loc[L]["key_ip"]["same_book"],
            "real_lag": real_loc[L]["consecutive_cosine_centred"],
            "cross_lag": cross[L]["cross_book"],
            "chance": None,
        }
        for flag in ("body", "tail"):
            rr = intrinsic_row(v1i, L, flag, "real")
            vv = v1_row(v1i, L, flag)
            r[f"real_{flag}"] = rr
            r[f"v1_{flag}"] = vv
            if L in V1S1_LAYERS:
                r[f"v1s1_{flag}"] = v1_row(v1s1i, L, flag)
        for s in SEEDS:
            loc = loc_entry(FILES[f"locality.json_L{L:02d}_s{s}"], L)
            r["chance"] = loc["chance"]["same_book"]
            ir = read_jsonl(FILES[f"intrinsic.jsonl_L{L:02d}_s{s}"])
            r[s] = {
                "sb": loc["h_l2"]["same_book"],
                "key_sb": loc["key_ip"]["same_book"],
                "lag": loc["consecutive_cosine_centred"],
                "body": intrinsic_row(ir, L, "body", f"v2_s{s}"),
                "tail": intrinsic_row(ir, L, "tail", f"v2_s{s}"),
                "v": d["verdict"]["runs"][f"L{L:02d}_s{s}"],
                "meta": read_json(FILES[f"run_metadata.json_L{L:02d}_s{s}"]),
            }
            # the verdict and the run files must agree
            if abs(r[s]["v"]["same_book"] - r[s]["sb"]) > 1e-9:
                raise SystemExit(f"verdict.json disagrees with locality at L{L} s{s}")
            if abs(r[s]["v"]["body_lid"] - r[s]["body"]["lid_median"]) > 1e-9:
                raise SystemExit(f"verdict.json disagrees with intrinsic at L{L} s{s}")
        rows[L] = r
    d["rows"] = rows
    d["seconds"] = {**run_seconds(FILES["pilot_log"]), **run_seconds(FILES["full_log"])}
    if len(d["seconds"]) != 2 * len(GLOBAL_LAYERS):
        raise SystemExit(
            f"expected 24 DONE lines across the logs, got {len(d['seconds'])}"
        )
    return d


def ax_layers(ax):
    ax.set_xticks(list(GLOBAL_LAYERS))
    ax.set_xlabel("layer")
    ax.grid(alpha=0.3)


def fig_c1(d):
    R, L = d["rows"], list(GLOBAL_LAYERS)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    real = [R[x]["real_sb"] for x in L]
    ax = axs[0]
    ax.fill_between(
        L,
        [v - bj.SAME_BOOK_TOL for v in real],
        [v + bj.SAME_BOOK_TOL for v in real],
        color="0.85",
        label=f"real ± {bj.SAME_BOOK_TOL}",
    )
    ax.plot(L, real, "k-o", label="real")
    ax.axhline(R[L[0]]["chance"], color="0.5", ls=":", label="chance")
    for s in SEEDS:
        ax.plot(
            L,
            [R[x][s]["sb"] for x in L],
            "-s",
            color=COL[f"s{s}"],
            ms=5,
            label=f"v2 seed {s}",
        )
    ax.set_ylabel("share of 10 nearest neighbours (L2) from the same book")
    ax.set_title("Criterion 1: same-book share of h")
    ax.set_ylim(0, 1.05)
    ax_layers(ax)
    ax.legend(fontsize=8, loc="lower right")
    ax = axs[1]
    ax.plot(L, [R[x]["real_key_sb"] for x in L], "k-o", label="real")
    ax.axhline(R[L[0]]["chance"], color="0.5", ls=":", label="chance")
    for s in SEEDS:
        ax.plot(
            L,
            [R[x][s]["key_sb"] for x in L],
            "-s",
            color=COL[f"s{s}"],
            ms=5,
            label=f"v2 seed {s}",
        )
    ax.set_title("Key side: same-book share of nearest keys (inner product)")
    ax.set_ylabel("same-book share")
    ax.set_ylim(0, 1.05)
    ax_layers(ax)
    ax.legend(fontsize=8)
    return png(fig)


def fig_c2(d):
    R, L = d["rows"], list(GLOBAL_LAYERS)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    v1 = [R[x]["v1_body"]["lid_median"] for x in L]
    ax.plot(L, [R[x]["real_body"]["lid_median"] for x in L], "k-o", label="real")
    ax.plot(L, v1, "-o", color=COL["v1"], label="v1 final, seed 42")
    ax.plot(
        L,
        [v - bj.LID_DROP_MAX for v in v1],
        "--",
        color=COL["v1"],
        label=f"v1 seed 42 − {bj.LID_DROP_MAX:g} (criterion 2 floor)",
    )
    ax.plot(
        V1S1_LAYERS,
        [R[x]["v1s1_body"]["lid_median"] for x in V1S1_LAYERS],
        "D",
        color=COL["v1s1"],
        mec="k",
        label="v1 final, seed 1",
    )
    ax.plot(
        V1S1_LAYERS,
        [R[x]["v1s1_body"]["lid_median"] - bj.LID_DROP_MAX for x in V1S1_LAYERS],
        "_",
        color="k",
        ms=14,
        mew=2,
        label=f"v1 seed 1 − {bj.LID_DROP_MAX:g} (seed-1 floor)",
    )
    for s in SEEDS:
        ax.plot(
            L,
            [R[x][s]["body"]["lid_median"] for x in L],
            "-s",
            color=COL[f"s{s}"],
            ms=5,
            label=f"v2 seed {s}",
        )
    ax.set_ylabel("median LID, body rows")
    ax.set_title("Criterion 2: body LID")
    ax_layers(ax)
    ax.legend(fontsize=8)
    return png(fig)


def fig_gate(d):
    P, L = d["per_layer"], list(GLOBAL_LAYERS)
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for ax, st in zip(axs, STATISTICS):
        ref = [P[42][x]["statistics"][st] for x in L]
        ax.fill_between(
            L,
            [r["min"] for r in ref],
            [r["max"] for r in ref],
            color="0.8",
            label="reference, 5 draws (min–max)",
        )
        ax.plot(L, [r["mean"] for r in ref], "k-", lw=1, label="reference mean")
        ax.plot(
            L,
            [P["v1"][x]["statistics"][st]["value"] for x in L],
            "-o",
            color=COL["v1"],
            ms=4,
            label="v1 final, seed 42",
        )
        ax.plot(
            V1S1_LAYERS,
            [P["v1s1"][x]["statistics"][st]["value"] for x in V1S1_LAYERS],
            "D",
            color=COL["v1s1"],
            mec="k",
            ms=5,
            label="v1 final, seed 1",
        )
        for s in SEEDS:
            ax.plot(
                L,
                [P[s][x]["statistics"][st]["value"] for x in L],
                "-s",
                color=COL[f"s{s}"],
                ms=4,
                label=f"v2 seed {s}",
            )
        ax.set_title(LABEL[st], fontsize=10)
        ax_layers(ax)
        ax.tick_params(axis="x", labelsize=7)
    axs[0].legend(fontsize=7)
    return png(fig)


def fig_intrinsic(d):
    R, L = d["rows"], list(GLOBAL_LAYERS)
    panels = [
        ("body", "hubness_skew", "hubness skew, body"),
        ("tail", "hubness_skew", "hubness skew, tail"),
        ("tail", "lid_median", "median LID, tail"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (flag, key, title) in zip(axs, panels):
        ax.plot(L, [R[x][f"real_{flag}"][key] for x in L], "k-o", label="real")
        ax.plot(
            L,
            [R[x][f"v1_{flag}"][key] for x in L],
            "-o",
            color=COL["v1"],
            label="v1 final, seed 42",
        )
        ax.plot(
            V1S1_LAYERS,
            [R[x][f"v1s1_{flag}"][key] for x in V1S1_LAYERS],
            "D",
            color=COL["v1s1"],
            mec="k",
            label="v1 final, seed 1",
        )
        for s in SEEDS:
            ax.plot(
                L,
                [R[x][s][flag][key] for x in L],
                "-s",
                color=COL[f"s{s}"],
                ms=5,
                label=f"v2 seed {s}",
            )
        ax.set_title(title)
        ax_layers(ax)
    axs[0].legend(fontsize=8)
    return png(fig)


def fig_lag(d):
    R = d["rows"]
    fig, axs = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    for ax, L in zip(axs.flat, GLOBAL_LAYERS):
        r = R[L]
        lags = sorted(r["real_lag"], key=int)
        x = [int(k) for k in lags]
        ax.plot(
            x,
            [r["real_lag"][k]["median"] for k in lags],
            "k-o",
            ms=4,
            label="real, same book",
        )
        cl = [k for k in sorted(r["cross_lag"], key=int) if int(k) > 0]
        ax.plot(
            [int(k) for k in cl],
            [r["cross_lag"][k] for k in cl],
            ":o",
            color="0.45",
            ms=4,
            label="real, other book",
        )
        for s in SEEDS:
            ax.plot(
                x,
                [r[s]["lag"][k]["median"] for k in lags],
                "-s",
                color=COL[f"s{s}"],
                ms=3,
                label=f"v2 seed {s}",
            )
        ax.set_xscale("log")
        ax.set_title(f"L{L}", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axs[-1]:
        ax.set_xlabel("lag (tokens)")
    for ax in axs[:, 0]:
        ax.set_ylabel("median cosine (centred h)")
    axs[0, 0].legend(fontsize=7)
    return png(fig)


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def table(head, body):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    b = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body
    )
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def build(d):
    R, L, V = d["rows"], list(GLOBAL_LAYERS), d["verdict"]
    c1_layers = [x for x in L if V["layers"][str(x)]["c1_both"]]
    c2_layers = [x for x in L if V["layers"][str(x)]["c2_both"]]
    c1_miss = [x for x in L if x not in c1_layers]
    gaps = {(x, s): abs(R[x][s]["sb"] - R[x]["real_sb"]) for x in L for s in SEEDS}
    tight = max((g, x, s) for (x, s), g in gaps.items() if x in c1_layers)
    pooled = {
        k: {s["statistic"]: s for s in d["check"][k]["statistics"]}
        for k in (42, 1, "v1")
    }
    n_pass = {k: sum(s["status"] == "pass" for s in pooled[k].values()) for k in pooled}
    secs = d["seconds"]
    total_h = sum(secs.values()) / 3600
    l19 = d["per_layer"]
    l19_rec = {k: l19[k][19]["statistics"]["hnsw_recall_k10_ef176"] for k in (42, 1)}
    hub_body = {
        s: sum(
            R[x][s]["body"]["hubness_skew"] > R[x]["real_body"]["hubness_skew"]
            for x in L
        )
        for s in SEEDS
    }
    tail_hub = {
        s: sum(
            R[x][s]["tail"]["hubness_skew"] > R[x]["real_tail"]["hubness_skew"]
            for x in L
        )
        for s in SEEDS
    }

    def layers_txt(xs):
        return ", ".join(f"L{x}" for x in xs) or "none"

    under = [x for x in L if all(R[x][s]["sb"] < R[x]["real_sb"] for s in SEEDS)]
    over = [x for x in L if all(R[x][s]["sb"] > R[x]["real_sb"] for s in SEEDS)]
    key_gap = max(
        (abs(R[x][s]["key_sb"] - R[x]["real_key_sb"]), x, s) for x in L for s in SEEDS
    )
    key_real_max = max((R[x]["real_key_sb"], x) for x in L)
    lid_below = [
        x
        for x in L
        if all(
            R[x][s]["body"]["lid_median"] < R[x]["v1_body"]["lid_median"] - 1
            for s in SEEDS
        )
    ]
    margin = min(
        (R[x][s]["v"]["body_lid"] - (R[x][s]["v"]["v1_lid"] - bj.LID_DROP_MAX), x, s)
        for x in L
        for s in SEEDS
    )
    deep = [x for x in L if x >= 15]
    real_hub = [R[x]["real_body"]["hubness_skew"] for x in deep]
    v2_hub = [R[x][s]["body"]["hubness_skew"] for x in deep for s in SEEDS]
    v1_hub = [R[x]["v1_body"]["hubness_skew"] for x in deep]
    tail_lid_above = {
        s: sum(
            R[x][s]["tail"]["lid_median"] > R[x]["real_tail"]["lid_median"] for x in L
        )
        for s in SEEDS
    }

    def drop(lag):
        return lag["1"]["median"] - lag["16"]["median"]

    real_drop = sum(drop(R[x]["real_lag"]) for x in deep) / len(deep)
    v2_drop = sum(drop(R[x][s]["lag"]) for x in deep for s in SEEDS) / (2 * len(deep))
    long_above = {
        s: sum(
            R[x][s]["lag"]["4096"]["median"] > R[x]["real_lag"]["4096"]["median"]
            for x in L
        )
        for s in SEEDS
    }
    ivf_cmp = []
    for st in ("ivf_ratio_k128", "ivf_recall_k128_np24"):
        a, b, v = (pooled[k][st]["value"] for k in (42, 1, "v1"))
        ivf_cmp.append(f"{LABEL[st]}: seed 42 {a:.3f}, seed 1 {b:.3f}, v1 {v:.3f}")
        if not abs(a - b) > max(abs(a - v), abs(b - v)):
            raise SystemExit(
                f"{st}: seed spread no longer exceeds the gap to v1; rewrite the gate paragraph"
            )

    claims = [
        (
            "aim (c1 at both seeds on ≥10 layers and c2 at both seeds on all 12)",
            str(V["aim"]).lower(),
            "verdict.json",
        ),
        (
            "layers meeting criterion 1 at both seeds",
            f"{len(c1_layers)} of 12 (miss: {', '.join(f'L{x}' for x in c1_miss) or 'none'})",
            "verdict.json",
        ),
        (
            "layers meeting criterion 2 at both seeds",
            f"{len(c2_layers)} of 12",
            "verdict.json",
        ),
        (
            "smallest criterion-1 margin among passing layers",
            f"L{tight[1]} seed {tight[2]}: gap {tight[0]:.3f} (limit {bj.SAME_BOOK_TOL})",
            "verdict.json + locality/locality.json",
        ),
        (
            "port check against the probe",
            "passed" if d["port"]["passed"] else "FAILED",
            "pilot/port.json",
        ),
        (
            "pooled gate bands passed, v2 seed 42 / seed 1 / v1 final",
            f"{n_pass[42]} / {n_pass[1]} / {n_pass['v1']} of 5",
            "check_s42.json, check_s1.json, ../v1/full/check_v1_final_seed0.json",
        ),
        (
            "L19 HNSW recall@10 ef 176, v2 seed 42 / seed 1 (reference max)",
            f"{l19_rec[42]['value']:.4f} / {l19_rec[1]['value']:.4f} ({l19_rec[42]['max']:.4f})",
            "gate_s*.jsonl via compare_layers",
        ),
        (
            "layers where v2 body hubness exceeds real, seed 42 / seed 1",
            f"{hub_body[42]} / {hub_body[1]} of 12",
            "runs/*/intrinsic.jsonl, ../v1/full/intrinsic.jsonl",
        ),
        (
            "compute, 24 runs (train + sample + score)",
            f"{total_h:.1f} h on one L40, {min(secs.values())}–{max(secs.values())} s per run",
            "runs/logs/*.log DONE lines",
        ),
    ]
    claim_rows = [
        (
            html.escape(c),
            html.escape(v),
            "MEASURED",
            f"<code>{html.escape(f)}</code>",
            "2026-10-28/29",
        )
        for c, v, f in claims
    ]

    vrows = []
    for x in L:
        cells = [f"L{x}", f3(R[x]["real_sb"])]
        for s in SEEDS:
            vv = R[x][s]["v"]
            cells.append(f"{vv['same_book']:.3f} {'✓' if vv['c1'] else '✗'}")
        cells.append(f"{R[x]['real_body']['lid_median']:.2f}")
        for s in SEEDS:
            vv = R[x][s]["v"]
            cells.append(
                f"{vv['body_lid']:.2f} (floor {vv['v1_lid'] - bj.LID_DROP_MAX:.2f}) {'✓' if vv['c2'] else '✗'}"
            )
        vrows.append(cells)

    prow = []
    for st in STATISTICS:
        b = d["bands"][st]
        band = f"{f3(b['min'])} – {f3(b['max'])}"
        prow.append(
            [html.escape(LABEL[st]), band]
            + [
                f"{pooled[k][st]['value']:.4f} {pooled[k][st]['status']}"
                for k in (42, 1, "v1")
            ]
        )

    css = """
:root{--fg:#1d1d1f;--bg:#fff;--muted:#555;--line:#ddd;--code:#f4f4f4}
@media (prefers-color-scheme:dark){:root{--fg:#e8e8e8;--bg:#161616;--muted:#aaa;--line:#444;--code:#262626}
img{background:#fff}}
body{font:15px/1.5 system-ui,sans-serif;color:var(--fg);background:var(--bg);max-width:1180px;margin:0 auto;padding:16px}
h1{font-size:1.5em}h2{margin-top:2em;border-bottom:1px solid var(--line)}
table{border-collapse:collapse;margin:1em 0;font-size:13px;display:block;overflow-x:auto}
td,th{border:1px solid var(--line);padding:4px 8px;text-align:left}code{background:var(--code);padding:0 3px}
img{max-width:100%;height:auto}.muted{color:var(--muted)}
"""
    miss_txt = ", ".join(f"L{x}" for x in c1_miss)
    body = f"""
<h1>scout_kv v2: book latent with a window critic</h1>
<p>v2 meets its aim: <b>aim = {str(V["aim"]).lower()}</b>. Criterion 1 (same-book share of h within
{bj.SAME_BOOK_TOL} of real) holds at both training seeds on {len(c1_layers)} of 12 layers; it misses at {miss_txt}.
Criterion 2 (body LID at least v1's final value minus {bj.LID_DROP_MAX:g}) holds at both seeds on {len(c2_layers)} of 12.
The search gate still fails at both seeds: v2 passes {n_pass[42]} of 5 pooled bands at seed 42 and {n_pass[1]} of 5
at seed 1, against {n_pass["v1"]} of 5 for v1 final.</p>
<p class="muted">Every number on this page was MEASURED on tig-gpu-alt (L40) from commit c5a86a2 on
2026-10-28/29 and computed by <code>make_report.py</code> from the committed files listed in its docstring.</p>

<h2>Claim table</h2>
{table(["claim", "value", "MEASURED / ESTIMATED", "file", "when"], claim_rows)}

<h2>Verdict per layer</h2>
{table(["layer", "real same-book", "v2 s42", "v2 s1", "real body LID", "v2 s42 body LID", "v2 s1 body LID"], vrows)}
<p>v1 seed 1 exists only at L19, L27, L35 and L43; elsewhere the seed-1 run is compared with v1 seed 42,
as the spec states.</p>

<h2>Criterion 1: book clustering</h2>
<img alt="same-book share by layer" src="data:image/png;base64,{fig_c1(d)}">
<p>v2 follows the real rise in same-book share with depth. It clusters less than real at both seeds at
{layers_txt(under)} and more than real at both seeds at {layers_txt(over)}. Only {miss_txt} falls outside the
{bj.SAME_BOOK_TOL} tolerance; the tightest passing case is L{tight[1]} at seed {tight[2]} (gap {tight[0]:.3f}).</p>
<p>On the key side (right; not a criterion) the real share rises to {key_real_max[0]:.3f} at L{key_real_max[1]} and
falls again in the last layers. The largest v2 gap to real is {key_gap[0]:.3f}, at L{key_gap[1]} seed {key_gap[2]}.</p>

<h2>Criterion 2: body LID</h2>
<img alt="body LID by layer" src="data:image/png;base64,{fig_c2(d)}">
<p>v2 body LID is above the floor at every layer and seed; the smallest margin is {margin[0]:.2f} at
L{margin[1]} seed {margin[2]}, which puts that case at the edge of the criterion. v2 is more than 1 below v1 (seed 42) at both seeds at {layers_txt(lid_below)}.</p>

<h2>Gate statistics</h2>
{table(["statistic", "band", "v2 seed 42", "v2 seed 1", "v1 final (seed 42)"], prow)}
<p>Pooled over the 12 layers. v1 seed 1 has only 4 layers, so it has no pooled value. On the two IVF
statistics the v2 seeds are farther apart than either is from v1 seed 42 ({"; ".join(ivf_cmp)}), so the
difference in bands passed between the v2 seeds says more about retrain noise than about v2. This matches the
v1 diagnostic finding that per-layer search statistics are dominated by retrain noise.</p>
<img alt="gate statistics per layer" src="data:image/png;base64,{fig_gate(d)}">
<p>L19 HNSW recall is too easy, as expected from the probe: {l19_rec[42]["value"]:.4f} at seed 42 and
{l19_rec[1]["value"]:.4f} at seed 1, against a reference maximum of {l19_rec[42]["max"]:.4f}. The probe found that
any generated token-to-token relation makes L19 search too easy; v2 keeps that gap.</p>

<h2>Hubness and tail LID</h2>
<img alt="hubness and tail LID" src="data:image/png;base64,{fig_intrinsic(d)}">
<p>Body hubness is the largest remaining gap. From L15 on, real body hubness skew is {min(real_hub):.1f}–{max(real_hub):.1f};
v2 is {min(v2_hub):.1f}–{max(v2_hub):.1f} across both seeds, and v1 final was {min(v1_hub):.1f}–{max(v1_hub):.1f}. v2 is above
real on {hub_body[42]} of 12 layers at seed 42 and {hub_body[1]} at seed 1, so it does not close v1's hubness gap.
Tail hubness is close to real (above it on {tail_hub[42]} and {tail_hub[1]} of 12 layers). Tail LID is above real on
{tail_lid_above[42]} and {tail_lid_above[1]} of 12 layers, by the most in the late layers.</p>

<h2>Consecutive similarity</h2>
<img alt="lag cosine curves" src="data:image/png;base64,{fig_lag(d)}">
<p>Median cosine between token t and t+lag of the same book, on h centred with the pooled mean. The dotted line is
the real cosine between a token and a token of another book at the same offset
(<code>v2-probe/control/cross_book_lag.json</code>), which shows how much of the real curve is shared position
structure rather than book identity.</p>
<p>From L15 on, the real same-book cosine drops by {real_drop:.3f} on average from lag 1 to lag 16; v2 drops by
{v2_drop:.3f}. v2 has the slow book-level component but not the short-range one. At lag 4096 v2 is above real on
{long_above[42]} of 12 layers at seed 42 and {long_above[1]} at seed 1: the latent keeps tokens of one book similar
over longer distances than real does.</p>
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>scout_kv v2 report</title>
<style>{css}</style></head><body>{body}</body></html>
"""


def main():
    d = collect()
    page = build(d)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({len(page)} bytes)")


if __name__ == "__main__":
    main()
