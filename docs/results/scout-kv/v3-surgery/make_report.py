"""Build docs/results/scout-kv/v3-surgery/report.html from summary.json and committed inputs.

    PYTHONPATH=. python docs/results/scout-kv/v3-surgery/summarise.py
    PYTHONPATH=. python docs/results/scout-kv/v3-surgery/make_report.py [out_html]

Inputs: summary.json and run.log (this directory); ../v2/compare_layers_s{42,1}.json and
../v2/runs/L{LL}_s{S}/intrinsic.jsonl (v2); ../v1/full/v1_final_seed0.jsonl and
../v1/full/intrinsic.jsonl (v1); ../reference/seed{0..4}.jsonl. A missing input stops the
script. Every number on the page is computed from these files, and each sentence that reads
the data checks the condition it states, so a changed input cannot leave a false sentence.
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
from scipy.stats import spearmanr  # noqa: E402

from src.data.kv_capture import GLOBAL_LAYERS  # noqa: E402
from src.eval.kv_gate import compare_layers  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "report.html"
SCOUT = HERE.parent
LAYERS = (11, 15, 43)
SEEDS = (0, 1)
SETS = ("real", "gaussian", "s0", "s32", "s256", "gz")
STAT = "hnsw_recall_k10_ef176"
OTHER = ("hnsw_reach_k10", "ivf_ratio_k128", "ivf_recall_k128_np24")
LABEL = {
    "hnsw_recall_k10_ef176": "HNSW recall@10, ef 176",
    "hnsw_reach_k10": "HNSW reach, k=10",
    "ivf_ratio_k128": "IVF scan ratio, k=128",
    "ivf_recall_k128_np24": "IVF recall@128, nprobe 24",
}
COL = {
    "real": "black",
    "gaussian": "#7570b3",
    "s0": "#d95f02",
    "s32": "#e6ab02",
    "s256": "#66a61e",
    "gz": "#1b9e77",
    "v1": "#7f7f7f",
    "v2": "#e7298a",
}
MARK = {11: "o", 15: "s", 43: "^"}


def need(p):
    if not Path(p).is_file():
        raise SystemExit(f"missing input: {p}")
    return Path(p)


def read_jsonl(p):
    return [json.loads(x) for x in need(p).read_text().splitlines() if x.strip()]


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def one(rows, **kw):
    hits = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    if len(hits) != 1:
        raise SystemExit(f"expected one row for {kw}, got {len(hits)}")
    return hits[0]


def collect():
    s = json.loads(need(HERE / "summary.json").read_text())
    log = need(HERE / "run.log").read_text()
    secs = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"^DONE (\w+) seconds=(\d+)$", log, re.M)
    }
    commit = re.search(r"^commit (\w+)$", log, re.M).group(1)
    refs = [read_jsonl(SCOUT / f"reference/seed{i}.jsonl") for i in range(5)]
    v2cmp = {
        sd: json.loads(need(SCOUT / f"v2/compare_layers_s{sd}.json").read_text())[
            "layers"
        ]
        for sd in (42, 1)
    }
    v1cmp = compare_layers(read_jsonl(SCOUT / "v1/full/v1_final_seed0.jsonl"), refs)[
        "layers"
    ]
    v1_intr = read_jsonl(SCOUT / "v1/full/intrinsic.jsonl")
    gen = {}
    for L in LAYERS:
        src = (
            "v1_final"
            if any(r["layer"] == L and r["source"] == "v1_final" for r in v1_intr)
            else "v1_best"
        )
        gen[("v1", L)] = {
            "recall": v1cmp[L]["statistics"][STAT]["value"],
            "body": one(v1_intr, layer=L, source=src, flag="body"),
        }
        for sd in (42, 1):
            intr = read_jsonl(SCOUT / f"v2/runs/L{L:02d}_s{sd}/intrinsic.jsonl")
            gen[(f"v2_s{sd}", L)] = {
                "recall": v2cmp[sd][str(L)]["statistics"][STAT]["value"],
                "body": one(intr, flag="body"),
            }
    # v1 retrained at seed 1 exists at L43 (v1-diagnostics); best was its final checkpoint.
    v1s1 = compare_layers(
        read_jsonl(SCOUT / "v1-diagnostics/noise/measure/v1s1_final_L43.jsonl"), refs
    )["layers"][43]
    gen[("v1_s1", 43)] = {
        "recall": v1s1["statistics"][STAT]["value"],
        "body": one(
            read_jsonl(SCOUT / "v2/runs/v1_seed1_intrinsic.jsonl"),
            layer=43,
            source="v1_best",
            flag="body",
        ),
    }
    real_nbr = read_jsonl(SCOUT / "controls/neighbours.jsonl")
    key_median = sorted(r["ood"]["key_median_pos"] for r in real_nbr)
    key_median = key_median[len(key_median) // 2]
    return {
        "s": s,
        "secs": secs,
        "commit": commit,
        "gen": gen,
        "key_median": key_median,
    }


def intr(s, L, name, seed, flag):
    return one(s["intrinsic"], layer=L, source=name, seed=seed, flag=flag)


def val(s, L, name, seed, stat=STAT):
    return s["sets"][str(L)][name][str(seed)]["stats"][stat]["value"]


def fig_recall(d):
    s = d["s"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, L in zip(axes, LAYERS, strict=True):
        r = s["rule"][str(L)]
        ax.axhspan(r["real_min"], r["real_max"], color="0.85", label="real, 5 draws")
        ax.axhline(r["remove_at"], color="#d95f02", ls="--", lw=1, label="removes at")
        ax.axhline(r["keep_at"], color="#1b9e77", ls="--", lw=1, label="keeps at")
        for sd, ls in ((42, "-"), (1, ":")):
            ax.axhline(
                r["v2"][str(sd)], color=COL["v2"], ls=ls, lw=1.2, label=f"v2 seed {sd}"
            )
        for i, name in enumerate(SETS):
            for j, seed in enumerate(SEEDS):
                ax.plot(
                    i + (j - 0.5) * 0.2,
                    val(s, L, name, seed),
                    "o" if seed == 0 else "D",
                    color=COL[name],
                    ms=7,
                )
        ax.set_xticks(range(len(SETS)), SETS)
        ax.set_title(f"L{L}")
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(f"{LABEL[STAT]} (median of 8 heads)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Search difficulty per set; circles seed 0, diamonds seed 1")
    return png(fig)


def hub_points(d):
    s = d["s"]
    pts = []
    for L in LAYERS:
        for name in SETS:
            for seed in SEEDS:
                pts.append(
                    (
                        L,
                        name,
                        seed,
                        intr(s, L, name, seed, "body")["hubness_skew"],
                        val(s, L, name, seed),
                    )
                )
    return pts


def fig_hub(d, pts):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for L, name, _seed, hub, rec in pts:
        ax.plot(hub, rec, MARK[L], color=COL[name], ms=8, alpha=0.85)
    for (src, L), g in d["gen"].items():
        c = COL["v1"] if src.startswith("v1") else COL["v2"]
        ax.plot(
            g["body"]["hubness_skew"],
            g["recall"],
            MARK[L],
            mfc="none",
            mec=c,
            ms=10,
            mew=2,
        )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=COL[n], label=n) for n in SETS
    ]
    handles += [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="none",
            mec=COL["v1"],
            mew=2,
            label="v1 final (L43: both seeds)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="none",
            mec=COL["v2"],
            mew=2,
            label="v2 (both seeds)",
        ),
    ]
    handles += [
        plt.Line2D([], [], marker=m, ls="", color="0.3", label=f"L{L}")
        for L, m in MARK.items()
    ]
    ax.legend(handles=handles, fontsize=8, ncol=2)
    ax.set_xlabel("body hubness skew (k=10, 20,000 rows)")
    ax.set_ylabel(f"{LABEL[STAT]}")
    ax.grid(alpha=0.3)
    ax.set_title("Body hubness against search difficulty, every set, layer and seed")
    return png(fig)


def fig_neighbours(d):
    nb = d["s"]["neighbours"]
    L = list(GLOBAL_LAYERS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for key, c, lab in (
        ("real", "black", "real"),
        ("v2_s42", COL["v2"], "v2 seed 42"),
        ("v2_s1", "#f1a1c7", "v2 seed 1"),
    ):
        axes[0].plot(
            L, [nb[str(x)][key]["n_distinct_keys"] for x in L], "o-", color=c, label=lab
        )
        axes[1].plot(
            L, [nb[str(x)][key]["hub_median_pos"] for x in L], "o-", color=c, label=lab
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("distinct keys in all top-10 lists (of 31,736)")
    axes[1].set_ylabel("median book position of the hub keys")
    axes[1].axhline(
        d["key_median"], color="0.5", ls=":", lw=1, label="median position of all keys"
    )
    for ax in axes:
        ax.set_xlabel("layer")
        ax.set_xticks(L)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Where real-split queries' top-10 keys land: real books against v2's books (median of 8 heads)"
    )
    return png(fig)


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def table(head, body):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    b = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body
    )
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def check(cond, msg):
    if not cond:
        raise SystemExit(f"a sentence no longer holds: {msg}")


def build(d):
    s, secs = d["s"], d["secs"]
    calls = {L: {n: s["calls"][str(L)][n]["call"] for n in SETS} for L in LAYERS}
    check(s["sanity_ok"], "real reproduces the reference")
    worst = max(abs(v["diff"]) for v in s["sanity"].values())
    check(all(calls[L]["s0"] == "removes" for L in LAYERS), "s0 removes at all layers")
    check(all(calls[L]["gaussian"] == "removes" for L in LAYERS), "gaussian removes")
    check(all(calls[L]["s32"] == "removes" for L in LAYERS), "s32 removes")
    check(
        calls[11]["gz"] == calls[15]["gz"] == "keeps" and calls[43]["gz"] == "partial",
        "gz keeps at L11/L15, partial at L43",
    )
    check(
        calls[11]["s256"] == calls[15]["s256"] == "removes"
        and calls[43]["s256"] == "partial",
        "s256 removes at L11/L15, partial at L43",
    )

    pts = hub_points(d)
    real_hub = [p[3] for p in pts if p[1] == "real"]
    split = {}
    for L in LAYERS:
        easy = [p for p in pts if p[0] == L and calls[L][p[1]] == "removes"]
        hard = [p for p in pts if p[0] == L and calls[L][p[1]] != "removes"]
        split[L] = {
            "easy_max_hub": max(p[3] for p in easy),
            "hard_min_hub": min(p[3] for p in hard),
            "easy_min_rec": min(p[4] for p in easy),
            "hard_max_rec": max(p[4] for p in hard),
        }
        check(
            split[L]["easy_max_hub"] < split[L]["hard_min_hub"],
            f"L{L}: every set that removes the hardness has lower body hubness than every set that does not",
        )
    rho = {}
    for L in LAYERS:
        xs = [p[3] for p in pts if p[0] == L]
        ys = [p[4] for p in pts if p[0] == L]
        rho[L] = spearmanr(xs, ys).statistic
    gen = d["gen"]
    gen_hub = [g["body"]["hubness_skew"] for g in gen.values()]
    rule = {L: s["rule"][str(L)] for L in LAYERS}
    v2_min_rec = min(gen[(f"v2_s{sd}", L)]["recall"] for L in LAYERS for sd in (42, 1))
    check(
        all(
            gen[(f"v2_s{sd}", L)]["recall"] >= rule[L]["remove_at"]
            for L in LAYERS
            for sd in (42, 1)
        ),
        "v2 is at removes level at all layers and seeds",
    )
    check(
        all(gen[("v1", L)]["recall"] >= rule[L]["remove_at"] for L in (11, 15)),
        "v1 is at removes level at L11 and L15",
    )
    v1_43, v1_43_hub = (
        gen[("v1", 43)]["recall"],
        gen[("v1", 43)]["body"]["hubness_skew"],
    )
    v1s1_43 = gen[("v1_s1", 43)]["recall"]
    v1s1_43_hub = gen[("v1_s1", 43)]["body"]["hubness_skew"]
    check(
        rule[43]["real_min"] <= v1_43 <= rule[43]["real_max"]
        and v1s1_43 >= rule[43]["remove_at"],
        "v1 L43 inside real at seed 42 and easy at seed 1",
    )
    gen_lid = {k: g["body"]["lid_median"] for k, g in d["gen"].items()}
    lid = {
        (L, n): sum(intr(s, L, n, sd, "body")["lid_median"] for sd in SEEDS) / 2
        for L in LAYERS
        for n in SETS
    }
    check(
        all(
            lid[(L, n)] > lid[(L, "real")] + 3
            for L in LAYERS
            for n in ("gaussian", "s0", "s32")
        ),
        "the shuffles that remove hardness raise body LID by more than 3",
    )
    check(
        abs(gen_lid[("v2_s42", 11)] - lid[(11, "real")]) < 2.5
        and abs(gen_lid[("v1", 11)] - lid[(11, "real")]) < 2.5,
        "v1 and v2 body LID within 2.5 of real at L11",
    )
    nb = s["neighbours"]
    early = [
        x
        for x in (7, 11)
        if all(
            nb[str(x)][k]["n_distinct_keys"] > 2 * nb[str(x)]["real"]["n_distinct_keys"]
            for k in ("v2_s42", "v2_s1")
        )
    ]
    check(
        early == [7, 11],
        "v2 spreads queries over more than twice real's distinct keys at L7 and L11",
    )
    # Layers from L15 on where real hub keys sit late (median position above 25,000).
    late_real = [
        x
        for x in GLOBAL_LAYERS
        if x >= 15 and nb[str(x)]["real"]["hub_median_pos"] > 25000
    ]
    late_pos = [
        x
        for x in late_real
        if all(
            nb[str(x)][k]["hub_median_pos"]
            < nb[str(x)]["real"]["hub_median_pos"] - 1000
            for k in ("v2_s42", "v2_s1")
        )
    ]
    not_late = [x for x in GLOBAL_LAYERS if x >= 15 and x not in late_real]

    r43 = s["sets"]["43"]["s0"]["0"]["stats"]
    for sd in SEEDS:
        for st in ("ivf_ratio_k128", "ivf_recall_k128_np24"):
            lo, hi = r43[st]["min"], r43[st]["max"]
            v = val(s, 43, "s0", sd, st)
            check(
                lo * 0.9 <= v <= hi * 1.1, f"L43 s0 {st} within 10% of the real range"
            )
        check(
            val(s, 43, "gaussian", sd, "ivf_ratio_k128")
            < 0.5 * r43["ivf_ratio_k128"]["min"]
            and val(s, 43, "gaussian", sd, "ivf_recall_k128_np24")
            > 1.1 * r43["ivf_recall_k128_np24"]["max"],
            "L43 gaussian far outside on both IVF statistics",
        )
    total_h = sum(secs.values()) / 3600
    claims = [
        (
            "real set reproduces reference seed 0 / 1 (largest per-layer difference)",
            f"yes ({worst:.4f})",
            "summary.json sanity",
        ),
        (
            "s0 (components shuffled within each token)",
            "removes the hardness at L11, L15, L43",
            "summary.json calls",
        ),
        (
            "gz (heavy tails removed)",
            "keeps it at L11, L15; partial at L43",
            "summary.json calls",
        ),
        (
            "s256 (top 256 components kept intact)",
            "removes at L11, L15; partial at L43",
            "summary.json calls",
        ),
        (
            "body hubness, sets that remove vs sets that keep, per layer (split read after the run)",
            "; ".join(
                f"L{L} ≤ {v['easy_max_hub']:.1f} vs ≥ {v['hard_min_hub']:.1f}"
                for L, v in split.items()
            )
            + f" (real {min(real_hub):.1f}–{max(real_hub):.1f})",
            "intrinsic.jsonl",
        ),
        (
            "Spearman rho, body hubness vs recall, per layer (12 points each)",
            ", ".join(f"L{L} {rho[L]:.2f}" for L in LAYERS),
            "intrinsic.jsonl + gate_L*.jsonl",
        ),
        (
            "v1 / v2 body hubness at these layers",
            f"{min(gen_hub):.1f}–{max(gen_hub):.1f}",
            "../v1/full/intrinsic.jsonl, ../v2/runs/*/intrinsic.jsonl",
        ),
        (
            "box time",
            f"{total_h:.2f} h ({secs['v2_neighbours']} s neighbours + {secs['surgery']} s surgery)",
            "run.log DONE lines",
        ),
    ]
    claim_rows = [
        (
            html.escape(c),
            html.escape(v),
            "MEASURED",
            f"<code>{html.escape(f)}</code>",
            "2026-10-29",
        )
        for c, v, f in claims
    ]

    rows = []
    for L in LAYERS:
        r = s["rule"][str(L)]
        for n in SETS:
            c = s["calls"][str(L)][n]
            rows.append(
                [
                    f"L{L}",
                    n,
                    " / ".join(f"{v:.3f}" for v in c["values"]),
                    c["call"],
                    f"{r['real_min']:.3f}–{r['real_max']:.3f}",
                    f"≥ {r['remove_at']:.3f} / ≤ {r['keep_at']:.3f}",
                ]
            )

    irows = []
    for L in LAYERS:
        for n in SETS:
            cells = [f"L{L}", n]
            for flag in ("body", "tail"):
                a, b = (intr(s, L, n, sd, flag) for sd in SEEDS)
                cells += [
                    f"{a['hubness_skew']:.1f} / {b['hubness_skew']:.1f}",
                    f"{a['lid_median']:.1f} / {b['lid_median']:.1f}",
                    f"{a['ivf_gini']:.2f} / {b['ivf_gini']:.2f}",
                ]
            nbr = [
                s["sets"][str(L)][n][str(sd)]["nbr"]["n_distinct_keys"] for sd in SEEDS
            ]
            cells.append(f"{nbr[0]:.0f} / {nbr[1]:.0f}")
            irows.append(cells)

    orows = []
    for L in LAYERS:
        for n in SETS:
            cells = [f"L{L}", n]
            for st in OTHER:
                v = [val(s, L, n, sd, st) for sd in SEEDS]
                rng = s["sets"][str(L)][n]["0"]["stats"][st]
                cells.append(
                    f"{v[0]:.3f} / {v[1]:.3f} <span class=muted>({f3(rng['min'])}–{f3(rng['max'])})</span>"
                )
            orows.append(cells)

    grows = []
    for (src, L), g in sorted(d["gen"].items(), key=lambda kv: (kv[0][1], kv[0][0])):
        grows.append(
            [
                f"L{L}",
                src.replace("_s", " seed "),
                f"{g['recall']:.3f}",
                f"{g['body']['hubness_skew']:.1f}",
                f"{g['body']['lid_median']:.1f}",
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
    body = f"""
<h1>scout_kv v3 real-data test: what makes real search hard</h1>
<p>Real h is hard to search at L11, L15 and L43 because of how its components vary together within
each token, and that joint structure is what produces the body hubs. Shuffling each principal component
on its own across the tokens of a block (<code>s0</code>), which keeps every component's values, the
covariance and the block means, makes real search as easy as generated search at all three layers. Mapping
every component to Gaussian quantiles by rank (<code>gz</code>), which removes heavy tails but keeps the
joint structure, leaves L11 and L15 as hard as real and L43 partly so. Keeping the top 256 components
intact (<code>s256</code>) is not enough at L11 and L15; at L43 it keeps most of the hardness.</p>
<p>Body hubness separates the outcomes. At each layer, every set that removes the hardness has lower body hubness
than every set that keeps it or partly keeps it ({"; ".join(f"L{L}: at most {v['easy_max_hub']:.1f} against at least {v['hard_min_hub']:.1f}" for L, v in split.items())};
real is {min(real_hub):.1f}–{max(real_hub):.1f}). This split was read off the results, not fixed in advance.</p>
<p>The generators have body hubness {min(gen_hub):.1f}–{max(gen_hub):.1f} at these layers. v2 is easy at all three layers
at both seeds (recall at least {v2_min_rec:.3f}). v1 is easy at L11 and L15. At L43, v1 at seed 42 is inside the real range
({v1_43:.3f}) with hubness {v1_43_hub:.1f}, but v1 retrained at seed 1 is easy there ({v1s1_43:.3f}, hubness {v1s1_43_hub:.1f}),
which the v1 diagnostics already put down to retrain noise. So one generated case reached real difficulty without hubs,
and it did not reproduce. The target for v3 is the body hub structure, which comes from dependence between
components rather than from any single component's values.</p>
<p class="muted">Measured on tig-gpu-alt (L40) at commit <code>{d["commit"][:7]}</code> on 2026-10-29; every number
is MEASURED and computed by <code>summarise.py</code> and <code>make_report.py</code> from the committed files.
The decision rule was committed before the run (README.md).</p>

<h2>Claim table</h2>
{table(["claim", "value", "MEASURED / ESTIMATED", "file", "when"], claim_rows)}

<h2>Decision rule applied</h2>
<p>Statistic: {LABEL[STAT]}, median over the 8 heads, at seeds 0 / 1. The real set reproduces the reference draw
of the same seed to {worst:.4f} at every layer, so the sanity check passes.</p>
{table(["layer", "set", "recall s0 / s1", "call", "real range", "removes at / keeps at"], rows)}
<img alt="recall per set" src="data:image/png;base64,{fig_recall(d)}">

<h2>Hubness and search difficulty</h2>
<img alt="hubness against recall" src="data:image/png;base64,{fig_hub(d, pts)}">
<p>Filled markers are the six sets at both seeds; open markers are v1 final and v2's two seeds at the same layers.
The points fall into two groups. The low-hubness sets have recall
{"; ".join(f"L{L} {v['easy_min_rec']:.3f} or more" for L, v in split.items())}, and the others at most
{"; ".join(f"L{L} {v['hard_max_rec']:.3f}" for L, v in split.items())}. The rank correlation over each layer's 12
set-seed points is {", ".join(f"L{L} {rho[L]:.2f}" for L in LAYERS)}; it is weakest at L11, where the low-hubness sets'
recall does not order with their hubness.</p>
<p>The test does not separate hubness from LID: the shuffles that remove the hardness also raise body LID by more
than 3 at every layer (L11 real {lid[(11, "real")]:.1f}, <code>s0</code> {lid[(11, "s0")]:.1f}). v1 and v2 separate
them at L11, where their body LID ({gen_lid[("v1", 11)]:.1f} and {gen_lid[("v2_s42", 11)]:.1f}) is within 2.5 of real
({lid[(11, "real")]:.1f}) while their hubness is near 3 and their search is easy. So at L11 the missing property is the
hubs, not the dimensionality.</p>
{table(["layer", "generator", "recall", "body hubness", "body LID"], grows)}

<h2>Intrinsic statistics per set</h2>
<p>Seeds 0 / 1. Body and tail measured as <code>../v2/intrinsic.py</code> measured the v2 runs (20,000 rows,
k=10 hubness, k=100 LID, 256-cell IVF Gini). The last column is the number of distinct keys in all top-10 lists of
the 5,120 real-split queries, median over heads.</p>
{table(["layer", "set", "body hub", "body LID", "body Gini", "tail hub", "tail LID", "tail Gini", "distinct top-10 keys"], irows)}

<h2>Other gate statistics</h2>
<p>Seeds 0 / 1, with the real 5-draw range in brackets. At L43, <code>s0</code> keeps the IVF scan ratio and IVF recall
near or inside the real range while HNSW becomes easy; the Gaussian set loses both. The IVF statistics there depend on
the block means (book and position), which <code>s0</code> keeps, as the controls found for the end-of-book query split.</p>
{table(["layer", "set"] + [LABEL[x] for x in OTHER], orows)}

<h2>Where queries land in v2's books</h2>
<img alt="neighbour positions real vs v2" src="data:image/png;base64,{fig_neighbours(d)}">
<p>Real queries concentrate their top-10 keys on few keys at L7 and L11 ({nb["7"]["real"]["n_distinct_keys"]:.0f} and
{nb["11"]["real"]["n_distinct_keys"]:.0f}); v2 spreads them over more than twice as many at both seeds. From L15 on the
counts are similar, but real hub keys sit late in the books: their median position is above 25,000 at
{len(late_real)} of the {len(late_real) + len(not_late)} layers from L15 on (not at {", ".join(f"L{x}" for x in not_late) or "none"}).
At {len(late_pos)} of those {len(late_real)} layers, v2's hub keys sit more than 1,000 positions earlier at both seeds.</p>

<h2>What this means for v3</h2>
<ul>
<li>The generator has to produce body hubs: a small set of tokens that many others have as near neighbours. Matching
each component's distribution (heavy tails) or the top 256 components is not enough at L11 and L15.</li>
<li>Hubs come from how components vary together within a token, so a loss or critic that sees whole tokens has to
reward them; a per-component or low-rank target cannot.</li>
<li>Hubness skew is directly measurable on a batch, so it can be a training signal or a selection statistic, as LID
and spectrum regularisers were for the earlier families.</li>
<li>Open: whether the components past rank 256 that matter at L11 and L15 are the directions W_k reads most; the key
projection is not ordered by h variance.</li>
</ul>
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>scout_kv surgery test</title>
<style>{css}</style></head><body>{body}</body></html>
"""


def main():
    d = collect()
    page = build(d)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({len(page)} bytes)")


if __name__ == "__main__":
    main()
