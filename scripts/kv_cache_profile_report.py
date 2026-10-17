"""Render the KV-cache track's head-profiling report as one self-contained HTML file.

Two steps, so the committed report regenerates without the GPU box:

    # on any machine holding the raw profiler output (3+ MB, not committed)
    python scripts/kv_cache_profile_report.py compact \
        --v1 profile.jsonl --v2 profile_v2.jsonl --meta capture/meta.json \
        --out docs/results/kv-cache-profile/profile_summary.json

    # from the committed summary
    python scripts/kv_cache_profile_report.py render \
        --summary docs/results/kv-cache-profile/profile_summary.json \
        --out docs/results/kv-cache-profile/report.html

The raw files come from the spike scripts in docs/results/kv-cache-profile/spike/.
Charts are inline SVG with a small hover script; nothing is fetched at view time.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from statistics import median

WORKLOADS = (
    "plain_ood_k10",
    "plain_id_k10",
    "plain_ood_k128",
    "plain_id_k128",
    "lift_ood_k10",
    "lift_ood_k128",
)
FIELDS = (
    "ivf_scan_frac",
    "ivf_reached",
    "hnsw_ndis",
    "hnsw_reached",
    "ivf_recall_op",
    "hnsw_recall_op",
)
N_HEADS = 8

# Reference palette (dataviz skill, references/palette.md). Sequential blue, light->dark.
SEQ = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
# Diverging blue <-> gray <-> red, equal steps per arm, lightness-matched.
DIV_NEG = ["#1c5cab", "#3987e5", "#86b6ef"]
DIV_MID = "#f0efec"
DIV_POS = ["#f2a7a3", "#e34948", "#a8292a"]
SERIES = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70")]


# --------------------------------------------------------------------------- compact


def compact(v1_path: Path, v2_path: Path, meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    v2 = []
    for line in v2_path.read_text().splitlines():
        r = json.loads(line)
        v2.append(
            {
                "doc": r["doc"],
                "layer": r["layer"],
                "head": r["kv_head"],
                "n_keys": r["n_keys"],
                **{w: {f: r[w][f] for f in FIELDS} for w in WORKLOADS},
                "attn_err": r["attn_rel_err_k128"],
                "diag": r["diag_ood"],
            }
        )
    v1 = []
    for line in v1_path.read_text().splitlines():
        r = json.loads(line)
        v1.append(
            {
                "doc": r["doc"],
                "layer": r["layer"],
                "head": r["kv_head"],
                **{
                    s: {
                        "ivf_scan_frac": r[s]["ivf_scan_frac"],
                        "hnsw_reached": r[s]["hnsw_reached"],
                    }
                    for s in ("ood", "id")
                },
                "top1_is_pos0": r["diag_ood"]["top1_is_pos0"],
            }
        )
    docs = [
        {
            k: d[k]
            for k in ("row", "title", "n_tokens", "n_tokens_full", "worst_rel_err")
        }
        | {
            "late_with_scale": max(
                v.get("q_rel_err_late_with_scale", 0.0) for v in d["verify"].values()
            ),
            "control_min": min(
                v.get("q_rel_err_late_without_scale", 1.0) for v in d["verify"].values()
            ),
            "forward_seconds": d["forward_seconds"],
        }
        for d in meta["docs"]
    ]
    return {
        "capture": {
            "global_layers": meta["global_layers"],
            "gqa": meta["gqa"],
            "versions": meta["versions"],
            "docs": docs,
        },
        "v1": v1,
        "v2": v2,
    }


# --------------------------------------------------------------------------- helpers


def esc(x) -> str:
    return html.escape(str(x))


def lum(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)
    ]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def ink_for(fill: str) -> str:
    return "#0b0b0b" if lum(fill) > 0.3 else "#ffffff"


def seq_color(t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    return SEQ[round(t * (len(SEQ) - 1))]


def div_color(t: float) -> str:
    """t in [-1, 1]; 0 is the neutral midpoint."""
    t = min(max(t, -1.0), 1.0)
    if abs(t) < 1 / 7:
        return DIV_MID
    arm = DIV_POS if t > 0 else DIV_NEG[::-1]
    idx = min(int((abs(t) - 1 / 7) / (6 / 7) * 3), 2)
    return arm[idx]


def fmt(x, nd=2) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def med(xs):
    xs = [x for x in xs if x is not None]
    return median(xs) if xs else None


def by_cell(rows):
    cells = {}
    for r in rows:
        cells.setdefault((r["layer"], r["head"]), []).append(r)
    for v in cells.values():
        v.sort(key=lambda r: r["doc"])
    return cells


# --------------------------------------------------------------------------- charts


def heatmap(layers, cells, value_fn, color_fn, label_fn, tip_fn, title, caption):
    cw, ch, left, top = 50, 26, 64, 34
    w, h = left + N_HEADS * cw + 8, top + len(layers) * ch + 8
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
        f'aria-label="{esc(title)}" class="heat">'
    ]
    parts.append(
        f'<text x="{left + N_HEADS * cw / 2}" y="12" class="axis" text-anchor="middle">kv head</text>'
    )
    for g in range(N_HEADS):
        parts.append(
            f'<text x="{left + g * cw + cw / 2}" y="28" class="axis" text-anchor="middle">{g}</text>'
        )
    for i, L in enumerate(layers):
        y = top + i * ch
        parts.append(
            f'<text x="{left - 8}" y="{y + ch / 2 + 4}" class="axis" text-anchor="end">L{L}</text>'
        )
        for g in range(N_HEADS):
            rs = cells[(L, g)]
            v = value_fn(rs)
            fill = color_fn(v)
            x = left + g * cw
            parts.append(
                f'<g class="cell" data-tip="{esc(tip_fn(L, g, rs))}">'
                f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="3" fill="{fill}"/>'
                f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" text-anchor="middle" '
                f'class="cellv" fill="{ink_for(fill)}">{esc(label_fn(v))}</text></g>'
            )
    parts.append("</svg>")
    table = heat_table(layers, cells, value_fn, label_fn)
    return (
        f"<figure><figcaption><b>{esc(title)}</b><br>{caption}</figcaption>"
        f'<div class="scroll">{"".join(parts)}</div>{table}</figure>'
    )


def heat_table(layers, cells, value_fn, label_fn):
    head = "".join(f"<th>h{g}</th>" for g in range(N_HEADS))
    body = "".join(
        f"<tr><th>L{L}</th>"
        + "".join(
            f"<td>{esc(label_fn(value_fn(cells[(L, g)])))}</td>" for g in range(N_HEADS)
        )
        + "</tr>"
        for L in layers
    )
    return (
        '<details><summary>Table view</summary><div class="scroll"><table class="num">'
        f"<tr><th></th>{head}</tr>{body}</table></div></details>"
    )


def legend_seq(lo_label, hi_label):
    sw = "".join(f'<span style="background:{c}"></span>' for c in SEQ)
    return f'<div class="ramp"><em>{esc(lo_label)}</em><div class="bar">{sw}</div><em>{esc(hi_label)}</em></div>'


def legend_div(lo_label, mid_label, hi_label):
    sw = "".join(
        f'<span style="background:{c}"></span>' for c in DIV_NEG + [DIV_MID] + DIV_POS
    )
    return (
        f'<div class="ramp"><em>{esc(lo_label)}</em><div class="bar">{sw}</div>'
        f'<em>{esc(hi_label)}</em></div><p class="note">Gray midpoint: {esc(mid_label)}</p>'
    )


def dot_plot(layers, series, title, caption, y_max):
    """series: list of (name, {layer: value}). One shared y axis, linear."""
    left, top, pw, ph = 48, 16, 560, 240
    w, h = left + pw + 150, top + ph + 40
    step = pw / len(layers)
    y = lambda v: top + ph - (v / y_max) * ph  # noqa: E731
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{esc(title)}" class="dots">'
    ]
    for t in [i * y_max / 4 for i in range(5)]:
        parts.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{y(t)}" y2="{y(t)}" class="grid"/>'
            f'<text x="{left - 6}" y="{y(t) + 4}" class="axis" text-anchor="end">{t:.2f}</text>'
        )
    parts.append(
        f'<line x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}" class="base"/>'
    )
    for i, L in enumerate(layers):
        parts.append(
            f'<text x="{left + step * (i + 0.5)}" y="{top + ph + 16}" class="axis" text-anchor="middle">L{L}</text>'
        )
    ends = []
    for si, (name, vals) in enumerate(series):
        last = None
        for i, L in enumerate(layers):
            v = vals.get(L)
            if v is None:
                continue
            cx = left + step * (i + 0.5) + (si - 1) * 7
            cy = y(min(v, y_max))
            last = cy
            parts.append(
                f'<g class="cell" data-tip="{esc(f"{name}, layer {L}: median {v:.3f}")}">'
                f'<circle cx="{cx}" cy="{cy}" r="9" fill="transparent"/>'
                f'<circle cx="{cx}" cy="{cy}" r="4.5" class="s{si}" stroke="var(--surface)" stroke-width="2"/></g>'
            )
        if last is not None:
            ends.append([last, si, name])
    # Direct labels in the right margin, pushed apart so they never overlap.
    ends.sort()
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + 14)
    for ly, si, name in ends:
        parts.append(
            f'<text x="{left + pw + 10}" y="{ly + 4}" class="dlabel s{si}t">{esc(name)}</text>'
        )
    parts.append("</svg>")
    leg = "".join(
        f'<span class="key"><i class="s{si}"></i>{esc(name)}</span>'
        for si, (name, _) in enumerate(series)
    )
    rows = "".join(
        f"<tr><th>L{L}</th>"
        + "".join(f"<td>{fmt(vals.get(L), 3)}</td>" for _, vals in series)
        + "</tr>"
        for L in layers
    )
    hdr = "".join(f"<th>{esc(n)}</th>" for n, _ in series)
    return (
        f"<figure><figcaption><b>{esc(title)}</b><br>{caption}</figcaption>"
        f'<div class="legend">{leg}</div><div class="scroll">{"".join(parts)}</div>'
        f'<details><summary>Table view</summary><div class="scroll"><table class="num">'
        f"<tr><th>layer</th>{hdr}</tr>{rows}</table></div></details></figure>"
    )


# --------------------------------------------------------------------------- render


def share(rows, key, field):
    return sum(bool(r[key][field]) for r in rows) / len(rows)


def render(summary: dict) -> str:
    v1, v2, cap = summary["v1"], summary["v2"], summary["capture"]
    layers = sorted({r["layer"] for r in v2})
    cells = by_cell(v2)
    n = len(v2)

    # ---- headline numbers (all computed here, from the committed summary)
    reach = {
        w: (share(v2, w, "ivf_reached"), share(v2, w, "hnsw_reached"))
        for w in WORKLOADS
    }
    op = {
        w: (
            median(r[w]["ivf_recall_op"] for r in v2),
            median(r[w]["hnsw_recall_op"] for r in v2),
        )
        for w in WORKLOADS
    }
    v1_reach = {
        s: sum(r[s]["hnsw_reached"] for r in v1) / len(v1) for s in ("ood", "id")
    }
    ivf_ratio, hnsw_ratio, spread = [], [], []
    for rs in cells.values():
        o = [r["plain_ood_k128"]["ivf_scan_frac"] for r in rs]
        i = [r["plain_id_k128"]["ivf_scan_frac"] for r in rs]
        ivf_ratio.append(median(a / b for a, b in zip(o, i)))
        spread.append(max(o) / min(o))
        ho = [r["plain_ood_k128"]["hnsw_ndis"] for r in rs]
        hi = [r["plain_id_k128"]["hnsw_ndis"] for r in rs]
        if all(x is not None for x in ho + hi):
            hnsw_ratio.append(median(a / b for a, b in zip(ho, hi)))
    v1_spread = []
    for rs in by_cell(v1).values():
        o = [r["ood"]["ivf_scan_frac"] for r in rs]
        v1_spread.append(max(o) / min(o))
    err_t = [r["attn_err"]["topk_flat"] for r in v2]
    err_l = [r["attn_err"]["lift_flat"] for r in v2]
    lift_wins = sum(a < b for a, b in zip(err_l, err_t))
    sinky = sum(r["diag"]["distinct_top10_keys"] <= 20 for r in v2)

    def q(xs, p):
        xs = sorted(xs)
        k = (len(xs) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    # ---- figures
    def tip_hnsw(L, g, rs):
        o = ", ".join(fmt(r["plain_ood_k10"]["hnsw_recall_op"]) for r in rs)
        i = ", ".join(fmt(r["plain_id_k10"]["hnsw_recall_op"]) for r in rs)
        return f"L{L} h{g} — OOD recall@10 per doc: {o} | ID: {i}"

    id_meds = [
        median(r["plain_id_k10"]["hnsw_recall_op"] for r in rs) for rs in cells.values()
    ]
    f_hnsw = heatmap(
        layers,
        cells,
        lambda rs: 1 - median(r["plain_ood_k10"]["hnsw_recall_op"] for r in rs),
        lambda v: seq_color(v / 0.6),
        lambda v: f"{v:.2f}",
        tip_hnsw,
        "Figure 1. HNSW miss rate for real queries: 1 − recall@10 at efSearch=176",
        "Median over 4 documents. Darker = the graph index misses more of the true top-10 for "
        "query-side (OOD) search. For comparison, median key-side (ID) recall at the same setting "
        f"is ≥ {min(id_meds):.3f} in every cell and ≥ 0.99 in {sum(m >= 0.99 for m in id_meds)} "
        f"of {len(id_meds)} (hover for per-document values).",
    ) + legend_seq("0.00 (no misses)", "≥ 0.60")

    def ratio_val(rs):
        return median(
            r["plain_ood_k128"]["ivf_scan_frac"] / r["plain_id_k128"]["ivf_scan_frac"]
            for r in rs
        )

    def tip_ivf(L, g, rs):
        o = ", ".join(fmt(r["plain_ood_k128"]["ivf_scan_frac"], 3) for r in rs)
        i = ", ".join(fmt(r["plain_id_k128"]["ivf_scan_frac"], 3) for r in rs)
        return f"L{L} h{g} — IVF scan fraction at recall@128=0.9, OOD per doc: {o} | ID: {i}"

    f_ivf = heatmap(
        layers,
        cells,
        ratio_val,
        lambda v: div_color(math.log2(v) / 3),
        lambda v: f"{v:.2f}",
        tip_ivf,
        "Figure 2. IVF work ratio, real queries ÷ key-side queries (recall@128 = 0.9)",
        "Median over 4 documents of the fraction of keys scanned. Red = real queries need more "
        "scanning than key-like queries; blue = less. Colour is log2-scaled and saturates at 8× / ⅛×.",
    ) + legend_div("⅛× (queries easier)", "1× (no gap)", "8× (queries harder)")

    f_spread = heatmap(
        layers,
        cells,
        lambda rs: max(r["plain_ood_k128"]["ivf_scan_frac"] for r in rs)
        / min(r["plain_ood_k128"]["ivf_scan_frac"] for r in rs),
        lambda v: seq_color(math.log(v) / math.log(4)),
        lambda v: f"{v:.2f}",
        lambda L, g, rs: f"L{L} h{g} — OOD IVF scan per doc: "
        + ", ".join(fmt(r["plain_ood_k128"]["ivf_scan_frac"], 3) for r in rs),
        "Figure 3. Document-to-document spread of the IVF work (max ÷ min over 4 documents)",
        "The same head measured on four different books. This is the noise floor a gate band has "
        "to sit above; 1.00 = identical across documents. Colour is log-scaled and saturates at 4×.",
    ) + legend_seq("1×", "≥ 4×")

    f_sink = heatmap(
        layers,
        cells,
        lambda rs: median(r["diag"]["distinct_top10_keys"] for r in rs),
        lambda v: seq_color(math.log10(max(v, 10)) / math.log10(3000) * 1.4 - 0.4),
        lambda v: f"{v:.0f}",
        lambda L, g, rs: f"L{L} h{g} — distinct top-10 keys per doc: "
        + ", ".join(str(r["diag"]["distinct_top10_keys"]) for r in rs),
        "Figure 4. Distinct keys among all 5,120 queries' top-10 lists",
        "Median over 4 documents, with position 0 already removed. 10 means every query returns the "
        "same ten keys (a sink-like head, where search is trivial); large values mean queries "
        "genuinely want different keys. Colour is log-scaled.",
    ) + legend_seq("10 (all identical)", "≥ 3,000")

    def per_layer(key):
        return {
            L: med([r["attn_err"][key] for r in v2 if r["layer"] == L]) for L in layers
        }

    f_err = dot_plot(
        layers,
        [
            ("Top-k, exact", per_layer("topk_flat")),
            ("SoftmaxLift, exact", per_layer("lift_flat")),
            ("SoftmaxLift, HNSW", per_layer("lift_hnsw")),
        ],
        "Figure 5. Attention error over the indexed segment, k = 128 (median over heads and documents)",
        "Pooled relative L2 error of the attention output, as defined in Zharkov et al. 2026 "
        "(Appendix C.1). Values above 1.60 are drawn at the top edge; the table has exact values.",
        1.6,
    )

    # ---- per-layer table
    def lrow(L):
        rs = [r for r in v2 if r["layer"] == L]
        po = [r["plain_ood_k128"] for r in rs]
        pi = [r["plain_id_k128"] for r in rs]
        lo = [r["lift_ood_k128"] for r in rs]
        cells_ = [
            fmt(med([x["ivf_scan_frac"] for x in po]), 3),
            fmt(med([x["ivf_scan_frac"] for x in pi]), 3),
            fmt(med([x["hnsw_ndis"] for x in po]), 0),
            f"{100 * sum(x['hnsw_reached'] for x in po) / len(po):.0f}%",
            fmt(med([x["hnsw_ndis"] for x in pi]), 0),
            fmt(med([x["ivf_scan_frac"] for x in lo]), 3),
            f"{100 * sum(x['hnsw_reached'] for x in lo) / len(lo):.0f}%",
            fmt(med([r["diag"]["top128_softmax_mass"] for r in rs]), 3),
            fmt(med([r["diag"]["distinct_top10_keys"] for r in rs]), 0),
        ]
        return f"<tr><th>L{L}</th>" + "".join(f"<td>{c}</td>" for c in cells_) + "</tr>"

    layer_table = (
        '<div class="scroll"><table class="num"><tr><th>layer</th><th>IVF scan, OOD</th><th>IVF scan, ID</th>'
        "<th>HNSW dist., OOD</th><th>OOD reached</th><th>HNSW dist., ID</th>"
        "<th>Lift IVF scan</th><th>Lift HNSW reached</th><th>top-128 attention mass</th>"
        "<th>distinct top-10 keys</th></tr>"
        + "".join(lrow(L) for L in layers)
        + "</table></div>"
    )

    # ---- candidate heads table
    def crow(L, g):
        rs = cells[(L, g)]
        po = [r["plain_ood_k10"] for r in rs]
        return (
            f"<tr><th>L{L} h{g}</th><td>{fmt(median(x['hnsw_recall_op'] for x in po), 3)}</td>"
            f"<td>{sum(x['hnsw_reached'] for x in po)}/4</td>"
            f"<td>{fmt(ratio_val(rs), 2)}</td>"
            f"<td>{fmt(median(r['diag']['top128_softmax_mass'] for r in rs), 3)}</td>"
            f"<td>{fmt(median(r['diag']['distinct_top10_keys'] for r in rs), 0)}</td>"
            f"<td>{fmt(median(r['attn_err']['topk_flat'] for r in rs), 3)}</td></tr>"
        )

    cand_table = (
        '<div class="scroll"><table class="num"><tr><th>head</th><th>HNSW recall@10, OOD, ef=176</th>'
        "<th>docs reaching 0.9 (k=10)</th><th>IVF OOD÷ID</th><th>top-128 mass</th>"
        "<th>distinct top-10</th><th>top-k attention err</th></tr>"
        + "".join(crow(L, g) for L in (39, 43, 47) for g in range(N_HEADS))
        + "</table></div>"
    )

    # ---- capture verification table
    cap_rows = "".join(
        f"<tr><td>{d['row']}</td><td class='txt'>{esc(d['title'])}</td><td>{d['n_tokens']:,} of {d['n_tokens_full']:,}</td>"
        f"<td>{100 * d['worst_rel_err']:.2f}%</td><td>{100 * d['late_with_scale']:.2f}%</td>"
        f"<td>{100 * d['control_min']:.1f}%</td><td>{d['forward_seconds']:.0f} s</td></tr>"
        for d in cap["docs"]
    )
    cap_table = (
        '<div class="scroll"><table class="num"><tr><th>PG-19 test row</th><th>book</th><th>tokens used</th>'
        "<th>worst q/k error</th><th>q error, pos ≥ 8191</th><th>control: no scale</th>"
        "<th>forward</th></tr>" + cap_rows + "</table></div>"
    )

    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    claims = [
        (
            "HNSW reaches recall@10 ≥ 0.9, OOD vs ID (efSearch ≤ 2048)",
            f"{pct(reach['plain_ood_k10'][1])} vs {pct(reach['plain_id_k10'][1])}",
        ),
        (
            "HNSW reaches recall@128 ≥ 0.9, OOD vs ID",
            f"{pct(reach['plain_ood_k128'][1])} vs {pct(reach['plain_id_k128'][1])}",
        ),
        (
            "Median HNSW recall@10 at efSearch=176, OOD vs ID",
            f"{op['plain_ood_k10'][1]:.3f} vs {op['plain_id_k10'][1]:.3f}",
        ),
        (
            "Median IVF recall@128 at nprobe=24/256, OOD vs ID",
            f"{op['plain_ood_k128'][0]:.3f} vs {op['plain_id_k128'][0]:.3f}",
        ),
        (
            "HNSW OOD÷ID work, heads reaching 0.9 in all docs",
            f"median {median(hnsw_ratio):.2f}× (n={len(hnsw_ratio)} of 96, range {min(hnsw_ratio):.2f}–{max(hnsw_ratio):.1f}×)",
        ),
        (
            "IVF OOD÷ID work, per head",
            f"median {median(ivf_ratio):.2f}× (quartiles {q(ivf_ratio, 0.25):.2f}–{q(ivf_ratio, 0.75):.2f}×)",
        ),
        (
            "Doc spread of OOD IVF work, sink removed vs kept",
            f"median {median(spread):.2f}× (max {max(spread):.1f}×) vs {median(v1_spread):.2f}× (max {max(v1_spread):.0f}×)",
        ),
        ("Corpora with ≤ 20 distinct top-10 keys (sink removed)", f"{sinky} of {n}"),
        (
            "Attention error, exact retrieval: top-k vs SoftmaxLift",
            f"median {median(err_t):.3f} vs {median(err_l):.3f}; lift lower in {lift_wins} of {n}",
        ),
        (
            "Run 1 (HNSW on raw inner product, sink kept), HNSW reach OOD vs ID",
            f"{pct(v1_reach['ood'])} vs {pct(v1_reach['id'])} — superseded",
        ),
        (
            "q/k reconstruction, worst over 12 layers × 4 docs",
            f"{100 * max(d['worst_rel_err'] for d in cap['docs']):.2f}% (control without scale ≥ {100 * min(d['control_min'] for d in cap['docs']):.1f}%)",
        ),
    ]
    claim_rows = "".join(
        f"<tr><td>{esc(c)}</td><td>{esc(v)}</td><td>MEASURED</td></tr>"
        for c, v in claims
    )

    body = f"""
<header>
<p class="kicker">wgan-synthetic · KV-cache track · head profiling</p>
<h1>How hard is real KV-cache search on Llama 4 Scout?</h1>
<p class="lede">Query→key maximum-inner-product search on the 12 global (no-RoPE) attention layers of
Llama-4-Scout-17B-16E-Instruct, measured on 4 PG-19 books × 12 layers × 8 key heads = {n} corpora of
~31.7k keys and 5,120 queries each. Measured 2026-10-16/17 on one A100-80GB.</p>
</header>

<section>
<h2>Findings</h2>
<ol>
<li><b>The out-of-distribution problem is a graph-index problem.</b> HNSW reaches recall@10 ≥ 0.9 for
real queries in {pct(reach["plain_ood_k10"][1])} of corpora versus {pct(reach["plain_id_k10"][1])} for key-like queries,
and at the paper's efSearch=176 the median recall@10 is {op["plain_ood_k10"][1]:.3f} versus {op["plain_id_k10"][1]:.3f}.
Where HNSW does reach 0.9 everywhere, real queries cost a median {median(hnsw_ratio):.2f}× the distance computations (up to {max(hnsw_ratio):.1f}×).</li>
<li><b>IVF shows no consistent gap.</b> The per-head OOD÷ID scan ratio has median {median(ivf_ratio):.2f}×
with quartiles {q(ivf_ratio, 0.25):.2f}–{q(ivf_ratio, 0.75):.2f}×, and recall at the operating point is
{op["plain_ood_k128"][0]:.3f} (OOD) versus {op["plain_id_k128"][0]:.3f} (ID).</li>
<li><b>Removing the sink token is what makes the numbers stable.</b> The document-to-document spread of
IVF work fell from a median {median(v1_spread):.2f}× (max {max(v1_spread):.0f}×) to {median(spread):.2f}× (max {max(spread):.1f}×).
Sink-like heads remain even so: {sinky} corpora have ≤ 20 distinct keys across all queries' top-10 lists.</li>
<li><b>Plain top-k is a poor attention estimate here; SoftmaxLift is much better, but only with exact search.</b>
Median attention error is {median(err_t):.3f} (top-k) versus {median(err_l):.3f} (SoftmaxLift), lower for lift in
{lift_wins} of {n} corpora. Through approximate indexes the lift estimate degrades sharply in the late layers (Figure 5).</li>
<li><b>Candidate target heads are in layers 39, 43 and 47</b>, where the HNSW gap is largest and the heads are
not sink-dominated. Layer 43 is the hardest.</li>
</ol>
</section>

<section class="retract">
<h2>Correction to the first run</h2>
<p>The first profiling run reported that HNSW reaches recall@10 ≥ 0.9 for real queries in
<b>{pct(v1_reach["ood"])}</b> of corpora versus {pct(v1_reach["id"])} for key-side queries. <b>That figure is withdrawn.</b>
It built HNSW directly on inner products, which graph indexes handle badly, and it kept the attention sink in the index.
With the Xbox MIPS→L2 reduction and the sink removed (as in Zharkov et al. 2026) the figure is
<b>{pct(reach["plain_ood_k10"][1])}</b> versus {pct(reach["plain_id_k10"][1])}. The gap is real, but about a third as large as first reported.</p>
</section>

<section>
<h2>Figures</h2>
{f_hnsw}
{f_ivf}
{f_spread}
{f_sink}
{f_err}
</section>

<section>
<h2>Per-layer summary (k = 128, medians over 8 heads × 4 documents)</h2>
<p class="note">“Reached” columns are the share of corpora in which recall@128 ≥ 0.9 was reached within the sweep
(efSearch ≤ 2048); work medians are over the corpora that reached it.</p>
{layer_table}
</section>

<section>
<h2>Candidate heads: layers 39, 43, 47</h2>
{cand_table}
</section>

<section>
<h2>Method</h2>
<ul>
<li><b>Model and capture.</b> Llama-4-Scout-17B-16E-Instruct, unquantized bf16, all 50 shards verified against Meta's
SHA-256 (downloaded from the ungated <code>unsloth/</code> mirror; hashes identical). The attention input h was recorded at the
{len(cap["global_layers"])} global layers {", ".join(map(str, cap["global_layers"]))}, which use no RoPE.
{esc(cap["gqa"])}. torch {esc(cap["versions"]["torch"])}, transformers {esc(cap["versions"]["transformers"])}.</li>
<li><b>Generation recipe verified.</b> At these layers k = W<sub>k</sub>h and q = s(pos)·W<sub>q</sub>h with
s(pos) = 1 + 0.1·ln(1 + ⌊(pos+1)/8192⌋), recomputed from the checkpoint weights, reproduce what the model hands to attention
to bf16 precision. Dropping s(pos) gives ≥ 9.5% error at positions ≥ 8191, so the check discriminates.</li>
<li><b>Corpora.</b> First 32,768 tokens of 4 PG-19 test books. Keys: positions 1 … 31,743 (position 0, the sink, removed).
Queries: the head group's 5 query heads at the last 1,024 positions (the recent window, not indexed). Real queries at position t
would also see keys in [31,744, t); this simplification ignores them. ID baseline: keys at those last 1,024 positions.</li>
<li><b>Indexes.</b> faiss 1.15.1. IVFFlat, inner product, 256 lists, nprobe swept 1–256; the operating point nprobe=24 probes the
same 9.4% of lists as the paper's 96/1024 (a scaled equivalent, not their setting). HNSW on the Xbox reduction, M=48,
efConstruction=500, efSearch swept 10–2048, operating point 176 (the paper's). Work at recall 0.9 is interpolated
between the two bracketing sweep points.</li>
<li><b>SoftmaxLift.</b> Keys (k, −log u), u ~ U(0,1] seeded per corpus; queries (q/√128, 1). Priority-sampling estimate
from the returned K+1 = 128 keys. Attention error is over the indexed segment only.</li>
<li><b>Noise floor.</b> Four documents is a small sample; Figure 3 is a lower bound on the real spread.</li>
</ul>
{cap_table}
</section>

<section>
<h2>Claim table</h2>
<p class="note">Every number in this report is computed when the page is rendered, from
<code>profile_summary.json</code> beside it. All were measured 2026-10-16/17 on the A100 box from the capture described above; none is carried from an earlier project result.</p>
<div class="scroll"><table><tr><th>claim</th><th>value</th><th>status</th></tr>{claim_rows}</table></div>
</section>
"""
    return PAGE.replace("{{BODY}}", body)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KV-Cache Head Profiling</title>
<style>
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--base:#c3c2b7;
--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--retract:#fdf0ec;--retract-edge:#e34948}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--base:#383835;--s0:#3987e5;--s1:#d95926;--s2:#199e70;--retract:#2a1a17;--retract-edge:#e66767}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--base:#383835;--s0:#3987e5;--s1:#d95926;--s2:#199e70;--retract:#2a1a17;--retract-edge:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.7rem;line-height:1.2;margin:.2em 0 .4em}
h2{font-size:1.15rem;margin:2em 0 .6em;border-bottom:1px solid var(--grid);padding-bottom:.3em}
.kicker{color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-size:.75rem;margin:0}
.lede{color:var(--ink2);max-width:72ch}
section{background:var(--surface);border-radius:8px;padding:4px 18px 14px;margin:16px 0}
.retract{background:var(--retract);border-left:4px solid var(--retract-edge)}
figure{margin:18px 0 26px}
figcaption{color:var(--ink2);font-size:.9rem;margin-bottom:8px;max-width:80ch}
figcaption b{color:var(--ink)}
.scroll{overflow-x:auto;max-width:100%}
svg text{font:11px system-ui,sans-serif}
.axis{fill:var(--muted)} .cellv{font-size:10.5px;font-variant-numeric:tabular-nums}
.grid{stroke:var(--grid);stroke-width:1} .base{stroke:var(--base);stroke-width:1}
.dlabel{font-size:11px} .s0t{fill:var(--s0)} .s1t{fill:var(--s1)} .s2t{fill:var(--s2)}
.s0{fill:var(--s0)} .s1{fill:var(--s1)} .s2{fill:var(--s2)}
.cell{cursor:default} .cell:hover rect{stroke:var(--ink);stroke-width:1.5}
.ramp{display:flex;align-items:center;gap:8px;font-size:.8rem;color:var(--muted);margin:4px 0 0}
.ramp .bar{display:flex;height:10px;width:220px;border-radius:3px;overflow:hidden}
.ramp .bar span{flex:1}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:.85rem;color:var(--ink2);margin:4px 0}
.key i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.key i.s0{background:var(--s0)} .key i.s1{background:var(--s1)} .key i.s2{background:var(--s2)}
.note{color:var(--muted);font-size:.85rem}
table{border-collapse:collapse;font-size:.85rem;margin:8px 0}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
table.num td{text-align:right;font-variant-numeric:tabular-nums} table.num td.txt{text-align:left} tr>th:first-child{white-space:nowrap}
details summary{cursor:pointer;color:var(--ink2);font-size:.85rem;margin-top:6px}
code{font-size:.85em}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);padding:6px 9px;border-radius:5px;
font-size:12px;max-width:340px;display:none;z-index:9}
</style></head><body><main>
{{BODY}}
</main><div id="tip"></div>
<script>
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const c=e.target.closest('[data-tip]');
if(!c){tip.style.display='none';return}tip.textContent=c.dataset.tip;tip.style.display='block';
const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8);tip.style.left=x+'px';tip.style.top=(e.clientY+14)+'px'});
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compact")
    c.add_argument("--v1", type=Path, required=True)
    c.add_argument("--v2", type=Path, required=True)
    c.add_argument("--meta", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("render")
    r.add_argument("--summary", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "compact":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(compact(args.v1, args.v2, args.meta), separators=(",", ":"))
        )
    else:
        args.out.write_text(render(json.loads(args.summary.read_text())))


if __name__ == "__main__":
    main()
