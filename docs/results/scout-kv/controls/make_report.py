"""B8: build report.html (self-contained, inline SVG) from summary.json.

Run from the repo root after summarise.py:
    PYTHONPATH=. python docs/results/scout-kv/controls/make_report.py
"""

import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "summary.json").read_text())
LAYERS = ("11", "19", "43")
STATS = (
    ("hnsw_reach_k10", "HNSW reach, recall@10 ≥ 0.9"),
    ("hnsw_recall_k10_ef176", "HNSW recall@10 at ef176"),
    ("ivf_ratio_k128", "IVF scan ratio, real-style / key-side"),
    ("ivf_recall_k128_np24", "IVF recall@128 at nprobe 24"),
    ("hnsw_recall_k10_ef176_id", "HNSW recall@10 at ef176, key-side"),
)
# Fixed categorical order (dataviz reference palette, slots 1-5), validated light + dark.
SERIES = (
    ("v0_best", "v0 generator", 1, "circle"),
    ("pooled_real", "pooled real, seed 0", 2, "square"),
    ("pooled_real_seed1", "pooled real, seed 1", 2, "square"),
    ("pooled_real_seed2", "pooled real, seed 2", 2, "square"),
    ("norm_equalised", "norm-equalised", 3, "diamond"),
    ("gaussian", "Gaussian fit", 4, "triangle"),
    ("train_books_real", "training books 0-7", 5, "cross"),
)
C = S["compare_layers"]


def esc(x):
    return html.escape(str(x))


def fmt(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def marker(shape, x, y, slot, tip):
    t = f"<title>{esc(tip)}</title>"
    a = f'class="mk s{slot}"'
    if shape == "circle":
        return f'<circle {a} cx="{x:.1f}" cy="{y:.1f}" r="5">{t}</circle>'
    if shape == "square":
        return f'<rect {a} x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" rx="1">{t}</rect>'
    if shape == "diamond":
        p = f"{x:.1f},{y - 6:.1f} {x + 6:.1f},{y:.1f} {x:.1f},{y + 6:.1f} {x - 6:.1f},{y:.1f}"
        return f'<polygon {a} points="{p}">{t}</polygon>'
    if shape == "triangle":
        p = f"{x:.1f},{y - 6:.1f} {x + 6:.1f},{y + 5:.1f} {x - 6:.1f},{y + 5:.1f}"
        return f'<polygon {a} points="{p}">{t}</polygon>'
    return (
        f'<g {a}><path d="M{x - 5:.1f},{y - 5:.1f} L{x + 5:.1f},{y + 5:.1f} '
        f'M{x + 5:.1f},{y - 5:.1f} L{x - 5:.1f},{y + 5:.1f}" class="xk s{slot}"/>'
        f'<rect x="{x - 7:.1f}" y="{y - 7:.1f}" width="14" height="14" fill="transparent">{t}</rect></g>'
    )


def dot_panel(stat, label, layer):
    present = [s for s in SERIES if s[0] in C]
    ref = C["v0_best"]["layers"][layer]["statistics"][stat]
    vals = [C[s[0]]["layers"][layer]["statistics"][stat]["value"] for s in present]
    vals = [v for v in vals if v is not None] + [ref["min"], ref["max"]]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 0.05
    lo, hi = lo - pad, hi + pad
    W, LW, row = 330, 128, 20
    H = 30 + row * len(present) + 26
    px = lambda v: LW + (v - lo) / (hi - lo) * (W - LW - 14)  # noqa: E731
    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label)}, layer {layer}">',
        f'<text x="{LW}" y="14" class="ptitle">Layer {layer}</text>',
        f'<rect x="{px(ref["min"]):.1f}" y="22" width="{max(px(ref["max"]) - px(ref["min"]), 2):.1f}" '
        f'height="{row * len(present)}" class="band"><title>real reference, 5 draws: '
        f'{fmt(ref["min"])} to {fmt(ref["max"])}</title></rect>',
    ]
    for i, (key, name, slot, shape) in enumerate(present):
        y = 32 + i * row
        e = C[key]["layers"][layer]["statistics"][stat]
        out.append(f'<text x="{LW - 8}" y="{y + 4}" class="rlab" text-anchor="end">{esc(name)}</text>')
        out.append(f'<line x1="{LW}" x2="{W - 14}" y1="{y}" y2="{y}" class="grid"/>')
        if e["value"] is not None:
            tip = f"{name}, layer {layer}: {fmt(e['value'])} ({e['status']} the real range)"
            out.append(marker(shape, px(e["value"]), y, slot, tip))
    yb = 32 + row * len(present) + 4
    for v in (lo + pad, (lo + hi) / 2, hi - pad):
        out.append(f'<text x="{px(v):.1f}" y="{yb + 12}" class="tick" text-anchor="middle">{fmt(v, 2)}</text>')
    out.append("</svg>")
    return "".join(out)


def heatmap(kind, title):
    tab = S["r5"]["enrichment"][kind]["per_layer_median_ratio"]
    edges = json.loads((HERE / "neighbours.jsonl").read_text().splitlines()[0])["edges"]
    layers = sorted(tab, key=int)
    bins = [b for b in range(len(edges) - 1) if any(tab[L][b] is not None for L in layers)]
    cw, ch, LW, TH = 74, 22, 44, 34
    W, H = LW + cw * len(bins) + 8, TH + ch * len(layers) + 30
    out = [f'<svg viewBox="0 0 {W} {H}" style="max-width:{W}px" role="img" aria-label="{esc(title)}">']
    for j, b in enumerate(bins):
        out.append(
            f'<text x="{LW + j * cw + cw / 2}" y="{TH - 8}" class="tick" text-anchor="middle">'
            f"{edges[b]}–{edges[b + 1]}</text>"
        )
    for i, L in enumerate(layers):
        y = TH + i * ch
        out.append(f'<text x="{LW - 6}" y="{y + 15}" class="rlab" text-anchor="end">L{L}</text>')
        for j, b in enumerate(bins):
            v = tab[L][b]
            x = LW + j * cw
            if v is None:
                out.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" class="na"/>')
                continue
            step = 0 if v < 0.5 else 1 if v < 1 else 2 if v < 2 else 3 if v < 4 else 4
            out.append(
                f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="2" class="q{step}">'
                f"<title>layer {L}, {edges[b]}–{edges[b + 1]}: {v:.2f}x baseline</title></rect>"
                f'<text x="{x + cw / 2}" y="{y + 15}" class="cell{" inv" if step >= 3 else ""}" '
                f'text-anchor="middle">{v:.1f}</text>'
            )
    out.append("</svg>")
    return "".join(out) + (
        '<p class="muted">Cell: median over the 8 heads of (neighbour share / baseline share); '
        "bins with baseline share below 0.01 are omitted.</p>"
    )


def values_table():
    head = "".join(f"<th>{esc(n)}</th>" for k, n, _, _ in SERIES if k in C)
    rows = []
    for L in LAYERS:
        for stat, label in STATS:
            ref = C["v0_best"]["layers"][L]["statistics"][stat]
            cells = "".join(
                f'<td class="{C[k]["layers"][L]["statistics"][stat]["status"]}">'
                f'{fmt(C[k]["layers"][L]["statistics"][stat]["value"])}</td>'
                for k, _, _, _ in SERIES
                if k in C
            )
            rows.append(
                f"<tr><td>{L}</td><td>{esc(label)}</td><td>{fmt(ref['min'])}–{fmt(ref['max'])}</td>{cells}</tr>"
            )
    return (
        "<table><thead><tr><th>layer</th><th>statistic</th><th>real, 5 draws</th>"
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def xbook_section():
    """R6/R7 (cross-book plan): verdict and a per-draw table. Empty until the run exists."""
    if "xbook" not in S:
        return ""
    X, V = S["xbook"], S["xbook"]["verdict"]
    draws = list(X["draws"])
    pooled = [k for k in ("pooled_real", "pooled_real_seed1", "pooled_real_seed2") if k in C]
    head = "".join(
        f"<th>{c.split('_')[0]} {d.replace('_s', '')}</th>"
        for c in ("same_book_tail", "cross_book_tail")
        for d in draws
    )
    rows = []
    short = {"hnsw_reach_k10": "reach", "hnsw_recall_k10_ef176": "recall@10", "ivf_ratio_k128": "IVF ratio"}
    for L in LAYERS:
        for stat, _ in STATS[:3]:
            label = short[stat]
            ref = C["v0_best"]["layers"][L]["statistics"][stat]
            pv = [C[k]["layers"][L]["statistics"][stat]["value"] for k in pooled]
            cells = "".join(
                f'<td class="{e["status"]}">{fmt(e["value"])}</td>'
                for c in ("same_book_tail", "cross_book_tail")
                for d in draws
                for e in [X["draws"][d][c]["compare_layers"]["layers"][L]["statistics"][stat]]
            )
            rows.append(
                f"<tr><td>{L}</td><td>{esc(label)}</td><td>{fmt(ref['min'])}–{fmt(ref['max'])}</td>"
                f"<td>{fmt(min(pv))}–{fmt(max(pv))}</td><td>{fmt(ref['value'])}</td>{cells}</tr>"
            )
    table = (
        "<table><thead><tr><th>layer</th><th>statistic</th><th>real, 5 draws</th><th>pooled, 3 seeds</th>"
        f"<th>v0</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )
    flags = "".join(
        f"<tr><td>{d}</td><td>{X['draws'][d]['same_book_tail']['n_outside']}</td>"
        f"<td>{X['draws'][d]['cross_book_tail']['rules']['statistics_with_2_of_3']}</td>"
        f"<td>{'yes' if X['draws'][d]['cross_book_tail']['flatters'] else 'no'}</td></tr>"
        for d in draws
    )
    flags = (
        "<table><thead><tr><th>draw</th><th>same_book_tail: statistics outside (either way)</th>"
        "<th>cross_book_tail: statistics on v0's side</th><th>cross flatters</th></tr></thead>"
        f"<tbody>{flags}</tbody></table>"
    )
    reading = {
        "reachable": "the reading is <b>reachable</b>: queries unrelated to the keys keep the real hardness",
        "own_book": "the reading is <b>own_book</b>: the hardness needs queries related to their own book's keys",
        "inconclusive": "the reading is <b>inconclusive</b>",
        None: "R7 is <b>not read</b>, because the positive control failed",
    }[V["reading"]]
    return (
        "<h2>R6 and R7: cross-book control</h2>"
        "<p>Keys from the bodies of four reference books; queries from the tails of the same four books "
        "(<code>same_book_tail</code>, the positive control) or of the other four (<code>cross_book_tail</code>). "
        "Within a draw both controls use the same key rows. Draws: partition a (key books 12-15) and b (16-19), "
        "seeds 0 and 1. Plan: <code>docs/ai/plans/2026-10-21-scout-kv-cross-book.md</code>.</p>"
        f"<div class='verdict'><p><b>R6 (positive control valid):</b> {'yes' if V['positive_control_valid'] else 'no'}. "
        f"{reading}. <b>For the record:</b> cross_book_tail is on v0's side in {V['n_cross_flatter']} of 4 draws.</p></div>"
        f"<div class='scroll'>{flags}</div><div class='scroll'>{table}</div>"
        "<p class='muted'>Cell shading: orange above the real range, blue below it. Draw labels: a0 is partition a, "
        "seed 0, and so on.</p>"
        "<p><b>Descriptive, not a rule result.</b> Same and cross share their key rows within a draw, so they can be paired. "
        "At layers 11 and 43, cross minus same is at most 0.045 in recall@10 and 0.08 in IVF ratio, far below the spread "
        "across draws (layer 43 IVF ratio 3.13-3.71). At layer 43 both tail controls keep an IVF ratio of 3.10-3.71, "
        "against 1.12-1.15 for pooled_real. Layer 19, the layer with the strongest same-book recency, differs more "
        "(up to 0.061 in recall and 0.092 in IVF ratio), with the sign set by the partition. Why R6 fails: with four "
        "key books, layer 19 depends on which books form the key set.</p>"
    )


# Descriptive paragraph for the held-out section, written after the run (B12).
HELDOUT_NOTE = ""


def heldout_section():
    """R6a/R6/R7 (held-out plan): verdict, flags, per-layer values, paired deltas."""
    if "heldout" not in S:
        return ""
    X, V = S["heldout"], S["heldout"]["verdict"]
    if not V["path_ok"]:
        pc = X["path_check"]
        return (
            "<h2>R6a, R6 and R7: held-out-book control</h2><div class='verdict'><p><b>R6a (path) FAILS:</b> "
            f"the seed-0 rebuild did not pair with the committed reference ({pc.get('n_compared')} compared, "
            f"missing {len(pc.get('missing', []))}, mismatched {len(pc.get('mismatched', []))}, metadata "
            f"mismatches {len(pc.get('meta_mismatch', []))}). Per the plan nothing else is read.</p></div>"
        )
    draws = list(X["draws"])
    ctl = ("rebuild", "held_out_tail")
    short = {"hnsw_reach_k10": "reach", "hnsw_recall_k10_ef176": "recall@10", "ivf_ratio_k128": "IVF ratio"}
    head = "".join(f"<th>{'rebuild' if c == 'rebuild' else 'held-out'} {d}</th>" for c in ctl for d in draws)
    rows = []
    for L in LAYERS:
        for stat, _ in STATS[:3]:
            ref = C["v0_best"]["layers"][L]["statistics"][stat]
            tb = C["train_books_real"]["layers"][L]["statistics"][stat]
            cells = "".join(
                f'<td class="{e["status"]}">{fmt(e["value"])}</td>'
                for c in ctl
                for d in draws
                for e in [X["draws"][d][c]["compare_layers"]["layers"][L]["statistics"][stat]]
            )
            rows.append(
                f"<tr><td>{L}</td><td>{short[stat]}</td><td>{fmt(ref['min'])}–{fmt(ref['max'])}</td>"
                f'<td class="{tb["status"]}">{fmt(tb["value"])}</td><td>{fmt(ref["value"])}</td>{cells}</tr>'
            )
    table = (
        "<table><thead><tr><th>layer</th><th>statistic</th><th>real, 5 draws</th><th>books 0-7 alone</th>"
        f"<th>v0</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )
    drows = []
    for L in LAYERS:
        for stat, _ in STATS[:3]:
            vals = [X["draws"][d]["held_out_minus_rebuild"][L][stat] for d in draws]
            drows.append(
                f"<tr><td>{L}</td><td>{short[stat]}</td>" + "".join(f"<td>{fmt(v)}</td>" for v in vals) + "</tr>"
            )
    deltas = (
        "<table><thead><tr><th>layer</th><th>statistic</th>"
        + "".join(f"<th>{d}</th>" for d in draws)
        + f"</tr></thead><tbody>{''.join(drows)}</tbody></table>"
    )
    flags = "".join(
        f"<tr><td>{d}</td><td>{X['draws'][d]['seed']}</td><td>{', '.join(map(str, X['draws'][d]['query_books']))}</td>"
        f"<td>{X['draws'][d]['rebuild']['n_outside']}</td>"
        f"<td>{X['draws'][d]['held_out_tail']['rules']['statistics_with_2_of_3']}</td>"
        f"<td>{'yes' if X['draws'][d]['held_out_tail']['flatters'] else 'no'}</td></tr>"
        for d in draws
    )
    flags = (
        "<table><thead><tr><th>draw</th><th>seed</th><th>held-out query books</th>"
        "<th>rebuild: statistics outside (either way)</th><th>held-out: statistics on v0's side</th>"
        f"<th>held-out flatters</th></tr></thead><tbody>{flags}</tbody></table>"
    )
    pc = X["path_check"]
    reading = {
        "reachable": "R7 reads <b>reachable</b>: queries from books outside the key set keep the real difficulty",
        "not_reachable": "R7 reads <b>not reachable</b>: queries from books outside the key set are too easy",
        "inconclusive": "R7 reads <b>inconclusive</b>",
        None: "R7 is <b>not read</b>",
    }[V["reading"]]
    pcv = "n/a" if V["positive_control_valid"] is None else ("yes" if V["positive_control_valid"] else "no")
    nflat = "" if V["n_held_out_flatter"] is None else (
        f" held_out_tail is on v0's side in {V['n_held_out_flatter']} of 4 draws."
    )
    return (
        "<h2>R6a, R6 and R7: held-out-book control</h2>"
        "<p>The rule-grade redo of the cross-book control. Keys are exactly the reference's: all eight reference "
        "books 12-19, built per KV head by the reference code at the reference's per-head seed. "
        "<code>rebuild</code> is the reference corpus itself at a fresh seed (the positive control); "
        "<code>held_out_tail</code> keeps the same keys and index seed and takes the queries from the tails of eight "
        "books from 0-11, none of which is in the key set. Plan: "
        "<code>docs/ai/plans/2026-10-21-scout-kv-held-out.md</code>.</p>"
        f"<div class='verdict'><p><b>R6a (path):</b> {'passes' if V['path_ok'] else 'FAILS'}: the seed-0 rebuild "
        f"pairs with the committed reference on {pc['n_compared']} corpora, {pc['n_exact']} exact, max own-book "
        f"share difference {fmt(pc['max_abs_diff'], 5)}. <b>R6 (positive control valid):</b> {pcv}. "
        f"{reading}.{nflat}</p></div>"
        f"<div class='scroll'>{flags}</div><div class='scroll'>{table}</div>"
        "<p class='muted'>Cell shading: orange above the real range, blue below it. "
        "“books 0-7 alone” is <code>train_books_real</code>: keys and queries both from books 0-7, seed 0.</p>"
        "<h3>Paired difference, held-out minus rebuild (same keys, same index seed)</h3>"
        f"<div class='scroll'>{deltas}</div>{HELDOUT_NOTE}"
    )


def rules_table():
    rows = []
    for name in ("pooled_real", "pooled_real_seed1", "pooled_real_seed2", "norm_equalised", "gaussian", "train_books_real"):
        r = S["rules"].get(name)
        if not r:
            continue
        cells = "".join(
            f"<td>{r['per_statistic'][s]['layers_in_v0_direction']} / 3</td>"
            for s in ("hnsw_reach_k10", "hnsw_recall_k10_ef176", "ivf_ratio_k128")
        )
        rows.append(f"<tr><td>{esc(name)}</td>{cells}<td>{r['statistics_with_2_of_3']}</td></tr>")
    return (
        "<table><thead><tr><th>control</th><th>reach</th><th>recall@10 ef176</th><th>IVF ratio</th>"
        "<th>statistics with ≥ 2 of 3 layers</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def r5_table():
    rows = "".join(
        f"<tr><td>{esc(r['scalar'])}</td><td>{esc(r['hardness'])}</td><td>{fmt(r['rho'])}</td>"
        f"<td>{r['n']}</td><td>{'yes' if r['associated'] else 'no'}</td></tr>"
        for r in S["r5"]["spearman"]
    )
    return (
        "<table><thead><tr><th>neighbour scalar (OOD queries)</th><th>hardness, seed 0</th>"
        f"<th>Spearman ρ</th><th>n</th><th>|ρ| ≥ 0.5</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def neighbours_table():
    rows = []
    for L, e in sorted(S["neighbours_per_layer"].items(), key=lambda kv: int(kv[0])):
        o = e["ood"]
        rows.append(
            f"<tr><td>{L}</td><td>{fmt(o['own_book_share'])}</td><td>{fmt(o['share_lag_le_1024'])}</td>"
            f"<td>{fmt(o['base_share_lag_le_1024'])}</td><td>{fmt(o['share_pos_lt_16'])}</td>"
            f"<td>{fmt(o['base_share_pos_lt_16'], 4)}</td><td>{fmt(o['hub_slot_share'])}</td>"
            f"<td>{fmt(e['id']['share_lag_le_1024'])}</td><td>{o['n_distinct_keys']:g}</td>"
            f"<td>{o['hub_median_pos']:,.0f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>layer</th><th>own-book share</th><th>lag ≤ 1024</th><th>baseline</th>"
        "<th>key pos &lt; 16</th><th>baseline</th><th>top-1% slot share</th><th>key-side lag ≤ 1024</th>"
        "<th>distinct keys</th><th>hub pos</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def legend():
    items = []
    seen = set()
    for key, name, slot, shape in SERIES:
        if key not in C or (slot == 2 and slot in seen):
            continue
        seen.add(slot)
        nm = "pooled real (seeds 0, 1, 2)" if slot == 2 else name
        items.append(
            f'<span class="lg"><svg viewBox="0 0 16 16" width="16" height="16">'
            f"{marker(shape, 8, 8, slot, nm)}</svg>{esc(nm)}</span>"
        )
    items.append('<span class="lg"><span class="sw band"></span>real reference, range of 5 draws</span>')
    return '<div class="legend">' + "".join(items) + "</div>"


def main():
    panels = []
    for stat, label in STATS[:3]:
        panels.append(
            f"<h3>{esc(label)}</h3><div class='row'>"
            + "".join(f"<div class='panel'>{dot_panel(stat, label, L)}</div>" for L in LAYERS)
            + "</div>"
        )
    r1 = S["R1_per_seed"]
    page = TEMPLATE.format(
        r1=", ".join(f"{k}: {'fires' if v else 'does not fire'}" for k, v in r1.items()),
        r2="fires" if S["R2_train_books_easier"] else "does not fire",
        v0m="matches" if S["v0_remeasure"]["match"] else "does NOT match",
        legend=legend(),
        panels="".join(panels),
        values=values_table(),
        rules=rules_table(),
        r5=r5_table(),
        ntab=neighbours_table(),
        xbook=xbook_section(),
        heldout=heldout_section(),
        heat_lag=heatmap("lag", "Same-book lag enrichment"),
        heat_pos=heatmap("pos", "Key position enrichment"),
        distinct=(HERE / "distinct_keys.txt").read_text().replace("<", "&lt;"),
    )
    (HERE / "report.html").write_text(page)
    print("wrote", HERE / "report.html")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>scout_kv controls</title>
<style>
:root {{ --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9;
  --band:#e1e0d9; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
  --q0:#f0efec; --q1:#cde2fb; --q2:#86b6ef; --q3:#2a78d6; --q4:#184f95; --above:#fbe3d9; --below:#d9e8fb; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec;
  --ink2:#c3c2b7; --grid:#2c2c2a; --band:#383835; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
  --q0:#383835; --q1:#184f95; --q2:#256abf; --q3:#5598e7; --q4:#9ec5f4; --above:#4a2a1c; --below:#1c2f4a; }} }}
:root[data-theme="dark"] {{ --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a; --band:#383835;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --q0:#383835; --q1:#184f95; --q2:#256abf;
  --q3:#5598e7; --q4:#9ec5f4; --above:#4a2a1c; --below:#1c2f4a; }}
body {{ background:var(--page); color:var(--ink); font:15px/1.5 system-ui,sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:1080px; margin:0 auto; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:19px; margin:32px 0 8px; }} h3 {{ font-size:15px; margin:18px 0 4px; color:var(--ink2); }}
p, li {{ color:var(--ink); max-width:76ch; }} .muted {{ color:var(--ink2); }}
.verdict {{ background:var(--surface); border:1px solid var(--grid); border-radius:8px; padding:12px 16px; }}
.row {{ display:flex; flex-wrap:wrap; gap:12px; }} .panel {{ flex:1 1 300px; min-width:0; background:var(--surface);
  border:1px solid var(--grid); border-radius:8px; padding:6px; }}
svg {{ width:100%; height:auto; display:block; }}
.ptitle {{ font-size:12px; font-weight:600; fill:var(--ink); }} .rlab {{ font-size:11px; fill:var(--ink2); }}
.tick {{ font-size:10px; fill:var(--muted); }} .grid {{ stroke:var(--grid); stroke-width:1; }}
.band {{ fill:var(--band); }} .sw.band {{ background:var(--band); }} .sw {{ display:inline-block; width:14px; height:10px; border-radius:2px; }}
.mk {{ stroke:var(--surface); stroke-width:2; }} .xk {{ stroke-width:2.5; fill:none; }}
.s1 {{ fill:var(--s1); stroke:var(--s1); }} .s2 {{ fill:var(--s2); }} .s3 {{ fill:var(--s3); }} .s4 {{ fill:var(--s4); }}
g.s5 path {{ stroke:var(--s5); }} circle.s1,rect.s2,polygon.s3,polygon.s4 {{ stroke:var(--surface); }}
.mk:hover {{ stroke:var(--ink); }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; margin:8px 0; color:var(--ink2); font-size:13px; }}
.lg {{ display:inline-flex; align-items:center; gap:6px; }} .lg svg {{ width:16px; height:16px; flex:none; }}
.q0 {{ fill:var(--q0); }} .q1 {{ fill:var(--q1); }} .q2 {{ fill:var(--q2); }} .q3 {{ fill:var(--q3); }} .q4 {{ fill:var(--q4); }}
.na {{ fill:none; stroke:var(--grid); stroke-dasharray:2 2; }} .cell {{ font-size:10px; fill:var(--ink); }} .cell.inv {{ fill:var(--surface); }}
.scroll {{ overflow-x:auto; }} table {{ border-collapse:collapse; font-size:12.5px; margin:8px 0; background:var(--surface); }}
th, td {{ border:1px solid var(--grid); padding:4px 7px; text-align:right; white-space:nowrap; }}
th {{ color:var(--ink2); font-weight:600; }} td:nth-child(-n+2) {{ text-align:left; }}
td.above {{ background:var(--above); }} td.below {{ background:var(--below); }}
pre {{ background:var(--surface); border:1px solid var(--grid); padding:10px; overflow-x:auto; font-size:12px; }}
</style></head><body><main>
<h1>scout_kv controls: why v0 looks too easy</h1>
<p class="muted">Box run B7 on tig-gpu-alt (L40), 2026-10-19. Every number on this page is MEASURED unless marked ESTIMATE.
Sources: <code>summary.json</code>, built by <code>summarise.py</code> from the JSONL files in this directory.</p>

<div class="verdict">
<p><b>R1 (hard stop): the gate's measurement path flatters synthetic data.</b> Per seed: {r1}. Real <code>h</code> from the
reference books, split into keys and queries the way synthetic corpora are (one pool, no position), is already outside the
real range in v0's direction for HNSW recall@10 on at least 2 of 3 layers and for the IVF ratio on all 3, at every seed.
Under the plan's rules, v1 is not designed; the gate question goes to the owner.</p>
<p><b>R2:</b> {r2} (the training books 0-7 are about as hard as the reference books).
<b>v0 re-measured on the L40:</b> {v0m} the committed records within the tolerance fixed before the run.</p>
</div>

<h2>Where each control sits against the real range</h2>
<p>One panel per layer. The grey band is the range of the five real reference draws. Hover a marker for its value.
The full table below has every value.</p>
{legend}
{panels}

<h2>All values</h2>
<p class="muted">Orange cells are above the real range, blue cells below it.</p>
<div class="scroll">{values}</div>

<h2>Decision rules R1 and R2</h2>
<p>Layers (of 11, 19, 43) on which the control is outside the real range in the same direction as v0.
R1 fires when <code>pooled_real</code> does this on ≥ 2 layers for ≥ 2 of the three statistics; R2 the same for the training books.</p>
<div class="scroll">{rules}</div>

<h2>R3 and R4 (descriptive)</h2>
<ul>
<li><b>R3, Gaussian fit:</b> at least as far from real as v0, in v0's direction, on 8 of the 9 rule cells; the exception is
layer 19's IVF ratio (0.097 against v0's 0.095). At layer 43 the gap is large: HNSW reach 1.000 against v0's 0.375 and real
0.000-0.125. Second-order structure alone is mostly easier than v0, so v0 has learned something beyond the covariance.</li>
<li><b>R4, norm-equalised:</b> it does not move <code>pooled_real</code> toward v0. Removing the spread of norms makes HNSW
much harder (recall@10 at layer 43: 0.296 against pooled 0.770 and real 0.574-0.637), so norm spread makes real search
easier, not harder.</li>
</ul>

{xbook}
{heldout}
<h2>Neighbour positions (all 12 layers, 96 corpora)</h2>
<p>Each corpus is the one <code>reference/seed0.jsonl</code> profiled: own-book shares matched on all 96 corpora, 92 exactly,
largest difference 3.9e-5 (<code>check_pairing.py</code>). Medians over the 8 heads per layer:</p>
<div class="scroll">{ntab}</div>
<p class="note">Distinct keys: keys that appear in any real-style query's top-10 (of 31,736). Hub pos: median position
of the most-used keys (all of them when fewer than 317 are used); keys sit at positions 1-31,743 with median about
15,900. Both columns come from the 2026-10-21 rerun, which also corrected an error in the hub position (see the README).</p>
<h3>Same-book lag, enrichment over the query's own-book keys</h3>
<div class="panel">{heat_lag}</div>
<h3>Key position, enrichment over all keys</h3>
<div class="panel">{heat_pos}</div>
<h3>R5: neighbour scalars against per-corpus hardness</h3>
{r5}
<p>No scalar reaches |ρ| ≥ 0.5. The top-1% slot share saturates near 1.0 at every layer, so it cannot carry an association.
A direct count shows why (head 0, seed 0):</p>
<pre>{distinct}</pre>
<p>All 5,120 real-style queries at layer 3 share 11 target keys of 31,736. Those keys are not simply the longest
(norm-count correlation about 0). Over all 8 heads, the rerun's median is 31-949 distinct keys per layer (table above).
At 8 of the 9 layers from 15 on, those keys sit at median positions 26,609-30,087, late in the books. The same-book share
stays near 1/8 at most of these layers, so the late keys come from any book. The distinct-key count was not part of the
pre-registered records; it is reported here as an observation and a candidate statistic for follow-up, not a result of
this plan.</p>

<h2>Caveats</h2>
<ul>
<li>Each control except <code>pooled_real</code> is a single draw (seed 0) against the range of five real draws.
Layer 11's <code>pooled_real</code> changes status across seeds on reach and recall@10; layers 19 and 43 do not.</li>
<li>The v0 match tolerances (reach ±1/8, recalls ±0.01, IVF ratio ±5% relative) are ESTIMATE (unverified) judgements fixed
before the run, not measured noise.</li>
<li>Layers 11, 19 and 43 only for the controls; the neighbour diagnostic covers all 12.</li>
</ul>
</main></body></html>
"""

if __name__ == "__main__":
    main()
