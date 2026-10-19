"""Build docs/results/scout-kv/v0/report.html from the committed B3 and B6 records.

    PYTHONPATH=. python docs/results/scout-kv/v0/make_report.py [data_dir]

Reads docs/results/scout-kv/reference/seed{0..4}.jsonl, and from this directory
best.jsonl, final.jsonl, check_best.json, check_final.json and intrinsic.jsonl. The
page is self-contained: every chart is inline SVG, with no external assets.
"""

import html
import json
import statistics as pystats
import sys
from pathlib import Path

from src.data.kv_capture import GLOBAL_LAYERS
from src.eval.kv_gate import STATISTICS, pooled_statistics

HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
REF = Path(__file__).resolve().parent.parent / "reference"
SEEDS = range(5)

LABEL = {
    "hnsw_reach_k10": "HNSW reach, k=10",
    "hnsw_recall_k10_ef176": "HNSW recall@10 at ef 176",
    "ivf_ratio_k128": "IVF scan ratio, k=128 (ood / id)",
    "ivf_recall_k128_np24": "IVF recall@128 at nprobe 24",
    "hnsw_recall_k10_ef176_id": "HNSW recall@10 at ef 176, key-side queries",
}
INTRINSIC = ["lid_median", "relative_contrast_median", "hubness_skew", "ivf_gini"]


def read_jsonl(p):
    return [
        json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()
    ]


def fmt(v, d=4):
    return "n/a" if v is None else f"{v:.{d}f}"


def per_layer(records):
    return {
        L: pooled_statistics([r for r in records if r["layer"] == L])
        for L in GLOBAL_LAYERS
    }


def head_values(r):
    o = r["plain_ood_k128"]["ivf_scan_frac"]
    i = r["plain_id_k128"]["ivf_scan_frac"]
    return {
        "hnsw_reach_k10": 1.0 if r["plain_ood_k10"]["hnsw_reached"] else 0.0,
        "hnsw_recall_k10_ef176": r["plain_ood_k10"]["hnsw_recall_op"],
        "ivf_ratio_k128": (o / i) if (o is not None and i) else None,
        "ivf_recall_k128_np24": r["plain_ood_k128"]["ivf_recall_op"],
        "hnsw_recall_k10_ef176_id": r["plain_id_k10"]["hnsw_recall_op"],
    }


# ---------------------------------------------------------------- SVG charts

W, H, PADL, PADR, PADT, PADB = 440, 210, 52, 12, 26, 34


def _band_text(band):
    lo, hi = band
    if lo is None and hi is None:
        return "unset"
    if hi is None:
        return f">= {lo:g}"
    if lo is None:
        return f"<= {hi:g}"
    return f"{lo:g} to {hi:g}"


def _scale(vals, lo_hint=None, hi_hint=None):
    vals = [v for v in vals if v is not None]
    lo, hi = min(vals), max(vals)
    if lo_hint is not None:
        lo = min(lo, lo_hint)
    if hi_hint is not None:
        hi = max(hi, hi_hint)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.05, hi + 0.05
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def layer_chart(title, real_draws, v0_best, v0_final=None):
    """real_draws: {layer: [values over draws]}; v0_*: {layer: value}."""
    allv = [v for vs in real_draws.values() for v in vs] + list(v0_best.values())
    if v0_final:
        allv += list(v0_final.values())
    lo, hi = _scale(allv)
    pw, ph = W - PADL - PADR, H - PADT - PADB

    def x(i):
        return PADL + pw * (i + 0.5) / len(GLOBAL_LAYERS)

    def y(v):
        return PADT + ph * (1 - (v - lo) / (hi - lo))

    out = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(title)}">'
    ]
    out.append(f'<text x="{PADL}" y="14" class="dlabel">{html.escape(title)}</text>')
    for t in range(5):
        v = lo + (hi - lo) * t / 4
        out.append(
            f'<line class="grid" x1="{PADL}" x2="{W - PADR}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
        )
        out.append(
            f'<text class="axis" x="{PADL - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.3g}</text>'
        )
    for i, L in enumerate(GLOBAL_LAYERS):
        out.append(
            f'<text class="axis" x="{x(i):.1f}" y="{H - 16}" text-anchor="middle">{L}</text>'
        )
        vs = [v for v in real_draws.get(L, []) if v is not None]
        if vs:
            out.append(
                f'<line stroke="var(--s0)" stroke-width="5" stroke-opacity=".35" x1="{x(i) - 5:.1f}" x2="{x(i) - 5:.1f}" '
                f'y1="{y(min(vs)):.1f}" y2="{y(max(vs)):.1f}"/>'
            )
            m = sum(vs) / len(vs)
            out.append(
                f'<circle class="s0" cx="{x(i) - 5:.1f}" cy="{y(m):.1f}" r="3"><title>real layer {L}: '
                f"mean {m:.4f}, range {min(vs):.4f}-{max(vs):.4f}</title></circle>"
            )
        if v0_best.get(L) is not None:
            out.append(
                f'<circle class="s1" cx="{x(i) + 5:.1f}" cy="{y(v0_best[L]):.1f}" r="3.5"><title>v0 best layer {L}: '
                f"{v0_best[L]:.4f}</title></circle>"
            )
        if v0_final and v0_final.get(L) is not None and v0_final[L] != v0_best.get(L):
            cx, cy = x(i) + 5, y(v0_final[L])
            out.append(
                f'<rect class="s2" x="{cx - 3:.1f}" y="{cy - 3:.1f}" width="6" height="6" transform="rotate(45 {cx:.1f} {cy:.1f})">'
                f"<title>v0 final layer {L}: {v0_final[L]:.4f}</title></rect>"
            )
    out.append(
        f'<text class="axis" x="{PADL + pw / 2:.1f}" y="{H - 2}" text-anchor="middle">layer</text>'
    )
    out.append("</svg>")
    return "".join(out)


def strip_chart(title, real_vals, best, final, band):
    """`band` is the gate's (min, max) for this statistic; either may be None."""
    lo, hi = _scale(real_vals + [best, final] + [v for v in band if v is not None])
    w, h, pl, pr = 440, 96, 70, 14
    pw = w - pl - pr

    def x(v):
        return pl + pw * (v - lo) / (hi - lo)

    rows = [
        ("real x5", real_vals, "s0"),
        ("v0 best", [best], "s1"),
        ("v0 final", [final], "s2"),
    ]
    out = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(title)}">'
    ]
    out.append(f'<text x="{pl}" y="12" class="dlabel">{html.escape(title)}</text>')
    b_lo = lo if band[0] is None else band[0]
    b_hi = hi if band[1] is None else band[1]
    out.append(
        f'<rect class="band" x="{x(b_lo):.1f}" y="20" width="{x(b_hi) - x(b_lo):.1f}" height="{h - 36}">'
        f"<title>gate band: {_band_text(band)}</title></rect>"
    )
    for t in range(5):
        v = lo + (hi - lo) * t / 4
        out.append(
            f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="20" y2="{h - 16}"/>'
        )
        out.append(
            f'<text class="axis" x="{x(v):.1f}" y="{h - 3}" text-anchor="middle">{v:.3g}</text>'
        )
    for j, (name, vals, cls) in enumerate(rows):
        yy = 30 + j * 20
        out.append(
            f'<text class="axis" x="{pl - 8}" y="{yy + 4}" text-anchor="end">{name}</text>'
        )
        for v in vals:
            if v is not None:
                out.append(
                    f'<circle class="{cls}" cx="{x(v):.1f}" cy="{yy}" r="4" fill-opacity=".8"><title>{name}: {v:.4f}</title></circle>'
                )
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- page

CSS = """
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--s0:#3987e5;--s1:#d95926;--s2:#199e70}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--s0:#3987e5;--s1:#d95926;--s2:#199e70}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.7rem;line-height:1.2;margin:.2em 0 .4em}
h2{font-size:1.15rem;margin:2em 0 .6em;border-bottom:1px solid var(--grid);padding-bottom:.3em}
.kicker{color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-size:.75rem;margin:0}
.lede{color:var(--ink2);max-width:72ch}
section{background:var(--surface);border-radius:8px;padding:4px 18px 14px;margin:16px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
svg text{font:11px system-ui,sans-serif;fill:var(--ink)} .axis{fill:var(--muted)}
.grid{stroke:var(--grid);stroke-width:1} .dlabel{font-size:12px;font-weight:600}
.s0{fill:var(--s0)} .s1{fill:var(--s1)} .s2{fill:var(--s2)} .band{fill:var(--grid);fill-opacity:.7}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:.85rem;color:var(--ink2);margin:4px 0}
.key i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.key i.s0{background:var(--s0)} .key i.s1{background:var(--s1)} .key i.s2{background:var(--s2)} .key i.band{background:var(--grid);border-radius:2px}
.note{color:var(--muted);font-size:.85rem}
.scroll{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;font-size:.85rem;margin:8px 0}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
table.num td{text-align:right;font-variant-numeric:tabular-nums} table.num td.txt{text-align:left}
details summary{cursor:pointer;color:var(--ink2)}
"""

LEGEND = (
    '<div class="legend"><span class="key"><i class="s0"></i>real, 5 draws (bar: range, dot: mean)</span>'
    '<span class="key"><i class="s1"></i>v0 best checkpoint</span>'
    '<span class="key"><i class="s2"></i>v0 final checkpoint (drawn only where it differs)</span></div>'
)


def table(header, rows, num=True):
    th = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"txt" if j == 0 else ""}">{c}</td>' for j, c in enumerate(r)
        )
        + "</tr>"
        for r in rows
    )
    return f'<div class="scroll"><table class="{"num" if num else ""}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def main():
    ref = {s: read_jsonl(REF / f"seed{s}.jsonl") for s in SEEDS}
    ref_pooled = {s: pooled_statistics(ref[s]) for s in SEEDS}
    v0 = {name: read_jsonl(HERE / f"{name}.jsonl") for name in ("best", "final")}
    v0_pooled = {name: pooled_statistics(v0[name]) for name in v0}
    checks = {
        name: json.loads((HERE / f"check_{name}.json").read_text()) for name in v0
    }
    intrinsic = read_jsonl(HERE / "intrinsic.jsonl")
    for name, c in checks.items():
        assert c["n_records"] == 96, (name, c["n_records"])
        assert {r["layer"] for r in v0[name]} == set(GLOBAL_LAYERS), name
        for st in c["statistics"]:
            assert abs(st["value"] - v0_pooled[name][st["statistic"]]) < 1e-12, (
                name,
                st,
            )
    for name in v0:
        weights = {r["generator_weights"] for r in v0[name]}
        assert weights == {"ema"}, (name, weights)

    ref_layer = {s: per_layer(ref[s]) for s in SEEDS}
    v0_layer = {name: per_layer(v0[name]) for name in v0}

    # pooled summary
    pooled_rows, strips = [], []
    for st in STATISTICS:
        vals = [ref_pooled[s][st] for s in SEEDS]
        b, f = v0_pooled["best"][st], v0_pooled["final"][st]
        mean = sum(vals) / len(vals)
        verdict = {
            name: next(x for x in c["statistics"] if x["statistic"] == st)
            for name, c in checks.items()
        }
        band = (verdict["best"]["band"]["min"], verdict["best"]["band"]["max"])
        assert band == (verdict["final"]["band"]["min"], verdict["final"]["band"]["max"])
        pooled_rows.append(
            [
                html.escape(LABEL[st]),
                fmt(mean),
                fmt(min(vals)),
                fmt(max(vals)),
                fmt(max(vals) - min(vals)),
                html.escape(_band_text(band)),
                fmt(b),
                fmt(b - mean),
                verdict["best"]["status"],
                fmt(f),
                fmt(f - mean),
                verdict["final"]["status"],
            ]
        )
        strips.append(strip_chart(LABEL[st], vals, b, f, band))

    layer_figs = []
    for st in STATISTICS:
        real_draws = {L: [ref_layer[s][L][st] for s in SEEDS] for L in GLOBAL_LAYERS}
        layer_figs.append(
            layer_chart(
                LABEL[st],
                real_draws,
                {L: v0_layer["best"][L][st] for L in GLOBAL_LAYERS},
                {L: v0_layer["final"][L][st] for L in GLOBAL_LAYERS},
            )
        )

    layer_rows = []
    for L in GLOBAL_LAYERS:
        row = [str(L)]
        for st in STATISTICS:
            rv = [ref_layer[s][L][st] for s in SEEDS]
            rv = [v for v in rv if v is not None]
            row += [
                fmt(sum(rv) / len(rv) if rv else None),
                fmt(v0_layer["best"][L][st]),
            ]
        layer_rows.append(row)
    layer_header = ["layer"] + [
        f"{h} {k}"
        for k in ("reach", "rec10", "ivf ratio", "ivf rec", "id rec")
        for h in ("real", "v0")
    ]

    # per head: real mean over 5 draws vs v0 best
    head_rows = []
    for L in GLOBAL_LAYERS:
        for g in range(8):
            rvals = [
                head_values(
                    next(r for r in ref[s] if r["layer"] == L and r["kv_head"] == g)
                )
                for s in SEEDS
            ]
            bv = head_values(
                next(r for r in v0["best"] if r["layer"] == L and r["kv_head"] == g)
            )
            row = [f"{L} / {g}"]
            for st in STATISTICS:
                xs = [d[st] for d in rvals if d[st] is not None]
                row += [fmt(sum(xs) / len(xs) if xs else None, 3), fmt(bv[st], 3)]
            head_rows.append(row)

    # intrinsic
    intr = {(r["layer"], r["source"]): r for r in intrinsic}
    intr_figs, intr_rows = [], []
    for m in INTRINSIC:
        intr_figs.append(
            layer_chart(
                m,
                {L: [intr[(L, "real")][m]] for L in GLOBAL_LAYERS},
                {L: intr[(L, "v0_best")][m] for L in GLOBAL_LAYERS},
                {
                    L: intr[(L, "v0_final")][m]
                    for L in GLOBAL_LAYERS
                    if (L, "v0_final") in intr
                },
            )
        )
    for (L, src), r in sorted(intr.items()):
        intr_rows.append(
            [f"{L} {src}"]
            + [fmt(r[m], 3) for m in INTRINSIC]
            + [str(r["n_rows"]), str(r["lid_discarded_queries"])]
        )

    differ = [L for L in GLOBAL_LAYERS if v0_layer["best"][L] != v0_layer["final"][L]]

    # findings, computed from the same records as the tables
    findings = []
    for st in STATISTICS:
        vals = [ref_pooled[s][st] for s in SEEDS]
        b = v0_pooled["best"][st]
        where = "above" if b > max(vals) else "below" if b < min(vals) else "inside"
        findings.append(
            f"<li>{html.escape(LABEL[st])}: v0 best {fmt(b)} is <b>{where}</b> the real range "
            f"{fmt(min(vals))}-{fmt(max(vals))}.</li>"
        )

    def intr_range(m, src, layers, d=1):
        xs = [intr[(L, src)][m] for L in layers]
        return f"{min(xs):.{d}f}-{max(xs):.{d}f}"

    late = [L for L in GLOBAL_LAYERS if L >= 15]
    early = [L for L in GLOBAL_LAYERS if L < 15]
    lid_lower = [
        L
        for L in GLOBAL_LAYERS
        if intr[(L, "v0_best")]["lid_median"] < intr[(L, "real")]["lid_median"]
    ]
    intr_findings = [
        f"<li>Hubness skew: real {intr_range('hubness_skew', 'real', late)} at layers 15-47 and "
        f"{intr_range('hubness_skew', 'real', early)} at layers 3-11; v0 best "
        f"{intr_range('hubness_skew', 'v0_best', GLOBAL_LAYERS)} across all layers. v0 does not "
        "reproduce the hubs of real h.</li>",
        f"<li>IVF Gini at layers 15-47: real {intr_range('ivf_gini', 'real', late, 3)}, v0 best "
        f"{intr_range('ivf_gini', 'v0_best', late, 3)}. v0 fills k-means cells more evenly.</li>",
        f"<li>LID median: v0 best is lower than real at layers {', '.join(map(str, lid_lower))} "
        f"and higher at the rest.</li>",
    ]

    noise_rows = []
    for st in STATISTICS:
        vals = [ref_pooled[s][st] for s in SEEDS]
        noise_rows.append(
            [
                html.escape(LABEL[st]),
                fmt(min(vals)),
                fmt(max(vals)),
                fmt(pystats.pstdev(vals)),
                fmt(max(vals) - min(vals)),
            ]
        )

    claims = [
        [
            "pooled v0 statistics (best, final)",
            "table above",
            "MEASURED",
            "check_best.json, check_final.json (kv_gate check)",
            "2026-10-19",
        ],
        [
            "real reference, 5 draws",
            "table above",
            "MEASURED",
            "../reference/check_seed{0..4}.json",
            "2026-10-18",
        ],
        [
            "per-layer and per-head values",
            "tables below",
            "MEASURED",
            "pooled_statistics over best.jsonl, final.jsonl, ../reference/seed*.jsonl",
            "2026-10-19",
        ],
        [
            "intrinsic statistics",
            "table below",
            "MEASURED",
            "intrinsic.jsonl (intrinsic.py)",
            "2026-10-19",
        ],
        [
            "best and final differ only at layers " + ", ".join(map(str, differ)),
            "",
            "MEASURED",
            "per-layer pooled values of best.jsonl vs final.jsonl",
            "2026-10-19",
        ],
        [
            "verdict",
            ", ".join(
                f"{n}: {c['verdict']} (exit {c['exit_code']})"
                for n, c in checks.items()
            ),
            "MEASURED",
            "check_*.json",
            "2026-10-19",
        ],
    ]

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>scout_kv v0 Gate</title>
<style>{CSS}</style></head><body><main>
<p class="kicker">scout_kv · v0 · B6</p>
<h1>scout_kv v0 against the real reference</h1>
<p class="lede">One WGAN-GP generator per Llama 4 Scout global layer (12 layers) samples the attention
input h. Keys and queries are projected through the real W_k and W_q of each KV head, giving 96 corpora,
the same as the real reference. Each is searched with IVF and HNSW under the canonical conditions in
<code>gates/scout_kv.yaml</code>. The owner set the bands in that file on 2026-10-19, after seeing this
page with the bands still null. Each band holds all five real draws with room on both sides.
Under them both checkpoints <b>fail</b>, on every statistic except IVF recall at nprobe 24.</p>

<section>
<h2>What the numbers show</h2>
<p>Gate statistics, v0 best checkpoint against the five real draws:</p>
<ul>{"".join(findings)}</ul>
<p>On out-of-distribution queries, HNSW finds v0 corpora easier than real ones, and IVF scans a smaller
share of them relative to key-side queries. IVF recall at the operating point matches. On key-side
queries, v0 is harder than real.</p>
<p>Intrinsic statistics of h (diagnostics, one subsample each):</p>
<ul>{"".join(intr_findings)}</ul>
<p class="note">These are descriptions of the measured values, not explanations. No cause has been tested.</p>
</section>

<section>
<h2>Pooled statistics</h2>
<p>Real: five independent draws over books 12-19. v0: <code>best_generator.pt</code> (best) and
<code>checkpoint_step_30000.pt</code> (final), both measured under their EMA weights, seed 0.
"diff" is v0 minus the mean of the five real draws. "band" is from <code>gates/scout_kv.yaml</code>;
the "best" and "final" columns are each checkpoint's status under it. The shaded area in each chart is the band.</p>
{table(["statistic", "real mean", "real min", "real max", "real spread", "band", "v0 best", "best diff", "best", "v0 final", "final diff", "final"], pooled_rows)}
<div class="legend"><span class="key"><i class="s0"></i>real, 5 draws</span><span class="key"><i class="s1"></i>v0 best</span><span class="key"><i class="s2"></i>v0 final</span><span class="key"><i class="band"></i>gate band</span></div>
<div class="grid2">{"".join(strips)}</div>
</section>

<section>
<h2>Noise floor of the real reference</h2>
<p>Spread of each pooled statistic over the five real draws. A band narrower than this spread would reject real data.</p>
{table(["statistic", "min", "max", "std (population)", "spread"], noise_rows)}
<p class="note">HNSW reach is the fraction of the 96 corpora that reach recall 0.9, so it moves in steps of 1/96.
The key-side statistic is a median over corpora. Its per-corpus values sit just below 1.0, and the median is 1.0 in every real draw.</p>
</section>

<section>
<h2>Per layer</h2>
<p>Each statistic pooled over the 8 KV heads of one layer. At this level reach moves in steps of 1/8.</p>
{LEGEND}
<div class="grid2">{"".join(layer_figs)}</div>
<details><summary>Per-layer values (real: mean of 5 draws; v0: best checkpoint)</summary>
{table(layer_header, layer_rows)}</details>
</section>

<section>
<h2>Per head</h2>
<p>One row per (layer, KV head). Real is the mean over the five draws. Reach is 1 if the corpus reached recall 0.9.
The 8 synthetic heads of one layer are all projected from one shared h sample, so their spread across heads
is not comparable with the real per-head spread, where each head's corpus is an independent mix of books.</p>
<details><summary>All 96 corpora</summary>
{table(["layer / head"] + [f"{h} {k}" for k in ("reach", "rec10", "ivf ratio", "ivf rec", "id rec") for h in ("real", "v0")], head_rows)}</details>
</section>

<section>
<h2>Diagnostics: intrinsic statistics of h</h2>
<p>The four intrinsic statistics from <code>src/eval/ann_difficulty.py</code> (k 100, k_hub 10, nlist 256, metric l2, seed 42),
on 20,000 real h rows from books 12-19 and 20,000 v0 samples per layer. These are not gate statistics. The real
values here come from one subsample, so they have no noise band.</p>
{LEGEND}
<div class="grid2">{"".join(intr_figs)}</div>
<details><summary>Values</summary>
{table(["layer / source"] + INTRINSIC + ["input rows (subsampled to 20,000)", "LID discarded"], intr_rows)}</details>
</section>

<section>
<h2>Claims</h2>
{table(["claim", "value", "MEASURED / ESTIMATE", "source", "when"], claims, num=False)}
<p class="note">Generated by <code>docs/results/scout-kv/v0/make_report.py</code> from records committed next to it.
The bands come from <code>check_best.json</code> and <code>check_final.json</code>, which copy them from <code>gates/scout_kv.yaml</code>.</p>
</section>
</main></body></html>
"""
    (HERE / "report.html").write_text(page, encoding="utf-8")
    print("wrote", HERE / "report.html", len(page), "bytes")


if __name__ == "__main__":
    main()
