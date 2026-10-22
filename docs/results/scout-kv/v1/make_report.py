"""Build docs/results/scout-kv/v1/report.html, the v1 results report for scout_kv.

    PYTHONPATH=. python docs/results/scout-kv/v1/make_report.py [data_dir] [out_html]

`data_dir` defaults to this directory. It must hold, under full/:
v1_best_seed0.jsonl, v1_final_seed0.jsonl, v0_best_seed0.jsonl, v0_best_seed1.jsonl,
check_v1_best_seed0.json, check_v1_final_seed0.json, check_v0_best_seed0.json,
verdict_best.json, verdict_final.json and intrinsic.jsonl; and pilot/verdict_best.json,
pilot/verdict_final.json, selector_noise.json. The shared inputs are read from the
repository: docs/results/scout-kv/reference/seed{0..4}.jsonl,
docs/results/scout-kv/controls/train_books_real.jsonl,
docs/results/scout-kv/v0/intrinsic.jsonl and gates/scout_kv.yaml.

A missing input file, or inputs that disagree with each other, stop the script with an
error; it never writes a partial page. The page is self-contained: every chart is inline
SVG and nothing is loaded from outside. Every number on it is computed from the inputs.
"""

import datetime
import html
import json
import statistics as pystats
import sys
from pathlib import Path

from src.data.kv_capture import GLOBAL_LAYERS
from src.eval.kv_gate import STATISTICS, load_kv_gate, pooled_statistics
from src.kv.judge import WIDENED_LAYER, distance, train_books_targets

HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA / "report.html"
SCOUT = HERE.parent  # docs/results/scout-kv
REPO = SCOUT.parent.parent.parent
REF = SCOUT / "reference"
SEEDS = range(5)

LABEL = {
    "hnsw_reach_k10": "HNSW reach, k=10",
    "hnsw_recall_k10_ef176": "HNSW recall@10 at ef 176",
    "ivf_ratio_k128": "IVF scan ratio, k=128 (ood / id)",
    "ivf_recall_k128_np24": "IVF recall@128 at nprobe 24",
    "hnsw_recall_k10_ef176_id": "HNSW recall@10 at ef 176, key-side queries",
}
INTRINSIC = ["lid_median", "relative_contrast_median", "hubness_skew", "ivf_gini"]
HUB = "hubness_skew"
HUB_MIN_LAYER = 15  # the spec's "layers 15-47"

FILES = {
    "v1_best": DATA / "full/v1_best_seed0.jsonl",
    "v1_final": DATA / "full/v1_final_seed0.jsonl",
    "v0_s0": DATA / "full/v0_best_seed0.jsonl",
    "v0_s1": DATA / "full/v0_best_seed1.jsonl",
    "check_best": DATA / "full/check_v1_best_seed0.json",
    "check_final": DATA / "full/check_v1_final_seed0.json",
    "check_v0": DATA / "full/check_v0_best_seed0.json",
    "verdict_best": DATA / "full/verdict_best.json",
    "verdict_final": DATA / "full/verdict_final.json",
    "intrinsic": DATA / "full/intrinsic.jsonl",
    "pilot_best": DATA / "pilot/verdict_best.json",
    "pilot_final": DATA / "pilot/verdict_final.json",
    "noise": DATA / "selector_noise.json",
    "train_books": SCOUT / "controls/train_books_real.jsonl",
    "v0_intrinsic": SCOUT / "v0/intrinsic.jsonl",
    "gate": REPO / "gates/scout_kv.yaml",
    **{f"ref{s}": REF / f"seed{s}.jsonl" for s in SEEDS},
}


def require_inputs():
    missing = [f"{k}: {p}" for k, p in FILES.items() if not p.is_file()]
    if missing:
        raise SystemExit(
            "missing input files, not writing a report:\n  " + "\n  ".join(missing)
        )


def read_jsonl(p):
    return [
        json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()
    ]


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(v, d=4):
    return "n/a" if v is None else f"{v:.{d}f}"


def per_layer(records):
    return {
        L: pooled_statistics([r for r in records if r["layer"] == L])
        for L in GLOBAL_LAYERS
    }


def check_records(records, name, seed):
    """96 corpora, every global layer x 8 heads once, one checkpoint kind, EMA, one seed."""
    corpora = {(r["layer"], r["kv_head"]) for r in records}
    want = {(L, g) for L in GLOBAL_LAYERS for g in range(8)}
    if len(records) != 96 or corpora != want:
        raise ValueError(f"{name}: not the 96 (layer, head) corpora once each")
    kinds = {(r["checkpoint"], r["generator_weights"], r["seed"]) for r in records}
    if len(kinds) != 1:
        raise ValueError(f"{name}: mixes checkpoint, weights or seed: {kinds}")
    ((ckpt, weights, s),) = kinds
    if weights != "ema" or s != seed:
        raise ValueError(
            f"{name}: weights {weights!r} seed {s}, expected ema seed {seed}"
        )
    return ckpt


def close(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-12


# ---------------------------------------------------------------- SVG charts

W, H, PADL, PADR, PADT, PADB = 440, 210, 52, 12, 26, 34
SERIES_STYLE = {  # css class, marker shape
    "real": ("s0", "circle"),
    "v0": ("s1", "circle"),
    "v1best": ("s3", "circle"),
    "v1final": ("s2", "diamond"),
}


def _band_text(band):
    lo, hi = band
    if lo is None and hi is None:
        return "unset"
    if hi is None:
        return f">= {lo:g}"
    if lo is None:
        return f"<= {hi:g}"
    return f"{lo:g} to {hi:g}"


def _scale(vals):
    vals = [v for v in vals if v is not None]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.05, hi + 0.05
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def _marker(kind, cx, cy, cls, tip):
    if kind == "diamond":
        return (
            f'<rect class="{cls}" x="{cx - 3:.1f}" y="{cy - 3:.1f}" width="6" height="6" '
            f'transform="rotate(45 {cx:.1f} {cy:.1f})"><title>{tip}</title></rect>'
        )
    return f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="3.2"><title>{tip}</title></circle>'


def layer_chart(title, real, series, mark=None):
    """real: {layer: [values]} drawn as a range bar with a mean dot.
    series: [(name, key into SERIES_STYLE, {layer: value})].
    mark: (layer, value, tip) for a hollow ring (the L19 training-books value)."""
    allv = [v for vs in real.values() for v in vs]
    for _, _, d in series:
        allv += list(d.values())
    if mark:
        allv.append(mark[1])
    lo, hi = _scale(allv)
    pw, ph = W - PADL - PADR, H - PADT - PADB
    slots = 1 + len(series)
    step = 4.6

    def x(i):
        return PADL + pw * (i + 0.5) / len(GLOBAL_LAYERS)

    def y(v):
        return PADT + ph * (1 - (v - lo) / (hi - lo))

    def off(j):
        return (j - (slots - 1) / 2) * step

    out = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{PADL}" y="14" class="dlabel">{html.escape(title)}</text>',
    ]
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
        vs = [v for v in real.get(L, []) if v is not None]
        if vs:
            cx = x(i) + off(0)
            out.append(
                f'<line stroke="var(--s0)" stroke-width="4" stroke-opacity=".35" x1="{cx:.1f}" x2="{cx:.1f}" '
                f'y1="{y(min(vs)):.1f}" y2="{y(max(vs)):.1f}"/>'
            )
            m = sum(vs) / len(vs)
            out.append(
                _marker(
                    "circle",
                    cx,
                    y(m),
                    "s0",
                    f"real layer {L}: mean {m:.4f}, range {min(vs):.4f}-{max(vs):.4f}",
                )
            )
        for j, (name, key, d) in enumerate(series, start=1):
            v = d.get(L)
            if v is None:
                continue
            cls, shape = SERIES_STYLE[key]
            out.append(
                _marker(
                    shape,
                    x(i) + off(j),
                    y(v),
                    cls,
                    f"{html.escape(name)} layer {L}: {v:.4f}",
                )
            )
        if mark and mark[0] == L:
            out.append(
                f'<circle cx="{x(i):.1f}" cy="{y(mark[1]):.1f}" r="6" fill="none" stroke="var(--ink)" stroke-width="1.5">'
                f"<title>{html.escape(mark[2])}</title></circle>"
            )
    out.append(
        f'<text class="axis" x="{PADL + pw / 2:.1f}" y="{H - 2}" text-anchor="middle">layer</text>'
    )
    out.append("</svg>")
    return "".join(out)


def strip_chart(title, rows, band):
    """rows: [(label, [values], css class)]; `band` is (min, max), either may be None."""
    allv = [v for _, vs, _ in rows for v in vs] + [v for v in band if v is not None]
    lo, hi = _scale(allv)
    w, pl, pr = 440, 78, 14
    h = 40 + 20 * len(rows)
    pw = w - pl - pr

    def x(v):
        return pl + pw * (v - lo) / (hi - lo)

    out = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{pl}" y="12" class="dlabel">{html.escape(title)}</text>',
    ]
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
            f'<text class="axis" x="{pl - 8}" y="{yy + 4}" text-anchor="end">{html.escape(name)}</text>'
        )
        for v in vals:
            out.append(
                f'<circle class="{cls}" cx="{x(v):.1f}" cy="{yy}" r="4" fill-opacity=".8">'
                f"<title>{html.escape(name)}: {v:.4f}</title></circle>"
            )
    out.append("</svg>")
    return "".join(out)


HUB_LINES = [  # label, css class, dash
    ("real body", "s0", ""),
    ("real tail", "s0", "5 3"),
    ("v1 body (best)", "s3", ""),
    ("v1 tail (best)", "s3", "5 3"),
    ("v0 pooled (best)", "s1", ""),
]


def hub_chart(lines):
    """lines: five {layer: value} dicts in HUB_LINES order."""
    w, h, pl, pr, pt, pb = 760, 330, 56, 16, 16, 40
    allv = [v for d in lines for v in d.values()]
    lo, hi = 0.0, max(allv) * 1.08
    pw, ph = w - pl - pr, h - pt - pb

    def x(i):
        return pl + pw * (i + 0.5) / len(GLOBAL_LAYERS)

    def y(v):
        return pt + ph * (1 - (v - lo) / (hi - lo))

    out = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="hubness skew per layer">'
    ]
    for t in range(6):
        v = lo + (hi - lo) * t / 5
        out.append(
            f'<line class="grid" x1="{pl}" x2="{w - pr}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
        )
        out.append(
            f'<text class="axis" x="{pl - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.3g}</text>'
        )
    for i, L in enumerate(GLOBAL_LAYERS):
        out.append(
            f'<text class="axis" x="{x(i):.1f}" y="{h - 22}" text-anchor="middle">{L}</text>'
        )
    out.append(
        f'<text class="axis" x="{pl + pw / 2:.1f}" y="{h - 6}" text-anchor="middle">layer</text>'
    )
    out.append(
        f'<text class="axis" x="14" y="{pt + ph / 2:.1f}" text-anchor="middle" transform="rotate(-90 14 {pt + ph / 2:.1f})">hubness skew</text>'
    )
    for (name, cls, dash), d in zip(HUB_LINES, lines):
        pts = " ".join(f"{x(i):.1f},{y(d[L]):.1f}" for i, L in enumerate(GLOBAL_LAYERS))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(
            f'<polyline fill="none" stroke="var(--{cls})" stroke-width="2"{da} points="{pts}"/>'
        )
        for i, L in enumerate(GLOBAL_LAYERS):
            out.append(
                f'<circle class="{cls}" cx="{x(i):.1f}" cy="{y(d[L]):.1f}" r="3"><title>{name} layer {L}: {d[L]:.3f}</title></circle>'
            )
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- page

CSS = """
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--s3:#8a4fc7;--bad:#b3261e;--good:#12795a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--s0:#3987e5;--s1:#d95926;--s2:#199e70;--s3:#a77be0;--bad:#f2837c;--good:#4fc39c}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--s0:#3987e5;--s1:#d95926;--s2:#199e70;--s3:#a77be0;--bad:#f2837c;--good:#4fc39c}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.7rem;line-height:1.2;margin:.2em 0 .4em}
h2{font-size:1.15rem;margin:2em 0 .6em;border-bottom:1px solid var(--grid);padding-bottom:.3em}
h3{font-size:1rem;margin:1.2em 0 .4em}
.kicker{color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-size:.75rem;margin:0}
.lede{color:var(--ink2);max-width:72ch}
section{background:var(--surface);border-radius:8px;padding:4px 18px 14px;margin:16px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
svg text{font:11px system-ui,sans-serif;fill:var(--ink)} .axis{fill:var(--muted)}
.grid{stroke:var(--grid);stroke-width:1} .dlabel{font-size:12px;font-weight:600}
.s0{fill:var(--s0)} .s1{fill:var(--s1)} .s2{fill:var(--s2)} .s3{fill:var(--s3)} .band{fill:var(--grid);fill-opacity:.7}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:.85rem;color:var(--ink2);margin:4px 0}
.key i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.key i.s0{background:var(--s0)} .key i.s1{background:var(--s1)} .key i.s2{background:var(--s2);border-radius:2px;transform:rotate(45deg)}
.key i.s3{background:var(--s3)} .key i.band{background:var(--grid);border-radius:2px}
.key i.ring{background:none;border:1.5px solid var(--ink)}
.note{color:var(--muted);font-size:.85rem}
.scroll{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;font-size:.85rem;margin:8px 0}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
table.num td{text-align:right;font-variant-numeric:tabular-nums} table.num td.txt{text-align:left}
.pass{color:var(--good);font-weight:600} .fail{color:var(--bad);font-weight:600}
details summary{cursor:pointer;color:var(--ink2)}
"""


def key(*items):
    return (
        '<div class="legend">'
        + "".join(
            f'<span class="key"><i class="{c}"></i>{html.escape(t)}</span>'
            for c, t in items
        )
        + "</div>"
    )


LAYER_KEY = key(
    ("s0", "real, 5 draws (bar: range, dot: mean)"),
    ("s1", "v0 best"),
    ("s3", "v1 best"),
    ("s2", "v1 final (drawn only where it differs from best)"),
)


def table(header, rows, num=True):
    th = "".join(f"<th>{h}</th>" for h in header)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"txt" if j == 0 else ""}">{c}</td>' for j, c in enumerate(r)
        )
        + "</tr>"
        for r in rows
    )
    return f'<div class="scroll"><table class="{"num" if num else ""}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def status(s):
    return f'<span class="{"pass" if s == "pass" else "fail"}">{html.escape(s)}</span>'


def move_label(c):
    """c: one entry of a `judge full` verdict."""
    if c["requirement"] == "inside":
        return (
            status("pass") + " (must stay inside its band)"
            if c["meets"]
            else status("fail") + " (left its band)"
        )
    if c["band_pass"]:
        return status("pass") + " (inside the band)"
    if c["moves"] and c["borderline"]:
        return '<span class="pass">moves</span> (<b>borderline</b>: closes less than twice the spread)'
    if c["moves"]:
        return '<span class="pass">moves</span>'
    return '<span class="fail">does not move</span>'


def main():
    require_inputs()
    recs = {k: read_jsonl(FILES[k]) for k in ("v1_best", "v1_final", "v0_s0", "v0_s1")}
    ckpt = {
        "v1_best": check_records(recs["v1_best"], "v1 best", 0),
        "v1_final": check_records(recs["v1_final"], "v1 final", 0),
        "v0_s0": check_records(recs["v0_s0"], "v0 seed 0", 0),
        "v0_s1": check_records(recs["v0_s1"], "v0 seed 1", 1),
    }
    if ckpt["v0_s0"] != ckpt["v0_s1"]:
        raise ValueError("v0 seeds 0 and 1 are different checkpoints")
    if ckpt["v1_best"] == ckpt["v1_final"]:
        raise ValueError("v1 best and final are the same checkpoint file name")
    pooled = {k: pooled_statistics(v) for k, v in recs.items()}
    lay = {k: per_layer(v) for k, v in recs.items()}
    ref = {s: read_jsonl(FILES[f"ref{s}"]) for s in SEEDS}
    ref_pooled = {s: pooled_statistics(ref[s]) for s in SEEDS}
    ref_layer = {s: per_layer(ref[s]) for s in SEEDS}
    bands_all = load_kv_gate(FILES["gate"])["statistics"]

    checks = {
        "best": read_json(FILES["check_best"]),
        "final": read_json(FILES["check_final"]),
        "v0": read_json(FILES["check_v0"]),
    }
    verdicts = {
        "best": read_json(FILES["verdict_best"]),
        "final": read_json(FILES["verdict_final"]),
    }
    pooled_of = {"best": "v1_best", "final": "v1_final", "v0": "v0_s0"}
    for name, c in checks.items():
        if c["n_records"] != 96:
            raise ValueError(f"check {name}: n_records {c['n_records']}")
        for st in c["statistics"]:
            if not close(st["value"], pooled[pooled_of[name]][st["statistic"]]):
                raise ValueError(f"check {name} disagrees with its records: {st}")
    for name, v in verdicts.items():
        if v["kind"] != "full" or v["v0_seeds"] != [0, 1]:
            raise ValueError(f"verdict {name}: unexpected kind or seeds")
        for st, c in v["statistics"].items():
            if not close(c["v1"], pooled[pooled_of[name]][st]):
                raise ValueError(
                    f"verdict {name} v1 value for {st} disagrees with records"
                )
            for s, key_ in (("0", "v0_s0"), ("1", "v0_s1")):
                got = (
                    c["v0_by_seed"][s]
                    if s in c["v0_by_seed"]
                    else c["v0_by_seed"][int(s)]
                )
                if not close(got, pooled[key_][st]):
                    raise ValueError(
                        f"verdict {name} v0 seed {s} for {st} disagrees with records"
                    )
    for st in STATISTICS:
        b = bands_all[st]
        for name in ("best", "final"):
            vb = verdicts[name]["statistics"][st]
            if (vb["target_min"], vb["target_max"]) != (b.get("min"), b.get("max")):
                raise ValueError(
                    f"verdict {name} band for {st} differs from gates/scout_kv.yaml"
                )

    intrinsic = read_jsonl(FILES["intrinsic"])
    intr = {}
    for r in intrinsic:
        k = (r["layer"], r["source"], r["flag"])
        if k in intr:
            raise ValueError(f"duplicate intrinsic record {k}")
        intr[k] = r
    for L in GLOBAL_LAYERS:
        for src in ("real", "v1_best"):
            for flag in ("body", "tail"):
                if (L, src, flag) not in intr:
                    raise ValueError(f"intrinsic.jsonl lacks layer {L} {src} {flag}")
    extra = {
        k
        for k in intr
        if k[0] not in GLOBAL_LAYERS
        or k[1] not in ("real", "v1_best", "v1_final")
        or k[2] not in ("body", "tail")
    }
    if extra:
        raise ValueError(f"unexpected intrinsic records {sorted(extra)}")
    v0i = {(r["layer"], r["source"]): r for r in read_jsonl(FILES["v0_intrinsic"])}
    for L in GLOBAL_LAYERS:
        for src in ("real", "v0_best"):
            if (L, src) not in v0i:
                raise ValueError(f"v0 intrinsic.jsonl lacks layer {L} {src}")

    train = train_books_targets(read_jsonl(FILES["train_books"]), WIDENED_LAYER)
    final_differs = [
        L for L in GLOBAL_LAYERS if lay["v1_best"][L] != lay["v1_final"][L]
    ]

    # ------------------------------------------------ verdict section
    gate_rows = []
    for st in STATISTICS:
        row = [html.escape(LABEL[st])]
        b = bands_all[st]
        row.append(html.escape(_band_text((b.get("min"), b.get("max")))))
        for name in ("best", "final"):
            cs = next(x for x in checks[name]["statistics"] if x["statistic"] == st)
            row += [fmt(cs["value"]), status(cs["status"])]
        gate_rows.append(row)
    move_rows = []
    for st in STATISTICS:
        row = [html.escape(LABEL[st])]
        for name in ("best", "final"):
            c = verdicts[name]["statistics"][st]
            row += [
                fmt(c["v1"]),
                fmt(c["v0"]),
                fmt(c["spread"]) + f" ({html.escape(c['spread_source'])})",
                fmt(c["gap_closed"]),
                move_label(c),
            ]
        move_rows.append(row)
    meet = {n: verdicts[n]["all_meet"] for n in verdicts}
    bands_pass = {n: verdicts[n]["all_bands_pass"] for n in verdicts}
    either = any(meet.values())
    v1_check_verdict = {n: checks[n]["verdict"] for n in ("best", "final")}
    n_fail = {
        n: sum(s["status"] != "pass" for s in checks[n]["statistics"])
        for n in ("best", "final")
    }
    n_fail_v0 = sum(s["status"] != "pass" for s in checks["v0"]["statistics"])
    n_stats = len(STATISTICS)
    verdict_text = (
        f"<p><b>Gate (kv_gate check):</b> v1 best {html.escape(v1_check_verdict['best'])} "
        f"({n_stats - n_fail['best']} of {n_stats} bands pass); v1 final "
        f"{html.escape(v1_check_verdict['final'])} ({n_stats - n_fail['final']} of {n_stats} pass); "
        f"v0 best {html.escape(checks['v0']['verdict'])} ({n_stats - n_fail_v0} of {n_stats} pass), "
        f"re-measured on this box.</p>"
        f"<p><b>Moves toward real (spec rule, <code>src/kv/judge.py full</code>):</b> all-best "
        f"{'meets' if meet['best'] else 'does not meet'} every criterion; all-final "
        f"{'meets' if meet['final'] else 'does not meet'} every criterion. The spec's rule is that "
        f"v1 meets a criterion if either checkpoint kind meets it in full, so v1 "
        f"<b>{'meets' if either else 'does not meet'}</b> it. "
        f"All five bands pass: best {'yes' if bands_pass['best'] else 'no'}, final {'yes' if bands_pass['final'] else 'no'}.</p>"
    )

    # ------------------------------------------------ pooled table and strips
    pooled_rows, strips = [], []
    for st in STATISTICS:
        vals = [ref_pooled[s][st] for s in SEEDS]
        b = bands_all[st]
        band = (b.get("min"), b.get("max"))
        pooled_rows.append(
            [
                html.escape(LABEL[st]),
                fmt(pooled["v1_best"][st]),
                fmt(pooled["v1_final"][st]),
                fmt(pooled["v0_s0"][st]),
                fmt(pooled["v0_s1"][st]),
                f"{fmt(min(vals))} to {fmt(max(vals))}",
                html.escape(_band_text(band)),
            ]
        )
        strips.append(
            strip_chart(
                LABEL[st],
                [
                    ("real x5", vals, "s0"),
                    ("v0 seed 0", [pooled["v0_s0"][st]], "s1"),
                    ("v0 seed 1", [pooled["v0_s1"][st]], "s1"),
                    ("v1 best", [pooled["v1_best"][st]], "s3"),
                    ("v1 final", [pooled["v1_final"][st]], "s2"),
                ],
                band,
            )
        )

    # ------------------------------------------------ per-layer charts
    layer_figs = []
    for st in STATISTICS:
        mark = (
            WIDENED_LAYER,
            train[st],
            f"L{WIDENED_LAYER} widened target: training books 0-7 value {train[st]:.4f}",
        )
        layer_figs.append(
            layer_chart(
                LABEL[st],
                {L: [ref_layer[s][L][st] for s in SEEDS] for L in GLOBAL_LAYERS},
                [
                    ("v0 best", "v0", {L: lay["v0_s0"][L][st] for L in GLOBAL_LAYERS}),
                    (
                        "v1 best",
                        "v1best",
                        {L: lay["v1_best"][L][st] for L in GLOBAL_LAYERS},
                    ),
                    (
                        "v1 final",
                        "v1final",
                        {
                            L: lay["v1_final"][L][st]
                            for L in GLOBAL_LAYERS
                            if lay["v1_final"][L][st] != lay["v1_best"][L][st]
                        },
                    ),
                ],
                mark,
            )
        )
    layer_rows = []
    for L in GLOBAL_LAYERS:
        row = [str(L)]
        for st in STATISTICS:
            rv = [ref_layer[s][L][st] for s in SEEDS]
            rv = [v for v in rv if v is not None]
            row += [
                f"{fmt(min(rv))}-{fmt(max(rv))}" if rv else "n/a",
                fmt(lay["v0_s0"][L][st]),
                fmt(lay["v1_best"][L][st]),
                fmt(lay["v1_final"][L][st]),
            ]
        layer_rows.append(row)
    layer_header = ["layer"] + [
        f"{k} {h}"
        for k in ("reach", "rec10", "ivf ratio", "ivf rec", "id rec")
        for h in ("real", "v0", "v1 best", "v1 final")
    ]
    # per-layer count inside the target interval (reference range; L19 stretched to the books value)
    inside_rows = []
    for st in STATISTICS:
        row = [html.escape(LABEL[st])]
        for k in ("v0_s0", "v1_best", "v1_final"):
            n = 0
            for L in GLOBAL_LAYERS:
                rv = [ref_layer[s][L][st] for s in SEEDS]
                lo, hi = min(rv), max(rv)
                if L == WIDENED_LAYER:
                    lo, hi = min(lo, train[st]), max(hi, train[st])
                n += distance(lay[k][L][st], lo, hi) == 0
            row.append(f"{n} of {len(GLOBAL_LAYERS)}")
        inside_rows.append(row)

    # ------------------------------------------------ intrinsic
    def iseries(flag, metric):
        best = {L: intr[(L, "v1_best", flag)][metric] for L in GLOBAL_LAYERS}
        final = {
            L: intr[(L, "v1_final", flag)][metric]
            for L in GLOBAL_LAYERS
            if (L, "v1_final", flag) in intr
        }
        v0 = {L: v0i[(L, "v0_best")][metric] for L in GLOBAL_LAYERS}
        real = {L: [intr[(L, "real", flag)][metric]] for L in GLOBAL_LAYERS}
        return real, [
            ("v0 pooled", "v0", v0),
            ("v1 best", "v1best", best),
            ("v1 final", "v1final", final),
        ]

    intr_figs = []
    for m in INTRINSIC:
        for flag in ("body", "tail"):
            real, ser = iseries(flag, m)
            intr_figs.append(layer_chart(f"{m}, {flag}", real, ser))
    intr_rows = []
    for (L, src, flag), r in sorted(intr.items()):
        intr_rows.append(
            [f"{L} {src} {flag}"]
            + [fmt(r[m], 3) for m in INTRINSIC]
            + [str(r["n_rows"]), str(r.get("lid_discarded_queries", "n/a"))]
        )
    for (L, src), r in sorted(v0i.items()):
        intr_rows.append(
            [f"{L} {src} pooled (v0 file)"]
            + [fmt(r[m], 3) for m in INTRINSIC]
            + [str(r["n_rows"]), str(r.get("lid_discarded_queries", "n/a"))]
        )
    INTR_KEY = key(
        ("s0", "real h, same flag (one subsample, no noise band)"),
        ("s1", "v0 best, all positions pooled (from v0/intrinsic.jsonl)"),
        ("s3", "v1 best, this flag"),
        ("s2", "v1 final, this flag (only where it differs from best)"),
    )

    # ------------------------------------------------ hub figure
    late = [L for L in GLOBAL_LAYERS if L >= HUB_MIN_LAYER]
    early = [L for L in GLOBAL_LAYERS if L < HUB_MIN_LAYER]

    def rng(d, layers, dg=1):
        xs = [d[L] for L in layers]
        return f"{min(xs):.{dg}f}-{max(xs):.{dg}f}"

    hub = {
        "real body": {L: intr[(L, "real", "body")][HUB] for L in GLOBAL_LAYERS},
        "real tail": {L: intr[(L, "real", "tail")][HUB] for L in GLOBAL_LAYERS},
        "v1 body": {L: intr[(L, "v1_best", "body")][HUB] for L in GLOBAL_LAYERS},
        "v1 tail": {L: intr[(L, "v1_best", "tail")][HUB] for L in GLOBAL_LAYERS},
        "v0 pooled": {L: v0i[(L, "v0_best")][HUB] for L in GLOBAL_LAYERS},
        "real pooled": {L: v0i[(L, "real")][HUB] for L in GLOBAL_LAYERS},
    }
    hub_svg = hub_chart(
        [hub[n] for n in ("real body", "real tail", "v1 body", "v1 tail", "v0 pooled")]
    )
    n_late = len(late)
    below_body = sum(hub["v1 body"][L] < hub["real body"][L] for L in late)
    below_tail = sum(hub["v1 tail"][L] < hub["real tail"][L] for L in late)
    ratio_body = [hub["v1 body"][L] / hub["real body"][L] for L in late]
    ratio_tail = [hub["v1 tail"][L] / hub["real tail"][L] for L in late]
    ratio_v0 = [hub["v0 pooled"][L] / hub["real pooled"][L] for L in late]
    v1_over_v0 = sum(hub["v1 body"][L] > hub["v0 pooled"][L] for L in late)
    hub_findings = [
        f"<li>Spec's numbers, checked against <code>v0/intrinsic.jsonl</code>: real pooled skew at layers "
        f"{late[0]}-{late[-1]} is {rng(hub['real pooled'], late)} (spec: 38-47); v0 best across all "
        f"layers is {rng(hub['v0 pooled'], GLOBAL_LAYERS)} (spec: 4-14); real pooled at layers "
        f"{early[0]}-{early[-1]} is {rng(hub['real pooled'], early)}.</li>",
        f"<li>Real h by flag at layers {late[0]}-{late[-1]} (this run's <code>intrinsic.jsonl</code>): body "
        f"{rng(hub['real body'], late)}, tail {rng(hub['real tail'], late)}.</li>",
        f"<li>v1 best at layers {late[0]}-{late[-1]}: body {rng(hub['v1 body'], late)}, tail "
        f"{rng(hub['v1 tail'], late)}. v1 body is below real body at {below_body} of {n_late} layers; v1 tail "
        f"is below real tail at {below_tail} of {n_late}. v1 body is above v0 pooled at {v1_over_v0} of {n_late}.</li>",
        f"<li>Ratio to real: v1 body / real body {min(ratio_body):.2f}-{max(ratio_body):.2f}; v1 tail / real tail "
        f"{min(ratio_tail):.2f}-{max(ratio_tail):.2f}; v0 pooled / real pooled {min(ratio_v0):.2f}-{max(ratio_v0):.2f}. "
        "A ratio of 1 would match real.</li>",
    ]
    hub_rows = []
    for L in GLOBAL_LAYERS:
        fin = [
            fmt(intr[(L, "v1_final", f)][HUB], 2)
            if (L, "v1_final", f) in intr
            else "same as best"
            for f in ("body", "tail")
        ]
        hub_rows.append(
            [str(L)]
            + [
                fmt(hub[n][L], 2)
                for n in ("real body", "real tail", "v1 body", "v1 tail")
            ]
            + fin
            + [fmt(hub["v0 pooled"][L], 2), fmt(hub["real pooled"][L], 2)]
        )

    # ------------------------------------------------ pilot and selector noise
    pilot = {n: read_json(FILES[f"pilot_{n}"]) for n in ("best", "final")}
    pilot_rows = [
        [n, "go" if v["go"] else "no go", f"seeds {', '.join(map(str, v['seeds']))}"]
        for n, v in pilot.items()
    ]
    noise = read_json(FILES["noise"])

    # ------------------------------------------------ reference noise
    noise_rows = []
    for st in STATISTICS:
        vals = [ref_pooled[s][st] for s in SEEDS]
        c = verdicts["best"]["statistics"][st]
        noise_rows.append(
            [
                html.escape(LABEL[st]),
                fmt(min(vals)),
                fmt(max(vals)),
                fmt(pystats.pstdev(vals)),
                fmt(max(vals) - min(vals)),
                fmt(abs(pooled["v0_s0"][st] - pooled["v0_s1"][st])),
            ]
        )
    today = datetime.date.today().isoformat()
    claims = [
        [
            "pooled v1 and v0 statistics",
            "pooled table",
            "MEASURED",
            "full/*.jsonl via pooled_statistics; full/check_*.json (kv_gate check)",
        ],
        [
            "moves-toward-real verdicts",
            "verdict section",
            "MEASURED",
            "full/verdict_best.json, full/verdict_final.json (python -m src.kv.judge full)",
        ],
        [
            "real reference, 5 draws",
            "pooled and per-layer figures",
            "MEASURED",
            "docs/results/scout-kv/reference/seed0..4.jsonl",
        ],
        [
            "intrinsic statistics per flag",
            "intrinsic and hub sections",
            "MEASURED",
            "full/intrinsic.jsonl (v1/intrinsic.py)",
        ],
        [
            "v0 pooled intrinsic statistics",
            "hub and intrinsic sections",
            "MEASURED",
            "docs/results/scout-kv/v0/intrinsic.jsonl",
        ],
        [
            "L19 training-books values",
            "per-layer figures",
            "MEASURED",
            "docs/results/scout-kv/controls/train_books_real.jsonl",
        ],
        [
            "v1 final differs from best at layers "
            + (", ".join(map(str, final_differs)) or "none"),
            "",
            "MEASURED",
            "per-layer pooled values of the two v1 record files",
        ],
    ]
    claims = [r + [today] for r in claims]
    checkpoints = ", ".join(f"{k}: {html.escape(v)}" for k, v in ckpt.items())

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>scout_kv v1 Results</title>
<style>{CSS}</style></head><body><main>
<p class="kicker">scout_kv · v1 · full run</p>
<h1>scout_kv v1 against v0 and the real reference</h1>
<p class="lede">v1 is v0 plus a body/tail position flag on the generator and critic. One generator per Llama 4 Scout
global layer (12 layers) samples the attention input h. Keys and queries are projected through the real W_k and W_q of
each KV head, giving 96 corpora, searched with IVF and HNSW under the conditions in <code>gates/scout_kv.yaml</code>.
v0 best, re-measured for this report, fails {n_fail_v0} of the {n_stats} bands (<code>full/check_v0_best_seed0.json</code>). Every number on this page
is computed from the record files named in each section.</p>

<section>
<h2>Verdict</h2>
{verdict_text}
<h3>Gate check against the bands</h3>
{table(["statistic", "band", "v1 best", "best", "v1 final", "final"], gate_rows)}
<h3>Moves toward real</h3>
<p>Rule (spec, "How v1 is judged"): a statistic moves toward real if v1 is inside the target (here the band), or v1's
distance outside the band is smaller than v0's by more than the spread. The spread is the pooled spread over the five
real draws, except for key-side recall, where it is v0's own difference between the two sampling seeds. A move that
closes the gap by less than twice the spread is labelled borderline; it still counts. v0 is the seed-0 measurement.
<code>ivf_recall_k128_np24</code> must stay inside its band. A checkpoint kind is judged as a whole: all-best or
all-final, never mixed across layers. v1 meets a criterion if either kind meets it in full.</p>
{table(["statistic", "best: v1", "v0", "spread", "gap closed", "best result", "final: v1", "v0", "spread", "gap closed", "final result"], move_rows)}
<p class="note">"Gap closed" is v0's distance from the band minus v1's, in the statistic's units. From
<code>full/verdict_best.json</code> and <code>full/verdict_final.json</code>.</p>
</section>

<section>
<h2>Pooled statistics</h2>
<p>Pooled over the 96 corpora. v1: {html.escape(ckpt["v1_best"])} (best) and {html.escape(ckpt["v1_final"])} (final), both
under EMA weights, sampling seed 0. v0: {html.escape(ckpt["v0_s0"])} re-measured on the same box at sampling seeds 0 and 1.
Real: five independent draws over books 12-19. The shaded area in each chart is the band.</p>
{table(["statistic", "v1 best", "v1 final", "v0 seed 0", "v0 seed 1", "real min to max", "band"], pooled_rows)}
{key(("s0", "real, 5 draws"), ("s1", "v0 (seeds 0 and 1)"), ("s3", "v1 best"), ("s2", "v1 final"), ("band", "gate band"))}
<div class="grid2">{"".join(strips)}</div>
<h3>Spread of the references</h3>
<p>Spread of each pooled statistic over the five real draws, and the absolute difference between v0's two sampling seeds.</p>
{table(["statistic", "real min", "real max", "real std (population)", "real spread", "v0 seed 0 vs 1"], noise_rows)}
</section>

<section>
<h2>Per layer</h2>
<p>Each statistic pooled over the 8 KV heads of one layer; reach moves in steps of 1/8. The open ring at layer
{WIDENED_LAYER} is the value measured on the training books 0-7 (<code>controls/train_books_real.jsonl</code>). At that
layer only, the spec widens the target interval from the reference range out to this value. Per-layer views are
diagnostics, not gate verdicts.</p>
{LAYER_KEY}{key(("ring", f"L{WIDENED_LAYER} training-books value (widened target)"))}
<div class="grid2">{"".join(layer_figs)}</div>
<h3>Layers inside the target interval</h3>
<p>Target interval per layer: min-max of the five real draws; at L{WIDENED_LAYER} stretched to the training-books
value. Counted over the 12 layers, with v0 at sampling seed 0.</p>
{table(["statistic", "v0", "v1 best", "v1 final"], inside_rows)}
<details><summary>Per-layer values</summary>
{table(layer_header, layer_rows)}</details>
<p class="note">v1 final differs from v1 best at layers {", ".join(map(str, final_differs)) or "none"}; elsewhere the selector chose the last step.</p>
</section>

<section>
<h2>Intrinsic diagnostics of h, per flag</h2>
<p>The four intrinsic statistics from <code>src/eval/ann_difficulty.py</code> (k 100, k_hub 10, nlist 256, metric l2,
seed 42), computed by <code>v1/intrinsic.py</code> separately for the body flag (20,000 rows) and the tail flag
(8,192 rows). Real h is books 12-19 under the same flag. The orange v0 points are v0's pooled values (all positions,
<code>v0/intrinsic.jsonl</code>), so they are context, not a like-for-like comparison. These are not gate statistics,
and real values come from one subsample, so there is no noise band.</p>
{INTR_KEY}
<div class="grid2">{"".join(intr_figs)}</div>
<details><summary>Values</summary>
{table(["layer / source / flag"] + INTRINSIC + ["input rows", "LID discarded"], intr_rows)}</details>
</section>

<section>
<h2>Hubs</h2>
<p>The spec's question: does the flag bring back the hubs of real h at layers {late[0]}-{late[-1]}, where real pooled skew
is high and v0's is low? Hubness skew per layer, by flag.</p>
{key(("s0", "real (solid: body, dashed: tail)"), ("s3", "v1 best (solid: body, dashed: tail)"), ("s1", "v0 best, pooled"))}
{hub_svg}
<ul>{"".join(hub_findings)}</ul>
<p class="note">These describe the measured values; no cause has been tested. The v1 samples are 20,000 body rows and 8,192 tail rows,
and skew may depend on the number of rows; real tail and v1 tail use equal row counts, while v0 pooled uses a different count.</p>
<details><summary>Values (hubness skew)</summary>
{table(["layer", "real body", "real tail", "v1 body (best)", "v1 tail (best)", "v1 body (final)", "v1 tail (final)", "v0 pooled (best)", "real pooled"], hub_rows)}</details>
</section>

<section>
<h2>Caveats</h2>
<ul>
<li><b>No v0 retrain spread.</b> The family has no measurement of how much a retrain of the same v0 configuration moves each statistic. The only noise controls are the five real draws, v0 at two sampling seeds (no retraining), and the borderline label. Same-config retrains in other families moved statistics by more than the reference spread (NYTimes v0, 30k LID 18.0 to 13.8, as quoted in the v1 spec, not re-measured here). The owner chose not to add a v0 retrain.</li>
<li><b>Hardware.</b> All v0 and v1 inputs on this page were measured on the same L40 box (<code>tig-gpu-alt</code>): v0 best was re-measured there, so v0 here may differ from the numbers in the committed v0 report ({html.escape(str(SCOUT / "v0" / "report.html").replace(str(REPO) + "/", ""))}). The real reference draws are earlier records, not re-measured.</li>
<li><b>One sampling seed for v1.</b> v1 is measured at sampling seed 0 only; the second seed exists for v0, and sets the key-side allowance.</li>
<li><b>Checkpoint kinds.</b> Best and final are judged separately and not mixed across layers. Checkpoints measured: {checkpoints}.</li>
<li><b>Intrinsic diagnostics</b> are single subsamples with no noise band, and the real v0-pooled and per-flag real values come from different subsamples.</li>
<li><b>Inputs.</b> The script stops with an error if any input file is missing or if the check, verdict and record files disagree; a page that exists was built from complete inputs.</li>
</ul>
<h3>Pilot and selector noise (committed earlier)</h3>
{table(["pilot checkpoint kind", "go/no-go", "sampling seeds"], pilot_rows, num=False)}
<p class="note">From <code>pilot/verdict_best.json</code> and <code>pilot/verdict_final.json</code> (layers 19 and 43). Selector-noise
file <code>selector_noise.json</code> covers layers {html.escape(", ".join(sorted(noise.get("layers", {}), key=int)))}.</p>
</section>

<section>
<h2>Claims</h2>
{table(["claim", "where", "MEASURED / ESTIMATE", "source", "report generated"], claims, num=False)}
<p class="note">Generated by <code>docs/results/scout-kv/v1/make_report.py</code>. Bands come from <code>gates/scout_kv.yaml</code>
through <code>src/eval/kv_gate.py</code>.</p>
</section>
</main></body></html>
"""
    OUT.write_text(page, encoding="utf-8")
    print("wrote", OUT, len(page), "bytes")


if __name__ == "__main__":
    main()
