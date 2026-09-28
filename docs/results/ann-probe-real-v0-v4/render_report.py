"""Render report.html for the real / v0 / v4 ANN probe from records.json.

    PYTHONPATH=. python docs/results/ann-probe-real-v0-v4/render_report.py

This exists instead of `src.eval.ann_benchmark.report.write_html` because that
figure lets plotly's colour cycle run across panels (so one corpus changes
colour from facet to facet) and shares one y range across indexes whose QPS
spans differ by 10x. Numbers come from the same `headline_rows` /
`qps_at_recall` the harness uses.
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.eval.ann_benchmark import report
from src.eval.ann_benchmark.runner import BuildRecord, SearchRecord

HERE = Path(__file__).resolve().parent
CORPORA = ("real", "v0", "v4")
# Categorical slots 1-3 of the dataviz default palette, fixed order; each
# corpus also gets its own marker so identity is never colour alone.
COLOR = {"real": "#2a78d6", "v0": "#eb6834", "v4": "#1baf7a"}
SYMBOL = {"real": "circle-open", "v0": "square", "v4": "diamond"}
# real is drawn last, dashed with open markers, so v0 lying on top of it
# cannot hide it; legendrank keeps it first in the legend.
DRAW_ORDER = ("v0", "v4", "real")
INDEXES = ("ivf_flat", "ivf_pq", "cagra", "cagra_iters")
TITLE = {
    "ivf_flat": "IVF-Flat (cluster) — n_probes",
    "ivf_pq": "IVF-PQ (cluster) — n_probes",
    "cagra": "CAGRA (graph) — itopk_size",
    "cagra_iters": "CAGRA (graph) — max_iterations",
}
TARGETS = (0.90, 0.95)

ENV = {
    "job": "wgan-synthetic-20260928T081320Z-1a2cfc (gpuq, exclusive GPU lane, exit 0)",
    "commit": "9f776dee6b5afc6be2ea7efb952b53756be403b7",
    "gpu": "NVIDIA GeForce RTX 3060 Ti, 8192 MiB, driver 580.178.04",
    "stack": "torch 2.13.0+cu130, cuvs 26.08.01, cupy 14.2.0",
    "records.json sha256": (
        "d6d085321f575d6cae94bab69df3b115eac24aa6dd2ddd19a5419d93a56fac88"
    ),
}


def load():
    rec = json.loads((HERE / "records.json").read_text())
    builds = [BuildRecord(**b) for b in rec["builds"]]
    searches = [SearchRecord(**s) for s in rec["searches"]]
    return builds, searches


def curves(searches):
    out = defaultdict(list)
    for s in searches:
        if s.recall is not None:
            out[(s.corpus, s.index)].append(s)
    for v in out.values():
        v.sort(key=lambda s: (s.param_value or 0))
    return out


def trace(corpus, x, y, show, hover):
    return go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name=corpus,
        legendgroup=corpus,
        showlegend=show,
        line={
            "color": COLOR[corpus],
            "width": 2,
            "dash": "dash" if corpus == "real" else "solid",
        },
        legendrank=CORPORA.index(corpus),
        marker={
            "color": COLOR[corpus],
            "symbol": SYMBOL[corpus],
            "size": 10 if corpus == "real" else 8,
            "line": {
                "color": COLOR[corpus] if corpus == "real" else "#fcfcfb",
                "width": 2 if corpus == "real" else 1.5,
            },
        },
        hovertemplate=hover + f"<extra>{corpus}</extra>",
    )


def fig_pareto(cur):
    fig = make_subplots(rows=1, cols=4, subplot_titles=[TITLE[i] for i in INDEXES])
    for col, index in enumerate(INDEXES, start=1):
        for corpus in DRAW_ORDER:
            pts = sorted(cur[(corpus, index)], key=lambda s: s.recall)
            fig.add_trace(
                trace(
                    corpus,
                    [s.recall for s in pts],
                    [s.qps_median for s in pts],
                    col == 1,
                    "recall %{x:.3f}<br>%{y:,.0f} QPS<br>"
                    + f"{pts[0].param_name}="
                    + "%{customdata}",
                ),
                row=1,
                col=col,
            )
            fig.data[-1].customdata = [s.param_value for s in pts]
        for t in TARGETS:
            fig.add_vline(x=t, line_dash="dot", line_color="#8a8984", row=1, col=col)
        fig.update_xaxes(title_text="recall@10", row=1, col=col)
        fig.update_yaxes(type="log", row=1, col=col)
    fig.update_yaxes(title_text="queries / second (median of 5)", row=1, col=1)
    # CAGRA's itopk sweep starts at recall 0.96; zoom it so the curves separate.
    fig.update_xaxes(range=[0.955, 1.002], row=1, col=3)
    return layout(fig)


def fig_equal_knob(cur):
    fig = make_subplots(rows=1, cols=4, subplot_titles=[TITLE[i] for i in INDEXES])
    for col, index in enumerate(INDEXES, start=1):
        for corpus in DRAW_ORDER:
            pts = cur[(corpus, index)]
            fig.add_trace(
                trace(
                    corpus,
                    [s.param_value for s in pts],
                    [s.recall for s in pts],
                    col == 1,
                    f"{pts[0].param_name}=" + "%{x}<br>recall %{y:.3f}",
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(
            title_text=cur[("real", index)][0].param_name, type="log", row=1, col=col
        )
    fig.update_yaxes(title_text="recall@10", row=1, col=1)
    return layout(fig)


def layout(fig):
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin={"t": 60, "l": 60, "r": 20, "b": 60},
        legend={"orientation": "h", "y": 1.18, "x": 0},
        font={"color": "#0b0b0b"},
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        hovermode="closest",
    )
    fig.update_annotations(font_size=13)
    return fig


def fmt(x, dp=0):
    return "—" if x is None else f"{x:,.{dp}f}"


def qps_cell(builds, searches, corpus, index, target):
    cur = [
        (s.recall, s.qps_median)
        for s in searches
        if s.corpus == corpus and s.index == index and s.recall is not None
    ]
    from src.eval.ann_benchmark.metrics import qps_at_recall

    p = qps_at_recall(cur, target)
    if p is None:
        return None, f"not reached (peak {max(r for r, _ in cur):.3f})"
    if not p.interpolated and p.recall > target:
        return p.qps, f"{p.qps:,.0f} (floor @ {p.recall:.3f})"
    return p.qps, f"{p.qps:,.0f}"


def tables(builds, searches):
    rows = {
        (r["corpus"], r["index"]): r
        for r in report.headline_rows(builds, searches, target_recall=0.90)
    }
    order = ("flat", "ivf_flat", "ivf_pq", "cagra", "cagra_iters")
    b = [
        "<tr><th>Index</th>"
        + "".join(f"<th>{c}</th>" for c in CORPORA)
        + "<th>index MB (est.)</th></tr>"
    ]
    for index in order:
        cells = "".join(
            f"<td>{fmt(rows[(c, index)]['build_seconds'], 2)}</td>" for c in CORPORA
        )
        mb = rows[("real", index)]["index_bytes_estimated"] / 1e6
        b.append(f"<tr><th>{index}</th>{cells}<td>{mb:,.0f}</td></tr>")

    q = [
        "<tr><th>Index</th><th>target</th>"
        + "".join(f"<th>{c}</th>" for c in CORPORA)
        + "<th>v0 / real</th><th>v4 / real</th></tr>"
    ]
    for index in order:
        if index == "flat":
            vals = [rows[(c, index)]["exact_qps"] for c in CORPORA]
            q.append(
                "<tr><th>flat</th><td>exact</td>"
                + "".join(f"<td>{fmt(v)}</td>" for v in vals)
                + f"<td>{vals[1] / vals[0]:.3f}</td><td>{vals[2] / vals[0]:.3f}</td></tr>"
            )
            continue
        for t in TARGETS:
            got = [qps_cell(builds, searches, c, index, t) for c in CORPORA]
            ratio = [
                "—" if got[0][0] is None or g[0] is None else f"{g[0] / got[0][0]:.2f}"
                for g in got[1:]
            ]
            q.append(
                f"<tr><th>{index}</th><td>{t:.2f}</td>"
                + "".join(f"<td>{html.escape(g[1])}</td>" for g in got)
                + "".join(f"<td>{r}</td>" for r in ratio)
                + "</tr>"
            )

    full = [
        "<tr><th>Index</th><th>knob</th>"
        + "".join(f"<th>{c} recall</th><th>{c} QPS</th>" for c in CORPORA)
        + "</tr>"
    ]
    by = defaultdict(dict)
    for s in searches:
        by[(s.index, s.param_name, s.param_value)][s.corpus] = s
    for index, pname, pval in sorted(by, key=lambda k: (order.index(k[0]), k[2] or 0)):
        r = by[(index, pname, pval)]
        full.append(
            f"<tr><th>{index}</th><td>{f'{pname}={pval}' if pname else 'exact'}</td>"
            + "".join(
                f"<td>{r[c].recall:.3f}</td><td>{fmt(r[c].qps_median)}</td>"
                for c in CORPORA
            )
            + "</tr>"
        )
    wrap = lambda rs: "<table>" + "".join(rs) + "</table>"  # noqa: E731
    return wrap(b), wrap(q), wrap(full)


CSS = """
:root { color-scheme: light; }
body { background:#fcfcfb; color:#0b0b0b; margin:0; padding:24px 16px;
  font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }
main { max-width:1280px; margin:0 auto; }
h1 { font-size:26px; margin:0 0 4px; } h2 { font-size:19px; margin:32px 0 8px; }
.meta { color:#52514e; font-size:13px; }
.note { border-left:3px solid #2a78d6; background:#f1f4f8; padding:8px 12px;
  margin:10px 0; }
.wrap { overflow-x:auto; }
table { border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:13px; }
th, td { border:1px solid #e3e2dd; padding:4px 10px; text-align:right; white-space:nowrap; }
th { background:#f4f3ef; text-align:left; }
ul { padding-left:20px; }
"""


def main():
    builds, searches = load()
    cur = curves(searches)
    build_t, qps_t, full_t = tables(builds, searches)
    env = "".join(
        f"<div><b>{html.escape(k)}</b>: {html.escape(v)}</div>" for k, v in ENV.items()
    )
    findings = (HERE / "findings.html").read_text()
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>SIFT ANN probe</title>"
        f"<style>{CSS}</style>{report._plotlyjs_script()}</head><body><main>"
        "<h1>ANN probe: real SIFT vs v0 vs v4</h1>"
        f"<div class='meta'>{env}</div>"
        f"{findings}"
        "<h2>Recall vs throughput</h2>"
        "<div class='note'>Dotted lines mark recall 0.90 and 0.95. Each panel "
        "has its own log QPS axis. The CAGRA itopk panel is zoomed to recall "
        "≥ 0.955 because its cheapest setting already clears 0.96.</div>"
        + fig_pareto(cur).to_html(full_html=False, include_plotlyjs=False)
        + "<h2>Recall at equal search settings</h2>"
        "<div class='note'>Same index, same knob value on each corpus. A curve "
        "above real means the corpus is easier to search than real SIFT at "
        "that setting.</div>"
        + fig_equal_knob(cur).to_html(full_html=False, include_plotlyjs=False)
        + "<h2>Build time (s)</h2><div class='wrap'>"
        + build_t
        + "</div>"
        "<p class='meta'>train + add seconds. Build params are identical on all "
        "three corpora: IVF n_lists=4096; IVF-PQ pq_dim=64, pq_bits=8; CAGRA "
        "graph_degree=64, intermediate_graph_degree=128. flat has no build "
        "beyond copying the vectors.</p>"
        "<h2>QPS at target recall</h2><div class='wrap'>" + qps_t + "</div>"
        "<p class='meta'>Interpolated linearly in log QPS between measured "
        "points. 'floor' = every measured point already exceeds the target, so "
        "the fastest point is shown at the recall it reached.</p>"
        "<h2>Every measured point</h2><div class='wrap'>" + full_t + "</div>"
        "</main></body></html>"
    )
    (HERE / "report.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
