"""Render the ANN benchmark's complete records as one self-contained HTML page.

The JSON files stay the source of truth; this is a readable view of them.
No external assets: everything is inline so the file works from disk.
"""

import html
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
BASE = OUT.parent
A = json.loads((BASE / "ann_benchmark.json").read_text())
B = json.loads((BASE / "ann_benchmark_repeat.json").read_text())

CORPORA = ("real", "v0", "v1", "v1_5", "v2", "v3", "v4")
INDEXES = ("flat", "ivf_flat", "ivf_pq", "cagra")
env = A["environment"]


def n(x, dp=0):
    if x is None:
        return "—"
    return f"{x:,.{dp}f}"


def key(r):
    return (
        r["corpus"],
        r["index"],
        r["param_value"] if r["param_value"] is not None else -1,
    )


b_by = {(r["corpus"], r["index"]): r for r in A["builds"]}
s_a = {key(r): r for r in A["searches"]}
s_b = {key(r): r for r in B["searches"]}

# ---------------------------------------------------------------- build table
build_rows = []
for idx in INDEXES:
    for c in CORPORA:
        r = b_by[(c, idx)]
        size = r.get("index_bytes_estimated") or r.get("index_bytes")
        vram = r.get("peak_vram_bytes")
        params = ", ".join(f"{k}={v}" for k, v in sorted(r["params"].items()))
        build_rows.append(
            f"<tr data-c='{c}' data-i='{idx}'>"
            f"<td class='k'>{c}</td><td class='k'>{idx}</td>"
            f"<td class='num'>{n(r['train_seconds'], 2)}</td>"
            f"<td class='num'>{n(r['add_seconds'], 2)}</td>"
            f"<td class='num'>{n(size / 1e6, 1)}</td>"
            f"<td class='num'>{n(vram / 2**20, 0) if vram else '—'}</td>"
            f"<td class='p'>{html.escape(params)}</td>"
            f"<td>{'<span class=bad>' + html.escape(str(r['failed'])) + '</span>' if r['failed'] else '<span class=ok>ok</span>'}</td>"
            "</tr>"
        )

# --------------------------------------------------------------- search table
search_rows = []
for c in CORPORA:
    for idx in INDEXES:
        pts = sorted(
            (k for k in s_a if k[0] == c and k[1] == idx),
            key=lambda t: t[2],
        )
        for k_ in pts:
            ra = s_a[k_]
            rb = s_b.get(k_)
            pv = ra["param_value"]
            pn = ra["param_name"] or "—"
            d_rec = (
                (rb["recall"] - ra["recall"])
                if rb and rb["recall"] is not None and ra["recall"] is not None
                else None
            )
            d_qps = (
                100 * (rb["qps_median"] - ra["qps_median"]) / ra["qps_median"]
                if rb and rb["qps_median"] and ra["qps_median"]
                else None
            )
            spread = (
                100 * (ra["qps_p95"] - ra["qps_min"]) / ra["qps_min"]
                if ra["qps_min"]
                else None
            )
            search_rows.append(
                f"<tr data-c='{c}' data-i='{idx}'>"
                f"<td class='k'>{c}</td><td class='k'>{idx}</td>"
                f"<td class='p'>{pn}</td>"
                f"<td class='num'>{pv if pv is not None else '—'}</td>"
                f"<td class='num'>{n(ra['recall'], 4)}</td>"
                f"<td class='num'>{n(ra['qps_min'])}</td>"
                f"<td class='num strong'>{n(ra['qps_median'])}</td>"
                f"<td class='num'>{n(ra['qps_p95'])}</td>"
                f"<td class='num dim'>{n(spread, 1) + '%' if spread is not None else '—'}</td>"
                f"<td class='num dim'>{f'{d_rec:+.4f}' if d_rec is not None else '—'}</td>"
                f"<td class='num dim'>{f'{d_qps:+.1f}%' if d_qps is not None else '—'}</td>"
                "</tr>"
            )

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--dim:#6b7280;--line:#e5e7eb;--head:#f9fafb;
--acc:#1d4ed8;--ok:#047857;--bad:#b91c1c;--zebra:#fcfcfd}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e6e8eb;--dim:#9aa1ab;
--line:#262a31;--head:#171a20;--acc:#7aa2ff;--ok:#34d399;--bad:#f87171;--zebra:#12151a}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
max-width:1200px;margin-inline:auto}
h1{font-size:1.6rem;margin:0 0 .35rem}
h2{font-size:1.15rem;margin:2.5rem 0 .5rem;padding-bottom:.3rem;border-bottom:1px solid var(--line)}
p{color:var(--dim);margin:.4rem 0}
.lede{color:var(--fg)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88em;
background:var(--head);padding:.1em .35em;border-radius:3px}
.meta{display:flex;flex-wrap:wrap;gap:.4rem 1.5rem;margin:1rem 0;padding:.85rem 1rem;
background:var(--head);border:1px solid var(--line);border-radius:6px;font-size:.87rem}
.meta div{white-space:nowrap}.meta b{color:var(--dim);font-weight:500}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px;margin-top:.6rem}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th{position:sticky;top:0;background:var(--head);text-align:left;font-weight:600;
padding:.5rem .6rem;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:.35rem .6rem;border-bottom:1px solid var(--line);white-space:nowrap}
tr:nth-child(even) td{background:var(--zebra)}
tr:last-child td{border-bottom:0}
.num{text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.k{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.p{color:var(--dim);font-size:.92em}
.dim{color:var(--dim)}.strong{font-weight:600}
.ok{color:var(--ok)}.bad{color:var(--bad)}
.filters{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;margin:.9rem 0 .2rem}
button{font:inherit;font-size:.82rem;padding:.25rem .6rem;border:1px solid var(--line);
background:var(--bg);color:var(--fg);border-radius:999px;cursor:pointer}
button:hover{border-color:var(--acc)}button[aria-pressed=true]{background:var(--acc);
border-color:var(--acc);color:#fff}
.filters span{color:var(--dim);font-size:.8rem;margin-right:.2rem}
.note{border-left:3px solid var(--acc);padding:.5rem 0 .5rem .8rem;margin:1rem 0;
color:var(--dim);font-size:.9rem}
"""

JS = """
function wire(id){
  const root=document.getElementById(id);
  root.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
    const g=b.dataset.group, on=b.getAttribute('aria-pressed')==='true';
    root.querySelectorAll(`button[data-group="${g}"]`).forEach(x=>x.setAttribute('aria-pressed','false'));
    b.setAttribute('aria-pressed', on?'false':'true');
    const c=root.querySelector('button[data-group=c][aria-pressed=true]');
    const i=root.querySelector('button[data-group=i][aria-pressed=true]');
    document.querySelectorAll(`#${root.dataset.table} tbody tr`).forEach(tr=>{
      const okc=!c||tr.dataset.c===c.dataset.v, oki=!i||tr.dataset.i===i.dataset.v;
      tr.style.display=(okc&&oki)?'':'none';
    });
  }));
}
wire('f-build');wire('f-search');
"""


def filters(fid, table):
    cs = "".join(
        f"<button data-group=c data-v='{c}' aria-pressed=false>{c}</button>"
        for c in CORPORA
    )
    is_ = "".join(
        f"<button data-group=i data-v='{i}' aria-pressed=false>{i}</button>"
        for i in INDEXES
    )
    return (
        f"<div class='filters' id='{fid}' data-table='{table}'>"
        f"<span>corpus</span>{cs}<span style='margin-left:.8rem'>index</span>{is_}</div>"
    )


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ANN benchmark — complete records</title>
<style>{CSS}</style></head><body>

<h1>GPU ANN benchmark — complete records</h1>
<p class="lede">Every measured cell from the run published in
<code>docs/results/ann-gpu-benchmark/</code>, rendered from
<code>ann_benchmark.json</code> and <code>ann_benchmark_repeat.json</code>.
Those files remain the source of truth; this page adds no numbers of its own.</p>

<div class="meta">
<div><b>corpora</b> {len(CORPORA)} × {env["num_vectors"]:,} × 128 float32</div>
<div><b>queries</b> {env["num_queries"]:,}</div>
<div><b>k</b> {env["k"]}</div>
<div><b>repeats</b> {env["repeats"]} (+1 discarded warmup)</div>
<div><b>target recall</b> {env["target_recall"]}</div>
<div><b>normalized</b> {env["normalized"]}</div>
<div><b>metric</b> squared L2</div>
<div><b>python</b> {env["python"]}</div>
</div>

<p>Run A is the published run (commit <code>7c00b45</code>, job
<code>…20260813T080256Z-3b6ec3</code>). Run B is the identical repeat used for the
noise floor (job <code>…20260813T084709Z-30d589</code>), over the same cached
corpora, so only the search is re-measured.</p>

<h2>Build records — {len(build_rows)} cells</h2>
<p>Train and add phases are timed separately; a partitioning index and a graph
index pay them in very different proportions.</p>
{filters("f-build", "t-build")}
<div class="wrap"><table id="t-build"><thead><tr>
<th>corpus</th><th>index</th><th>train (s)</th><th>add (s)</th>
<th>index size (MB, est.)</th><th>peak VRAM (MiB)</th><th>build params</th><th>status</th>
</tr></thead><tbody>
{"".join(build_rows)}
</tbody></table></div>
<div class="note"><b>Index size is an analytic estimate</b>, not a measured
allocation: it omits IVF coarse centroids and PQ codebooks, so IVF-PQ's figure
understates its true footprint. <b>Peak VRAM is a card-wide delta</b> sampled
around the build, not a per-index allocation.</div>

<h2>Search sweeps — {len(search_rows)} cells</h2>
<p>One row per (corpus, index, swept parameter). QPS is over
{env["repeats"]} timed repeats of all {env["num_queries"]:,} queries issued in a
single batch, each region fenced with a stream synchronization on both sides.
The last two columns are run B minus run A.</p>
{filters("f-search", "t-search")}
<div class="wrap"><table id="t-search"><thead><tr>
<th>corpus</th><th>index</th><th>knob</th><th>value</th><th>recall@10</th>
<th>QPS min</th><th>QPS median</th><th>QPS p95</th><th>spread</th>
<th>Δrecall (B−A)</th><th>ΔQPS (B−A)</th>
</tr></thead><tbody>
{"".join(search_rows)}
</tbody></table></div>
<div class="note"><b>spread</b> is (p95 − min) / min for run A — how much the five
repeats of that one cell varied. <b>Δ columns</b> compare the two independent
runs: they bound run-to-run noise, not generator-seed noise, since both runs
reuse the same cached corpora.</div>

<h2>Reading these numbers</h2>
<p>All figures describe <b>L2-normalized</b> corpora, the real one included, and
are not comparable with published SIFT1M results. The <code>flat</code> rows are
exact search at recall 1.000 by construction — the check that ground truth and
search share a distance space. A search cell reaching the target recall nowhere
in its sweep is a real result, reported as a null headline rather than an
extrapolation; see the README's IVF-PQ section.</p>
<p>This page is a measurement record. It does not adjudicate the ANN-difficulty
gate and does not rank the variants.</p>

<script>{JS}</script>
</body></html>
"""

OUT.write_text(HTML)
print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
print(f"  build rows  {len(build_rows)}")
print(f"  search rows {len(search_rows)}")
