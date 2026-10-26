"""Tables and report for the v2 probe: real vs v0 / v1 / probe arms, per layer.

Reads the probe runs (copied from /workspace/kv/v2probe, layout L{LL}/{ARM}_s{SEED}/)
and the committed real, v0 and v1 records. Writes summary.json and report.html.

    python docs/results/scout-kv/v2-probe/compare.py --runs docs/results/scout-kv/v2-probe/runs \
        --out docs/results/scout-kv/v2-probe
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

from src.eval.kv_gate import pooled_statistics

HERE = Path(__file__).resolve().parent
SK = HERE.parent
GATE_STATS = (
    ("hnsw_reach_k10", "HNSW reach"),
    ("hnsw_recall_k10_ef176", "HNSW recall"),
    ("ivf_ratio_k128", "IVF ratio"),
    ("ivf_recall_k128_np24", "IVF recall"),
    ("hnsw_recall_k10_ef176_id", "key-side recall"),
)
LAGS = ("1", "16", "256", "1024", "4096")
ARM_NAMES = {"A": "A token-by-token", "B": "B drifting latent", "C": "C position only"}


def jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def gate_of(records, layer):
    rs = [r for r in records if r["layer"] == layer]
    assert len(rs) == 8, (layer, len(rs))
    return pooled_statistics(rs)


def locality_of(path, layer):
    d = json.loads(Path(path).read_text())
    (r,) = [x for x in d["layers"] if x["layer"] == layer]
    c = r["consecutive_cosine_centred"]
    return {
        **{f"cos_lag{k}": c[k]["median"] for k in LAGS},
        "cos_random_same_book": r["random_same_book_cosine_centred"]["median"],
        "h_nn_same_book": r["h_l2"]["same_book"],
        "h_nn_within_256": r["h_l2"]["same_book_within_256"],
        "key_nn_same_book": r["key_ip"]["same_book"],
        "key_nn_within_2048": r["key_ip"]["same_book_within_2048"],
        "key_nn_median_dpos": r["key_ip"]["median_abs_dpos_same_book"],
        "query_hits_own_book": r["query_hits"]["same_book"],
        "query_hits_last_eighth": r["query_hits"]["hits_in_last_eighth_of_body"],
    }


def hubs_of(path, layer):
    d = json.loads(Path(path).read_text())
    (r,) = [x for x in d["layers"] if x["layer"] == layer]
    b = r["share_hubs_by_bin"]
    return {
        "hub_skew": r["hubness_skew"],
        "hubs_first_eighth": b[0],
        "hubs_last_eighth": b[-1],
        "hub_median_position": r["median_position"],
    }


def intrinsic_of(records, layer, source):
    out = {}
    for r in records:
        if r["layer"] == layer and r["source"] == source:
            f = r["flag"]
            out[f"{f}_lid"] = r["lid_median"]
            out[f"{f}_contrast"] = r["relative_contrast_median"]
            out[f"{f}_hubness"] = r["hubness_skew"]
    return out


def collect(runs: Path, layers):
    v1full = SK / "v1" / "full"
    noise = SK / "v1-diagnostics" / "noise" / "measure"
    v1_intr = jsonl(v1full / "intrinsic.jsonl")
    out = {}
    for L in layers:
        ref = [gate_of(jsonl(SK / "reference" / f"seed{s}.jsonl"), L) for s in range(5)]
        band = {
            k: [min(r[k] for r in ref), max(r[k] for r in ref)] for k, _ in GATE_STATS
        }
        rows = {
            "real": {
                "gate_band": band,
                **locality_of(SK / "locality" / "locality.json", L),
                **hubs_of(SK / "v1-diagnostics" / "hub_position.json", L),
                **intrinsic_of(v1_intr, L, "real"),
            },
            "v0": {"gate": gate_of(jsonl(v1full / "v0_best_seed0.jsonl"), L)},
            "v1 s42 best": {
                "gate": gate_of(jsonl(v1full / "v1_best_seed0.jsonl"), L),
                **intrinsic_of(v1_intr, L, "v1_best"),
            },
            "v1 s42 final": {"gate": gate_of(jsonl(v1full / "v1_final_seed0.jsonl"), L)},
        }
        if (noise / f"v1s1_best_L{L}.jsonl").exists():
            rows["v1 s1 best"] = {"gate": gate_of(jsonl(noise / f"v1s1_best_L{L}.jsonl"), L)}
            rows["v1 s1 final"] = {"gate": gate_of(jsonl(noise / f"v1s1_final_L{L}.jsonl"), L)}
        for arm in "ABC":
            for seed in (42, 1):
                r = runs / f"L{L:02d}" / f"{arm}_s{seed}"
                if not (r / "intrinsic.jsonl").exists():
                    continue
                rows[f"{arm} s{seed}"] = {
                    "gate": gate_of(jsonl(r / "gate.jsonl"), L),
                    **locality_of(r / "locality.json", L),
                    **hubs_of(r / "hub_position.json", L),
                    **intrinsic_of(jsonl(r / "intrinsic.jsonl"), L, f"{arm}_s{seed}"),
                    "train_tail": jsonl(r / "train_log.jsonl")[-1],
                }
        out[L] = rows
    return out


def fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


def gate_table(L, rows):
    band = rows["real"]["gate_band"]
    head = "<tr><th>source</th>" + "".join(f"<th>{n}</th>" for _, n in GATE_STATS) + "</tr>"
    body = ["<tr class=real><td>real (5 draws)</td>" + "".join(
        f"<td>{fmt(band[k][0])}–{fmt(band[k][1])}</td>" for k, _ in GATE_STATS) + "</tr>"]
    for name, r in rows.items():
        if "gate" not in r:
            continue
        cells = []
        for k, _ in GATE_STATS:
            v = r["gate"][k]
            inside = v is not None and band[k][0] <= v <= band[k][1]
            cells.append(f"<td class={'in' if inside else 'out'}>{fmt(v)}</td>")
        body.append(f"<tr><td>{html.escape(name)}</td>{''.join(cells)}</tr>")
    return f"<table><caption>Layer {L}: gate statistics over its 8 KV-head corpora (green = inside the real range)</caption>{head}{''.join(body)}</table>"


def plain_table(L, rows, keys, caption):
    srcs = [n for n, r in rows.items() if any(k in r for k, _ in keys)]
    head = "<tr><th>source</th>" + "".join(f"<th>{html.escape(n)}</th>" for _, n in keys) + "</tr>"
    body = []
    for n in srcs:
        r = rows[n]
        cls = " class=real" if n == "real" else ""
        body.append(f"<tr{cls}><td>{html.escape(n)}</td>" + "".join(
            f"<td>{fmt(r.get(k))}</td>" for k, _ in keys) + "</tr>")
    return f"<table><caption>Layer {L}: {caption}</caption>{head}{''.join(body)}</table>"


LOC_KEYS = [(f"cos_lag{k}", f"cos lag {k}") for k in LAGS] + [
    ("cos_random_same_book", "cos random pair"),
    ("h_nn_same_book", "h-NN same book"),
    ("key_nn_same_book", "key-NN same book"),
    ("key_nn_within_2048", "key-NN ≤2048 tok"),
    ("key_nn_median_dpos", "key-NN median |Δpos|"),
    ("query_hits_own_book", "query hits own book"),
    ("query_hits_last_eighth", "query hits last 1/8"),
]
DIST_KEYS = [
    ("body_lid", "body LID"), ("body_contrast", "body contrast"), ("body_hubness", "body hubness"),
    ("tail_lid", "tail LID"), ("tail_contrast", "tail contrast"), ("tail_hubness", "tail hubness"),
    ("hubs_first_eighth", "hubs in first 1/8"), ("hubs_last_eighth", "hubs in last 1/8"),
]


def lag_figure(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(data)
    cp = HERE / "control" / "cross_book_lag.json"
    cross = json.loads(cp.read_text()) if cp.exists() else None
    fig, axes = plt.subplots(1, len(layers), figsize=(5.2 * len(layers), 3.8), squeeze=False)
    lags = [int(k) for k in LAGS]
    style = {"A": ("#d95f02", "-"), "B": ("#1b9e77", "-"), "C": ("#7570b3", "-")}
    for ax, L in zip(axes[0], layers):
        rows = data[L]
        ax.plot(lags, [rows["real"][f"cos_lag{k}"] for k in LAGS], "k-o", lw=2.5, label="real, same book", zorder=5)
        if cross is not None:
            (cb,) = [x for x in cross["layers"] if x["layer"] == L]
            ax.plot(lags, [cb["cross_book"][k] for k in LAGS], color="0.45", ls=":", marker="o", lw=2,
                    label="real, different books (control)", zorder=4)
        for n, r in rows.items():
            if n[0] in style and f"cos_lag{LAGS[0]}" in r:
                col, _ = style[n[0]]
                ls = "-" if n.endswith("s42") else "--"
                ax.plot(lags, [r[f"cos_lag{k}"] for k in LAGS], color=col, ls=ls, marker=".",
                        label=f"{ARM_NAMES[n[0]]} ({n.split()[1]})")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.0)
        ax.set_xlabel("lag (tokens)")
        ax.set_ylabel("median cosine of centred h")
        ax.set_title(f"Layer {L}")
        ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def report(data, path: Path):
    parts = []
    for L in sorted(data):
        rows = data[L]
        parts.append(f"<h2>Layer {L}</h2>")
        parts.append(gate_table(L, rows))
        parts.append(plain_table(L, rows, LOC_KEYS, "sequence locality (real: books 12–19; arms: their 8 synthetic books)"))
        parts.append(plain_table(L, rows, DIST_KEYS, "distribution and hubs"))
    img = lag_figure(data)
    doc = f"""<!doctype html><html><head><meta charset=utf-8><title>scout_kv v2 probe</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2em auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;margin:1em 0;font-size:13px}}td,th{{border:1px solid #ccc;padding:3px 7px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}caption{{text-align:left;font-weight:600;margin-bottom:4px}}
tr.real td{{background:#f2f2f2;font-weight:600}}td.in{{background:#d9f2d9}}td.out{{background:#fbe3e3}}
img{{max-width:100%}}</style></head><body>
<h1>scout_kv v2 probe: token-by-token vs drifting latent</h1>
{{SUMMARY}}
<h2>Consecutive similarity</h2><img alt="lag curves" src="data:image/png;base64,{img}">
{''.join(parts)}
</body></html>"""
    path.write_text(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="27,19")
    ap.add_argument("--summary-html", type=Path, default=None)
    a = ap.parse_args()
    data = collect(a.runs, [int(x) for x in a.layers.split(",")])
    (a.out / "summary.json").write_text(json.dumps(data, indent=2, default=str) + "\n")
    report(data, a.out / "report.html")
    if a.summary_html:
        p = a.out / "report.html"
        p.write_text(p.read_text().replace("{SUMMARY}", a.summary_html.read_text()))
    else:
        p = a.out / "report.html"
        p.write_text(p.read_text().replace("{SUMMARY}", ""))
    print("wrote", a.out / "summary.json", a.out / "report.html")


if __name__ == "__main__":
    main()
