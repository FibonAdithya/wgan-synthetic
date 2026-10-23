"""Where in the book do real h's hubs sit?

v1 (a body/tail flag) does not reproduce the hubs of real body h: skew 26.7-52.3
at layers 7-47 for real against 2.6-6.4 for v1
(docs/results/scout-kv/v1/full/intrinsic.jsonl). A finer position condition
can only help if hubness depends on position inside the body. This script
measures that on real data, and checks two other candidate drivers at the
same time: vector norm and alignment with the mean (GloVe's hubs were
short-norm and mean-aligned).

For each layer: 20,000 body rows of books 12-19 (positions 1-31,743), drawn
with a fixed seed, as the intrinsic diagnostics subsample. N_10 = how often a
row is among the other rows' 10 nearest neighbours (L2), the quantity whose
skew is `hubness_skew`. Hubs = the top 1% by N_10.

Reported per layer:
- hubness skew of the subsample (comparable to intrinsic.jsonl);
- share of rows and of hubs in each of 8 equal position bins over 1-31,743;
- Spearman correlation of N_10 with position, with ||h||, and with the cosine
  of h to the subsample mean;
- the hubs' median position and median norm against all rows'.

Run on the box from the repo root:
    python docs/results/scout-kv/v1-diagnostics/hub_position.py \
        --capture-dir /workspace/kv/capture --out /workspace/kv/v1/hub_position.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.data.kv_capture import GLOBAL_LAYERS, load_kv_capture
from src.eval.ann_difficulty import hubness_skew, k_occurrence, knn
from src.models.conditional import BODY

BOOKS = list(range(12, 20))
N_SINK = 1
N_TAIL = 1024
N_ROWS = 20000
K_HUB = 10
HUB_SHARE = 0.01
N_BINS = 8
SEED = 42


def layer_report(capture_dir: Path, layer: int, n_rows: int = N_ROWS) -> dict:
    x, flags = load_kv_capture(capture_dir, layer, BOOKS, n_sink=N_SINK, n_tail=N_TAIL)
    per_book = x.shape[0] // len(BOOKS)
    pos = np.tile(np.arange(N_SINK, N_SINK + per_book), len(BOOKS))
    book = np.repeat(BOOKS, per_book)
    body = np.flatnonzero(flags == BODY)
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(body, size=n_rows, replace=False))
    h = np.ascontiguousarray(x[pick], dtype=np.float32)
    p, b = pos[pick], book[pick]
    del x, flags

    _, idx, _ = knn(h, K_HUB)
    n10 = k_occurrence(idx, n_rows, K_HUB)
    norm = np.linalg.norm(h, axis=1)
    mu = h.mean(axis=0)
    cos_mean = (h @ mu) / (norm * np.linalg.norm(mu))

    n_hubs = int(round(HUB_SHARE * n_rows))
    hubs = np.argsort(-n10, kind="stable")[:n_hubs]
    body_max = int(p.max())
    edges = np.linspace(N_SINK, body_max + 1, N_BINS + 1)
    all_hist = np.histogram(p, bins=edges)[0] / n_rows
    hub_hist = np.histogram(p[hubs], bins=edges)[0] / n_hubs

    def rho(v):
        r = spearmanr(n10, v)
        return {"rho": float(r.statistic), "p": float(r.pvalue)}

    return {
        "layer": layer,
        "n_rows": n_rows,
        "n_hubs": n_hubs,
        "hubness_skew": float(hubness_skew(n10)),
        "max_n10": int(n10.max()),
        "bin_edges": [float(e) for e in edges],
        "share_rows_by_bin": [float(v) for v in all_hist],
        "share_hubs_by_bin": [float(v) for v in hub_hist],
        "spearman_n10_position": rho(p),
        "spearman_n10_norm": rho(norm),
        "spearman_n10_cos_mean": rho(cos_mean),
        "median_position": {
            "all": float(np.median(p)),
            "hubs": float(np.median(p[hubs])),
        },
        "median_norm": {
            "all": float(np.median(norm)),
            "hubs": float(np.median(norm[hubs])),
        },
        "median_cos_mean": {
            "all": float(np.median(cos_mean)),
            "hubs": float(np.median(cos_mean[hubs])),
        },
        "hub_books": {
            int(k): int(v) for k, v in zip(*np.unique(b[hubs], return_counts=True))
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--layers", default=",".join(str(x) for x in GLOBAL_LAYERS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-rows", type=int, default=N_ROWS)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    out = {
        "books": BOOKS,
        "seed": SEED,
        "k_hub": K_HUB,
        "hub_share": HUB_SHARE,
        "layers": [],
    }
    for layer in (int(v) for v in a.layers.split(",")):
        t0 = time.time()
        rec = layer_report(a.capture_dir, layer, a.n_rows)
        rec["seconds"] = round(time.time() - t0, 1)
        out["layers"].append(rec)
        print(
            layer,
            "skew",
            round(rec["hubness_skew"], 1),
            "rho(pos)",
            round(rec["spearman_n10_position"]["rho"], 3),
            "rho(norm)",
            round(rec["spearman_n10_norm"]["rho"], 3),
            "rho(cos)",
            round(rec["spearman_n10_cos_mean"]["rho"], 3),
            "hub bins",
            [round(v, 3) for v in rec["share_hubs_by_bin"]],
            flush=True,
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("DONE", a.out)


if __name__ == "__main__":
    main()
