"""Is real h sequence-local? The case for generating KV caches token by token.

v1 samples every h independently. A generator that writes a book token by token
(each h conditioned on the previous ones) only adds something if real KV caches
have structure tied to the sequence: neighbouring tokens alike, or nearest
neighbours that are nearby tokens of the same book. This script measures that
on real h from the reference books 12-19, per global layer.

1. Consecutive similarity: cosine of centred h at positions t and t+lag inside
   one book's body (positions 1-31,743), for lags 1 to 4,096, against random
   same-book pairs.
2. Neighbour locality in h: 20,000 body rows (seed 42) against all 253,944
   body rows of the 8 books, 10 nearest by L2 (self excluded). Share of
   neighbours from the same book, and from within 16 / 256 / 2,048 positions
   of the query in the same book, with the chance rate for each.
3. Neighbour locality in key space: the same rows projected to each KV head's
   keys (real W_k, 128 dims), 10 largest inner products among all body keys.
   Averaged over the 8 KV heads; per-head values kept in the JSON.
4. Query hits: real tail queries (real W_q, last 1,024 positions of each book,
   5 query heads per KV head) against all body keys of the 8 books, top 10 by
   inner product, as the gate searches. Share of hits from the query's own
   book, within 1,024 / 4,096 positions of the query, and in the last eighth
   of the body.

Run on the box from the repo root (uses the GPU when present):
    python docs/results/scout-kv/locality/locality.py --capture-dir /workspace/kv/capture \
        --out /workspace/kv/locality/locality.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.kv_capture import GLOBAL_LAYERS, load_kv_capture
from src.kv.transforms import HEAD_DIM, load_layer_weights, n_rep
from src.models.conditional import BODY, TAIL

BOOKS = list(range(12, 20))
N_SINK = 1
N_TAIL = 1024
N_QUERY_ROWS = 20000
K = 10
LAGS = (1, 2, 4, 16, 64, 256, 1024, 4096)
WINDOWS = (16, 256, 2048)
QUERY_WINDOWS = (1024, 4096)
N_RANDOM_PAIRS = 200000
SEED = 42
CHUNK = 1000


def topk_neighbours(q, db, q_rows, *, largest, metric):
    """Indices into db of the K best rows for each q row, excluding q_rows (self)."""
    out = torch.empty((q.shape[0], K), dtype=torch.long, device=q.device)
    db_sq = (db * db).sum(1) if metric == "l2" else None
    for s in range(0, q.shape[0], CHUNK):
        qc = q[s : s + CHUNK]
        sim = qc @ db.T
        if metric == "l2":
            sim = (qc * qc).sum(1, keepdim=True) + db_sq[None, :] - 2 * sim
        if q_rows is not None:
            r = torch.arange(qc.shape[0], device=q.device)
            sim[r, q_rows[s : s + CHUNK]] = -float("inf") if largest else float("inf")
        out[s : s + CHUNK] = sim.topk(K, dim=1, largest=largest).indices
    return out


def locality_stats(nb, q_book, q_pos, db_book, db_pos, windows):
    nb_book, nb_pos = db_book[nb], db_pos[nb]
    same = nb_book == q_book[:, None]
    d = (nb_pos - q_pos[:, None]).abs()
    rec = {"same_book": float(same.float().mean())}
    for w in windows:
        rec[f"same_book_within_{w}"] = float((same & (d <= w)).float().mean())
    same_d = d[same]
    rec["median_abs_dpos_same_book"] = (
        float(same_d.float().median()) if same_d.numel() else None
    )
    return rec


def layer_report(capture_dir: Path, layer: int, dev: torch.device) -> dict:
    x, flags = load_kv_capture(capture_dir, layer, BOOKS, n_sink=N_SINK, n_tail=N_TAIL)
    per_book = x.shape[0] // len(BOOKS)
    pos_all = np.tile(np.arange(N_SINK, N_SINK + per_book), len(BOOKS))
    book_all = np.repeat(BOOKS, per_book)
    body = flags == BODY
    tail = flags == TAIL
    hb = torch.from_numpy(np.ascontiguousarray(x[body], dtype=np.float32)).to(dev)
    ht = torch.from_numpy(np.ascontiguousarray(x[tail], dtype=np.float32)).to(dev)
    del x
    b_pos = torch.from_numpy(pos_all[body]).to(dev)
    b_book = torch.from_numpy(book_all[body]).to(dev)
    t_pos = torch.from_numpy(pos_all[tail]).to(dev)
    t_book = torch.from_numpy(book_all[tail]).to(dev)
    n_body = hb.shape[0]
    body_per_book = n_body // len(BOOKS)
    rec = {"layer": layer, "n_body": int(n_body), "n_tail": int(ht.shape[0])}

    # 1. consecutive similarity on centred h
    hc = hb - hb.mean(0, keepdim=True)
    hn = torch.nn.functional.normalize(hc, dim=1)
    cons = {}
    for lag in LAGS:
        vals = []
        for i in range(len(BOOKS)):
            blk = hn[i * body_per_book : (i + 1) * body_per_book]
            vals.append((blk[:-lag] * blk[lag:]).sum(1))
        v = torch.cat(vals)
        cons[str(lag)] = {"median": float(v.median()), "mean": float(v.mean())}
    g = torch.Generator(device="cpu").manual_seed(SEED)
    bi = torch.randint(0, len(BOOKS), (N_RANDOM_PAIRS,), generator=g)
    a = (
        torch.randint(0, body_per_book, (N_RANDOM_PAIRS,), generator=g)
        + bi * body_per_book
    )
    b = (
        torch.randint(0, body_per_book, (N_RANDOM_PAIRS,), generator=g)
        + bi * body_per_book
    )
    rnd = (hn[a.to(dev)] * hn[b.to(dev)]).sum(1)
    rec["consecutive_cosine_centred"] = cons
    rec["random_same_book_cosine_centred"] = {
        "median": float(rnd.median()),
        "mean": float(rnd.mean()),
    }

    # sampled query rows for 2 and 3
    rng = np.random.default_rng(SEED)
    q_rows = torch.from_numpy(
        np.sort(rng.choice(n_body, N_QUERY_ROWS, replace=False))
    ).to(dev)
    q_book, q_pos = b_book[q_rows], b_pos[q_rows]

    # 2. neighbour locality in h (L2 on centred h)
    nb = topk_neighbours(hc[q_rows], hc, q_rows, largest=False, metric="l2")
    rec["h_l2"] = locality_stats(nb, q_book, q_pos, b_book, b_pos, WINDOWS)
    del hn

    # 3 and 4. key space and query hits, per KV head
    w = load_layer_weights(capture_dir, layer)
    wk, wq = w["k_proj"].to(dev).float(), w["q_proj"].to(dev).float()
    rep = n_rep(wq, wk)
    n_kv = wk.shape[0] // HEAD_DIM
    per_head_k, per_head_q = [], []
    for gi in range(n_kv):
        keys = hb @ wk[gi * HEAD_DIM : (gi + 1) * HEAD_DIM].T
        nb = topk_neighbours(keys[q_rows], keys, q_rows, largest=True, metric="ip")
        per_head_k.append(locality_stats(nb, q_book, q_pos, b_book, b_pos, WINDOWS))
        hits = []
        for j in range(gi * rep, (gi + 1) * rep):
            qv = ht @ wq[j * HEAD_DIM : (j + 1) * HEAD_DIM].T
            hits.append(topk_neighbours(qv, keys, None, largest=True, metric="ip"))
        hits = torch.cat(hits)
        qb, qp = t_book.repeat(rep), t_pos.repeat(rep)
        st = locality_stats(hits, qb, qp, b_book, b_pos, QUERY_WINDOWS)
        last_eighth = b_pos[hits] > (7 * body_per_book) // 8
        st["hits_in_last_eighth_of_body"] = float(last_eighth.float().mean())
        per_head_q.append(st)
        del keys
    rec["key_ip_per_head"] = per_head_k
    rec["query_hits_per_head"] = per_head_q

    def mean_over(heads):
        return {
            k: float(np.mean([h[k] for h in heads if h[k] is not None]))
            for k in heads[0]
        }

    rec["key_ip"] = mean_over(per_head_k)
    rec["query_hits"] = mean_over(per_head_q)
    rec["chance"] = {
        "same_book": (body_per_book - 1) / (n_body - 1),
        **{f"same_book_within_{w}": 2 * w / (n_body - 1) for w in WINDOWS},
        **{
            f"query_same_book_within_{w}": min(w, body_per_book) / n_body
            for w in QUERY_WINDOWS
        },
        "hits_in_last_eighth_of_body": 1 / 8,
    }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--layers", default=",".join(str(x) for x in GLOBAL_LAYERS))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"books": BOOKS, "seed": SEED, "k": K, "device": str(dev), "layers": []}
    for layer in (int(v) for v in a.layers.split(",")):
        t0 = time.time()
        rec = layer_report(a.capture_dir, layer, dev)
        rec["seconds"] = round(time.time() - t0, 1)
        out["layers"].append(rec)
        c = rec["consecutive_cosine_centred"]
        print(
            layer,
            "cos lag1/16/256/4096",
            [round(c[k]["median"], 3) for k in ("1", "16", "256", "4096")],
            "rand",
            round(rec["random_same_book_cosine_centred"]["median"], 3),
            "| h-NN same book",
            round(rec["h_l2"]["same_book"], 3),
            "within16",
            round(rec["h_l2"]["same_book_within_16"], 3),
            "| key-NN same book",
            round(rec["key_ip"]["same_book"], 3),
            "within16",
            round(rec["key_ip"]["same_book_within_16"], 3),
            "| query hits own book",
            round(rec["query_hits"]["same_book"], 3),
            "within1024",
            round(rec["query_hits"]["same_book_within_1024"], 3),
            f"({rec['seconds']} s)",
            flush=True,
        )
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("DONE", a.out)


if __name__ == "__main__":
    main()
