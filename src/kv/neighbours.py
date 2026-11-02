"""Where in its book a real-style query's exact top-10 keys sit.

Three readings, each structure that an unconditional generator cannot produce
because it draws no positions: recency (same-book neighbours at a small lag
`q_pos - key_pos`), early tokens (neighbours at low positions, after the
dropped sink) and hubs (a few keys taking many top-10 slots). Each is reported
next to a baseline computed from all keys, so a box run can relate it to the
measured search hardness corpus by corpus.

Ground truth is exact inner-product top-k on the raw vectors, computed as
`kv_gate._own_book_share_top10` computes it, so `ood.own_book_share` equals the
gate record's `own_book_share_top10` for the same corpus.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.eval import kv_gate
from src.kv.corpus import Corpus, corpus_seed, mixed_real_corpus
from src.kv.transforms import HEAD_DIM, load_layer_weights

# 16 edges, 15 log2 bins, for both positions and lags. np.histogram closes the
# last bin, so every position 1..32768 and every lag 1..32768 falls inside.
EDGES = np.array([2**i for i in range(16)], dtype=np.int64)
RECENT_LAG = 1024
EARLY_POS = 16


def _share_hist(values: np.ndarray) -> list[float]:
    return (np.histogram(values, bins=EDGES)[0] / len(values)).tolist()


def _check_inside(name: str, values: np.ndarray) -> None:
    if len(values) and (values.min() < EDGES[0] or values.max() > EDGES[-1]):
        raise ValueError(
            f"{name} span {values.min()}..{values.max()}, outside the histogram "
            f"edges {EDGES[0]}..{EDGES[-1]}"
        )


def _top_k(q: np.ndarray, keys: np.ndarray, k: int) -> np.ndarray:
    sims = (
        np.ascontiguousarray(q, np.float32) @ np.ascontiguousarray(keys, np.float32).T
    )
    return np.argpartition(-sims, k - 1, axis=1)[:, :k]


def _query_set(c: Corpus, q, q_pos, q_book, k: int, n_hub: int) -> dict:
    key_pos, key_book = np.asarray(c.key_pos), np.asarray(c.key_book)
    q_pos, q_book = np.asarray(q_pos), np.asarray(q_book)
    gt = _top_k(q, c.keys, k)
    slot_pos = key_pos[gt]
    same = key_book[gt] == q_book[:, None]
    lags = (q_pos[:, None] - slot_pos)[same]

    # Baseline: each query's lags to every key of its own book, as a per-query
    # histogram normalised to sum 1, then averaged over queries.
    base_hist = np.zeros((len(q_pos), len(EDGES) - 1))
    base_recent = np.zeros(len(q_pos))
    for b in np.unique(q_book):
        rows = np.flatnonzero(q_book == b)
        own = key_pos[key_book == b]
        if len(own) == 0:
            raise ValueError(f"queries from book {b} but no keys from it")
        base_lags = q_pos[rows][:, None] - own[None, :]
        _check_inside("own-book lags", base_lags.ravel())
        # Bin index per lag with np.histogram's convention (last bin closed).
        n_bins = len(EDGES) - 1
        idx = np.minimum(
            np.searchsorted(EDGES, base_lags, side="right") - 1, n_bins - 1
        )
        idx += n_bins * np.arange(len(rows))[:, None]
        per_query = np.bincount(idx.ravel(), minlength=len(rows) * n_bins)
        base_hist[rows] = per_query.reshape(len(rows), n_bins) / len(own)
        base_recent[rows] = (base_lags <= RECENT_LAG).mean(axis=1)

    counts = np.bincount(gt.ravel(), minlength=len(key_pos))
    n_distinct = int((counts > 0).sum())
    # Hubs are the n_hub most-used keys, but never keys no query used: when the
    # top-k lists use fewer than n_hub keys, every used key is a hub.
    hubs = np.argsort(-counts, kind="stable")[: min(n_hub, n_distinct)]
    n_same = int(same.sum())
    return {
        "own_book_share": float(same.mean()),
        "nbr_pos_share": _share_hist(slot_pos.ravel()),
        "key_pos_share": _share_hist(key_pos),
        "nbr_lag_share": _share_hist(lags) if n_same != 0 else None,
        "base_lag_share": base_hist.mean(axis=0).tolist(),
        "share_lag_le_1024": (
            float((lags <= RECENT_LAG).mean()) if n_same != 0 else None
        ),
        "base_share_lag_le_1024": float(base_recent.mean()),
        "share_pos_lt_16": float((slot_pos < EARLY_POS).mean()),
        "base_share_pos_lt_16": float((key_pos < EARLY_POS).mean()),
        "n_distinct_keys": n_distinct,
        "hub_slot_share": float(np.sort(counts)[::-1][:n_hub].sum() / gt.size),
        "hub_median_pos": float(np.median(key_pos[hubs])),
        "key_median_pos": float(np.median(key_pos)),
    }


def neighbour_positions(c: Corpus, k: int = 10) -> dict:
    """Neighbour-position statistics of one real corpus, for its OOD (`ood`)
    and key-side (`id`) query sets. Every value is JSON-serialisable."""
    for name in ("key_pos", "q_pos", "q_id_pos"):
        pos = np.asarray(getattr(c, name))
        if len(pos) and pos.min() < 0:
            raise ValueError(
                f"corpus has no positions ({name} is negative); "
                "neighbour positions need a real corpus"
            )
    _check_inside("key positions", np.asarray(c.key_pos))
    n_hub = max(1, len(c.keys) // 100)
    return {
        "ood": _query_set(c, c.q_ood, c.q_pos, c.q_book, k, n_hub),
        "id": _query_set(c, c.q_id, c.q_id_pos, c.q_id_book, k, n_hub),
        "edges": EDGES.tolist(),
        "n_hub": n_hub,
    }


def measure_neighbours(
    capture_dir, books, layers, seed, out, *, n_sink=1, n_tail=1024, source="real"
) -> None:
    """One record per (layer, KV head) on the same real corpus `measure_real`
    builds for the same books, seed, n_sink and n_tail. `source` labels the
    records, e.g. `v2_s42` for a v2 run's capture-format books."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to append duplicate records")
    dev = kv_gate._device()
    for layer in layers:
        h_books = [
            h.to(dev).float() for h in kv_gate.load_books(capture_dir, layer, books)
        ]
        w = load_layer_weights(capture_dir, layer)
        wk, wq = w["k_proj"].to(dev).float(), w["q_proj"].to(dev).float()
        for g in range(wk.shape[0] // HEAD_DIM):
            t0 = time.time()
            c = mixed_real_corpus(
                h_books,
                wk,
                wq,
                g,
                n_sink=n_sink,
                n_tail=n_tail,
                seed=corpus_seed(layer, g, seed),
            )
            rec = {
                "layer": layer,
                "kv_head": g,
                "source": source,
                "seed": seed,
                "books": list(books),
                "n_keys": len(c.keys),
                "n_sink": n_sink,
                "n_tail": n_tail,
                "k": 10,
                **neighbour_positions(c, k=10),
                "seconds": round(time.time() - t0, 1),
            }
            kv_gate._append(out, rec)
            o = rec["ood"]
            print(
                layer,
                g,
                f"own_book {o['own_book_share']:.3f}",
                f"lag<=1024 {o['share_lag_le_1024']}",
                f"hub {o['hub_slot_share']:.3f}",
                rec["seconds"],
                "s",
                flush=True,
            )
