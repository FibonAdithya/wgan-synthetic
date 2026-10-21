"""Control sets of h for diagnosing why v0 is too easy to search.

Each control yields (h_keys, h_query, h_id) and is profiled through
`kv_gate.measure_h_layer`, so it is measured under the locked conditions.
"""

import time
from pathlib import Path

import numpy as np
import torch

from src.data.kv_capture import h_path, read_safetensor
from src.eval import kv_gate
from src.kv.corpus import Corpus, _np, mixed_real_corpus
from src.kv.transforms import head_keys, head_queries, n_rep

CONTROLS = ("pooled_real", "norm_equalised", "gaussian")
# Not in CONTROLS, so the script default and b7.sh keep their original runs.
TAIL_CONTROLS = ("same_book_tail", "cross_book_tail")


def pooled_split(books, n_keys, n_tail, seed, n_sink=1):
    """Pool every book after dropping its first `n_sink` rows, then split one
    seeded permutation into keys, queries and key-side queries (disjoint).

    Returns (h_keys, h_query, h_id, idx_keys, idx_query, idx_id); the indices
    address the concatenation of the books with the sink rows removed.
    """
    pool = torch.cat([torch.as_tensor(b)[n_sink:] for b in books])
    need = n_keys + 2 * n_tail
    if len(pool) < need:
        raise ValueError(f"pool has {len(pool)} rows, need {need}")
    perm = np.random.default_rng(seed).permutation(len(pool))
    ik = perm[:n_keys]
    iq = perm[n_keys : n_keys + n_tail]
    ii = perm[n_keys + n_tail : need]
    take = lambda i: pool[torch.from_numpy(i)]  # noqa: E731
    return take(ik), take(iq), take(ii), ik, iq, ii


def tail_split(key_books, query_books, n_keys, n_tail, seed, n_sink=1):
    """Keys from the bodies of `key_books`, queries from the tails of `query_books`.

    Every book has the same length T. The body is positions n_sink .. T-n_tail-1
    and the tail is positions T-n_tail .. T-1, as in mixed_real_corpus. Each key
    book gives n_keys // len(key_books) rows and each query book gives
    n_tail // len(query_books) rows, sampled without replacement. Key positions
    come from default_rng([seed, 0]) and tail positions from default_rng([seed, 1]),
    so the key rows do not depend on query_books. h_id is h_query (the same rows),
    as in the reference.

    Returns (h_keys, h_query, h_id, key_pos, q_pos); key_pos and q_pos are (n, 2)
    int arrays of (index into the books argument list, position).
    """
    books = [torch.as_tensor(b) for b in (*key_books, *query_books)]
    T = books[0].shape[0]
    if any(b.shape[0] != T for b in books):
        raise ValueError(f"books differ in length: {[b.shape[0] for b in books]}")
    if n_keys % len(key_books) or n_tail % len(query_books):
        raise ValueError(
            f"n_keys {n_keys} / n_tail {n_tail} not divisible by "
            f"{len(key_books)} key / {len(query_books)} query books"
        )
    Tk = T - n_tail
    per_k, per_q = n_keys // len(key_books), n_tail // len(query_books)
    if per_k > Tk - n_sink:
        raise ValueError(f"each body has {Tk - n_sink} rows, need {per_k}")

    def take(src, lo, hi, per, rng):
        rows, pos = [], []
        for i, b in enumerate(src):
            p = np.sort(rng.choice(np.arange(lo, hi), per, replace=False))
            rows.append(torch.as_tensor(b)[torch.from_numpy(p)])
            pos.append(np.stack([np.full(per, i), p], 1))
        return torch.cat(rows), np.concatenate(pos)

    hk, kp = take(key_books, n_sink, Tk, per_k, np.random.default_rng([seed, 0]))
    hq, qp = take(query_books, Tk, T, per_q, np.random.default_rng([seed, 1]))
    return hk, hq, hq, kp, qp


def held_out_tail_corpus(
    h_ref_books, h_query_books, wk, wq, g, *, n_sink, n_tail, seed
) -> Corpus:
    """The reference corpus's keys, with queries from the tails of other books.

    Keys, key_book and key_pos are exactly mixed_real_corpus(h_ref_books, ...) at
    `seed`. Tail positions come from default_rng([seed, 1]): n_tail // Q per query
    book, in order. q_id is the k-projection of the same tail rows as q_ood, as in
    the reference. Query book j is labelled len(h_ref_books) + j, so no query
    shares a label with any key.
    """
    Q = len(h_query_books)
    if Q == 0:
        raise ValueError("h_query_books is empty")
    T = h_ref_books[0].shape[0]
    if any(h.shape[0] != T for h in h_query_books):
        raise ValueError(
            f"query books differ in length from the reference books ({T}): "
            f"{[h.shape[0] for h in h_query_books]}"
        )
    if n_tail % Q:
        raise ValueError(f"n_tail {n_tail} is not divisible by {Q} query books")
    ref = mixed_real_corpus(
        h_ref_books, wk, wq, g, n_sink=n_sink, n_tail=n_tail, seed=seed
    )
    B, per, r = len(h_ref_books), n_tail // Q, n_rep(wq, wk)
    rng = np.random.default_rng([seed, 1])
    qs, ids, qb, qp, idp, idb = [], [], [], [], [], []
    for j, h in enumerate(h_query_books):
        tpos = np.sort(rng.choice(np.arange(T - n_tail, T), per, replace=False))
        ht = h[torch.as_tensor(tpos).to(h.device)]
        qs.append(_np(head_queries(ht, wq, g, r)))
        ids.append(_np(head_keys(ht, wk, g)))
        qb.append(np.full(per * r, B + j))
        qp.append(np.repeat(tpos, r))
        idp.append(tpos)
        idb.append(np.full(per, B + j))
    return Corpus(
        ref.keys,
        np.vstack(qs),
        np.vstack(ids),
        ref.key_book,
        np.concatenate(qb),
        ref.key_pos,
        np.concatenate(qp),
        np.concatenate(idp),
        np.concatenate(idb),
    )


def equalise_norms(h_keys, h_query, h_id):
    """Rescale every row to the mean row norm of the three tensors together."""
    ts = [torch.as_tensor(t).float() for t in (h_keys, h_query, h_id)]
    norms = [t.norm(dim=1, keepdim=True) for t in ts]
    mean = torch.cat(norms).double().mean().float()
    return tuple(t * (mean / n) for t, n in zip(ts, norms, strict=True))


def fit_gaussian(arrays, device=None):
    """Mean and full covariance (float64, divisor n) from an iterable of
    row chunks, accumulating sums so the chunks need not be resident together."""
    n, s, ss = 0, None, None
    for a in arrays:
        x = torch.as_tensor(a).to(device=device, dtype=torch.float64)
        if s is None:
            s = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
            ss = torch.zeros(
                x.shape[1], x.shape[1], dtype=torch.float64, device=x.device
            )
        n += len(x)
        s += x.sum(0)
        ss += x.T @ x
    if n == 0:
        raise ValueError("fit_gaussian got no rows")
    mu = s / n
    cov = ss / n - torch.outer(mu, mu)
    return {"mean": mu.cpu().numpy(), "cov": cov.cpu().numpy(), "n": n}


def sample_gaussian(fit, n, seed, device=None):
    """n draws from N(mean, cov) via eigh (singular covariances are fine)."""
    lam, v = np.linalg.eigh(fit["cov"])
    lam = np.clip(lam, 0.0, None)
    dev = torch.device(device) if device else kv_gate._device()
    a = torch.from_numpy(v * np.sqrt(lam)).to(dev, torch.float32)
    mu = torch.from_numpy(fit["mean"]).to(dev, torch.float32)
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(n, len(mu), generator=g, device=dev)
    return mu + z @ a.T


def _book_rows(capture_dir, book, layer):
    return torch.from_numpy(
        read_safetensor(h_path(Path(capture_dir), book, layer), "h")
    )


def measure_controls(
    capture_dir,
    layers,
    controls,
    seed,
    out_dir,
    *,
    threads=15,
    train_books=range(12),
    ref_books=range(12, 20),
    key_books=(12, 13, 14, 15),
    n_keys=31736,
    n_tail=1024,
    nlist=256,
    hnsw_m=48,
    ef_construction=500,
) -> None:
    import faiss

    controls = list(controls)
    known = CONTROLS + TAIL_CONTROLS
    bad = [c for c in controls if c not in known]
    if bad:
        raise ValueError(f"unknown controls {bad}; choose from {list(known)}")
    train_books, ref_books = list(train_books), list(ref_books)
    key_books = list(key_books)
    if not key_books or not set(key_books) < set(ref_books):
        raise ValueError(
            f"key_books {key_books} must be a non-empty proper subset of {ref_books}"
        )
    query_books = {
        "same_book_tail": key_books,
        "cross_book_tail": [b for b in ref_books if b not in key_books],
    }
    out_dir = Path(out_dir)
    paths = {c: out_dir / f"{c}.jsonl" for c in controls}
    for p in paths.values():
        if p.exists():
            raise FileExistsError(f"{p} exists; refusing to append duplicate records")
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(threads)
    dev = kv_gate._device()
    idx = dict(nlist=nlist, hnsw_m=hnsw_m, ef_construction=ef_construction)

    def run(h, layer, name, books, extra=None):
        t0 = time.time()
        recs = kv_gate.measure_h_layer(
            *h,
            capture_dir,
            layer,
            source=name,
            seed=seed,
            extra={"control": name, "books": books, **(extra or {})},
            **idx,
        )
        for r in recs:
            kv_gate._append(paths[name], r)
        print(f"L{layer} {name}: {time.time() - t0:.1f} s", flush=True)

    for layer in layers:
        if "pooled_real" in controls or "norm_equalised" in controls:
            ref = [_book_rows(capture_dir, b, layer) for b in ref_books]
            hk, hq, hi, *_ = pooled_split(ref, n_keys, n_tail, seed)
            del ref
            pooled = tuple(t.to(dev).float() for t in (hk, hq, hi))
            if "pooled_real" in controls:
                run(pooled, layer, "pooled_real", ref_books)
            if "norm_equalised" in controls:
                run(equalise_norms(*pooled), layer, "norm_equalised", ref_books)
        if "gaussian" in controls:
            fit = fit_gaussian(
                (_book_rows(capture_dir, b, layer)[1:] for b in train_books),
                device=dev,
            )
            s = sample_gaussian(fit, n_keys + 2 * n_tail, seed, device=dev)
            h = (s[:n_keys], s[n_keys : n_keys + n_tail], s[n_keys + n_tail :])
            run(h, layer, "gaussian", train_books)
        tails = [c for c in TAIL_CONTROLS if c in controls]
        if tails:
            need = sorted({b for c in tails for b in query_books[c]} | set(key_books))
            rows = {b: _book_rows(capture_dir, b, layer) for b in need}
            for name in tails:
                qb = query_books[name]
                hk, hq, hi, *_ = tail_split(
                    [rows[b] for b in key_books],
                    [rows[b] for b in qb],
                    n_keys,
                    n_tail,
                    seed,
                    n_sink=1,
                )
                h = tuple(t.to(dev).float() for t in (hk, hq, hi))
                run(h, layer, name, key_books, {"query_books": list(qb)})
            del rows


HELD_OUT_CONTROLS = ("rebuild", "held_out_tail")


def measure_held_out(
    capture_dir,
    layers,
    seed,
    out_dir,
    *,
    query_books,
    controls=HELD_OUT_CONTROLS,
    ref_books=range(12, 20),
    n_sink=1,
    n_tail=1024,
    nlist=256,
    hnsw_m=48,
    ef_construction=500,
    threads=15,
) -> None:
    """Profile `rebuild` (the reference corpus, as measure_real builds it) and
    `held_out_tail` (the same keys, queries from the tails of `query_books`) at
    `seed`. Both use corpus_seed(layer, g, seed) for the corpus and the index."""
    import faiss

    controls, ref_books, query_books = (
        list(controls),
        list(ref_books),
        list(query_books),
    )
    bad = [c for c in controls if c not in HELD_OUT_CONTROLS]
    if bad:
        raise ValueError(
            f"unknown controls {bad}; choose from {list(HELD_OUT_CONTROLS)}"
        )
    if not query_books:
        raise ValueError("query_books is empty")
    shared = sorted(set(query_books) & set(ref_books))
    if shared:
        raise ValueError(f"query_books {shared} are also reference books")
    out_dir = Path(out_dir)
    paths = {c: out_dir / f"{c}.jsonl" for c in controls}
    for p in paths.values():
        if p.exists():
            raise FileExistsError(f"{p} exists; refusing to append duplicate records")
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(threads)
    dev = kv_gate._device()
    for layer in layers:
        load = lambda bs: [  # noqa: E731
            h.to(dev).float() for h in kv_gate.load_books(capture_dir, layer, bs)
        ]
        h_ref = load(ref_books)
        h_q = load(query_books) if "held_out_tail" in controls else None
        w = kv_gate.load_layer_weights(capture_dir, layer)
        wk, wq = w["k_proj"].to(dev).float(), w["q_proj"].to(dev).float()
        for g in range(wk.shape[0] // kv_gate.HEAD_DIM):
            s = kv_gate.corpus_seed(layer, g, seed)
            for name in controls:
                t0 = time.time()
                kw = dict(n_sink=n_sink, n_tail=n_tail, seed=s)
                if name == "rebuild":
                    c = mixed_real_corpus(h_ref, wk, wq, g, **kw)
                    qbooks = ref_books
                else:
                    c = held_out_tail_corpus(h_ref, h_q, wk, wq, g, **kw)
                    qbooks = query_books
                prof = kv_gate.profile_corpus(
                    c.keys,
                    c.q_ood,
                    c.q_id,
                    nlist=nlist,
                    hnsw_m=hnsw_m,
                    ef_construction=ef_construction,
                    seed=s,
                    target=kv_gate.TARGET,
                )
                rec = {
                    "layer": layer,
                    "kv_head": g,
                    "source": "control",
                    "seed": seed,
                    "books": ref_books,
                    "n_keys": len(c.keys),
                    "n_q_ood": len(c.q_ood),
                    "n_q_id": len(c.q_id),
                    "n_sink": n_sink,
                    "n_tail": n_tail,
                    "conditions": kv_gate.conditions(
                        nlist, hnsw_m, ef_construction, kv_gate.TARGET
                    ),
                    "own_book_share_top10": kv_gate._own_book_share_top10(c),
                    **prof,
                    "seconds": round(time.time() - t0, 1),
                    "control": name,
                    "query_books": qbooks,
                }
                kv_gate._append(paths[name], rec)
                print(f"L{layer} g{g} {name}: {rec['seconds']} s", flush=True)
        del h_ref, h_q
