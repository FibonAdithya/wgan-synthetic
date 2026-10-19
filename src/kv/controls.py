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

CONTROLS = ("pooled_real", "norm_equalised", "gaussian")


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
    n_keys=31736,
    n_tail=1024,
    nlist=256,
    hnsw_m=48,
    ef_construction=500,
) -> None:
    import faiss

    controls = list(controls)
    bad = [c for c in controls if c not in CONTROLS]
    if bad:
        raise ValueError(f"unknown controls {bad}; choose from {list(CONTROLS)}")
    out_dir = Path(out_dir)
    paths = {c: out_dir / f"{c}.jsonl" for c in controls}
    for p in paths.values():
        if p.exists():
            raise FileExistsError(f"{p} exists; refusing to append duplicate records")
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(threads)
    dev = kv_gate._device()
    train_books, ref_books = list(train_books), list(ref_books)
    idx = dict(nlist=nlist, hnsw_m=hnsw_m, ef_construction=ef_construction)

    def run(h, layer, name, books):
        t0 = time.time()
        recs = kv_gate.measure_h_layer(
            *h,
            capture_dir,
            layer,
            source=name,
            seed=seed,
            extra={"control": name, "books": books},
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
