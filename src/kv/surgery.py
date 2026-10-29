"""Real h with one property removed, to find what makes real search hard (scout_kv v3 test).

Each set maps the reference books (full length, sink row included) to books of the
same shape, which are then measured through the gate's real split
(`kv_gate.real_layer_records`), so every set is compared with the real reference
range under the same queries-from-the-tail condition. Row 0 (the sink) is never
changed. Components are the principal axes of h, fitted on all non-sink rows of
the books being transformed.

- `real`: unchanged.
- `gaussian`: every row drawn from N(mean, cov).
- `s0`, `s32`, `s256`: in the principal basis, each component past the first
  0 / 32 / 256 is permuted on its own across the rows of a block (one book,
  1,024 body positions, or the whole tail). Covariance, each component's
  values and block means stay; the joint structure of those components within
  a token goes.
- `gz`: each component mapped, by rank over all rows, to Gaussian quantiles with
  its own mean and standard deviation. Dependence between components stays;
  heavy tails go.

Outputs are rounded to bf16, the precision the capture stores.

    python -m src.kv.surgery --capture-dir /workspace/kv/capture --layers 11,15,43 \
        --seeds 0,1 --out-dir /workspace/kv/v3surgery
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.kv_capture import kv_flags
from src.eval import kv_gate
from src.eval.ann_difficulty import compute, summary
from src.kv.neighbours import neighbour_positions
from src.kv.transforms import load_layer_weights
from src.models.conditional import BODY, TAIL

SETS = ("real", "gaussian", "s0", "s32", "s256", "gz")
KEEP = {"s0": 0, "s32": 32, "s256": 256}
BLOCK = 1024
# The setting docs/results/scout-kv/v2/intrinsic.py measured the v2 runs with.
INTRINSIC = dict(k=100, k_hub=10, nlist=256, max_rows=20000, seed=42, metric="l2")
NEIGHBOUR_FIELDS = (
    "n_distinct_keys",
    "hub_slot_share",
    "hub_median_pos",
    "own_book_share",
    "share_lag_le_1024",
)


@dataclass
class PCA:
    mean: torch.Tensor  # (D,)
    basis: torch.Tensor  # (D, D), column j is component j
    var: torch.Tensor  # (D,), descending


def block_ids(T: int, n_sink: int, n_tail: int, block: int = BLOCK) -> torch.Tensor:
    """Block of each position: -1 for the sink, body blocks of `block` rows counted
    from the first non-sink row, and the whole tail as the last block."""
    ids = torch.full((T,), -1, dtype=torch.long)
    body = torch.arange(n_sink, T - n_tail)
    ids[body] = (body - n_sink) // block
    ids[T - n_tail :] = int(ids[body].max()) + 1 if len(body) else 0
    return ids


def fit_pca(books, n_sink: int) -> PCA:
    """Principal axes of the books' non-sink rows (float64 sums, divisor n)."""
    dev = books[0].device
    D = books[0].shape[1]
    n, s = 0, torch.zeros(D, dtype=torch.float64, device=dev)
    ss = torch.zeros(D, D, dtype=torch.float64, device=dev)
    for b in books:
        x = b[n_sink:].to(torch.float64)
        n += len(x)
        s += x.sum(0)
        ss += x.T @ x
    mean = s / n
    cov = ss / n - torch.outer(mean, mean)
    var, vec = torch.linalg.eigh(cov)
    order = torch.argsort(var, descending=True)
    return PCA(
        mean.float(),
        vec[:, order].float().contiguous(),
        var[order].clamp_min(0).float(),
    )


def _restore_sink(out, book, n_sink):
    out[:n_sink] = book[:n_sink]
    return out


def shuffle_components(books, pca: PCA, *, keep, n_sink, n_tail, seed, block=BLOCK):
    """Permute each component past the first `keep` independently within each block."""
    dev = books[0].device
    g = torch.Generator(device=dev).manual_seed(int(seed))
    T, D = books[0].shape
    keep = min(keep, D)
    ids = block_ids(T, n_sink, n_tail, block).to(dev)
    out = []
    for b in books:
        z = (b - pca.mean) @ pca.basis
        if keep < D:
            for blk in range(int(ids.max()) + 1):
                rows = torch.nonzero(ids == blk).squeeze(1)
                sub = z[rows, keep:]
                perm = torch.argsort(
                    torch.rand(sub.shape, generator=g, device=dev), dim=0
                )
                z[rows, keep:] = torch.gather(sub, 0, perm)
        out.append(_restore_sink(z @ pca.basis.T + pca.mean, b, n_sink))
    return out


def gaussianise_components(books, pca: PCA, *, n_sink, chunk=512):
    """Map each component, by rank over all non-sink rows, to Gaussian quantiles
    with that component's mean and standard deviation."""
    sizes = [len(b) - n_sink for b in books]
    x = torch.cat([b[n_sink:] for b in books]) - pca.mean
    N, D = x.shape
    u = (torch.arange(N, device=x.device, dtype=torch.float64) + 0.5) / N
    quant = (np.sqrt(2.0) * torch.erfinv(2 * u - 1)).float()
    acc = torch.zeros_like(x)
    for j0 in range(0, D, chunk):
        v = pca.basis[:, j0 : j0 + chunk]
        z = x @ v
        ranks = torch.argsort(torch.argsort(z, dim=0), dim=0)
        q = quant[ranks]
        q = (q - q.mean(0)) / q.std(0) * z.std(0) + z.mean(0)
        acc += q @ v.T
        del z, ranks, q
    acc += pca.mean
    out, start = [], 0
    for b, n in zip(books, sizes, strict=True):
        o = torch.empty_like(b)
        o[n_sink:] = acc[start : start + n]
        out.append(_restore_sink(o, b, n_sink))
        start += n
    return out


def gaussian_books(books, pca: PCA, *, n_sink, seed):
    """Replace every non-sink row with a draw from N(mean, cov)."""
    dev = books[0].device
    g = torch.Generator(device=dev).manual_seed(int(seed))
    scale = pca.basis * pca.var.sqrt()
    out = []
    for b in books:
        z = torch.randn(len(b) - n_sink, b.shape[1], generator=g, device=dev)
        o = torch.empty_like(b)
        o[n_sink:] = z @ scale.T + pca.mean
        out.append(_restore_sink(o, b, n_sink))
    return out


def transform(name, books, pca: PCA, *, n_sink, n_tail, seed, block=BLOCK):
    """The books with set `name`'s property removed, rounded to bf16."""
    if name == "real":
        return books
    if name == "gaussian":
        out = gaussian_books(books, pca, n_sink=n_sink, seed=seed)
    elif name in KEEP:
        out = shuffle_components(
            books,
            pca,
            keep=KEEP[name],
            n_sink=n_sink,
            n_tail=n_tail,
            seed=seed,
            block=block,
        )
    elif name == "gz":
        out = gaussianise_components(books, pca, n_sink=n_sink)
    else:
        raise ValueError(f"unknown set {name!r}; choose from {SETS}")
    return [o.bfloat16().float() for o in out]


def transform_seed(layer: int, seed: int, name: str) -> int:
    return 1_000_003 * seed + 1009 * layer + SETS.index(name)


def flat_rows(books, *, n_sink, n_tail):
    """Rows and flags laid out as `load_kv_capture(..., n_tail=...)` lays them out."""
    x = np.vstack([np.asarray(b[n_sink:].cpu(), dtype=np.float32) for b in books])
    flags = np.concatenate([kv_flags(len(b), n_sink, n_tail) for b in books])
    return x, flags


def _neighbour_fields(c) -> dict:
    o = neighbour_positions(c, k=10)["ood"]
    return {f"nbr_{k}": o[k] for k in NEIGHBOUR_FIELDS}


def _done(path: Path, n_per_unit: int) -> set:
    if not path.exists():
        return set()
    counts: dict = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        key = (r["layer"], r["source"], r["seed"])
        counts[key] = counts.get(key, 0) + 1
    return {k for k, n in counts.items() if n == n_per_unit}


def _append_all(path: Path, recs) -> None:
    with path.open("a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def run_surgery(
    capture_dir,
    *,
    books,
    layers,
    sets,
    seeds,
    out_dir,
    n_sink=1,
    n_tail=1024,
    nlist=256,
    hnsw_m=48,
    ef_construction=500,
    threads=15,
    block=BLOCK,
    intrinsic_kw=None,
) -> None:
    """Measure every (layer, set, seed) once into out_dir/gate.jsonl and
    out_dir/intrinsic.jsonl. A unit's records are written together after it
    finishes, and finished units are skipped, so a rerun resumes."""
    import faiss

    bad = [s for s in sets if s not in SETS]
    if bad:
        raise ValueError(f"unknown sets {bad}; choose from {SETS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_path, intr_path = out_dir / "gate.jsonl", out_dir / "intrinsic.jsonl"
    faiss.omp_set_num_threads(threads)
    intrinsic_kw = {**INTRINSIC, **(intrinsic_kw or {})}
    dev = kv_gate._device()
    for layer in layers:
        w = load_layer_weights(capture_dir, layer)
        wk, wq = w["k_proj"].to(dev).float(), w["q_proj"].to(dev).float()
        n_heads = wk.shape[0] // kv_gate.HEAD_DIM
        todo = [
            (name, seed)
            for seed in seeds
            for name in sets
            if (layer, name, seed) not in _done(gate_path, n_heads)
            or (layer, name, seed) not in _done(intr_path, 2)
        ]
        if not todo:
            continue
        h_books = [
            h.to(dev).float() for h in kv_gate.load_books(capture_dir, layer, books)
        ]
        t0 = time.time()
        pca = fit_pca(h_books, n_sink)
        print(f"L{layer} pca {time.time() - t0:.1f} s", flush=True)
        for name, seed in todo:
            t0 = time.time()
            hb = transform(
                name,
                h_books,
                pca,
                n_sink=n_sink,
                n_tail=n_tail,
                seed=transform_seed(layer, seed, name),
                block=block,
            )
            t_tr = time.time() - t0
            if (layer, name, seed) not in _done(gate_path, n_heads):
                recs = list(
                    kv_gate.real_layer_records(
                        hb,
                        wk,
                        wq,
                        layer,
                        books,
                        seed,
                        n_sink=n_sink,
                        n_tail=n_tail,
                        nlist=nlist,
                        hnsw_m=hnsw_m,
                        ef_construction=ef_construction,
                        source=name,
                        extra_fn=_neighbour_fields,
                    )
                )
                _append_all(gate_path, recs)
            if (layer, name, seed) not in _done(intr_path, 2):
                x, flags = flat_rows(hb, n_sink=n_sink, n_tail=n_tail)
                lines = []
                for flag, f in (("body", BODY), ("tail", TAIL)):
                    ti = time.time()
                    rec = {
                        "layer": layer,
                        "source": name,
                        "seed": seed,
                        "flag": flag,
                        "n_rows": int((flags == f).sum()),
                    }
                    rec |= summary(compute(x[flags == f], **intrinsic_kw))
                    rec["seconds"] = round(time.time() - ti, 1)
                    lines.append(rec)
                _append_all(intr_path, lines)
                del x
            del hb
            print(
                f"L{layer} {name} seed {seed}: transform {t_tr:.1f} s, "
                f"total {time.time() - t0:.1f} s",
                flush=True,
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument(
        "--books", type=kv_gate.parse_books, default=kv_gate.parse_books("12-19")
    )
    ap.add_argument("--layers", required=True, help="comma-separated, e.g. 11,15,43")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--threads", type=int, default=15)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    run_surgery(
        a.capture_dir,
        books=a.books,
        layers=[int(x) for x in a.layers.split(",")],
        sets=a.sets.split(","),
        seeds=[int(x) for x in a.seeds.split(",")],
        out_dir=a.out_dir,
        threads=a.threads,
    )
    print("SURGERY_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
