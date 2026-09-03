"""Sweep the per-nonce query-set size for one TIG vector_search track.

The TIG challenge answers `n_queries` queries per nonce against a fixed
database, gates on recall@1 >= 0.90, and ranks qualifiers on solve time. Two
things scale with `n_queries` and pull in opposite directions:

* Throughput. A GPU index needs a large enough batch to saturate the card;
  below that knee the measured solve time is launch overhead and
  under-occupancy, not the algorithm. The per-nonce floor with a do-nothing
  solver was measured at ~23 ms on this card (tig-monorepo,
  docs/measurements/2026-08-31-c004-post-split-nonce-time.md), so search must
  cost several times that before speed differences between algorithms are
  visible at all.
* Cost. Generation and verification are linear in `n_queries`.

This script measures, for one corpus at the challenge's shape (700,000
database rows), recall calibration of two indexes to the 0.90 bar, the
device-resident search time of each as a function of batch size, and the
nonce-to-nonce spread of recall at each size, from per-query hit indicators
over a held-out pool.

Corpora are L2-normalized, as everywhere in `src.eval.ann_benchmark`, so
squared L2 is monotone in the angular metric DEEP and GloVe are searched
with. Queries are held-out rows of the same corpus: the challenge draws both
halves of an instance from one generator, so held-out rows are the right
stand-in, and the ann-benchmarks `test` split is only 10,000 rows in any
case.

Example (on the GPU box, through the queue):
    python -m src.eval.query_set_sweep --corpus deep \
        --real-path /workspace/data-cache/deep_1m.npy \
        --output-dir docs/results/query-set-sweep/deep
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.eval.ann_benchmark import indexes
from src.eval.ann_benchmark.corpora import normalize
from src.eval.ann_benchmark.metrics import recompute_exact_distances

DATABASE_SIZE = 700_000
POOL_SIZE = 100_000
SIZES = (500, 1_000, 2_000, 4_000, 7_000, 10_000, 20_000, 40_000, 70_000, 100_000)
K = 1
TARGET_RECALL = 0.90
# The challenge's `recall_tolerance`, applied here to squared distances. A
# relative 1e-6 on d is ~2e-6 on d^2; the difference is far below any float
# noise this comparison is meant to absorb.
TIE_EPS = 1e-6
REPEATS = 7
DRAWS = 400
TRUTH_CHUNK = 10_000
# Measured per-nonce floor with a do-nothing solver, RTX 3060, post-split.
NONCE_FLOOR_MS = 22.7


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", required=True, help="Label, e.g. sift/deep/glove.")
    parser.add_argument("--real-path", required=True, help=".npy with >= 800k rows.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--database-size", type=int, default=DATABASE_SIZE)
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def split_corpus(
    path: Path, database_size: int, pool_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint database and query pool from one seeded permutation."""
    rows = np.load(path, mmap_mode="r")
    need = database_size + pool_size
    if rows.shape[0] < need:
        raise SystemExit(
            f"{path} has {rows.shape[0]} rows; need {need} for a "
            f"{database_size} database plus a {pool_size} held-out pool."
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(rows.shape[0])
    db_idx = np.sort(perm[:database_size])
    pool_idx = np.sort(perm[database_size:need])
    database = normalize(np.asarray(rows[db_idx], dtype=np.float32))
    pool = normalize(np.asarray(rows[pool_idx], dtype=np.float32))
    return database, pool


class Timed:
    """Device-resident search timing, one cuVS call per index kind.

    The adapters in `indexes.py` copy queries host->device inside `search`;
    the challenge generates queries on the device, so that copy is not part
    of a nonce and is excluded here.
    """

    def __init__(self) -> None:
        self.cupy, resources_cls = indexes._require_cuvs()
        self.res = resources_cls()

    def sync(self) -> None:
        self.res.sync()

    def to_device(self, x: np.ndarray):
        return self.cupy.asarray(np.ascontiguousarray(x, dtype=np.float32))

    def to_host(self, x) -> np.ndarray:
        return self.cupy.asnumpy(x)

    def search(self, kind: str, handle, device_queries, k: int, param: int | None):
        if kind == "flat":
            from cuvs.neighbors import brute_force

            return brute_force.search(handle, device_queries, k)
        if kind == "ivf_flat":
            from cuvs.neighbors import ivf_flat

            return ivf_flat.search(
                ivf_flat.SearchParams(n_probes=int(param)), handle, device_queries, k
            )
        if kind == "cagra_iters":
            from cuvs.neighbors import cagra

            return cagra.search(
                cagra.SearchParams(
                    itopk_size=indexes.CAGRA_ITOPK_FLOOR, max_iterations=int(param)
                ),
                handle,
                device_queries,
                k,
            )
        raise ValueError(kind)

    def time_search(
        self, kind: str, handle, device_queries, k: int, param: int | None, repeats: int
    ) -> list[float]:
        # Warmup, discarded: first-call JIT and allocator growth.
        self.sync()
        self.search(kind, handle, device_queries, k, param)
        self.sync()
        seconds: list[float] = []
        for _ in range(repeats):
            self.sync()
            started = time.perf_counter()
            self.search(kind, handle, device_queries, k, param)
            self.sync()
            seconds.append(time.perf_counter() - started)
        return seconds


def exact_truth(
    timed: Timed, flat_handle, database: np.ndarray, pool: np.ndarray, k: int
) -> np.ndarray:
    """Exact squared-L2 distance to the true k-th neighbour, chunked.

    Recomputed from the returned ids rather than read off cuVS's reported
    distances, so truth and every index's found distances are in one float
    space (see `recompute_exact_distances`; commit 7c00b45).
    """
    ids = np.empty((pool.shape[0], k), dtype=np.int64)
    for start in range(0, pool.shape[0], TRUTH_CHUNK):
        chunk = pool[start : start + TRUTH_CHUNK]
        _d, found = timed.search("flat", flat_handle, timed.to_device(chunk), k, None)
        ids[start : start + chunk.shape[0]] = timed.to_host(found)
    timed.sync()
    return recompute_exact_distances(database, pool, ids)


def hits_for(
    timed: Timed,
    kind: str,
    handle,
    database: np.ndarray,
    pool: np.ndarray,
    truth: np.ndarray,
    k: int,
    param: int | None,
) -> np.ndarray:
    """Per-query hit indicator, scored from recomputed exact distances."""
    ids = np.empty((pool.shape[0], k), dtype=np.int64)
    for start in range(0, pool.shape[0], TRUTH_CHUNK):
        chunk = pool[start : start + TRUTH_CHUNK]
        _d, found = timed.search(kind, handle, timed.to_device(chunk), k, param)
        ids[start : start + chunk.shape[0]] = timed.to_host(found)
    timed.sync()
    found_d = recompute_exact_distances(database, pool, ids)
    threshold = truth[:, -1:] * (1.0 + TIE_EPS)
    return (found_d[:, :k] <= threshold).all(axis=1)


def calibrate(
    timed, kind, handle, database, pool, truth, k, params, target
) -> tuple[list[dict], int | None, np.ndarray | None]:
    """Walk `params` ascending; stop at the first clearing `target`."""
    curve: list[dict] = []
    chosen = None
    chosen_hits = None
    for param in params:
        hits = hits_for(timed, kind, handle, database, pool, truth, k, param)
        recall = float(hits.mean())
        curve.append({"param": int(param), "recall": recall})
        print(f"  {kind} param={param} recall@{k}={recall:.4f}", flush=True)
        if recall >= target:
            chosen, chosen_hits = int(param), hits
            break
    return curve, chosen, chosen_hits


def spread_table(hits: np.ndarray, sizes: Sequence[int], draws: int, seed: int) -> list[dict]:
    """Recall spread over `draws` subsets of each size, without replacement."""
    rng = np.random.default_rng(seed)
    pool_n = hits.shape[0]
    p = float(hits.mean())
    rows = []
    for n in sizes:
        if n > pool_n:
            continue
        recalls = np.empty(draws)
        for i in range(draws):
            recalls[i] = hits[rng.choice(pool_n, size=n, replace=False)].mean()
        fpc = np.sqrt(max(0.0, 1.0 - (n - 1) / (pool_n - 1)))
        rows.append(
            {
                "n": int(n),
                "mean": float(recalls.mean()),
                "sd": float(recalls.std(ddof=1)),
                "p05": float(np.percentile(recalls, 5)),
                "p95": float(np.percentile(recalls, 95)),
                "binomial_sd": float(np.sqrt(p * (1 - p) / n) * fpc),
            }
        )
    return rows


def fit_linear(ns: Sequence[int], secs: Sequence[float]) -> dict:
    """Least squares t = t0 + n / qps on medians."""
    x = np.asarray(ns, dtype=np.float64)
    y = np.asarray(secs, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    return {
        "t0_ms": float(intercept * 1e3),
        "qps_asymptotic": float(1.0 / slope) if slope > 0 else None,
        "rms_resid_ms": float(np.sqrt(np.mean(resid**2)) * 1e3),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indexes.require_device_stack()
    timed = Timed()

    print(f"loading {args.real_path}", flush=True)
    database, pool = split_corpus(
        Path(args.real_path), args.database_size, args.pool_size, args.seed
    )
    dim = database.shape[1]
    print(f"database {database.shape}, pool {pool.shape}", flush=True)

    adapters = {
        "flat": indexes.FlatAdapter(),
        "ivf_flat": indexes.IvfFlatAdapter(),
        "cagra_iters": indexes.CagraIterationsAdapter(),
    }
    builds: dict[str, dict] = {}
    handles: dict[str, object] = {}
    for kind, adapter in adapters.items():
        built = adapter.build(database)
        adapter.sync()
        handles[kind] = built.handle
        builds[kind] = {
            "train_seconds": built.train_seconds,
            "add_seconds": built.add_seconds,
            "peak_vram_bytes": built.peak_vram_bytes,
            "describe": adapter.describe(),
        }
        print(f"built {kind} in {built.train_seconds + built.add_seconds:.1f}s", flush=True)
        # Keep the device buffer alive; the handle points into it.
        builds[kind]["_dataset"] = built.dataset

    truth = exact_truth(timed, handles["flat"], database, pool, K)

    calibration: dict[str, dict] = {}
    hits: dict[str, np.ndarray] = {}
    hits["flat"] = hits_for(timed, "flat", handles["flat"], database, pool, truth, K, None)
    calibration["flat"] = {"curve": [{"param": None, "recall": float(hits["flat"].mean())}], "chosen": None}
    for kind, params in (
        ("ivf_flat", indexes.IVF_N_PROBES),
        ("cagra_iters", indexes.CAGRA_MAX_ITERATIONS),
    ):
        curve, chosen, chosen_hits = calibrate(
            timed, kind, handles[kind], database, pool, truth, K, params, args.target_recall
        )
        calibration[kind] = {"curve": curve, "chosen": chosen}
        if chosen_hits is not None:
            hits[kind] = chosen_hits

    spread = {kind: spread_table(h, args.sizes, args.draws, args.seed) for kind, h in hits.items()}

    timing: dict[str, list[dict]] = {}
    fits: dict[str, dict] = {}
    for kind in ("flat", "ivf_flat", "cagra_iters"):
        param = calibration[kind]["chosen"]
        if kind != "flat" and param is None:
            print(f"{kind}: no param reached {args.target_recall}; timing skipped", flush=True)
            continue
        rows = []
        for n in args.sizes:
            if n > pool.shape[0]:
                continue
            dq = timed.to_device(pool[:n])
            secs = timed.time_search(kind, handles[kind], dq, K, param, args.repeats)
            del dq
            med = float(np.median(secs))
            rows.append(
                {
                    "n": int(n),
                    "median_ms": med * 1e3,
                    "min_ms": float(min(secs)) * 1e3,
                    "p95_ms": float(np.percentile(secs, 95)) * 1e3,
                    "qps_median": n / med,
                    "seconds": [float(s) for s in secs],
                }
            )
            print(f"  {kind} n={n}: {med * 1e3:.2f} ms median, {n / med:,.0f} qps", flush=True)
        timing[kind] = rows
        fits[kind] = fit_linear([r["n"] for r in rows], [r["median_ms"] / 1e3 for r in rows])
        peak = max(r["qps_median"] for r in rows)
        knee = next((r["n"] for r in rows if r["qps_median"] >= 0.9 * peak), None)
        fits[kind]["qps_peak_measured"] = peak
        fits[kind]["knee_n_90pct_of_peak"] = knee
        fits[kind]["n_where_search_ge_3x_floor"] = next(
            (r["n"] for r in rows if r["median_ms"] >= 3 * NONCE_FLOOR_MS), None
        )

    for b in builds.values():
        b.pop("_dataset", None)

    result = {
        "corpus": args.corpus,
        "real_path": str(args.real_path),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": indexes.stack_versions(),
            "gpu": _gpu_name(timed),
        },
        "shape": {
            "database_size": int(database.shape[0]),
            "pool_size": int(pool.shape[0]),
            "dim": int(dim),
            "k": K,
            "sizes": list(args.sizes),
            "repeats": args.repeats,
            "draws": args.draws,
            "target_recall": args.target_recall,
            "tie_eps": TIE_EPS,
            "nonce_floor_ms": NONCE_FLOOR_MS,
            "seed": args.seed,
        },
        "builds": builds,
        "calibration": calibration,
        "spread": spread,
        "timing": timing,
        "fits": fits,
    }
    (out_dir / "query_set_sweep.json").write_text(json.dumps(result, indent=2))
    (out_dir / "query_set_sweep.md").write_text(render_markdown(result))
    print(f"wrote {out_dir / 'query_set_sweep.json'}", flush=True)


def _gpu_name(timed: Timed) -> str | None:
    try:
        props = timed.cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:  # noqa: BLE001 - provenance must not fail the run
        return None


def render_markdown(result: dict) -> str:
    lines = [f"# Query-set sweep: {result['corpus']}", ""]
    s = result["shape"]
    lines.append(
        f"database {s['database_size']:,} x {s['dim']}, pool {s['pool_size']:,}, "
        f"k={s['k']}, target recall {s['target_recall']}, {s['repeats']} repeats."
    )
    lines.append("")
    lines.append("## Calibration")
    for kind, cal in result["calibration"].items():
        lines.append(f"- {kind}: chosen={cal['chosen']}; " + ", ".join(
            f"{c['param']}->{c['recall']:.4f}" for c in cal["curve"]
        ))
    lines.append("")
    lines.append("## Search time vs n (median ms, qps)")
    kinds = list(result["timing"])
    lines.append("| n | " + " | ".join(f"{k} ms | {k} qps" for k in kinds) + " |")
    lines.append("|---|" + "---|---|" * len(kinds))
    by_n: dict[int, dict[str, dict]] = {}
    for kind in kinds:
        for r in result["timing"][kind]:
            by_n.setdefault(r["n"], {})[kind] = r
    for n in sorted(by_n):
        cells = []
        for kind in kinds:
            r = by_n[n].get(kind)
            cells.append(f"{r['median_ms']:.2f} | {r['qps_median']:,.0f}" if r else " | ")
        lines.append(f"| {n:,} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Fits (t = t0 + n/qps)")
    for kind, f in result["fits"].items():
        lines.append(f"- {kind}: {json.dumps(f)}")
    lines.append("")
    lines.append("## Recall spread vs n (sd over draws; binomial sd)")
    for kind, rows in result["spread"].items():
        lines.append(f"- {kind}: " + ", ".join(
            f"n={r['n']}: {r['sd']:.4f} ({r['binomial_sd']:.4f})" for r in rows
        ))
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
