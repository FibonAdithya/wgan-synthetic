"""Measure what building one GPU ANN index over a real corpus costs.

One index build per process invocation (one CUDA context per process, as the
GPU box notes require), repeated `--repeats` times inside that context. For
each build the probe records:

* wall time of `build()`, fenced with a device sync on both sides;
* peak device memory, three ways, because no single one is complete:
    - `rmm_peak_bytes`: the high-water mark of every allocation that went
      through RMM, which is where cuVS and (here, by routing cupy through
      RMM) the dataset copy both allocate. Exact and process-local, but
      blind to the CUDA context and to anything cuVS allocates outside RMM.
    - `card_peak_delta_bytes`: max over samples of card-wide used memory
      minus the pre-dataset baseline, sampled at ~200 Hz by a child process
      (cuVS holds the GIL through `build()`, so a thread cannot). Card-wide,
      so it needs the job to hold the card alone; then it is the one figure
      that sees allocations made outside RMM.
    - `context_mib_before_dataset`: one nvidia-smi reading of this pid before
      the dataset copy, i.e. the CUDA context.

With `--cap-mib`, the probe also enforces the cap the way the design's
balloon does and reports whether the build survived it.
* device bytes still held after the build (`resident_after_bytes`), the
  index plus the dataset it points into;
* host RSS high-water mark, which is dominated by the corpus load.

The c004 index-build split caps the build at 2 GiB of device memory and 600 s
of wall clock (tig-monorepo, docs/superpowers/specs/2026-08-31-c004-index-
build-split-design.md); the numbers here are what to hold against that.

Example, through the queue:
    python -m tools.probes.index_build_probe --real-path /workspace/data-cache/sift_1m.npy \
        --kind cagra --build-algo nn_descent --output docs/results/index-build-1m/cagra_nn_descent.json
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

METRIC = "sqeuclidean"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kind", required=True, choices=["ivf_flat", "ivf_pq", "cagra", "flat"])
    p.add_argument("--n", type=int, default=1_000_000, help="rows to build over (first n of the file)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--n-lists", type=int, default=4096, help="ivf_flat / ivf_pq coarse clusters")
    p.add_argument("--pq-dim", type=int, default=64)
    p.add_argument("--pq-bits", type=int, default=8)
    p.add_argument("--graph-degree", type=int, default=64)
    p.add_argument("--intermediate-graph-degree", type=int, default=128)
    p.add_argument("--build-algo", default="nn_descent", choices=["nn_descent", "ivf_pq", "iterative_cagra_search"])
    p.add_argument("--normalize", action="store_true", help="L2-normalize rows, as the ann benchmarks do")
    p.add_argument(
        "--cap-mib", type=int, default=0,
        help="Enforce the c004 build cap the way the design does: after the dataset is on the device, "
        "allocate free - (cap + 64 MiB) with raw cudaMalloc and hold it for the whole build. 0 disables.",
    )
    return p.parse_args(argv)


SAMPLER_SRC = r"""
import sys, time
import cupy
out = open(sys.argv[1], "w")
interval = float(sys.argv[2])
while True:
    free, total = cupy.cuda.runtime.memGetInfo()
    out.write(f"{time.time():.6f} {total - free}\n")
    out.flush()
    time.sleep(interval)
"""


class CardSampler:
    """Card-wide used bytes, sampled by a child process.

    A thread cannot do this: cuVS holds the GIL for the whole of `build()`,
    so an in-process sampler gets one sample at each end and nothing in
    between (measured 2026-09-04: 1-4 samples per 20 s build). The child has
    its own CUDA context, which is constant and part of the baseline.

    Each job holds the whole card (no --vram-mb), so the delta from the
    pre-build baseline is this process's growth, including anything cuVS
    allocates outside RMM."""

    def __init__(self, path: Path, interval_s: float = 0.005) -> None:
        self.path = path
        self.proc = subprocess.Popen([sys.executable, "-c", SAMPLER_SRC, str(path), str(interval_s)])

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=10)

    def samples(self) -> list[tuple[float, int]]:
        out = []
        try:
            for line in self.path.read_text().splitlines():
                t, used = line.split()
                out.append((float(t), int(used)))
        except (OSError, ValueError):
            pass
        return out

    def window(self, t0: float, t1: float) -> list[int]:
        return [b for t, b in self.samples() if t0 <= t <= t1]


def own_mib_now(pid: int) -> int:
    """This pid's memory as nvidia-smi reports it (about 2 s per call)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:  # noqa: BLE001 - must never kill the run
        return -1
    for line in out.splitlines():
        parts = [s.strip() for s in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            return int(parts[1])
    return 0


def build_once(kind: str, args, device_vectors, res):
    if kind == "flat":
        from cuvs.neighbors import brute_force
        return brute_force.build(device_vectors, metric=METRIC, resources=res)
    if kind == "ivf_flat":
        from cuvs.neighbors import ivf_flat
        params = ivf_flat.IndexParams(n_lists=args.n_lists, metric=METRIC)
        return ivf_flat.build(params, device_vectors, resources=res)
    if kind == "ivf_pq":
        from cuvs.neighbors import ivf_pq
        params = ivf_pq.IndexParams(n_lists=args.n_lists, metric=METRIC, pq_dim=args.pq_dim, pq_bits=args.pq_bits)
        return ivf_pq.build(params, device_vectors, resources=res)
    if kind == "cagra":
        from cuvs.neighbors import cagra
        params = cagra.IndexParams(
            metric=METRIC,
            graph_degree=args.graph_degree,
            intermediate_graph_degree=args.intermediate_graph_degree,
            build_algo=args.build_algo,
        )
        return cagra.build(params, device_vectors, resources=res)
    raise ValueError(kind)


def describe(kind: str, args) -> dict:
    if kind == "flat":
        return {"metric": METRIC}
    if kind == "ivf_flat":
        return {"n_lists": args.n_lists, "metric": METRIC}
    if kind == "ivf_pq":
        return {"n_lists": args.n_lists, "pq_dim": args.pq_dim, "pq_bits": args.pq_bits, "metric": METRIC}
    return {
        "graph_degree": args.graph_degree,
        "intermediate_graph_degree": args.intermediate_graph_degree,
        "build_algo": args.build_algo,
        "metric": METRIC,
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    import os

    import cupy
    import cuvs
    import rmm
    from cuvs.common import Resources
    from rmm.allocators.cupy import rmm_cupy_allocator

    # No pool: a pool would hide frees and make the peak a pool high-water
    # mark rather than a live-bytes high-water mark. Every allocation, cuVS's
    # and cupy's, goes through the one statistics adaptor.
    stats = rmm.mr.StatisticsResourceAdaptor(rmm.mr.CudaMemoryResource())
    rmm.mr.set_current_device_resource(stats)
    cupy.cuda.set_allocator(rmm_cupy_allocator)
    res = Resources()
    gpu = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    _, total_bytes = cupy.cuda.runtime.memGetInfo()

    t0 = time.perf_counter()
    rows = np.load(args.real_path, mmap_mode="r")
    if rows.shape[0] < args.n:
        raise SystemExit(f"{args.real_path} has {rows.shape[0]} rows; need {args.n}")
    host = np.ascontiguousarray(rows[: args.n], dtype=np.float32)
    if args.normalize:
        host /= np.maximum(np.linalg.norm(host, axis=1, keepdims=True), 1e-12)
    load_s = time.perf_counter() - t0
    print(f"loaded {host.shape} in {load_s:.1f}s", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sampler = CardSampler(Path(tempfile.mkdtemp(prefix="index_build_probe_")) / "samples.txt")
    context_mib = own_mib_now(os.getpid())
    time.sleep(1.0)
    baseline_samples = sampler.samples()
    if not baseline_samples:
        raise SystemExit("card sampler produced no samples; is cupy importable in the child?")
    baseline_card = min(b for _, b in baseline_samples)

    res.sync()
    t0 = time.perf_counter()
    device_vectors = cupy.asarray(host)
    res.sync()
    h2d_s = time.perf_counter() - t0
    dataset_bytes = int(device_vectors.nbytes)

    balloon = None
    balloon_bytes = 0
    free_before_balloon, _ = cupy.cuda.runtime.memGetInfo()
    if args.cap_mib:
        cap = args.cap_mib * 2**20
        headroom = 64 * 2**20
        if free_before_balloon < cap + headroom:
            raise SystemExit(f"free {free_before_balloon} < cap + headroom {cap + headroom}; cannot enforce")
        balloon_bytes = free_before_balloon - (cap + headroom)
        balloon = cupy.cuda.runtime.malloc(balloon_bytes)  # raw cudaMalloc: invisible to the RMM stats
        free_after, _ = cupy.cuda.runtime.memGetInfo()
        print(f"balloon {balloon_bytes / 2**20:.0f} MiB held; {free_after / 2**20:.0f} MiB free for the build", flush=True)

    builds = []
    handle = None
    for i in range(args.repeats):
        handle = None  # free the previous index before rebuilding
        res.sync()
        counts_before = stats.allocation_counts
        t0 = time.perf_counter()
        wall0 = time.time()
        error = None
        try:
            handle = build_once(args.kind, args, device_vectors, res)
            res.sync()
        except Exception as exc:  # noqa: BLE001 - an OOM under the cap is a result, not a crash
            error = f"{type(exc).__name__}: {exc}"[:500]
            handle = None
        elapsed = time.perf_counter() - t0
        counts_after = stats.allocation_counts
        card = sampler.window(wall0, wall0 + elapsed)
        builds.append(
            {
                "repeat": i,
                "build_seconds": elapsed,
                "error": error,
                "rmm_current_before_bytes": int(counts_before.current_bytes),
                "rmm_peak_bytes": int(counts_after.peak_bytes),
                "resident_after_bytes": int(counts_after.current_bytes),
                "card_peak_delta_bytes": max(card, default=baseline_card) - baseline_card,
                "card_samples": len(card),
            }
        )
        print(
            f"{args.kind} build {i}: {elapsed:.2f}s, rmm peak {counts_after.peak_bytes / 2**20:.0f} MiB, "
            f"card delta peak {builds[-1]['card_peak_delta_bytes'] / 2**20:.0f} MiB over {len(card)} samples, "
            f"resident {counts_after.current_bytes / 2**20:.0f} MiB"
            + (f", FAILED: {error}" if error else ""),
            flush=True,
        )
        if error:
            break
    sampler.stop()
    del handle
    if balloon is not None:
        cupy.cuda.runtime.free(balloon)

    result = {
        "kind": args.kind,
        "params": describe(args.kind, args),
        "real_path": args.real_path,
        "shape": {"n": int(host.shape[0]), "dim": int(host.shape[1]), "dtype": "float32", "normalized": args.normalize},
        "dataset_bytes": dataset_bytes,
        "load_seconds": load_s,
        "h2d_seconds": h2d_s,
        "context_mib_before_dataset": context_mib,
        "baseline_card_bytes": int(baseline_card),
        "cap_mib": args.cap_mib,
        "balloon_bytes": int(balloon_bytes),
        "free_before_balloon_bytes": int(free_before_balloon),
        "builds": builds,
        "host_maxrss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "environment": {
            "gpu": gpu,
            "gpu_total_bytes": int(total_bytes),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cuvs": cuvs.__version__,
            "cupy": cupy.__version__,
            "rmm": rmm.__version__,
        },
        "notes": [
            "rmm_peak_bytes is the process-wide RMM high-water mark since process start, so repeat i>0 "
            "cannot report a lower peak than repeat 0.",
            "card_peak_delta_bytes is card-wide used memory minus the pre-dataset baseline, from a child "
            "process sampling the driver; it includes the dataset and anything allocated outside RMM, and "
            "excludes both CUDA contexts (parent and sampler), which are in the baseline.",
        ],
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
