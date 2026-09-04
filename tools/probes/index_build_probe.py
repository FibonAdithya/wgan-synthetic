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
    - `smi_peak_mib`: the largest `nvidia-smi` reading for this pid during
      the build, sampled as fast as nvidia-smi answers (~0.5 Hz on the box).
      Includes the context; misses anything shorter than one sample.
    - `card_peak_delta_bytes`: max over samples of card-wide used memory
      minus the pre-build baseline, sampled at ~200 Hz. Card-wide, so it
      needs the job to hold the card alone; then it is the one figure that
      sees allocations made outside RMM.
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
import threading
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
    return p.parse_args(argv)


class SmiSampler:
    """Two samplers: the driver's card-wide used bytes at ~200 Hz, and this
    pid's nvidia-smi figure from a second thread. nvidia-smi takes ~2 s per
    call on the GPU box (measured 2026-09-04), so it must never sit in the
    fast loop: the first version of this probe did that and produced 0 to 1
    samples per build.

    Each job holds the whole card (no --vram-mb), so the card-wide delta from
    the pre-build baseline is this process's growth, including anything
    allocated outside RMM."""

    def __init__(self, pid: int, cupy) -> None:
        self.pid = pid
        self.cupy = cupy
        self.samples: list[tuple[float, int]] = []  # (t, card_used_bytes)
        self.smi_samples: list[tuple[float, int]] = []  # (t, own_mib)
        self._stop = threading.Event()
        self._fast = threading.Thread(target=self._run_fast, daemon=True)
        self._slow = threading.Thread(target=self._run_slow, daemon=True)

    def _read_own_mib(self) -> int:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout
        except Exception:  # noqa: BLE001 - sampler must never kill the run
            return -1
        for line in out.splitlines():
            parts = [s.strip() for s in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == self.pid:
                return int(parts[1])
        return 0

    def _run_fast(self) -> None:
        while not self._stop.is_set():
            free, total = self.cupy.cuda.runtime.memGetInfo()
            self.samples.append((time.perf_counter(), int(total - free)))
            self._stop.wait(0.005)

    def _run_slow(self) -> None:
        while not self._stop.is_set():
            own = self._read_own_mib()
            self.smi_samples.append((time.perf_counter(), own))

    def start(self) -> None:
        self._fast.start()
        self._slow.start()

    def stop(self) -> None:
        self._stop.set()
        self._fast.join()
        self._slow.join(timeout=15)

    def window(self, t0: float, t1: float) -> tuple[list[int], list[int]]:
        card = [b for t, b in self.samples if t0 <= t <= t1]
        own = [m for t, m in self.smi_samples if t0 <= t <= t1 + 3.0]
        return card, own


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

    sampler = SmiSampler(os.getpid(), cupy)
    sampler.start()
    time.sleep(2.5)
    baseline_card = min(b for _, b in sampler.samples) if sampler.samples else 0
    context_mib = max((m for _, m in sampler.smi_samples), default=-1)

    res.sync()
    t0 = time.perf_counter()
    device_vectors = cupy.asarray(host)
    res.sync()
    h2d_s = time.perf_counter() - t0
    dataset_bytes = int(device_vectors.nbytes)

    builds = []
    handle = None
    for i in range(args.repeats):
        handle = None  # free the previous index before rebuilding
        res.sync()
        counts_before = stats.allocation_counts
        t0 = time.perf_counter()
        handle = build_once(args.kind, args, device_vectors, res)
        res.sync()
        elapsed = time.perf_counter() - t0
        counts_after = stats.allocation_counts
        card, own = sampler.window(t0, t0 + elapsed)
        builds.append(
            {
                "repeat": i,
                "build_seconds": elapsed,
                "rmm_current_before_bytes": int(counts_before.current_bytes),
                "rmm_peak_bytes": int(counts_after.peak_bytes),
                "resident_after_bytes": int(counts_after.current_bytes),
                "smi_peak_mib": max(own, default=-1),
                "card_peak_delta_bytes": max(card, default=baseline_card) - baseline_card,
                "card_samples": len(card),
                "smi_samples": len(own),
            }
        )
        print(
            f"{args.kind} build {i}: {elapsed:.2f}s, rmm peak {counts_after.peak_bytes / 2**20:.0f} MiB, "
            f"smi peak {builds[-1]['smi_peak_mib']} MiB, resident {counts_after.current_bytes / 2**20:.0f} MiB",
            flush=True,
        )
    sampler.stop()
    del handle

    result = {
        "kind": args.kind,
        "params": describe(args.kind, args),
        "real_path": args.real_path,
        "shape": {"n": int(host.shape[0]), "dim": int(host.shape[1]), "dtype": "float32", "normalized": args.normalize},
        "dataset_bytes": dataset_bytes,
        "load_seconds": load_s,
        "h2d_seconds": h2d_s,
        "context_mib_before_dataset": context_mib,
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
            "smi_peak_mib includes the CUDA context; card_peak_delta_bytes is card-wide and may include other jobs.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
