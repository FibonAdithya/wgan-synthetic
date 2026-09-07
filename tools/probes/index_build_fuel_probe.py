"""Measure what an IVF-Flat index build costs in TIG *fuel* over a real corpus.

TIG fuel is not time. For GPU code it is a static per-PTX-instruction cost
table that `tig-binary/scripts/build_ptx` injects into every basic block of an
algorithm's PTX; each thread accumulates the cost of the blocks it executes
and adds it to a global counter when it returns, and tig-runtime divides that
counter by `GPU_FUEL_SCALE` (20). It can therefore only be measured on
hand-written kernels compiled through that script -- cuVS ships SASS and
cannot be metered. This probe:

1. compiles `ivf_build_fuel_kernels.cu` (k-means on a training subset, one
   assignment pass over the whole database, CSR lists, list-order copy: the
   build half of an IVF-Flat index as cuVS constructs one) together with the
   monorepo's `framework.cu`, and runs the monorepo's own `build_ptx`
   parse/inject functions over the PTX, so the accounting is byte-for-byte
   TIG's, including its cost table and its per-return atomics;
2. loads the instrumented PTX with cupy, runs the build over the first `--n`
   rows of a corpus, and reads `gbl_FUELUSAGE` after every phase;
3. records raw counter values, the value tig-runtime would report
   (raw // 20), wall time per phase, and the injector's static per-block
   costs, plus a straight-line calibration kernel that pins down how many
   times the injector adds each thread's fuel.

Fuel is a count of executed instructions, so it is data-independent to first
order: the same build over generator output would cost the same. It is not
kernel-independent -- a differently written build kernel costs differently,
and this is one reasonable implementation, not a bound.

One build per process (one CUDA context per process, as the GPU box requires).

Example:
    python tools/probes/index_build_fuel_probe.py \
        --real-path /workspace/data-cache/sift_1m.npy --corpus sift --n 1000000 \
        --n-lists 1024 --output docs/results/index-build-fuel/sift_n1000000_nlist1024.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KERNELS_CU = HERE / "ivf_build_fuel_kernels.cu"
GPU_FUEL_SCALE = 20  # tig-structs/src/config.rs
THREADS = 256
A_DB_TILE = 8
UPD_THREADS = 128


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-path", required=True, help=".npy (rows) or ann-benchmarks .hdf5 (uses 'train')")
    p.add_argument("--corpus", required=True, help="label: sift, deep, glove")
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, required=True, help="rows to build over (first n of the file)")
    p.add_argument("--n-lists", type=int, default=1024, help="cuVS ivf_flat default is 1024")
    p.add_argument("--n-iters", type=int, default=20, help="k-means iterations; cuVS kmeans_n_iters default is 20")
    p.add_argument("--trainset-fraction", type=float, default=0.5, help="cuVS kmeans_trainset_fraction default is 0.5")
    p.add_argument("--monorepo", default=os.environ.get("TIG_MONOREPO", "/workspace/tig-bench"),
                   help="tig-monorepo checkout providing build_ptx and framework.cu")
    p.add_argument("--nvcc", default=os.environ.get("NVCC", "/usr/local/cuda-12.6/bin/nvcc"))
    p.add_argument("--fuel-limit", type=int, default=1 << 62, help="value patched over 0xdeadbeefdeadbeef")
    p.add_argument("--keep-ptx", default="", help="also copy the instrumented PTX here")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Instrumentation, using the monorepo's own build_ptx
# ---------------------------------------------------------------------------

def load_build_ptx(path: Path):
    """Import build_ptx (no .py suffix) as a module without running main()."""
    os.environ.setdefault("CHALLENGE", "vector_search")  # it exits at import without one
    loader = importlib.machinery.SourceFileLoader("tig_build_ptx", str(path))
    spec = importlib.util.spec_from_loader("tig_build_ptx", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def instrument(args, workdir: Path) -> dict:
    monorepo = Path(args.monorepo)
    build_ptx_path = monorepo / "tig-binary" / "scripts" / "build_ptx"
    framework_cu = monorepo / "tig-binary" / "src" / "framework.cu"
    bp = load_build_ptx(build_ptx_path)

    framework = framework_cu.read_text()
    kernels = KERNELS_CU.read_text()
    # build_ptx skips (does not meter) framework and challenge functions; the
    # algorithm's own kernels are metered. Same regex, same split.
    kernel_regex = r'(?:extern\s+"C"\s+__global__|__device__)\s+\w+\s+(?P<func>\w+)\s*\('
    to_ignore = [m.group("func") for m in re.finditer(kernel_regex, framework)]
    temp_cu = workdir / "temp.cu"
    temp_ptx = workdir / "temp.ptx"
    temp_cu.write_text(framework + "\n" + kernels + "\n\n")
    cmd = [args.nvcc, "-ptx", str(temp_cu), "-o", str(temp_ptx),
           "-arch", "compute_70", "-code", "sm_70", "--use_fast_math", "-dopt=on"]
    subprocess.run(cmd, check=True)
    nvcc_version = subprocess.run([args.nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]

    lines = temp_ptx.read_text().splitlines(keepends=True)
    parsed = bp.parse_ptx_code(lines)
    buf = io.StringIO()
    with redirect_stdout(buf):
        modified = bp.inject_fuel_and_runtime_sig(parsed, to_ignore)
    inject_log = buf.getvalue()
    ptx = "".join(modified)
    n_patched = ptx.count("0xdeadbeefdeadbeef")
    ptx = ptx.replace("0xdeadbeefdeadbeef", f"0x{args.fuel_limit:016x}")
    out_ptx = workdir / "instrumented.ptx"
    out_ptx.write_text(ptx)
    if args.keep_ptx:
        Path(args.keep_ptx).parent.mkdir(parents=True, exist_ok=True)
        Path(args.keep_ptx).write_text(ptx)

    # Static per-block costs, per kernel, from the injector's own log.
    static = {}
    current = None
    for line in inject_log.splitlines():
        m = re.match(r"kernel: (\S+), #blocks: (\d+), status: (\w+)", line)
        if m:
            current = m.group(1)
            static[current] = {"blocks": int(m.group(2)), "status": m.group(3), "block_fuel": []}
            continue
        m = re.match(r"\s*block (\d+): fuel_usage: (\d+)", line)
        if m and current:
            static[current]["block_fuel"].append(int(m.group(2)))
    for k in static.values():
        k["static_total"] = sum(k["block_fuel"])

    # Which PTX opcodes in the metered kernels have no cost at all.
    table = bp.instruction_fuel_cost
    opcodes = {}
    for item in parsed:
        if isinstance(item, dict):
            for block in item["blocks"]:
                for instr in block:
                    op = instr.strip().split()[0] if instr.strip() else ""
                    if op and not op.startswith(("//", ".", "$", "{", "}")):
                        opcodes[op] = opcodes.get(op, 0) + 1
    uncosted = {op: n for op, n in opcodes.items() if op not in table}
    costed = {op: {"count": n, "cost": table[op]} for op, n in opcodes.items() if op in table}

    try:
        head = subprocess.run(["git", "-C", str(monorepo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        head = ""
    return {
        "ptx_path": out_ptx,
        "static": static,
        "uncosted_opcodes": dict(sorted(uncosted.items(), key=lambda kv: -kv[1])),
        "costed_opcodes": dict(sorted(costed.items(), key=lambda kv: -kv[1]["count"])),
        "patched_limit_sites": n_patched,
        "nvcc_version": nvcc_version,
        "nvcc_cmd": " ".join(cmd),
        "build_ptx_sha256": hashlib.sha256(build_ptx_path.read_bytes()).hexdigest(),
        "framework_cu_sha256": hashlib.sha256(framework_cu.read_bytes()).hexdigest(),
        "kernels_cu_sha256": hashlib.sha256(KERNELS_CU.read_bytes()).hexdigest(),
        "instrumented_ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        "monorepo_head": head,
        "inject_log": inject_log,
    }


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def load_rows(path: str, n: int) -> np.ndarray:
    if path.endswith(".npy"):
        rows = np.load(path, mmap_mode="r")
        if rows.shape[0] < n:
            raise SystemExit(f"{path} has {rows.shape[0]} rows; need {n}")
        return np.ascontiguousarray(rows[:n], dtype=np.float32)
    import h5py
    with h5py.File(path, "r") as f:
        ds = f["train"]
        if ds.shape[0] < n:
            raise SystemExit(f"{path} has {ds.shape[0]} rows; need {n}")
        return np.ascontiguousarray(ds[:n], dtype=np.float32)


# ---------------------------------------------------------------------------
# The build, metered
# ---------------------------------------------------------------------------

class Meter:
    def __init__(self, module):
        import cupy
        self.cupy = cupy
        self.fuel = cupy.ndarray((1,), dtype=cupy.uint64, memptr=module.get_global("gbl_FUELUSAGE"))
        self.err = cupy.ndarray((1,), dtype=cupy.uint64, memptr=module.get_global("gbl_ERRORSTAT"))
        self.phases = []

    def zero(self):
        self.cupy.cuda.runtime.deviceSynchronize()
        self.fuel[0] = 0
        self.err[0] = 0
        self.cupy.cuda.runtime.deviceSynchronize()

    def read(self) -> int:
        self.cupy.cuda.runtime.deviceSynchronize()
        return int(self.fuel[0])

    def error(self) -> int:
        self.cupy.cuda.runtime.deviceSynchronize()
        return int(self.err[0])


def grid(n: int, block: int) -> int:
    return (n + block - 1) // block


def main(argv=None) -> None:
    args = parse_args(argv)
    import cupy

    gpu = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    workdir = Path(tempfile.mkdtemp(prefix="index_build_fuel_"))
    t0 = time.perf_counter()
    inst = instrument(args, workdir)
    compile_s = time.perf_counter() - t0
    print(f"instrumented PTX in {compile_s:.1f}s; metered kernels: "
          f"{[k for k, v in inst['static'].items() if v['status'] == 'PROCESSING']}", flush=True)

    t0 = time.perf_counter()
    host = load_rows(args.real_path, args.n)
    load_s = time.perf_counter() - t0
    n, dims = int(host.shape[0]), int(host.shape[1])
    if dims > 128:
        raise SystemExit(f"dims {dims} > 128; the assign kernel's shared tiles are sized for <= 128")
    print(f"loaded {host.shape} in {load_s:.1f}s", flush=True)

    mod = cupy.RawModule(path=str(inst["ptx_path"]))
    k = {name: mod.get_function(name) for name in
         ["fuel_calibrate", "ivf_pick_centroids", "ivf_assign", "ivf_count", "ivf_scan",
          "ivf_scatter", "kmeans_update", "ivf_gather"]}
    meter = Meter(mod)

    # -- calibration: how many times does the injector add a thread's fuel? --
    calib = {}
    static_calib = inst["static"]["fuel_calibrate"]["static_total"]
    for threads in (1, 1024, 65536):
        cin = cupy.arange(threads, dtype=cupy.float32)
        cout = cupy.zeros(threads, dtype=cupy.float32)
        meter.zero()
        k["fuel_calibrate"]((grid(threads, 256),), (min(threads, 256),), (cin, cout))
        raw = meter.read()
        calib[str(threads)] = {"raw": raw, "static_block_total": static_calib,
                               "raw_per_thread_over_static": (raw / threads / static_calib) if static_calib else None}
    print(f"calibration: {calib}", flush=True)

    # -- device buffers --
    cupy.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    d_db = cupy.asarray(host)
    cupy.cuda.runtime.deviceSynchronize()
    h2d_s = time.perf_counter() - t0
    n_cent = int(args.n_lists)
    n_train = max(n_cent, int(n * args.trainset_fraction))
    d_cent = cupy.zeros(n_cent * dims, dtype=cupy.float32)
    d_assign = cupy.zeros(n, dtype=cupy.uint32)
    d_counts = cupy.zeros(n_cent, dtype=cupy.uint32)
    d_offsets = cupy.zeros(n_cent + 1, dtype=cupy.uint32)
    d_cursor = cupy.zeros(n_cent, dtype=cupy.uint32)
    d_ids = cupy.zeros(n, dtype=cupy.uint32)
    d_gathered = cupy.zeros(n * dims, dtype=cupy.float32)
    i32 = np.int32

    def csr(n_rows: int):
        d_counts.fill(0)
        d_cursor.fill(0)
        k["ivf_count"]((grid(n_rows, THREADS),), (THREADS,), (d_assign, i32(n_rows), d_counts))
        k["ivf_scan"]((1,), (1,), (d_counts, i32(n_cent), d_offsets))
        k["ivf_scatter"]((grid(n_rows, THREADS),), (THREADS,),
                         (d_assign, i32(n_rows), d_offsets, d_cursor, d_ids))

    def assign(n_rows: int):
        k["ivf_assign"]((grid(n_rows, A_DB_TILE),), (THREADS,),
                        (d_db, i32(n_rows), d_cent, i32(n_cent), i32(dims), d_assign))

    phases = []

    def phase(name: str, fn, **extra):
        meter.zero()
        cupy.cuda.runtime.deviceSynchronize()
        t = time.perf_counter()
        fn()
        cupy.cuda.runtime.deviceSynchronize()
        secs = time.perf_counter() - t
        raw = meter.read()
        rec = {"phase": name, "raw_fuel": raw, "reported_fuel": raw // GPU_FUEL_SCALE,
               "seconds": secs, "errorstat": meter.error(), **extra}
        phases.append(rec)
        print(f"  {name:>22}: raw {raw:>16,d}  reported {raw // GPU_FUEL_SCALE:>15,d}  {secs:8.3f}s", flush=True)
        return rec

    print(f"build: n={n} dims={dims} n_lists={n_cent} n_train={n_train} n_iters={args.n_iters}", flush=True)
    build_t0 = time.perf_counter()
    phase("pick_centroids", lambda: k["ivf_pick_centroids"](
        (grid(n_cent * dims, THREADS),), (THREADS,), (d_db, i32(n_train), i32(dims), i32(n_cent), d_cent)))
    for it in range(args.n_iters):
        phase(f"kmeans_assign[{it}]", lambda: assign(n_train), iteration=it, rows=n_train)
        phase(f"kmeans_csr[{it}]", lambda: csr(n_train), iteration=it, rows=n_train)
        phase(f"kmeans_update[{it}]", lambda: k["kmeans_update"](
            (n_cent,), (UPD_THREADS,), (d_db, i32(dims), d_offsets, d_ids, i32(n_cent), d_cent)),
            iteration=it, rows=n_train)
    phase("final_assign", lambda: assign(n), rows=n)
    phase("final_csr", lambda: csr(n), rows=n)
    phase("gather", lambda: k["ivf_gather"](
        (min(n, 65535 * 4),), (UPD_THREADS,), (d_db, i32(dims), d_ids, i32(n), d_gathered)), rows=n)
    cupy.cuda.runtime.deviceSynchronize()
    build_s = time.perf_counter() - build_t0

    # -- sanity: lists cover n, assignments are the true nearest centroid on a sample --
    counts = cupy.asnumpy(d_counts)
    offsets = cupy.asnumpy(d_offsets)
    sample = cupy.arange(0, n, max(1, n // 512))[:512]
    dbs = d_db.reshape(n, dims)[sample]
    cen = d_cent.reshape(n_cent, dims)
    d2 = (dbs[:, None, :] - cen[None, :, :]) ** 2
    ref = cupy.asnumpy(cupy.argmin(d2.sum(axis=2), axis=1))
    got = cupy.asnumpy(d_assign[sample])
    mismatches = int((ref != got).sum())
    sanity = {
        "counts_sum": int(counts.sum()), "n": n, "offsets_last": int(offsets[-1]),
        "empty_lists": int((counts == 0).sum()), "max_list": int(counts.max()), "min_list": int(counts.min()),
        "assign_check_sample": int(sample.size), "assign_check_mismatches": mismatches,
    }
    print(f"sanity: {sanity}", flush=True)
    if sanity["counts_sum"] != n or mismatches > 0:
        print("SANITY FAILURE", flush=True)

    def total(pred):
        sel = [p for p in phases if pred(p)]
        return {"raw_fuel": sum(p["raw_fuel"] for p in sel), "reported_fuel": sum(p["reported_fuel"] for p in sel),
                "seconds": sum(p["seconds"] for p in sel), "phases": len(sel)}

    summary = {
        "total": total(lambda p: True),
        "kmeans": total(lambda p: p["phase"].startswith("kmeans_")),
        "kmeans_assign": total(lambda p: p["phase"].startswith("kmeans_assign")),
        "kmeans_update": total(lambda p: p["phase"].startswith("kmeans_update")),
        "kmeans_csr": total(lambda p: p["phase"].startswith("kmeans_csr")),
        "final": total(lambda p: p["phase"] in ("final_assign", "final_csr", "gather")),
        "final_assign": total(lambda p: p["phase"] == "final_assign"),
    }
    tot = summary["total"]
    summary["fuel_per_second_reported"] = tot["reported_fuel"] / tot["seconds"] if tot["seconds"] else None
    print(f"TOTAL reported fuel {tot['reported_fuel']:,d} in {build_s:.2f}s "
          f"({summary['fuel_per_second_reported']:.3e} reported fuel/s)", flush=True)

    result = {
        "corpus": args.corpus,
        "real_path": args.real_path,
        "shape": {"n": n, "dim": dims, "dtype": "float32"},
        "params": {"n_lists": n_cent, "n_iters": args.n_iters, "trainset_fraction": args.trainset_fraction,
                   "n_train": n_train, "assign_tile": [A_DB_TILE, 32], "threads": THREADS, "update_threads": UPD_THREADS},
        "gpu_fuel_scale": GPU_FUEL_SCALE,
        "fuel_limit_patched": args.fuel_limit,
        "calibration": calib,
        "summary": summary,
        "phases": phases,
        "build_wall_seconds": build_s,
        "sanity": sanity,
        "load_seconds": load_s,
        "h2d_seconds": h2d_s,
        "compile_seconds": compile_s,
        "instrumentation": {key: val for key, val in inst.items() if key not in ("ptx_path", "inject_log")},
        "environment": {"gpu": gpu, "platform": platform.platform(), "python": platform.python_version(),
                        "cupy": cupy.__version__, "numpy": np.__version__},
        "notes": [
            "raw_fuel is gbl_FUELUSAGE as the injected PTX leaves it; reported_fuel is raw // 20, "
            "which is what tig-runtime prints as 'gpu fuel used'.",
            "Fuel is a static instruction count and is data-independent to first order; it is specific "
            "to these kernels, not a property of IVF-Flat in general.",
            "calibration.raw_per_thread_over_static is the factor by which the injector's per-return "
            "atomics multiply a thread's accumulated block cost.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    (out.with_suffix(".inject.log")).write_text(inst["inject_log"])
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
