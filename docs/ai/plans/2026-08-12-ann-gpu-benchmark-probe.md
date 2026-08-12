> **AI-generated working note.** Written by Claude during development and kept
> for provenance. It is not the source of truth: where this file disagrees with
> `PROJECT_DOCUMENTATION.md`, the latter wins.

# GPU ANN Benchmark — box probe

**Date:** 2026-08-12
**Branch:** `benchmark-algos`
**Purpose:** Task 1 of `2026-08-12-ann-gpu-benchmark.md`. Confirm the box, the
corpora and the real cuVS API *before* six modules are written against guessed
signatures.

Everything below was measured, not assumed. Where it contradicts the plan, the
plan is wrong and the plan's tasks are to be adjusted.

## Box

    hostname   73021f80c133   (vast.ai container, unprivileged Docker)
    GPU        NVIDIA GeForce RTX 4060
    VRAM       8188 MiB total (7805 MiB visible to CUDA)
    driver     580.95.05
    torch      2.12.0+cu130  ->  CUDA 13.0
    python     /venv/main/bin/python
    disk       /workspace, 80 G free of 130 G

## Corrections to the plan

| Plan assumed | Reality |
|---|---|
| `pip install cuvs-cu12 cupy-cuda12x` | **`cuvs-cu13 cupy-cuda13x`.** torch is built against CUDA 13.0; cuVS must match, or a single process holding both loads two CUDA runtimes. |
| repo at `~/wgan-synthetic` | `/workspace/checkouts/wgan-synthetic` (git, at `3d11e36`) — but its `data/` is empty and `runs/` holds only `glove/` and `sift/`. |
| `data/sift_1m.npy` | `/workspace/data-cache/sift_1m.npy` |
| `data/cache/*.hdf5` | `/workspace/data-cache/sift-128-euclidean.hdf5` |
| run dirs under one `runs/` | Scattered: see the ladder table below. |

Data staging per job must therefore name `/workspace/data-cache` and the two
`keep/` trees explicitly; a fresh worktree has none of it.

## Installed

    cuvs-cu13      26.08.01
    libcuvs-cu13   26.8.1
    libraft-cu13   26.8.0
    rmm-cu13       26.8.0
    cupy-cuda13x   14.1.1

## Corpora

`/workspace/data-cache/sift_1m.npy` — `(1000000, 128)` float32.

`/workspace/data-cache/sift-128-euclidean.hdf5`:

    train       (1000000, 128)
    test        (10000, 128)      <- the real query set the plan wants
    neighbors   (10000, 100)
    distances   (10000, 100)

The `test` key exists, so the design's query protocol stands unchanged and no
fallback to a held-out slice is needed. The cached `neighbors`/`distances` are
ground truth for *raw* SIFT and are deliberately **not** used: this benchmark
measures L2-normalized corpora and computes its own exact k=10.

## The ladder — which rungs exist

From `configs/eval/sift.yaml`:

| Variant | `run_dir` in manifest | Found at | Status |
|---|---|---|---|
| `v0` | `runs/long_baseline` | `/workspace/keep/wgan-synthetic/long_baseline` | **OK** |
| `v1` | `runs/x100k_ema_only` | `/workspace/keep/wgan-synthetic/x100k_ema_only` | **OK** |
| `v1_5` | `runs/x100k_improved` | `/workspace/keep/wgan-synthetic/x100k_improved` | **OK** |
| `v2` | `runs/x100k_sparse_clamp4` | `/workspace/keep/wgan-sparse-v2/x100k_sparse_clamp4` (31 M) | **OK** |
| `v3` | not in manifest (`runs/sift_gan_v3`) | — | **ABSENT** |
| `v4` | not in manifest (`runs/x100k_structured`) | — | **ABSENT** |

All four present rungs have both `best_generator.pt` and `run_config.yaml`
beside each other, satisfying invariant 4.

`v2` is **not** under the same root as `v0`/`v1`/`v1_5`. A second copy exists at
`/workspace/glyph-root/runs/x100k_sparse_clamp4` but is only 4 K — use the
`keep/wgan-sparse-v2` copy.

An exhaustive `find / -name best_generator.pt` confirms `sift_gan_v3` and
`x100k_structured` are on no path on this box. **The table has five rows:**
`real`, `v0`, `v1`, `v1_5`, `v2` — the design's anticipated five-row case.

## cuVS API, as measured

Signatures (`cuvs 26.08.01`):

    ivf_flat.build   (index_params, dataset, resources=None)
    ivf_flat.search  (search_params, index, queries, k, neighbors=None,
                      distances=None, resources=None, filter=None)
    ivf_pq.build     (index_params, dataset, resources=None)
    ivf_pq.search    (search_params, index, queries, k, neighbors=None,
                      distances=None, resources=None)
    cagra.build      (index_params, dataset, resources=None)
    cagra.search     (search_params, index, queries, k, neighbors=None,
                      distances=None, resources=None, filter=None)
    brute_force.build  (dataset, metric='sqeuclidean', metric_arg=2.0,
                        resources=None)
    brute_force.search (index, queries, k, neighbors=None, distances=None,
                        resources=None, prefilter=None)

Note the argument order: **`search` takes the params first and the index
second**, and `brute_force.build` takes the metric while the others take it on
`IndexParams`.

`IndexParams`/`SearchParams` are Cython classes whose signatures introspect as
`(*args, **kwargs)`, so every name the plan assumes was verified by
construction instead. All six constructed without error:

    ivf_flat.IndexParams(n_lists=4096, metric="sqeuclidean")          OK
    ivf_flat.SearchParams(n_probes=16)                                OK
    ivf_pq.IndexParams(n_lists=4096, pq_dim=64, pq_bits=8,
                       metric="sqeuclidean")                          OK
    ivf_pq.SearchParams(n_probes=16)                                  OK
    cagra.IndexParams(graph_degree=64,
                      intermediate_graph_degree=128,
                      metric="sqeuclidean")                           OK
    cagra.SearchParams(itopk_size=64)                                 OK

The `"sqeuclidean"` metric spelling is confirmed, as is the stream fence:
`cuvs.common.Resources` exposes exactly `get_c_obj` and `sync`, so
`Resources().sync()` is the adapter's `sync()`.

## Capacity — 1M fits, comfortably

Measured at the full 1,000,000 × 128 float32 corpus, resident on device:

    corpus on device                free 7206 MiB / 7805 MiB
    CAGRA build (64/128)      9.5 s  free 6826 MiB
    CAGRA search, 10k queries        free 6792 MiB
    brute-force exact k=10,
      10k queries x 1M        1.3 s  free 7034 MiB

The plan never checked VRAM and an 8 GB card looked like a risk; it is not.
Roughly 87% of the card is still free at the peak measured here, so the
benchmark runs at the canonical N and invariant 3's locked N is preserved.
CAGRA emits an informational `reducing IVF-PQ search max_internal_batch_size
131072 -> 104857 to fit the workspace` during build — it is a workspace
adjustment, not a warning, and the build succeeds.

## Warmup is required, and the plan omits it

Timing the same CAGRA search six times over 1M vectors, each region fenced with
`Resources().sync()` on both sides:

    iter 0    126.2 ms     79 266 QPS
    iter 1     82.2 ms    121 642 QPS
    iter 2     82.2 ms    121 588 QPS
    iter 3     82.2 ms    121 591 QPS
    iter 4     82.2 ms    121 636 QPS
    iter 5     82.2 ms    121 633 QPS

The first timed call is 1.5x the steady state, and the very first cuVS search
in a *process* costs seconds of one-time initialization. The plan's protocol —
5 repeats reporting min, median and p95, with no warmup — would let that cold
call inflate p95 and drag the median, so p95 would describe a one-off
initialization rather than a deadline worth budgeting against.

**Amendment (approved 2026-08-12):** each cell issues one untimed, discarded
warmup search before its timed repeats. `min`/`median`/`p95` then describe
steady-state throughput, which is what the table claims to report.

## Job staging — a single root the manifest already understands

`compare_variants.resolve_variants` joins one root to every manifest entry
(`run_dir = root / variant.run_dir`), but the rungs are spread over two trees:
`v0`/`v1`/`v1_5` under `keep/wgan-synthetic`, `v2` under `keep/wgan-sparse-v2`.
Rather than teach the manifest about per-variant roots — a code change, and one
that would outlive this benchmark — a staging root of symlinks is assembled and
`--root` points at it. The manifest is then correct as written.

    R=/workspace/annbench-root
    mkdir -p $R/runs $R/data
    ln -sfn /workspace/checkouts/wgan-synthetic/configs      $R/configs
    ln -sfn /workspace/checkouts/wgan-synthetic/src          $R/src
    ln -sfn /workspace/keep/wgan-synthetic/long_baseline     $R/runs/long_baseline
    ln -sfn /workspace/keep/wgan-synthetic/x100k_ema_only    $R/runs/x100k_ema_only
    ln -sfn /workspace/keep/wgan-synthetic/x100k_improved    $R/runs/x100k_improved
    ln -sfn /workspace/keep/wgan-sparse-v2/x100k_sparse_clamp4 \
                                                             $R/runs/x100k_sparse_clamp4
    ln -sfn /workspace/data-cache/sift_1m.npy                $R/data/sift_1m.npy

Verified against the real loader:

    load_variants(configs/eval/sift.yaml)  ->  ['v0', 'v1', 'v1_5', 'v2']
    resolve_variants(..., /workspace/annbench-root)
        RESOLVED: ['v0', 'v1', 'v1_5', 'v2']
        SKIPPED:  (none)

All four rungs carry `best_generator.pt`, `run_config.yaml` *and*
`run_metadata.json`, so `_inversion_blocker` passes and no variant is silently
dropped. This is the check that would otherwise have failed late, inside the
grid, after the expensive corpus draw.

## Decisions taken from this probe

1. Install `cuvs-cu13` / `cupy-cuda13x`, not the `cu12` variants.
2. Five corpora: `real`, `v0`, `v1`, `v1_5`, `v2`. `v3`/`v4` are absent and are
   reported as absent rather than substituted.
3. Keep the full 1M corpus; the card has ample headroom.
4. Add a discarded warmup search per cell.
5. Paths for the job: python `/venv/main/bin/python`, real corpus
   `/workspace/data-cache/sift_1m.npy`, query HDF5
   `/workspace/data-cache/sift-128-euclidean.hdf5`, run dirs as tabulated above.
