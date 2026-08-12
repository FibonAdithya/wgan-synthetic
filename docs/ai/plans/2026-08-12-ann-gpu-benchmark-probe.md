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
| `v3` | not in manifest (`runs/sift_gan_v3`) | `/workspace/keep/v34-sift1m/v3` | **OK** |
| `v4` | not in manifest (`runs/x100k_structured`) | `/workspace/keep/v34-sift1m/v4` | **OK** |

> **Correction (2026-08-12).** An earlier revision of this document claimed
> `v3` and `v4` were "absent everywhere on the box" and fixed the table at five
> rows. **That was wrong.** The search behind it was
> `find / -name best_generator.pt | head -40` — the `head -40` truncated the
> results and cut off `/workspace/keep/v34-sift1m/`, which holds both. A second
> copy of each also exists under `/workspace/keep/v3-structured/` named
> `sift_gan_v3-best_generator.pt` and `x100k_structured-best_generator.pt`,
> which an exact-name search misses on top of that. Use the
> `keep/v34-sift1m/{v3,v4}` copies: they carry `best_generator.pt`,
> `run_config.yaml` and `run_metadata.json` in the canonical layout.
>
> **The ladder is six rungs and the table has seven corpora** — the design's
> full case, not its reduced one.

All four present rungs have both `best_generator.pt` and `run_config.yaml`
beside each other, satisfying invariant 4.

`v2` is **not** under the same root as `v0`/`v1`/`v1_5`. A second copy exists at
`/workspace/glyph-root/runs/x100k_sparse_clamp4` but is only 4 K — use the
`keep/wgan-sparse-v2` copy.

**The table has seven rows:** `real`, `v0`, `v1`, `v1_5`, `v2`, `v3`, `v4`.

`configs/sift/v3.yaml` and `configs/sift/v4.yaml` are both git-tracked, so the
only thing missing is two manifest entries — which Task 8 adds, and which the
design already places in scope as a manifest edit rather than a change to what
any variant number means.

All six configs carry the same `data.preprocess` block
(`center: false, whiten: false, l2_normalize: true`), so every rung emits
unit-norm vectors and the comparability decision applies uniformly. Note that
`preprocess` is nested under `data:`, not at the top level of the config — a
top-level lookup returns `None` for *every* rung including `v0` and invites the
false conclusion that `v3`/`v4` differ.

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

## cuVS does not own its dataset — and the failure is silent

`metric="sqeuclidean"` does return squared L2, sorted ascending, matching numpy
to 1e-4 with identical neighbour ids. But only if the caller keeps the device
dataset alive.

cuVS indexes hold a *pointer* into the device buffer passed to `build()`; they
neither copy it nor take a reference. If that buffer is a temporary, cupy frees
it when `build()` returns, the pool hands the block to the next allocation, and
the index reads whatever now lives there. Measured, 2000x128 unit vectors,
5 queries, k=10:

    # dataset passed as a temporary: brute_force.build(cp.asarray(x), ...)
    cuvs  D[0][:4]  [-0.0, 1.475755, 1.4782941, 1.4862771]
    numpy sq [:4]   [1.4757549, 1.4782941, 1.4862771, 1.5096332]
    match: False        ids match: 0.02

    # same code, dataset held in a live local across the call
    cuvs  D[0][:4]  [1.475755, 1.4782941, 1.4862771, 1.5096332]
    match: True         ids match: 1.0

Nothing raises. The distances are finite, correctly shaped and correctly
sorted; they are answers to a different question. In the failing run the query
buffer had landed in the freed dataset block, so query 0 "found itself" at
distance `-0.0` and every other result shifted by one — which in a benchmark
table looks like one corpus scoring suspiciously well rather than like a bug.

**Any `BuiltIndex` must therefore retain the device dataset for the index's
whole lifetime.** This is not in the plan, and it is not reproducible on a
CPU-only box, so no test in `make check` can catch it; the guard is a test
pinning that the reference is retained.

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
    ln -sfn /workspace/keep/v34-sift1m/v3                    $R/runs/sift_gan_v3
    ln -sfn /workspace/keep/v34-sift1m/v4                    $R/runs/x100k_structured
    ln -sfn /workspace/data-cache/sift_1m.npy                $R/data/sift_1m.npy

Verified against the real loader, with `v3`/`v4` appended to the manifest:

    resolve_variants(..., /workspace/annbench-root)
        RESOLVED: ['v0', 'v1', 'v1_5', 'v2', 'v3', 'v4']
        SKIPPED:  (none)

All six rungs carry `best_generator.pt`, `run_config.yaml` *and*
`run_metadata.json`, so `_inversion_blocker` passes and no variant is silently
dropped. This is the check that would otherwise have failed late, inside the
grid, after the expensive corpus draw — and note the failure mode is a *skip*,
not an exception: an unresolvable rung would quietly shrink the table rather
than stop the run.

## Decisions taken from this probe

1. Install `cuvs-cu13` / `cupy-cuda13x`, not the `cu12` variants.
2. Seven corpora: `real`, `v0`, `v1`, `v1_5`, `v2`, `v3`, `v4`. Task 8 adds the
   two manifest entries.
3. Keep the full 1M corpus; the card has ample headroom.
4. Add a discarded warmup search per cell.
5. `BuiltIndex` retains the device dataset. Without it every number in the
   table is silently wrong.
5. Paths for the job: python `/venv/main/bin/python`, real corpus
   `/workspace/data-cache/sift_1m.npy`, query HDF5
   `/workspace/data-cache/sift-128-euclidean.hdf5`, run dirs as tabulated above.
