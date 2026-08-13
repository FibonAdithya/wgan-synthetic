> **AI-generated working note.** Written by Claude during development and kept
> for provenance. It is not the source of truth: where this file disagrees with
> `PROJECT_DOCUMENTATION.md`, the latter wins.

# GPU ANN-Algorithm Benchmark

**Date:** 2026-08-12
**Branch:** `benchmark-algos`
**Status:** Design approved, pending implementation plan

## Problem

The gate for this project is four statistics in `src/eval/ann_difficulty.py` —
LID median, relative contrast, hubness skew, IVF cell-balance Gini. That module
says what it is in its own first paragraph: *"Everything here is computed from
the vectors alone; no index is built."* The four numbers are published
*predictors* of nearest-neighbour difficulty, standing in for the thing the
project actually cares about, which is how a real index behaves.

Nothing in the repo has ever built an index over a synthetic corpus and
measured it. There is no number for how long an index takes to build over
`v2`, what recall it reaches, or how many queries per second it serves —
neither for the synthetic sets nor for real SIFT.

This is a measurement exercise. It produces a table of index build time,
recall and QPS for real SIFT against each trained SIFT variant, measured on
GPU. It does not tune anything, and it does not adjudicate the gate.

## Scope

**In scope.** A benchmark harness that sweeps GPU ANN index against corpus,
traces a recall-vs-QPS curve per cell, and emits JSON, a markdown table and a
self-contained HTML report.

**Out of scope, each deliberately.**

- **Validating the proxy gate.** Whether the four statistics predict these
  measurements is a genuine and interesting question, and this benchmark
  produces the evidence needed to ask it. Asking it is separate work with a
  separate spec. Nothing here should be read as adjudicating a gate.
- **Ranking the variants.** The deliverable is a table, not a verdict about
  which variant is the best stand-in.
- **Tuning index parameters** for any corpus. Build parameters are fixed and
  identical across corpora; see "Sweep shape".
- **CPU indexes.** The roster is GPU-only. HNSW has no GPU equivalent — CAGRA
  is the GPU graph index and stands in that slot.
- **A CI performance gate.** The grid needs a GPU, a full corpus and roughly
  an hour. `make check` stays CPU-only and dataset-free.

## The comparability problem, and what is done about it

This is the correctness issue that decides whether any recall number in the
table means anything, and it is not obvious from either side of the code.

Every SIFT config sets `preprocess.l2_normalize: true` with `center: false`
and `whiten: false`. Generators therefore emit **unit-norm** vectors.
`src/data/dataset.py:invert_preprocess` deliberately does not invert L2
normalization, and says why: normalizing discards each vector's norm, so the
information needed to undo it is gone.

`data/sift_1m.npy` is raw SIFT — non-negative, uint8-valued, with norms in
the hundreds.

So the synthetic corpora live on the unit sphere and the real corpus does
not. Building indexes over both as they sit on disk and comparing the recall
would be measuring the scale difference, not the corpora.

**Decision: every corpus, real included, is L2-normalized before indexing.**
All measurement happens in the trained space. The search metric stays `l2`,
which on the unit sphere is monotone in cosine, so the choice of metric costs
nothing there.

The cost of this decision is real and must be stated in the report rather
than buried: **normalizing real SIFT discards its norm distribution, which is
itself part of SIFT's search difficulty.** The numbers describe normalized
SIFT. They are not SIFT1M figures, and per invariant 3 they were never going
to be comparable with published values anyway.

The rejected alternative was to rescale each synthetic vector by a norm drawn
from the real corpus's empirical norm distribution, putting everything in raw
space instead. It was rejected because no generator ever modelled the norm:
the scale would be invented data dressed as a measurement.

## Corpora

Seven, each 1,000,000 × 128 float32, L2-normalized.

| Corpus | Source |
|---|---|
| `real` | `data/sift_1m.npy` |
| `v0`, `v1`, `v1_5`, `v2`, `v3`, `v4` | Drawn from each variant's `best_generator.pt`, rebuilt against its `run_config.yaml` |

Note that the ladder's six rungs are `v0`, `v1`, `v1_5`, `v2`, `v3`, `v4` —
`v1_5` is a rung, not a typo, and the names are not a numeric sequence.

Generator loading reuses `src/eval/compare_variants.py`'s loader rather than
growing a second copy. A checkpoint is only loadable beside its
`run_config.yaml` (invariant 4), and that module already enforces it.

Which variants exist is read from `configs/eval/sift.yaml`. **That manifest is
currently stale**: it stops at `v2`, while `docs/datasets/sift.md` documents a
ladder through `v3` (`runs/sift_gan_v3`, `runs/x100k_structured`) and `v4`.
Extending the manifest to name the runs that exist is a manifest edit, not a
change to what any variant number means, so it is inside scope. If the `v3` or
`v4` run directories are not on the box, those corpora drop out and the table
has five rows.

`data/sift_base.npy` is absent from the box; `data/sift_1m.npy` was verified
to hold bit-identical data and is what `real` reads.

## Queries and ground truth

**10,000 queries per corpus.**

- `real` uses the real query set — the `test` key of the cached
  ann-benchmarks HDF5. `src/data/fetch.py` reads only `train` and never writes
  the queries to disk, so this is a direct `h5py` read of the existing cache.
  `h5py` is already a pinned dependency.
- Each synthetic corpus uses a **fresh draw from its own generator** under a
  seed disjoint from the corpus draw.

This mirrors the relationship SIFT's query set has with its base set — same
distribution, different sample — so each corpus is searched the way it would
actually be used. The rejected alternative, holding out queries from the
corpus itself, was consistent with the existing self-queried-subsample
convention but discards the real query set for no gain.

Queries are L2-normalized identically to their corpus.

**Ground truth** is exact k=10, computed by GPU brute force once per corpus
and cached. The exact index is also a reported benchmark row, so this
computation is not overhead: it is the recall-1.0 ceiling that makes every
other QPS number interpretable.

## Roster

`cuvs` (NVIDIA cuVS / RAPIDS). One library, one API, one device, so every
number in the table is measured the same way.

| Index | Fixed build params | Swept knob |
|---|---|---|
| Flat (exact) | — | none; single point at recall 1.0 |
| IVF-Flat | `n_lists=4096` | `n_probes` ∈ 1…256 |
| IVF-PQ | `n_lists=4096, pq_dim=64, pq_bits=8` | `n_probes` ∈ 1…256 |
| CAGRA | `graph_degree=64, intermediate_graph_degree=128` | `itopk_size` ∈ 32…512 |

`n_lists=4096` is roughly 4·√1e6, the conventional IVF choice at this scale.

**cuVS is not pinned in `requirements.txt`.** It is CUDA-12-only and installs
from NVIDIA's package index; adding it would break the CPU-only install that
CI runs. It is documented as a box-side extra, and the CLI fails at entry with
the install command if it is missing.

## Sweep shape

Build parameters are **fixed and identical across all seven corpora**. Only
the search-time knob is swept, tracing a recall@10-vs-QPS curve per cell.

A single operating point would confound speed with accuracy: an index that
looks twice as fast may simply be returning worse results. The curve is the
only form in which "faster" is a meaningful claim, and it is what
ann-benchmarks reports.

The headline number per cell is **QPS at recall@10 = 0.90**, interpolated on
the curve. When a corpus's curve never reaches 0.90 under the swept range,
the cell reports `null` — not the nearest point, and not an extrapolation.
Fabricating a number there would hide the most interesting possible result,
which is a corpus an index cannot reach target recall on at all.

The Flat index has no swept knob and sits at recall 1.0 by construction, so it
has no interpolated headline. Its row reports its single measured QPS, labelled
as the exact-search ceiling rather than as a value at the 0.90 target.

## Measurement protocol

**QPS.** All 10,000 queries issued in a single batched call. GPU indexes are
throughput devices; a batch-1 measurement would time launch latency rather
than the index. The batch size is recorded in the output rather than left
implicit. Each point runs 5 repeats reporting **min, median and p95** — min
is the machine's ceiling, median the typical case, p95 what a deadline should
be budgeted against. This matches the reporting shape of
`src/sample/benchmark.py`.

**Every timed region is fenced with a CUDA stream synchronization.** cuVS
calls are asynchronous; an unfenced region times the launch queue rather than
the work, and the whole benchmark becomes fiction. This is the single most
important correctness property in the harness, and it is the same one the
generation-timing spec identifies.

**Build time** is reported as train and add phases separately, since they
scale differently in N and a partitioning index pays them in very different
proportions to a graph index.

**Per cell** the harness also records index size in bytes and peak VRAM. Both
are free at that point in the code, and how much device memory a corpus needs
is the sibling question to how long it takes — a 1M × 128 float32 corpus is
512 MB before any index is built over it.

Because the whole roster runs on GPU, the box's 10.24-core cgroup quota is
not on the critical path and no CPU thread pinning is required.

## Architecture

New package `src/eval/ann_benchmark/`, following the `src/eval/eda/`
precedent — the one other place in this repo where a job outgrew a flat
module. Each unit is testable without a GPU, without data, and without cuVS
installed.

| Module | Responsibility | Depends on |
|---|---|---|
| `corpora.py` | Materialize the seven corpora and their query sets: load real from `.npy` and the HDF5 `test` key, draw synthetic from each manifest entry's checkpoint, L2-normalize everything. | `compare_variants` loader, `dataset.apply_preprocess` |
| `groundtruth.py` | Exact k=10 per corpus via GPU brute force, with a numpy fallback used by tests. | cuVS (lazy import) |
| `indexes.py` | One adapter per algorithm: `build(vectors)`, `search(handle, queries, k, param)`, `sweep_params()`, `describe()`. cuVS is imported inside methods so the module imports on a CPU-only box. | cuVS (lazy import) |
| `metrics.py` | recall@k, QPS, interpolation to a target recall. Pure numpy, no device and no I/O. | — |
| `runner.py` | The grid loop, all timing and all stream fencing. Reaches indexes only through the adapter interface. | the above |
| `report.py` | JSON, markdown headline table, self-contained HTML with recall-vs-QPS curves. | plotly |
| `cli.py` | `python -m src.eval.ann_benchmark` | the above |

The adapter interface in `indexes.py` is the boundary that matters: `runner.py`
never names a cuVS type, so the grid loop is drivable end-to-end by a fake
adapter in tests.

### Data flow

`corpora.py` writes each corpus, query set and ground-truth array to a work
directory once and reuses them on a later run. Drawing seven 1.01M-vector sets
and running seven exact-kNN passes is the expensive and fully deterministic
half of the job; caching it means a crash inside the grid does not re-pay it.

`runner.py` consumes those arrays and emits one record per (corpus, index,
search-parameter) cell. `report.py` consumes records and writes files. Neither
knows how corpora were produced.

### Error handling

- **Missing run directory or checkpoint** aborts before any GPU work, naming
  the path and the command that would produce it — the behaviour
  `compare_variants` already has. `--allow-missing` proceeds on a deliberately
  partial roster.
- **Missing cuVS** fails at CLI entry with the install command, not at cell 40
  of the grid.
- **A cell that OOMs or throws** is recorded as a failed cell with its
  exception and the grid continues. One bad IVF-PQ cell must not cost six
  corpora of completed work.
- **Records append to the JSON incrementally**, so a killed job keeps what it
  had.

### Testing

All CPU-only, so `make check` stays green with no GPU and no dataset.

- `metrics.py` against hand-computed ground truth: recall@k, QPS arithmetic,
  and interpolation — including the case where a curve never reaches the
  target recall and must return `None` rather than a fabricated value.
- `corpora.py`: normalization is applied to real and synthetic alike, and the
  query draw is disjoint from the corpus draw.
- `runner.py`: driven end-to-end by a fake adapter, covering record shape,
  failed-cell handling and resume.
- `report.py`: JSON and markdown shape from fixed records.
- The cuVS-touching code stays a thin edge, skipped when the import is
  unavailable.

## Execution

One `gpuq` job on `tig-gpu`, with data staged per-job into a fresh worktree.
Artifacts are the JSON, markdown and HTML outputs; `runs/` is not declared as
an artifact.

Wall clock is dominated by CAGRA graph construction at roughly 2–3 minutes per
corpus. Budget 40–60 minutes for the full grid.

## Deliverable

`docs/results/ann-gpu-benchmark/`, matching the layout established by
`docs/results/generation-timing/`:

    README.md              what was run, on what, and how to read it
    ann_benchmark.json     every cell, unaggregated
    ann_benchmark.md       headline table
    report.html            recall-vs-QPS curves, self-contained
    gpuq_job_spec.json     the job that produced it

The README states plainly that all figures are measured on L2-normalized
corpora and are not comparable with published SIFT1M results.
