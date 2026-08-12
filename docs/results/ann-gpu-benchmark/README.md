# GPU ANN benchmark: real SIFT against the variant ladder

Measured 2026-08-12 from commit `159f1ce` on the exclusive `gpuq` GPU lane
(job `wgan-synthetic-20260812T140307Z-46a363`, exit 0, 8 minutes wall clock).
The machine had one NVIDIA GeForce RTX 4060 (8188 MiB, driver 580.95.05),
PyTorch 2.13.0+cu130, CUDA 13.0 and cuVS 26.08.01.

This is a measurement exercise. It reports index build time, recall@10 and QPS
for real SIFT and each trained SIFT variant. **It does not adjudicate the
ANN-difficulty gate, and it does not rank the variants.** Whether the four gate
statistics predict these measurements is a real question; this benchmark
produces the evidence for asking it, and asking it is separate work.

## Read this first: every corpus was L2-normalized

**All seven corpora, the real one included, were L2-normalized before
indexing, and all figures below describe normalized SIFT.** They are not
SIFT1M figures and are not comparable with published SIFT1M results.

This is not incidental. Every SIFT config sets `preprocess.l2_normalize: true`,
so generators emit unit-norm vectors, and `dataset.invert_preprocess`
deliberately does not invert L2 normalization — normalizing discards each
vector's norm, so the information needed to undo it is gone. Meanwhile
`data/sift_1m.npy` is raw SIFT with norms in the hundreds. Indexing both as
they sit on disk would have measured the scale difference, not the corpora.

The cost is real and is stated rather than buried: **normalizing real SIFT
discards its norm distribution, which is itself part of SIFT's search
difficulty.** The rejected alternative — rescaling synthetic vectors by norms
drawn from the real corpus — was rejected because no generator ever modelled
the norm, so the scale would have been invented data dressed as a measurement.

Per invariant 3, these figures were never going to be comparable with published
values anyway.

## What was run

Seven corpora, each 1,000,000 x 128 float32, unit-norm: `real` plus the full
six-rung ladder `v0`, `v1`, `v1_5`, `v2`, `v3`, `v4`. **No rung was skipped.**
`v3` and `v4` required two new entries in `configs/eval/sift.yaml`, which had
stopped at `v2`; naming runs that exist is a manifest edit and does not change
what any variant number means.

Queries are 10,000 per corpus. `real` uses SIFT's own query set (the `test` key
of the cached ann-benchmarks HDF5); each synthetic corpus uses a fresh draw
from its own generator under a seed salted separately from its corpus draw, so
queries are never members of the index. Ground truth is exact k=10 by GPU brute
force, computed per corpus.

Build parameters are fixed and identical across all seven corpora; only the
search-time knob is swept.

| Index | Fixed build params | Swept knob |
|---|---|---|
| Flat (exact) | — | none; single point at recall 1.0 |
| IVF-Flat | `n_lists=4096` | `n_probes` ∈ 1…256 |
| IVF-PQ | `n_lists=4096, pq_dim=64, pq_bits=8` | `n_probes` ∈ 1…256 |
| CAGRA | `graph_degree=64, intermediate_graph_degree=128` | `itopk_size` ∈ 32…512 |

Search metric is squared L2 throughout, which on the unit sphere is monotone in
cosine. Each cell issues one untimed, discarded warmup search and then five
timed repeats; every timed region is fenced with a CUDA stream synchronization
on both sides. All 10,000 queries are issued in one batched call.

## How to read the files

`ann_benchmark.json` is every cell, unaggregated, plus the environment block.
`ann_benchmark.md` is the headline table. `report.html` is self-contained and
carries the recall-vs-QPS curve per cell — the curve, not the headline, is
where "faster" becomes a meaningful claim. `gpuq_job_spec.json` pins the
command, commit, lane, timeout and exit status.

The headline number is QPS at recall@10 = 0.90, interpolated on the curve. A
cell whose curve never reaches 0.90 would report null rather than the nearest
point or an extrapolation. **No cell reported null in this run**, and no cell
failed: 28 build records and 168 search records, all successful.

The Flat row is the exact-search ceiling at recall 1.0 by construction, not an
interpolated value at the 0.90 target, and is labelled as such.

## Correctness checks

Run before reading anything into the numbers:

- `flat` reports recall exactly **1.000 on all seven corpora**. Had it not,
  ground truth and search would be in different distance spaces and every
  figure here would be wrong.
- Recall rises monotonically with the swept knob in **21 of 21** sweeps.
- **Zero failed cells**, so the table is complete rather than filtered.

## What was measured

QPS at recall@10 = 0.90, as a ratio to `real` on the same index:

| Index | v0 | v1 | v1_5 | v2 | v3 | v4 |
|---|---:|---:|---:|---:|---:|---:|
| IVF-Flat | 0.93 | 0.93 | 0.93 | **1.50** | **2.58** | 1.11 |
| IVF-PQ | 1.02 | 1.00 | 1.02 | **1.40** | **2.07** | 1.05 |
| CAGRA | 0.92 | 0.92 | 0.90 | 1.01 | 1.04 | 0.96 |

The clearest signal is that **the divergence is specific to the partitioning
indexes.** Under CAGRA, a graph index, all six synthetic corpora sit within
0.90x–1.04x of real. Under IVF-Flat and IVF-PQ, `v2` and especially `v3` are
substantially *cheaper* to search than real SIFT at the same recall — IVF-Flat
serves `v3` at 159,160 QPS against real's 61,695.

`v3` also reaches the lowest peak recall of any corpus under both partitioning
indexes (0.958 IVF-Flat, 0.919 IVF-PQ, against real's 0.971 and 0.939), so it
is both faster and less accurate there: the shape of a corpus whose cell
occupancy differs from real's, where a probe budget buys more of the true
neighbours sooner.

`v0`, `v1` and `v1_5` cluster tightly at 0.92–1.02x of real across all three
approximate indexes. `v4` sits within 1.05–1.11x on the partitioning indexes.

What this does and does not say: an index finding a corpus easier at matched
recall is a measured difference in search behaviour between that corpus and
real SIFT. It is **not** a gate verdict, **not** a statement that any variant
is a better or worse stand-in overall, and it is measured on normalized
corpora. One benchmark on one card is also one sample; nothing here is a
seed-noise study.

## Cost

| Index | Build (train + add) | Index size | Peak VRAM delta |
|---|---|---:|---:|
| Flat (exact) | 0.00 s + 0.09 s | 488.3 MB | 4 MB |
| IVF-Flat | 1.44–1.65 s | 488.3 MB | 690–1316 MB |
| IVF-PQ | 3.28–3.53 s | 61.0 MB | 94–98 MB |
| CAGRA | 6.46–8.54 s | 732.4 MB | 242–248 MB |

Build time is reported as train and add phases separately because they scale
differently in N, and a partitioning index pays them in very different
proportions to a graph index.

IVF-PQ is the only index that compresses: 61.0 MB against the corpus's own
488.3 MB, an 8x reduction, at the cost of the lowest peak recall of the three
approximate indexes on every corpus.

**Peak VRAM figures are card-wide deltas, not per-index allocations.** They are
sampled around the build and include whatever else the process had resident, so
treat them as an order-of-magnitude guide rather than an allocation profile.
The Flat row's 4 MB reflects that brute force builds almost nothing beyond
referencing the corpus already on the device.

The full grid took 8 minutes, well under the 40–60 minutes budgeted: CAGRA
graph construction over 1M vectors takes 6–9 seconds on this card, not the 2–3
minutes assumed when the work was planned.

## Reproducing

The run directories are spread over two trees on the box, while
`compare_variants` resolves every rung against a single `--root`. A staging
directory of symlinks reconciles that without changing the manifest format;
the recipe is in
`docs/superpowers/plans/2026-08-12-ann-gpu-benchmark-probe.md`, together with
the measured cuVS API surface and the box paths.

cuVS is **not** in `requirements.txt`: it is CUDA-only and installs from
NVIDIA's package index, so pinning it would break the CPU-only install CI runs.
It is a box-side extra (`cuvs-cu13`, `cupy-cuda13x`), and the CLI fails at
entry with the install command if it is absent. `make check` stays CPU-only,
dataset-free and cuVS-free.
