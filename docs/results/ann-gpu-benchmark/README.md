# GPU ANN benchmark: real SIFT against the variant ladder

Measured 2026-08-13 from commit `7c00b45` on the exclusive `gpuq` GPU lane
(job `wgan-synthetic-20260813T080256Z-3b6ec3`, exit 0). The machine had one
NVIDIA GeForce RTX 4060 (8188 MiB, driver 580.95.05), PyTorch 2.13.0+cu130,
CUDA 13.0 and cuVS 26.08.01.

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

### How recall is scored, and why it matters here

Recall is computed from **exact squared-L2 distances recomputed from each
search's returned neighbour ids**, on both the found and the ground-truth side,
using identical arithmetic.

This is not a detail. `ivf_pq.search` returns distances computed from the
quantized PQ codes, not distances to the stored vectors; scoring recall on
those inflates it, by a margin that grows with the probe budget (measured on
this box at +0.006 at 1 probe, +0.031 at 8, +0.137 at 64). Every IVF-PQ
number below would be optimistic without the recomputation, and the null
results in the next section would have been hidden behind it.

Recall remains **distance-based, never id-matching**: cuVS and numpy break ties
between equidistant neighbours differently, and real SIFT contains duplicate
vectors, so an id-equality test would depress recall for no real reason.

## Correctness checks

Run before reading anything into the numbers:

- `flat` reports recall exactly **1.000 on all seven corpora**. Had it not,
  ground truth and search would be in different distance spaces and every
  figure here would be wrong. This check is what caught a scoring regression
  during development, when the two sides of the comparison were briefly
  computed by different float arithmetic.
- Recall rises monotonically with the swept knob in **21 of 21** sweeps.
- **Zero failed cells**: 28 build records and 168 search records, all
  successful.

## What was measured

### IVF-PQ cannot reach the target recall on five of seven corpora

At `pq_dim=64, pq_bits=8` and `n_probes` up to 256, IVF-PQ never reaches
recall@10 = 0.90 on `real`, `v0`, `v1`, `v1_5` or `v4`. Those cells report
**null**, which is a result rather than a gap:

| Corpus | IVF-PQ @ recall 0.90 | Peak recall reached |
|---|---|---:|
| real | — never reached | 0.881 |
| v0 | — never reached | 0.877 |
| v1 | — never reached | 0.879 |
| v1_5 | — never reached | 0.879 |
| v4 | — never reached | 0.894 |
| **v2** | **42,066 QPS** | 0.906 |
| **v3** | **129,842 QPS** | 0.916 |

`v2` and `v3` are the only corpora on which this quantized index clears 0.90 at
all. The separation is qualitative, not a ratio.

### IVF-Flat: a genuine matched-recall comparison

Both endpoints bracket 0.90, so these are interpolated at the target and
directly comparable. QPS at recall@10 = 0.90, and as a ratio to `real`:

| Corpus | QPS @ 0.90 | Ratio to real |
|---|---:|---:|
| real | 73,063 | 1.00 |
| v0 | 71,586 | 0.98 |
| v1 | 74,450 | 1.02 |
| v1_5 | 73,744 | 1.01 |
| v2 | 128,793 | **1.76** |
| v3 | 224,925 | **3.08** |
| v4 | 87,644 | 1.20 |

### CAGRA: not a matched-recall comparison, and labelled as such

CAGRA's swept knob cannot reach 0.90. `itopk_size` must be at least k and a
multiple of 32, so 32 is its floor, and every corpus already exceeds 0.90 there.
Each CAGRA cell therefore reports its fastest measured point at **the recall
that point was actually measured at** — marked `floor` in the table — rather
than an interpolated value at the target or an extrapolation.

| Corpus | QPS (floor) | at recall |
|---|---:|---:|
| real | 250,469 | 0.963 |
| v0 | 231,341 | 0.966 |
| v1 | 235,762 | 0.963 |
| v1_5 | 237,524 | 0.963 |
| v2 | 245,221 | 0.997 |
| v3 | 269,465 | 0.999 |
| v4 | 241,336 | 0.989 |

**These QPS figures are not comparable across corpora**, because each is
measured at a different recall. What is comparable is the recall itself at a
fixed search cost: at `itopk_size=32`, CAGRA reaches 0.997 on `v2` and 0.999 on
`v3` against 0.963 on `real`. The graph index saturates on those two corpora at
a search budget where real SIFT has not.

### The pattern across all three approximate indexes

`v2` and `v3` separate from real under every approximate index, and the
direction is consistent: at matched recall they are cheaper to search
(IVF-Flat, 1.76x and 3.08x), at matched cost they reach higher recall (CAGRA,
0.997/0.999 against 0.963), and they are the only corpora a quantized index can
push to 0.90 at all (IVF-PQ). `v0`, `v1` and `v1_5` track real closely on every
index — within noise on IVF-Flat, within 0.003 recall on CAGRA, and failing to
reach 0.90 on IVF-PQ exactly as real does. `v4` sits between the two groups: it
separates from real on the partitioning indexes (1.20x on IVF-Flat, seven times
the noise floor) but, like real, never reaches 0.90 under IVF-PQ.

What this does and does not say: an index finding a corpus easier at matched
recall, or reaching higher recall at matched cost, is a measured difference in
search behaviour between that corpus and real SIFT. It is **not** a gate
verdict, **not** a statement that any variant is a better or worse stand-in
overall, and it is measured on normalized corpora.

### Which of these differences survive the noise floor

The grid was run a second time, identically — same commit, same parameters,
same cached corpora, so the only thing re-measured is the search itself (job
`wgan-synthetic-20260813T084709Z-30d589`). `ann_benchmark_repeat.json` holds
that run in full. Cell by cell against the published run:

| Quantity | Run-to-run movement |
|---|---|
| `flat` recall | **0.000** — bit-identical, as exact search must be |
| recall, any approximate cell | ≤ 0.005 (mean 0.001) |
| QPS at a fixed operating point | ≤ 4.6% |
| exact-search ceiling QPS | ≤ 0.4% |
| ratio-to-real @ 0.90 (IVF-Flat) | ≤ 0.074 |

Applying that floor to the results above:

- **The IVF-PQ pattern is exactly reproducible.** All seven null/non-null
  verdicts agree between runs, and peak recalls move by at most 0.001. `real`,
  `v0`, `v1`, `v1_5` and `v4` fail to reach 0.90 in both runs; `v2` and `v3`
  reach it in both. This is the most robust finding in the table.
- **`v2`, `v3` and `v4` separate from real for real.** Their IVF-Flat ratios
  moved 1.76→1.70, 3.08→3.00 and 1.20→1.17 between runs — movement of 0.03–0.07
  against separations of 0.17–2.00. The separations are 7–27x the noise.
- **`v0`, `v1` and `v1_5` are *not* distinguishable from real, or from each
  other.** Their ratios moved 0.98→0.98, 1.02→1.00 and 1.01→0.98. The spread
  within that group is the same size as the run-to-run movement, so the
  earlier-looking "v1 is 1.02x, v0 is 0.98x" ordering does not survive a second
  sample. Read those three as: indistinguishable from real under IVF-Flat.

Two samples bound run-to-run variation on one card; they do not bound
generator-seed variation, since both runs reuse the same cached corpora.
A seed-noise study remains separate work.

## Cost

| Index | Train (s) | Add (s) | Index size (est.) | Peak VRAM delta |
|---|---:|---:|---:|---:|
| Flat (exact) | 0.00 | 0.09–0.11 | 512.0 MB | 4–496 MB |
| IVF-Flat | 1.43–1.59 | 0.00 | 512.0 MB | 692–1316 MB |
| IVF-PQ | 3.18–3.69 | 0.00 | 64.0 MB | 96–102 MB |
| CAGRA | 7.02–7.57 | 0.00 | 768.0 MB | 242–246 MB |

Sizes are decimal MB (10^6 bytes); VRAM figures are MiB as reported by the
driver. Build time is reported as train and add phases separately because they
scale differently in N, and a partitioning index pays them in very different
proportions to a graph index.

**Index size is an analytic estimate, not a measured allocation.** It is
computed from the corpus and index geometry and omits the IVF coarse centroids
(4096 x 128) and the PQ codebooks, so IVF-PQ's 64.0 MB understates its true
footprint. Treat the 8x gap against the corpus as indicative, not measured.

**Peak VRAM figures are card-wide deltas, not per-index allocations.** They are
sampled around the build and include whatever else the process had resident, so
they are an order-of-magnitude guide. The exact index's wide 4–496 MB range
reflects that: brute force allocates almost nothing of its own, so the figure
mostly tracks what else happened to be resident at sample time.

The exact-search ceiling is 7,948–7,996 QPS across the seven corpora — the
price of perfect recall, and roughly 9x slower than IVF-Flat at 0.90 and 30x
slower than CAGRA at its floor.

## How to read the files

`ann_benchmark.json` is every cell, unaggregated, plus the environment block.
`ann_benchmark_repeat.json` is the identical second run used for the noise
floor above; it is evidence for that section and is not otherwise part of the
result. `ann_benchmark.md` is the headline table. `report.html` is self-contained and
carries the recall-vs-QPS curve per cell — the curve, not the headline, is
where "faster" becomes a meaningful claim, and it is the right place to look at
the five null IVF-PQ cells. `gpuq_job_spec.json` pins the command, commit,
lane, timeout and exit status.

A cell whose curve never reaches the target reports null rather than the
nearest point or an extrapolation. A cell whose curve never drops below the
target reports its fastest measured point, marked `floor`, with the recall it
was measured at.

## Reproducing

The run directories are spread over two trees on the box, while
`compare_variants` resolves every rung against a single `--root`. A staging
directory of symlinks reconciles that without changing the manifest format;
the recipe is in
`docs/superpowers/plans/2026-08-12-ann-gpu-benchmark-probe.md`, together with
the measured cuVS API surface and the box paths.

Corpora, query sets and ground truth are cached in the work directory and keyed
by a `manifest.json` recording the seed, batch size, sizes, k and source, so a
rerun under different parameters rematerializes rather than silently serving
the previous draw.

cuVS is **not** in `requirements.txt`: it is CUDA-only and installs from
NVIDIA's package index, so pinning it would break the CPU-only install CI runs.
It is a box-side extra (`cuvs-cu13`, `cupy-cuda13x`), and the CLI fails at
entry with the install command if it is absent. `make check` stays CPU-only,
dataset-free and cuVS-free.
