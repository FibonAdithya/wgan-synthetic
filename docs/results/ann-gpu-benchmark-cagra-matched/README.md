# CAGRA at matched recall 0.90

NVIDIA GeForce RTX 3060 (12,288 MiB, driver 580.173.02), PyTorch 2.13.0+cu130,
cuVS 26.08.01, cupy 14.2.0, Linux 6.8.0-59. SIFT1M, L2-normalized, 1,000,000
vectors, 10,000 queries, k=10, 5 repeats, squared-L2 throughout — the same
measurement discipline as `docs/results/ann-gpu-benchmark/`, on different
hardware. See "Not comparable with the published table" below before reading a
ratio across the two.

## The question

`docs/results/ann-gpu-benchmark/` reports IVF-Flat at recall@10 = 0.90 and
CAGRA at a *floor* — its fastest measured point, at whatever recall that point
happened to land on (0.963 on `real`). That asymmetry is not a presentation
choice: `itopk_size` must be a multiple of 32 and at least k, so 32 is its
floor, and at 32 every corpus already clears 0.90. The sweep bottoms out above
the target, and `metrics.qps_at_recall` refuses to extrapolate past its
cheapest measured point. So the shipped table can say how fast IVF-Flat is at
0.90 and cannot say it for CAGRA.

This run answers it with the second search-side knob cuVS exposes.

## What was swept, and why that knob

`max_iterations` caps CAGRA's graph traversal. cuVS auto-selects the cap from
`itopk_size` and `search_width` when it is left at 0; measured here that
auto-selected value lands somewhere below 48, since an explicit cap of 48 gives
slightly *more* recall than auto and 64 gives no more than 48. Capping it below
that trades recall for speed **without touching the graph**. The index is
byte-for-byte the one `CagraAdapter` builds (`graph_degree=64`,
`intermediate_graph_degree=128`); `CagraIterationsAdapter` inherits `build`
rather than reimplementing it, so a QPS read off this curve is the published
index searched more cheaply, not a different index.

Lowering `graph_degree` would also reach 0.90 and would not answer the same
question: it builds a different index, moving build time, VRAM and the whole
recall curve with it.

## Result

| Index | Knob at the target | Recall | QPS | vs brute force | vs IVF-Flat |
|---|---|---:|---:|---:|---:|
| CAGRA (`cagra_iters`) | `max_iterations` 20→22 | **0.9000** | **467,926** | 65.7x | 5.1x |
| IVF-Flat | `n_probes` 16→32 | 0.9000 | 91,128 | 12.8x | 1.0x |
| Flat (exact) | — | 1.0000 | 7,124 | 1.0x | 0.08x |
| CAGRA (`cagra`, floor) | `itopk_size=32` | 0.9652 | 273,276 | 38.4x | 3.0x |

Both 0.90 figures are interpolated between measured points that bracket the
target, in log(QPS); neither is an extrapolation. CAGRA is **1.71x faster at
0.90 than at its own published floor**, which is the size of the gap the
shipped table could not quote.

## The measured curve

`max_iterations`, at `itopk_size=32`:

| `max_iterations` | Recall | QPS (median of 5) |
|---:|---:|---:|
| 1 | 0.0001 | 3,098,499 |
| 2 | 0.0047 | 2,566,332 |
| 3 | 0.0416 | 2,270,012 |
| 4 | 0.1443 | 1,888,582 |
| 6 | 0.4214 | 1,361,555 |
| 8 | 0.6030 | 1,069,589 |
| 12 | 0.7748 | 746,643 |
| 16 | 0.8535 | 580,414 |
| 18 | 0.8786 | 519,919 |
| **20** | **0.8981** | **473,350** |
| 22 | 0.9137 | 430,286 |
| 24 | 0.9253 | 397,770 |
| 32 | 0.9552 | 302,587 |
| 48 | 0.9658 | 267,751 |
| 64 | 0.9658 | 267,981 |

Recall rises monotonically across all fifteen points. The target crossing is
bracketed by 20 (0.8981) and 22 (0.9137) — 0.0156 of recall — so the headline
is very nearly a measured point rather than an interpolation. An earlier run of
this same sweep without 18/20/22 bracketed it between 16 and 24 (0.07 of
recall) and interpolated 448,547, 4% low; that run's artifact was superseded by
this one and is not kept.

## Two checks the numbers pass

**The curves join.** `max_iterations` 48 and 64 give identical results (0.9658,
~268k QPS): both exceed the auto cap, so the search runs to convergence and the
knob stops binding. That converged point sits just above the published-knob
floor at `itopk_size=32` (0.9652, 273,276 QPS) — auto stops at ~38 iterations,
an explicit larger cap buys the last couple. The two adapters describe one
continuous curve through the same index, which is what makes reading a matched
figure off one of them and a floor off the other legitimate.

**Run-to-run agreement.** The earlier run above shares all twelve of its points
with this one. Across them QPS moved by at most 4.7% (mean 1.4%) and recall by
at most 0.0023. Both worst cases are at the cheap end of the sweep — 4.7% at
`max_iterations=3`, 0.0023 at 6 — where a whole 10,000-query batch takes 4–7 ms
and timing noise is a large fraction of it. Over the eight points from 12
upwards, which is the half the headline is read from, QPS moved by at most 1.5%
and recall by at most 0.0010. The 1.71x over CAGRA's own floor and the 5.1x
over IVF-Flat are both far outside that.

## Not comparable with the published table

`docs/results/ann-gpu-benchmark/` was measured on an **RTX 4060**; the box was
rebuilt on 2026-08-19 and now has an **RTX 3060**. Absolute QPS does not carry
across. That is why `flat` and `ivf_flat` were re-measured here rather than
quoted: every ratio in this document is between cells measured on the same card
in the same job. For orientation, the same cells on the two machines:

| Cell | RTX 4060 (published) | RTX 3060 (here) |
|---|---:|---:|
| flat, exact | 7,996 | 7,124 |
| ivf_flat @ 0.90 | 73,063 | 91,128 |
| cagra floor (`itopk_size=32`) | 250,469 @ 0.963 | 273,276 @ 0.965 |

Recall tracks closely between the two — it is a property of the corpus and the
index, not the card — while QPS does not, in both directions.

**Only `real` was run.** The variant ladder (`v0`–`v4`) is absent: the 2026-08-19
rebuild wiped `/workspace/keep`, and those checkpoints are gone with it. So this
document says nothing about whether the synthetic corpora separate from real
SIFT under a matched-recall CAGRA — that question is still open, and answering
it means retraining the ladder first.

## Files

`ann_benchmark.json` is every cell unaggregated plus the environment block.
`ann_benchmark.md` is the headline table as the runner emitted it.
`report.html` is self-contained and carries the recall-vs-QPS curve per cell.
`gpuq_job_spec.json` pins the command, commit, lane, timeout and exit status.

## Reproducing

    python -m src.eval.ann_benchmark \
        --real-path /workspace/data-cache/sift_1m.npy \
        --real-hdf5-path /workspace/data-cache/hdf5/sift-128-euclidean.hdf5 \
        --work-dir /workspace/annbench-cagra-work \
        --output-dir docs/results/ann-gpu-benchmark-cagra-matched \
        --indexes flat ivf_flat cagra cagra_iters \
        --allow-missing

`cagra_iters` is opt-in via `--indexes` and is not in `DEFAULT_INDEX_NAMES`, so
the shipped grid's shape does not move because this adapter exists.
