# Is the exact-search ceiling real? Three torch baselines against cuVS

Measured 2026-08-17 from commit `df53c97` on the exclusive `gpuq` GPU lane
(job `wgan-synthetic-20260817T082226Z-e3dfd5`, exit 0). One NVIDIA GeForce
RTX 4060 (driver 580.95.05), CUDA 13.0, cuVS 26.08.01, PyTorch 2.12.0+cu130,
Python 3.12.3. Seven corpora, 28 build and 28 search cells, zero failures.

This is a probe, not a benchmark of the variant ladder. It exists to test one
assumption the main grid rests on, and it reports a null result.

## The question

`docs/results/ann-gpu-benchmark/` prices every approximate index against cuVS
brute force, at 7,996 QPS on 1M x 128. How good that baseline is decides how
much of ANN's measured advantage is real — and at matched recall the advantage
is already thin, with CAGRA at 2.6x and IVF-Flat at break-even.

7,996 QPS is about **14% of the card's FP32 peak**, which invites the
objection that the ceiling is just an unoptimized implementation. Decomposing
one 10,000-query batch sharpens it: the batch takes 1,251 ms, of which the
distance GEMM (2.56 TFLOP at a 15.1 TFLOP/s peak) accounts for only ~170 ms.
Roughly **86% of the time is not the distance computation** — it is top-10
selection over 10^10 candidate distances, a matrix that would be 40 GB if
materialized and so must be scanned in fused tiles.

If that 86% were addressable, the ceiling would move and the ladder's
conclusions would move with it. So it was measured.

## What was run

Three torch implementations beside cuVS `flat`, in one process, over the same
cached corpora and query sets, through the same runner — same warmup, same
five fenced repeats, same distance-based recall scored from recomputed exact
distances.

| Adapter | What it isolates |
|---|---|
| `flat` | cuVS brute force: the existing baseline, re-measured as the control |
| `torch_flat` | FP32, TF32 explicitly off — tiling strategy alone |
| `torch_flat_tf32` | TF32 tensor cores |
| `torch_flat_fp16` | FP16 storage and matmul, FP32 accumulate and FP32 norms |

Splitting it three ways matters: a slow baseline could be a poor selection
strategy *or* simply not using the tensor cores, and one alternative
implementation could not tell those apart. The torch adapters tile at
`query_chunk=2048` x `corpus_tile=65536`, keeping the score tile at 537 MB.

## Result: cuVS is the fastest of the four

On `real`:

| Index | Recall | QPS min | QPS median | QPS p95 | vs cuVS |
|---|---:|---:|---:|---:|---:|
| `flat` (cuVS) | **1.0000** | 7,944.8 | **7,984.5** | 8,006.4 | 1.000x |
| `torch_flat` (FP32) | 1.0000 | 5,555.6 | 5,559.1 | 5,563.5 | 0.696x |
| `torch_flat_tf32` | 0.9934 | 5,897.2 | 5,904.7 | 5,907.1 | 0.740x |
| `torch_flat_fp16` | 0.9845 | 5,431.2 | 5,436.5 | 5,437.4 | 0.681x |

**The control reproduces.** cuVS measured 7,984.5 here against 7,996.2 in the
published run — a 0.15% move, inside that grid's own 0.4% run-to-run floor. So
this is a like-for-like comparison on one card, not two different setups.

### Reduced precision does not help, and that is the informative part

TF32 bought **+6.2%** over torch FP32; FP16 was **2.2% slower**. Were the
distance GEMM the bottleneck, halving its precision should have moved a large
share of the runtime. It did not move at all.

That confirms the diagnosis the arithmetic suggested: the work is selection,
not multiplication, and **selection is precision-independent**. The whole
class of tensor-core optimizations cannot reach this problem. FP16 being
*slower* is consistent with the same picture — it halves a cost that was
already 14% of the total while adding a conversion.

Both reduced-precision arms also **lose exactness** (0.9934 and 0.9845). That
disqualifies them from the exact-ceiling role regardless of throughput: search
at recall 0.984 is not exact search, and could not be used as ground truth.
Only `flat` and `torch_flat` return recall 1.0000.

### Brute force is data-independent, as expected

Across all seven corpora:

| Index | QPS range | Spread | Recall range |
|---|---|---:|---|
| `flat` | 7,886.5–7,984.5 | 1.24% | 1.0000 |
| `torch_flat` | 5,552.9–5,559.1 | **0.11%** | 1.0000 |
| `torch_flat_tf32` | 5,901.0–5,905.1 | 0.07% | 0.9934–0.9976 |
| `torch_flat_fp16` | 5,433.9–5,436.8 | 0.05% | 0.9845–0.9881 |

This is the expected null and the reason brute force is a sound control: a
full scan's cost depends on N and d, which are pinned at 1,000,000 x 128
across every corpus, so there is no mechanism for the distribution to show up.
The torch adapters' near-zero spread is the cleaner demonstration; cuVS's
wider 1.24% is most likely ordering or thermal drift within the job (`real`
runs first) rather than data dependence, but it is not separately established
here.

## What this does and does not establish

It establishes that the ~8,000 QPS exact-search ceiling is **not an artifact
of one unoptimized implementation**. Three independent attempts, including
both tensor-core paths, failed to beat it. The main grid's matched-recall
figures — CAGRA 2.6x at recall 1.000, IVF-Flat break-even — are therefore real
numbers against a defensible baseline.

It does **not** prove cuVS optimal. This is one implementation family plus two
precision variants, at one tiling. A hand-written fused selection kernel could
still do better, and the evidence points at exactly that as the only place
headroom could be, since selection is where the time goes. What can be said is
that the headroom is not reachable by the two cheapest routes — a different
tiling strategy, or lower-precision arithmetic.

It is also measured at one point: N = 1M, d = 128, batch 10,000, one RTX 4060.
Brute force is unusually well served by every one of those choices. Nothing
here transfers to a larger corpus or a batch-1 latency regime.

## Cost

| Index | Add (s) | Index size (est.) |
|---|---:|---:|
| `flat` | 0.084 | 512.0 MB |
| `torch_flat` | 0.158 | 516.0 MB |
| `torch_flat_tf32` | 0.117 | 516.0 MB |
| `torch_flat_fp16` | 0.117 | 260.0 MB |

None has a training phase. The torch figures include the FP32 squared norms
(4.0 MB), which stay FP32 in the FP16 adapter too: they are a per-vector
constant added once per distance, so computing them in half precision would
cost accuracy on every distance and buy no throughput.

Peak VRAM is omitted rather than tabulated. It is a card-wide delta sampled
around each build, and torch's caching allocator does not return memory
between adapters, so the second and third torch builds both sample a delta of
zero — a fact about the allocator, not about their footprint.

## Files

`ann_benchmark.json` is every cell plus the environment block, and is the
source of truth. `ann_benchmark.md` is the headline table. `report.html`
carries the per-cell view.

This run's environment block does **not** carry a `versions` field: the field
was added in response to this run (`indexes.stack_versions`, wired in by
`cli.environment_block`) and postdates the job by a commit. The versions in
the header above were read off the box directly and are not recoverable from
the artifact. Runs after `df53c97` record them in the JSON.

`report.html` was **regenerated locally from `ann_benchmark.json`** rather than
copied off the box: `scp` truncated it three times at exact KiB boundaries
while exiting 0. The JSON transferred whole — `ann_benchmark.md` reproduces
from it byte-for-byte — and the regenerated HTML is byte-identical to the
box's in 74 of its 75 64-KiB blocks, differing only in Plotly's random div id.
Anything pulled off that box should be checksummed against the remote; the
exit code is not a transfer-integrity signal.

## Reproducing

```
python -m src.eval.ann_benchmark \
    --real-path /workspace/data-cache/sift_1m.npy \
    --cache-dir /workspace/data-cache \
    --root /workspace/annbench-root \
    --work-dir /workspace/annbench-work \
    --output-dir docs/results/ann-gpu-benchmark-bruteforce \
    --indexes flat torch_flat torch_flat_tf32 torch_flat_fp16
```

The torch adapters are registered but off by default; `--indexes` defaults to
the four cuVS indexes the published grid reports, so adding this probe did not
change that artifact's shape. Unlike the cuVS adapters they run under
`make check`, since torch installs CPU-only: the tiling, the cross-tile merge
and `k` larger than one tile are covered on CPU in
`tests/test_ann_benchmark_indexes.py`.
