# SIFT generation timing

Measured 2026-08-13 from commit `3a53874` on the exclusive `gpuq` GPU lane,
job `wgan-synthetic-20260813T091823Z-f87eb0`. The machine had one NVIDIA
GeForce RTX 4060, PyTorch 2.13.0+cu130, and CUDA 13.0. The benchmark used
batch size 4096, seed 42, and five repeats per cell. The generators were
randomly initialized from the matched SIFT v1, v2, and v4 configs; these are
architecture timing measurements, not quality comparisons.

`generation_benchmark.json` is the full phase-level record and
`generation_benchmark.md` is the budgeting view. `gpuq_job_spec.json` is the
queue's own record of the job: command, commit, lane, timeout, exit status,
and log paths.

All twelve corpus sizes — 1,000 and 10,000 plus 100k to 1M in 100k steps —
come from **one run**, so no figure here is stitched together from separately
measured grids.

## What changed from the 2026-08-11 measurements

The earlier runs in this directory were taken with a harness that mismeasured
three things; they are superseded and were removed in the same commit as this
rewrite (they remain in git history at `eed10fc`). The conclusions they
supported have been withdrawn:

- The warmup ran `generator(z)` while the timed region ran
  `normalize_l2(generator(z))`, so the normalization kernels initialized inside
  the first measured cell. That is the whole of the old "startup-sized outlier
  even though the configured warmup had already run": v1 at N=1,000 had a
  p95/median ratio of 80x, and now has 1.03x.
- Each batch was copied to the host twice — once into a fresh pageable tensor,
  once into the preallocated array — with both copies inside `to_host_seconds`.
  Host time at N=1M has dropped from 0.29–0.58 s to 0.126 s.
- Every cell ran its five repeats back-to-back in ascending-N order, so
  slow-varying machine state landed on whole cells and aliased onto the N axis
  the linearity fit uses. Repeats are now interleaved as re-randomized rounds
  over all 36 cells.

The clearest evidence that the old host numbers were an artifact: they differed
per architecture (v1 0.29 s, v2 0.45 s, v4 0.58 s at N=1M) for a transfer of the
same 488.3 MiB. They are now 0.1255, 0.1261, and 0.1265 s — identical, as a
device-to-host copy of a fixed byte count should be.

## Headline figures at N=1M

| Version | Generate median | To host median | p95 budget | Generate vectors/s | End-to-end vectors/s |
|---|---:|---:|---:|---:|---:|
| v1 (dense) | 0.533 s | 0.128 s | 0.895 s | 1.88M/s | 1.51M/s |
| v2 (gated) | 0.616 s | 0.126 s | 0.822 s | 1.62M/s | 1.35M/s |
| v4 (structured) | 0.868 s | 0.129 s | 1.041 s | 1.15M/s | 1.00M/s |

Quote the end-to-end figure when sizing a deadline. The generate-only rate
excludes the host phase, which is a real cost whenever a CPU-side ANN index
needs the corpus as a NumPy array.

CUDA initialization was 0.264 s once per process. Build was 0.018–0.025 s and
warmup 0.09–0.15 s per generator; both are separate from the per-corpus budget.

## Linearity

Linear regressions over the ten 100k–1M points, all from the single run:

| Version | Phase | Seconds / 1M vectors | R² |
|---|---|---:|---:|
| v1 | generation median | 0.535 | 0.9999 |
| v2 | generation median | 0.619 | 0.9982 |
| v4 | generation median | 0.866 | 1.0000 |
| v1 | host median | 0.128 | 0.9993 |
| v2 | host median | 0.118 | 0.3947 |
| v4 | host median | 0.128 | 0.9998 |
| v1 | p95 budget | 0.751 | 0.9147 |
| v2 | p95 budget | 0.810 | 0.9361 |
| v4 | p95 budget | 0.998 | 0.9657 |

**Both phases are linear in N.** The previous claim that host materialization
is not reliably linear — and the specific claim that v1 drops sharply at 800k
and v4 at 700k — do not survive the fixed harness. Those kinks were the
back-to-back repeat schedule, not a property of host transfer.

The one low R² left is v2's host fit, and it is a single contaminated cell
rather than curvature. At N=500k that cell's five repeats have a minimum of
0.127 µs/vector — the same rate as every other cell — but a median of 0.408.
Elevated p95 values also appear at v2/600k, v2/900k, and v2/1M. That is a
time-localized slow episode landing on scattered repeats, which is exactly the
shape interleaving is meant to expose: under the old schedule it would have
presented as a clean-looking kink at one N. Fitting the per-cell **minimum**
instead of the median gives host R² of 0.99994, 0.99993, and 0.99996 for v1,
v2, and v4.

Use the median for a typical cost, the p95 for a deadline, and treat the p95
fits (R² 0.91–0.97) as tail-sensitive rather than as evidence of curvature.

## Memory

| Version | Model parameters | Sampling peak above baseline | 1M host output |
|---|---:|---:|---:|
| v1 | 6.76 MiB (7,088,640 B) | 36.0 MiB | 488.3 MiB |
| v2 | 7.26 MiB (7,613,440 B) | 40.0 MiB | 488.3 MiB |
| v4 | 7.27 MiB (7,618,268 B) | 38.5 MiB | 488.3 MiB |

**Model parameters** is now counted directly from the module's parameters and
buffers. The earlier figures of 14.9/15.4/15.4 MiB were post-warmup
`torch.cuda.memory_allocated`, which carries roughly 8.1 MiB of one-time
library workspace that does not scale per model — so they overstated the cost
of holding a generator beside a GPU ANN index by about 2x.

**Sampling peak above baseline** is the per-repeat peak allocation minus the
allocation at that repeat's start, taken as the worst of the five repeats. It
stays flat from 100k to 1M because generation is streamed in fixed 4,096-vector
batches. It is the figure to use when deciding whether generation can coexist
with a GPU index.

Peak *reserved* memory is not reported per architecture. This run holds all
three generators resident so their cells can be interleaved, and the caching
allocator's pool is already warm by the time any cell runs, so the incremental
reserved figure is 0 for every cell. Reserved-peak attribution needs a
single-generator run; the allocated numbers above do not.

## Open question: generation is ~10% slower than the 2026-08-11 run

v2 and v4 device-side generation is about 10% slower here than in the
superseded run (v2 0.554→0.609 s, v4 0.777→0.855 s at N=1M, comparing minima);
v1 is unchanged at 0.98x. One plausible mechanism is that the old run's much
slower host phase left 0.3–0.45 s of GPU idle time per 1M vectors, and the
heavier v2/v4 graphs benefited from that thermal headroom on a 4060 in a way
the light v1 graph did not. **This is untested.** It does not affect the
linearity conclusions or the relative ordering of the three architectures, but
do not treat the absolute generate figures as hardware-independent.

These timings are operational diagnostics only. They do not measure or change
the ANN-difficulty gate.

## Reproduce

    python -m src.sample.benchmark --device cuda:0 \
      --num-samples 1000 --num-samples 10000 \
      --num-samples 100000 --num-samples 200000 --num-samples 300000 \
      --num-samples 400000 --num-samples 500000 --num-samples 600000 \
      --num-samples 700000 --num-samples 800000 --num-samples 900000 \
      --num-samples 1000000 \
      --output-dir benchmark-output

Submit it to the GPU lane rather than running it directly. The full 36-cell
grid took under two minutes on an otherwise idle RTX 4060 (submitted
09:18:23Z, results written 09:20:00Z, including the runner's checkout).
