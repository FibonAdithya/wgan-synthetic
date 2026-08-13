# SIFT v0, v1 and v4 against the real corpus

Run 2026-08-13 from commit `2bf36eb` on branch `benchmark-algos`, CPU-lane `gpuq`
jobs. `job_spec.json` and `run_eda3.sh` hold the invocation verbatim.

`report.html` is the written report, self-contained, with the figures embedded.
This file is the metric record.

## Why this exists

`docs/results/v4-logratio/` compared v4 against v3 and found the track's first
positive result. It did not look further down the ladder. Reading v0, v1 and v4
against real SIFT in one measurement asks a different question — does the ladder
monotonically approach real SIFT? — and the answer is no.

## Conditions

Canonical SIFT conditions from `docs/datasets/sift.md`: N=20000, k=100, hubness
k=10, nlist=256, `--preprocess l2`, seed 42, real corpus `sift_1m.npy` subsampled
to 50000. All four series were measured in a **single** `eda_report` invocation,
so they share one real-side subsample and rung-to-rung differences carry no
sampling noise of their own.

| Series | Run | Steps | Delta |
|---|---|---|---|
| `v0` | `runs/long_baseline` | 30k | plain WGAN-GP |
| `v1` | `runs/x100k_ema_only` | 100k | + generator EMA |
| `v4` | `runs/sift/v4_sift1m` | 30k | + structured gate + log-ratio regularizer |

**One confound to hold in mind:** v1 is a 100k-step run while v0 and v4 are both
30k, so the v0→v1 step mixes generator EMA with 3.3x more training. Run length is
an independent axis, not a variant, and nothing here separates the two.

## Result

| Statistic | Real | v0 | v1 | v4 |
|---|---|---|---|---|
| LID median | 17.7383 | 17.2997 | 17.4497 | 15.5465 |
| Relative contrast | 2.2673 | 2.2734 | 2.2526 | 2.4140 |
| Hubness skew | 1.8839 | 2.0880 | 2.0958 | 1.5984 |
| IVF cell-balance Gini | 0.3040 | 0.2718 | 0.2842 | 0.2681 |

**v0 — plain WGAN-GP, no EMA, no gated generator, no regularizer — is closer to
real SIFT than v4 on all four gate statistics.** In absolute gap:

| Statistic | gap v0 | gap v1 | gap v4 |
|---|---|---|---|
| LID median | **0.4386** | 0.2886 | 2.1918 |
| Relative contrast | **0.0061** | 0.0146 | 0.1467 |
| Hubness skew | **0.2041** | 0.2119 | 0.2855 |
| IVF Gini | 0.0322 | **0.0198** | 0.0359 |

### Is it noise?

Floors are per architecture and do not transfer. The MLP rungs take the v0
seed-to-seed floor from `docs/datasets/sift.md` (LID 0.164, contrast 0.048,
hubness 0.062, Gini 0.007); v4 takes the architecture-matched v3-pair spread from
`docs/results/v4-logratio/` (LID 0.0930, contrast 0.0203, hubness 0.0382, Gini
0.0131).

| Statistic | v0 | v1 | v4 |
|---|---|---|---|
| LID median | 2.67x | 1.76x | 23.57x |
| Relative contrast | **0.13x** | **0.30x** | 7.23x |
| Hubness skew | 3.29x | 3.42x | 7.47x |
| IVF Gini | 4.60x | 2.83x | 2.74x |

Because the two floors differ, these are **not** a cross-rung ranking — they
answer "is this gap real?" only. Rank on the absolute gaps above.

Both MLP rungs land **within noise of real on relative contrast**. v4 is 7.2x out.
The log-ratio regularizer moved relative contrast away from real relative to the
baseline it was built on top of.

**v4's LID is not evidence**, on any floor. `lid_reg` fits the mean log-ratio
profile of within-batch neighbours, the sufficient statistic the Hill estimator
reduces to a scalar, so a good `lid_median` was guaranteed by construction. It
checks the penalty worked. It does not rank the rung.

## What the gate does not see

| | Real | v0 | v1 | v4 |
|---|---|---|---|---|
| Exact-zero fraction | 0.2298 | 0.0000 | 0.0000 | 0.2441 |
| Negative fraction | 0.0000 | 0.1182 | 0.0983 | 0.0000 |
| Minimum coordinate | 0.0000 | −0.1221 | −0.0974 | 0.0000 |
| Duplicate rows | 0.00062 | 0.0 | 0.0 | 0.0 |
| Effective rank | 27.99 | 27.11 | 27.50 | 26.81 |
| Median 5-NN distance | 0.5153 | 0.5091 | 0.5140 | 0.4894 |

Real SIFT is non-negative and quantized. v0 and v1 produce **zero** exact zeros
and put 10–12% of their mass below zero, down to −0.12. v4 reproduces the zero
mass to within 0.014 and never goes negative.

The descriptor glyph panel shows this directly, and is the only panel that can:
the two real rows carry no negative bins, the two MLP rows are shot through with
them, and v4's rows are clean.

**This is the finding.** The four gate statistics measure neighbourhood geometry
and do not constrain support, so a rung can win them while producing 128 numbers
that are not a plausible gradient histogram.

## The EMA rung moves nothing

On the gate, v0→v1 is a wash: LID +0.1501 (0.91x floor), relative contrast −0.0208
(0.43x), hubness skew +0.0078 (0.13x), IVF Gini +0.0124 (1.77x). Three of four are
under 1x — smaller than a reseed — and Gini, the only one above, is the weakest
discriminator of the four.

That is despite v1 carrying both generator EMA *and* 3.3x the training. The two
are not separated here, so this says the combination lands within noise of the
baseline; it does not say which of them is inert.

## Distribution diagnostics

Not the gate (`AGENTS.md` invariant 1). Reported here because measuring them shows
why that rule exists.

A distribution diagnostic is unreadable without knowing how far real is from
itself, so the corpus was split into two disjoint 20000-row halves: one is the
reference, the other is scored against it the same way and is the **floor**.
Computed by `dist_diag.py`, equal-N at 20000, seed 42, same L2 preprocessing.

| Diagnostic | real vs real | v0 | v1 | v4 |
|---|---|---|---|---|
| Per-dimension W1, mean | 0.000678 | 0.004303 | 0.002285 | 0.003558 |
| Per-dimension W1, worst dim | 0.001777 | 0.007808 | 0.003765 | 0.007309 |
| Covariance Frobenius | 0.006382 | 0.020765 | 0.005804 | 0.006588 |
| MMD (RBF, γ=1) | 0.000430 | 0.001244 | 0.000284 | 0.000234 |
| Pairwise-distance hist L1 | 0.4050 | 0.7893 | 0.4100 | 0.5365 |

As a multiple of the floor:

| Diagnostic | v0 | v1 | v4 |
|---|---|---|---|
| Per-dimension W1, mean | 6.3x | 3.4x | 5.2x |
| Per-dimension W1, worst dim | 4.4x | 2.1x | 4.1x |
| Covariance Frobenius | 3.3x | **0.9x** | **1.0x** |
| MMD (RBF, γ=1) | 2.9x | **0.7x** | **0.5x** |
| Pairwise-distance hist L1 | 1.9x | **1.0x** | **1.3x** |

**Three of the five are saturated.** Covariance Frobenius, MMD and the
pairwise-distance histogram cannot separate v1 or v4 from a disjoint sample of
real SIFT. On MMD, both score *better than real does against itself*.

**MMD ranks v4 first; the gate ranks it last.** Both are correct — they measure
different things, and only one is the project's question. This is the "MMD
improved, looks good" trap from `AGENTS.md`, made concrete.

Only the per-dimension marginals retain resolution: every rung clears the floor,
and they rank **v1 best, v0 worst**. The correlation-structure panel agrees — real
SIFT's 4x4 grid of 8-bin histograms correlates at lags of 8 and 32, v0's residual
against it is heavily structured, v1's is close to flat noise, v4 recovers most of
it but leaves visible diagonal bands.

### Three lenses, three winners

| Question | Instrument | Winner |
|---|---|---|
| Neighbourhood geometry | the four gated statistics | v0 |
| Support | exact zeros, negatives, glyphs | v4, uniquely |
| Distribution | per-dimension marginals, correlation structure | v1 |

No rung wins two. Any single scalar would have reported progress that is not there.

## A panel that cannot be read

The relative-contrast histogram renders with its x-axis to ~6.5e7 and every bar
in the first bin. Relative contrast divides by the nearest-neighbour distance,
and SIFT's quantized lattice puts near-coincident pairs in the corpus, so a few
real queries divide by almost zero. Six real queries were dropped outright by
`survivor_mask`; **every synthetic rung dropped zero** — no generator here
produces a single exact duplicate or fully tied neighbourhood. The median the
gate uses is unaffected; only the histogram is. Worth an issue against the panel.

## What this does not say

- **Not an argument for v0.** It is a finding about what four aggregate
  statistics fail to constrain.
- **No band was set or moved.** `gates/sift.yaml` is untouched; every band there
  is still null. Choosing a number is reserved for a human.
- **IVF Gini ranks nothing.** The architecture-matched floor makes it the weakest
  of the four, as `docs/results/v4-logratio/` already found.
- **n=1 per rung.** Both gate floors are themselves n=2 and are lower bounds: two
  runs at identical seed still diverge 1–8% per loss column by step 250 from
  nondeterministic CUDA reduction order.
- **The real-vs-real floor is one split.** It resolves "saturated" from "clearly
  above", which is all it is used for. It does not support fine distinctions near
  1x, and the ordering of v1 and v4 on the three saturated diagnostics should not
  be read at all.

## Suggested follow-up

The gate needs a support check beside the four statistics. `exact_zero_fraction`
and `negative_fraction` are already computed by `eda_report` and would have
caught v0 immediately. Whether that becomes a fifth gated statistic or a
precondition is a banding decision, and belongs to a human.

## Artifacts on the box

`/workspace/keep/sift-v0-v1-v4/eda3/` on `tig-gpu` holds the stock
`eda_report.html` and all twelve per-panel PNGs. All three draws are reused
verbatim from `/workspace/keep/sift-ladder/`.
