# SIFT v0, v1 and v4 at 100k steps

Run 2026-08-13 from commit `2bf36eb` on branch `benchmark-algos`. `job_spec.json`,
`run_100k.sh`, `run_floor100k.sh` and `dist_diag.py` hold the invocations verbatim.

`report.html` is the written report, self-contained, with the figures embedded.
This file is the metric record.

**This supersedes a 30k measurement and reverses its headline.** That version is in
this page's git history (`010ec22`..`6ade0d2`). At 30k, v0 — plain WGAN-GP — was
closer to real than v4 on all four gated statistics. At 100k it is closer on one.

## Why

`docs/results/v4-logratio/` compared v3 against v4 at 30k. The first version of this
page compared v0, v1 and v4, also at 30k for v0/v4, and found the ladder did not
progress. Both were too short: **v4 was undertrained at 30k.**

Between 30k and 100k, v4 closes 38–98% of each gate gap; v0 closes 15–28%.

| Rung | LID | Relative contrast | Hubness skew | IVF Gini |
|---|---|---|---|---|
| v0 | 0.4386 → 0.3508 (+20%) | 0.0061 → 0.0052 (+15%) | 0.2041 → 0.1700 (+17%) | 0.0322 → 0.0231 (+28%) |
| v4 | 2.1918 → 1.3526 (+38%) | 0.1467 → 0.0483 (+67%) | 0.2855 → 0.1478 (+48%) | 0.0359 → 0.0005 (+98%) |

Absolute gap to real at 30k → at 100k, and the fraction closed. The regularizers and
the structured gate need well past 30k steps to express themselves; the plain
adversarial baseline has largely stopped moving by then.

**A ladder comparison at a run length short enough to leave the regularized rung
undertrained will systematically favour the simpler rung.** That is the
methodological finding, and it outweighs either rung's ranking.

## Conditions

Canonical SIFT conditions from `docs/datasets/sift.md`: N=20000, k=100, hubness
k=10, nlist=256, `--preprocess l2`, seed 42, real corpus `sift_1m.npy` subsampled
to 50000. All four series were measured in a **single** `eda_report` invocation, so
they share one real-side subsample and rung-to-rung differences carry no sampling
noise of their own.

| Series | Run | Steps | Delta |
|---|---|---|---|
| `v0` | `v0_x100k` (trained for this report) | 100k | plain WGAN-GP |
| `v1` | `runs/x100k_ema_only` | 100k | + generator EMA |
| `v4` | `v4_x100k` | 100k | + structured gate + log-ratio regularizer |

All three are 100k steps, seed 42, on `sift_1m.npy`, same architecture. The v0
config is v1's own config with `ema_decay` removed and nothing else changed, so v0
and v1 differ in exactly one key. The ladder had no 100k v0, so it was trained here
(112 min on one RTX 4060). **The run-length confound in the 30k version is gone.**

## Result

| Statistic | Real | v0 | v1 | v4 |
|---|---|---|---|---|
| LID median | 17.7383 | 17.3875 | 17.4497 | 16.3857 |
| Relative contrast | 2.2673 | 2.2621 | 2.2526 | 2.3156 |
| Hubness skew | 1.8839 | 2.0539 | 2.0958 | 1.7361 |
| IVF cell-balance Gini | 0.3040 | 0.2810 | 0.2842 | 0.3035 |

Closest to real, by absolute gap:

| Statistic | Closest | By absolute gap |
|---|---|---|
| LID median (v4 fitted — not counted) | v1 | v1 0.2886 < v0 0.3508 < v4 1.3526 |
| Relative contrast | **v0** | v0 0.0052 < v1 0.0146 < v4 0.0483 |
| Hubness skew | **v4** | v4 0.1478 < v0 0.1700 < v1 0.2119 |
| IVF cell-balance Gini | **v4** | v4 0.0005 < v1 0.0198 < v0 0.0231 |

No rung sweeps the gate. **v4 takes hubness skew and IVF Gini**, and its Gini is
nearly exact — 0.3035 against 0.3040, a gap of 0.0005, a fifth of v1's and a
twentieth of v0's. **v0 keeps relative contrast**, though only just: at 1.30x the
measured 100k floor it is marginal, not indistinguishable from real as the 30k
floor suggested. Discounting LID leaves three usable statistics, split 2–1 for v4.

### In noise-floor units

The floor was measured at this run length: two 100k v0 runs identical in every
hyperparameter except the training seed (42 and 43), both sampled at a fixed seed
of 42, measured in one `eda_report` invocation. `floor_100k.json` holds it.

**It does not transfer across run length.** The 30k floor this page previously
divided by was wrong in both directions:

| Statistic | 30k floor | 100k floor | ratio |
|---|---|---|---|
| LID median | 0.1640 | 0.1026 | 0.63x |
| Relative contrast | 0.0480 | 0.0040 | **0.08x** |
| Hubness skew | 0.0620 | 0.1245 | **2.01x** |
| IVF Gini | 0.0070 | 0.0072 | 1.03x |

Relative contrast's floor is 12x tighter at 100k; hubness skew's is twice as loose.

| Statistic | v0 | v1 | v4 |
|---|---|---|---|
| LID median | 3.42x | 2.81x | — |
| Relative contrast | 1.30x | 3.68x | — |
| Hubness skew | 1.37x | 1.70x | — |
| IVF Gini | 3.21x | 2.75x | — |

**Against the measured floor, no v0 or v1 gap is inside noise.** The narrowest are
v0's relative contrast at 1.30x and its hubness at 1.37x; the rest run to 3.7x.

The floor is also **per architecture**, so it does not cover v4. No 100k floor
exists for the structured-gate architecture, and the 30k v3-pair spread cannot
stand in now that run length is known to move a floor 12x. **v4's multiples are
withheld** rather than computed against a floor known to be wrong at this run
length; its standing rests on the absolute gaps above, which need no floor.

**v4's LID is not evidence and ranks nothing.** `lid_reg` fits the mean log-ratio
profile of within-batch neighbours, the sufficient statistic the Hill estimator
reduces to a scalar, so a good `lid_median` was guaranteed by construction — and v4
is in fact *furthest* out on it.

## What the gate does not see

| | Real | v0 | v1 | v4 |
|---|---|---|---|---|
| Exact-zero fraction | 0.2298 | 0.0000 | 0.0000 | 0.2310 |
| Negative fraction | 0.0000 | 0.1234 | 0.0983 | 0.0000 |
| Minimum coordinate | 0.0000 | −0.0979 | −0.0974 | 0.0000 |
| Duplicate rows | 0.00062 | 0.0 | 0.0 | 0.0 |
| Effective rank | 27.99 | 27.42 | 27.50 | 28.18 |
| Median 5-NN distance | 0.5153 | 0.5142 | 0.5140 | 0.5073 |

v0 and v1 produce **zero** exact zeros and put 12.3% and 9.8% of their mass below
zero. v4 never goes negative and matches the zero mass to **0.0012** — tighter than
at 30k, where it overshot by 0.0143.

The descriptor glyph panel shows this directly, and is the only panel that can: the
real rows carry no negative bins, the v0 and v1 rows are shot through with them at
100k just as at 30k, and v4's are clean.

**What survives from the 30k reading**, narrower but still worth acting on: v0 is
the closest rung to real on relative contrast while being this far off-support. Four
statistics measuring neighbourhood geometry do not constrain support.

## EMA alone does nothing

With run length matched, v0→v1 isolates generator EMA — one config key, all else
equal. Against the measured 100k floor it moves LID 0.0622 (0.61x), hubness skew
0.0419 (0.34x) and IVF Gini 0.0033 (0.46x) — all under 1x, each smaller than a
reseed. **Relative contrast is the exception at 0.0095, which is 2.38x the floor**,
and it moves the wrong way: v1 lands further from real than v0.

On the 30k floor that term read 0.20x and this section said EMA moved nothing. The
measured floor turns the one statistic EMA affects into its only real effect, and
that effect is a regression. It does not help the support problem either.

At 30k the question was also confounded, because the ladder's v1 was a 100k run.

## Distribution diagnostics

Not the gate (`AGENTS.md` invariant 1). The corpus was split into two disjoint
20000-row halves: one is the reference, the other is scored against it the same way
and is the **floor**. Computed by `dist_diag.py`, equal-N at 20000, seed 42.

| Diagnostic | real vs real | v0 | v1 | v4 |
|---|---|---|---|---|
| Per-dimension W1, mean | 0.000678 | 0.003217 | 0.002285 | 0.001718 |
| Per-dimension W1, worst dim | 0.001777 | 0.007829 | 0.003765 | 0.003048 |
| Covariance Frobenius | 0.006382 | 0.017085 | 0.005804 | 0.005477 |
| MMD (RBF, γ=1) | 0.000430 | 0.000903 | 0.000284 | 0.000313 |
| Pairwise-distance hist L1 | 0.4050 | 0.2871 | 0.4100 | 0.2193 |

As a multiple of the floor:

| Diagnostic | v0 | v1 | v4 |
|---|---|---|---|
| Per-dimension W1, mean | 4.7x | 3.4x | **2.5x** |
| Per-dimension W1, worst dim | 4.4x | 2.1x | **1.7x** |
| Covariance Frobenius | 2.7x | **0.9x** | **0.9x** |
| MMD (RBF, γ=1) | 2.1x | **0.7x** | **0.7x** |
| Pairwise-distance hist L1 | **0.7x** | **1.0x** | **0.5x** |

**Two of the five are saturated.** Covariance Frobenius and MMD cannot separate v1 or
v4 from a disjoint sample of real SIFT.

**The pairwise-distance histogram is worse than saturated:** v0 (0.7x) and v4 (0.5x)
both score "better than real", which is not a coherent thing for a distance to do. It
is measuring sampling variation, not fidelity, and cannot be read at this resolution.

This is the "MMD improved, looks good" trap from `AGENTS.md`, made concrete: these
diagnostics have no resolution left, because every rung already matches real SIFT's
global second-order structure.

Only the per-dimension marginals retain resolution, and they rank **v4 best**, then
v1, then v0. Unlike at 30k, this lens now agrees with the support check. In the
correlation panel, v0's residual is pale over most of the matrix but carries a hard
cross near dim 95 — one dimension whose covariance is badly wrong, which is what puts
its Frobenius norm at 2.7x the floor while v1 and v4 sit on it.

### Three lenses, and where they now agree

| Question | Instrument | Winner |
|---|---|---|
| Neighbourhood geometry | the four gated statistics | v4 (2 of 3 usable); v0 keeps contrast |
| Support | exact zeros, negatives, glyphs | v4, uniquely |
| Distribution | per-dimension marginals | v4 |

At 30k the three lenses picked three different rungs. Matched at 100k they converge
on v4, and the ladder is doing what a ladder should. The dissent worth keeping is
relative contrast, where the plain baseline is closest to real — 0.0052 against v4's
0.0483 — and is the only gap on the page within shouting distance of its own noise
floor at 1.30x.

## What the lattice does to relative contrast

Relative contrast divides by the nearest-neighbour distance, and SIFT's quantized
lattice puts near-coincident pairs in the corpus, so a few real queries divide by
almost zero and reach ~6.5e7. Those points used to set the axis and collapse every
curve into the first bin; the panel now bins to the 99.5th percentile and states
what it left out — 130 of 79994 queries, 0.16%.

The tail belongs to the real corpus, not the generators. Six real queries were
dropped outright by `survivor_mask` as exact duplicates or full ties, and **every
synthetic rung dropped zero** — no generator here produces a single exact duplicate
or fully tied neighbourhood. The gated median is computed from the full array and
was never affected; only the histogram's axis was.

## What this does not say

- **v4 has no floor at this run length.** The v0/v1 multiples come from a 100k pair
  measured here; v4's architecture has none, and the 30k v3-pair spread is not a
  substitute. Its rows give gaps only. Producing one means a second-seed 100k v4.
- **The 100k floor is n=2.** One paired difference has one degree of freedom: it
  fixes the order of magnitude and nothing more, and it is a lower bound — two runs
  at identical seed still diverge 1–8% per loss column by step 250 from
  nondeterministic CUDA reduction order. Three to five seeds are needed before any of
  this justifies a band.
- **No band was set or moved.** `gates/sift.yaml` is untouched; every band is null.
- **v4's LID ranks nothing** — fitted by construction.
- **The real-vs-real floor is one split.** It resolves "saturated" from "clearly
  above" and nothing finer; the ordering of v1 and v4 on the saturated diagnostics
  should not be read.
- **Run length is not a variant.** The 100k v0 is a new run of the existing `v0`
  configuration, not a new rung.

## What follows

**Compare rungs at matched run length, long enough for the regularizers to act.** 30k
inverted the ranking of the two most distant rungs on this ladder. Any future rung
carrying a regularizer should be assumed to need the same headroom.

**The gate still needs a support check.** v0 sits nearer real on relative contrast
than any other rung while carrying 12.3% negative components and no exact zeros; that
combination should not look like a pass on any statistic. `exact_zero_fraction` and `negative_fraction`
are already computed by `eda_report`. Whether that becomes a fifth gated statistic or
a precondition is a banding decision, and belongs to a human.

## Figures

The overlay panels draw each set as a binned-density curve rather than overlapping
translucent bars, with `real` as the filled reference. Four overlapping bar series
were unreadable — the fills multiplied into mud and no single set could be followed.
Bins are unchanged, so every number a panel encodes is the same; only the mark
differs. Shipped in `src/eval/eda/figures.py`, so every family's report gets it.

## Artifacts on the box

`/workspace/keep/sift-v0-v1-v4/` on `tig-gpu`: `eda100k/` holds the stock
`eda_report.html` and twelve per-panel PNGs; `v0_x100k/` holds the newly trained
checkpoint beside the `run_config.yaml` it needs to be loadable; `samples_v0_100k.npy`
and `samples_v4_100k.npy` are the draws. v1's draw is the ladder's existing
`samples_v1.npy`, already a seed-42 draw from the 100k `x100k_ema_only` checkpoint.
