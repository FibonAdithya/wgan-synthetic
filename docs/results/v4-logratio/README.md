# v4 `structured_gated` + log-ratio regularizer — paired against v3

Run 2026-08-10 from commit `a86c2ba` on branch `sift/gan-v4`, both arms 30k steps
on one RTX 4060 via the `gpuq` queue. `jobs/gpuq_job_specs.txt` holds the three
job specs verbatim — the exact commands, lanes, timeouts and exit codes — so the
invocations are recoverable without access to the box.

Checkpoints (`best_generator.pt`) for both runs are at
`/workspace/keep/v34-sift1m/` on `tig-gpu`, beside the `run_config.yaml` each
needs to be loadable (`AGENTS.md` invariant 4). The 6.8MB `eda_report.html` is
there too. What is committed here is the metric and configuration record.

## Why v3 was re-run

`data/sift_base.npy`, the corpus v3's recorded numbers were measured against, is
gone from the box. v4 therefore had to train on `sift_1m.npy`, and comparing a
`sift_1m` v4 against a `sift_base` v3 would have confounded the regularizer's
effect with a change of corpus. Both arms were re-run from the same commit on the
same corpus instead. The v3 arm doubles as a replicate of the August run.

**The corpus substitution turned out to be a non-issue, and this run proves it.**
The `real` column measured here against `sift_1m.npy` reproduces the August
`sift_base.npy` report to all 16 digits:

| | this run | `docs/results/v3-structured/eda_v3_30k` |
|---|---|---|
| `lid_median` | 17.738288203443098 | 17.738288203443098 |
| `relative_contrast_median` | 2.267281799988775 | 2.267281799988775 |
| `hubness_skew` | 1.8839015220354034 | 1.8839015220354034 |
| `ivf_gini` | 0.30403671874999993 | 0.30403671874999993 |
| `duplicate_row_fraction` | 0.0006199999999999539 | 0.0006199999999999539 |

A seeded 20,000-row subsample of a *different* 1M corpus cannot do that.
`sift_1m.npy` is the same data as the missing `sift_base.npy`. The hedge written
into `configs/sift/noisefloor_a.yaml` and into this run's own config headers is
obsolete.

## Result

Canonical SIFT conditions — N=20000, k=100, hubness k=10, nlist=256 — from
`eda_v3_v4/summary.json`. Both arms were measured in a **single** `eda_report`
invocation, so they share one real-side subsample and the v3–v4 difference
carries no sampling noise of its own.

| Statistic | Real | v3 | v4 | gap v3 | gap v4 |
|---|---|---|---|---|---|
| LID median | 17.7383 | 10.7234 | 15.5465 | 7.0148 | **2.1918** |
| Relative contrast | 2.2673 | 3.1208 | 2.4140 | 0.8535 | **0.1467** |
| Hubness skew | 1.8839 | 0.8933 | 1.5984 | 0.9906 | **0.2855** |
| IVF cell-balance Gini | 0.3040 | 0.2478 | 0.2681 | 0.0563 | **0.0359** |

**v4 moves every statistic toward real.** That is the result, and it is the first
positive one this track has produced.

**It is not, however, the best rung on this gate.** See "Against the baseline"
below before quoting any of the above: measured against `v0`, the whole
structured-gate line is behind the plain WGAN-GP it was built to improve on.

### Is it noise?

Two independent estimates, and they disagree in a way that matters.

The seed-to-seed floor in `docs/datasets/sift.md` was measured on **v0**, a plain
WGAN-GP MLP — a different architecture. This run supplies an
**architecture-matched** estimate for free: the v3 arm re-ran the August v3
configuration, so the two v3 draws differ only by run-to-run nondeterminism.

| Statistic | v3→v4 improvement | ÷ v0 floor | ÷ v3-pair spread | verdict |
|---|---|---|---|---|
| LID median | 4.8230 | 29.4x | 51.9x | **fitted — not evidence** |
| Relative contrast | 0.7068 | 14.7x | 34.9x | clears both |
| Hubness skew | 0.7051 | 11.4x | 18.5x | clears both |
| IVF Gini | 0.0203 | 2.9x | **1.6x** | marginal — do not claim |

(v3-pair spread: LID 0.0930, contrast 0.0203, hubness 0.0382, Gini 0.0131.)

Note the architecture-matched estimate makes the IVF Gini column **weaker**, not
stronger, than the v0 floor implies. Gini is the one statistic here that should
not be used to rank the two arms.

### How to read it

**Discard LID.** v4 trains on the mean log-ratio profile of within-batch
neighbours, which is the sufficient statistic the Hill LID estimator reduces to a
scalar. A good `lid_median` was guaranteed by construction, and the v4 config said
so before the run. It is a check that the penalty worked, not evidence that it
helped.

The evidence is **relative contrast and hubness skew** — untouched by the penalty,
improved by 83% and 71%, at 35x and 18x the architecture-matched noise.

This is what v3's own write-up predicted. It read v3's uniform failure as one
cause rather than four: "the generator is not reproducing the local density
variation that makes real descriptor neighbourhoods hard." v4 targets exactly that
and all four statistics moved together. The hypothesis survived its test.

## Against the baseline

The comparison above is v3 against v4, which is what the pair was run to measure.
It says nothing about whether either beats the rung they descend from. Measured
2026-08-10, |gap to real|, `v0` sampled from `runs/long_baseline` at the same 30k
length as v3 and v4 (`eda_v0_v3_v4_30k_summary.json`):

| Statistic | v0 | v3 | v4 |
|---|---|---|---|
| LID median | **0.4386** | 7.0148 | 2.1918 |
| Relative contrast | **0.0061** | 0.8535 | 0.1467 |
| Hubness skew | **0.2041** | 0.9906 | 0.2855 |
| IVF cell-balance Gini | **0.0322** | 0.0563 | 0.0359 |

**`v0` is closer to real than either, on all four.** On LID -- the one statistic
the noise floor calls comfortably usable -- it is ahead of `v4` by 10.7x that
floor, so this is not a marginal call. No length confound: all three are 30k runs.

Widening to every rung (`eda_ladder_all_summary.json`) locates where it starts:

| Statistic | v0 | v1 | v1_5 | v2 | v3 | v4 |
|---|---|---|---|---|---|---|
| LID median | 0.44 | 0.29 | **0.27** | 2.95 | 7.01 | 2.19 |
| Relative contrast | **0.006** | 0.015 | 0.014 | 0.245 | 0.854 | 0.147 |
| Hubness skew | 0.20 | 0.21 | 0.27 | **0.06** | 0.99 | 0.29 |
| IVF Gini | 0.032 | 0.020 | 0.025 | **0.012** | 0.056 | 0.036 |
| exact-zero fraction | 0.000 | 0.000 | 0.000 | 0.152 | 0.179 | **0.244** |

Real exact-zero fraction is 0.230. `v1`, `v1_5` and `v2` are 100k runs against
`v0`, `v3` and `v4` at 30k, so read across that boundary with care -- but the
`v1_5` -> `v2` step is clean, both being 100k, and it is where the gate breaks:
LID gap 0.27 -> 2.95.

Read on its own that table says the dense rungs win every gate column while
producing *no exact zeros at all* against real SIFT's 23%. **That reading does
not survive matching the run lengths.** It was comparing an undertrained `v3`
and `v4` against fully trained dense rungs.

### At matched length

`configs/sift/v4_sift1m_x100k.yaml` was written to remove exactly that confound.
With `v4` at 100k, every rung below is a 100k run and the comparison is clean
(`eda_ladder_100k_summary.json`):

| Statistic | v1 | v1_5 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| LID median | 0.2886 | **0.2738** | 2.9478 | 4.4100 | 1.4026 |
| Relative contrast | 0.0146 | **0.0137** | 0.2449 | 0.4122 | 0.0592 |
| Hubness skew | 0.2119 | 0.2702 | 0.0608 | 0.5611 | **0.1069** |
| IVF Gini | 0.0198 | 0.0246 | 0.0118 | 0.0124 | **0.0042** |
| exact-zero gap | 0.2298 | 0.2298 | 0.0775 | 0.0388 | **0.0023** |

Against `v1_5`, the best dense rung, in the architecture-matched noise units
established above:

| Statistic | v1_5 | v4 | difference | verdict |
|---|---|---|---|---|
| LID median | 0.2738 | 1.4026 | 12.1x noise | **`v1_5` better** |
| Relative contrast | 0.0137 | 0.0592 | 2.2x | too close to call |
| Hubness skew | 0.2702 | 0.1069 | 4.3x | **`v4` better** |
| IVF Gini | 0.0246 | 0.0042 | 1.6x | too close to call |

One clear win each, not a rout. **And `v4` has not plateaued**: from 30k to 100k
every gate statistic improved by more than the noise floor -- LID -36%, contrast
-60%, hubness -63%, Gini -88%. Against `v2` at matched length it now wins three
of four, losing only hubness.

Its support match is close to exact: exact-zero fraction 0.2321 against real
0.2298, a gap 34x smaller than `v2`'s, and effective rank 28.109 against real
27.994, the closest of any rung.

### What the ladder actually shows

A trade, not a verdict. `v4` reproduces SIFT's *support* almost exactly and wins
hubness and IVF Gini; `v1_5` reproduces its *intrinsic dimensionality* far better
while emitting no exact zeros at all. LID being the one statistic the noise floor
calls comfortably usable tilts this toward `v1_5` -- but `v4` was still improving
at 100k, so the remaining gap is not demonstrably terminal.

The sharpest datum is this: **`v4` trains directly on LID's sufficient statistic
and still loses LID to a plain MLP by 12x the noise floor.** That points at an
architectural ceiling -- the gate mechanism itself costing intrinsic
dimensionality -- rather than a tuning problem, and it is the thing to attack
next if this line continues.

Duplicate-row fraction is 0.00000 for all five rungs against real SIFT's
0.00062. No continuous generator here has ever touched it.

The open question stands, and is not ours to settle: `AGENTS.md` invariant 1
makes ANN difficulty the gate and everything else a diagnostic. Read literally it
still prefers `v1_5`, on the strength of LID. But it is ranking a generator that
cannot emit a single exact zero above one that reproduces the support to 0.0023 --
either a real finding about what drives search difficulty, or evidence the four
statistics are not sufficient on their own.

## What this does not say

- **The corpus is still easier to search than real SIFT, on all four statistics.**
  Every one still points the same direction it did for v3 — lower LID, higher
  relative contrast, flatter hubness, lower Gini. v4 shrank the gap; it did not
  close it, and v4's LID gap is still 13x the v0 floor.
- **Duplicate rows are still exactly zero**, against real SIFT's 0.00062. Real
  descriptors are quantized to a lattice so exact duplicates occur; a continuous
  generator cannot produce them. The regularizer does not touch this and was never
  going to.
- **n=1 per arm.** One seed each. The improvements are large enough relative to
  both noise estimates that they are unlikely to be reseeds, but a paired
  comparison of two single runs is not a seed sweep. The open issue asking for 3–5
  seeds still applies before any of this justifies a band in `gates/sift.yaml`.
- **No band was set.** `gates/sift.yaml` is unchanged and every band there is
  still null.

## Supporting diagnostics

Not the gate — `AGENTS.md` invariant 1 — but they move consistently with it, and
the support statistics are where the structured gate was supposed to act.

| | Real | v3 | v4 |
|---|---|---|---|
| Exact-zero fraction | 0.229803 | 0.179097 | 0.244086 |
| Effective rank | 27.9939 | 22.3411 | 26.8097 |
| Median 5-NN distance | 0.515315 | 0.405476 | 0.489393 |
| Median pairwise distance | 1.099533 | 1.091138 | 1.098394 |

v4 slightly *overshoots* the exact-zero fraction (0.2441 against 0.2298) where v3
undershot it (0.1791); the absolute gap falls from 0.0507 to 0.0143.

Final training-loop diagnostics at step 30000 improve across the board — `cov_fro`
−71%, `mean_l2` −69%, `var_l2` −74%, `zero_fraction_gap` −79%, `nnz_std_gap` −64%.
These are distribution-matching diagnostics and are reported here only because
they corroborate the gate; on their own they would prove nothing, which is exactly
the trap v3 fell into.

One number that is *not* meaningful: v4's generator `adv_loss` settles near 0.40
against v3's 2.31. The critic-side quantities are nearly identical (`wasserstein`
0.0357 vs 0.0366, `d_loss` −0.0256 vs −0.0277, `gp` 0.0020 vs 0.0018), so this is
a level shift in a critic whose output has an arbitrary additive constant, not a
change in the discrepancy being measured.

## Reproducing

Both training arms, on the box:

    python -m src.train.train_wgan_gp --config configs/sift/v3_sift1m.yaml
    python -m src.train.train_wgan_gp --config configs/sift/v4_sift1m.yaml

Those are the paths *now*. At `a86c2ba`, the commit the runs were actually pinned
to, these files were `configs/sift_gan_v{3,4}_sift1m.yaml` — the SIFT ladder was
moved into `configs/sift/` when this branch was reconciled onto main, which had
already made that move for `v0`–`v2`. `jobs/gpuq_job_specs.txt` records the
original invocations unedited, so it names the old paths. The file contents are
unchanged by the move.

Then sample 20,000 from each at seed 7 and measure both in one report. Seed 7
matches `docs/results/v3-structured/logs/eval_v3.sh`, which is why the v3 column
here stays comparable to the August numbers. The full commands, including the
`device: cpu` rewrite that keeps sampling off the card, are in
`jobs/gpuq_job_specs.txt`.
