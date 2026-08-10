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

Both training arms, from this commit, on the box:

    python -m src.train.train_wgan_gp --config configs/sift_gan_v3_sift1m.yaml
    python -m src.train.train_wgan_gp --config configs/sift_gan_v4_sift1m.yaml

Then sample 20,000 from each at seed 7 and measure both in one report. Seed 7
matches `docs/results/v3-structured/logs/eval_v3.sh`, which is why the v3 column
here stays comparable to the August numbers. The full commands, including the
`device: cpu` rewrite that keeps sampling off the card, are in
`jobs/gpuq_job_specs.txt`.
