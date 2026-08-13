# SIFT EDA: v0, v1 and v4 against the real corpus

Design for a comparison report over three SIFT ladder rungs. Written
2026-08-13. Non-authoritative working note, per `AGENTS.md` — it records the
reasoning, not the behaviour.

## The question

`docs/results/v4-logratio/` compared v4 against v3 and found the track's first
positive result. It did not look further down the ladder. Reading v0, v1 and v4
against real SIFT in one measurement answers a different question: does the
ladder monotonically approach real SIFT, and does the gate say what we think it
says?

It does not, and the gate does not.

## What the data already showed

`/workspace/keep/sift-ladder/eda/summary.json` holds a six-rung run at canonical
conditions. Restricted to the three rungs of interest:

| | real | v0 | v1 | v4 |
|---|---|---|---|---|
| LID median | 17.738 | 17.300 | 17.450 | 15.547 |
| Relative contrast | 2.2673 | 2.2734 | 2.2526 | 2.4140 |
| Hubness skew | 1.884 | 2.088 | 2.096 | 1.598 |
| IVF Gini | 0.3040 | 0.2718 | 0.2842 | 0.2681 |
| Exact-zero fraction | 0.2298 | 0.0000 | 0.0000 | 0.2441 |
| Negative fraction | 0.0000 | 0.1182 | 0.0983 | 0.0000 |

v0 — plain WGAN-GP, no EMA, no gated generator, no regularizer — is the closest
rung to real SIFT on LID and relative contrast. Its contrast gap is 0.13x the
seed-to-seed noise floor: indistinguishable from real. It achieves this while
emitting 11.8% negative components and not one exact zero, on a corpus whose
defining structural fact is heavy mass at exactly zero.

That is the report's subject. It is a finding about the gate, not an endorsement
of v0.

## Method

One `eda_report` invocation, real `sift_1m.npy` plus four overlays, at the
canonical SIFT conditions locked in `docs/datasets/sift.md` — N=20000, k=100,
hubness k=10, nlist=256, `--preprocess l2`, seed 42. A single invocation is
load-bearing: it gives every rung one shared real-side subsample, so
rung-to-rung differences carry no sampling noise of their own.

Three of the four sample sets already exist in `/workspace/keep/sift-ladder/`
as seed-42 draws. The fourth is new, and is the reason for the job.

### The fourth overlay

The ladder pins v1 to `runs/x100k_ema_only` — 100k generator steps — while v0
(`runs/long_baseline`) and v4 are both 30k. So the ladder's own v0→v1 step
confounds generator EMA with 3.3x more training. `runs/long_ema_only` is a 30k
EMA run that exists on the box, and adding it as `v1_30k` separates the two.

It is a control, not a ladder rung. Nothing here renames or repoints v1:
`AGENTS.md` reserves any change to what a variant number means for a human.

### Noise floors

Every claim is divided by a floor, and the floor depends on the architecture:

- **v0, v1, v1_30k** — the v0-architecture floor in `docs/datasets/sift.md`:
  LID 0.164, contrast 0.048, hubness 0.062, Gini 0.007.
- **v4** — the v3-pair spread from `docs/results/v4-logratio/`: LID 0.0930,
  contrast 0.0203, hubness 0.0382, Gini 0.0131.

Both are n=2 and establish an order of magnitude, nothing more. Both are lower
bounds: they vary the training seed, and two runs at identical seed still
diverge 1-8% per column by step 250 from nondeterministic CUDA reduction order.

v4's LID is discounted regardless of floor. `lid_reg` fits the mean log-ratio
profile of within-batch neighbours, which is the sufficient statistic the Hill
estimator reduces to a scalar, so a good `lid_median` was guaranteed by
construction. It checks the penalty worked; it is not evidence the rung helped.

## The argument

Four claims, in order:

1. **The gate ranks v0 first.** Two of four statistics, one of them within
   noise of real.
2. **v0 is not on SIFT's support.** Negative components, no exact zeros, no
   duplicate rows. Carried by the pooled-value and descriptor-glyph panels —
   glyphs are the only panel that can show a vector is not a plausible 4x4x8
   orientation histogram, which no aggregate can.
3. **v1 changes nothing measurable.** v0→v1 moves LID 0.15, which is 0.9x the
   floor. `v1_30k` says whether the 100k step count contributes anything.
4. **v4 trades gate numbers for support fidelity.** It is the only rung that
   reproduces the exact-zero mass, and it is measurably worse on LID and
   contrast than the rung with no structure at all.

## What this must not do

- **Set or move a gate band.** `gates/sift.yaml` stays untouched, every band
  still null. The report recommends; a human decides.
- **Claim v0 is the better generator.** The finding is that four aggregate
  statistics do not constrain support, so a rung can win them while being
  obviously wrong. That argues for adding a support check, not for shipping v0.
- **Rank rungs on IVF Gini.** The architecture-matched floor makes Gini the
  weakest discriminator of the four, and `docs/results/v4-logratio/` already
  ruled it out for the v3/v4 pair.
- **Rank rungs on v4's LID.** Fitted, per above.

## Deliverable

`docs/results/sift-v0-v1-v4/`, following the shape of `docs/results/v4-logratio/`:

| File | Contents |
|---|---|
| `README.md` | The record: conditions, table, per-claim noise arithmetic, caveats |
| `report.html` | Self-contained written report with the figures embedded |
| `summary.json` | Copied verbatim from the run |
| `job_spec.json` | The `gpuq` spec, so the invocation is recoverable |

The stock 6.8MB `eda_report.html` stays on the box and is cited by path, as
`v4-logratio` does with its own.
