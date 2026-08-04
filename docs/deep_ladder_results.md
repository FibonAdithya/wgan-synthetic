# DEEP variant ladder: ANN-difficulty results

## Setup

Three WGAN-GP variants ("rungs") were trained to synthesize 96-D descriptors
matching the `deep-image-96-angular`-like DEEP embedding distribution. Each
rung differs from the previous by exactly one config key:

| rung | delta from previous | key value |
|------|----------------------|-----------|
| v0   | baseline             | `training.spectrum_reg_alpha: 0.0`, `data.preprocess.whiten: false` |
| v1   | + covariance-spectrum regularizer | `training.spectrum_reg_alpha: 0.1` |
| v2   | + PCA whitening of the training space | `data.preprocess.whiten: true` |

The spectrum regularizer penalizes the L1 gap between the sorted,
trace-normalized eigenvalue spectra of the real and generated batches'
covariance matrices — i.e. it targets the shape of variance decay across
directions, which is the property most specific to a PCA-derived embedding
set like DEEP. Final logged `spectrum_reg` at step 30000: v0 = 0.0 (disabled
by design), v1 = 0.000351, v2 = 0.000778.

Fixed across all three rungs: `latent_dim: 128`, generator `[512, 1024,
1024]`, critic `[1024, 512, 256]`, `batch_size: 512`, `n_critic: 3`,
`lambda_gp: 5.0`, `ema_decay: 0.999`, `distance_reg_alpha: 0.0`,
`num_gen_steps: 30000`, `seed: 42`. Trained on `data/deep96_1m.npy`
(1,000,000 x 96 float32), one rung at a time on an RTX 4060, ~35 minutes per
run, all three reaching step 30000.

v2's training data lives in a PCA-whitened space, so its samples are drawn
through the inverting sampler in `src/deep/sample.py`, which undoes the
whitening before emitting vectors. `src/sample/generate.py` does not perform
this inversion and would emit vectors in whitened (not original DEEP)
coordinates — it was not used for v2. All three rungs are therefore compared
in the same, original DEEP coordinate space.

## Measured results

From `runs/eda_deep/summary.json`: 50,000 sampled vectors per variant
compared against 50,000 real DEEP vectors (`preprocess: l2`, `seed: 42`, ANN
settings `k=100, k_hub=10, max_rows=20000, nlist=256`).

| dataset | lid_median | relative_contrast_median | hubness_skew | ivf_gini |
|---------|------------|---------------------------|--------------|----------|
| real    | 17.561218  | 1.832256                  | 1.940139     | 0.304576 |
| v0      | 16.851780  | 1.868597                  | 1.918069     | 0.321115 |
| v1      | 16.804635  | 1.870771                  | 1.940117     | 0.297048 |
| v2      | 16.908998  | 1.867279                  | 1.839952     | 0.299153 |

## Which rung is closest to real DEEP

Absolute gap to real, primary metrics (LID, hubness skew, IVF gini):

| rung | \|Δ lid_median\| | \|Δ hubness_skew\| | \|Δ ivf_gini\| | \|Δ relative_contrast_median\| (secondary) |
|------|-------------------:|---------------------:|-----------------:|---------------------------------------------:|
| v0   | 0.709438            | 0.022070              | 0.016539          | 0.036341 |
| v1   | 0.756583            | 0.000022              | 0.007528          | 0.038515 |
| v2   | 0.652220            | 0.100187              | 0.005423          | 0.035023 |

(e.g. v1 LID gap = \|17.561218 − 16.804635\| = 0.756583; v1 gini gap =
\|0.304576 − 0.297048\| = 0.007528; same arithmetic for the rest.)

**Per-metric winner:**

- **LID:** v2 closest (gap 0.652), then v0 (0.709), then v1 (0.757). v2's
  margin over v0 (≈0.06) and v1 (≈0.10) is a meaningful fraction of the
  overall real-to-v0 gap, so this looks like a real effect rather than noise
  — but see the single-run caveat below.
- **Hubness skew:** v1 is essentially exact — gap 0.000022, i.e. v1's
  hubness skew (1.940117) matches real (1.940139) to the fourth decimal. v0
  is also close (gap 0.022). v2 is the outlier here (gap 0.100), notably
  worse than either v0 or v1.
- **IVF gini:** v2 closest (gap 0.005423), v1 a close second (gap
  0.007528), v0 clearly worst (gap 0.016539). The v1/v2 gap difference here
  (≈0.002) is small enough that it should not be read as a confident
  ordering between those two specifically.

**No single rung wins all three primary metrics.** v2 is closest on 2 of 3
(LID, IVF gini); v1 is closest on the third (hubness skew), and wins it by a
striking margin. Taking the ladder as a whole, v2 has the best overall
showing on the primary metrics, but the result is not a clean sweep —
whitening (v2) improves the two metrics tied to local geometry and index
structure (LID, IVF gini) while making hubness skew markedly worse than
either v0 or v1.

**Honesty about statistical confidence:** all three rungs share `seed: 42`
and there is exactly one run per rung. There is no way to separate a small
gap from run-to-run seed variance with this data. The LID differences
between rungs (spread ≈0.10, i.e. ~15% of the total real-to-v0 LID gap) are
plausibly larger than typical single-seed noise, so the v2 > v0 > v1
ordering on LID is reported with moderate confidence. The IVF gini gap
between v1 and v2 (0.007528 vs 0.005423, a difference of ~0.002) is small
enough that it should not be treated as a confirmed ranking between those
two rungs specifically — it could plausibly flip under a different seed.
The v0-vs-v1 and v0-vs-v2 gini gaps (both roughly 2-3x larger than the
v1-vs-v2 difference) are more likely to be real.

## Notable close match

v1's hubness skew (1.940117) essentially reproduces real DEEP's hubness skew
(1.940139) — a gap of 0.000022, two to three orders of magnitude tighter
than any other rung/metric pair in this table. This is consistent with the
covariance-spectrum regularizer targeting a global second-order property
(the shape of the eigenvalue spectrum) that plausibly correlates with
hub-formation behavior in ANN search. That said, this is a single run at a
single seed: it is not possible from this data alone to say whether the
regularizer *causes* the hubness-skew match or whether v1 simply landed
there by chance for this seed. A repeat run (different seed, same config)
would be needed to confirm the effect is not incidental.

## Bottom line

- On the primary metrics, **v2 (whitening) is closest to real DEEP most
  often** — winning LID and IVF gini — but does noticeably worse than v0 or
  v1 on hubness skew.
- **v1 (spectrum regularizer) produces the single most striking match to
  real DEEP** of any rung/metric pair, on hubness skew, though this is not
  confirmed as causal given only one run.
- **v0 (baseline) is not closest on any primary metric**, but it is not
  drastically worse than the other two either — its metrics all sit within
  the same rough range.
- With one run per rung, this ladder supports "whitening plus regularizer
  each help on different metrics, and neither is a strict improvement over
  the other" more than it supports a single confident overall ranking.
