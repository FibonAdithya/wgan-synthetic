# GloVe

100-dimensional GloVe word embeddings, dense and signed. The vectors come
from co-occurrence statistics over a text corpus, and the one structural
fact that decides how this family is modelled is the density gradient that
word frequency produces across the space.

## Source

    python -m src.data.fetch glove

Fetches `glove-100-angular` into the shared cache and writes
`data/glove_250k.npy` and `data/glove_1m.npy`. The HDF5 is large and
immutable; the fetcher downloads it once and is safe to run concurrently.

| | |
|---|---|
| Dimension | `100` |
| Search metric | `angular` |
| Upstream | `glove-100-angular` |

## Structure

100-dimensional word embeddings, dense and signed. Word frequency produces a
pronounced density gradient across the space, which is the mechanism that
generates hubs. Hubness skew is therefore the statistic this family is most
likely to fail, and the most informative one when it does.

## Measured profile

Read from the file rather than quoted from a paper. Canonical N and k are
locked here so a gate result stays readable against an older one.

| | |
|---|---|
| Canonical N | `20000` |
| Canonical k | `100` (`10` for hubness) |

| Statistic | Real | Synthetic (`v0`, mean of 5 seeds) |
|---|---|---|
| LID median | 35.077082 | 16.438292 |
| Relative contrast | 1.389420 | 1.839151 |
| Hubness skew | 5.639746 (see below) | 1.695891 (see below) |
| IVF cell-balance Gini | 0.580732 | 0.262707 |

Measured over 50,000 vectors at `preprocess: l2`, `seed: 42`, `nlist: 256`,
subsampled from `glove_250k.npy`. The report output these came from is
committed as `docs/datasets/glove_profile_summary.json`, and the noise-floor
draws below as `docs/datasets/glove_noise_floor.json`, so both of those tables
are checkable without access to the training box. The synthetic column is the
mean of `v0`'s five-seed sweep; see `## Noise floor` below for the per-seed
range, and `docs/datasets/glove_v0_noise_floor.json` for the committed
figures. Reproduce the real column with:

    python -m src.eval.eda_report \
        --real-path data/glove_250k.npy \
        --output-dir runs/glove/profile \
        --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular

Read the four values out of runs/glove/profile/summary.json (written by the command above).

### Hubness skew is below the noise floor at this N

**Do not read the 5.639746 above as a property of GloVe.** Eight independent
20,000-row draws from the same real corpus, changing nothing but which rows
were drawn, give:

| Statistic | mean | spread across draws | as % of mean |
|---|---|---|---|
| LID median | 35.1238 | 35.0318 -- 35.2086 | 0.50% |
| Relative contrast | 1.38951 | 1.38754 -- 1.39201 | 0.32% |
| Hubness skew | 4.4976 | 3.4630 -- 8.3308 | 108.2% |
| IVF cell-balance Gini | 0.59324 | 0.58157 -- 0.60339 | 3.68% |

Those draws are committed as `docs/datasets/glove_noise_floor.json`. Hubness
skew is a third moment of the k-occurrence distribution, so a handful of
extreme hubs in the tail set it; at N=20000 with `k_hub` 10, whether a few
land in the draw moves it by more than a factor of two. The other three are
stable enough to gate on.

This is a problem for this family in particular, because the structural
section above names hubness as the statistic GloVe is most likely to fail and
the most informative when it does. Choosing what to do about it -- raising N
for the hubness pass, or adding a hub statistic that is not a third moment --
changes locked measurement conditions or the gate's contents, so it needs a
human. See the `## Gate` section. The synthetic side's own hubness spread --
a separate measurement, from a five-seed `v0` sweep rather than a real-data
subsample -- is in `## Noise floor` below.

### Measured under this corpus's own metric

`src/eval/ann_difficulty.py` measures this family under its `data.metric`, which is
`angular`: L2 between unit-norm rows. On the unit sphere Euclidean distance
is a strictly increasing function of cosine distance, so it ranks neighbours
identically -- the corpus is measured under the distance it is searched with.
Measuring requires `--preprocess l2`, and `ann_difficulty.compute` refuses
rows that are neither unit-norm nor exactly zero rather than normalizing
them itself -- an exact zero is what `maybe_l2_normalize` leaves behind, so
it is accepted rather than treated as a caller mistake.

Two of the four will not move at all. On L2-normalized vectors the two
distances are related by a strictly monotone map, and hubness skew and Gini
read only neighbour identity and cluster assignment, which such a map cannot
reorder. That is an argument, not a measurement, and it holds for any corpus
preprocessed this way.

One draw, measured before this branch shipped its own `angular`, compares L2
against a different angular definition -- not the chord distance
`ann_difficulty.compute` implements now. It is a historical record, not a
reproduction target, and it puts a size on the two statistics a monotone map
cannot fix:

| Statistic | under L2 | under angular | change |
|---|---|---|---|
| Hubness skew | 4.1839 | 4.1839 | none |
| IVF cell-balance Gini | 0.56835 | 0.56835 | none |
| Relative contrast | 1.38872 | 1.33333 | -3.99% |
| LID median | 35.2928 | 31.4624 | -10.85% |

Unlike the two tables above, these eight figures are not backed by anything
committed here: the script that produced them is not in this tree, so they
cannot be reproduced from a pinned commit, and its measurement conditions are
otherwise unknown. Two of its four figures fall outside the committed noise
floor above -- the LID median of 35.2928 sits above the eight-draw range of
35.0318--35.2086, and the IVF cell-balance Gini of 0.56835 sits below the
eight-draw range of 0.58157--0.60339 -- so this draw was not taken under the
canonical measurement conditions, and should not be read as a draw comparable
to the tables above. They are also not reproducible with `--metric angular`
as shipped: on unit-norm rows that flag returns the L2 column exactly, for
the reason given above, not the separate angular column this one-off
recorded. The two changed figures are consistent with the script's angular
definition having been geodesic (`arccos`) rather than cosine -- cosine
exactly halves LID on unit-norm rows, a larger and differently-shaped move
than -10.85%, while a geodesic definition drops LID by a comparable amount on
synthetic isotropic unit vectors. Consistent, not confirmed: the script
cannot be inspected. The zeroes are the part that does not need re-measuring,
for the reason given above.

## Model family

`mlp` today, `spherical` not yet adopted.

## Ladder

| Variant | Delta | Config | Run | Status |
|---|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/glove/v0_seed42.yaml` through `configs/glove/v0_seed46.yaml`, five-seed instruments of `configs/glove/v0.yaml` | `runs/glove/v0_seed{42..46}` | trained -- n=5 seeds, see `## Noise floor` |
| `v1` | `training.spectrum_reg_alpha: 5.0` | `configs/glove/v1_seed42.yaml` through `configs/glove/v1_seed46.yaml`, five-seed instruments of `configs/glove/v1.yaml` | `runs/glove/v1_seed{42..46}` | trained -- n=5 seeds, see `## v1: the covariance-spectrum regularizer` |

Train a rung:

    python -m src.train.train_wgan_gp --config configs/glove/v0.yaml
    python -m src.train.train_wgan_gp --config configs/glove/v1.yaml

## Gate

`gates/glove.yaml` is the gate. The bands live there rather than in this
prose so a program can read them, and this section does not repeat the
numbers: two copies of a threshold is one copy too many.

Pass bands are per statistic, not a combined score, because the four fail in
different directions. A set can look too easy on relative contrast while being
too clustered on Gini, and a single score would average that away instead of
naming it. The gate file also pins the measurement conditions the bands were
set under, since these statistics are not comparable across different N, k or
nlist.

Every band is currently null. Bands are set once this family has a trained
ladder to show what is achievable; until then the gate file records that they
are unset, and the checker says so instead of passing.

When they are set, `hubness_skew` is the one to be careful with, though not
for the reason once written here. The measured profile above shows its
subsample noise at the canonical N spans 3.46 to 8.33 on the real corpus
alone. That does not make a band on it useless, only coarse: the `v0`
five-seed sweep in `## Noise floor` below put `v0`'s own hubness skew at
1.535 to 1.798, entirely clear of the real-side range, so a band admitting
3.46--8.33 would reject `v0` decisively rather than pass it. A band this wide
can only catch a generator whose hubness deficit exceeds the real corpus's
own noise floor -- which is what happened here -- and would go blind again
against a later rung that closed most of that gap and landed inside
3.46--8.33, indistinguishable there from a reseed of real itself.

Check a run against it:

    python -m src.eval.check_gate --dataset glove --run-dir runs/glove/profile

It reads `summary.json` from that run directory, prints a JSON verdict, and
exits non-zero when the run fails -- or, as now, when the bands are still
unset, which is verdict `unset` and exit code 2. Pass `--allow-unset` to get
the report without the non-zero exit, and `--stats-name <label>` to check a
synthetic series rather than `real`.

## Noise floor

How far each statistic moves when *nothing* changes but the training seed. A
band tighter than this spread is unenforceable, and a ladder rung whose
improvement is smaller than it is indistinguishable from a reseed.

`docs/datasets/sift.md#noise-floor` is this measurement's methodological
precedent: it ran the same seed-reseed test at n=2 and said three to five
seeds were needed before any of its numbers could justify a band. This
sweep, at n=5, is that follow-through.

Measured 2026-08-10 from five 30k-step `v0` runs, identical in every training
hyperparameter except seed (42 through 46), configs `configs/glove/v0_seed42.yaml`
through `configs/glove/v0_seed46.yaml`. Each run's checkpoint was sampled for
50,000 vectors at a *fixed* sampling seed of 42, so only the training seed
varies between the five, and `real` plus all five were measured together in a
single `eda_report` invocation under the canonical conditions above. The
result is committed as `docs/datasets/glove_v0_noise_floor.json`.

| Statistic | mean | spread across draws (n=5) | as % of mean | distance from real, in units of the training-seed spread |
|---|---|---|---|---|
| LID median | 16.438292 | 15.697354 -- 17.453755 | 10.68% | 10.6x |
| Relative contrast | 1.839151 | 1.781112 -- 1.895765 | 6.23% | 3.9x |
| Hubness skew | 1.695891 | 1.535497 -- 1.797769 | 15.47% | 15.0x |
| IVF cell-balance Gini | 0.262707 | 0.254021 -- 0.277913 | 9.09% | 13.3x |

The last column's denominator is the **training-seed spread measured in this
sweep** -- how far a statistic moves when only the training seed changes and
the sample stays fixed. That is the right yardstick for one specific
question: could a later ladder rung's improvement be told from a reseed of
the same generator. By it, all four clear a wide margin: relative contrast at
3.9x its own spread, LID median at 10.6x, hubness skew at 15.0x, IVF Gini at
13.3x. None of `v0`'s five draws, on any statistic, is close enough to real to
be mistaken for a reseed -- that is a fact about this measurement, not a
recommendation to set a band: see below.

Training-seed spread is not the spread a gate *band* is judged against,
though, and the two must not be conflated. `gates/glove.yaml` bands reject a
generator by comparing it to draws of the *real* corpus, so the denominator
that decides whether a statistic can carry a band is the real-side subsample
spread measured in `## Hubness skew is below the noise floor at this N`
above -- a different noise source, measured by a different procedure. For
LID median, relative contrast and IVF Gini the distinction is academic: both
spreads are small relative to `v0`'s gap from real, so either denominator
gives the same verdict. It is not academic for hubness skew. Measured
against the real-side range (3.463--8.331, a spread of 4.8678), `v0`'s gap
from the single real draw this page's tables call "Real" is 3.9439 --
**0.81x** of that range, and 0.58x against the real-side mean (4.4976 across
eight draws) instead. Both are under 1x, which by `docs/datasets/sift.md`'s
own convention is the bolded "noise exceeds signal" case. Hubness's 15.0x in
the table above is a true number about training-seed noise; it is not
evidence that hubness clears the real-side noise floor, and quoted alone it
says the opposite of the truth. See "Hubness skew is coarse, not useless"
below for what the real-side comparison does and does not license.

`v0` misses real on all four statistics, every one in the direction that
makes the synthetic set easier to search: LID median low (16.438292 against
35.077082), relative contrast high (1.839151 against 1.389420 -- higher means
neighbours stand out more, i.e. easier), hubness skew low (1.695891 against
5.639746), and IVF Gini low (0.262707 against 0.580732). The `## Structure`
section above names hubness skew as the statistic this family is most likely
to fail, because the density gradient that produces real hub structure is
exactly what an undifferentiated `mlp` generator smooths away; that held. The
same smoothing is why relative contrast overshoots rather than undershoots: a
generator that has not learned the density gradient spreads its mass more
evenly than the real corpus does, so the typical neighbour ends up standing
out from the nearest one more than it should. LID median and IVF Gini
undershoot too, for the same underlying reason -- an evenly smoothed density
is too easy to search on every axis measured here, not only the one the
structural section calls out.

### Hubness skew is coarse, not useless

This is the statistic that needed the most care, because the result here
partly corrects a claim made in `## Gate` and in `gates/glove.yaml`'s
`hubness_skew` comment: that any band wide enough not to reject real data
against itself would pass nearly any generator. That is too strong. The
real-side floor's eight draws (`docs/datasets/glove_noise_floor.json`) span
3.463 to 8.331; `v0`'s five draws above span 1.535 to 1.798. **The two ranges
do not overlap**, so a band admitting the real range would reject `v0`
decisively rather than pass it.

That margin is worth sizing, because it is what makes the separation fragile
rather than comfortable: the gap between the two ranges is 3.463 - 1.798 =
1.6653, which is 0.342 of the real-side range (4.8678). With only eight real
draws underestimating the true real-side spread, a margin of a third of the
range is not a large one.

**Hubness distinguishes a generator this far off, and cannot resolve one
much closer.** Both halves of that are true at once and neither stands in
for the whole. The accurate statement is that a hubness band is coarse, not
useless: it can only catch a generator whose deficit exceeds the real-side
spread, which `v0`'s does by a wide margin. A later rung that closed most of
that gap would land inside the real-side noise (3.463--8.331) and stop being
judgeable by this statistic alone -- indistinguishable there from a reseed of
the real corpus.

**This is n=5 on the synthetic side.** Five points bound the floor's order of
magnitude with more confidence than a single paired difference would, but
they are still five points; treat the ranges above as bounds on the floor's
width, not as its exact shape.

**This is a different measurement from the real-side floor.** The floor in
`docs/datasets/glove_noise_floor.json` holds the generator fixed (there was
none, at the time it was measured) and resamples which 20,000 rows are drawn
from the real corpus. This one holds the sample fixed (one fixed sampling
seed, 50,000 vectors per run) and reseeds training instead. Subsampling noise
and seed-to-seed training variance are different sources of variance,
measured by different procedures, which is why the two are reported as
separate tables above rather than merged into one.

No band is set from this measurement. Every band in `gates/glove.yaml`
remains null: setting one is reserved for a human working from a full ladder,
not from `v0` alone.

Reproduce with. First, train each of the five seeds:

    for seed in 42 43 44 45 46; do
        python -m src.train.train_wgan_gp --config configs/glove/v0_seed${seed}.yaml
    done

then sample each run's `best_generator.pt` for 50,000 vectors at the fixed
sampling seed:

    for seed in 42 43 44 45 46; do
        python -m src.sample.generate \
            --checkpoint runs/glove/v0_seed${seed}/best_generator.pt \
            --config configs/glove/v0_seed${seed}.yaml \
            --num-samples 50000 --seed 42 \
            --output-path runs/glove/v0_seed${seed}/samples.npy
    done

then measure `real` and all five seeds in one `eda_report` invocation:

    python -m src.eval.eda_report \
        --real-path data/glove_250k.npy \
        --synthetic-path v0_seed42=runs/glove/v0_seed42/samples.npy \
        --synthetic-path v0_seed43=runs/glove/v0_seed43/samples.npy \
        --synthetic-path v0_seed44=runs/glove/v0_seed44/samples.npy \
        --synthetic-path v0_seed45=runs/glove/v0_seed45/samples.npy \
        --synthetic-path v0_seed46=runs/glove/v0_seed46/samples.npy \
        --output-dir runs/glove/v0_sweep \
        --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10

then difference the five labelled series against `real`:

    python -m src.eval.noise_floor \
        --summary runs/glove/v0_sweep/summary.json \
        --series v0_seed42 --series v0_seed43 --series v0_seed44 \
        --series v0_seed45 --series v0_seed46 \
        --output docs/datasets/glove_v0_noise_floor.json

## v1: the covariance-spectrum regularizer

`v1` is `v0` plus `training.spectrum_reg_alpha: 5.0`, and nothing else
(`tests/test_glove_configs.py` checks this). It was chosen from two
single-change candidates measured side by side at five seeds each.

### Why this delta

A comparison of `v0`'s samples against the corpus on the unit sphere
(`v0_seed42`, 50,000 rows each; not committed, measured 2026-09-16) showed
that `v0` learns GloVe's shared mean direction -- the mean unit vector has norm
0.364 against the corpus's 0.355, and the two directions have cosine 0.99 --
but spreads its variance over too few directions: participation ratio 50
against 88, and the top 50 eigenvalues hold 81% of the variance against 60%.
Its nearest neighbours are correspondingly too close (median cosine 0.74
against 0.55). The deficit is dimensionality, not the mean offset, so both
candidates target dimensionality:

- `spectrum_reg` matches the normalized eigenvalue spectrum of the generated
  batch to the real one. It has no term for LID, contrast, hubness or Gini,
  so where those move, the movement is evidence. Its alpha, 5.0, is the value
  DEEP's sweep found binding (`docs/datasets/deep.md`); it was not tuned here.
- `lid_reg` matches the mean log-ratio profile of within-batch neighbours,
  LID's sufficient statistic, so its LID is fitted rather than evidence. Its
  alpha, 0.01858, puts the penalty at 2.5% of `v0`'s mean |adv_loss| (0.538
  over steps 20,000-30,000) at the gap `tools/probes/lid_reg_scale_probe.py`
  measured between trained `v0` and real: 0.7238 against a real-vs-real
  floor of 0.0322. This is the same share `configs/sift/v4.yaml` chose.

### Five-seed result

Measured 2026-09-17 on the RTX 3060 Ti box, under the canonical conditions
(N 20000, k 100, k_hub 10, nlist 256). Seeds 43-46 of all three families
were trained at commit `b17bd5f`; seed 42 of `v1` and `lid_reg` at `5774227`,
whose seed-42 configs differ from `b17bd5f`'s only in comments; and
`v0_seed42` at `f0b47ec`. The data, model, training and sampling code
(`src/data/`, `src/models/`, `src/train/`, `src/sample/generate.py`,
`src/device.py`) is identical across the three commits; `f0b47ec` also carries
benchmark tooling the other two lack, none of it on the training path. Every
run was sampled for 50,000 vectors at sampling seed 42, and `real` and all
fifteen series were measured in one `eda_report` invocation. The report is
committed as `docs/datasets/glove_reg_sweep_summary.json`, and each family's
seed spread as `docs/datasets/glove_v1_noise_floor.json`,
`docs/datasets/glove_lidreg_probe_noise_floor.json` and
`docs/datasets/glove_v0_new_box_noise_floor.json`.

**Series names.** The `v1` seeds were trained as `probe_spectrum_seed42`
through `probe_spectrum_seed46` and renamed to `v1_seed42` through
`v1_seed46` afterwards; only the file name and `output_dir` changed. The
committed JSONs are byte-for-byte what the measurement produced, so they
still label the `v1` series `probe_spectrum_seed<N>` and the `lid_reg` series
`probe_lidreg_seed<N>`.

The real column is the eight-draw subsample floor from
`docs/datasets/glove_noise_floor.json` (measured 2026-08-10), because a
generator is judged against draws of the corpus, not one draw. Synthetic
cells are mean ± sample standard deviation over five seeds, with the
min--max range and how many seeds fall inside the real range.

| Statistic | real, 8 draws: mean (range) | `v0` | `v1` | `lid_reg` probe |
|---|---|---|---|---|
| LID median | 35.1238 (35.0318--35.2086) | 18.1422 ± 2.2867 (16.5308--22.0032), 0/5 | 37.4090 ± 1.1781 (36.0803--38.6250), 0/5 | 33.6437 ± 0.2491 (33.3343--33.8552), 0/5, fitted |
| Relative contrast | 1.3895 (1.3875--1.3920) | 1.7623 ± 0.0828 (1.6393--1.8475), 0/5 | 1.3647 ± 0.0086 (1.3572--1.3744), 0/5 | 1.3974 ± 0.0026 (1.3940--1.4001), 0/5 |
| Hubness skew | 4.4976 (3.4630--8.3308) | 1.8805 ± 0.2927 (1.6210--2.3560), 0/5 | 4.2606 ± 0.3036 (3.9470--4.6353), 5/5 | 3.4521 ± 0.1671 (3.2793--3.6797), 2/5 |
| IVF cell-balance Gini | 0.5932 (0.5816--0.6034) | 0.2730 ± 0.0385 (0.2367--0.3369), 0/5 | 0.5968 ± 0.0221 (0.5690--0.6289), 2/5 | 0.5508 ± 0.0294 (0.5049--0.5843), 1/5 |

What the table supports:

- **`v1` moves all four statistics from far off to near the corpus.** Its
  gap to the real mean, as a share of the real mean, is +6.51% on LID,
  -1.78% on contrast, -5.27% on hubness and +0.61% on Gini, against `v0`'s
  -48.35%, +26.83%, -58.19% and -53.98%.
- **Hubness is inside the real range for every `v1` seed; Gini is inside on
  the mean but for only two of five seeds.** Gini's `v1` spread (sd 0.0221)
  is wider than the real range (0.0218 wide), so single seeds land on either
  side of it.
- **`v1` misses LID and contrast, in the harder-than-real direction.** LID is
  too high and contrast too low for all five seeds. The real LID and contrast
  ranges are narrow (0.5% and 0.3% of their means), so a miss of this size is
  still well outside them.
- **`lid_reg` misses in the easier-than-real direction** on contrast (too
  high), hubness (too low) and Gini (too low). It is closer to real on
  contrast than `v1`, and its LID is not evidence.
- **`v1` and `lid_reg` are distinguishable** on LID (+3.77, 4.4 pooled sd),
  contrast (-0.0327, 5.2 pooled sd) and hubness (+0.81, 3.3 pooled sd), where
  their five-seed ranges do not overlap. On Gini they are not: +0.0461, 1.8
  pooled sd, with overlapping ranges.

`v1` was chosen over `lid_reg` because it reaches the corpus on the two
statistics tied to hub structure without being trained on either, and
because its remaining misses make the synthetic set harder to search, not
easier. It does not meet the corpus on LID or contrast. No gate band is set
from this; see `## Gate`.

### v0 measures differently on the new box

The `v0` column above is not the `v0` sweep in `## Noise floor`. That one was
measured on 2026-08-10 on an earlier instance of the training box, whose card,
driver and torch build this page does not record; this one ran on an RTX 3060
Ti with driver 580.178.04 and torch 2.13.0+cu130. The two disagree on LID (old
mean 16.4383, range 15.6974--17.4538; new mean 18.1422, range
16.5308--22.0032) and contrast (old mean 1.8392, range 1.7811--1.8958; new
mean 1.7623, range 1.6393--1.8475). The environments differ in more than one
way, so the cause is not known. The old-box figures stay committed as measured; compare `v1`
against the new-box `v0` in the table above, which was measured with it.

### Reproduce

Train each seed of each family, for example:

    for seed in 42 43 44 45 46; do
        python -m src.train.train_wgan_gp --config configs/glove/v1_seed${seed}.yaml
        python -m src.sample.generate \
            --checkpoint runs/glove/v1_seed${seed}/best_generator.pt \
            --config configs/glove/v1_seed${seed}.yaml \
            --num-samples 50000 --seed 42 \
            --output-path runs/glove/v1_seed${seed}/samples.npy
    done

and the same for `configs/glove/v0_seed<N>.yaml` and
`configs/glove/probe_lidreg_seed<N>.yaml`. Train one seed at a time:
`train_wgan_gp` takes an exclusive per-card lock, so a second run on the same
card waits and fails after `gpu_lock_timeout_s` (1800 s). Then measure all
fifteen series in one invocation, labelling them as the committed JSONs do:

    python -m src.eval.eda_report \
        --real-path data/glove_250k.npy \
        --synthetic-path v0_seed42=runs/glove/v0_seed42/samples.npy \
        --synthetic-path probe_spectrum_seed42=runs/glove/v1_seed42/samples.npy \
        --synthetic-path probe_lidreg_seed42=runs/glove/probe_lidreg_seed42/samples.npy \
        ... (the same three for seeds 43-46) \
        --output-dir runs/glove/reg_sweep \
        --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular

and difference each family against `real`:

    python -m src.eval.noise_floor --summary runs/glove/reg_sweep/summary.json \
        --series probe_spectrum_seed42 --series probe_spectrum_seed43 \
        --series probe_spectrum_seed44 --series probe_spectrum_seed45 \
        --series probe_spectrum_seed46 \
        --output docs/datasets/glove_v1_noise_floor.json
