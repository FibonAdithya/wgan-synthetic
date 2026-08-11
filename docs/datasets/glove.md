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
        --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10

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

### These are L2 measurements of an angular corpus

`ann_difficulty.py` measures everything under L2, including this family's
`angular` corpus, so these numbers will need re-measuring once angular
distance support lands (phase (c)).

Two of the four will not move at all. On L2-normalized vectors the two
distances are related by a strictly monotone map, and hubness skew and Gini
read only neighbour identity and cluster assignment, which such a map cannot
reorder. That is an argument, not a measurement, and it holds for any corpus
preprocessed this way.

One draw measured against both distances agrees, and puts a size on the two
that do move:

| Statistic | under L2 | under angular | change |
|---|---|---|---|
| Hubness skew | 4.1839 | 4.1839 | none |
| IVF cell-balance Gini | 0.56835 | 0.56835 | none |
| Relative contrast | 1.38872 | 1.33333 | -3.99% |
| LID median | 35.2928 | 31.4624 | -10.85% |

Unlike the two tables above, these eight figures are not backed by anything
committed here: they came from a one-off script that is not in this tree, so
they cannot be reproduced from a pinned commit. Treat the two percentages as
indicative and re-measure them when phase (c) lands. The zeroes are the part
that does not need re-measuring, for the reason given above.

## Model family

`mlp` today, `spherical` when phase (b) lands.

## Ladder

| Variant | Delta | Config | Run | Status |
|---|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/glove/v0.yaml` | `runs/glove/v0_seed{42..46}` | trained -- n=5 seeds, see `## Noise floor` |

Train `v0`:

    python -m src.train.train_wgan_gp --config configs/glove/v0.yaml

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

Measured 2026-08-10 from five 30k-step `v0` runs, identical in every training
hyperparameter except seed (42 through 46), configs `configs/glove/v0_seed42.yaml`
through `configs/glove/v0_seed46.yaml`. Each run's checkpoint was sampled for
50,000 vectors at a *fixed* sampling seed of 42, so only the training seed
varies between the five, and `real` plus all five were measured together in a
single `eda_report` invocation under the canonical conditions above. The
result is committed as `docs/datasets/glove_v0_noise_floor.json`.

| Statistic | mean | spread across draws (n=5) | as % of mean | distance from real, in units of that spread |
|---|---|---|---|---|
| LID median | 16.438292 | 15.697354 -- 17.453755 | 10.68% | 10.6x |
| Relative contrast | 1.839151 | 1.781112 -- 1.895765 | 6.23% | 3.9x |
| Hubness skew | 1.695891 | 1.535497 -- 1.797769 | 15.47% | 15.0x |
| IVF cell-balance Gini | 0.262707 | 0.254021 -- 0.277913 | 9.09% | 13.3x |

The last column is the one that decides gateability: it compares the
seed-to-seed spread against how far `v0`'s mean sits from real. Relative
contrast has the smallest margin, at 3.9x its own spread; LID median, hubness
skew and IVF Gini clear it further still, at 10.6x, 15.0x and 13.3x. All four
separate cleanly from noise at `v0`'s current distance from real -- this
sweep does not disqualify any of the four statistics. That is a fact about
this measurement, not a recommendation to set a band: see below.

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

The accurate statement is that a hubness band is coarse, not useless: it can
only catch a generator whose deficit exceeds the real-side spread, which
`v0`'s does by a wide margin. A later rung that closed most of that gap would
land inside the real-side noise (3.463--8.331) and stop being judgeable by
this statistic alone -- indistinguishable there from a reseed of the real
corpus.

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
