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
land in the draw moves it by more than a factor of two. The other three
looked stable enough to gate on at this N and draw count. `## Hub statistic
stability` below qualifies that for IVF Gini: its own range grows with N
rather than shrinking on a wider sweep, reaching 19.45% on GloVe and clearing
20% on DEEP, so a band set from the 3.68% figure above would be tighter than
the statistic actually supports once N or corpus changes.

This is a problem for this family in particular, because the structural
section above names hubness as the statistic GloVe is most likely to fail and
the most informative when it does. Choosing what to do about it -- raising N
for the hubness pass, or adding a hub statistic that is not a third moment --
changes locked measurement conditions or the gate's contents, and that choice
has now been made from a measurement rather than argument. See
`## Hub statistic stability` below for the sweep and its answer: `hubness_gini`
qualifies at the locked N and `hubness_skew` does not, at any N tried. The
synthetic side's own hubness spread -- a separate measurement, from a
five-seed `v0` sweep rather than a real-data subsample -- is in
`## Noise floor` below.

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
| `v0` | plain WGAN-GP | `configs/glove/v0_seed42.yaml` through `configs/glove/v0_seed46.yaml`, five-seed instruments of `configs/glove/v0.yaml` | `runs/glove/v0_seed{42..46}` | trained -- n=5 seeds, see `## Noise floor` |

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

**Confirmed, not overturned, at sixteen draws.** The stability sweep in
`## Hub statistic stability` below repeats this comparison at the locked N
with twice the draws and finds the same non-overlap: real 3.41340--10.33706
against `v0` 1.55003--1.90223. What sixteen draws add is a size on how thin
that margin is: the gap between the two ranges is 0.2x the real-side range's
own width, thinner than the eight-draw estimate above suggested, where the
sweep's replacement candidate, `hubness_gini`, manages 14.1x on the same pair
of real and `v0` ranges. That is the case for changing which statistic
carries the band, not for trusting this one at a wider setting -- see
`## Hub statistic stability` for the full comparison and the rule it was
decided under.

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

## Hub statistic stability

Issue #29 named four fixes for the hubness problem above and said choosing
between them needed a measurement nobody had taken. This is that measurement:
two corpora (GloVe and DEEP), four N (20,000 / 50,000 / 100,000 / 250,000),
16 draws per cell, `--backend torch`, at commit `8a90daf`. It answers, for
GloVe's canonical N, which of three hub statistics -- the incumbent
`hubness_skew` and two candidates, `hubness_gini` and `hub_share_top1pct` --
can carry a gate band at all.

The rule was fixed before the sweep ran and is not revisable after seeing
numbers (`docs/superpowers/specs/2026-08-13-glove-hub-statistic-stability-design.md#the-pre-registered-rule`).
A statistic qualifies at a given N when both hold: its real-side
`range_pct_of_mean` across the 16 draws is at most **10.0%** (stable), and
`|mean(real) - mean(v0)|` is at least **1.0x** the real-side range (max -
min) (discriminating). The tie-break, also pre-registered: take the cheapest
qualifying cell -- if anything qualifies at the locked N=20,000, adopt it and
leave canonical conditions alone.

### GloVe: three hub statistics, four N

| N | statistic | range % of mean | separation | draws disjoint | verdict |
|---|---|---|---|---|---|
| 20,000 | `hubness_skew` | 151.33 | 0.42x | yes | rejected |
| 20,000 | `hubness_gini` | 1.56 | 15.43x | yes | **qualified** |
| 20,000 | `hub_share_top1pct` | 10.46 | 4.39x | yes | rejected |
| 50,000 | `hubness_skew` | 73.89 | 0.81x | yes | rejected |
| 50,000 | `hubness_gini` | 0.58 | 40.78x | yes | qualified |
| 50,000 | `hub_share_top1pct` | 3.68 | 12.30x | yes | qualified |
| 100,000 | `hubness_skew` | 40.57 | 1.49x | no | rejected |
| 100,000 | `hubness_gini` | 0.44 | 53.51x | no | provisional |
| 100,000 | `hub_share_top1pct` | 2.41 | 18.65x | no | provisional |
| 250,000 | `hubness_skew` | 21.33 | 2.81x | no | rejected |
| 250,000 | `hubness_gini` | 0.30 | 76.35x | no | provisional |
| 250,000 | `hub_share_top1pct` | 2.43 | 18.56x | no | provisional |

Full artifact: `docs/datasets/glove_hub_stability.json`.

**One statistic qualifies at the locked N, so the tie-break stops there.**
`hubness_gini` is stable (1.56% against the 10.0% bar) and discriminating
(15.43x) at N=20,000. Because the pre-registered tie-break takes the
cheapest qualifying cell, canonical measurement conditions are not touched.
`hubness_skew` is rejected at every N, and raising N does not rescue it on
this corpus: its range falls from 151.33% at 20,000 to 21.33% at 250,000, but
its cv sits near 35% at every scale it is measured in (see the cv table
below), so the instability is a property of the statistic on this corpus,
not a small-sample artifact that a bigger draw would fix. `hub_share_top1pct`
is rejected at the locked N by a narrow margin -- 10.46% against the 10.0%
bar -- and qualifies at 50,000.

**At N=100,000 and 250,000 the sixteen draws no longer fit GloVe's 1M pool
without overlap.** Sixteen disjoint draws exhaust the pool at N=62,500; at
100,000 they need 1.6M rows and at 250,000 they need 4M, so the draws share
rows and their measured spread is a **lower bound** on the true subsample
spread. That biases the sweep toward finding statistics stable at large N --
the direction that would wrongly favour "raise N" as the fix -- which is why
every verdict at those two N above is reported `provisional` rather than
`qualified`, and why a provisional pass is not grounds for moving canonical N
on its own.

**`hub_share_top1pct`'s rejection at N=20,000 turns on which noise metric the
rule uses.** `range_pct_of_mean` grows with the number of draws sampled,
because a wider draw is more likely to catch an extreme one; `cv_pct` does
not. At the eight draws the provenance cell used, `hub_share_top1pct`
measured 7.23% and would have passed the 10.0% bar; at the sixteen draws
this sweep used, it measured 10.46% and failed. The rule was applied exactly
as pre-registered, on the draw count the sweep actually ran, and the outcome
does not change: `hubness_gini` is the better statistic on every metric
regardless, cv 0.53% against `hub_share_top1pct`'s 2.72% at N=20,000. But
this particular rejection is a real artifact of range-over-cv, not a wide
margin, and is worth stating plainly rather than leaving as a clean 10.0%
cutoff would imply.

### DEEP: condition 1 only

DEEP has no synthetic series in this study, so it votes on whether a
statistic's instability is a property of the statistic or of a
high-hubness corpus -- it is evidence about generality, not a second vote
on GloVe's gate.

| N | statistic | range % of mean | cv % | draws disjoint | verdict |
|---|---|---|---|---|---|
| 20,000 | `hubness_skew` | 22.13 | 5.45 | yes | unstable |
| 20,000 | `hubness_gini` | 1.25 | 0.31 | yes | stable |
| 20,000 | `hub_share_top1pct` | 6.60 | 1.88 | yes | stable |
| 50,000 | `hubness_skew` | 10.89 | 2.92 | yes | unstable |
| 50,000 | `hubness_gini` | 0.71 | 0.19 | yes | stable |
| 50,000 | `hub_share_top1pct` | 3.86 | 1.05 | yes | stable |
| 100,000 | `hubness_skew` | 8.45 | 2.21 | no | stable |
| 100,000 | `hubness_gini` | 0.64 | 0.16 | no | stable |
| 100,000 | `hub_share_top1pct` | 3.25 | 0.88 | no | stable |
| 250,000 | `hubness_skew` | 5.61 | 1.38 | no | stable |
| 250,000 | `hubness_gini` | 0.46 | 0.11 | no | stable |
| 250,000 | `hub_share_top1pct` | 2.24 | 0.56 | no | stable |

Full artifact: `docs/datasets/deep_hub_stability.json`.

`hubness_skew` is salvageable on DEEP -- 22.13% at N=20,000, falling under
the 10.0% bar by N=100,000 -- and is not salvageable on GloVe at any N tried.
The difference tracks the two corpora's own hubness: DEEP's mean
`hubness_skew` is 1.91 against GloVe's 4.58, a much less skewed
k-occurrence distribution to begin with. That is the difference between
"this statistic is corpus-dependent" and "raise N for GloVe" -- raising N
helps on DEEP and does not rescue GloVe within the range tried.
`hubness_gini` is stable on both corpora at every N, which is the argument
for replacing the statistic everywhere rather than re-tuning N per family.

### Provenance and the fresh `v0` re-sample

One N=20,000 cell was drawn from `glove_250k.npy` at eight draws (not
sixteen), matching the pool and draw count `docs/datasets/glove_noise_floor.json`
used, to check that this sweep's numbers land where that one's do before
anything else in the sweep is trusted. Three of four incumbent statistics
reproduced the committed means; LID median missed by 0.28%, diagnosed as a
difference in how many times the corpus is subsampled rather than a code
defect. The same cell was re-run under `--backend sklearn` and agreed with
`--backend torch` to within 8.7e-05 on every draw of every statistic. Both
are committed: `docs/datasets/glove_hub_stability_provenance.json` and
`docs/datasets/glove_hub_stability_sklearn_control.json`.

The `v0` figures in the tables above come from a **fresh** 250,000-vector
re-sample of the same five `v0` checkpoints, at the fixed sampling seed of
42 -- not from the 50,000-vector samples behind
`docs/datasets/glove_v0_noise_floor.json`, which cannot support N above
50,000. At N=20,000 the two measurements of `v0` agree closely: `lid_median`,
`relative_contrast_median` and `hubness_skew` all differ by 0.12% or less
between the committed five-seed means and the fresh re-sample, and
`ivf_gini` differs by -3.72%. That is close enough to call the fresh
re-sample trustworthy, while it remains a different measurement from the one
`glove_v0_noise_floor.json` records, and the two are not spliced into one
table.

### An unlooked-for finding: `ivf_gini`'s noise, carried along as a control

The four incumbent statistics were carried through this sweep to
re-measure the committed eight-draw table at sixteen draws and larger N, for
free. `ivf_gini` -- currently gated as one of the four ANN-difficulty
statistics, per `AGENTS.md`'s first invariant -- fails the same 10.0% bar
this study used to disqualify `hubness_skew`.

On GloVe its real-side range runs 9.16 / 9.61 / 19.45 / 10.97% across
N = 20,000 / 50,000 / 100,000 / 250,000; on DEEP it runs 12.01 / 13.79 /
24.33 / 19.34%, over the bar at every single N. In cv terms -- comparable
across draw counts, unlike range -- it sits at 2.56-5.21% on GloVe and
3.06-5.88% on DEEP, several times `lid_median`'s 0.26-1.53% over the same
grid, so this is not a draw-count artifact. Its noise also **rises** with N
up to 100,000 on both corpora rather than falling, the opposite of what
subsample noise usually does as N grows.

`## Hubness skew is below the noise floor at this N` above used to call
`ivf_gini` "stable enough to gate on" at 3.68%, and has been corrected above
to qualify that: a band set from 3.68% would be tighter than the statistic
supports once N or corpus changes, since the same statistic reaches 19.45%
on GloVe and clears 20% on DEEP within the grid this sweep measured.

A plausible mechanism, consistent with the numbers and the code but **not
proven here**: `cell_occupancy` in `src/eval/ann_difficulty.py` clusters
each draw with `MiniBatchKMeans(n_clusters=nlist_eff, random_state=seed,
n_init=3, batch_size=1024)`, at sklearn's default `max_iter`. The clustering
budget is fixed while N grows, so the partition has progressively more
points to place in the same number of iterations and batches -- a
progressively less converged clustering, not a progressively less balanced
corpus. This is a hypothesis worth testing directly, not a cause this study
established.

This finding does not change any gate: `gates/glove.yaml` already carries
`ivf_gini` with a null band, and setting or tightening a band is reserved
for a human working from a trained ladder, unchanged by this sweep.
