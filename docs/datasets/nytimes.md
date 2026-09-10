# NYTimes

256-dimensional document embeddings of New York Times articles, dense and
signed. The vectors come from a text embedding model over news articles, and
the one structural fact that decides how this family is modelled is its
strong cluster structure by topic.

## Source

    python -m src.data.fetch nytimes

Fetches `nytimes-256-angular` into the shared cache and writes
`data/nytimes_250k.npy` and `data/nytimes_1m.npy`. The HDF5 is large and
immutable; the fetcher downloads it once and is safe to run concurrently.

The `_1m` name is the requested size, not a guarantee: if the upstream
corpus holds fewer than 1,000,000 rows, the fetcher caps the subset at
whatever the corpus actually has and prints a notice saying so — it does not
silently write a smaller file under a `_1m` name with nothing to flag it.

| | |
|---|---|
| Dimension | `256` |
| Search metric | `angular` |
| Upstream | `nytimes-256-angular` |

## Structure

256-dimensional document embeddings, dense and signed. The page used to say
"strong cluster structure by topic"; the spectrum does not show it. Measured
on the cleaned corpus (see below), the mean vector has norm `0.124`, random
pairs sit at cosine `0.015`, the top principal component carries `2.1%` of
the variance and the participation ratio is `224` of 256 -- close to
isotropic, the opposite of `openai`'s narrow cone. What the corpus does have
is the most lopsided IVF partition of any family measured so far (Gini
`0.80` against `0.58` for glove and `0.55` for openai), so cell balance is
still the panel this family stresses hardest.

Two facts about the file matter more than its geometry, because they set
every gate statistic at the canonical N:

- **239 of the 290,000 upstream rows are exactly zero** (197 of the 250k
  subset). `maybe_l2_normalize` leaves a zero row at zero, so after
  `preprocess: l2` each one sits at L2 distance exactly `1.0` from every
  other row.
- **41,031 upstream rows are exact repeats of another row** (14.1%), in
  26,558 groups; the largest non-zero group has 95 copies. 2,384 of the
  10,000 upstream queries have an exact copy in the train split.

Raw rows are not unit-norm either: norms run `0.90` (1st percentile) to
`1.10` (99th), so `--preprocess l2` is doing real work here, unlike on
`openai`.

## Measured profile

Read from the file rather than quoted from a paper. Canonical N and k are
locked here so a gate result stays readable against an older one.

| | |
|---|---|
| Canonical N | `20000` |
| Canonical k | `100` (`10` for hubness) |

| Statistic | Real | Synthetic (best variant) |
|---|---|---|
| LID median | 15.813480 (see below) | — |
| Relative contrast | 1.405519 | — |
| Hubness skew | 31.578471 (see below) | — |
| IVF cell-balance Gini | 0.978471 (see below) | — |

Measured 2026-09-10 over 50,000 vectors at `preprocess: l2`, `seed: 42`,
`nlist: 256`, `metric: angular`, subsampled from `nytimes_250k.npy`. The
report output these came from is committed as
`docs/datasets/nytimes_profile_summary.json`. Reproduce the real column with:

    python -m src.eval.eda_report \
        --real-path data/nytimes_250k.npy \
        --output-dir runs/nytimes/profile \
        --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular

Read the four values out of runs/nytimes/profile/summary.json (written by the command above).

### Three of the four numbers above measure the zero rows, not the corpus

**Do not read the LID, hubness or Gini above as properties of NYTimes.** In
the canonical 20,000-row draw, ten rows appear in 12,179 to 12,309 of the
20,000 neighbour lists each and 72% of rows appear in none; one k-means cell
holds 19,532 of the 20,000 points. Those ten rows are zero rows. Within a
20k draw the median row's 5th-nearest neighbour is at L2 distance exactly
`1.0` -- most rows have fewer than five real neighbours closer than the
origin -- so the 15-odd zero rows in any draw fill the top of nearly every
10-NN list, and sklearn's index-order tie-break hands all of that
k-occurrence to the ten lowest-indexed of them. The same rows sit ahead of
real neighbours in the 100-NN lists and pull LID down.

Measured directly with `ann_difficulty.compute` at the locked conditions on
the 250k subset, one 20,000-row draw at seed 42 (committed as
`docs/datasets/nytimes_row_filters.json`):

| Rows kept | rows | zero rows in draw | LID median | Relative contrast | Hubness skew | IVF Gini | max relative contrast | LID queries discarded |
|---|---|---|---|---|---|---|---|---|
| as is | 250,000 | 15 | `19.22` | `1.4047` | `36.47` | `0.982` | `9.4e+07` | 33 |
| drop zero rows | 249,803 | 0 | `55.36` | `1.2733` | `2.48` | `0.802` | `9.47e+07` | 10 |
| drop exact duplicates | 217,855 | 0 | `55.77` | `1.2710` | `2.64` | `0.816` | `313` | 0 |
| drop both | 217,854 | 0 | `55.84` | `1.2692` | `2.59` | `0.796` | `144` | 0 |

The as-is row differs from the profile table because `eda_report` cuts to
50,000 rows before `compute` cuts to 20,000, so the two draws are not the
same rows. The zero rows alone account for the LID, hubness and Gini
figures; the duplicates alone account for the relative-contrast tail
(pairs of near-identical rows leave r_1 at float32 rounding, ~1e-8, so a
handful of queries score a contrast of 10^7) and the discarded LID queries.
The medians of LID and contrast are unaffected by the duplicates.

### Noise floor

Ten disjoint 20,000-row draws of the 250k subset at the locked conditions,
committed as `docs/datasets/nytimes_noise_floor.json`, computed with
`ann_difficulty.compute` directly rather than through `eda_report` (the
`openai_structure` module does the same, but it refuses this family's raw
rows because they are not unit-norm, and its noise floor on an
L2-normalised copy died without output).

As is, the draw's statistic is a function of how many zero rows landed in
it -- 12 zero rows give hubness skew `40.8`, 24 give `28.8`:

| Statistic | min | median | max | spread | % of median |
|---|---|---|---|---|---|
| LID median | `13.88` | `20.07` | `22.05` | `8.17` | `40.7%` |
| Relative contrast | `1.405` | `1.406` | `1.406` | `0.00108` | `0.1%` |
| Hubness skew | `28.83` | `37.76` | `40.79` | `12` | `31.7%` |
| IVF Gini | `0.7999` | `0.9813` | `0.9821` | `0.182` | `18.6%` |

With zero rows and exact duplicates dropped before the draw (217,854 rows
left, so ten disjoint draws still fit):

| Statistic | min | median | max | spread | % of median |
|---|---|---|---|---|---|
| LID median | `54.97` | `56.06` | `56.86` | `1.89` | `3.4%` |
| Relative contrast | `1.265` | `1.269` | `1.276` | `0.0107` | `0.8%` |
| Hubness skew | `2.315` | `2.525` | `2.779` | `0.464` | `18.4%` |
| IVF Gini | `0.7832` | `0.8011` | `0.8231` | `0.0399` | `5.0%` |

On the cleaned rows all four are usable, with hubness skew the loosest at
18%, in line with openai's 9% and far from glove's 108%. On the file as
shipped, only relative contrast is stable, and it is stable at a value the
zero rows set.

### Angular versus L2, measured

On the cleaned, L2-normalised rows (`docs/datasets/nytimes_structure.json`,
written by `openai_structure` run against that copy): neighbour-set
agreement at k=100 between L2 and cosine is `0.999999`, hubness skew is
identical under both (`2.5938`), and LID median under cosine is `27.92`,
half the L2 `55.84`, as `cos = L2^2/2` implies for a log-ratio estimator.
The argument behind `--metric angular` holds here as it did on openai.

It holds only after normalisation. On the raw rows, whose norms spread
`0.90`--`1.10`, the plain-Euclidean nearest neighbour is the angular
nearest neighbour for only 57.6% of 2,000 upstream queries against the full
290,000-row train split, and Euclidean top-10 recalls 38% of the angular
top-10. That figure came from a one-off script outside this tree, so treat
it as indicative; the direction is not in doubt.

### What the geometry says about `v0`

Measured 2026-09-10 on the cleaned rows. Readings, not decisions: no config
is changed here.

- **Isotropic, so centering is not a lever.** The participation ratio is
  `224` about the origin and `238` about the mean, and the top component
  goes from `2.1%` to `1.0%`. Compare openai, where centering moved it from
  2 to 175. `preprocess.center: false` costs nothing here.
- **Intrinsic dimension is 14 to 56 against an ambient 256**: two-NN gives
  `14.3`, LID median `55.8`. `latent_dim: 256` equals the ambient dimension;
  a rung at 64 would be the cheap comparison.
- **The zero rows and duplicates reach the generator as shipped.**
  `l2_normalize` keeps a zero row at the origin, a point off the sphere the
  generator cannot reproduce, and 13% of the training rows are repeats of
  another row. Whether to drop them belongs to the same decision as the
  measurement question below.

### What needs a human

Every gate statistic at the locked conditions is set by 0.08% of the rows.
The fix is a row filter -- drop exact-zero rows, and probably exact
duplicates -- applied before the draw, which changes the locked measurement
conditions and what `v0` is trained on. That is a decision about what this
family's profile *is*, so it is raised here and not made. Until it is made,
the profile table above is the number the locked procedure produces, and
the cleaned figures are what the corpus looks like without the artefact.

## Model family

`mlp` today, `spherical` when phase (b) lands.

## Ladder

| Variant | Delta | Config | Run | Status |
|---|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/nytimes/v0.yaml` | — | not trained |

Train `v0`:

    python -m src.train.train_wgan_gp --config configs/nytimes/v0.yaml

## Gate

`gates/nytimes.yaml` is the gate. The bands live there rather than in this
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

Check a run against it:

    python -m src.eval.check_gate --dataset nytimes --run-dir runs/nytimes/profile

It reads `summary.json` from that run directory, prints a JSON verdict, and
exits non-zero when the run fails -- or, as now, when the bands are still
unset, which is verdict `unset` and exit code 2. Pass `--allow-unset` to get
the report without the non-zero exit, and `--stats-name <label>` to check a
synthetic series rather than `real`.
