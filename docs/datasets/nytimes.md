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

`mlp` for `v0`, `linear_skip` for `v1`; `spherical` when phase (b) lands.

## Ladder

| Variant | Delta | Config | Run | Status |
|---|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/nytimes/v0_seed42.yaml`, instrument of `configs/nytimes/v0.yaml` | `runs/nytimes/v0_seed42` (box: `/workspace/nytimes-v0/v0_seed42`) | trained -- n=1 seed, misses the gate on every statistic; see `## v0, measured` |
| `v0` at 100k steps | same rung, budget raised | `configs/nytimes/v0_seed42_100k.yaml`, resumed from the row above | `runs/nytimes/v0_seed42_100k` (box: `/workspace/nytimes-v0/v0_seed42_100k`) | trained -- gate statistics worse than at 30k; see `### Continued to 100,000 steps` |
| `v1` | + linear skip path (`generator_type: linear_skip`) and `select_on: gate` | `configs/nytimes/v1.yaml`; box instrument `configs/nytimes/v1_seed42.yaml` | `runs/nytimes/v1_seed42` (box: `/workspace/nytimes-v1/v1_seed42`) | trained -- n=1 seed, misses the gate on LID and hubness, contrast within 3%; see `## v1, measured` |

Train `v0`:

    python -m src.train.train_wgan_gp --config configs/nytimes/v0.yaml

## v0, measured

Trained 2026-09-10 on `tig-gpu` (RTX 3060) as one gpuq job,
`scripts/nytimes_v0_seed42_job.sh` at commit `49171d4`: 30,000 generator
steps of `configs/nytimes/v0_seed42.yaml`, 37 minutes wall, on the corpus as
shipped (zero rows and duplicates included). 50,000 samples at sampling seed
42 from `best_generator.pt`, then from the step-26,000 and step-30,000
checkpoints, all measured together against the cleaned real corpus in one
`eda_report` invocation at the canonical conditions. The summaries,
`run_config.yaml` and `run_metadata.json` are committed under
`docs/results/nytimes-v0-seed42/`. The as-shipped comparison is committed
too, but it is meaningless in the way the noise-floor section explains: the
generator emits unit-norm rows, so it has no zero rows and its column is
identical in both reports while the real column is set by them.

The parenthesised figure after each v0 value is its distance from real in
units of the real corpus's own ten-draw spread (the range in the real
column). GloVe's page uses the training-seed spread for this; NYTimes has one
seed, so the real-side spread is the only yardstick available and it is the
looser of the two.

| Statistic | real, cleaned (10-draw range) | `best_generator.pt` (step 1,000) | step 26,000 | step 30,000 |
|---|---|---|---|---|
| LID median | `55.97` (54.97 -- 56.86) | `26` (15.9x) | `18.82` (19.7x) | `17.98` (20.1x) |
| Relative contrast | `1.271` (1.265 -- 1.276) | `1.598` (30.5x) | `2.101` (77.3x) | `2.078` (75.2x) |
| Hubness skew | `2.529` (2.315 -- 2.779) | `1.536` (2.1x) | `2.459` (0.2x) | `2.035` (1.1x) |
| IVF cell-balance Gini | `0.7767` (0.7832 -- 0.8231) | `0.2946` (12.1x) | `0.3516` (10.7x) | `0.295` (12.1x) |
| Effective rank | `247.4` | `52.1` | `30.3` | `34.7` |
| Median pairwise distance | `1.404` | `0.992` | `1.404` | `1.404` |
| Median 5-NN distance | `1.205` | `0.658` | `0.710` | `0.719` |

**v0 does not reproduce NYTimes' search difficulty at any checkpoint.** LID,
contrast and Gini sit far outside the real spread at every checkpoint, in the
same direction GloVe `v0` missed: lower LID, higher contrast, flatter IVF
partition. Hubness skew is the exception -- inside the real range at step
26,000, just under it at 30,000 -- and on its own it says little, since a
low-rank cloud can have the same hub profile as the corpus while being far
easier to search on the other three. (The real column's own Gini, `0.7767`,
sits just under its ten-draw range because `eda_report` draws its 20,000 rows
from a 50,000-row cut rather than from the whole subset; the draws behind the
range are of the whole subset.) The samples
are easier to search than the corpus, and the reason is visible in the
effective rank. The corpus fills the sphere (rank 247 of 256, participation
ratio 236); the generator never does. At step 1,000 it emits a cone -- mean
vector norm `0.71`, random pairs at cosine `0.51`, where real has `0.12` and
`0.015` -- and by step 30,000 it has learned the mean direction away (random
pairs land on the real median distance of `1.404` exactly) but only by
spreading a ~30-dimensional cloud over the sphere. A 30-dimensional cloud on a
256-dimensional sphere has near neighbours that are much nearer than its bulk
(contrast `2.1` against `1.27`), which is precisely the property an index
exploits.

**`best_generator.pt` is the step-1,000 checkpoint.** The trainer selects the
checkpoint by the smallest covariance Frobenius gap (`cov_fro`), and on this
family that gap is lowest at the first evaluation (`0.094`) and rises to
`0.25` by the end, while the mean gap falls from `0.69` to `0.13` and the
sample pairwise distance climbs onto the real value. The two diagnostics
disagree about which checkpoint is best, the selector follows the one that
gets worse, and neither is the gate (AGENTS.md invariant 1). Read the
step-30,000 column as what v0 trained to; read the first column as what
`src.sample.generate` on `best_generator.pt` would hand anyone who did not
check. EMA is off in `v0` (`ema_decay` unset), so every checkpoint, `best_generator.pt`
included, holds live weights and the columns are directly comparable.

What would move this is a ladder decision, not a rerun: the rank collapse is
the thing to attack, and the per-family ladders in `PROJECT_DOCUMENTATION.md`
already have rungs for it. Whether the training set should also have its zero
rows and duplicates removed is the open question from `## Measured profile`;
this run shows it does not decide the outcome, since the generator's failure
is rank, not the 0.08% of rows at the origin.

### Continued to 100,000 steps

Asked for after the table above. `configs/nytimes/v0_seed42_100k.yaml` is the
same instrument with the budget raised, run with `--resume` from the 30k run's
`checkpoint_step_30000.pt` (`scripts/nytimes_v0_seed42_100k_job.sh`, commit
`aec4cbf`, 87 minutes for the remaining 70,000 steps). Summary,
`run_config.yaml` and `run_metadata.json` are under
`docs/results/nytimes-v0-seed42-100k/`. `best_generator.pt` never changed:
the restored selection score of `0.094` from step 1,000 was not beaten (the
continuation's lowest `cov_fro` was `0.162`, at step 95,000), so the sampled
"best" is byte-identical to the 30k run's and is not repeated here.

| Statistic | real, cleaned (10-draw range) | step 30,000 | step 100,000 |
|---|---|---|---|
| LID median | `55.97` (54.97 -- 56.86) | `17.98` (20.1x) | `13.75` (22.3x) |
| Relative contrast | `1.271` (1.265 -- 1.276) | `2.078` (75.2x) | `2.076` (75.0x) |
| Hubness skew | `2.529` (2.315 -- 2.779) | `2.035` (1.1x) | `1.362` (2.5x) |
| IVF cell-balance Gini | `0.7767` (0.7832 -- 0.8231) | `0.295` (12.1x) | `0.2201` (14.0x) |
| Effective rank | `247.4` | `34.7` | `83.8` |
| Participation ratio | `236` | `14.3` | `31.4` |
| Components for 90% of variance | `219` | `47` | `125` |
| Mean-vector norm | `0.124` | `0.183` | `0.134` |
| Median 5-NN distance | `1.205` | `0.719` | `0.766` |

**More steps made the global moments better and the search difficulty
worse.** By 100,000 steps the generator's mean is on target (mean gap `0.05`,
down from `0.13`), random pairs land on the real cosine (`-0.007` against
`0.015`), and the spectrum has broadened from 47 to 125 components for 90% of
variance, with `cov_fro` still falling at the end -- the run had not
converged on the diagnostics the trainer watches. On the gate it went the
other way: LID median fell from `18` to `13.7`, hubness from `2.0` to `1.4`,
Gini from `0.30` to `0.22`, while contrast held at `2.08`. The 5-NN distance
histogram shows what happened: the generator now spreads its cloud across
more global directions, but each sample's neighbours sit at `0.77` where the
real corpus's sit at `1.2` -- locally the samples lie on a ~14-dimensional
sheet, and that local dimension, not the global rank, is what LID, contrast
and an IVF partition read.

So the answer to "does it improve with a longer budget" is no, for this
family and this rung: the objective is pulling the first two moments onto
the corpus, and the first two moments of an isotropic corpus are satisfied
by a low-dimensional cloud with the right mean and covariance envelope. The
missing constraint is on local structure, which is what the later ladder
rungs (`distance_reg_alpha`, `lid_reg`, `spectrum_reg`, all `0.0` in `v0`)
exist to supply. Choosing one is a ladder decision.

## v1, measured

Trained 2026-09-11 on `tig-gpu` (RTX 3060) as one gpuq job,
`wgan-synthetic-20260911T134101Z-32dbda`
(`scripts/nytimes_v1_seed42_job.sh` at commit `a9ea30f`): 30,000 generator
steps of `configs/nytimes/v1_seed42.yaml` (`generator_type: linear_skip`,
`select_on: gate`), submitted 13:41:01Z and complete at 14:19:17Z -- **38
minutes wall for the whole job** (training, two 50,000-row samples and one
`eda_report`), against v0's 37 minutes for training alone, so the extra
12,500-row holdout k-NN each eval adds under `select_on: gate` cost nothing
measurable. No `gate_error` on any eval; `resumed_from_step` 0; EMA is off in
this config, so every checkpoint holds live weights. 50,000 samples at
sampling seed 42 were drawn from `best_generator.pt` (the gate selector's
pick, step 20,000, `best_score` `0.0114`) and from the step-30,000
checkpoint, measured together against the cleaned real corpus in one
`eda_report` invocation at the canonical conditions. The summaries,
`run_config.yaml` and `run_metadata.json` are committed under
`docs/results/nytimes-v1-seed42/`; the raw samples are gitignored under
`runs/nytimes/v1_seed42/`.

The parenthesised figure after each `v1` value follows the same convention
`v0`'s section uses: distance from real in units of the real corpus's own
ten-draw spread (the range in the real column). Alongside it is the percentage
off real median the spec's 3% bar is stated in, since that is the number the
success criterion is measured against; both are computed from the same
`docs/datasets/nytimes_noise_floor.json` range as `v0`'s table.

| Statistic | real, cleaned (10-draw range) | `v1_best` (step 20,000, gate-selected) | step 30,000 |
|---|---|---|---|
| LID median | `55.97` (54.97 -- 56.86) | `62.58` (11.8% off, 3.5x) | `45.8` (18.2% off, 5.4x) |
| Relative contrast | `1.271` (1.265 -- 1.276) | `1.242` (2.3% off, 2.7x) | `1.42` (11.8% off, 13.9x) |
| Hubness skew | `2.529` (2.315 -- 2.779) | `10.37` (310.0% off, 16.9x) | `5.164` (104.2% off, 5.7x) |
| IVF cell-balance Gini | `0.7767` (0.7832 -- 0.8231) | `0.8273` (6.5% off, 1.3x) | `0.7239` (6.8% off, 1.3x) |

Effective rank: real `247.4`, `v1_best` `152.9`, step 30,000 `95.7`. Median
5-NN distance: real `1.205`, `v1_best` `1.159`, step 30,000 `1.017`. Both sit
much closer to real than `v0` ever got (effective rank `84` at step 30,000
on the as-shipped corpus) -- the skip path is doing the full-rank-output job
the design asked of it.

**The gate selector picked step 20,000 (score `0.0114`); `cov_fro` would have
picked step 3,000** (`cov_fro` `0.0365`, whose gate score is `0.2774` -- a
Gaussian-like checkpoint reading holdout LID `72`, contrast `1.17`). The
reference the selector measured against (12,500-row holdout, 7 exact-zero
rows dropped) reads LID `58.97`, contrast `1.241`, hubness `2.11`, Gini
`0.813`; that reference is `5.4%` above the canonical 20k-row real median of
`55.97`, and the selected step reads `59.32` on the holdout but `62.58` at
20k rows -- two different measurement conditions, which is why "score
`0.011`" and "`11.8%` off" both describe the same checkpoint without
contradicting each other.

Trajectory on the holdout (`run_metadata.json` `eval`, full table there):

| step | fake LID | fake RC | hubness | Gini | score | cov_fro |
|---|---|---|---|---|---|---|
| 1000 | 64.39 | 1.1925 | 2.04 | 0.765 | 0.1312 | 0.0424 |
| 3000 | 71.99 | 1.1709 | 11.87 | 0.867 | 0.2774 | 0.0365 |
| 10000 | 62.95 | 1.2079 | 13.46 | 0.847 | 0.0943 | 0.0874 |
| 15000 | 60.4 | 1.2385 | 8.27 | 0.861 | 0.0264 | 0.1147 |
| 20000 | 59.32 | 1.2345 | 7.02 | 0.841 | 0.0114 | 0.1152 |
| 21000 | 55.77 | 1.2475 | 5.69 | 0.843 | 0.0593 | 0.1194 |
| 25000 | 51.56 | 1.2829 | 5.45 | 0.807 | 0.1593 | 0.1386 |
| 30000 | 42.56 | 1.4024 | 4.74 | 0.761 | 0.4081 | 0.1835 |

The generator starts at the Gaussian's numbers (LID `72` / contrast `1.17`
around steps 2,000--7,000, close to the Gaussian row's `82.5` / `1.158` on
this page) and drifts monotonically toward `v0`'s sheet (LID `42.6` /
contrast `1.40` at step 30,000; `v0` read `14` / `2.08` at the same step). It
passes through the real point around step 20,000--21,000 and does not stay
there. The gate selector caught the crossing; `cov_fro` would have caught the
Gaussian end instead.

Skip map `W` (256x256) singular values, live weights, 4,096 fixed latents:

| step | max | median | min | # singular values > 0.1 |
|---|---|---|---|---|
| 1000 | 2.08 | 0.837 | 0.464 | 256 |
| 10000 | 1.721 | 0.905 | 0.24 | 256 |
| 20000 | 1.473 | 0.979 | 0.091 | 255 |
| 30000 | 1.561 | 0.918 | 0.028 | 254 |

`W` stays full rank throughout training. The design's rank argument held --
rank is not what moved.

Trunk-versus-skip output energy (mean squared norm over the same 4,096
latents) and the skip term's share of the pre-normalisation output:

| step | \|\|trunk\|\|^2 | \|\|skip\|\|^2 | skip share | \|cos(trunk, skip)\| |
|---|---|---|---|---|
| 1000 | 24.7 | 250.3 | 0.91 | 0.053 |
| 8000 | 115.3 | 256.0 | 0.689 | 0.046 |
| 15000 | 145.8 | 257.2 | 0.638 | 0.048 |
| 20000 | 213.6 | 254.3 | 0.544 | 0.047 |
| 26000 | 415.1 | 247.1 | 0.373 | 0.047 |
| 30000 | 944.3 | 239.8 | 0.202 | 0.045 |

The skip term's energy stays pinned near `256` (orthogonal `W`, unit
latents) while the trunk's grows `38x` over the run. After the trainer's L2
normalisation, the residual's share of each output vector falls from `91%`
to `20%`. The noise-sweep on this page found LID `56` at noise-to-signal
`1.4`; this run crosses that ratio near step 20,000 and keeps going. The two
terms stay orthogonal (`cos` `0.05` throughout), so this is a balance drift
between two terms that never learn to interact, not the trunk learning to
cancel the skip.

**`v1` misses the bar.** At the selected checkpoint, contrast is within `3%`
of real (`2.3%`) but LID is `11.8%` high, hubness skew is `10.4` against a
real range topping out at `2.78` and worse than the noise-sweep bound of
`4.0`, and Gini sits just above the real range. The architecture did what the
spec claimed: the output stays full rank (`W`'s singular values never
collapse), the generator reaches LID `72` at step 3,000 instead of the `14`
`v0` was stuck near, and the gate selector found a checkpoint that passes
through the real LID/contrast point -- something no `v0` checkpoint ever did.
The real-side cleaning was necessary to see any of this: without it the
reference LID the selector measures against is `29.9`, not `59`. But the
checkpoint that matches is a transient. The trunk-to-skip balance is
unconstrained under a per-vector critic, so the trunk's output energy climbs
monotonically (`38x` over the run) while the skip's holds still, and the
"real-like" checkpoint is a way station on the trunk's path back to `v0`'s
sheet, not a stable point. Per the spec's own rule for this outcome: the
quantity that drifted (the trunk/skip energy balance, read only pairwise
through the residual) is one a per-vector critic cannot see, so the next
rung is the neighbourhood-aware critic (approach B), not another change to
this generator. Pinning the skip share (or the trunk's output scale) is worth
trying as a cheap diagnostic to see whether hubness comes down on its own,
but it is a diagnostic, not a rung -- it does not give the critic the
neighbourhood information the failure mode needs.

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
