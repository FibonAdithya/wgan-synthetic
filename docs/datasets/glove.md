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

| Statistic | Real | Synthetic (best variant) |
|---|---|---|
| LID median | 35.077082 | — |
| Relative contrast | 1.389420 | — |
| Hubness skew | 5.639746 (see below) | — |
| IVF cell-balance Gini | 0.580732 | — |

Measured over 50,000 vectors at `preprocess: l2`, `seed: 42`, `nlist: 256`,
subsampled from `glove_250k.npy`. The report output these came from is
committed as `docs/datasets/glove_profile_summary.json`, and the noise-floor
draws below as `docs/datasets/glove_noise_floor.json`, so both of those tables
are checkable without access to the training box. Reproduce the real column
with:

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
human. See the `## Gate` section.

### Measured under this corpus's own metric

`ann_difficulty.py` measures this family under its `data.metric`, which is
`angular`: L2 between unit-norm rows. On the unit sphere Euclidean distance
is a strictly increasing function of cosine distance, so it ranks neighbours
identically -- the corpus is measured under the distance it is searched with.
Measuring requires `--preprocess l2`, and `ann_difficulty.compute` refuses
rows that are not on the sphere rather than normalizing them itself.

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
cannot be reproduced from a pinned commit. They are also not reproducible
with `--metric angular` as shipped: on unit-norm rows that flag returns the
L2 column exactly, for the reason given above, so a reader running this tool
should expect 35.2928, not 31.4624. The two changed figures are consistent
with the script's angular definition having been geodesic (`arccos`) rather
than cosine -- cosine exactly halves LID on unit-norm rows, a larger and
differently-shaped move than -10.85%, while a geodesic definition drops LID
by a comparable amount on synthetic isotropic unit vectors. Consistent, not
confirmed: the script cannot be inspected. The zeroes are the part that does
not need re-measuring, for the reason given above.

## Model family

`mlp` today, `spherical` when phase (b) lands.

## Ladder

| Variant | Delta | Config | Run | Status |
|---|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/glove/v0.yaml` | — | not trained |

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

When they are set, `hubness_skew` is the one to be careful with. The measured
profile above shows its subsample noise at the canonical N spans 3.46 to 8.33
on the real corpus alone, so any band wide enough not to reject real data
against itself would pass nearly any generator. A band on it would be
measuring the draw, not the model.

Check a run against it:

    python -m src.eval.check_gate --dataset glove --run-dir runs/glove/profile

It reads `summary.json` from that run directory, prints a JSON verdict, and
exits non-zero when the run fails -- or, as now, when the bands are still
unset, which is verdict `unset` and exit code 2. Pass `--allow-unset` to get
the report without the non-zero exit, and `--stats-name <label>` to check a
synthetic series rather than `real`.
