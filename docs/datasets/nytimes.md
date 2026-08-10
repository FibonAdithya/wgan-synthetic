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

256-dimensional document embeddings, dense and signed, with strong cluster
structure by topic. Cluster structure is what IVF cell balance measures, so
this family stresses that panel hardest.

## Measured profile

Read from the file rather than quoted from a paper. Canonical N and k are
locked here so a gate result stays readable against an older one.

| | |
|---|---|
| Canonical N | `20000` |
| Canonical k | `100` (`10` for hubness) |

| Statistic | Real | Synthetic (best variant) |
|---|---|---|
| LID median | not yet measured | — |
| Relative contrast | not yet measured | — |
| Hubness skew | not yet measured | — |
| IVF cell-balance Gini | not yet measured | — |

Fill the real column with:

    python -m src.eval.eda_report \
        --real-path data/nytimes_250k.npy \
        --output-dir runs/nytimes/profile \
        --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular

Read the four values out of runs/nytimes/profile/summary.json (written by the command above).

`ann_difficulty.py` measures this family under its `data.metric`, which is
`angular`: L2 between unit-norm rows. On the unit sphere Euclidean distance
is a strictly increasing function of cosine distance, so it ranks neighbours
identically -- the corpus is measured under the distance it is searched with.
Measuring requires `--preprocess l2`, and `ann_difficulty.compute` refuses
rows that are not on the sphere rather than normalizing them itself.

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
