# DEEP

96-dimensional image descriptors produced by a deep network, dense, signed
and unit-norm. The vectors come from a deep image embedding model, and the
one structural fact that decides how this family is modelled is that they
all lie exactly on the unit sphere.

## Source

    python -m src.data.fetch deep

Fetches `deep-image-96-angular` into the shared cache and writes
`data/deep_250k.npy` and `data/deep_1m.npy`. The HDF5 is large and
immutable; the fetcher downloads it once and is safe to run concurrently.

| | |
|---|---|
| Dimension | `96` |
| Search metric | `angular` |
| Upstream | `deep-image-96-angular` |

## Structure

96-dimensional image descriptors from a deep network, dense, signed and
unit-norm. The smallest angular family, which makes it the right first
target for the `spherical` generator.

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
        --real-path data/deep_250k.npy \
        --output-dir runs/deep/profile \
        --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10

Read the four values out of runs/deep/profile/summary.json (written by the command above).

`ann_difficulty.py` currently measures everything under L2, including this
family's `angular` corpus, so these numbers will need re-measuring once
angular distance support lands (phase (c)).

## Model family

`mlp` today, `spherical` when phase (b) lands — being the smallest angular
family, this is the first candidate for the unit-norm-native generator.

## Ladder

| Variant | Delta | Config | Run | Status |
|---|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/deep/v0.yaml` | — | not trained |

Train `v0`:

    python -m src.train.train_wgan_gp --config configs/deep/v0.yaml

## Gate

Pass bands are per statistic, not a combined score, because the four fail in
different directions. Bands are set once this family has a trained ladder to
show what is achievable; until then this section records that they are unset.
