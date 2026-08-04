# OpenAI

1536-dimensional text embeddings from an OpenAI embedding model, already
unit-norm. The vectors come from a DBpedia text corpus, and the one
structural fact that decides how this family is modelled is that ambient
dimension is very high while intrinsic dimension is low.

## Source

    python -m src.data.fetch openai

Fetches `dbpedia-openai-1000k-angular` into the shared cache and writes
`data/openai_250k.npy` and `data/openai_1m.npy`. The HDF5 is large and
immutable; the fetcher downloads it once and is safe to run concurrently.

| | |
|---|---|
| Dimension | `1536` |
| Search metric | `angular` |
| Upstream | `dbpedia-openai-1000k-angular` |

## Structure

1536-dimensional text embeddings, already unit-norm. Ambient dimension is
very high while intrinsic dimension is low, so LID and relative contrast are
the statistics that carry information; per-dimension marginals say almost
nothing at this width.

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
        --real-path data/openai_250k.npy \
        --output-dir runs/openai/profile \
        --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10

Read the four values out of runs/openai/profile/summary.json (written by the command above).

`ann_difficulty.py` currently measures everything under L2, including this
family's `angular` corpus, so these numbers will need re-measuring once
angular distance support lands (phase (c)).

## Model family

`mlp` today, `spherical` when phase (b) lands.

## Ladder

| Variant | Delta | Config | Run | Status |
|---|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/openai/v0.yaml` | — | not trained |

Train `v0`:

    python -m src.train.train_wgan_gp --config configs/openai/v0.yaml

## Gate

Pass bands are per statistic, not a combined score, because the four fail in
different directions. Bands are set once this family has a trained ladder to
show what is achievable; until then this section records that they are unset.
