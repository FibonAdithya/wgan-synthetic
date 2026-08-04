# GIST

960-dimensional GIST image descriptors, non-negative dense floats. The
vectors summarize global scene structure, and the one structural fact that
decides how this family is modelled is its high ambient dimension paired
with the near-absence of exact-zero mass.

## Source

    python -m src.data.fetch gist

Fetches `gist-960-euclidean` into the shared cache and writes
`data/gist_250k.npy` and `data/gist_1m.npy`. The HDF5 is large and
immutable; the fetcher downloads it once and is safe to run concurrently.

| | |
|---|---|
| Dimension | `960` |
| Search metric | `l2` |
| Upstream | `gist-960-euclidean` |

## Structure

960-dimensional GIST descriptors, non-negative dense float with little
exact-zero mass. Shares SIFT's non-negativity but not its sparsity, so the
thing `gated` was built to fix may not be present; the ladder starts on
`mlp` and tries `gated` as a rung rather than assuming it. The high ambient
dimension is the dominant cost.

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
        --real-path data/gist_250k.npy \
        --output-dir runs/gist/profile \
        --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10

Read the four values out of runs/gist/profile/summary.json (written by the command above).

## Model family

`mlp` — starting point for this family; `gated` is tried as a rung only if
the profile shows the sparsity it was built to fix.

## Ladder

| Variant | Delta | Config | Run | Status |
|---|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/gist/v0.yaml` | — | not trained |

Train `v0`:

    python -m src.train.train_wgan_gp --config configs/gist/v0.yaml

## Gate

Pass bands are per statistic, not a combined score, because the four fail in
different directions. Bands are set once this family has a trained ladder to
show what is achievable; until then this section records that they are unset.
