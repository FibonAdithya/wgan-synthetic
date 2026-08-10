# OpenAI

1536-dimensional text embeddings from an OpenAI embedding model, already
unit-norm. The vectors come from a DBpedia text corpus, and the one
structural fact that decides how this family is modelled is that ambient
dimension is very high while intrinsic dimension is low.

## Source

    python -m src.data.fetch openai

Writes `data/openai_250k.npy`. This is the one family ann-benchmarks does
not publish as an HDF5: it names `dbpedia-openai-*-angular` as datasets but
generates them on demand from a HuggingFace dataset, so the fetcher reads
that dataset's parquet shards instead. They are large and immutable; the
fetcher downloads them once and is safe to run concurrently.

Only the 250k subset is written, unlike the five HDF5-backed families, which
default to 250k and 1M. `configs/openai/v0.yaml` names `openai_250k.npy` and
this family's canonical N is 20,000, so a 1M subset would cost 6GB to go
unread.

| | |
|---|---|
| Dimension | `1536` |
| Search metric | `angular` |
| Upstream | `KShivendu/dbpedia-entities-openai-1M` (HuggingFace) |
| Benchmark name | `dbpedia-openai-1000k-angular` (generated, not hosted) |

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

At this width the report drops its per-dimension marginals and correlation
panels, which are quadratic in the dimension and say little here; pass
`--max-panel-dim 1536` to force them back on.

## Structural profile

The four gate statistics say how hard this corpus is to search. They do not
say what a generator for it should look like. That question -- how many
directions the data actually uses, whether it fills the sphere or a cone,
and which statistics are stable enough to gate on -- is answered by:

    python -m src.eval.openai_structure \
        --real-path data/openai_250k.npy \
        --output-dir runs/openai/structure

It writes `openai_structure.html` and `structure.json`, and reports the
noise floor: the spread of each gate statistic across disjoint draws of the
same real corpus. A band narrower than that spread would reject real data,
so it bounds how tight any band can be. It sets no bands.

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

`gates/openai.yaml` is the gate. The bands live there rather than in this
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

    python -m src.eval.check_gate --dataset openai --run-dir runs/openai/profile

It reads `summary.json` from that run directory, prints a JSON verdict, and
exits non-zero when the run fails -- or, as now, when the bands are still
unset, which is verdict `unset` and exit code 2. Pass `--allow-unset` to get
the report without the non-zero exit, and `--stats-name <label>` to check a
synthetic series rather than `real`.
