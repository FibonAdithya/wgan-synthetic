# OpenAI embeddings: EDA and the fetcher it depends on

Date: 2026-08-10
Status: design, approved for planning
Base: `origin/main` at `fb5ec5d`

## Problem

`docs/datasets/openai.md` records the openai family's measured profile as
"not yet measured" in all four gate statistics, and names the command that
would fill it. That command cannot work. It has never worked.

`src/data/fetch.py:61` registers openai through `_ann_benchmarks()`, the
helper that builds a URL of the form
`http://ann-benchmarks.com/<slug>.hdf5`. For the other five families that
file exists. For `dbpedia-openai-1000k-angular` it does not, and no mirror
serves it:

    dbpedia-openai-1000k-angular  ->  404
    dbpedia-openai-100k-angular   ->  404
    sift-128-euclidean            ->  206
    glove-100-angular             ->  206

Upstream ann-benchmarks does not host these. It *generates* them, in
`ann_benchmarks/datasets.py:564-578`, by loading the HuggingFace dataset
`KShivendu/dbpedia-entities-openai-1M` and calling `write_output`. The
`dbpedia-openai-*-angular` names are real dataset names; they are simply not
downloadable artefacts. So `_ann_benchmarks()` is the correct helper for five
of six families and the wrong one for the sixth.

Nothing catches this. The fetcher is not exercised against the network in
CI — correctly, since the files are gigabytes — so a registry entry that
describes a nonexistent file lints clean, tests clean, and fails only when
somebody tries to use it. That somebody is this EDA.

The goal is design inputs for openai's GAN training: what the geometry
implies for `latent_dim`, for `mlp` versus `spherical`, for centering and
whitening, and which of the four gate statistics are stable enough to carry
a band at all. None of that can start until the corpus exists on disk.

## Non-goals

- **Setting gate bands.** `AGENTS.md` reserves this for a human, and the
  reasoning is sound: choosing a number is a judgement about what counts as
  success. This work reports what the measured noise floor *permits* and
  stops. `gates/openai.yaml` keeps every band `null`.
- **Training openai `v0`.** This is the EDA that precedes planning the
  training, not the training.
- **Implementing angular distance in `ann_difficulty.py`.** That is phase (c)
  and issue #16. This work measures how much the L2-versus-angular choice
  actually changes each statistic, which is an input to that decision.
- **Re-pointing any config's `data.real_path`.** `AGENTS.md` lists this as
  needing a human, and openai `v0` already names `data/openai_250k.npy`,
  which is what this work produces.
- **Generalising the EDA report with new high-dimensional panels.** Rejected
  during design in favour of a separate analysis script; revisit only if the
  script's panels prove worth sharing across families.

## Architecture

Three independent units, in dependency order. Unit 1 gates the rest; units 2
and 3 are independent of each other.

    Unit 1  fetcher fix        ->  data/openai_250k.npy exists
    Unit 2  width guard        ->  the report is readable at 1536 dimensions
    Unit 3  measurement        ->  the profile, the noise floor, the design read

## Unit 1 — a parquet-backed source in the fetcher

`Source` today assumes one downloadable file read through `h5py` with an
`hdf5_key`. openai needs a second kind: a set of parquet shards, one column
of which holds the embedding.

The shard list comes from the HuggingFace API endpoint

    https://huggingface.co/api/datasets/<repo>/parquet/default/train

which returns stable, hash-free URLs (`.../train/0.parquet` upward). This is
deliberately preferred over the repository's own filenames
(`data/train-00000-of-00026-3c7b99d1c7eda36e.parquet`), whose embedded
hashes would have to be scraped from the siblings listing and would change
if the dataset were ever re-uploaded.

Shards honour HTTP range requests, so each one downloads through the
existing `fetch()` helper unchanged — already atomic and single-flight, which
is what makes the documented "safe to run concurrently" claim stay true.
Shard 0 is 366 MB; the full set is roughly 9.5 GB, cached under
`<cache-dir>/dbpedia-openai-1M/<i>.parquet`.

**All shards are downloaded, not a prefix.** `subset()` takes a seeded
random sample of the full corpus, and reading only the first *k* shards
would silently substitute a different sampling scheme. DBpedia entities may
be ordered by topic; a prefix could be structurally biased in exactly the
way that would corrupt an intrinsic-dimension or hubness measurement without
looking wrong. Matching every other family's semantics is worth the
bandwidth.

The `openai` column is stacked into `[N, 1536]` float32 and handed to the
existing `subset()` path, so openai's 250k rows are drawn exactly as sift's
and glove's are.

`pyarrow` is added to `requirements.txt`, pinned, with a comment saying what
needs it — matching how `h5py` is already annotated there.

Only the 250k subset is written. `default_rows` for this family drops the
1M entry: openai `v0` names `openai_250k.npy`, the gate's canonical N is
20,000, and the noise floor needs ten disjoint 20k draws, which 250k
supplies with room to spare. The 1M subset would cost 6 GB to sit unused.

### Testing

Unit tests build a tiny synthetic parquet on a `tmp_path` and assert the
dimension check fires on a mismatch, the row count is right, the requested
count clamps when the corpus is smaller, and two runs at one seed produce
identical output. No test touches the network, matching how the HDF5 path is
tested today.

## Unit 2 — a width guard on two panels

At 1536 dimensions, `fig_per_dim_marginals` (`src/eval/eda/figures.py:73`)
emits one trace per dimension plus a dropdown whose 1536 buttons each carry
a 1536-entry visibility list — about 2.4M booleans of JSON. `fig_correlation`
(`src/eval/eda/figures.py:204`) builds a full 1536x1536 heatmap, another
2.4M cells. Neither has a dimension guard.

The report does not break; it becomes tens of megabytes and slow to open,
and it spends that on the two panels `docs/datasets/openai.md` singles out as
least informative at this width: "per-dimension marginals say almost nothing
at this width".

Both builders return `None` above a width threshold. This needs no new
mechanism: `Panel.build` returning `None` already means "does not apply to
this run", which is how the glyph panel handles non-128-dimensional data.

The threshold defaults to **256** and is exposed as a CLI flag. That keeps
both panels for sift (128), deep (96), glove (100) and nytimes (256), and
drops them for gist (960) and openai (1536). A skipped panel is omitted
silently, consistent with the glyph panel's existing behaviour.

### Testing

A test asserts both builders return `None` above the threshold and a figure
at or below it, and that the report still renders with the panels absent.

## Unit 3 — the measurement

Run on the GPU box's **cpu** lane via `gpuq`, at a pinned commit. Nothing
here touches CUDA; `ann_difficulty.py` imports only numpy and scikit-learn
(`src/eval/ann_difficulty.py:27-29`), so no faiss is needed.

### 3a. The canonical profile

`eda_report` on `openai_250k.npy` at the conditions locked in
`gates/openai.yaml`: `--ann-max-rows 20000 --ann-k 100 --ann-hub-k 10
--ivf-nlist 256`. Fills the four rows of the measured-profile table in
`docs/datasets/openai.md`, read from `summary.json`.

### 3b. The noise floor

Ten disjoint 20,000-row draws cut from the 250k corpus, each passed through
`ann_difficulty.summary()`, reporting min, median and max per statistic.

This is the method already applied to sift and glove, and both times it
changed the conclusion: sift's relative contrast and IVF Gini turned out to
have noise larger than their gap to real, and glove's hubness skew swung
between 3.46 and 8.33 across draws of the *same real data* — which is now
issue #29. A statistic whose spread across redraws of real data exceeds the
distance a generator would have to close cannot carry a band, and finding
that out before training is much cheaper than finding it out after.

Output is a table of the four statistics against their spread, plus a plain
statement per statistic of whether a band is viable. No band is written.

### 3c. The overlay

One `eda_report` run with a second disjoint real draw overlaid as `real-b`.
This is the established idiom — the glyph panel's note already describes the
`real-a`/`real-b` pair as "the baseline for how much natural variation to
expect" — and it makes 3b's numbers legible as figures.

## Unit 4 — the design-input analysis

`scripts/openai_structure.py`, deliberately outside `src/eval/eda/`: these
questions are about openai specifically, and shared code should not grow to
answer them until they prove general. Writes one self-contained HTML file.

| Measurement | What it decides |
|---|---|
| Norm distribution, `max abs(norm - 1)` | Whether "already unit-norm" is true, or inherited from the docs |
| `norm(mean vector)`, cosine of each vector to the mean direction | Anisotropy: whether the corpus is a narrow cone rather than a sphere |
| Cosine similarity between random pairs | The geometry `metric: angular` actually refers to |
| PCA components at 90/95/99% variance, participation ratio | Whether `latent_dim: 512` is sized to the data |
| LID median, global two-NN intrinsic dimension | The same question locally, and the doc's "intrinsic dimension is low" claim |
| The four statistics recomputed under cosine, diffed against L2 | Whether the recorded profile survives phase (c) |
| PCA spectrum and cosine distribution after centering | Evidence for or against a `center: true` rung |

The anisotropy and cosine-similarity rows are the `mlp`-versus-`spherical`
evidence. `v0` uses an `mlp` generator with `l2_normalize: true`, which
projects onto the sphere after the fact. If the corpus occupies a narrow
cone — the usual finding for text embeddings — that is a harder target for
post-hoc projection than a well-spread sphere, and it is the concrete reason
to prefer a native spherical parameterisation in phase (b).

The angular-versus-L2 row has a prior worth stating so the measurement can
contradict it. For unit-norm vectors `||a-b||^2 = 2 - 2cos(a,b)`, a strictly
monotone map, so k-NN *sets* are identical under both metrics. Hubness
depends only on those sets and should be exactly invariant; IVF Gini should
be near-invariant; LID and relative contrast are ratios of distances and
should shift by a predictable factor. If that holds, the profile recorded
now survives phase (c) and issue #16 shrinks. If it does not, the reason is
worth knowing before any band is set.

## Deliverables

- `docs/datasets/openai.md` — the measured-profile table filled, with the
  conditions it was measured under.
- A self-contained HTML report, pulled back off the box.
- A written read on `v0`'s design: `latent_dim`, generator family,
  preprocessing.
- The noise-floor table, and per statistic whether a band is viable.

Split into three pull requests: the fetcher fix, the width guard, and the
analysis plus documentation update. They are independent and reviewable
separately; the fetcher fix is the only one that blocks anything.

## Risks

- **The box's `/venv/main` reports numpy 2.4.6 against the pinned 2.5.1.**
  The glove fetch succeeded there on 2026-08-10, so the runner probably uses
  a different environment. Confirm before trusting a measured number, rather
  than assuming.
- **9.5 GB of download and roughly 11 GB on disk**, on a box with 93 GB free
  and another agent holding ten queued GPU jobs. The cpu lane does not
  contend for the card, but it does share bandwidth and disk.
- **The HF parquet endpoint is a third-party interface** not covered by
  tests. If it changes, the fetcher's openai path breaks the same way the
  ann-benchmarks URL did. The failure is loud and at fetch time, which is
  the acceptable kind.

## Success criteria

1. `python -m src.data.fetch openai` writes `data/openai_250k.npy` with
   shape `[250000, 1536]`, as `docs/datasets/openai.md` already claims it
   does.
2. `make check` passes.
3. An `eda_report` run at 1536 dimensions produces a report without the two
   uninformative panels.
4. The four measured statistics are in `docs/datasets/openai.md`.
5. Each of the four carries a statement about whether its noise floor leaves
   room for a band, with the numbers behind it.
6. `gates/openai.yaml` still has every band `null`.
