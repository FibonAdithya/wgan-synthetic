> **AI-generated working note.** Written by Claude during development and kept
> for provenance. It is not the source of truth: where this file disagrees with
> `PROJECT_DOCUMENTATION.md`, the latter wins.

# Multi-dataset ANN-difficulty emulation

Date: 2026-08-04
Branch: `ann/difficulty-panels`

## Problem

The repo trains a WGAN-GP on SIFT1M and is documented as if that were the
goal. It is not. The goal is to produce synthetic corpora that reproduce the
nearest-neighbour search difficulty of the benchmark sets people develop ANN
algorithms against — SIFT, GIST, DEEP Image, GloVe, NYTimes and OpenAI
embeddings — so those algorithms can be developed and stressed without the
real corpora.

Three gaps follow from the mismatch:

1. **Framing.** `README.md` and `PROJECT_DOCUMENTATION.md` name SIFT1M
   throughout, present the distributional metrics as the deliverable, and
   treat `src/eval/ann_difficulty.py` as one panel group among many. The ANN
   metrics are the point; everything else is diagnosis.
2. **Dataset knowledge lives nowhere.** `src/deep/download.py` was added for
   DEEP alone. Nothing records what the other four sets are, what shape they
   have, what their real ANN-difficulty profile is, or which model family
   suits them.
3. **SIFT-shaped code.** `src/data/sift1m_dataset.py` is dimension-agnostic
   but named for one dataset. `generator_type` offers `mlp` and `gated`, both
   of which suit L2 non-negative data; nothing suits the four angular
   families. `ann_difficulty.py` assumes L2 distance. Configs and run
   directories have no dataset axis.

## What this repo is for

A synthetic set succeeds when an ANN algorithm finds it as hard, and hard in
the same way, as the real set. It does not succeed by having matching
marginals.

This inverts the current metric hierarchy. `mmd_rbf`, `cov_fro`,
`pairwise_hist_l1` and the per-dimension panels become diagnostics that
explain a failed gate; the gate itself is ANN-difficulty parity.

### Why distributional fidelity does not imply ANN difficulty

Recorded here because it is the single assumption the whole design rests on.

- **Scale mismatch.** MMD, covariance error and the pairwise-distance
  histogram are dominated by the bulk of the distance distribution, around
  the median. ANN difficulty is set by the far-left tail: the gap between a
  query's 1st and k-th neighbour relative to the typical distance. An RBF
  kernel at median bandwidth is nearly flat across that tail, so a large
  relative error there barely moves the metric.
- **Ratios of small differences.** Relative contrast is a ratio of mean to
  nearest-neighbour distance; LID is estimated from ratios within the top-k
  distances. Matching the first two moments exactly leaves both unconstrained.
- **Symmetric statistics cannot see hubness.** k-occurrence is a property of
  the directed k-NN graph. No two-sample statistic over unordered distances
  constrains it. Hubness arises from mild density gradients that coarse
  density matching does not reproduce.
- **Smoothing is invisible globally, fatal locally.** A generator is a smooth
  pushforward of a Gaussian and tends to add a small full-rank noise floor.
  Globally that hardly moves the covariance; locally it makes every
  neighbourhood look isotropic and full-dimensional, inflating LID toward the
  ambient dimension and collapsing relative contrast toward 1. The mirror
  failure, partial mode collapse, makes the synthetic set artificially easy.
  Both pass a marginals check.
- **Support and ties.** Real SIFT is quantized non-negative integers with
  heavy exact-zero mass, so points sit on a lattice and exact ties are common.
  Real text and image corpora carry near-duplicates. Both dominate the top of
  the neighbour list. IID samples from a smooth generator are in general
  position — no ties, no duplicates. This is what the `gated` generator was
  built to attack.
- **N-dependence.** Metrics computed on a 20k subsample say nothing about the
  1M-point neighbour graph. Nearest-neighbour distance shrinks with N at a
  rate governed by intrinsic dimension, so two sets that agree at 20k can
  diverge sharply at 1M — and 1M is where the algorithm runs.

## Dataset catalogue

Fixed list. No plugin extensibility; adding a seventh family is a code change.

| Family | Dim | Metric | Structure | Model family | Source |
|---|---|---|---|---|---|
| SIFT1M | 128 | L2 | non-negative, uint8-quantized, heavy exact-zero mass, ties | `gated` | corpus-texmex `.fvecs` |
| GIST1M | 960 | L2 | non-negative dense float, little zero mass, high dim | `mlp`, `gated` as a ladder rung | corpus-texmex `.fvecs` |
| DEEP Image | 96 | angular | dense signed, unit-norm | `spherical` | ann-benchmarks HDF5 |
| GloVe | 100 | angular | dense signed; frequency-driven density gradient, strong hubness | `spherical` | ann-benchmarks HDF5 |
| NYTimes | 256 | angular | dense signed, document embeddings, cluster-heavy | `spherical` | ann-benchmarks HDF5 |
| OpenAI | 1536 | angular | unit-norm, very high ambient dim, low intrinsic dim | `spherical` | dbpedia-openai set |

GIST is listed apart from SIFT despite the shared provenance: 960-dimensional
non-negative float behaves nothing like 128-dimensional quantized
non-negative, and its model family is an open question the ladder settles.

Exact URLs, HDF5 keys and split names are verified when `src/data/fetch.py`
is written, not assumed from this table.

### Per-dataset documentation

Each family gets `docs/datasets/<name>.md`, human-maintained, holding:

- Source URL and provenance, and how to fetch it.
- Measured shape, dtype, zero rate and norm distribution — read from the file,
  not quoted from a paper.
- The **real** ANN-difficulty profile at the dataset's canonical N and k:
  LID median, relative contrast, hubness skew, IVF cell-balance Gini.
- The model family it uses and why.
- Its variant ladder and which run directory holds each rung.
- Current gate status: real-vs-synthetic for each of the four statistics.

## Code shape

Native dimension throughout. Hidden dimensions scale from config; high-dim
families cost more and may need retuning, which the ladder surfaces.

### `src/data/fetch.py`

Generalizes `src/deep/download.py`. The valuable part of that module is its
acquisition discipline and it is kept verbatim in behaviour:

- Atomic — body written to a sibling `.part` file and `os.replace`d, so a
  concurrent reader sees a complete file or none.
- Single-flight — the `.part` file doubles as an `O_EXCL` lock, so a second
  caller waits rather than starting its own multi-gigabyte fetch, bounded by
  a timeout.
- Output is `.npy`, which the existing loader already reads, so neither the
  loader nor the trainer learns about HDF5 or `.fvecs` acquisition.

What generalizes: a source table keyed by dataset name giving URL, container
format (HDF5 key, or `.fvecs`), dimension and default subset sizes. Subsets
are written as `data/<dataset>_<rows>.npy` with a seeded row sample.

`src/deep/download.py` is removed once `fetch.py` covers DEEP; its CLI
behaviour is preserved under the new entry point rather than kept as a shim.

### `src/data/dataset.py`

`sift1m_dataset.py` renamed. The module is already dimension-agnostic — the
name is the only SIFT-specific thing about it. The preprocess config grows a
`metric` field (`l2` | `angular`) that evaluation reads, so distance choice
follows the dataset rather than being hardcoded per call site.

### `src/models/generator.py`

`generator_type` becomes the model-family axis, chosen per dataset:

- `mlp` — dense signed baseline. Unchanged.
- `gated` — softplus magnitude times a sampled binary gate, giving exact
  zeros. Unchanged. SIFT, and GIST if its ladder shows non-negativity matters.
- `spherical` — **new.** Explicit unit-norm output projection, so the
  generator spends capacity on direction rather than on learning to land near
  the sphere. Serves the four angular families.

`spherical` is the one genuinely new model and the main implementation risk:
projecting to the sphere changes the gradient geometry the critic sees, and a
naive normalize-at-the-end can starve the magnitude path. Its ladder starts
from `mlp` on the same dataset so the delta is attributable.

### Configs and runs

- `configs/<dataset>/v0.yaml`, `v1.yaml`, … Existing `configs/sift_gan_v*.yaml`
  move under `configs/sift/`.
- `runs/<dataset>/<variant>_<length>`. Existing run directories stay where
  they are; the docs record the old names as historical, as they already do.

### Evaluation

`src/eval/ann_difficulty.py` is dataset-agnostic in shape but assumes L2. The
metric comes from the dataset config instead, and the angular path uses cosine
distance for the k-NN cache, LID, relative contrast and hubness. IVF
cell-balance clustering follows the same metric.

`src/eval/compare_variants.py` gains a `--dataset` argument selecting which
ladder to resolve, replacing the hardcoded four-variant SIFT table.

## The gate

Per dataset, at that dataset's canonical N and k, four statistics compared
real against synthetic:

- LID median
- relative contrast
- hubness skew (k-occurrence)
- IVF cell-balance Gini

**Pass is a documented relative band per statistic, not a single combined
score.** The four fail in different directions — a smoothed generator inflates
LID and collapses contrast, a collapsed one does the reverse — and collapsing
them into one number hides which broke. Bands are recorded per dataset in its
doc page and start deliberately wide, tightened as a dataset's ladder shows
what is achievable.

**Canonical N and k are locked per dataset** and written into the dataset doc.
These metrics are self-queried subsample statistics with no absolute meaning;
they are comparable only within one report at one N and k. Without a locked
pair, a gate result from last month cannot be read against today's. The lock
is cheap to revise — it is a number in a doc — and the cost of not having one
is that no gate result outlives its report.

## Testing

- `fetch.py`: subset determinism under a fixed seed; the single-flight lock
  path (second caller waits, stale `.part` times out); atomicity, i.e. no
  destination file exists after an interrupted body write. Network fetch
  itself is stubbed.
- `spherical` generator: output rows are unit-norm to tolerance; gradients
  flow to all layers; the factory builds it from config at several dimensions.
- Metric-aware eval: cosine and L2 paths agree on L2-normalized input, where
  the two orderings coincide; LID and contrast on a synthetic set of known
  intrinsic dimension land near the analytic value.
- Existing SIFT tests must pass unchanged after the rename, which is what
  proves the rename was a rename.

## Scope

More than one implementation plan. Split into three, in order:

- **(a) Docs, fetch, rename.** README and `PROJECT_DOCUMENTATION.md`
  reframed; `docs/datasets/*.md` created; `src/data/fetch.py`;
  `sift1m_dataset.py` → `dataset.py`; configs and runs reorganized.
- **(b) `spherical` generator.** Model, factory wiring, tests, and a first
  ladder on DEEP, which is the smallest angular set.
- **(c) Metric-aware evaluation.** Angular distance through
  `ann_difficulty.py`, `--dataset` in `compare_variants.py`, gate bands
  recorded.

(a) is a prerequisite for both others. (b) and (c) are independent.

## Out of scope

- Downstream index benchmarking — measuring recall-versus-latency on a real
  IVF or HNSW index over synthetic data. It is the most direct test of the
  thesis and worth doing later, but it is a separate deliverable from the
  generator work.
- Retuning the existing SIFT ladder. It stays as-is; the reorganization must
  not change its results.
- Any seventh dataset family.
