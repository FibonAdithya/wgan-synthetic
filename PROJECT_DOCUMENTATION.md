# WGAN ANN-difficulty Synthetic Data Project Documentation

## Goal

Train Wasserstein GANs with gradient penalty (WGAN-GP) to produce synthetic
corpora that reproduce the *nearest-neighbour search difficulty* of six
benchmark families, so ANN algorithms can be developed and stressed without
the real corpora. A synthetic set succeeds when an index finds it as hard,
and hard in the same way, as the real set. It does not succeed by having
matching marginals.

The gate is ANN-difficulty parity on four statistics, measured per dataset at
that dataset's canonical N and k: LID (local intrinsic dimensionality),
relative contrast, hubness skew (k-occurrence), and IVF cell-balance Gini.
The distributional metrics below — `mmd_rbf`, `cov_fro`, `pairwise_hist_l1`
and the per-dimension marginals — are diagnostics. They explain why a gate
failed; they do not decide whether it passed.

They cannot decide it because they measure the wrong part of the distance
distribution. MMD, covariance error and the pairwise-distance histogram are
dominated by the bulk of that distribution, around the median, while ANN
difficulty is set by its far-left tail: the gap between a query's 1st and
k-th neighbour relative to the typical distance. An RBF kernel at median
bandwidth is nearly flat across that tail, so a large relative error there
barely moves the metric. Hubness is worse still — k-occurrence is a property
of the *directed* k-NN graph, and no symmetric two-sample statistic over
unordered distances constrains it at all. A generator is a smooth pushforward
of a Gaussian and tends to add a full-rank noise floor: globally invisible in
the covariance, locally it inflates LID toward the ambient dimension and
collapses relative contrast toward 1.

The full argument, including the failure modes that pass a marginals check in
both directions, is in
`docs/superpowers/specs/2026-08-04-multi-dataset-ann-emulation-design.md`.

Primary deliverable:

- Per dataset family, a trained generator checkpoint (`best_generator.pt`)
  and associated run metadata/config to reproducibly synthesize vectors at
  scale, together with the ANN-difficulty report that shows the synthetic set
  clears that family's gate.

---

## Datasets

Six benchmark families. Each has its own page under `docs/datasets/`,
carrying its structure, canonical N and k, measured profile, ladder and gate
bands; the pages are the source of truth for anything family-specific.

| Family | Dim | Metric | Structure | Model family | Page |
|---|---|---|---|---|---|
| `sift` | 128 | `l2` | non-negative uint8, heavy exact-zero mass, ties common | `gated` | `docs/datasets/sift.md` |
| `gist` | 960 | `l2` | non-negative dense float, little zero mass, high ambient dim | `mlp` | `docs/datasets/gist.md` |
| `deep` | 96 | `angular` | dense signed unit-norm image embeddings | `mlp` today, `spherical` when built | `docs/datasets/deep.md` |
| `glove` | 100 | `angular` | dense signed word vectors, strong density gradient | `mlp` today, `spherical` when built | `docs/datasets/glove.md` |
| `nytimes` | 256 | `angular` | dense signed document embeddings, strong topic clusters | `mlp` today, `spherical` when built | `docs/datasets/nytimes.md` |
| `openai` | 1536 | `angular` | unit-norm text embeddings, very high ambient dim, low intrinsic dim | `mlp` today, `spherical` when built | `docs/datasets/openai.md` |

### Fetching

`src/data/fetch.py` holds a single source registry with one entry per family,
each naming an ann-benchmarks HDF5 mirror, the dimension and the search
metric. Running

```bash
python -m src.data.fetch <dataset>
```

downloads that family's HDF5 into a shared cache once and cuts two
reproducible random subsets out of it, writing `data/<dataset>_250k.npy` and
`data/<dataset>_1m.npy`. The download is atomic — the body goes to a sibling
`.part` file and is `os.replace`d into position, so a concurrent reader sees
either nothing or a complete file — and single-flight, since the `.part` file
doubles as an exclusive lock and a second caller waits rather than starting
its own multi-gigabyte fetch. An existing destination is left alone; these
files are large and immutable. Subsets are drawn with a seeded RNG, so the
same seed gives the same rows.

All six come from ann-benchmarks HDF5 so the module handles one container
format. Descriptor sets obtained another way — corpus-texmex `.fvecs`, say —
are read directly by the loader and do not come through here.

---

## Technical choices

## Why WGAN-GP

- Wasserstein objective stabilizes adversarial training better than vanilla GAN losses on continuous vector spaces.
- Gradient penalty avoids brittle weight clipping and enforces approximate 1-Lipschitz critic behavior.

Critic objective:

- Maximize `E[D(real)] - E[D(fake)] - lambda_gp * GP`

Generator objective:

- Minimize `-E[D(fake)]`

Optional generator regularizer (implemented):

- Pairwise-distance mean matching on minibatches:
  - `L_G = adv_loss + alpha * |mean_pairdist(real) - mean_pairdist(fake)|`
  - Controlled by `training.distance_reg_alpha` and `training.distance_reg_max_points`.

---

## Data contract and preprocessing

Accepted input formats:

- `.fvecs` (Faiss style vectors)
- `.npy` with shape `[N, D]`, where `D` is the family's dimension

Implemented in:

- `src/data/dataset.py`

Preprocessing options:

- `center` (train-split mean subtraction)
- `whiten` (train-split covariance whitening)
- `l2_normalize` (per-vector normalization)

Training/eval split:

- Configurable holdout (`data.holdout_fraction`, default `0.05`)

### `data.metric`

A field in the `data` block, not under `data.preprocess`, taking `l2` or
`angular` and defaulting to `l2`. It records the distance the real corpus is
searched under — a property of the family, which is why it sits beside
`real_path` and `descriptor_dim` rather than among the transforms.

It is not a preprocessing instruction. `l2_normalize` is set independently,
and an `angular` corpus is not thereby normalized nor an `l2` one left alone;
the two settings answer different questions. The value is validated at load
time against the two accepted strings and is otherwise inert today: nothing
consumes it yet. Reading it in `src/eval/ann_difficulty.py`, so difficulty is
measured under the metric the corpus is actually searched with, is phase (c).

---

## Model architecture

Implemented in:

- Generator: `src/models/generator.py`
- Critic: `src/models/critic.py`

Both are MLPs with Linear + LeakyReLU blocks.

The models are dimension-agnostic. `data.descriptor_dim`, `model.latent_dim`
and both hidden-dim lists come from the config, so a 960d GIST config and a
1536d OpenAI config build the same architecture at a different width with no
code change.

The numbers below are what SIFT's configs use (`configs/sift/v0.yaml`), not a
repo-wide constant:

- `latent_dim: 128`
- Generator hidden dims: `[512, 1024, 1024]`
- Critic hidden dims: `[1024, 512, 256]`
- Output descriptor dimension: `128`

No sigmoid on critic output.

---

## Model variants: the per-dataset ladder

Every family has its own ladder, numbered independently from `v0`. Each rung
is exactly one config change from the one above it, so a difference visible
in an EDA overlay attributes to a single cause. Because the ladders are
independent, a variant number means nothing across families: SIFT's `v2` and
a future GIST `v2` are unrelated, and only ever compare within one dataset.
Each family's ladder and its status live in its page under `docs/datasets/`;
SIFT
is the only family with trained rungs today, and the other five have a `v0`
baseline config only.

The SIFT ladder:

| Variant | Delta from previous | Config | Runs |
|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/sift/v0.yaml` | `long_baseline`, `bench_baseline` |
| `v1` | + generator EMA (`ema_decay: 0.999`) | `configs/sift/v1.yaml` | `long_ema_only`, `x100k_ema_only` |
| `v1_5` | + distance reg (`distance_reg_alpha: 0.1`, 256 points) | `configs/sift/v1_5.yaml` | `long_improved`, `x100k_improved`, `bench_improved` |
| `v2` | + gated generator (`generator_type: gated`) | `configs/sift/v2.yaml` | `x100k_sparse_clamp4` |

Run length is an independent axis and is not a variant: `bench_*` are 3k
generator steps, `long_*` are 30k, `x100k_*` are 100k. The run directory
names predate this scheme and are kept as-is because the artifacts under
them are already named that way.

The four `configs/sift/` configs above are the variant definitions, all at 30k
steps. Two further configs are run-length or ablation arms of them, not
variants of their own:

| Config | What it is |
|---|---|
| `configs/x100k_gated.yaml` | v2 at 100k steps with `logit_clamp: 10.0`, the value the design called for. Untrained — the v2 run that exists (`x100k_sparse_clamp4`) used 4.0, which is what `configs/sift/v2.yaml` reproduces. Kept so the clamp comparison can be run. |
| `configs/wgan_gp_sift1m_smoke_improved.yaml` | 200-step smoke test on synthetic data (`synthetic_if_missing: true`), with EMA, the distance regularizer, `num_workers` and the collapse monitor all switched on, so the new training-loop paths get exercised without the dataset. Small model, unrelated hyperparameters to the variants above — not a variant and not for evaluation. Output lands in `runs/wgan_sift1m_smoke_improved`. |

### Why v2 exists

Raw SIFT descriptors carry heavy mass at exactly zero. A dense MLP generator
cannot reproduce that support — it emits smooth values everywhere — and the
critic does not reliably penalize it, so Wasserstein estimates look
flattering while the marginals are plainly wrong. v2's generator multiplies a
softplus magnitude by a sampled binary gate, producing exact zeros. See
`src/models/generator.py` (`GatedGenerator`).

### `generator_type`

The architecture axis in the `model` config block, accepting `mlp` (default)
and `gated`. It sits underneath the variant numbering: v0, v1 and v1_5 all
use `mlp` and differ only in training settings.

A third value, `spherical`, is planned and not built. It is phase (b) of the
multi-dataset design: a generator whose output is unit-norm by construction
rather than by a normalization applied afterwards, for the four `angular`
families. Until it exists, `deep`, `glove`, `nytimes` and `openai` all start
their ladders on `mlp`, and any dataset page naming `spherical` is describing
the intended rung, not a trained one.

Checkpoints do not record `generator_type` — the architecture is rebuilt from
the run config at load time. A checkpoint is therefore only loadable
alongside the `run_config.yaml` written next to it. Checkpoints do record
`generator_weights` (`"live"` or `"ema"`), which says which weights the file
holds, not which architecture produced them.

---

## Optimizer and training setup

Default (current promoted config):

- `n_critic: 3`
- `lr_g: 1e-4`
- `lr_d: 1e-4`
- `betas: (0.0, 0.9)`
- `lambda_gp: 5.0`
- `num_gen_steps: 3000`
- `batch_size: 512`
- `distance_reg_alpha: 0.0` (disabled by default)

Training entrypoint:

- `src/train/train_wgan_gp.py`

---

## Device behavior

Auto device selection (implemented):

1. CUDA (if available)
2. MPS (Apple Metal, if available)
3. CPU fallback

Applied in:

- `src/train/train_wgan_gp.py`
- `src/sample/generate.py`
- `src/eval/evaluate_distribution.py`

---

## Evaluation stack

## ANN difficulty — the gate

Implemented in `src/eval/ann_difficulty.py` and surfaced as panels in the EDA
report. This is the decision procedure: whether a synthetic set would
*behave* like the real one under nearest-neighbour search. Everything under
"Metrics" below is diagnostic and explains a failure here.

Four statistics, compared real against synthetic:

- LID median (local intrinsic dimensionality)
- relative contrast
- hubness skew (k-occurrence)
- IVF cell-balance Gini

Pass is a documented relative band per statistic, recorded in that family's
page under `docs/datasets/`, not a single combined score. The four fail in
different directions — a smoothed generator inflates LID and collapses
contrast, a collapsed one does the reverse — and one number would hide which
broke. Bands start wide and tighten as a family's ladder shows what is
achievable; a family with no trained ladder records its bands as unset.

Canonical N and k are locked per dataset and written into its page. These are
self-queried subsample statistics with no absolute meaning: they are
comparable only within one report at one N and k, and are not comparable with
published figures, which are measured on the full corpus against the real
query set. Without a locked pair, a gate result from last month cannot be
read against today's.

`ann_difficulty.py` currently computes everything under L2, including for the
four `angular` families. Reading `data.metric` and measuring under the
corpus's own distance is phase (c) of the multi-dataset design; until it
lands, angular-family numbers are internally consistent within a report but
are not the distance the corpus is searched with.

The knobs (`--ann-k`, `--ann-hub-k`, `--ann-max-rows`, `--ivf-nlist`) are
documented under "Visualization tools" with the EDA report that exposes them.

## Checkpoint-based eval

- Script: `src/eval/evaluate_distribution.py`
- Inputs: real dataset + checkpoint + run config
- Samples fake vectors directly from generator.

## File-to-file eval

- Script: `src/eval/evaluate_file_to_file.py`
- Inputs: real file + synthetic file
- Compares dataset artifacts directly (no model required).

## Metrics

Diagnostics, not the gate. A synthetic set can score well on all of these and
still be far easier or harder to search than the real one; their use is to
localize a difficulty mismatch once the gate above has flagged it.

- `mean_l2`, `var_l2`
- `cov_fro`
- `mmd_rbf`
- `pairwise_hist_l1`
- `knn_recall`
- `ann_proxy_recall`

### Metric definitions

Let `X = {x_i}` be real samples and `Y = {y_j}` be synthetic samples, with vectors in `R^D` for the dataset's dimension `D`.

- `mean_l2` (lower is better)
  - Per-dimension mean mismatch:
  - `||mu_X - mu_Y||_2`, where `mu_X = (1/|X|) sum_i x_i`.

- `var_l2` (lower is better)
  - Per-dimension variance mismatch:
  - `||var_X - var_Y||_2`.

- `cov_fro` (lower is better)
  - Full covariance mismatch:
  - `||Sigma_X - Sigma_Y||_F`, where `Sigma` is sample covariance.

- `mmd_rbf` (lower is better)
  - Maximum Mean Discrepancy with RBF kernel:
  - `MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]`
  - `k(a,b) = exp(-gamma * ||a-b||^2)`.
  - Captures global distribution mismatch beyond first/second moments.

- `pairwise_hist_l1` (lower is better)
  - Compare histograms of pairwise distances:
    - real-real distances (`RR`)
    - real-fake distances (`RF`)
  - Metric is L1 gap between density histograms over shared bins:
  - `sum_b |hist_RR[b] - hist_RF[b]|`.
  - This is the most directly aligned metric with the distance-CDF objective.

- `knn_recall` (higher is better)
  - For each real query `q`, compute:
    - `r_k(q)`: distance to k-th nearest neighbor in real train set
    - `d_fake(q)`: distance to nearest synthetic point
  - Count hit if `d_fake(q) <= r_k(q)`.
  - Report mean hit rate over queries.

- `ann_proxy_recall` (higher is better)
  - Compares average top-k neighborhood distances for queries in real vs synthetic indices.
  - Uses a normalized ratio mapped to `(0,1]` via `exp(-|ratio-1|)`.
  - Closer to `1` indicates synthetic neighborhoods have similar scale to real.

Practical interpretation:

- Use `pairwise_hist_l1` + CDF plots for distance-structure matching.
- Use `mmd_rbf` and `cov_fro` for global distributional similarity.
- Use `knn_recall` / `ann_proxy_recall` for neighborhood utility.

Memory-safe note:

- Pairwise histogram computation is chunked to avoid allocating large `[n, n, d]` tensors.
- MMD computation is internally capped by sample count for safety.

---

## Visualization tools

- Distance CDF envelope plot:
  - `src/eval/plot_distance_cdf_pillow.py`
  - Plots q10/q50/q90 CDF curves for real vs synthetic
  - Supports config label and caption in the plot.

- Embedding/clustering visualization:
  - `src/eval/plot_embedding_clusters.py`
  - t-SNE (or UMAP if installed) for real and synthetic subsets.

- Distributional EDA report:
  - `src/eval/eda_report.py`
  - One self-contained interactive HTML file (plotly bundled inline, opens
    offline) plus a `summary.json` and best-effort PNGs.
  - Panels: descriptor glyphs, local intrinsic dimensionality and relative
    contrast (ANN difficulty), hubness (k-occurrence), IVF cell balance,
    pooled value distribution, per-dimension marginals with a dropdown over
    every dimension, per-dim mean/std/zero-rate profiles, pairwise distances,
    within-set kNN distances, PCA spectrum, correlation heatmaps, and a
    Wasserstein-1 ranking of the worst-matching dimensions.
  - The descriptor glyph panel comes first because it frames the rest: every
    other panel is an aggregate over tens of thousands of vectors, and all of
    them can look healthy while the generator emits descriptors that are
    structurally wrong. It draws a handful of individual descriptors instead
    -- two real rows and one per synthetic set -- using the geometry in
    `src/eval/descriptor_glyph.py`. `--glyph-samples` (default 8) sets
    descriptors per row, and `0` turns the panel off. It is skipped
    automatically unless every series is 128-dimensional and large enough for
    its rows, since the (cell, orientation bin) mapping exists only for SIFT
    descriptors while the rest of the report is dimension-agnostic.
  - Caveat the glyph panel cannot check: it assumes the arrays it is handed
    are raw descriptors. `eda_report` sees materialised `.npy` files, not run
    configs, so a set that was centered or whitened before being written out
    no longer maps dimension to (cell, orientation bin) and would be drawn as
    a plausible-looking lie. `plot_descriptor_grid` reads the run config and
    refuses such a run outright; prefer it when that risk is live.
  - The ANN-difficulty panels carry the gate described above; their knobs
    are set here. `--ann-k` (default 100) sets the neighbour depth
    for LID and relative contrast, `--ann-hub-k` (default 10) the depth for
    the hubness k-occurrence count, `--ann-max-rows` (default 20000) the
    equal-N truncation every set is cut to so the metrics stay comparable
    across series, and `--ivf-nlist` (default 256) the cluster count for the
    IVF cell-balance panel. These numbers are self-queried subsample
    statistics, not published benchmark figures, and are only comparable
    across the series in one report; each family's locked values are in its
    page under `docs/datasets/`.
  - `--synthetic-path` is optional; without it the report is pure dataset EDA.
    With it, every panel overlays the two so mismatch is visible by eye.
  - `--preprocess l2` (default) matches the training contract, since generator
    output is unit-norm. Use `--preprocess none` to inspect raw integer SIFT.
  - Purpose: reject a generator by eye when the critic cannot separate the
    sets. A weak critic yields flattering Wasserstein estimates over samples
    whose marginals are plainly wrong -- most visibly SIFT's heavy exact-zero
    mass, which smooth generators do not reproduce.
  - `src/eval/compare_variants.py` drives this across all four SIFT variants
    at once, labelling the overlays `v0`/`v1`/`v1_5`/`v2` to match the SIFT
    ladder. It resolves each variant's `best_generator.pt` and
    `run_config.yaml`, samples the generator, and calls the report in
    process. Variants whose checkpoints are not on the local machine are
    skipped with a message, so a partial comparison still produces a report.
    Each variant's latents are seeded from `--seed` and its own name, so a
    variant's samples do not change depending on which other variants were
    present. `--num-samples` defaults to `--max-vectors`, since the report
    subsamples to that; raise it only to keep a larger `.npy` under
    `<output-dir>/samples`.

```bash
.venv/bin/python -m src.eval.eda_report \
  --real-path data/sift_base.npy \
  --synthetic-path runs/bench_improved/synthetic_1m.npy \
  --output-dir runs/bench_improved/eda
```

```bash
.venv/bin/python -m src.eval.compare_variants \
  --real-path data/sift_base.npy \
  --output-dir runs/eda_variants
```

PNG export uses kaleido, which drives a headless Chrome. Without a Chrome
install the HTML report is still written and the export is skipped with a
message; run `plotly_get_chrome` once if the static images are wanted, or pass
`--no-png` to skip it outright.

`--plotlyjs` controls how plotly.js ships, which matters when pulling reports
off a remote training box:

- `inline` (default) -- self-contained, opens offline, ~4.5MB overhead.
- `cdn` -- roughly 4x smaller, needs internet to view. Use over a slow link.
- `directory` -- writes `plotly.min.js` once beside the report; several
  reports in one output directory then share a single copy.

When running on a remote box, generating with `--plotlyjs cdn` and gzipping
before transfer took the three-report set from 5.7MB to 1.5MB.

- Descriptor glyph grid (standalone):
  - `src/eval/plot_descriptor_grid.py`
  - The same figure as the `eda_report` glyph panel -- `eda_report.
    fig_descriptor_glyphs` is the single implementation both use -- but
    sourced differently. This CLI loads generator checkpoints and samples
    them directly, so it works with no materialised `.npy` files and can read
    each run config. That is what lets it refuse a run trained with centering
    or whitening, a check `eda_report` structurally cannot make. Use the
    report panel in the normal `compare_variants` flow; use this when
    rendering straight from a checkpoint, or when the preprocessing of a set
    is in doubt.
  - Every aggregate panel is over tens of thousands of vectors. This
    one instead draws individual SIFT descriptors as glyphs: each 128-value
    descriptor becomes a 4x4 grid of spatial cells, and each cell an 8-ray
    star, one ray per orientation bin, using the index convention
    `index = (row * 4 + col) * 8 + orientation_bin`. Ray length is a shared
    99th-percentile scale computed across every descriptor in the figure
    (not per-glyph normalisation, which would make a flat generated
    descriptor look as structured as a real one), and is clipped so a ray
    never crosses into a neighbouring cell.
  - Writes `descriptor_grid.html` into `--output-dir`, plus a `png/`
    subdirectory unless `--no-png`. The figure has two rows of real
    descriptors (`real-a`, `real-b`) above one row per resolvable variant
    checkpoint.
  - How to read it: real SIFT is sparse and spiky, with most cells dominated
    by one or two directions. Even, bushy stars mean the generator matched
    the marginals without the structure. **Red rays are negative bins --
    impossible for a gradient histogram**, and expected from v0/v1/v1_5,
    which use the unactivated MLP generator. The real-a/real-b pair is the
    baseline for how much natural variation to expect before comparing it to
    a variant row.
  - Negative rays are drawn at a minimum length
    (`descriptor_glyph.NEGATIVE_RAY_FLOOR`, 35% of the half-cell) rather than
    their true one. Measured on the real checkpoints, v0/v1/v1_5 put around
    10% of their bins below zero but at a median magnitude of 0.003 against a
    scale reference of 0.26 -- roughly 0.4px in a 2400px export, which made
    the figure's headline defect invisible. Length therefore does **not**
    encode magnitude for negative rays; their presence and count are what to
    read. Positive rays keep the honest shared scale, since flooring those
    would give every near-zero bin a ray and erase the real-vs-generated
    sparsity difference.
  - It refuses to run against a variant trained with centering or
    whitening, or one that generates a width other than 128, because either
    breaks the dimension-to-(cell, bin) mapping the glyph depends on. It also
    refuses a real or generated array containing NaN or inf, since either
    would draw a spurious or blank glyph rather than a true picture of the
    descriptor. A variant whose checkpoint or run config is not on this
    machine is skipped with a printed message, like `compare_variants`.
  - Flags: `--num-samples` (default 8) sets descriptors per row; `--seed`
    (default 42); `--root` (default `.`), the repo root that variant config
    and run paths resolve against; `--real-format` (default `auto`, or `npy`
    / `fvecs`), same meaning as in `eda_report`; `--plotlyjs` (default
    `inline`), likewise. PNG export follows the same best-effort
    stance as `eda_report`: without a Chrome install the HTML report is still
    written and the export is skipped with a printed message.

```bash
.venv/bin/python -m src.eval.plot_descriptor_grid \
    --real-path data/sift_base.npy \
    --output-dir runs/descriptor_grid
```

---

## Run artifact structure

Typical run directory contents:

- `best_generator.pt`
- `checkpoint_step_*.pt` (periodic snapshots)
- `run_config.yaml`
- `run_metadata.json`
- `synthetic_1m.npy` (if generated)
- `eval/metrics.json`
- `eval/distance_cdf*.png`
- `eval_file_to_file/metrics.json`
- `eval_embeddings/*.png`

Current promoted default run:

- `runs/wgan_sift1m_real_default/`

---

## Workflow for a new user (historical SIFT walkthrough)

This walkthrough predates the multi-dataset reframe and only ever covered
SIFT. For any of the six families, including SIFT, `README.md`'s quick start
(fetch, then train `configs/<dataset>/v0.yaml`) is the current path; this
section is kept because steps 4-7 below (sampling, file-to-file eval, CDF
plot, t-SNE) are still accurate once you have a checkpoint, and are spelled
out here in more detail than the README.

## 1) Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Fetch the dataset

```bash
python -m src.data.fetch sift
```

Writes `data/sift_250k.npy` and `data/sift_1m.npy`. See `data/README.md` for
what the fetcher does and `docs/datasets/sift.md` for the family's specifics.

## 3) Train with the SIFT `v0` config

```bash
.venv/bin/python -m src.train.train_wgan_gp --config configs/sift/v0.yaml
```

`configs/sift/v0.yaml` names `data/sift_base.npy` as `data.real_path` — the
corpus the trained checkpoints in this repo actually used, not the fetched
`data/sift_250k.npy` subset above. Point `real_path` at whichever file you
have, per `FOLLOWUPS.md` ("`data.real_path` names a file the fetcher does not
produce").

## 4) Generate synthetic dataset

```bash
.venv/bin/python -m src.sample.generate \
  --checkpoint runs/wgan_sift1m_real_default/best_generator.pt \
  --config runs/wgan_sift1m_real_default/run_config.yaml \
  --num-samples 1000000 \
  --output-path runs/wgan_sift1m_real_default/synthetic_1m.npy
```

## 5) Run file-to-file evaluation

```bash
.venv/bin/python -m src.eval.evaluate_file_to_file \
  --real-path data/sift_base.fvecs \
  --real-format fvecs \
  --synthetic-path runs/wgan_sift1m_real_default/synthetic_1m.npy \
  --synthetic-format npy \
  --output-dir runs/wgan_sift1m_real_default/eval_file_to_file \
  --num-samples 5000
```

## 6) Generate distance CDF visualization

```bash
.venv/bin/python -m src.eval.plot_distance_cdf_pillow \
  --real-path data/sift_base.fvecs \
  --real-format fvecs \
  --synthetic-path runs/wgan_sift1m_real_default/synthetic_1m.npy \
  --config-path runs/wgan_sift1m_real_default/run_config.yaml \
  --config-label wgan_gp_sift1m_real_default \
  --caption "default promoted configuration" \
  --num-queries 200 \
  --num-targets 5000 \
  --output-path runs/wgan_sift1m_real_default/eval/distance_cdf_labeled.png
```

## 7) Generate t-SNE visualizations

```bash
.venv/bin/python -m src.eval.plot_embedding_clusters \
  --real-path data/sift_base.fvecs \
  --real-format fvecs \
  --synthetic-path runs/wgan_sift1m_real_default/synthetic_1m.npy \
  --synthetic-format npy \
  --method tsne \
  --sample-size 3000 \
  --output-dir runs/wgan_sift1m_real_default/eval_embeddings
```

---

## Reproducibility notes

- Seed is configured in each YAML and saved in `run_metadata.json`.
- Always keep:
  - checkpoint
  - run config
  - run metadata
  together for reproducible sampling/evaluation.

---

## Tuning guidance (no model size increase)

Most impactful levers observed:

- `lambda_gp` (around 4-6 was useful in focused sweeps)
- `n_critic` and critic LR
- training length (`num_gen_steps`)
- optional distance regularizer (`distance_reg_alpha`)

Use separate output directories for every sweep run to avoid artifact overwrite.
