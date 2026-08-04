# WGAN SIFT1M Synthetic Data Project Documentation

## Goal

Train a Wasserstein GAN with gradient penalty (WGAN-GP) to generate synthetic 128D vectors that match the distributional and neighborhood structure of SIFT1M descriptors.

Primary deliverable:

- A trained generator checkpoint (`best_generator.pt`) and associated run metadata/config to reproducibly synthesize vectors at scale.

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
- `.npy` with shape `[N, 128]`

Implemented in:

- `src/data/sift1m_dataset.py`

Preprocessing options:

- `center` (train-split mean subtraction)
- `whiten` (train-split covariance whitening)
- `l2_normalize` (per-vector normalization)

Training/eval split:

- Configurable holdout (`data.holdout_fraction`, default `0.05`)

---

## Model architecture

Implemented in:

- Generator: `src/models/generator.py`
- Critic: `src/models/critic.py`

Both are MLPs with Linear + LeakyReLU blocks.

Current default config (`configs/wgan_gp_sift1m_real.yaml`):

- `latent_dim: 128`
- Generator hidden dims: `[512, 1024, 1024]`
- Critic hidden dims: `[1024, 512, 256]`
- Output descriptor dimension: `128`

No sigmoid on critic output.

---

## Model variants

Four variants were trained. Each is exactly one config change from the one
above it, so a difference visible in an EDA overlay attributes to a single
cause.

| Variant | Delta from previous | Config | Runs |
|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/sift_gan_v0.yaml` | `long_baseline`, `bench_baseline` |
| `v1` | + generator EMA (`ema_decay: 0.999`) | `configs/sift_gan_v1.yaml` | `long_ema_only`, `x100k_ema_only` |
| `v1_5` | + distance reg (`distance_reg_alpha: 0.1`, 256 points) | `configs/sift_gan_v1_5.yaml` | `long_improved`, `x100k_improved`, `bench_improved` |
| `v2` | + gated generator (`generator_type: gated`) | `configs/sift_gan_v2.yaml` | `x100k_sparse_clamp4` |
| `v3` | + structured gate (`generator_type: structured_gated`) | `configs/sift_gan_v3.yaml` | *(untrained)* |
| `v4` | + log-ratio regularizer (`lid_reg_alpha > 0`) | `configs/sift_gan_v4.yaml` | *(untrained)* |

Run length is an independent axis and is not a variant: `bench_*` are 3k
generator steps, `long_*` are 30k, `x100k_*` are 100k. The run directory
names predate this scheme and are kept as-is because the artifacts under
them are already named that way.

The `sift_gan_*` configs above are the variant definitions, all at 30k
steps. Further configs are run-length or ablation arms of them, not
variants of their own:

| Config | What it is |
|---|---|
| `configs/x100k_gated.yaml` | v2 at 100k steps with `logit_clamp: 10.0`, the value the design called for. Untrained — the v2 run that exists (`x100k_sparse_clamp4`) used 4.0, which is what `sift_gan_v2.yaml` reproduces. Kept so the clamp comparison can be run. |
| `configs/x100k_structured.yaml` | v3 at 100k steps — the arm the experiment plan calls for. Sole delta from `sift_gan_v3.yaml` is run length and logging cadence. |
| `configs/wgan_gp_sift1m_smoke_improved.yaml` | 200-step smoke test on synthetic data (`synthetic_if_missing: true`), with EMA, the distance regularizer, `num_workers` and the collapse monitor all switched on, so the new training-loop paths get exercised without the dataset. Small model, unrelated hyperparameters to the variants above — not a variant and not for evaluation. Output lands in `runs/wgan_sift1m_smoke_improved`. |

### Why v2 exists

Raw SIFT descriptors carry heavy mass at exactly zero. A dense MLP generator
cannot reproduce that support — it emits smooth values everywhere — and the
critic does not reliably penalize it, so Wasserstein estimates look
flattering while the marginals are plainly wrong. v2's generator multiplies a
softplus magnitude by a sampled binary gate, producing exact zeros. See
`src/models/generator.py` (`GatedGenerator`).

### Why v3 exists

v2 produces exact zeros but gets their *distribution* wrong. Its gates are
sampled independently per coordinate, so the non-zero count per vector is
Binomial(128, p) with standard deviation about 4.76. Real SIFT measures
14.45 -- three times as variable -- and its zero pattern is locally
correlated: +0.32 between adjacent orientation bins and +0.27 between the
same bin in neighbouring spatial cells. Neither is reachable by tuning v2.

v3 adds a per-vector sparsity level (over-dispersion), a 3x3x3 convolution
over the (4,4,8) descriptor grid with circular orientation padding (local
correlation), and fixed smoothing of the gate noise so sampling is
correlated too. Measurements are in `tools/probes/`; the design is in
`docs/superpowers/specs/2026-08-03-structured-gate-generator-design.md`.

`noise_kernel_sigma` is calibrated, not tuned during training: the noise
kernel is deliberately fixed (a learnable one could be driven to zero,
killing gate stochasticity), so the value has to be right up front. Sweeping
sigma against the measured correlation profile puts it at `0.65`, which
reproduces +0.329 at separation 1 and +0.271 at offset 8 against the real
+0.317 and +0.275.

### Why v4 exists

v3 bets that the right architecture reaches SIFT's local geometry without
being driven there. Nothing in the objective sees local structure: the critic
is pointwise (`128 -> 1`) and judges each vector alone, and v1_5's distance
regularizer matches one global scalar. A generator can satisfy both while
getting neighbourhood geometry wrong.

v4 hedges that bet. Within each minibatch it computes every point's `k`
nearest neighbours *among the other points in that batch* and matches the
mean log-ratio profile `p_i = mean[log(r_i / r_k)]`, penalising
`alpha * ||p_fake - p_real||_1` on the generator loss. `p` is exactly the
sufficient statistic the Hill estimator collapses to a scalar
(`LID = -1 / mean_i(p_i)`), so matching it moves LID — but it is bounded and
smooth where LID's `-1/x` blows up, and being a `(k-1)`-vector it constrains
the neighbourhood's *shape*, not just its scale. The real-side target is an
EMA (decay 0.99) rather than a fresh per-batch estimate, which is the same
cost at lower gradient variance. See `src/train/log_ratio.py`.

**v3 and v4 are not judged the same way.** v3 trains on none of the
read-out metrics, so if it hits the LID gate that is genuine independent
evidence. v4 trains on LID's sufficient statistic, so its `lid_median` is a
*fitted* number and hitting the gate is close to tautological. The real
question for v4 is what comes with it: `hubness_skew`, `ivf_gini` and
`relative_contrast` are untouched by the penalty and stay independent under
both arms. Any write-up of a v4 result must lead with those.

### `generator_type`

The architecture axis in the `model` config block, accepting `mlp` (default),
`gated`, and `structured_gated`. It sits underneath the variant numbering:
v0, v1 and v1_5 all use `mlp` and differ only in training settings.

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

### Generator regularizers

Both are off by default, so v0–v3 behaviour is unchanged when they are absent.

| Config key | Default | Meaning |
|---|---|---|
| `training.distance_reg_alpha` | `0.0` | Weight on `\|mean pairwise distance real − fake\|`, a single global scalar. |
| `training.distance_reg_max_points` | `128` | Batch subsample the distance matrix is computed on. |
| `training.lid_reg_alpha` | `0.0` | Weight on the local log-ratio penalty (v4). Off means the term is never computed and the real batch is never even transferred. |
| `training.lid_reg_k` | `20` | Neighbour depth; the profile has `k − 1` entries. |
| `training.lid_reg_max_points` | `256` | Batch subsample the within-batch neighbour search runs on. |

Both terms are logged per step alongside `wasserstein` and `adv_loss`, as
`distance_reg` and `lid_reg`.

`lid_reg_alpha` **cannot be set by analogy to `distance_reg_alpha`.**
`distance_reg` is one scalar; `log_ratio_penalty` is an L1 *sum* over `k − 1`
components — 19 of them at the default `k` — so it sits on a different scale
and grows with `k`. Pick it from a measured `lid_reg` value on real batches at
the config's actual `lid_reg_k`. See `FOLLOWUPS.md`.

Degenerate batches are dropped rather than clamped, matching
`src.eval.ann_difficulty.survivor_mask`: a query whose nearest neighbour sits
at distance zero (duplicates, entirely plausible under mode collapse) or whose
neighbours all tie contributes nothing, and the penalty is exactly zero when no
query survives on either side.

Training entrypoint:

- `src/train/train_wgan_gp.py`

---

## Device behavior

Resolved by `src/device.py` (`resolve_device`), shared by training, sampling
and eval.

Order for `device: auto`:

1. CUDA (if available)
2. MPS (Apple Metal, if available)
3. CPU fallback

Training passes `strict=True`, which **rejects `auto` when CUDA is present
and `CUDA_VISIBLE_DEVICES` is unset**. Plain `auto` resolves to a bare
`cuda`, i.e. `cuda:0`, so on a shared box two runs silently land on the same
card. Name the device in the config (`device: cuda:0`) or pin the process.
Sampling and eval stay permissive -- they are short and read-only.

### GPU claiming

`src/train/gpu_lock.py` takes an exclusive `flock` for the duration of a run,
keyed on the card's **UUID** rather than its index, since two processes with
different `CUDA_VISIBLE_DEVICES` mappings both see their card as index 0. The
lock is acquired in `main()`, so it covers every CLI launch but not direct
`train()` calls from tests.

| Config key | Default | Meaning |
|---|---|---|
| `training.gpu_lock_timeout_s` | `1800` | Seconds to queue for a busy card before giving up. |
| `training.gpu_memory_fraction` | `0.9` | Per-process cap, so a bypassed lock degrades a run rather than taking the card down. |

`flock` is advisory and host-local: it coordinates cooperating processes on
one machine, and does nothing across hosts or against a process that does not
take the lock.

`run_metadata.json` records a `gpu` block with the card's name, UUID and free
and total memory at launch.

### Resume

`--resume <checkpoint>` continues a run. `num_gen_steps` is the target
**total**, not an additional budget. Checkpoints carry both optimiser states,
the EMA shadow, `ema_step` and `best_cov`. Resuming an EMA-enabled run from a
checkpoint without a shadow is refused rather than silently restarting the
average.

---

## Evaluation stack

## Checkpoint-based eval

- Script: `src/eval/evaluate_distribution.py`
- Inputs: real dataset + checkpoint + run config
- Samples fake vectors directly from generator.

## File-to-file eval

- Script: `src/eval/evaluate_file_to_file.py`
- Inputs: real file + synthetic file
- Compares dataset artifacts directly (no model required).

## Metrics

- `mean_l2`, `var_l2`
- `cov_fro`
- `mmd_rbf`
- `pairwise_hist_l1`
- `knn_recall`
- `ann_proxy_recall`

### Metric definitions

Let `X = {x_i}` be real samples and `Y = {y_j}` be synthetic samples, with vectors in `R^128`.

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
  - Panels: local intrinsic dimensionality and relative contrast (ANN
    difficulty), hubness (k-occurrence), IVF cell balance, pooled value
    distribution, per-dimension marginals with a dropdown over all 128 dims,
    per-dim mean/std/zero-rate profiles, pairwise distances, within-set kNN
    distances, PCA spectrum, correlation heatmaps, and a Wasserstein-1
    ranking of the worst-matching dimensions.
  - The ANN-difficulty panels (`src/eval/ann_difficulty.py`) ask whether a
    synthetic set would *behave* like SIFT under nearest-neighbour search,
    not just look like it: `--ann-k` (default 100) sets the neighbour depth
    for LID and relative contrast, `--ann-hub-k` (default 10) the depth for
    the hubness k-occurrence count, `--ann-max-rows` (default 20000) the
    equal-N truncation every set is cut to so the metrics stay comparable
    across series, and `--ivf-nlist` (default 256) the cluster count for the
    IVF cell-balance panel. These numbers are self-queried subsample
    statistics, not published SIFT1M figures, and are only comparable across
    the series in one report.
  - `--synthetic-path` is optional; without it the report is pure dataset EDA.
    With it, every panel overlays the two so mismatch is visible by eye.
  - `--preprocess l2` (default) matches the training contract, since generator
    output is unit-norm. Use `--preprocess none` to inspect raw integer SIFT.
  - Purpose: reject a generator by eye when the critic cannot separate the
    sets. A weak critic yields flattering Wasserstein estimates over samples
    whose marginals are plainly wrong -- most visibly SIFT's heavy exact-zero
    mass, which smooth generators do not reproduce.
  - `src/eval/compare_variants.py` drives this across all four variants at
    once, labelling the overlays `v0`/`v1`/`v1_5`/`v2` to match the variant
    table. It resolves each variant's `best_generator.pt` and
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

## Workflow for a new user

## 1) Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Put dataset in place

- Expected path by default config:
  - `data/sift_base.fvecs`

## 3) Train with default config

```bash
.venv/bin/python -m src.train.train_wgan_gp --config configs/wgan_gp_sift1m_real.yaml
```

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
