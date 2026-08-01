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

Run length is an independent axis and is not a variant: `bench_*` are 3k
generator steps, `long_*` are 30k, `x100k_*` are 100k. The run directory
names predate this scheme and are kept as-is because the artifacts under
them are already named that way.

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
  - Panels: pooled value distribution, per-dimension marginals with a dropdown
    over all 128 dims, per-dim mean/std/zero-rate profiles, pairwise distances,
    within-set kNN distances, PCA spectrum, correlation heatmaps, and a
    Wasserstein-1 ranking of the worst-matching dimensions.
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

```bash
.venv/bin/python -m src.eval.eda_report \
  --real-path data/sift_base.npy \
  --synthetic-path runs/bench_improved/synthetic_1m.npy \
  --output-dir runs/bench_improved/eda
```

```bash
.venv/bin/python -m src.eval.compare_variants \
  --real-path data/sift_base.npy \
  --output-dir runs/eda_variants \
  --num-samples 100000
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
