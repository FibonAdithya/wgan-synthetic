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

Measured on 50,000 rows of the real train split (9,990,000 x 96 float32):

| | |
|---|---|
| Norm | exactly 1.0 (std 2.2e-08 — machine precision, not approximately) |
| Negative fraction | 0.508 |
| Exact zeros | none |
| Mean abs. per-dim mean | 0.021 — effectively centered already |
| Effective rank | 65.3 |

Covariance spectrum, which is what `training.spectrum_reg_alpha` targets:

| | |
|---|---|
| Participation ratio | 45.3 of a possible 96 |
| Largest normalized eigenvalue | 0.0731 (7.0x the isotropic 1/96) |
| Smallest | 0.0014 (0.13x isotropic) |
| First-to-last ratio | 53.6 |
| Variance in top 10 / 25 / 50 dims | 36.6% / 60.1% / 81.9% |

Worth stating plainly because it is easy to assume otherwise: a PCA-compressed
set is often taken to have a steeply decaying spectrum, and DEEP's decay is
real but **moderate**. Variance is spread across roughly half the available
directions. That tempers how much the spectrum regularizer could be expected
to do here, and is consistent with the `v1` results below.

## Measured profile

Read from the file rather than quoted from a paper. Canonical N and k are
locked here so a gate result stays readable against an older one.

| | |
|---|---|
| Canonical N | `20000` |
| Canonical k | `100` (`10` for hubness) |

| Statistic | Real | Synthetic (closest rung) |
|---|---|---|
| LID median | 17.561218 | 16.908998 (`v2`) |
| Relative contrast | 1.832256 | 1.867279 (`v2`) |
| Hubness skew | 1.940139 | 1.940117 (`v1`) |
| IVF cell-balance Gini | 0.304576 | 0.299153 (`v2`) |

Measured over 50,000 vectors per set at `preprocess: l2`, `seed: 42`,
`nlist: 256`. The full report output these came from is committed as
`docs/datasets/deep_ladder_summary.json`, so every figure on this page is
checkable without access to the training box. Reproduce the real column with:

    python -m src.eval.eda_report \
        --real-path data/deep_1m.npy \
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

Each rung differs from the one below it by exactly one config key, which is
what makes a difference in the report attributable to a single cause.
`tests/test_deep_configs.py` machine-checks that invariant by flattening both
configs and diffing the full key set.

| Variant | Delta | Config | Run | Status |
|---|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/deep/v0.yaml` | `runs/deep/v0` | trained, 30k steps |
| `v1` | `training.spectrum_reg_alpha: 0.1` | `configs/deep/v1.yaml` | `runs/deep/v1` | trained, 30k steps |
| `v2` | `data.preprocess.whiten: true` | `configs/deep/v2.yaml` | `runs/deep/v2` | trained, 30k steps |

Train a rung:

    python -m src.train.train_wgan_gp --config configs/deep/v0.yaml

Compare the whole ladder against real DEEP:

    python -m src.eval.compare_variants --dataset deep \
        --real-path data/deep_1m.npy \
        --output-dir runs/eda_deep

`v2` trains in a PCA-whitened space, so its samples only mean anything once
that transform is undone. `compare_variants` does this via `invert_samples`,
reading the fitted transform out of the run's `run_metadata.json`.
`src/sample/generate.py` does **not**, and would emit `v2` vectors in
whitened coordinates.

### Results

All three rungs at 30,000 steps, `seed: 42`, on 1M real DEEP vectors
(RTX 4060, ~35 min each). Gap to real, primary statistics:

| rung | ΔLID | Δhubness skew | ΔIVF gini |
|---|---:|---:|---:|
| `v0` | 0.709438 | 0.022070 | 0.016539 |
| `v1` | 0.756583 | **0.000022** | 0.007528 |
| `v2` | **0.652220** | 0.100187 | **0.005423** |

`v2` is closest on LID and IVF gini; `v1` is closest on hubness skew. The
baseline `v0` is closest on none, so the added machinery does move the ladder
toward real DEEP — but no rung sweeps, and whitening makes hubness skew
markedly worse than either other rung.

**One run per rung at a single seed.** The gaps above cannot be separated from
run-to-run seed variance, and `v1`'s near-exact hubness match is one draw, not
a demonstrated property of the regularizer. Reading any ordering here as
settled would be over-reading the data; see FOLLOWUPS.md.

## Gate

Pass bands are per statistic, not a combined score, because the four fail in
different directions. Still unset: with one seed per rung there is no variance
estimate to set a band against, and the numbers will move again when phase (c)
re-measures this family under angular distance.
