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
| LID median | 17.561218 | 16.935708 (`v2`) |
| Relative contrast | 1.832256 | 1.864716 (`v2`) |
| Hubness skew | 1.940139 | 1.939737 (`v1`) |
| IVF cell-balance Gini | 0.304576 | 0.302923 (`v0`) |

Measured over 50,000 vectors per set at `preprocess: l2`, `seed: 42`,
`nlist: 256`. The full report output these came from is committed as
`docs/datasets/deep_ladder_summary.json`, so every figure on this page is
checkable without access to the training box. Reproduce the real column with:

    python -m src.eval.eda_report \
        --real-path data/deep_1m.npy \
        --output-dir runs/deep/profile \
        --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular

Read the four values out of runs/deep/profile/summary.json (written by the command above).

`ann_difficulty.py` measures this family under its `data.metric`, which is
`angular`: L2 between unit-norm rows. On the unit sphere Euclidean distance
is a strictly increasing function of cosine distance, so it ranks neighbours
identically -- the corpus is measured under the distance it is searched with.
Measuring requires `--preprocess l2`, and `ann_difficulty.compute` refuses
rows that are not on the sphere rather than normalizing them itself.

The figures above were measured at `preprocess: l2`, as
`deep_ladder_summary.json` records, so they were already measured under this
geometry and stand unchanged.

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

All three rungs at 30,000 steps, `seed: 42`, `latent_dim: 96`, on 1M real DEEP
vectors (RTX 4060, ~35 min each). Gap to real, primary statistics:

| rung | ΔLID | Δhubness skew | ΔIVF gini |
|---|---:|---:|---:|
| `v0` | 0.836308 | 0.045223 | **0.001653** |
| `v1` | 0.748235 | **0.000402** | 0.007974 |
| `v2` | **0.625510** | 0.041541 | 0.006477 |

### How much of this is noise

An earlier draft of this ladder ran the same three configs at
`latent_dim: 128`, inherited from the SIFT ladder. Correcting that to 96 gave
a near-replicate: three runs, same seed, same data, same everything except a
latent width the target's effective rank of 65 says should barely bind. The
two sets can therefore be read as two draws, which is the only variance
estimate this family has.

Gap to real, 128 → 96:

| rung | ΔLID | Δhubness skew | ΔIVF gini |
|---|---|---|---|
| `v0` | 0.709 → 0.836 | 0.022 → 0.045 | 0.017 → 0.002 |
| `v1` | 0.757 → 0.748 | 0.000022 → 0.000402 | 0.0075 → 0.0080 |
| `v2` | 0.652 → 0.626 | 0.100 → 0.042 | 0.0054 → 0.0065 |

**Some of these swings are larger than the differences between rungs.** `v0`'s
IVF gini improved tenfold and its hubness gap doubled; `v2`'s hubness gap
halved. Nothing about the change should have produced that, which puts a
floor under how finely these numbers can be read.

What survives both draws:

- **`v2` is closest on LID** in both (0.652, 0.626). Holds.
- **`v1` is closest on hubness skew** in both, by two orders of magnitude
  (0.000022, 0.000402). Holds, and is the one result the spectrum regularizer
  can plausibly claim.
- **IVF gini reorders completely** — `v0` went from worst (0.017) to best
  (0.002), `v2` from best to middle. No ordering here is supported.

So the honest reading is narrower than "v2 wins two of three": `v2` helps LID,
`v1` helps hubness skew, and the IVF gini column should not be used to rank
rungs at all. Two draws is still two, and neither is a seed sweep.

## Gate

`gates/deep.yaml` is the gate. The bands live there rather than in this
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
are unset, and the checker says so instead of passing. The two draws above show
why that caution is warranted here: the IVF gini gap moved tenfold for `v0`
under a change that should barely have bound, so a band set from either draw
alone would be fitted to noise. Setting them needs a real seed sweep.

Check a run against it:

    python -m src.eval.check_gate --dataset deep --run-dir runs/deep/profile

It reads `summary.json` from that run directory, prints a JSON verdict, and
exits non-zero when the run fails -- or, as now, when the bands are still
unset, which is verdict `unset` and exit code 2. Pass `--allow-unset` to get
the report without the non-zero exit, and `--stats-name <label>` to check a
synthetic series rather than `real`.
