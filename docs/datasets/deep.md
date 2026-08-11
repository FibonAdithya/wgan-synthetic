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

Synthetic is quoted as the mean over three seeds, not a single run, and
without naming a "closest rung": the seed sweep below shows the rungs are
indistinguishable on all four of these statistics, so attributing a column to
one of them reports noise.

| Statistic | Real | Synthetic (3-seed mean, spread across rungs) |
|---|---|---|
| LID median | 17.561218 | 16.78 – 16.85 |
| Relative contrast | 1.832256 | 1.868 – 1.873 |
| Hubness skew | 1.940139 | 1.890 – 1.942 |
| IVF cell-balance Gini | 0.304576 | 0.307 – 0.314 |

Measured over 50,000 vectors per set at `preprocess: l2`, `nlist: 256`, seeds
42/43/44. Two report outputs are committed, so every figure on this page is
checkable without access to the training box:
`docs/datasets/deep_ladder_summary.json` (the original single-seed ladder) and
`docs/datasets/deep_seed_sweep_summary.json` (the three-seed sweep these
numbers come from). Reproduce the real column with:

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
rows that are neither unit-norm nor exactly zero rather than normalizing
them itself -- an exact zero is what `maybe_l2_normalize` leaves behind, so
it is accepted rather than treated as a caller mistake.

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

**Measured, 2026-08-10.** The whole ladder was re-run at seeds 42, 43 and 44 —
nine runs, 30,000 steps each, everything else identical — which replaces the
two-draw guess this section used to carry. Committed as
`docs/datasets/deep_seed_sweep_summary.json`.

Per-rung mean and the spread across seeds:

| statistic | v0 | v1 | v2 | worst seed range | pooled sd |
|---|---:|---:|---:|---:|---:|
| LID median | 16.7829 | 16.8319 | 16.8505 | 0.1625 | 0.0640 |
| relative contrast | 1.8734 | 1.8709 | 1.8683 | 0.0168 | 0.0063 |
| hubness skew | 1.8903 | 1.9422 | 1.9111 | 0.1617 | 0.0570 |
| IVF gini | 0.3088 | 0.3069 | 0.3144 | 0.0236 | 0.0087 |
| effective rank | 63.3108 | 63.3228 | 64.0460 | 0.1614 | 0.3704 |

**Both claims this page previously reported as surviving are withdrawn.** The
single-seed numbers that supported them are inside one run-to-run swing:

- *"`v2` is closest on LID."* The v2−v0 difference is **0.0676** against a seed
  range of **0.1625**. Not supported.
- *"`v1` is closest on hubness skew, by two orders of magnitude."* v1's gap of
  0.000402 was where seed 42 happened to land; across three seeds the rung's
  gap swings by ±0.16, some four hundred times that. Not supported.

Only **effective rank** separates the rungs at all: its spread across rung
means (0.7353) exceeds the worst seed range (0.1614), and it orders them
`v2` closer than `v1` closer than `v0`. Whitening is the one rung delta this
ladder can demonstrate does something.

For the other four statistics the practical reading is that **v0, v1 and v2
are indistinguishable**. A one-key ladder is still the right shape — it just
turns out these three keys do not move these four numbers by more than noise.

### Does the spectrum regularizer bind?

`v1`'s `spectrum_reg_alpha: 0.1` was suspected of being too small to act. Two
extra runs at seed 42 settle it, read on effective rank — the property the
term actually targets:

| α | effective rank | vs shipped 0.1 |
|---|---:|---:|
| 0.1 (shipped `v1`) | 63.3809 | — |
| 1.0 | 63.3694 | −0.0115 |
| 5.0 | **63.7668** | **+0.3860** |

Seed noise on that rung is 0.0912, so α=5.0 clears it roughly fourfold while
α=1.0 does nothing at all. **The term works, but only above roughly fifty
times the shipped value** — the response is a threshold, not a slope, which is
why `test_enabling_the_regularizer_changes_the_generator` needed `alpha: 5.0`
to see any effect.

It is still the weaker lever. α=5.0 buys +0.39 of effective rank; whitening
buys +0.74 with no penalty term, and real DEEP is 1.99 above `v0` — so even
both together leave most of the gap open.

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

**Two of the four bands are now set**, calibrated from the seed sweep above:
`lid_median` and `relative_contrast_median`. They encode what the ladder
achieves *today* — best rung mean ± 3 pooled standard deviations, so all nine
sweep cells sit inside — which makes them a regression guard, not a
certificate that the synthetic set resembles real DEEP. It does not: LID is
still 0.71 short, and that gap is the open modelling problem.

`hubness_skew` and `ivf_gini` stay null, and the reason is worth stating
because it is the opposite of the usual one. They are not too noisy to
measure; **the ladder already reproduces both to within run-to-run noise** —
gaps to real of 0.04 and 0.27 pooled standard deviations respectively. There
is no gap left to gate, so any band loose enough to admit an honest run would
also admit one that missed.

`check_gate` therefore still exits 2 with verdict `unset` while those two are
null, which is correct: a partially calibrated gate must not report a pass.
Pass `--allow-unset` to get the report alone.

The most discriminating statistic of all, **effective rank**, cannot go in the
gate at present: `check_gate.GATE_STATISTICS` is a fixed four-name tuple and
rejects anything else, and widening it would require every other family's gate
file to grow the key too. Tracked as an issue rather than done here, since it
is a schema change to shared code.

These numbers do not move under the metric-aware `ann_difficulty`: they were
measured at `preprocess: l2`, which on unit-norm rows is the `angular`
geometry this family is searched with, exactly as the profile section above
records.

Check a run against it:

    python -m src.eval.check_gate --dataset deep --run-dir runs/deep/profile

It reads `summary.json` from that run directory, prints a JSON verdict, and
exits non-zero when the run fails -- or, as now, when the bands are still
unset, which is verdict `unset` and exit code 2. Pass `--allow-unset` to get
the report without the non-zero exit, and `--stats-name <label>` to check a
synthetic series rather than `real`.
