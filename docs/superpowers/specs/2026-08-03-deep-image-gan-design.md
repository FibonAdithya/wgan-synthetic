> **AI-generated working note.** Written by Claude during development and kept
> for provenance. It is not the source of truth: where this file disagrees with
> `PROJECT_DOCUMENTATION.md`, the latter wins.

> **Status: not implemented.** Nothing in this spec ships yet.

# A WGAN track for DEEP image descriptors

Date: 2026-08-03
Branch: `worktree-gan+next-iteration`
Status: approved, ready for implementation planning

## Problem

This repo synthesizes SIFT1M-like descriptors: 128-D, non-negative, sparse,
L2-normalized. Four variants (`v0`–`v2`) were trained, and the hardest thing
about SIFT turned out to be its *support* — the heavy mass at exactly zero that
a dense MLP cannot reproduce, which `GatedGenerator` was built to fix.

A synthetic ANN benchmark built on one dataset proves little. The natural second
target is `deep-image-96-angular` (DEEP1B descriptors), the other standard
ann-benchmarks vector set. Its properties are almost the inverse of SIFT's:

| | SIFT1M | DEEP |
|---|---|---|
| dimension | 128 | 96 |
| support | sparse, many exact zeros | dense |
| sign | non-negative | real-valued, signed |
| norm | L2-normalized in preprocessing | already unit-norm; points lie on S^95 |
| origin | hand-crafted gradient histograms | PCA-compressed CNN embeddings |
| spectrum | mild anisotropy | strong PCA variance decay |

Nothing SIFT-specific in the codebase transfers. `GatedGenerator`'s
non-negativity is actively wrong here, and the sparsity metrics in
`tensor_stats` measure a property DEEP does not have. What *does* transfer is
the WGAN-GP machinery, the preprocessing contract, and the whole evaluation
suite — none of which is dimension-bound.

## Scope

A parallel `src/deep/` track: data acquisition, a DEEP-appropriate generator, a
four-rung variant ladder, and an ANN-difficulty comparison report. Trained to
completion on the `tig-gpu` box.

Explicitly out of scope:

- Any change to SIFT configs, the SIFT variant table, or SIFT behaviour. The
  existing 118 tests are the regression guard.
- Generalizing the codebase into one dataset-agnostic pipeline. That was
  considered and rejected (see Options).
- Recall-vs-latency curves against a real faiss/hnswlib index. Worth doing
  later; the ANN-difficulty panels answer the same question more cheaply.
- Retiring the SIFT-flavoured entries in `tensor_stats`. See Known warts.

## Options considered

**Generalize the existing pipeline in place.** Make the loader, trainer, and
eval genuinely dataset-agnostic; rename the SIFT-specific modules that lie
about their contents; add DEEP as a second config family. One codebase, two
datasets, no duplication.

Rejected: it puts a working, validated SIFT track at risk for a benefit that is
mostly aesthetic. The trainer is *already* dimension-agnostic in substance —
"SIFT" appears in module names, docstrings, and config defaults far more than
in logic — so generalizing buys less than the rename churn costs.

**Config-only.** Add `configs/deep_gan_*.yaml` with `descriptor_dim: 96`, reuse
the plain MLP, change nothing else.

Rejected: no DEEP-appropriate inductive bias, and no answer to where the data
comes from. It is the `deep_v0` rung of the chosen design, not a design.

**Parallel deep track (chosen).** New `src/deep/` modules that *call* the
existing trainer rather than reimplement it. SIFT files take three additive
touches and no behavioural change.

This is cheaper than it first appears because `train(config: Dict)`
(`src/train/train_wgan_gp.py:299`) is already a clean, config-driven entry
point. The deep track supplies a config and a generator; the loop is reused
as-is.

## Architecture

```
src/deep/
  download.py    fetch deep-image-96-angular.hdf5 -> data/deep96_{1m,250k}.npy
  dataset.py     HDF5 reader + inverse-preprocess
  generator.py   SphericalGenerator
  spectrum.py    covariance-spectrum regularizer
  sample.py      sampler that inverts preprocessing before writing
  report.py      deep variant table -> ANN-difficulty comparison
configs/deep_gan_{v0,v1,v1_5,v2}.yaml
tests/test_deep_*.py
```

Each module has one job and can be tested without the others:

- `download.py` is I/O only. Input: a URL and a subset size. Output: a
  `[N, 96]` float32 `.npy` on disk. It emits `.npy` deliberately, so the
  existing loader needs no HDF5 support and the trainer needs no change.
- `dataset.py` adds `invert_preprocess(x, state)`, the inverse of the existing
  `apply_preprocess`. This is the piece the pipeline genuinely lacks today, and
  the `deep_v2` rung cannot be correct without it.
- `generator.py` holds `SphericalGenerator`: an MLP trunk with an explicit
  L2-normalizing output head. Unlike `GatedGenerator` it is sign-free and has
  no stochastic gate, so it is deterministic given `z`.
- `spectrum.py` is a pure function of two batches, returning a scalar loss. No
  model state, no config coupling.
- `sample.py` mirrors `src/sample/generate.py` but applies
  `invert_preprocess` before writing, and does not unconditionally
  L2-normalize.
- `report.py` holds the deep variant table and drives the comparison, mirroring
  `src/eval/compare_variants.py`.

### Touches to shared files

Three, all additive:

1. `src/models/generator.py` — a `"spherical"` branch in `build_generator`.
   The existing `"mlp"` and `"gated"` branches are untouched.
2. `src/train/train_wgan_gp.py` — an optional `spectrum_reg_alpha` term,
   following the `distance_reg_alpha` precedent already in the loop. Defaults
   to `0.0`, so SIFT configs behave identically.
3. `data/README.md` — document the DEEP data contract alongside the SIFT one.

## Variant ladder

Each rung is exactly one config change from the previous, matching the
methodology of the SIFT table in `PROJECT_DOCUMENTATION.md`.

| Variant | Delta | Tests |
|---|---|---|
| `deep_v0` | plain MLP; EMA and GP settings carried over from SIFT | baseline; post-hoc normalize only |
| `deep_v1` | `model.generator_type: spherical` | critic sees on-manifold samples during training |
| `deep_v1_5` | `+ training.spectrum_reg_alpha` | explicit pressure on the PCA variance decay |
| `deep_v2` | `+ data.preprocess.whiten: true`, inverse at sample time | anisotropy exact by construction |

`deep_v0` is derived from `configs/sift_gan_v1.yaml` — the SIFT rung that has
EMA but not the pairwise-distance regularizer — with `descriptor_dim: 96` and
the real path pointed at the DEEP subset. Concretely that fixes, for every
rung: `latent_dim: 128`, generator `[512, 1024, 1024]`, critic
`[1024, 512, 256]`, `batch_size: 512`, `n_critic: 3`, `lambda_gp: 5.0`,
`ema_decay: 0.999`, `distance_reg_alpha: 0.0`, `num_gen_steps: 30000`. Only the
row's stated delta varies.

Carrying these forward rather than re-ablating them is deliberate: those
questions were answered on SIFT, and re-running them costs GPU hours for no new
information. `spectrum_reg_alpha` defaults to `0.0` everywhere except
`deep_v1_5`, where its starting value is `0.1`, matching the scale
`distance_reg_alpha` uses in `sift_gan_v1_5.yaml`.

`deep_v1_5` and `deep_v2` attack the same problem — the PCA variance decay —
by different means: one by penalty, one by construction. The ladder is
informative precisely because they can be compared.

## Success criteria

A variant is judged by **ANN-difficulty parity**, not distributional fidelity.
The three metrics from `src/eval/ann_difficulty.py` — LID, hubness skew, and
IVF cell balance — computed on real DEEP and on each variant's samples, overlaid
in one report. A variant wins by landing closest to real DEEP on those three.

This choice follows the reasoning in
`2026-07-31-ann-difficulty-panels-design.md`: on SIFT, every generator matched
real data on global geometry while saying nothing about search behaviour.
Distributional metrics are reported for diagnosis, not for ranking.

`deep_v0` is expected to be a weak baseline. That is the point of including it;
a ladder where every rung wins tells you nothing.

## Training and data

`deep-image-96-angular.hdf5` is ~4 GB (9,990,000 train vectors, 10,000 test).
`download.py` fetches it once on the GPU box and writes two subsets: 250k for
pipeline smoke runs and 1M for the real runs. The full 10M is not used — a
WGAN never sees a full epoch anyway, so beyond ~1M the extra rows change
sample diversity marginally while making every load heavier.

Training runs on `tig-gpu` (RTX 4060, 8 GB). These MLPs are ~2.5M parameters
total, so the constraint is wall clock, not memory.

### Persistence

`workspace_is_volume=false` on that instance. **Nothing on the container
filesystem survives a recycle or destroy**, including the repo copy at
`/workspace/wgan-synthetic` and every checkpoint written there. The plan
therefore:

- pushes work up as a git bundle, matching the `sparse.bundle` pattern already
  in that box's home directory;
- runs training as a **supervisor service**, not a loose background process —
  the instance's own agent guide is explicit that a bare `python … &` dies with
  the shell and its logs never reach the portal;
- pulls each variant's checkpoint back down as it finishes, rather than
  collecting them all at the end.

The 4 GB download and the subsetting happen on the GPU box. Neither the HDF5
nor the `.npy` subsets are committed.

## Testing

Unit tests per module, following the existing `tests/` conventions (imports at
the top of the file, per FOLLOWUPS item 3):

- `SphericalGenerator` emits unit-norm rows, admits negative values, and is
  deterministic given `z` — the last being the contrast with `GatedGenerator`.
- `invert_preprocess(apply_preprocess(x)) ~= x` for each combination of
  center/whiten, within float32 tolerance. This is the round-trip that
  `deep_v2` depends on.
- The spectrum regularizer is zero for identical batches and positive for
  batches with mismatched variance decay.
- `download.py` subsetting is tested against a small synthetic HDF5 fixture,
  not a network fetch.
- The deep sampler writes `[N, 96]` float32 and, under a whitening config,
  produces output in the original space rather than the whitened one.

End-to-end: a 250k smoke run of each rung before any long run is launched.

## Known warts

`tensor_stats` (`src/train/train_wgan_gp.py:66`) computes `zero_fraction_gap`,
`negative_fraction`, `per_dim_zero_rate_l1`, and `nnz_std_gap` — all measures of
the sparse support that DEEP does not have. On deep runs these log as
uninformative near-constants. Fixing it properly means restructuring a SIFT
file this work is meant to leave alone, so the deep report simply does not
surface them. Worth revisiting if a third dataset ever arrives, at which point
the "generalize in place" option deserves a second look.
