# Linear-skip generator for full local rank

Design, 2026-09-11. Not authoritative (see `AGENTS.md`); the code and
`PROJECT_DOCUMENTATION.md` win on any conflict once built.

## Problem

`v0` reproduces the first two moments of an angular corpus and not its
search difficulty, because the MLP generator's local rank is about 15 on
every family. Measured on NYTimes (`docs/datasets/nytimes.md`, cleaned real
corpus, canonical conditions):

| | LID median | relative contrast | hubness skew | IVF Gini | effective rank |
|---|---|---|---|---|---|
| real | 55.8 | 1.269 | 2.59 | 0.80 | 247 |
| v0, 100k steps | 13.7 | 2.08 | 1.36 | 0.22 | 84 |
| Gaussian, real covariance, on the sphere | 82.5 | 1.158 | 1.21 | 0.86 | 235 |

The critic scores one vector at a time, so a low-dimensional sheet with the
right mean and covariance envelope is indistinguishable from the corpus to
it, and the gradient penalty rewards the smooth maps that produce sheets.
v0's LID sits at 14 to 17 on SIFT, DEEP, GloVe and NYTimes alike; it matched
the first two only because their real LID is about 17.

A bound measured without training (`runs/nytimes/noise_sweep.json`, same
conditions): adding isotropic noise to the 100k samples on the sphere and
renormalising, at noise-to-signal 1.4, gives LID 56.1 and contrast 1.234
against real 55.8 and 1.269, with hubness 4.0 (real 2.6) and Gini 0.72 (real
0.80). Covariance-shaped noise behaves the same, because the real covariance
is nearly isotropic. So a sheet plus a full-rank residual lands the two
local statistics together to within 3%, at the cost of the residual carrying
two thirds of the output variance. The remaining question is whether a
generator that owns the residual can keep enough structure to close the
last 3% and the hubness and Gini gaps, which pure noise cannot.

## Design

### `generator_type: linear_skip`

A new generator class in `src/models/generator.py`, selected by
`generator_type: linear_skip` and built by `build_generator` alongside the
existing three. The latent `z` of size `latent_dim` is split into a trunk
block of `latent_dim - skip_dim` entries and a skip block of `skip_dim`
entries:

    x = trunk(z[:, :latent_dim - skip_dim]) + W z[:, latent_dim - skip_dim:]

`trunk` is today's `Generator` MLP (same `generator_hidden_dims` and
`negative_slope`). `W` is `nn.Linear(skip_dim, output_dim, bias=False)`.
The trainer's `normalize_l2` after the generator is unchanged, so the output
reaches the critic on the sphere as before.

Config keys, all under `model`:

| key | default | meaning |
|---|---|---|
| `skip_dim` | `output_dim` | width of the skip block; `latent_dim - skip_dim` must be positive |
| `skip_init` | `orthogonal` | `orthogonal` (gain `skip_init_gain`) or `identity` (`skip_dim` must equal `output_dim`) |
| `skip_init_gain` | `1.0` | scale of the orthogonal init |

Rationale for the split rather than a second latent argument: the two
sampling sites in the trainer and `src.sample.generate` draw
`randn(n, latent_dim)` and know nothing about generator internals. Splitting
inside the generator keeps every caller, checkpoint loader and the
`run_config.yaml` contract unchanged. A `linear_skip` config simply sets
`latent_dim` to trunk width plus `skip_dim`.

Why the Jacobian is full rank: `d x / d z_skip = W`, so the output's local
dimension is at least `rank(W)` whatever the trunk does. Orthogonal init
makes `rank(W) = min(skip_dim, output_dim)` at step 0; training can lower it
only by driving singular values of `W` to zero, which the critic's view of
the covariance envelope pushes against on a corpus whose spectrum is flat.

What the trunk is for: the residual alone lands on the Gaussian's numbers
(LID 82, contrast 1.16). The trunk supplies the structure that pulls LID
down to 56 and contrast up to 1.27. The critic constrains the *sum*, so the
balance between the two is learned, not set.

### Gate-aware checkpoint selection

`best_generator.pt` is chosen by the smallest `cov_fro` over evaluations.
On NYTimes that chose step 1,000 of a 100,000-step run, because the
covariance gap is lowest for a tight cone. An architecture whose point is
local dimension needs a selector that can see local dimension.

New key `training.select_on`, default `cov_fro` (today's behaviour, every
existing config unchanged), or `gate`. Under `gate`, each evaluation also
runs `ann_difficulty.compute` on the fake holdout sample and on the real
holdout (the real side once, cached), at `k=100`, `k_hub=10`, `nlist=256`,
`max_rows=len(holdout)`, `metric=data.metric`, and records the four
statistics for both as `eval.gate_fake_*` / `eval.gate_real_*`. The
selection score is

    score = |lid_fake - lid_real| / lid_real + |rc_fake - rc_real| / rc_real

over the two statistics that read local dimension; hubness and Gini are
recorded but not selected on, because their noise floors are wider and on
this family Gini is mostly a k-means artefact. Lower is better; the
checkpoint with the lowest score is written as `best_generator.pt`, and
`best_score` replaces `best_cov` in the checkpoint's resume state under a
key that names which selector produced it, so a resume cannot silently mix
the two.

The holdout is 5% of the training file (12,500 rows for `nytimes_250k`),
below the canonical 20,000. The selection is relative between checkpoints
of one run, so this is acceptable; the numbers it logs are not the family's
profile and must not be copied into a dataset page. Cost: one 12,500-row
brute-force k-NN per evaluation, ~10 s on CPU; at `eval_every: 1000` over
30,000 steps that is 5 minutes on a 35-minute run.

### The rung

`configs/nytimes/v1.yaml`: `v0.yaml` plus `generator_type: linear_skip`,
`latent_dim: 512` (256 trunk, `skip_dim: 256`), `select_on: gate`. The
seed-42 box instrument `configs/nytimes/v1_seed42.yaml` follows the
`v0_seed42.yaml` pattern (absolute `real_path`, own `output_dir`). Everything
else identical to v0, so the rung is one architectural delta plus the
selector; the final checkpoint is sampled and measured alongside the
selected one, so the selector's effect is visible separately from the
architecture's.

Trained on the corpus as shipped, like v0, since the row-filter question on
the family page is still open and v0 showed it does not decide the outcome.

## Testing

- `tests/test_generator.py`: `linear_skip` output shape; with the trunk's
  last layer zeroed, output equals `W z_skip`; the output covariance over
  4,096 random latents has all `output_dim` eigenvalues above a floor
  (full rank), where the plain `mlp` of the same width has fewer -- the
  discriminating assertion is the rank gap, not "not None".
- `tests/test_generator_factory.py`: `build_generator` dispatches
  `linear_skip`, rejects `skip_dim >= latent_dim`, and rejects `identity`
  init when `skip_dim != output_dim`.
- Selection: a unit test on the scoring function with hand-built stats
  (a checkpoint with lid and rc nearer real scores lower); a smoke test that
  `select_on: gate` writes `best_generator.pt` at the step with the lowest
  score and records `gate_*` keys in `run_metadata.json`, using the existing
  `test_train_smoke.py` fixture sizes.
- Mutation checks after implementing: zero `W` and confirm the rank test
  fails; invert the score comparison and confirm the selection test fails.

## What this does not do

- No change to the critic. If `v1` lands on the Gaussian's numbers rather
  than the real ones, the next rung is the neighbourhood-aware critic
  (approach B in the discussion), not more of this.
- No spherical parameterisation; normalisation stays post-hoc.
- No change to `v0`, to any other family's configs, or to the default
  selector.

## Success

All four gate statistics of the selected `v1` checkpoint, measured against
the cleaned real corpus at the canonical conditions, inside the real ten-draw range, or for LID and contrast within 3% of the real
median, in `docs/datasets/nytimes_noise_floor.json`:
LID 55.0--56.9, contrast 1.265--1.276, hubness 2.32--2.78, Gini 0.78--0.82.
LID and contrast are the ones the design targets; hubness and Gini are
reported and read against the noise-sweep bound (4.0 and 0.72) to say
whether the trunk added structure the residual lacks.
