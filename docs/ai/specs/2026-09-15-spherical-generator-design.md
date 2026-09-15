# Spherical generator with a constant radius and a location-conditioned residual

Design, 2026-09-15. Not authoritative (see `AGENTS.md`); the code and
`PROJECT_DOCUMENTATION.md` win on any conflict once built. Scope: the next
NYTimes rung, `v3`. The class is the first instance of the `generator_type:
spherical` that `PROJECT_DOCUMENTATION.md` names as planned; the other
angular families are not designed for here.

## Problem

Every trained NYTimes rung misses hubness skew by 4x to 7x, and neither the
neighbourhood critics (`v2`, `v2b`, `v2c`) nor a longer budget (`v2c` to
100,000 steps) moved it. Two probes on the `v2c` checkpoints, run 2026-09-15
and committed under `docs/results/nytimes-trunk-scale-probe/`, locate the
cause in the `linear_skip` generator.

The generator's output is `trunk(z_t) + W z_s`, normalised. The probe fixes a
checkpoint and a latent draw, rescales the trunk term by `s` before the sum,
and measures each sample with `ann_difficulty.compute` at the canonical
conditions (20,000 rows, k 100, hub k 10, nlist 256, angular). The second
probe also rescales each row's trunk and skip terms to their mean norms
before the sum, so every sample carries the same trunk-to-skip ratio.

Step-30,000 checkpoint (`trunk_scale_v2c_eq.json`); the other two
checkpoints read the same way:

| set | LID median | relative contrast | hubness skew | IVF Gini | effective rank |
|---|---|---|---|---|---|
| real, cleaned | 55.8 | 1.269 | 2.59 | 0.796 | 247.9 |
| skip only (`s = 0`) | 64.7 | 1.205 | 1.26 | 0.735 | 184.5 |
| `s = 1`, free ratio | 60.2 | 1.233 | 10.25 | 0.786 | 184.9 |
| `s = 1`, equalised | 62.6 | 1.215 | 4.67 | 0.762 | 184.8 |
| `s = 2`, free ratio | 56.7 | 1.259 | 17.31 | 0.832 | 185.2 |
| `s = 2`, equalised | 63.1 | 1.213 | 4.42 | 0.779 | 185.3 |
| trunk only | 22.9 | 1.911 | 2.70 | 0.351 | 23.6 |

Three facts follow, and together they fix the design.

1. **The hubs are rows with a slightly smaller residual share.** Equalising
   the per-row ratio drops hubness from 17 to 30 down to 3.3 to 5.7 on every
   checkpoint and scale. The per-row trunk norm has a coefficient of
   variation of only 0.06 to 0.08: in 256 dimensions a row whose residual is
   a few percent smaller is uniformly closer to every other row, and lands
   in thousands of 10-NN lists. Neither term alone is hubby (skip only 1.3,
   trunk only 2.5 to 2.7); the free-ratio sum is.
2. **The LID readings near real were hub artefacts.** With the ratio
   equalised, every mix short of trunk-only reads LID 62 to 69, and
   trunk-only is a 24-rank sheet at LID 22. Hubs fill neighbour lists and
   pull the estimator down, the same mechanism the family page documents
   for the corpus's zero rows.
3. **A fixed linear residual cannot reach LID 56 at real's global rank.**
   Skip-only reads LID 62 to 70 at effective rank 184 to 210; a Gaussian
   with the real covariance reads 82 at rank 235; real reads 56 at rank
   248. With one global map `W`, lowering LID costs global rank. Real is
   globally isotropic and locally 56-dimensional, so the local covariance
   must vary with position while the global covariance does not.

So the next generator needs a residual whose per-sample magnitude is held
constant (removes the hubs) and whose shape is conditioned on the sample's
position (reaches the local dimension without losing global rank). Neither a
critic nor a selector can supply either. The equalised rows also give the
floor this design starts from: a uniform residual of the current shape
lands hubness at about 4 against real's 2.6, so a constant ratio is
necessary and not sufficient, and the conditioning has to close the rest.

## Design

### `generator_type: spherical`

A new class `SphericalGenerator` in `src/models/generator.py`, built by
`build_generator` under `generator_type: spherical`. The latent `z` of size
`latent_dim` splits as `LinearSkipGenerator`'s does: the first
`latent_dim - skip_dim` entries are the trunk block `z_t`, the last
`skip_dim` entries the skip block `z_s`. Four parts:

- **Trunk.** The hidden-layer stack from `generator_hidden_dims` with
  `negative_slope` LeakyReLU, applied to `z_t`, ending in a hidden state `h`
  of width `generator_hidden_dims[-1]`. A direction head, one bias-free
  linear layer from `h` to `output_dim`, normalised per row, gives the unit
  direction `u`.
- **Tangent head.** A linear layer maps `z_s` to `tangent_hidden_dim`
  channels. A per-channel scale and shift computed from `h` (two linear
  layers from `h`, one for each) modulate those channels as
  `a * (1 + gamma(h)) + beta(h)`, then LeakyReLU, then a linear layer to
  `output_dim`. The result `v` is projected orthogonal to `u`,
  `v - (v . u) u`, and normalised per row to the unit tangent `t`. The
  modulation is where the location dependence lives: a per-channel scale
  reshapes the residual's local covariance directly, where a plain
  concatenation of `h` and `z_s` would reach it only through activation
  patterns. `gamma` and `beta` are initialised with the module's default
  weights, not zero, so the dependence exists from the first step and the
  location-dependence test below has something to measure.
- **Radius.** One learned scalar `radius_raw`, mapped as
  `r = radius_min + (radius_max - radius_min) * sigmoid(radius_raw)` and
  initialised so `r = radius_init`. Every sample uses the same `r`. No
  per-sample radius on this rung: the probe shows a 6% spread already makes
  hubs.
- **Output.** `x = cos(r) u + sin(r) t`. Unit-norm to float precision, so
  the trainer's `normalize_l2` and the sampler's normalisation are no-ops on
  it. The local dimension comes from the Jacobian of `t` with respect to
  `z_s`, which is full rank in the tangent space at every `u`; its shape
  follows `h` through the modulation while its magnitude, `sin(r)`, does
  not.

### Config keys

All under `model`, beside the existing `latent_dim`,
`generator_hidden_dims` and `negative_slope`:

| key | default | meaning |
|---|---|---|
| `skip_dim` | `output_dim` | width of the skip block; `latent_dim - skip_dim` must be positive |
| `tangent_hidden_dim` | `512` | hidden width of the tangent head |
| `radius_init` | `0.95` | starting angle in radians; `tan(0.95) = 1.40`, the noise-sweep ratio in the linear-skip design |
| `radius_min` | `0.2` | lower edge of the band; keeps the generator off the sheet |
| `radius_max` | `1.5` | upper edge of the band; keeps it off pure residual (pi/2 is 1.571) |

Construction validates `0 < skip_dim < latent_dim`,
`0 < radius_min < radius_init < radius_max < pi/2`, and
`tangent_hidden_dim > 0`, and raises `ValueError` naming the offending
value, as the existing classes do.

### Configs and scripts

- `configs/nytimes/v3.yaml`: `v2c` with `generator_type: spherical` and the
  four generator keys at their defaults, plus `skip_dim: 256`. Every other
  key byte-identical to `v2c`, including the set critic
  (`critic_type: neighbourhood_set`, edge dim 128, max pool),
  `drop_zero_rows: true`, `select_on: gate`, 30,000 steps, `amp: false`.
- `configs/nytimes/v3_seed42.yaml`: the box instrument, differing from `v3`
  only in the absolute real path and the output dir, as `v2c_seed42` does.
- `scripts/nytimes_v3_seed42_job.sh`: one gpuq job on the GPU lane that
  trains, samples 50,000 rows at seed 42 from `best_generator.pt` and from
  the step-30,000 checkpoint, and runs `eda_report` against the cleaned
  corpus at the canonical conditions, the way the `v2c` script does. Budget
  from the `v2c` job: 67 minutes wall for 30,000 steps; the tangent head
  adds three small linear layers and is not expected to change that
  measurably (ESTIMATE until the job runs).

### Data flow

Nothing in the trainer's loop, the sampler, or the checkpoint contract
changes. The trainer already branches on the generator class to log
`LinearSkipGenerator.component_energies` at each evaluation; the new class
gets the same branch with a `diagnostics(z)` method returning two scalars:

- `radius`: the current `r`.
- `direction_effective_rank`: the effective rank (the report's definition,
  `src/eval/eda/metrics.effective_rank`) of `u` over the evaluation batch,
  which shows whether the trunk is developing a sheet the way
  `linear_skip`'s did.

Both land in `run_metadata.json` under `eval`. Checkpoints hold the state
dict, `radius_raw` included, and the architecture is rebuilt from
`run_config.yaml` at load time, so invariant 4 of `AGENTS.md` holds
unchanged.

### Numerical guards

The direction head's norm and the projected tangent's norm are clamped at
`1e-8` before division. In 256 dimensions a tangent output parallel to `u`
does not occur in practice; the clamp keeps a degenerate row finite rather
than NaN. Under `amp: false`, which the set critic already requires, every
operation is float32.

## Tests

Each test names the mutation it catches. After implementing, break the code
as described, confirm the test fails, restore.

| test | assertion | mutation it catches |
|---|---|---|
| unit norm | output row norms within `1e-5` of 1 on random `z` | dropping the normalisation of `t` or of `u` |
| orthogonality | `(u . t)` within `1e-5` of 0 per row | removing the projection |
| constant angle | `arccos(x . u)` equals `radius_init` on every row, spread under `1e-5` across rows | a per-sample radius; a wrong sigmoid mapping |
| band enforcement | `radius_raw = +-50` leaves `r` strictly inside `(radius_min, radius_max)` | an unclamped radius |
| local rank | the Jacobian of one output row with respect to `z_s` has rank at least `min(skip_dim, output_dim - 1)` at init | zeroing the tangent head's skip weight |
| location dependence | `gamma(h)` differs across trunk latents; with `gamma` and `beta` frozen to constants, `t` for a fixed `z_s` stops varying with `z_t` beyond what re-projection onto the new `u` explains | a modulation that is wired but inert |
| factory | `build_generator` builds the class from `spherical`, honours the five keys, rejects `skip_dim` outside `(0, latent_dim)` and a band with `min >= init`, `init >= max` or `max >= pi/2`; a state dict round-trips through a rebuild from the run config | a missing factory branch; a key-name mismatch |
| config pinning (`tests/test_nytimes_configs.py`) | `v3` equals `v2c` except the generator keys; `v3_seed42` differs from `v3` only in real path and output dir; the job script names `v3_seed42` | a rung that changes more than it says |
| trainer smoke | the existing smoke fixture with `generator_type: spherical` runs its few steps and writes `radius` and `direction_effective_rank` into `run_metadata.json` `eval` | the diagnostics branch not firing |

The local-rank test follows the shape of the existing `linear_skip` rank
test. The location-dependence test is the one most likely to be written
tautologically; its second clause (freeze the modulation, confirm `t` stops
depending on `z_t`) is what makes it discriminating, and the plan must keep
it.

## Mechanism checks the trained run reports

The family page's `v3` section reports these beside the gate table, so the
design's claims are tested in training and not only on frozen checkpoints:

- **Holdout hubness on every evaluation.** Predicted at or below about 4
  for the whole run. Holding there confirms the constant-ratio claim under
  training.
- **Radius trajectory.** Where `r` settles and whether it pins to a band
  edge.
- **Direction rank trajectory.** Whether `u` collapses into a sheet.
- **Transient test** as the earlier rungs used: the selected step's
  neighbouring evaluations within a factor of two of its score.
- **Wall time** measured from the job log timestamps, against `v2c`'s 67
  minutes.

## Success

The selected `v3` checkpoint, measured against the cleaned real corpus at
the canonical conditions, inside the real ten-draw range on all four gate
statistics, with LID and contrast allowed within 3% of the real median, and
not a transient. From `docs/datasets/nytimes_noise_floor.json`
(`zero_and_duplicate_rows_removed`): LID 55.0 to 56.9, contrast 1.265 to
1.276, hubness 2.32 to 2.78, Gini 0.78 to 0.82.

## Fallbacks

Decided from the mechanism checks, not from the gate table alone:

- **Hubness at or below 4 but LID above 60 for the run.** The residual is
  uniform but not shaped enough. Follow-up rung: a wider
  `tangent_hidden_dim` or a second modulated layer in the tangent head.
- **Hubness back above 10 despite the constant radius.** The hubs come from
  the direction distribution, not the ratio, and the probe's conclusion was
  incomplete. Re-run `tools/probes/trunk_scale_probe.py` adapted to the new
  class on the `v3` checkpoints before designing further.
- **Radius pinned at `radius_min`.** The set critic prefers the sheet. That
  is a critic finding for the family page, not a generator change.
- **Radius pinned at `radius_max`.** The critic prefers pure residual, and
  the trunk is contributing nothing the critic values; read with the
  direction rank before deciding.

## What this does not do

- No change to the critic, the selector, or any other family's configs.
- No per-sample radius.
- No handling of openai's narrow cone; that family's `spherical` rung is a
  separate design.
- No change to `linear_skip` or to the `v0` to `v2c` configs.
