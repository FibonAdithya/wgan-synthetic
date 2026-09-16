# Holding the local neighbourhood profile: NYTimes `v4`

Design, 2026-09-16. Not authoritative (see `AGENTS.md`); the code and
`PROJECT_DOCUMENTATION.md` win on any conflict once built. Scope: the next
NYTimes rung, `v4`. One change from `v3`: the log-ratio regulariser is turned
on. No other family's configs are touched.

## Problem

`v3` reaches the real corpus's local geometry and then loses it. At its
selected step (9,000) it reads LID `55.68` against real's `55.84`, the first
NYTimes rung inside the real range, at near-real global rank. By step 30,000
LID reads `24.21`. The selection is disqualified on the family's transient
test: it is a point the run passes through, not a state it reaches.

`docs/datasets/nytimes.md` originally read the run's monotonically rising
shared radius (`0.976` to `1.448` against a `1.5` ceiling) as the mechanism.
**That reading is wrong**, and the correction is recorded in the same section.
`tools/probes/radius_probe.py` (job
`wgan-synthetic-20260916T135324Z-a06c18`, commit `a06db84`, lane `cpu`,
exit 0; table under `docs/results/nytimes-radius-probe/`) decoded three frozen
checkpoints at seven shared angles each, one latent draw per checkpoint reused
at every angle, at the canonical conditions.

LID median, by checkpoint and overridden angle:

| | r = 1.1 | r = 1.2 | r = 1.25 | r = 1.3 | r = 1.4 | r = 1.448 |
|---|---|---|---|---|---|---|
| step 9,000 | 52.12 | 55.75 | 57.11 | 58.25 | 59.47 | 59.65 |
| step 20,000 | 29.69 | 31.17 | 31.82 | 32.39 | 33.35 | 33.70 |
| step 30,000 | 21.82 | 22.63 | 23.04 | 23.38 | 23.97 | 24.21 |

Holding the angle at `1.2`, LID falls 33 points as the weights advance.
Holding the weights, the entire angle band moves LID by at most `7.52` (step
9,000), `4.01` (20,000), `2.40` (30,000). The reverse control settles it: on
the trajectory `r = 1.448` coincided with LID `23.90`, but the frozen
step-9,000 weights pushed to that same angle read `59.65` -- better than at
their own learned `1.1957`. LID rises monotonically with `r` at every
checkpoint, so the radius climb masked part of the decay rather than causing
it. **Freezing or re-banding the radius is a dead end.**

What decays is the tangent head, and it decays locally while holding globally:

| | LID | contrast | IVF Gini | effective rank |
|---|---|---|---|---|
| real | 55.84 | 1.269 | 0.796 | 247.9 |
| `tangent_only`, step 9,000 | 59.70 | 1.210 | 0.749 | 221.0 |
| `tangent_only`, step 20,000 | 34.20 | 1.327 | 0.609 | 213.9 |
| `tangent_only`, step 30,000 | 24.57 | 1.441 | 0.474 | 203.6 |

LID falls 59% while effective rank falls 8%. The set keeps its global spread
and loses its local dimension, and contrast rises as neighbours clump. The
trunk collapses too (`trunk_only` effective rank `65.3` to `48.9`) but
contributes almost nothing to output geometry at any step -- `tangent_only` at
step 9,000 is within a point of the full output on every statistic -- so its
collapse is not what the gate sees.

Two further readings from the same table fix where a lever must act:

- **The deficit is local, not global.** At step 9,000 median 1-NN distance is
  `1.1529` against real's `1.1046` while the global distance scale matches
  (median pairwise `1.4054` against `1.4038`, from the committed
  `eda_clean_summary.json`). The synthetic set's nearest neighbours are too
  far apart, which is what its 4%-low contrast measures.
- **`distance_reg` is therefore not the lever.** It penalises
  `|mean_pairwise_real - mean_pairwise_fake|`, a single global scalar. The
  committed summaries record medians rather than means, so the two are not
  the identical quantity, but a global distance-scale penalty has nothing to
  push on when the global scale already agrees to 0.1% and the miss is at
  1-NN. It stays at `0.0`.

## Design

### The one change

`configs/nytimes/v4.yaml` is `configs/nytimes/v3.yaml` with three training
keys changed, and nothing else:

| key | v3 | v4 |
|---|---|---|
| `lid_reg_alpha` | `0.0` (absent) | set by the sizing procedure below |
| `lid_reg_k` | -- | `20` |
| `lid_reg_max_points` | -- | `256` |

`log_ratio_penalty` matches the fake batch's log-ratio profile,
`mean_i log(r_i / r_k)` for `i = 1 .. k-1`, against an EMA of the real
batches' profile. That vector is the sufficient statistic the Hill estimator
reduces to the LID scalar, so it constrains the *shape* of the local
neighbourhood rather than only its scale -- which is what the probe says
collapsed.

### Why `k = 20` and not `100`

The gate measures LID at `k = 100`. Penalising the profile at `k = 20`
constrains genuinely local structure while leaving the gate's `k = 100`
measurement less directly fitted; matching the gate's own `k` would make the
LID result close to tautological. This is a deliberate trade against the
chance of holding LID, not a default inherited from SIFT, and the fallback
table below says what to do if `k = 20` proves too local to hold `k = 100`
LID.

### Sizing `lid_reg_alpha`

**SIFT's `0.015` must not be carried over.** `log_ratio_penalty` is an L1 sum
over `k - 1` terms and its magnitude depends on the family's actual gap;
`tools/probes/lid_reg_scale_probe.py` exists precisely to size it per family
and its docstring says so. It runs as its own `--lane cpu` job before the
training run, pinned with `--device cpu` (see the device fix below) so it
never touches the card, and reports three numbers: the real-vs-real floor (the penalty's
noise level, which alpha must not be multiplying), the untrained-generator
gap, and a trained-generator gap.

Two departures from the probe's defaults, both stated because they change the
answer:

- **Trained reference is `v3`'s step-9,000 checkpoint**, passed through the
  probe's `--v2-checkpoint` / `--v2-config` pair (the argument names are
  SIFT-era; `build_generator` reads whatever model config it is given, so
  no code change is needed for this). The penalty has to do its work in the
  collapse window, not at initialisation.
- **`--adv-loss 0.87`**, `v3`'s run-mean `|adv_loss|` from its committed
  `run_metadata.json`. Sizing at launch instead would use the first-20-step
  mean of `0.33` and yield an alpha roughly three times larger than the
  operating point warrants. (`v3`'s last-50-step mean is `0.96`; the run mean
  is the middle reading and the one this spec uses.)

`--target-fraction` stays at the default `0.05`. The resulting alpha is
written into `v4.yaml` with the probe's three numbers quoted in a comment
beside it, so a reader can see what it was sized against.

### Two bugs that block the sizing probe

`lid_reg_scale_probe.py` resolves its device with
`torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")`. That
line has two separate problems, and fixing only the first leaves the probe
worse than before.

**It cannot parse `auto`.** Every NYTimes config says `device: auto`, and
`torch.device("auto")` raises `RuntimeError: Expected one of cpu, cuda, ...`.
SIFT's configs name `cuda:0`, which is why this never surfaced. Fix the
probe, not the config: use `src.device.resolve_device`, which exists to handle
`auto` and is what the rest of the project calls. Changing `v4.yaml`'s device
instead would make `v4` a two-change rung and leave the trap for the next
family.

**It has no way to stay off the card.** The probe picks its device from the
config alone, so on this box it resolves to CUDA whatever lane it was
submitted to -- and `resolve_device("auto")` would do the same. A `--lane cpu`
job that quietly allocates on the GPU is the exact failure the queue exists to
prevent, and it would land beside whatever holds the per-card lock. So the fix
is both parts: delegate to `resolve_device`, **and** add a `--device` argument
that overrides the config, defaulting to the config's value. The job script
then passes `--device cpu` explicitly, as `radius_probe.py` already does.

Note for the record: `device: auto` does **not** hard-fail training on the
box. `resolve_device` only refuses under `strict=True` with
`CUDA_VISIBLE_DEVICES` unset, and the runner pins it -- `v3` trained to
completion with `device: auto` on 2026-09-16.

### Configs and scripts

Following the family's existing convention:

- `configs/nytimes/v4.yaml` -- the rung.
- `configs/nytimes/v4_seed42.yaml` -- differs from `v4.yaml` only in
  `data.real_path` and `output_dir`, as `v3_seed42.yaml` does from `v3.yaml`.
- `scripts/nytimes_v4_seed42_job.sh` -- modelled on the `v3` job script,
  targeting `/venv/main/bin/python` (the 2026-09-16 box rebuild removed
  `/opt/venvs/wgan-synthetic`), and submitted **without** `--vram-mb` so the
  scheduler does not admit a second job onto the card against
  `train_wgan_gp`'s per-card lock.

Seed 42 and 30,000 steps are unchanged from `v3`, so `v4` pairs exactly
against the existing `v3_seed42` run. A 100,000-step continuation is a
follow-up only if 30,000 holds, not part of this rung.

## Tests

Trainer-side behaviour of `lid_reg` is already covered by
`tests/test_lid_reg_training.py` (absent keys leave the loss unchanged, an
explicit zero behaves the same, an enabled regulariser contributes to the
generator loss, and training completes with it on) and by
`tests/test_log_ratio.py`. This rung does not re-test any of that. It adds:

| test | assertion | mutation it catches |
|---|---|---|
| config pinning (`tests/test_nytimes_configs.py`) | `v4` equals `v3` except exactly the three `lid_reg` keys, and `lid_reg_alpha > 0` | a rung that changes more than it says; an alpha left at zero, which would make `v4` a duplicate of `v3` |
| seed-variant pinning (same file) | `v4_seed42` differs from `v4` only in `data.real_path` and `output_dir` | a seed variant that drifts a training key |
| job script names its config (same file) | the job script references `v4_seed42` | a script pinned to the wrong rung |
| probe device resolution (`tests/test_lid_reg_scale_probe.py`, new) | the probe resolves a config whose `device` is `auto` without raising, and returns what `resolve_device` returns | reverting to `torch.device(cfg["device"])`; a fix that special-cases the string `"auto"` in the probe rather than delegating |
| probe device override (same file) | `--device cpu` wins over a config naming `cuda:0`, and omitting `--device` falls back to the config | an override that is parsed but ignored, which would put a cpu-lane job on the card |

The alpha value itself is deliberately **not** pinned by a test. It is a
measured quantity from the sizing probe, and a test asserting a specific float
would pin a measurement rather than a contract -- it would pass against a
wrong-but-unchanged number and fail on a legitimate re-measure.

## Mechanism checks the trained run reports

The family page's `v4` section reports these beside the gate table:

- **Holdout LID across all 30 evaluations.** The success criterion. Predicted
  flat and near the real holdout reading (`59.44` at these conditions) rather
  than peaking near step 9,000 and decaying. `v3`'s trajectory for comparison:
  `33.07` at step 1,000, peak `54.44` at 9,000, `23.90` at 30,000.
- **`lid_reg` logged per step, against the probe's real-vs-real floor.** If
  the penalty sits at or below the floor, alpha is multiplying noise and the
  run says nothing about the mechanism, whatever LID does.
- **Contrast -- the falsifiable prediction.** The penalty constrains a
  19-component profile, not the LID scalar; contrast is another function of
  the same neighbourhood. Predicted to move from `v3`'s `1.217` toward real's
  `1.269`. If LID holds and contrast does not move, the penalty reaches scale
  but not shape, and that is a finding for the family page.
- **Transient test**, same form as earlier rungs: the selected step's
  neighbouring evaluations within a factor of two of its score.
- **Radius trajectory**, reported for continuity but explicitly no longer
  load-bearing: the probe showed it is not the mechanism.
- **Wall time**, against `v3`'s 68 minutes. The penalty adds an O(n^2)
  pairwise distance over `lid_reg_max_points` rows per generator step.

## Success

The selected `v4` checkpoint is not a transient on the family's existing test,
and holdout LID holds near the real reading across the run rather than peaking
and decaying. Contrast and hubness are reported, not required.

This bar is deliberately narrower than `v3`'s, which demanded all four
statistics inside the real ten-draw range. Nothing in this rung targets
hubness, which the radius probe measured 18% to 29% high across every angle
it swept on the step-9,000 checkpoint -- and rising with `r`, so no angle
escapes it -- so requiring it would set `v4` up to fail by construction.

One measurement note for whoever reads the gate table: IVF Gini carries about
0.8% partition noise -- in the radius probe, an input change of one part in
ten thousand moved it from `0.4942` to `0.4901` -- which is the same size as
`v3_best`'s Gini "miss". Gini differences below about 1% should not be read as
signal in either direction.

## Fallbacks

Decided from the mechanism checks, not the gate table alone:

- **LID holds and contrast moves toward real.** The profile penalty reaches
  shape. The next rung targets hubness, the remaining free miss.
- **LID holds, contrast does not move.** The penalty reaches scale but not
  shape at `k = 20`. Next rung raises `lid_reg_k` toward the gate's `100`,
  accepting that this makes LID more directly fitted, and says so.
- **LID does not hold, and `lid_reg` sits near the real-vs-real floor.** Alpha
  is too small. Re-size against a larger `--target-fraction` before concluding
  anything about the mechanism; this is a sizing failure, not a result.
- **LID does not hold and `lid_reg` sits well above the floor.** The local
  collapse is not preventable by a generator-side penalty on the profile, and
  the critic is the remaining lever. That is the strongest possible argument
  for the critic rung, and it is why this cheap rung runs first.
- **Wall time far above `v3`'s 68 minutes.** Reduce `lid_reg_max_points`
  before reducing steps; the penalty is O(n^2) in that key alone.

## What this does not do

- No change to the generator, the critic, the selector, or the radius band.
- No change to `v0` to `v3` configs, or to any other family's configs.
- No new band in `gates/nytimes.yaml`; every band stays null, as it is for
  the whole family.
- No 100,000-step continuation. That is a follow-up if 30,000 holds.
- No attempt on hubness.
