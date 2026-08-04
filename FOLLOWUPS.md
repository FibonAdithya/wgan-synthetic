# Follow-ups

Tracked here rather than in GitHub Issues, which are disabled on this repo.

## ANN-difficulty panels (`src/eval/ann_difficulty.py`, PR #4)

### 1. `k_eff == 1` silently discards every query

`survivor_mask` requires `dist[:, 0] < dist[:, -1]`. At `k_eff == 1` those are the
same column, so the condition is never true and every query is dropped:

```
compute(x, k=1, ...) -> lid_median None, discarded 500/500
```

Reachable via `--ann-k 1`, and more plausibly via any series with only 2 rows,
where `knn` clamps `k_eff` to 1. The report renders `n/a` and the panel trace
vanishes, so nothing is *wrong* — but nothing tells the reader why either.

Either reject `k < 2` in `compute`, or have `ann_condition_note` call out a
series whose `discarded_queries == num_rows`.

### 2. `ann_condition_note`'s divergent branch is untested

`eda_report.ann_condition_note` is the most branch-heavy new function in the
report, and the unequal-N case was verified by hand rather than by test. Both
branches are pure string assembly over a `Series` / `AnnMetrics` pair, so a
direct unit test with two stub metrics objects is a few lines. Today only the
uniform branch is exercised, incidentally, through the end-to-end HTML test.

### 3. Scattered imports in `tests/test_ann_difficulty.py`

Six `from src.eval.ann_difficulty import ...` statements are interleaved between
test groups, recording the TDD order rather than the finished artefact.
`tests/test_eda_report.py` puts its imports at the top; match that.

Related: `test_compute_discards_all_tied_neighbour_queries_without_producing_inf_or_nan`
calls `compute` and `summary` before the line that imports them. It works because
module-level imports all execute before any test runs, but that is exactly the
fragility the convention prevents.

### 4. `--ann-max-rows` governs a non-ANN panel

`nn_distances` lost its `max_rows=20000` default and now takes `args.ann_max_rows`,
so the flag silently controls the pre-existing within-set k-NN distance panel too.
The help text says so and the default is unchanged, so nothing regresses — but a
user tuning ANN cost will move that panel without expecting to. Either rename to
`--eval-max-rows`, or give `nn_distances` its own flag defaulting from the same
constant.

### 5. Minor

- `lid_discarded_queries` is a count returned as `float` for `Dict[str, Optional[float]]`
  homogeneity. Harmless at current scale, but `format(1200000, '.6g')` renders
  `1.2e+06` in the stats table.
- `fig_ann_profile`: when `populated` is empty for a column, `continue` leaves a
  bare subplot with no axis titles and no note. Only reachable if every series is
  fully degenerate.

## Run infrastructure

### Off-box sync is specified but not implemented

`docs/superpowers/specs/2026-08-03-structured-gate-generator-design.md` calls
for syncing run artifacts off the vast.ai box, whose `workspace_is_volume` is
`false` -- nothing survives recycle or destroy. Resume protects against
contention and preemption, but only if checkpoints leave the machine.

Still needed: sync `summary.json`, logs and `run_metadata.json` on the
`eval_every` cadence, and pull `best_generator.pt` after each arm. Deferred
because it is an operational concern with no natural home in this codebase
and no test that would prove it works. A shell script beside
`data/sample_sift1m_100k.sh` is the likely shape.

### Known gaps from the final whole-branch review (2026-08-03)

These were deliberately left unfixed when finishing `ann/difficulty-panels`.
They are honest known gaps, not planned work with an owner or timeline.

1. **Resume does not restore RNG state, dataloader position, or `GradScaler`
   state.** A resumed run replays the same shuffle order and the same `z`
   sequence from wherever `set_seed` puts the global RNG, rather than
   continuing the stream that produced the checkpoint. Currently zero impact
   on AMP specifically, since every shipped config sets `amp: false` and
   `GradScaler` is a no-op when disabled.

2. **`WGAN_GPU_LOCK_DIR` is a test hook that doubles as a mutual-exclusion
   bypass.** Two agents on one box, one with the env var set and one without,
   get different lock files for the same physical card -- the lock only
   coordinates processes that agree on the lock directory.

3. **The CUDA context is created before the lock is acquired.**
   `gpu_lock_key` calls `torch.cuda.get_device_properties`, which triggers
   PyTorch's lazy CUDA init. A process about to be refused the lock still
   allocates a context on the card first, which can push the current holder
   into OOM.

4. **All 13 configs in `configs/` still say `device: auto`**, and the launch
   snippets in `PROJECT_DOCUMENTATION.md` and `README.md` still show it, so
   the documented happy path now hard-fails under `strict=True` unless
   `CUDA_VISIBLE_DEVICES` is exported first. The refusal itself is
   intentional; no config or doc snippet was updated to match it.

5. **`tests/test_gpu_lock.py::test_lock_is_released_even_when_the_body_raises`
   passes even with the `finally` body deleted**, because CPython's refcount
   GC closes the file handle and drops the flock as the `with` block unwinds
   via exception. The asserted behaviour (the lock is released) is real, but
   the test does not prove the `finally` clause is what does it.

### The log-ratio EMA target is not checkpointed

`LogRatioTarget` in `src/train/log_ratio.py` holds an EMA of the real
batches' log-ratio profile, and `--resume` does not restore it: a resumed
run rebuilds the average from scratch.

This is deliberate, not an oversight. At decay 0.99 the average resettles
within roughly a hundred steps, so the cost is a brief transient on a 100k
step run. It is explicitly unlike the generator weight EMA at decay 0.999,
which `save_checkpoint` does persist and where a silent restart degrades
`best_generator.pt` with no error.

If `lid_reg_decay` is ever raised toward 0.999, revisit this — the argument
above stops holding.

### `lid_reg_alpha` cannot be started from `distance_reg_alpha`'s scale

The plan for `configs/sift_gan_v4.yaml` suggests starting `lid_reg_alpha` at
`0.1`, calling it "v1_5's proven scale" for `distance_reg_alpha`. That analogy
does not hold and should not be used to pick the starting value.

`distance_reg` is a single scalar, `|dist_real - dist_fake|`. `log_ratio_penalty`
returns an L1 **sum** over `k - 1` components — 19 of them at the default
`lid_reg_k=20` — so its magnitude sits on a different scale than
`distance_reg` and grows with `k`. Measured for calibration (`k=10`, so 9
terms, 256-point batches): an L1 gap of roughly `1.75` between a 32-D blob and
a rank-4 blob padded to 32 dimensions, against roughly `0.01` for a batch
measured against a fresh draw from its own distribution. Neither number is
anywhere near `distance_reg`'s typical scale, and both would scale up further
at `k=20`.

Recommend picking `lid_reg_alpha` from a measured `lid_reg` value on a real
batch at the config's actual `lid_reg_k`, not by analogy to `distance_reg_alpha`.

Design observation, not a change to make now (the plan specifies a `.sum()`
reduction and that is what shipped): a `.mean()` reduction in
`log_ratio_penalty` would have made `alpha` invariant to `k`, removing this
whole caveat. Left as `.sum()` because that is what the ratified plan called
for.
