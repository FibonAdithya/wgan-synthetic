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
