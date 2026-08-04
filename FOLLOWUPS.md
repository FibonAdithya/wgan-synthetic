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

## SIFT configs are out of step with the per-dataset conventions

Two inconsistencies in `configs/sift/v0.yaml` through `v2.yaml`, recorded
together because they should be decided in one deliberate change rather than
as a tail on an unrelated commit. Both were left alone during the
documentation rewrite that found them.

### 1. `output_dir` keeps the flat historical names

The four SIFT configs write to `runs/sift_gan_v0` .. `runs/sift_gan_v2`,
while the five newer families use `runs/<dataset>/v*` — `runs/deep/v0`,
`runs/gist/v0`, and so on. Nothing reads these values: no such directory
exists, and `compare_variants` hard-codes the historical run directories
(`long_baseline`, `x100k_improved`, `x100k_sparse_clamp4`, …) instead. So
this only decides where a *future* SIFT training run would land.

### 2. `data.real_path` names a file the fetcher does not produce

All four set `data.real_path: data/sift_base.npy`, the path the trained runs
used. `python -m src.data.fetch sift` writes `data/sift_250k.npy` and
`data/sift_1m.npy`, so the obvious quick start — fetch, then train
`configs/sift/v0.yaml` — fails on a fresh machine with a missing file. The
five newer families are self-consistent: `configs/deep/v0.yaml` names
`data/deep_250k.npy`, exactly what `fetch deep` produces. `README.md`
currently works around this by telling the reader to check the path and edit
it by hand.

Repointing `real_path` is the part that needs a decision rather than a patch:
it would redefine what SIFT's `v0` reproduces, since the existing checkpoints
were trained against `sift_base.npy` and a 250k fetched subset is not the
same corpus. The alternatives are to have the fetcher also emit a
`sift_base.npy`, or to accept the redefinition and re-record what each SIFT
variant means. Either way the run-directory rename above should ride along,
so the two land as one reviewed change.

## Phase (c) prerequisite: re-measure the angular families' real profiles

`ann_difficulty.py` currently measures everything under L2, including for
the four `angular` families (`deep`, `glove`, `nytimes`, `openai`) — see
"`data.metric`" in `PROJECT_DOCUMENTATION.md`. Once `ann_difficulty.py` reads
`data.metric` and measures under the corpus's own distance (phase (c)), any
"Measured profile" numbers already filled in on those four families' pages
(`docs/datasets/deep.md`, `glove.md`, `nytimes.md`, `openai.md`) will be
L2-measured figures sitting next to figures measured under the metric the
corpus is actually searched with, and will need re-measuring so the report
stays internally comparable. Each of those four pages now says as much in
its "Measured profile" section.
