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

## Fold the remaining `l2_normalize` copies onto `eda_report.maybe_l2_normalize`

`plot_descriptor_grid` now reuses `eda_report.maybe_l2_normalize` rather than
carrying its own copy, but four remain: `evaluate_file_to_file.py:43`,
`plot_distance_cdf.py:30`, `plot_distance_cdf_pillow.py:46` and
`plot_embedding_clusters.py:33`. All four are byte-identical to each other
and encode the same rule as `dataset.apply_preprocess`, so each is a place
that rule can drift out of step with training preprocessing without any test
noticing. Mechanical, but it touches four eval entry points that the glyph
work has no reason to disturb, so it wants its own change.

## `build_generator` rejects the `sparse` run configs it wrote itself

`b29e317` renamed `generator_type: sparse` to `gated` in the code but left the
already-written run configs alone, and `build_generator` raises
`ValueError: Unknown generator_type: sparse` rather than accepting the old
name. v2's only trained checkpoint on tig-gpu
(`/workspace/wgan-sparse-v2/runs/x100k_sparse_clamp4/run_config.yaml`) still
says `sparse`, so **no current tool can load v2 from its real run config** --
`compare_variants` and `plot_descriptor_grid` both fail on it. Verifying the
glyph grid against v2 needed a hand-patched copy of that config.

The fix is an alias in `build_generator` (`sparse` -> `gated`) rather than
editing checkpoints on disk: a run config is a historical record of a run that
happened, and the rename should not have invalidated it. Wants its own change
since it touches model loading for every consumer.

## v2's checkpoint lives outside the repo the variant table points at

`cv.VARIANTS` resolves every variant under one `--root`, but on tig-gpu
v0/v1/v1_5 are in `/workspace/wgan-synthetic/runs/` while v2 is in
`/workspace/wgan-sparse-v2/runs/`. No single `--root` resolves all four, so
the four-row comparison needs the run directories collected under one tree
first. Either move v2's run into the main checkout, or let `Variant` carry an
absolute run dir.
## DEEP ladder results rest on one seed per rung (PR #6)

`docs/datasets/deep.md` reports three trained rungs at `seed: 42`. The ladder
was run twice — once at `latent_dim: 128` (inherited from SIFT, since
corrected) and once at 96 — which gives two draws rather than one, and the
comparison is not reassuring: `v0`'s IVF gini gap moved tenfold and its
hubness gap doubled under a change the target's effective rank of 65 says
should barely bind. Several of those swings exceed the differences between
rungs.

Two claims survive both draws (`v2` closest on LID, `v1` closest on hubness
skew by two orders of magnitude); the IVF gini ordering does not survive at
all and should not be used to rank rungs. Two draws is still two.

Three or four seeds per rung (~35 min each on the RTX 4060) would settle it,
and are a prerequisite for setting this family's gate bands — a band fitted to
either existing draw would be fitted to noise.

Resolved in passing: the `summary.json` those numbers were read out of is now
committed as `docs/datasets/deep_ladder_summary.json`, so the table is
checkable from this repo alone.

## `spectrum_reg_alpha: 0.1` is too small to bind, and v1 may be measuring noise (PR #6)

The seed sweep above should be an **alpha sweep** as well, because there is
reason to think `v1`'s rung is not currently testing what it claims to.

`spectrum_distance` compares trace-normalized spectra, so at `d = 96` each
entry is O(1/96) ≈ 0.0104 and the mean absolute gap between two spectra is
inherently small. Measured on v1's actual architecture (512×96 batch,
DEEP-like anisotropy, at initialization):

| quantity | value |
|---|---|
| `spectrum_distance`, isotropic fake vs anisotropic real | 0.0039 |
| `spectrum_distance`, untrained generator vs real | 0.0014 |
| ‖∇_G adv_loss‖ | 0.170 |
| ‖∇_G spectrum_reg‖ | 0.0205 |
| ‖∇_G (0.1 × spectrum_reg)‖ | **0.0020** — ~1.2% of the adversarial gradient |

So the term contributes about one percent of the generator gradient at its
strongest, and shrinks from there as the spectra converge. `deep_ladder_summary.json`
is consistent with it having done nothing to the property it targets:

| | real | v0 | v1 | v2 |
|---|---|---|---|---|
| effective rank | 65.30 | 63.34 | **63.36** | 64.22 |

`v1` moved effective rank by 0.02 against a 1.96 gap to real. The whitening
rung moved it by 0.88 — roughly 45× more, with no penalty term at all. That
makes the one claim surviving both draws (`v1` closest on hubness skew) most
likely a consequence of a perturbed optimization trajectory rather than of the
regularizer acting through the spectrum. Note that
`test_enabling_the_regularizer_changes_the_generator` needed `alpha: 5.0` to
show the term reaching the weights at all.

Two things to decide together, since the second changes what `alpha` means:

1. Sweep `alpha` over something like `{0.1, 1, 10}` alongside the seeds, and
   report effective rank per rung so the term is judged on the property it
   targets rather than only on downstream ANN metrics.
2. Consider making the penalty scale-free — a relative gap rather than the
   absolute mean, e.g. dividing by the real spectrum's mean entry — so that
   `spectrum_reg_alpha` is comparable to `distance_reg_alpha` and does not
   quietly depend on `descriptor_dim`.

Not urgent: `v1` is not *wrong*, and nothing in `docs/datasets/deep.md`
overclaims for it. But the rung costs 35 minutes a draw and currently cannot
support the conclusion it exists to test.

## `src/eval/ann_difficulty.py` is the last consumer that could inherit `--dataset`

`compare_variants` now selects a family's ladder with `--dataset`, and reads
the search metric off each config as `data.metric`. `ann_difficulty.py` still
measures everything under L2 (phase (c), above). When it learns to read
`data.metric`, `compare_variants` is where the value should be threaded
through from — it already resolves the config for every variant it samples.
