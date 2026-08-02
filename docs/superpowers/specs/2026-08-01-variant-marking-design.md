> **AI-generated working note.** Written by Claude during development and kept
> for provenance. It is not the source of truth: where this file disagrees with
> `PROJECT_DOCUMENTATION.md`, the latter wins.

# Variant marking, sparse cleanup, and doc split

Date: 2026-08-01
Branch: `eda/sift-eda`

## Problem

Three lines of work sit on three branches and cannot be read against each
other:

- `main` — plain WGAN-GP.
- `feat-wgan-gp-v2-ema` (`420f008`) — generator EMA, training safety fixes,
  and the sparse generator v2 merged in.
- `worktree-sparse-generator` (`05812d9`) — an older parallel sparse line.
- `eda/sift-eda` — the EDA report and ANN difficulty panels, branched off the
  EMA commits but before the sparse merge.

Three consequences:

1. The model variants that were actually trained are identified only by run
   directory name (`long_improved`, `x100k_sparse_clamp4`), and those names do
   not say what changed. No config in `configs/` reproduces them one-to-one.
2. `src/eval/eda_report.py` can already overlay any number of synthetic sets on
   the real data, but nothing drives it across the variants, so the comparison
   is re-typed by hand each time.
3. `docs/superpowers/` holds AI-generated specs and plans alongside
   human-maintained `README.md` and `PROJECT_DOCUMENTATION.md`, with nothing
   marking which is which or which one wins on a disagreement.

## What was actually run

Read from the `run_config.yaml` files under `runs/`:

| run dir | EMA | `distance_reg_alpha` | `generator_type` | steps |
|---|---|---|---|---|
| `bench_baseline` | — | 0.0 | dense | 3k |
| `long_baseline` | — | 0.0 | dense | 30k |
| `long_ema_only` | 0.999 | 0.0 | dense | 30k |
| `x100k_ema_only` | 0.999 | 0.0 | dense | 100k |
| `bench_improved` | 0.999 | 0.1 (256 pts) | dense | 3k |
| `long_improved` | 0.999 | 0.1 (256 pts) | dense | 30k |
| `x100k_improved` | 0.999 | 0.1 (256 pts) | dense | 100k |
| `x100k_sparse_clamp4` | 0.999 | 0.1 (256 pts) | sparse | 100k |

Four distinct configurations, not three. The distance regularizer is its own
change, and the sparse run differs from the EMA-only run by *two* changes. Run
length (3k / 30k / 100k) is an independent axis and is not a variant.

## Design

### 1. Variant scheme

Four variants, v-numbered, each exactly one delta from the one above so an EDA
overlay attributes each change:

| Variant | Delta | Config | Runs |
|---|---|---|---|
| `v0` | plain WGAN-GP | `configs/sift_gan_v0.yaml` | `long_baseline`, `bench_baseline` |
| `v1` | + generator EMA (0.999) | `configs/sift_gan_v1.yaml` | `long_ema_only`, `x100k_ema_only` |
| `v1_5` | + distance reg (α 0.1, 256 pts) | `configs/sift_gan_v1_5.yaml` | `long_improved`, `x100k_improved`, `bench_improved` |
| `v2` | + gated non-negative generator | `configs/sift_gan_v2.yaml` | `x100k_sparse_clamp4` |

Each config is reconciled against the `run_config.yaml` of its runs so it
reproduces work already done. `configs/bench_sparse.yaml` and
`configs/sweeps/` are benchmark and sweep artifacts; they stay as they are.

The variant names are used identically in config filenames, the
`PROJECT_DOCUMENTATION.md` table, and the EDA report legend.

### 2. `generator_type` hard rename

The generator architecture axis is renamed `mlp | sparse` → `mlp | gated`, and
`SparseGenerator` → `GatedGenerator`.

`gated` rather than `v2`: `generator_type` is an architecture axis sitting
underneath the variant numbering, and `v1_5` and `v2` differ at that layer by
exactly this one flag. The v-numbers name variants; `generator_type` names the
mechanism.

This is a hard rename with no compatibility alias, chosen deliberately. Blast
radius:

- `build_generator` in `src/models/generator.py` (the `kind == "sparse"` branch
  and the `SparseGenerator` class).
- `configs/bench_sparse.yaml`.
- `tests/test_generator.py`, `test_generator_factory.py`,
  `test_evaluate_distribution.py`, `test_ema.py`, `test_train_smoke.py`.
- `runs/x100k_sparse_clamp4/run_config.yaml` — the one artifact that genuinely
  breaks. It is gitignored and local, so it is hand-edited as part of this
  change and the checkpoint is confirmed to still load afterwards.
- Checkpoint provenance labelling added in `a194ad9` — verify whether it
  persists the generator type string and fix that load path if so.

### 3. Variant comparison driver

`src/eval/compare_variants.py`, invoked as `python -m src.eval.compare_variants`
to match every other tool in the repo.

- Imports `src.sample.generate` and `src.eval.eda_report` directly rather than
  shelling out, so it is testable and has no shell dependency.
- Resolves each variant's checkpoint, generates N samples per variant, and
  calls the report with `v0=…`, `v1=…`, `v1_5=…`, `v2=…` labels.
- Checkpoint paths, sample count and output directory are CLI arguments with
  defaults pointing at the run directories in the table above.
- Variants absent from disk are skipped with a message rather than aborting the
  run, so a partial comparison still produces a report.

### 4. Documentation split

Human-maintained, source of truth:

- `README.md` — entry point. Gains a doc map naming which files are
  human-maintained and which are AI working notes, plus a variant
  quick-reference pointing into `PROJECT_DOCUMENTATION.md`.
- `PROJECT_DOCUMENTATION.md` — technical reference. Gains a **Model variants**
  section carrying the table from §1, and an updated EDA section covering
  multi-set overlay and the new driver.
- `data/README.md` — data contract. Documents what the driver expects on disk.

AI working notes, kept for provenance, explicitly not authoritative:

- `docs/superpowers/` — new `README.md` stating that the specs and plans there
  are AI-generated, kept for provenance, and superseded by
  `PROJECT_DOCUMENTATION.md` on any disagreement. A one-line banner is added at
  the top of each of the four existing spec/plan files and this one.
- `.superpowers/sdd/` — gitignored tooling state. The `docs/superpowers/README.md`
  notes that it exists and why it is not tracked.

### 5. Git integration and cleanup

1. Merge `feat-wgan-gp-v2-ema` into `eda/sift-eda`. Conflicts are expected in
   `requirements.txt` (both sides appended), `src/eval/evaluate_distribution.py`,
   `src/train/train_wgan_gp.py`, and `pytest.ini` (EMA side only). Resolution
   keeps both sides' additions.
2. Diff `worktree-sparse-generator` against the merged tree and confirm it holds
   nothing unique — its factory, support metrics, `bench_sparse.yaml` and tests
   all appear to have reached the EMA branch via merge `b9d9ef7`. Verify rather
   than assume, then delete the branch and its worktree.
3. Review `src/models/generator.py` and `src/train/train_wgan_gp.py` for dead
   paths and duplicated sparse logic left by the two lineages, and make
   docstrings name the variants consistently with §1.
4. `preserve-wgan-improvements-wip` and `experiment/wgan-improvements` are left
   untouched.

## Verification

- `python3 -m pytest` green across the combined suite after the merge and the
  rename: `test_ann_difficulty`, `test_ema`, `test_generator`,
  `test_generator_factory`, `test_tensor_stats`, `test_train_smoke`,
  `test_evaluate_distribution`.
- A smoke run of `python -m src.eval.compare_variants` against the
  synthetic-fallback data, producing a report with all present variants
  labelled.
- `runs/x100k_sparse_clamp4` still loads and evaluates after the rename.

## Out of scope

- Retraining any variant. The configs are reconciled against runs that already
  exist; no new runs are produced.
- Changing the EDA report's panels or metrics.
- Refactoring unrelated to the variant marking, the rename, or the doc split.
