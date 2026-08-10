# Agent guide

Read this first, then follow the links. This file is a router: it says which
document is authoritative, what must not be broken, and what needs a human.
It deliberately restates as little as possible, because a second copy of a
fact is a copy that goes stale.

## What this project is

Train WGAN-GP models that reproduce the *nearest-neighbour search difficulty*
of six benchmark vector families, so ANN algorithms can be stressed against
synthetic corpora instead of the real ones. The target is not a matching
distribution: a synthetic set succeeds when an index finds it as hard, and
hard in the same way, as the real set. Only SIFT has trained models today;
the other five families have a `v0` config and a documented profile.

## Source of truth, in order

`README.md#documentation-map` holds the documentation map. When two documents disagree,
the one higher in this list wins:

1. **The code** in `src/` and the configs in `configs/`. If the docs describe
   behaviour the code does not have, the docs are wrong.
2. **`PROJECT_DOCUMENTATION.md`** — technical reference: architecture,
   training objective, data contract, evaluation, variant table.
3. **`README.md`**, **`data/README.md`**, **`docs/datasets/*.md`** — setup and
   day-to-day commands, the on-disk data contract, and one page per family.
4. **`docs/superpowers/`** — *not authoritative*. Design specs and plans
   written by Claude during development, kept for the reasoning behind
   decisions. They are not updated as the code changes, and they lose to
   `PROJECT_DOCUMENTATION.md` on any conflict. See
   `docs/superpowers/README.md#ai-working-notes`.

GitHub Issues on the public mirror (`FibonAdithya/wgan-synthetic`) are the
issue tracker; they are disabled on `upstream`. They record known problems;
they are not a description of how things work.

## Five invariants

These are silent until violated. Nothing in the test suite catches them, and
each is easy to break while believing you are making progress.

1. **ANN-difficulty is the gate; everything else is a diagnostic.** The
   decision procedure is the four statistics in `src/eval/ann_difficulty.py`
   (LID median, relative contrast, hubness skew, IVF cell-balance Gini),
   compared against a per-family band. `mmd_rbf`, `cov_fro`,
   `pairwise_hist_l1` and the rest are diagnostics that explain *why* a gate
   failed. Reporting "MMD improved, looks good" is a misreading of the
   project. See `PROJECT_DOCUMENTATION.md#ann-difficulty--the-gate`.
2. **Variant numbers are per-family.** Each family's ladder is numbered
   independently from `v0`. SIFT `v2` and a future GIST `v2` are unrelated,
   and comparing variant numbers across families is meaningless. Compare
   within one family only. See `PROJECT_DOCUMENTATION.md#model-variants-the-per-dataset-ladder`.
3. **Canonical N and k are locked per dataset, and the measured statistics
   are self-queried subsample figures.** They have no absolute meaning and
   are **not** comparable with published benchmark values, which are measured
   on the full corpus against the real query set. Do not "sanity check" a
   measured LID or recall against the literature. The locked pair lives in
   the family's page under `docs/datasets/`.
4. **A checkpoint is only loadable beside its `run_config.yaml`.**
   `generator_type` is not recorded in the checkpoint; the architecture is
   rebuilt from the run config at load time. Moving a `.pt` file away from
   its config makes it unloadable. See `PROJECT_DOCUMENTATION.md#generator_type`.
5. **`data/sift_base.npy` and `data/sift_250k.npy` are different corpora.**
   `sift_base.npy` is what the trained SIFT checkpoints were fit against;
   `python -m src.data.fetch sift` produces `sift_250k.npy` and
   `sift_1m.npy`. The SIFT configs still name `sift_base.npy`, so a fresh
   fetch-then-train fails on a missing file until you edit the path — and
   editing it changes which corpus the variant reproduces. See issue #15,
   "SIFT configs are out of step with the per-dataset conventions".

## What "done" means

Run from the repo root, on Python 3.12:

    make check

That is ruff lint, ruff format check, and the pytest suite. It runs in
seconds and is CPU-only — no GPU and no dataset needed. It is the same
command CI runs (`.github/workflows/ci.yml`). No target uses `|| true`; a red
suite is a failure, not a warning.

`make format` rewrites files and is not part of `check`. Only format the
files you touched; never run it repo-wide.

## What requires a human

Do not decide these yourself. Raise them and stop.

- Anything in an open issue described as needing a decision rather than a
  patch. Repointing SIFT's `data.real_path` ("`data.real_path` names a file
  the fetcher does not produce") is the clearest case: it redefines what
  SIFT's `v0` reproduces, because the existing checkpoints were trained on a
  different corpus.
- **Setting or tightening gate bands.** Bands start wide and tighten as a
  family's ladder shows what is achievable. Choosing a number is a judgement
  about what counts as success.
- **Any change to what a variant number means** — re-pointing a config,
  renaming a rung, or changing a rung's delta. A ladder rung is a historical
  record of a run that happened.
- Changing a pinned version in `requirements.txt`.

## Where to look

| Task | Start here |
|---|---|
| Understand the goal and the gate | `PROJECT_DOCUMENTATION.md#ann-difficulty--the-gate` |
| Set up and run day-to-day commands | `README.md` (quick start) |
| Get data onto disk | `data/README.md`, `src/data/fetch.py` |
| Facts about one family (N, k, profile, bands) | `docs/datasets/<family>.md` |
| Preprocessing contract | `data/README.md`, `src/data/dataset.py` |
| Model architectures and `generator_type` | `src/models/generator.py`, `PROJECT_DOCUMENTATION.md#model-architecture`, `PROJECT_DOCUMENTATION.md#generator_type` |
| The variant ladder | `PROJECT_DOCUMENTATION.md#model-variants-the-per-dataset-ladder`, `configs/<family>/` |
| Training loop and config keys | `src/train/train_wgan_gp.py` |
| Evaluation and metric definitions | `src/eval/`, `PROJECT_DOCUMENTATION.md#metric-definitions` |
| Compare SIFT variants in one report | `src/eval/compare_variants.py` |
| The EDA report's panels and prose | `src/eval/eda/panels.py` |
| Known bugs and open questions | GitHub Issues (`gh issue list`) |
| Why a decision was made (non-authoritative) | `docs/superpowers/specs/` |
