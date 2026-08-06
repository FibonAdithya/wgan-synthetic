# WGAN ANN-difficulty synthesizer

Train WGAN-GP models that reproduce the nearest-neighbour search difficulty of
six benchmark vector families, so ANN algorithms can be developed and stressed
against synthetic corpora instead of the real ones. The target is not a
matching distribution. A synthetic set succeeds when an index finds it as
hard, and hard in the same way, as the real set; matching marginals is
evidence about why a gate failed, not the gate itself.

## Documentation map

Human-maintained, and the source of truth:

- `README.md` — this file. Setup and the commands you run day to day.
- `PROJECT_DOCUMENTATION.md` — technical reference: architecture, training
  objective, data contract, evaluation, and the model variant table.
- `data/README.md` — the data contract and what the evaluation tools expect
  on disk.
- `docs/datasets/` — one page per benchmark family: structure, source,
  canonical N and k, measured profile, model family, ladder and gate bands.

AI working notes, kept for provenance and **not** authoritative:

- `docs/superpowers/` — design specs and implementation plans written by
  Claude during development. See `docs/superpowers/README.md`. Where these
  disagree with `PROJECT_DOCUMENTATION.md`, the latter wins.

Reviews and vendored external references, also **not** authoritative:

- `AGENTIC-REVIEW.md` — a cold-read review of how ready this repo is for
  autonomous agents. Written against one commit; its counts are a snapshot.
- `docs/ai-first-development-workflow.md` — a general AI-workflow guide copied
  from the sibling `tig-cpu` repository. It describes no part of this project
  and is kept only so the citations in `AGENTIC-REVIEW.md` resolve.

## Datasets

Each family gets its own ladder of variants and its own gate. SIFT and DEEP
have trained ladders; the other four have a `v0` baseline config and a
documented profile waiting to be measured.

| Family | Dim | Metric | Ladder | Page |
|---|---|---|---|---|
| `sift` | 128 | `l2` | `v0`–`v2` trained | `docs/datasets/sift.md` |
| `gist` | 960 | `l2` | `v0` defined, not trained | `docs/datasets/gist.md` |
| `deep` | 96 | `angular` | `v0`–`v2` trained | `docs/datasets/deep.md` |
| `glove` | 100 | `angular` | `v0` defined, not trained | `docs/datasets/glove.md` |
| `nytimes` | 256 | `angular` | `v0` defined, not trained | `docs/datasets/nytimes.md` |
| `openai` | 1536 | `angular` | `v0` defined, not trained | `docs/datasets/openai.md` |

Variant numbers are per dataset and are comparable only within one family.
The SIFT and DEEP ladders live in `configs/sift/` and `configs/deep/`; every
other family has a single `v0.yaml` under its own directory in `configs/`. To see all four SIFT variants overlaid on real SIFT
in one report:

    python -m src.eval.compare_variants \
        --real-path data/sift_base.npy \
        --output-dir runs/eda_variants

Which variants that overlays, and where each one's trained run lives, is
`configs/eval/<dataset>.yaml` — `configs/eval/sift.yaml` by default, or
`--dataset deep` for the DEEP ladder, or `--variants-manifest <path>` for a
file of your own. `runs/` is gitignored, so a fresh clone has none of the run
directories a manifest names: the command above will stop and tell you which
paths are missing and what would produce them. Copy the runs in, edit the
manifest to name runs you do have, or pass `--allow-missing` to report on
whichever variants resolved.

`data/sift_base.npy` is what the four trained SIFT checkpoints were actually
trained against, not the fetcher's `data/sift_250k.npy` subset — those are
different corpora. See issue #15 ("`data.real_path` names a file the
fetcher does not produce") for why the SIFT configs still point at
`sift_base.npy` rather than a fetched subset.

## What this project provides

- Descriptor loader and preprocessing pipeline.
- A fetcher that pulls any of the six families and cuts reproducible subsets.
- WGAN-GP generator and critic models (MLPs).
- Training script with checkpoints and reproducible configs.
- Evaluation script for distribution and neighborhood fidelity, including the
  ANN-difficulty statistics that decide whether a synthetic set is usable.
- Deterministic sampling script from a frozen generator checkpoint.

## Quick start

1. Create environment and install dependencies:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Fetch a dataset (see `data/README.md`):
   - `python -m src.data.fetch <dataset>` — `<dataset>` is one of `sift`,
     `gist`, `deep`, `glove`, `nytimes`, `openai`.
3. Train:
   - `python -m src.train.train_wgan_gp --config configs/<dataset>/v0.yaml`
   - Check `data.real_path` in the config points at a file you have. The five
     non-SIFT families name the fetcher's output (`data/deep_1m.npy` and so
     on); the SIFT ladder configs still name `data/sift_base.npy`, the path
     the trained runs used, so edit it if you fetched instead. See issue #15
     for the open question of reconciling that.
4. Evaluate:
   - `python -m src.eval.evaluate_distribution --real-path <path_to_real.npy_or_fvecs> --checkpoint runs/x100k_improved/best_generator.pt --config runs/x100k_improved/run_config.yaml --output-dir runs/x100k_improved/eval`
   - `python -m src.eval.evaluate_file_to_file --real-path <path_to_real.npy_or_fvecs> --synthetic-path <path_to_synthetic.npy_or_fvecs> --output-dir runs/file_eval`
5. Sample:
   - `python -m src.sample.generate --checkpoint runs/x100k_improved/best_generator.pt --config runs/x100k_improved/run_config.yaml --num-samples 1000000 --output-path runs/x100k_improved/synthetic.npy`
   - Runs whose config sets `preprocess.whiten` or `preprocess.center` (e.g.
     `configs/deep/v2.yaml`) must be sampled through
     `python -m src.eval.compare_variants --dataset <family>` instead, which
     inverts the fitted transform. `src.sample.generate` does not, and would
     emit vectors in the transformed space.

## Notes

- Use exact preprocessing parity with your source descriptors.
- For full 1M-row training, ensure enough GPU memory for larger batch sizes.
- Default config includes a synthetic fallback for smoke tests.
