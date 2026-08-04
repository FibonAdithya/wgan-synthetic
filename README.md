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

## Datasets

Each family gets its own ladder of variants and its own gate. Only SIFT has
been trained so far; the other five have a `v0` baseline config and a
documented profile waiting to be measured.

| Family | Dim | Metric | Ladder | Page |
|---|---|---|---|---|
| `sift` | 128 | `l2` | `v0`–`v2` trained | `docs/datasets/sift.md` |
| `gist` | 960 | `l2` | `v0` defined, not trained | `docs/datasets/gist.md` |
| `deep` | 96 | `angular` | `v0` defined, not trained | `docs/datasets/deep.md` |
| `glove` | 100 | `angular` | `v0` defined, not trained | `docs/datasets/glove.md` |
| `nytimes` | 256 | `angular` | `v0` defined, not trained | `docs/datasets/nytimes.md` |
| `openai` | 1536 | `angular` | `v0` defined, not trained | `docs/datasets/openai.md` |

Variant numbers are per dataset and are comparable only within one family.
The SIFT ladder lives in `configs/sift/`; every other family has a single
`v0.yaml` under its own directory in `configs/`. To see all four SIFT variants overlaid on real SIFT
in one report:

    python -m src.eval.compare_variants \
        --real-path data/sift_250k.npy \
        --output-dir runs/eda_variants

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
   - `python -m src.data.fetch sift`
3. Train:
   - `python -m src.train.train_wgan_gp --config configs/sift/v0.yaml`
   - Check `data.real_path` in the config points at a file you have. The five
     newer families name the fetcher's output (`data/deep_250k.npy` and so
     on); the SIFT ladder configs still name `data/sift_base.npy`, the path
     the trained runs used, so edit it if you fetched instead.
4. Evaluate:
   - `python -m src.eval.evaluate_distribution --real-path <path_to_real.npy_or_fvecs> --checkpoint runs/wgan_sift1m/best_generator.pt --config runs/wgan_sift1m/run_config.yaml --output-dir runs/wgan_sift1m/eval`
   - `python -m src.eval.evaluate_file_to_file --real-path <path_to_real.npy_or_fvecs> --synthetic-path <path_to_synthetic.npy_or_fvecs> --output-dir runs/file_eval`
5. Sample:
   - `python -m src.sample.generate --checkpoint runs/wgan_sift1m/best_generator.pt --config runs/wgan_sift1m/run_config.yaml --num-samples 1000000 --output-path runs/wgan_sift1m/synthetic.npy`

## Notes

- Use exact preprocessing parity with your source descriptors.
- For full 1M-row training, ensure enough GPU memory for larger batch sizes.
- Default config includes a synthetic fallback for smoke tests.
