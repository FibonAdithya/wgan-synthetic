# WGAN SIFT1M Synthesizer

Train a WGAN-GP model to generate synthetic 128D vectors with statistics and neighborhood behavior similar to SIFT1M descriptors.

## What this project provides

- SIFT-like descriptor loader and preprocessing pipeline.
- WGAN-GP generator and critic models (MLPs).
- Training script with checkpoints and reproducible configs.
- Evaluation script for distribution and neighborhood fidelity.
- Deterministic sampling script from a frozen generator checkpoint.

## Quick start

1. Create environment and install dependencies:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Add data (see `data/README.md`).
3. Train:
   - `python -m src.train.train_wgan_gp --config configs/wgan_gp_sift1m.yaml`
4. Evaluate:
   - `python -m src.eval.evaluate_distribution --real-path <path_to_real.npy_or_fvecs> --checkpoint runs/wgan_sift1m/best_generator.pt --config runs/wgan_sift1m/run_config.yaml --output-dir runs/wgan_sift1m/eval`
   - `python -m src.eval.evaluate_file_to_file --real-path <path_to_real.npy_or_fvecs> --synthetic-path <path_to_synthetic.npy_or_fvecs> --output-dir runs/file_eval`
5. Sample:
   - `python -m src.sample.generate --checkpoint runs/wgan_sift1m/best_generator.pt --config runs/wgan_sift1m/run_config.yaml --num-samples 1000000 --output-path runs/wgan_sift1m/synthetic.npy`

## Notes

- Use exact preprocessing parity with your source descriptors.
- For true SIFT1M training, ensure enough GPU memory for larger batch sizes.
- Default config includes a synthetic fallback for smoke tests.
