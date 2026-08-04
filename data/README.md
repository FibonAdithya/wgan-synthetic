# Data contract for SIFT1M-like training

This project expects 128-dimensional descriptors in one of these formats:

- `.npy` file with shape `[N, 128]`, dtype float32 preferred.
- `.fvecs` file (Faiss format): each vector stored as `[int32 dim][float32 * dim]`.

## Preprocessing contract

The training and generation pipeline assumes:

1. Input vectors are converted to `float32`.
2. Optional centering is computed from train split only.
3. Optional whitening uses train covariance only.
4. Final L2 normalization is applied per vector.

This contract is implemented in `src/data/sift1m_dataset.py` and saved into training artifacts so sampling/evaluation uses the same transform.

## Example layout

- `data/sift1m_base.fvecs`
- `data/sift1m_base.npy`

## Smoke mode

If no dataset path is supplied, training script can generate synthetic unit-norm vectors for sanity checks (`data.synthetic_if_missing: true` in config).

## What the variant comparison expects

`python -m src.eval.compare_variants` reads, for each variant, a run
directory containing both:

- `best_generator.pt` — the checkpoint.
- `run_config.yaml` — written by the training script. Required, because the
  generator architecture is rebuilt from it; the checkpoint alone is not
  enough to reconstruct the model.

The directories it looks in are listed in the variant table in
`PROJECT_DOCUMENTATION.md`. Pass `--root` to point at a different tree.
Variants missing either file are skipped with a message rather than failing
the run, since checkpoints commonly live only on the training box.

Samples are written to `<output-dir>/samples/<variant>.npy` and reused as the
report's input, so they can be inspected independently.

## DEEP image descriptors

The deep track expects 96-dimensional descriptors from
`deep-image-96-angular`, as `.npy` with shape `[N, 96]`, float32.

Fetch and subset them with:

    python -m src.deep.download --out-dir data --cache-path data/deep-image-96-angular.hdf5

That downloads `deep-image-96-angular.hdf5` once (here, into `data/`) and
writes `data/deep96_250k.npy` (pipeline smoke runs) and `data/deep96_1m.npy`
(the real runs). Neither the HDF5 nor the subsets are committed.

`--cache-path` defaults to `/workspace/data-cache/deep-image-96-angular.hdf5`,
a location that only exists on the GPU box and that a normal machine cannot
create (`mkdir` under `/workspace` fails with `PermissionError`). Pass
`--cache-path` explicitly, as above, anywhere else. On the GPU box,
`scripts/deep_gpu_setup.sh` already passes `--cache-path` pointing at the
shared cache, so the download there is not repeated per agent.

DEEP vectors arrive already L2-normalized, dense, and signed — the opposite of
SIFT on every count. The `deep_gan_v2` config additionally PCA-whitens; its
samples must be drawn with `src.deep.sample`, which inverts that transform.
`src.sample.generate` does not, and would return vectors in whitened space.
