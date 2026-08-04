# Data contract

This project expects descriptors in one of these formats:

- `.npy` file with shape `[N, D]`, dtype float32 preferred.
- `.fvecs` file (Faiss format): each vector stored as `[int32 dim][float32 * dim]`.

`D` is whatever the dataset family is — 128 for `sift`, 960 for `gist`, 96
for `deep`, 100 for `glove`, 256 for `nytimes`, 1536 for `openai`. It is
declared per config as `data.descriptor_dim` and checked on load; nothing in
the pipeline assumes a particular width.

## Fetched subsets

    python -m src.data.fetch <dataset>

downloads the family's ann-benchmarks HDF5 into a shared cache once and cuts
two reproducible subsets from it, landing at `data/<dataset>_<rows>.npy` —
`data/deep_250k.npy`, `data/sift_1m.npy`, and so on. These files are not
tracked in git; treat `data/` as a local cache and refetch rather than
copying subsets between machines. The HDF5 download is atomic and
single-flight, so several agents on one box can run the command concurrently
without racing.

## Preprocessing contract

The training and generation pipeline assumes:

1. Input vectors are converted to `float32`.
2. Optional centering is computed from train split only.
3. Optional whitening uses train covariance only.
4. Final L2 normalization is applied per vector.

This contract is implemented in `src/data/dataset.py` and saved into training artifacts so sampling/evaluation uses the same transform.

`data.metric` (`l2` or `angular`, default `l2`) sits alongside these but is
not one of them: it records the distance the real corpus is searched under,
not a transform to apply. Setting it does not normalize anything, and
`l2_normalize` is configured independently.

## Example layout

- `data/sift_250k.npy`
- `data/sift_base.fvecs`

## Smoke mode

If no dataset path is supplied, training script can generate synthetic unit-norm vectors for sanity checks (`data.synthetic_if_missing: true` in config).

## What the variant comparison expects

`python -m src.eval.compare_variants` reads, for each variant, a run
directory containing both:

- `best_generator.pt` — the checkpoint.
- `run_config.yaml` — written by the training script. Required, because the
  generator architecture is rebuilt from it; the checkpoint alone is not
  enough to reconstruct the model.

The variants it drives are the SIFT ladder, defined by `configs/sift/v0.yaml`
through `configs/sift/v2.yaml`; the table naming each variant's run directory
now lives in `docs/datasets/sift.md`. Pass `--root` to point at a different
tree. Variants missing either file are skipped with a message rather than
failing the run, since checkpoints commonly live only on the training box.

Samples are written to `<output-dir>/samples/<variant>.npy` and reused as the
report's input, so they can be inspected independently.
