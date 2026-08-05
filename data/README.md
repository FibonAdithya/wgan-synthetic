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

`--cache-dir` (default `data/cache`) is where the downloaded HDF5 itself
lives — a single large, immutable file per family, separate from the `.npy`
subsets cut from it. On a multi-user box it is worth pointing at a shared
location (e.g. `--cache-dir /shared/ann-cache`) so the multi-gigabyte
download happens once for everyone instead of once per user.

If a family's corpus holds fewer rows than requested (NYTimes is the one
case among the six where this is plausible), the fetcher clamps to what
exists and prints a notice naming the requested count, the actual count, and
the file it wrote — check for that notice rather than assuming a filename
like `nytimes_1m.npy` always holds exactly 1,000,000 rows.

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

`src/eval/plot_descriptor_grid.py` depends on `center: false` and
`whiten: false`, since it interprets dimension `(row * 4 + col) * 8 + bin` as
a specific spatial cell and orientation bin; centering or whitening would
break that mapping.

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
- `run_metadata.json` — required only when the run's config sets
  `preprocess.center` or `preprocess.whiten`. It records the transform that
  was fitted on the training split, which is what the samples have to be
  mapped back through before they can be compared against a real corpus in
  its original coordinates. A run that needs it and lacks it is skipped, not
  silently sampled in the transformed space.

Which ladder it drives is chosen with `--dataset` (`sift` by default, `deep`
for the DEEP ladder); the table naming each variant's run directory lives in
that family's page under `docs/datasets/`. Pass `--root` to point at a
different tree. Variants missing a required file are skipped with a message
rather than failing the run, since checkpoints commonly live only on the
training box.

Samples are written to `<output-dir>/samples/<variant>.npy` and reused as the
report's input, so they can be inspected independently.
