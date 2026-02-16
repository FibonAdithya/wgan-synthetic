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
