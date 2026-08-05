"""Match the covariance eigenvalue spectrum of generated batches to real ones.

Sits alongside the pairwise-distance regularizer in `train_wgan_gp` as the
second optional generator penalty. Nothing here is family-specific -- it reads
a batch and returns a scalar -- but the property it targets is one some
families have more of than others. DEEP descriptors are PCA-compressed CNN
embeddings, so their variance is unevenly distributed across directions in a
way the WGAN critic does not reliably enforce. Same failure mode that
motivated the distance regularizer on the SIFT track.

How uneven, measured on 50k rows of the real DEEP train split rather than
assumed: participation ratio 45.3 of a possible 96, largest normalized
eigenvalue 7.0x the isotropic 1/96, smallest 0.13x, ratio first-to-last 53.6.
The top 10 directions hold 36.6% of the variance and the top 50 hold 82%.
That is a real anisotropy but a moderate one -- not the steep decay a
PCA-compressed set is often assumed to have, which is worth knowing before
attributing much to this term. See docs/datasets/deep.md.

The penalty compares the *shape* of the two spectra: each is divided by its own
trace before comparison, so it measures how variance is distributed across
directions rather than how much there is in total. Overall scale is already
pinned by the unit-norm constraint.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _normalized_spectrum(x: Tensor, eps: float) -> Tensor:
    """Sorted eigenvalues of x's covariance, scaled to sum to one."""
    # float() unconditionally: under AMP this runs inside an autocast region,
    # and eigvalsh on an fp16 covariance is both badly conditioned and, on
    # some builds, unimplemented. The cast is free at realistic batch shapes
    # (512x96 here) and makes the term safe to enable alongside amp: true.
    x = x.float()
    centered = x - x.mean(dim=0, keepdim=True)
    n = max(centered.shape[0] - 1, 1)
    cov = (centered.T @ centered) / n
    # eigvalsh, not eigvals: the covariance is symmetric, and the symmetric
    # solver is both cheaper and free of the complex-valued output that would
    # otherwise have to be discarded. Clamp because a singular covariance --
    # any batch with fewer rows than dimensions -- yields eigenvalues that are
    # zero up to float error, and float error can put them slightly negative.
    eigenvalues = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    return eigenvalues / (eigenvalues.sum() + eps)


def spectrum_distance(real: Tensor, fake: Tensor, *, eps: float = 1.0e-8) -> Tensor:
    """Mean absolute gap between the two trace-normalized spectra.

    Non-negative, zero when the spectra match, and differentiable with respect
    to `fake`. Returns a float32 scalar.
    """
    real_spectrum = _normalized_spectrum(real.detach(), eps)
    fake_spectrum = _normalized_spectrum(fake, eps)
    return torch.abs(real_spectrum - fake_spectrum).mean()
