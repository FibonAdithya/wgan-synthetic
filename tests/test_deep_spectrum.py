import torch

from src.deep.spectrum import spectrum_distance


def _anisotropic(n: int, d: int, decay: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scale = torch.linspace(1.0, decay, d)
    return torch.randn(n, d, generator=g) * scale


def test_distance_is_near_zero_for_identical_batches():
    x = _anisotropic(256, 32, 0.1, seed=0)
    assert spectrum_distance(x, x).item() < 1e-5


def test_distance_is_positive_for_mismatched_spectra():
    real = _anisotropic(256, 32, 0.02, seed=0)
    fake = _anisotropic(256, 32, 1.0, seed=1)
    assert spectrum_distance(real, fake).item() > 0.01


def test_distance_grows_as_the_spectra_diverge():
    real = _anisotropic(256, 32, 0.05, seed=0)
    near = _anisotropic(256, 32, 0.10, seed=1)
    far = _anisotropic(256, 32, 1.00, seed=1)
    assert spectrum_distance(real, near) < spectrum_distance(real, far)


def test_distance_is_invariant_to_overall_scale():
    """Trace normalization means a rescaled batch has the same spectrum shape."""
    real = _anisotropic(256, 32, 0.1, seed=0)
    fake = _anisotropic(256, 32, 0.1, seed=1)
    base = spectrum_distance(real, fake)
    scaled = spectrum_distance(real, fake * 10.0)
    torch.testing.assert_close(base, scaled, rtol=1e-4, atol=1e-4)


def test_distance_is_differentiable_with_respect_to_fake():
    real = _anisotropic(128, 16, 0.05, seed=0)
    fake = _anisotropic(128, 16, 0.5, seed=1).requires_grad_(True)
    spectrum_distance(real, fake).backward()
    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()
    assert fake.grad.abs().sum() > 0


def test_distance_returns_a_scalar():
    x = _anisotropic(64, 8, 0.1, seed=0)
    assert spectrum_distance(x, x).shape == torch.Size([])


def test_distance_is_finite_for_a_rank_deficient_batch():
    """Fewer rows than dimensions makes the covariance singular; eigvalsh must
    still return finite values rather than NaN."""
    real = _anisotropic(8, 32, 0.1, seed=0)
    fake = _anisotropic(8, 32, 0.5, seed=1)
    assert torch.isfinite(spectrum_distance(real, fake)).all()
