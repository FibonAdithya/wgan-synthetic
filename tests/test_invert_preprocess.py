import numpy as np
import pytest

from src.data.dataset import (
    PreprocessConfig,
    _fit_preprocess_state,
    apply_preprocess,
    invert_preprocess,
)


def _sample(n: int = 400, d: int = 96, seed: int = 0) -> np.ndarray:
    """Anisotropic data, so whitening is a non-trivial transform."""
    rng = np.random.default_rng(seed)
    scale = np.linspace(1.0, 0.05, d).astype(np.float32)
    return (rng.normal(size=(n, d)) * scale).astype(np.float32)


@pytest.mark.parametrize(
    "center,whiten",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_invert_round_trips_apply_without_l2(center: bool, whiten: bool):
    x = _sample()
    cfg = PreprocessConfig(center=center, whiten=whiten, l2_normalize=False)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=x.shape[1], cfg=cfg)
    round_tripped = invert_preprocess(apply_preprocess(x, state), state)
    np.testing.assert_allclose(round_tripped, x, rtol=1e-3, atol=1e-3)


def test_invert_returns_float32():
    x = _sample()
    cfg = PreprocessConfig(center=True, whiten=True, l2_normalize=False)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=x.shape[1], cfg=cfg)
    assert invert_preprocess(apply_preprocess(x, state), state).dtype == np.float32


def test_invert_moves_whitened_data_off_the_identity_covariance():
    """Guards against a no-op inverse silently passing the round-trip test."""
    x = _sample()
    cfg = PreprocessConfig(center=True, whiten=True, l2_normalize=False)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=x.shape[1], cfg=cfg)
    whitened = apply_preprocess(x, state)
    # Whitened data has near-unit variance in every direction; the original
    # does not, because _sample builds in a decaying scale.
    assert np.ptp(whitened.var(axis=0)) < 0.5
    assert np.ptp(invert_preprocess(whitened, state).var(axis=0)) > 0.5


def test_invert_is_a_no_op_when_no_transform_was_fitted():
    x = _sample()
    cfg = PreprocessConfig(center=False, whiten=False, l2_normalize=False)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=x.shape[1], cfg=cfg)
    np.testing.assert_array_equal(invert_preprocess(x, state), x)


def test_l2_normalization_is_documented_as_not_invertible():
    """apply_preprocess discards vector norms; invert cannot restore them.

    The inverse undoes centering and whitening only. This test pins that
    contract so a future change does not quietly claim a full inverse.
    """
    x = _sample()
    cfg = PreprocessConfig(center=False, whiten=False, l2_normalize=True)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=x.shape[1], cfg=cfg)
    normalized = apply_preprocess(x, state)
    recovered = invert_preprocess(normalized, state)
    np.testing.assert_allclose(recovered, normalized, rtol=1e-6, atol=1e-6)
