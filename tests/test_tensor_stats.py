import numpy as np

from src.train.train_wgan_gp import tensor_stats


def test_existing_keys_are_preserved():
    rng = np.random.default_rng(0)
    stats = tensor_stats(rng.normal(size=(200, 8)), rng.normal(size=(200, 8)))
    for key in ("mean_l2", "var_l2", "cov_fro"):
        assert key in stats


def test_identical_inputs_give_zero_gaps():
    rng = np.random.default_rng(1)
    x = np.abs(rng.normal(size=(200, 8)))
    x[x < 0.5] = 0.0
    stats = tensor_stats(x, x.copy())
    assert stats["zero_fraction_gap"] == 0.0
    assert stats["per_dim_zero_rate_l1"] == 0.0
    assert stats["nnz_std_gap"] == 0.0
    assert stats["negative_fraction"] == 0.0


def test_negative_fraction_measures_fake_only():
    stats = tensor_stats(np.ones((10, 4)), -np.ones((10, 4)))
    assert stats["negative_fraction"] == 1.0


def test_zero_fraction_gap():
    stats = tensor_stats(np.zeros((10, 4)), np.ones((10, 4)))
    assert stats["zero_fraction_gap"] == 1.0


def test_per_dim_zero_rate_catches_misplaced_sparsity():
    real = np.ones((10, 4))
    real[:, :2] = 0.0
    fake = np.ones((10, 4))
    fake[:, 2:] = 0.0
    stats = tensor_stats(real, fake)
    assert stats["zero_fraction_gap"] == 0.0
    assert stats["per_dim_zero_rate_l1"] == 1.0


def test_nnz_std_gap_catches_uncorrelated_sparsity():
    real = np.zeros((10, 4))
    real[:, :2] = 1.0
    fake = np.zeros((10, 4))
    fake[:5, :] = 1.0
    stats = tensor_stats(real, fake)
    assert stats["zero_fraction_gap"] == 0.0
    assert stats["nnz_std_gap"] == 2.0


def test_all_values_are_plain_floats():
    rng = np.random.default_rng(2)
    stats = tensor_stats(rng.normal(size=(50, 4)), rng.normal(size=(50, 4)))
    assert all(type(value) is float for value in stats.values())
