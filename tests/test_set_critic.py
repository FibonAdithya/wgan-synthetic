"""The learned set critic: one EdgeConv layer over within-batch neighbour
differences. Each test names the mutation it catches."""

import numpy as np
import pytest
import torch

from src.models.critic import neighbourhood_indices


def _unit(seed, n, dim):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return torch.from_numpy(x)


def test_indices_match_brute_force_and_exclude_self():
    """Catches a lost diagonal mask (self would be index 0 of every row).
    torch.cdist is the brute-force oracle here, as in
    test_neighbourhood_critic.py; the no-cdist rule is for src/."""
    x = _unit(0, n=9, dim=6)
    idx = neighbourhood_indices(x, k=3)
    full = torch.cdist(x, x)
    full.fill_diagonal_(float("inf"))
    _, expected = torch.topk(full, 3, dim=1, largest=False, sorted=True)
    assert idx.shape == (9, 3) and idx.dtype == torch.long
    assert torch.equal(idx, expected)
    assert (idx != torch.arange(9)[:, None]).all()


def test_indices_keep_an_exact_copy_as_a_neighbour():
    """Catches self-exclusion by dropping the nearest column, which would
    drop a copy in the row's place."""
    x = _unit(1, n=8, dim=5)
    x[3] = x[0]
    idx = neighbourhood_indices(x, k=2)
    assert idx[0, 0] == 3 and idx[3, 0] == 0


def test_indices_require_more_than_k_rows():
    with pytest.raises(ValueError, match="k"):
        neighbourhood_indices(_unit(2, n=4, dim=3), k=4)
