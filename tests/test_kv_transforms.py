# tests/test_kv_transforms.py
import numpy as np
import torch

from src.kv.transforms import head_keys, head_queries, n_rep, pad0, pos_scale, xbox


def _w(seed, rows, cols):
    return torch.as_tensor(
        np.random.default_rng(seed).normal(size=(rows, cols)), dtype=torch.float32
    )


def test_head_keys_takes_rows_g_times_128():
    """Catches slicing the wrong 128-row block of W_k."""
    h, wk = _w(0, 3, 16), _w(1, 2 * 128, 16)
    np.testing.assert_allclose(head_keys(h, wk, 1), h @ wk[128:256].T, rtol=1e-6)


def test_head_queries_gqa_mapping_and_row_order():
    """Catches query head j reading KV head other than j // n_rep, or the 5
    queries of one h not being consecutive rows."""
    h, wk, wq = _w(0, 2, 16), _w(1, 2 * 128, 16), _w(2, 2 * 5 * 128, 16)
    r = n_rep(wq, wk)
    assert r == 5
    q = head_queries(h, wq, 1, r)
    assert q.shape == (2 * 5, 128)
    for i in range(2):
        for j in range(5):
            jj = 5 * 1 + j
            np.testing.assert_allclose(
                q[i * 5 + j],
                h[i] @ wq[jj * 128 : (jj + 1) * 128].T,
                rtol=1e-5,
                atol=1e-5,
            )


def test_pos_scale_steps_at_floor_scale():
    """Catches an off-by-one in floor((pos+1)/8192): pos 8190 is unscaled, 8191 is the first scaled."""
    s = pos_scale(torch.tensor([0, 8190, 8191, 16383]), 0.1, 8192.0)
    np.testing.assert_allclose(
        s, [1.0, 1.0, 1 + 0.1 * np.log(2), 1 + 0.1 * np.log(3)], rtol=1e-6
    )


def test_xbox_preserves_mips_ranking():
    """Catches a wrong augmentation (e.g. norm instead of sqrt(M^2 - |x|^2))."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 8)).astype(np.float32) * rng.uniform(
        0.2, 3, (200, 1)
    ).astype(np.float32)
    q = rng.normal(size=(20, 8)).astype(np.float32)
    mips = np.argsort(-(q @ base.T), 1)[:, :10]
    xb, qx = xbox(base).astype(np.float64), pad0(q).astype(np.float64)
    l2 = np.argsort(((qx[:, None, :] - xb[None]) ** 2).sum(-1), 1)[
        :, :10
    ]  # float64: no near-tie flips
    np.testing.assert_array_equal(mips, l2)
