"""How many latent factors does SIFT's zero-pattern correlation need?

Sets the rank for a low-rank correlated-gate design instead of hard-coding a
grouping the data does not support.
"""
import numpy as np

PATH = "/workspace/wgan-synthetic/data/sift_base.npy"
N = 200_000

x = np.load(PATH, mmap_mode="r")
rng = np.random.default_rng(0)
rows = np.sort(rng.choice(x.shape[0], size=N, replace=False))
zero = (np.asarray(x[rows], dtype=np.float32) == 0.0)

z = zero.astype(np.float64)
z -= z.mean(0, keepdims=True)
sd = z.std(0, keepdims=True)
sd[sd == 0] = 1.0
z /= sd
corr = (z.T @ z) / z.shape[0]

w = np.linalg.eigvalsh(corr)[::-1]
total = w.sum()
print("eigenvalue spectrum of the 128x128 zero-indicator correlation:")
for r in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
    print(f"  rank {r:3d}: {w[:r].sum() / total:6.2%} of total correlation mass")
print(f"  top 8 eigenvalues: {np.round(w[:8], 2)}")

# How much is the single global sparsity level worth on its own?
nnz = (~zero).sum(1).astype(np.float64)
lead = np.abs(np.linalg.eigh(corr)[1][:, -1])
print(f"\nleading eigenvector: min={lead.min():.3f} max={lead.max():.3f} "
      f"(flat => a single global level)")
print(f"nnz std = {nnz.std():.2f}; independent-gate equivalent = "
      f"{np.sqrt(128 * (~zero).mean() * zero.mean()):.2f}")

# Residual structure after removing the global level, per grouping.
resid = corr - np.outer(lead, lead) * w[0]
d = np.arange(128)
np.fill_diagonal(resid, np.nan)
for name, mask in (
    ("same group of 4", (d[:, None] // 4) == (d[None, :] // 4)),
    ("same orientation (i%8)", (d[:, None] % 8) == (d[None, :] % 8)),
    ("same cell (i//8)", (d[:, None] // 8) == (d[None, :] // 8)),
):
    m = mask & ~np.eye(128, dtype=bool)
    print(f"residual after global level, {name:24s} {np.nanmean(resid[m]):+.4f}")
print(f"residual after global level, {'all off-diagonal':24s} "
      f"{np.nanmean(resid):+.4f}")
