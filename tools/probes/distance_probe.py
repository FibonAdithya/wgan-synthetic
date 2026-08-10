"""Is the group-of-4 effect a hard block, or smooth adjacency decay?

Distinguishes a block-structured gate from a local/convolutional one.
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
w, v = np.linalg.eigh(corr)
resid = corr - np.outer(v[:, -1], v[:, -1]) * w[-1]
np.fill_diagonal(resid, np.nan)

d = np.arange(128)
sep = np.abs(d[:, None] - d[None, :])
print("residual correlation (global level removed) vs |i - j|:")
for gap in range(1, 13):
    m = sep == gap
    print(f"  |i-j| = {gap:2d}: {np.nanmean(resid[m]):+.4f}")
print(f"  |i-j| >= 16: {np.nanmean(resid[sep >= 16]):+.4f}")

# Does crossing a group-of-4 boundary matter, holding |i-j| fixed at 1?
adj = sep == 1
same4 = (d[:, None] // 4) == (d[None, :] // 4)
print("\nneighbouring dims (|i-j| = 1), split by group-of-4 boundary:")
print(f"  same group of 4:     {np.nanmean(resid[adj & same4]):+.4f} "
      f"(n={(adj & same4).sum()})")
print(f"  crosses a boundary:  {np.nanmean(resid[adj & ~same4]):+.4f} "
      f"(n={(adj & ~same4).sum()})")

# Circular orientation distance within a cell, assuming 8 bins per cell.
same_cell = (d[:, None] // 8) == (d[None, :] // 8)
o = d % 8
odist = np.minimum(np.abs(o[:, None] - o[None, :]), 8 - np.abs(o[:, None] - o[None, :]))
print("\nwithin-cell residual correlation vs circular orientation distance:")
for k in range(1, 5):
    m = same_cell & (odist == k) & ~np.eye(128, dtype=bool)
    if m.sum():
        print(f"  orientation gap {k}: {np.nanmean(resid[m]):+.4f} (n={m.sum()})")
