"""Read-only probe: does SIFT's zero pattern show 4x4x8 cell structure?

Tests competing groupings rather than assuming dim // 8 is right.
"""
import numpy as np

PATH = "/workspace/wgan-synthetic/data/sift_base.npy"
N = 200_000

x = np.load(PATH, mmap_mode="r")
print(f"shape={x.shape} dtype={x.dtype}")
rng = np.random.default_rng(0)
rows = np.sort(rng.choice(x.shape[0], size=min(N, x.shape[0]), replace=False))
sub = np.asarray(x[rows], dtype=np.float32)

zero = (sub == 0.0)
print(f"overall exact_zero_fraction = {zero.mean():.4f}")
print(f"per-dim zero rate: min={zero.mean(0).min():.4f} max={zero.mean(0).max():.4f}")

nnz = (~zero).sum(axis=1)
print(f"nnz per vector: mean={nnz.mean():.2f} std={nnz.std():.2f} "
      f"min={nnz.min()} max={nnz.max()}")
p = 1.0 - zero.mean()
print(f"binomial-equivalent nnz std if gates were independent = "
      f"{np.sqrt(128 * p * (1 - p)):.2f}")

# Correlation of the zero-indicator across dimensions.
z = zero.astype(np.float32)
z = z - z.mean(0, keepdims=True)
sd = z.std(0, keepdims=True)
sd[sd == 0] = 1.0
z /= sd
corr = (z.T @ z) / z.shape[0]
np.fill_diagonal(corr, np.nan)

d = np.arange(128)
pairs = {
    "same cell (i//8)": (d[:, None] // 8) == (d[None, :] // 8),
    "same orientation (i%8)": (d[:, None] % 8) == (d[None, :] % 8),
    "same group of 16": (d[:, None] // 16) == (d[None, :] // 16),
    "same group of 4": (d[:, None] // 4) == (d[None, :] // 4),
}
cell = d // 8
row, col = cell // 4, cell % 4
grid_dist = np.abs(row[:, None] - row[None, :]) + np.abs(col[:, None] - col[None, :])
pairs["adjacent cells (grid dist 1)"] = (grid_dist == 1)
pairs["distant cells (grid dist >=3)"] = (grid_dist >= 3)

print("\nmean zero-indicator correlation by pair class:")
for name, mask in pairs.items():
    m = mask & ~np.eye(128, dtype=bool)
    print(f"  {name:32s} {np.nanmean(corr[m]):+.4f}  (n={m.sum()})")
print(f"  {'all off-diagonal':32s} {np.nanmean(corr):+.4f}")

# Does an entire cell go empty together more often than independence predicts?
cell_zero = zero.reshape(-1, 16, 8).all(axis=2)
print(f"\nwhole-cell-empty rate (observed)   = {cell_zero.mean():.5f}")
per_dim_p = zero.mean(0).reshape(16, 8)
print(f"whole-cell-empty rate (independent) = {per_dim_p.prod(axis=1).mean():.5f}")
print(f"empty cells per vector: mean={cell_zero.sum(1).mean():.3f} "
      f"std={cell_zero.sum(1).std():.3f} max={cell_zero.sum(1).max()}")
