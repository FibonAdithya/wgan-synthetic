"""Ground reference_size and neighbour_k for the neighbour-aware critic.

Three questions:
  1. LEAK  - if R is drawn from the training split, how often does a real
             training batch point land *in* R and get distance exactly 0?
  2. STABILITY - how much do the features move when R is resampled? A buffer
             too small makes every refresh a moving target for the critic.
  3. SIGNAL - how well do k distance features separate real from v2 output?
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REAL = "/workspace/wgan-synthetic/data/sift_base.npy"
FAKE = "/workspace/eda-out/samples/x100k_sparse_clamp4.npy"
KMAX = 20
NQ = 2000
SIZES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
KS = [1, 3, 5, 10, 20]


def l2(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1e-8)


def knn_dist(q, r, k):
    """Sorted distances from each q to its k nearest rows of r."""
    out = np.empty((q.shape[0], k), dtype=np.float64)
    q_sq = np.einsum("ij,ij->i", q, q)
    r_sq = np.einsum("ij,ij->i", r, r)
    step = 512
    for s in range(0, q.shape[0], step):
        blk = q[s : s + step]
        d2 = q_sq[s : s + step, None] + r_sq[None, :] - 2.0 * (blk @ r.T)
        np.maximum(d2, 0.0, out=d2)
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        part = np.take_along_axis(d2, idx, axis=1)
        out[s : s + step] = np.sqrt(np.sort(part, axis=1))
    return out


rng = np.random.default_rng(0)
real = l2(np.asarray(np.load(REAL, mmap_mode="r")[:300_000], dtype=np.float32))
fake = l2(np.asarray(np.load(FAKE, mmap_mode="r"), dtype=np.float32))
print(f"real {real.shape}  fake {fake.shape}")

# Disjoint pools: buffer drawn from 'pool', queries never overlap it.
pool = real[:250_000]
q_real = real[250_000 : 250_000 + NQ]
q_fake = fake[rng.choice(fake.shape[0], size=NQ, replace=False)]

print("\n=== 1. LEAK: buffer drawn from the split real batches come from ===")
print("expected zero-distance hits per 512-batch, if R subset of train split:")
for m in SIZES:
    print(f"  M={m:6d}: {512 * m / 950_000:6.3f} points per batch")

print("\n=== 2. STABILITY across an independent resample ===")
print(f"{'M':>7} {'rel RMS change':>15} {'corr(r1)':>10} {'corr(r5)':>10}")
stability = {}
for m in SIZES:
    a = knn_dist(q_real, pool[rng.choice(pool.shape[0], m, replace=False)], KMAX)
    b = knn_dist(q_real, pool[rng.choice(pool.shape[0], m, replace=False)], KMAX)
    rel = np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a**2))
    c1 = np.corrcoef(a[:, 0], b[:, 0])[0, 1]
    c5 = np.corrcoef(a[:, 4], b[:, 4])[0, 1]
    stability[m] = (rel, c1, c5)
    print(f"{m:7d} {rel:15.4f} {c1:10.4f} {c5:10.4f}")

print("\n=== 3. SIGNAL: real vs v2 separability from k distance features ===")
print(f"{'M':>7} " + " ".join(f"{'AUC k=' + str(k):>10}" for k in KS))
for m in SIZES:
    r = pool[rng.choice(pool.shape[0], m, replace=False)]
    fr = np.log1p(knn_dist(q_real, r, KMAX))
    ff = np.log1p(knn_dist(q_fake, r, KMAX))
    x = np.vstack([fr, ff])
    y = np.concatenate([np.zeros(NQ), np.ones(NQ)])
    perm = rng.permutation(x.shape[0])
    x, y = x[perm], y[perm]
    half = x.shape[0] // 2
    row = []
    for k in KS:
        clf = LogisticRegression(max_iter=2000).fit(x[:half, :k], y[:half])
        row.append(roc_auc_score(y[half:], clf.decision_function(x[half:, :k])))
    print(f"{m:7d} " + " ".join(f"{v:10.4f}" for v in row))

print("\n=== cost: distances per critic call at batch 512 ===")
for m in SIZES:
    print(f"  M={m:6d}: {512 * m / 1e6:7.2f}M distance evals")
