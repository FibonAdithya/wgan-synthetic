"""Control: do neighbour features add anything a pointwise critic lacks?

If the raw 128 dims already separate real from fake near-perfectly, the
neighbour features are solving a problem the critic does not have.
Also pushes reference size far beyond 2048 to test the trend properly.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REAL = "/workspace/wgan-synthetic/data/sift_base.npy"
FAKES = {
    "v2 gated": "/workspace/eda-out/samples/x100k_sparse_clamp4.npy",
    "v1_5 mlp": "/workspace/eda-out/samples/x100k_improved.npy",
}
NQ = 4000
KMAX = 20
SIZES = [2048, 16384, 65536, 200_000]


def l2(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1e-8)


def knn_dist(q, r, k):
    out = np.empty((q.shape[0], k), dtype=np.float64)
    q_sq = np.einsum("ij,ij->i", q, q)
    r_sq = np.einsum("ij,ij->i", r, r)
    for s in range(0, q.shape[0], 256):
        blk = q[s : s + 256]
        d2 = q_sq[s : s + 256, None] + r_sq[None, :] - 2.0 * (blk @ r.T)
        np.maximum(d2, 0.0, out=d2)
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        out[s : s + 256] = np.sqrt(np.sort(np.take_along_axis(d2, idx, axis=1), axis=1))
    return out


def auc(xr, xf, rng):
    x = np.vstack([xr, xf])
    y = np.concatenate([np.zeros(len(xr)), np.ones(len(xf))])
    p = rng.permutation(len(x))
    x, y = x[p], y[p]
    h = len(x) // 2
    mu, sd = x[:h].mean(0), x[:h].std(0) + 1e-9
    clf = LogisticRegression(max_iter=5000).fit((x[:h] - mu) / sd, y[:h])
    return roc_auc_score(y[h:], clf.decision_function((x[h:] - mu) / sd))


rng = np.random.default_rng(0)
real = l2(np.asarray(np.load(REAL, mmap_mode="r")[:300_000], dtype=np.float32))
pool = real[:250_000]
q_real = real[250_000 : 250_000 + NQ]

for name, path in FAKES.items():
    fake = l2(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32))
    q_fake = fake[rng.choice(fake.shape[0], size=NQ, replace=False)]
    print(f"\n===== {name} =====")
    print(f"raw 128 dims (what the pointwise critic already sees): "
          f"AUC = {auc(q_real, q_fake, rng):.4f}")
    for m in SIZES:
        r = pool[rng.choice(pool.shape[0], m, replace=False)]
        fr = np.log1p(knn_dist(q_real, r, KMAX))
        ff = np.log1p(knn_dist(q_fake, r, KMAX))
        a5 = auc(fr[:, :5], ff[:, :5], rng)
        a20 = auc(fr[:, :20], ff[:, :20], rng)
        comb = auc(np.hstack([q_real, fr[:, :5]]), np.hstack([q_fake, ff[:, :5]]), rng)
        print(f"  M={m:7d}: neigh k=5 AUC={a5:.4f}  k=20 AUC={a20:.4f}  "
              f"raw+neigh AUC={comb:.4f}  median r1={np.median(fr[:, 0]):.4f}")

# For scale: how far is the nearest of M, vs the true 1-NN in the full set?
print("\n=== nearest-neighbour distance vs reference size (real queries) ===")
for m in SIZES:
    r = pool[rng.choice(pool.shape[0], m, replace=False)]
    d = knn_dist(q_real, r, 1)[:, 0]
    print(f"  M={m:7d}: median nearest distance = {np.median(d):.4f}")
print("  (documented median 5-NN distance within the full real set = 0.5153)")
