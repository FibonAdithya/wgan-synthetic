"""Can k-means anchors buy the same signal as a huge random buffer, cheaply?

Random reference points sample the density; centroids summarise it. If 2048
centroids match 65k random points, the mechanism becomes affordable.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REAL = "/workspace/wgan-synthetic/data/sift_base.npy"
FAKE = "/workspace/eda-out/samples/x100k_sparse_clamp4.npy"
NQ = 4000
KMAX = 20


def l2(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1e-8)


def knn_dist(q, r, k):
    k = min(k, r.shape[0])
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
fake = l2(np.asarray(np.load(FAKE, mmap_mode="r"), dtype=np.float32))
q_fake = fake[rng.choice(fake.shape[0], size=NQ, replace=False)]

print(f"raw 128 dims baseline: AUC = {auc(q_real, q_fake, rng):.4f}\n")
print(f"{'buffer':>28} {'k=5':>8} {'k=20':>8} {'raw+k5':>8}")
for m in (2048, 16384):
    r = pool[rng.choice(pool.shape[0], m, replace=False)]
    fr, ff = np.log1p(knn_dist(q_real, r, KMAX)), np.log1p(knn_dist(q_fake, r, KMAX))
    print(f"{'random ' + str(m):>28} {auc(fr[:, :5], ff[:, :5], rng):8.4f} "
          f"{auc(fr[:, :20], ff[:, :20], rng):8.4f} "
          f"{auc(np.hstack([q_real, fr[:, :5]]), np.hstack([q_fake, ff[:, :5]]), rng):8.4f}")

    km = MiniBatchKMeans(n_clusters=m, random_state=0, n_init=3, batch_size=4096)
    c = km.fit(pool).cluster_centers_.astype(np.float32)
    fr, ff = np.log1p(knn_dist(q_real, c, KMAX)), np.log1p(knn_dist(q_fake, c, KMAX))
    print(f"{'kmeans ' + str(m):>28} {auc(fr[:, :5], ff[:, :5], rng):8.4f} "
          f"{auc(fr[:, :20], ff[:, :20], rng):8.4f} "
          f"{auc(np.hstack([q_real, fr[:, :5]]), np.hstack([q_fake, ff[:, :5]]), rng):8.4f}")
