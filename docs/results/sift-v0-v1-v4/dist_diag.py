"""Distribution diagnostics for the SIFT v0/v1/v4 report, each against a floor.

Every diagnostic here answers "how far is this rung from real" in units nobody
can read without knowing how far *real is from itself*. So the real corpus is
split into two disjoint equal-size halves: one is the reference every rung is
scored against, the other is scored against it the same way and becomes the
baseline row. A rung only differs from real on a diagnostic where it exceeds
that row.

These are diagnostics, not the gate (AGENTS.md invariant 1).
"""

import json
from pathlib import Path

import numpy as np

from src.eval.eda.metrics import wasserstein1
from src.eval.evaluate_distribution import covariance_fro, mmd_rbf, pairwise_hist_l1

OUT = Path("/workspace/keep/sift-v0-v1-v4/distribution.json")
CORPUS = "/workspace/data-cache/sift_1m.npy"
LADDER = "/workspace/keep/sift-ladder"
RUNGS = {
    "v0": f"{LADDER}/samples_v0.npy",
    "v1": f"{LADDER}/samples_v1.npy",
    "v1_30k": "/workspace/keep/sift-v0-v1-v4/samples_v1_30k.npy",
    "v4": f"{LADDER}/samples_v4.npy",
}
N = 20000  # equal-N with the eda_report run
SEED = 42


def l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def per_dim_w1(a, b):
    # The same estimator eda_report's worst_dimensions uses, so the numbers here
    # and the ones in summary.json are directly comparable.
    return np.array([wasserstein1(a[:, d], b[:, d]) for d in range(a.shape[1])])


def diagnostics(ref, x, rng):
    w1 = per_dim_w1(ref, x)
    return {
        "w1_mean": float(w1.mean()),
        "w1_max": float(w1.max()),
        "w1_median": float(np.median(w1)),
        "cov_fro": float(covariance_fro(ref, x)),
        "mmd_rbf": float(mmd_rbf(ref, x, gamma=1.0)),
        "pairwise_hist_l1": float(pairwise_hist_l1(ref, x)),
    }


def main():
    rng = np.random.default_rng(SEED)
    real = l2(np.load(CORPUS).astype(np.float32))
    idx = rng.choice(real.shape[0], size=2 * N, replace=False)
    ref, probe = real[idx[:N]], real[idx[N:]]
    print(f"real reference {ref.shape}, disjoint probe {probe.shape}")

    out = {
        "conditions": {"n": N, "seed": SEED, "preprocess": "l2", "corpus": CORPUS},
        "real_vs_real": diagnostics(ref, probe, rng),
    }
    print("real_vs_real", json.dumps(out["real_vs_real"], indent=1))
    for name, path in RUNGS.items():
        x = l2(np.load(path).astype(np.float32))[:N]
        out[name] = diagnostics(ref, x, rng)
        print(name, json.dumps(out[name], indent=1))

    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
