"""Binned distribution data for the SIFT figure in the write-up.

Writes a small JSON holding three panels per corpus, so the figure can be drawn
anywhere without moving sample files off the box:

  values      coordinate-value histogram, with the mass at exactly zero and the
              mass below zero kept OUT of the histogram and reported as scalars.
              A continuous density cannot show a point mass, and the point mass
              at zero is the whole distinction between the dense and the gated
              generator, so it is carried separately.
  counts      histogram of the per-row non-zero count, over the integers 0..d.
  profile     mean over query rows of log(r_m / r_k), m = 1..k, at the same
              conditions the gate uses. This is the curve R_LR matches and the
              curve Eq. (LID) reduces to a scalar.

Bin edges are fixed in this file, not derived from the data, so every corpus is
binned identically and the curves can be overlaid.

Usage:

    python scripts/sift_dist_panels.py \
        --real-path /workspace/data-cache/sift_1m.npy \
        --synthetic-path v0=<...>.npy --synthetic-path v4=<...>.npy \
        --out runs/sift/panels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Locked to the gate's canonical conditions (docs/datasets/sift.md), so the
# profile panel is comparable with the LID medians in the results table.
N_ROWS = 20000
K_NN = 100
SEED = 42

# Fixed so every corpus is binned identically. Real SIFT lives in [0, 0.42]
# after L2 normalisation; the dense generator reaches about -0.10.
VALUE_LO, VALUE_HI, VALUE_BINS = -0.15, 0.50, 130


def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def value_panel(x: np.ndarray) -> dict:
    """Coordinate values, with the zero point mass and the negative mass split out."""
    flat = x.reshape(-1)
    total = flat.size
    is_zero = flat == 0.0
    zero_frac = float(is_zero.sum()) / total
    neg_frac = float((flat < 0.0).sum()) / total
    # The histogram covers the non-zero coordinates only. Including the exact
    # zeros would pile 23% of the mass into whichever bin straddles 0 and make
    # a point mass look like a tall bar, which is a different claim.
    edges = np.linspace(VALUE_LO, VALUE_HI, VALUE_BINS + 1)
    counts, _ = np.histogram(flat[~is_zero], bins=edges)
    width = edges[1] - edges[0]
    density = counts / (total * width)  # normalised over ALL coordinates
    return {
        "edges": edges.tolist(),
        "density": density.tolist(),
        "zero_fraction": zero_frac,
        "negative_fraction": neg_frac,
    }


def count_panel(x: np.ndarray) -> dict:
    """Per-row non-zero count."""
    nz = (x != 0.0).sum(axis=1)
    d = x.shape[1]
    counts = np.bincount(nz, minlength=d + 1)[: d + 1]
    return {
        "support": list(range(d + 1)),
        "fraction": (counts / counts.sum()).tolist(),
        "mean": float(nz.mean()),
        "std": float(nz.std(ddof=1)),
    }


def profile_panel(x: np.ndarray, rng: np.random.Generator) -> dict:
    """Mean over rows of log(r_m / r_k), m = 1..k, exact kNN in chunks."""
    n = min(N_ROWS, x.shape[0])
    idx = rng.choice(x.shape[0], size=n, replace=False)
    q = np.ascontiguousarray(x[idx])
    sq = (q * q).sum(axis=1)
    out = np.empty((n, K_NN), dtype=np.float64)
    chunk = 512
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        # Squared distances; the constant sq[s:e] term does not change the order.
        d2 = sq[None, :] - 2.0 * (q[s:e] @ q.T) + sq[s:e, None]
        np.maximum(d2, 0.0, out=d2)
        # Drop the self-match, which is the zero on the diagonal.
        d2[np.arange(e - s), np.arange(s, e)] = np.inf
        part = np.partition(d2, K_NN - 1, axis=1)[:, :K_NN]
        part.sort(axis=1)
        out[s:e] = np.sqrt(part)
    # Real SIFT sits on an integer lattice and has exact duplicate rows, so a
    # query can have r_1 = 0 and log(r_1 / r_k) = -inf, which would poison the
    # mean at m = 1. Drop those rows, the same rule R_LR applies.
    keep = (out[:, 0] > 0) & (out[:, K_NN - 1] > 0)
    r = out[keep]
    prof = np.log(r / r[:, [K_NN - 1]]).mean(axis=0)
    return {
        "m": list(range(1, K_NN + 1)),
        "mean_log_ratio": prof.tolist(),
        "rows_used": int(keep.sum()),
        "rows_dropped": int((~keep).sum()),
    }


def panels(path: str, rng: np.random.Generator) -> dict:
    x = l2(np.load(path).astype(np.float32))
    return {
        "path": path,
        "num_vectors": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "values": value_panel(x),
        "counts": count_panel(x),
        "profile": profile_panel(x, rng),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-path", required=True)
    ap.add_argument("--synthetic-path", action="append", default=[],
                    metavar="NAME=PATH", help="repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    out = {
        "conditions": {"n": N_ROWS, "k": K_NN, "seed": SEED, "preprocess": "l2"},
        "value_bins": {"lo": VALUE_LO, "hi": VALUE_HI, "bins": VALUE_BINS},
        "corpora": {},
    }
    # The real corpus is subsampled to N_ROWS for the profile panel but binned
    # whole for the value and count panels, where more rows only sharpen it.
    out["corpora"]["real"] = panels(args.real_path, rng)
    print(f"real: {out['corpora']['real']['num_vectors']} rows")
    for item in args.synthetic_path:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--synthetic-path needs NAME=PATH, got {item!r}")
        out["corpora"][name] = panels(path, rng)
        print(f"{name}: {out['corpora'][name]['num_vectors']} rows")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
