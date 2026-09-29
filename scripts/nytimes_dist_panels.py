"""Binned distribution data for the NYTimes figures in the write-up.

The SIFT figure (scripts/sift_dist_panels.py) shows coordinate values, because
the SIFT base WGAN fails on support. On NYTimes the support is right by
construction -- every generator ends on the unit sphere, like the corpus -- and
the base WGAN fails on rank and local dimension instead. So these panels show
the distributions the four gate statistics are summaries of, plus the
covariance spectrum that explains them:

  spectrum    eigenvalues of the covariance of the measured rows, sorted
              descending and divided by their sum. Effective rank is
              exp(entropy) of this vector.
  lid         histogram of the per-row LID estimates whose median is the
              gate's LID.
  contrast    histogram of per-row relative contrast (mean distance to a fixed
              target sample / nearest-neighbour distance), whose median is the
              gate's relative contrast. Log-spaced bins.
  koccurrence fraction of rows with each 10-NN k-occurrence count, whose
              skewness is the gate's hubness.
  cells       IVF cell sizes sorted descending, whose Gini is the gate's IVF
              Gini.

Every per-row array comes from ONE call to `ann_difficulty.compute` at the
gate's locked conditions (20,000 rows, k 100, hub k 10, nlist 256, seed 42,
angular), so each panel is exactly the distribution its gate scalar
summarises. `ann_difficulty.summary` of the same call is written alongside as
a cross-check against the eda_report run in the same job.

Bin edges are fixed in this file, not derived from the data, so every corpus is
binned identically and the curves can be overlaid.

Usage:

    python scripts/nytimes_dist_panels.py \
        --real-path /workspace/data-cache/nytimes_250k_l2_clean.npy \
        --synthetic-path v0=<...>.npy --synthetic-path v3=<...>.npy \
        --out runs/nytimes/panels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval import ann_difficulty  # noqa: E402
from src.eval.eda.series import maybe_l2_normalize, subsample  # noqa: E402

# eda_report's default --max-vectors: every set is cut to this before measuring.
MAX_VECTORS = 50000
# The gate's locked conditions (gates/nytimes.yaml `canonical`).
N_ROWS = 20000
K_NN = 100
K_HUB = 10
NLIST = 256
SEED = 42
METRIC = "angular"

# Fixed so every corpus is binned identically. Real cleaned NYTimes has median
# LID 56 and the collapsed runs reach about 5; contrast runs from about 1.1 on
# real to above 15 on a collapsed sheet.
LID_LO, LID_HI, LID_BINS = 0.0, 120.0, 120
CONTRAST_LO, CONTRAST_HI, CONTRAST_BINS = 1.0, 20.0, 120
KOCC_MAX = 200  # counts above this are pooled into the last bin


def spectrum_panel(x: np.ndarray) -> dict:
    """Normalised covariance eigenvalues, descending, and the effective rank."""
    xc = x.astype(np.float64) - x.mean(axis=0, dtype=np.float64)
    cov = xc.T @ xc / (x.shape[0] - 1)
    ev = np.clip(np.linalg.eigvalsh(cov)[::-1], 0.0, None)
    p = ev / ev.sum()
    nz = p[p > 0]
    return {
        "normalised_eigenvalues": p.tolist(),
        "effective_rank": float(np.exp(-(nz * np.log(nz)).sum())),
    }


def hist_panel(values: np.ndarray, edges: np.ndarray) -> dict:
    """Density over the given edges; the mass outside them is reported, not dropped silently."""
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    n = values.size
    return {
        "edges": edges.tolist(),
        "density": (counts / (n * widths)).tolist(),
        "below": int((values < edges[0]).sum()),
        "above": int((values >= edges[-1]).sum()),
        "n": int(n),
        "median": float(np.median(values)),
    }


def koccurrence_panel(counts: np.ndarray) -> dict:
    pooled = np.minimum(counts, KOCC_MAX)
    freq = np.bincount(pooled, minlength=KOCC_MAX + 1)
    return {
        "support": list(range(KOCC_MAX + 1)),
        "fraction": (freq / freq.sum()).tolist(),
        "max": int(counts.max()),
        "pooled_above": int((counts > KOCC_MAX).sum()),
    }


def panels(path: str) -> dict:
    # Exactly eda_report's load path: cut to MAX_VECTORS, then normalise. The
    # cut is a no-op on the 50,000-row sample files but not on the 217,854-row
    # real corpus, and skipping it would measure a different real draw from the
    # one the report in the same job measures.
    x = maybe_l2_normalize(subsample(np.load(path), MAX_VECTORS, SEED), "l2")
    m = ann_difficulty.compute(
        x, k=K_NN, k_hub=K_HUB, nlist=NLIST, max_rows=N_ROWS, seed=SEED, metric=METRIC
    )
    # The same rows compute() measured, for the spectrum.
    sub = ann_difficulty._subsample(x, N_ROWS, SEED)
    return {
        "path": path,
        "num_vectors": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "summary": ann_difficulty.summary(m),
        "spectrum": spectrum_panel(sub),
        "lid": hist_panel(m.lid, np.linspace(LID_LO, LID_HI, LID_BINS + 1)),
        "contrast": hist_panel(
            m.relative_contrast,
            np.geomspace(CONTRAST_LO, CONTRAST_HI, CONTRAST_BINS + 1),
        ),
        "koccurrence": koccurrence_panel(m.k_occurrence),
        "cells": {
            "sizes_descending": m.cell_occupancy[::-1].tolist(),
            "nlist": m.nlist,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-path", required=True)
    ap.add_argument(
        "--synthetic-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="repeatable",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {
        "conditions": {
            "n": N_ROWS,
            "k": K_NN,
            "k_hub": K_HUB,
            "nlist": NLIST,
            "seed": SEED,
            "metric": METRIC,
            "preprocess": "l2",
        },
        "corpora": {},
    }
    out["corpora"]["real"] = panels(args.real_path)
    print(f"real: {out['corpora']['real']['summary']}")
    for item in args.synthetic_path:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--synthetic-path needs NAME=PATH, got {item!r}")
        out["corpora"][name] = panels(path)
        print(f"{name}: {out['corpora'][name]['summary']}")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
