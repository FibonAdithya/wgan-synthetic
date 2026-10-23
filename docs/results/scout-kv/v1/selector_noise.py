"""B1: noise floor of the v1 checkpoint selector, real against real.

The v1 trainer scores each checkpoint as selection_score(body) +
selection_score(tail), each half comparing a fake sample with its own real
holdout class (1,024 tail rows, 7,618 body rows; configs/scout_kv/v1/). This
script asks how large each half is when both sides are real: five seeded
pairs of disjoint real subsets of the holdout sizes, drawn from books 0-11.
A checkpoint's half-score below this floor cannot be told apart from real by
the selector.

The statistics are measured on raw h. The trainer measures centered h, but
LID and relative contrast are computed from L2 distances, which centering
does not change.

Usage (on the box):
    python docs/results/scout-kv/v1/selector_noise.py --capture-dir /workspace/kv/capture \
        --out /workspace/kv/v1/selector_noise.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data.kv_capture import load_kv_capture
from src.models.conditional import BODY, TAIL
from src.train.selection import gate_statistics, selection_score

LAYERS = (19, 43)
BOOKS = tuple(range(12))
N_TAIL = 1024
SIZES = {"tail": 1024, "body": 7618}  # kv_tail_holdout; round(0.02 * 380,916)
N_PAIRS = 5
GATE_SEED = 42  # the configs' seed, which the trainer passes to gate_statistics


def pair_scores(x: np.ndarray, size: int, n_pairs: int, layer: int) -> list[dict]:
    out = []
    for p in range(n_pairs):
        rng = np.random.default_rng(1000 * layer + p)
        idx = rng.permutation(x.shape[0])
        if 2 * size > idx.size:
            raise ValueError(
                f"need {2 * size} rows for a disjoint pair, have {idx.size}"
            )
        a, b = x[idx[:size]], x[idx[size : 2 * size]]
        sa = gate_statistics(a, metric="l2", seed=GATE_SEED)
        sb = gate_statistics(b, metric="l2", seed=GATE_SEED)
        out.append(
            {
                "pair": p,
                "score": selection_score(sa, sb),
                "lid_median": [sa["lid_median"], sb["lid_median"]],
                "relative_contrast_median": [
                    sa["relative_contrast_median"],
                    sb["relative_contrast_median"],
                ],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    result = {"books": list(BOOKS), "n_pairs": N_PAIRS, "sizes": SIZES, "layers": {}}
    for layer in LAYERS:
        t0 = time.time()
        x, flags = load_kv_capture(a.capture_dir, layer, BOOKS, n_sink=1, n_tail=N_TAIL)
        per = {}
        for name, flag in (("tail", TAIL), ("body", BODY)):
            pairs = pair_scores(x[flags == flag], SIZES[name], N_PAIRS, layer)
            scores = [q["score"] for q in pairs]
            per[name] = {
                "pairs": pairs,
                "min": min(scores),
                "median": float(np.median(scores)),
                "max": max(scores),
            }
            print(layer, name, [round(s, 4) for s in scores], flush=True)
        per["seconds"] = round(time.time() - t0, 1)
        result["layers"][str(layer)] = per
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("DONE", a.out)


if __name__ == "__main__":
    main()
