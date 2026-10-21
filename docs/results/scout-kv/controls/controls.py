"""Control sets of h for the v0 diagnosis: pooled_real, norm_equalised, gaussian.

Each control is profiled through the gate machinery under the locked conditions.
Run on the box from the repo root:
    PYTHONPATH=. python docs/results/scout-kv/controls/controls.py --capture-dir /workspace/kv/capture \
        --out-dir /workspace/kv/controls
"""

import argparse
from pathlib import Path

from src.eval.kv_gate import parse_books
from src.kv.controls import CONTROLS, measure_controls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--layers", type=parse_books, default=[11, 19, 43])
    ap.add_argument("--controls", default=",".join(CONTROLS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=15)
    a = ap.parse_args()
    measure_controls(
        a.capture_dir,
        a.layers,
        a.controls.split(","),
        a.seed,
        a.out_dir,
        threads=a.threads,
    )


if __name__ == "__main__":
    main()
