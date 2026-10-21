"""Held-out-book tail control (docs/ai/plans/2026-10-21-scout-kv-held-out.md):
`rebuild` (the reference corpus) and `held_out_tail` (the same keys, queries from
the tails of books outside the key set), on the rule layers.

Run on the box from the repo root:
    PYTHONPATH=. python docs/results/scout-kv/controls/heldout.py --capture-dir /workspace/kv/capture \
        --seed 5 --query-books 0-7 --out-dir /workspace/kv/controls/heldout/s5
"""

import argparse
from pathlib import Path

from src.eval.kv_gate import parse_books
from src.kv.controls import HELD_OUT_CONTROLS, measure_held_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--layers", type=parse_books, default=[11, 19, 43])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--query-books", type=parse_books, required=True)
    ap.add_argument("--controls", default=",".join(HELD_OUT_CONTROLS))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=15)
    a = ap.parse_args()
    measure_held_out(
        a.capture_dir,
        a.layers,
        a.seed,
        a.out_dir,
        query_books=a.query_books,
        controls=a.controls.split(","),
        threads=a.threads,
    )


if __name__ == "__main__":
    main()
