"""Neighbour positions of real queries: where each real-style query's exact
top-10 keys sit in its book (recency, early tokens, hubs), per (layer, KV head).

Records pair one-to-one with `kv_gate measure-real` records of the same books,
seed, n_sink and n_tail (same corpus seed). No faiss needed. Run from the repo root:
    PYTHONPATH=. python docs/results/scout-kv/controls/neighbours.py --capture-dir /workspace/kv/capture \
        --out /workspace/kv/controls/neighbours.jsonl
"""

import argparse
from pathlib import Path

from src.data.kv_capture import GLOBAL_LAYERS
from src.eval.kv_gate import parse_books
from src.kv.neighbours import measure_neighbours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--books", type=parse_books, default=parse_books("12-19"))
    ap.add_argument("--layers", type=parse_books, default=list(GLOBAL_LAYERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-sink", type=int, default=1)
    ap.add_argument("--n-tail", type=int, default=1024)
    a = ap.parse_args()
    measure_neighbours(
        a.capture_dir,
        a.books,
        a.layers,
        a.seed,
        a.out,
        n_sink=a.n_sink,
        n_tail=a.n_tail,
    )


if __name__ == "__main__":
    main()
