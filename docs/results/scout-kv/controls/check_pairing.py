"""B7 acceptance check: the neighbour run rebuilt the corpora that reference/seed0 profiled.

For every (layer, kv_head), ``ood.own_book_share`` in neighbours.jsonl (``--neighbours``),
or ``own_book_share_top10`` in a records file (``--records``, the held-out plan's R6a),
must be within TOL of ``own_book_share_top10`` in docs/results/scout-kv/reference/seed0.jsonl.
``--layers`` keeps only those layers on both sides; a corpus missing inside them still fails. One
flipped top-10 slot moves the share by 1/51,200; TOL allows for float32 ties flipping
when the projection runs on a different GPU. Also compares seed, books, n_sink, n_tail and n_keys. Exits 1 on any missing,
mismatched or meta-mismatched corpus.
"""

import argparse
import json
import sys
from pathlib import Path

from src.eval.kv_gate import parse_books

TOL = 1e-3
META = ("seed", "books", "n_sink", "n_tail", "n_keys")
REPO = Path(__file__).resolve().parents[4]


def _load(path):
    return {
        (r["layer"], r["kv_head"]): r
        for r in map(json.loads, filter(str.strip, Path(path).read_text().splitlines()))
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--neighbours", type=Path)
    src.add_argument("--records", type=Path)
    p.add_argument("--layers", type=parse_books, default=None)
    p.add_argument(
        "--reference",
        type=Path,
        default=REPO / "docs/results/scout-kv/reference/seed0.jsonl",
    )
    a = p.parse_args(argv)
    nb, ref = _load(a.neighbours or a.records), _load(a.reference)
    if a.layers is not None:
        keep = set(a.layers)
        nb = {k: v for k, v in nb.items() if k[0] in keep}
        ref = {k: v for k, v in ref.items() if k[0] in keep}

    def share(r):
        return r["ood"]["own_book_share"] if a.neighbours else r["own_book_share_top10"]

    missing = sorted(set(ref) ^ set(nb))
    both = sorted(set(ref) & set(nb))
    diffs = {k: abs(share(nb[k]) - ref[k]["own_book_share_top10"]) for k in both}
    mismatched = [k for k in both if diffs[k] > TOL]
    meta_mismatch = [
        k for k in both if any(nb[k].get(m) != ref[k].get(m) for m in META)
    ]
    print(
        json.dumps(
            {
                "n_compared": len(both),
                "n_exact": sum(d == 0.0 for d in diffs.values()),
                "max_abs_diff": max(diffs.values(), default=0.0),
                "missing": missing,
                "mismatched": mismatched,
                "meta_mismatch": meta_mismatch,
            }
        )
    )
    bad = missing or mismatched or meta_mismatch
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
