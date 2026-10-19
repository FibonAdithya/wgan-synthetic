"""B7 acceptance check: the neighbour run rebuilt the corpora that reference/seed0 profiled.

For every (layer, kv_head), ``ood.own_book_share`` in neighbours.jsonl must be within
TOL of ``own_book_share_top10`` in docs/results/scout-kv/reference/seed0.jsonl. One
flipped top-10 slot moves the share by 1/51,200; TOL allows for float32 ties flipping
when the projection runs on a different GPU. Exits 1 on any mismatch or missing corpus.
"""

import argparse
import json
import sys
from pathlib import Path

TOL = 1e-3
REPO = Path(__file__).resolve().parents[4]


def _load(path):
    return {
        (r["layer"], r["kv_head"]): r
        for r in map(json.loads, Path(path).read_text().splitlines())
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--neighbours", required=True, type=Path)
    p.add_argument(
        "--reference",
        type=Path,
        default=REPO / "docs/results/scout-kv/reference/seed0.jsonl",
    )
    a = p.parse_args(argv)
    nb, ref = _load(a.neighbours), _load(a.reference)
    bad = sorted(set(ref) ^ set(nb))
    worst = 0.0
    for key in sorted(set(ref) & set(nb)):
        d = abs(nb[key]["ood"]["own_book_share"] - ref[key]["own_book_share_top10"])
        worst = max(worst, d)
        if d > TOL:
            bad.append(key)
    print(
        json.dumps(
            {"n_compared": len(set(ref) & set(nb)), "max_abs_diff": worst, "bad": bad}
        )
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
