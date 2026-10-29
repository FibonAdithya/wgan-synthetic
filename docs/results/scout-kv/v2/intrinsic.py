"""LID, contrast and hubness per flag of a capture-format directory (books 12-19).

Copies `intrinsic` in docs/results/scout-kv/v2-probe/probe.py setting for setting, so a v2 run
is measured as the probe's runs were. One JSON line per flag (body, tail).

Run from the repo root:
    python docs/results/scout-kv/v2/intrinsic.py --capture-dir R/capture --layer 19 \
        --source v2_s42 --out R/intrinsic.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from src.data.kv_capture import load_kv_capture
from src.eval.ann_difficulty import compute, summary
from src.models.conditional import BODY, TAIL

OUT_BOOKS = range(12, 20)
N_TAIL = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    x, flags = load_kv_capture(a.capture_dir, a.layer, OUT_BOOKS, n_sink=1, n_tail=N_TAIL)
    for name, f in (("body", BODY), ("tail", TAIL)):
        t = time.time()
        rec = {
            "layer": a.layer,
            "source": a.source,
            "flag": name,
            "n_rows": int((flags == f).sum()),
        }
        rec |= summary(
            compute(x[flags == f], k=100, k_hub=10, nlist=256, max_rows=20000, seed=42, metric="l2")
        )
        rec["seconds"] = round(time.time() - t, 1)
        with a.out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
