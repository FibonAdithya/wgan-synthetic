"""Where real-split queries' top-10 keys sit in v2's generated books.

Each v2 run wrote books 12-19 in capture format to <v2-dir>/L{LL}_s{S}/capture, with
`weights` linked to the real capture's. `measure_neighbours` builds the gate's real
split on them at seed 0, the seed of `controls/neighbours.jsonl`, which holds the real
values. One file per run; a finished file is skipped, so a rerun resumes.

Run from the repo root:
    PYTHONPATH=. python docs/results/scout-kv/v3-surgery/v2_neighbours.py \
        --v2-dir /workspace/kv/v2 --out-dir /workspace/kv/v3surgery/v2_neighbours
"""

import argparse
import time
from pathlib import Path

from src.data.kv_capture import GLOBAL_LAYERS
from src.kv.neighbours import measure_neighbours

SEEDS = (42, 1)
BOOKS = range(12, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", type=Path, default=Path("/workspace/kv/v2"))
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        for layer in GLOBAL_LAYERS:
            out = a.out_dir / f"L{layer:02d}_s{s}.jsonl"
            if out.exists():
                continue
            tmp = out.with_suffix(".tmp")
            tmp.unlink(missing_ok=True)
            t0 = time.time()
            measure_neighbours(
                a.v2_dir / f"L{layer:02d}_s{s}" / "capture",
                list(BOOKS),
                [layer],
                0,
                tmp,
                source=f"v2_s{s}",
            )
            tmp.rename(out)
            print(f"L{layer} s{s}: {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
