"""B6 diagnostic: the four intrinsic difficulty statistics of h, real vs v0, per layer.

Real: 20,000 rows of h from books 12-19 (sink dropped), subsampled inside
`ann_difficulty.compute`. v0: 20,000 samples from each layer's generator under EMA
weights, in original coordinates. Canonical settings: k 100, k_hub 10, nlist 256,
metric l2, seed 42.

The final checkpoint (step 30,000, EMA) is measured only where it differs from
best_generator.pt, which is where best_generator.pt is not step 30,000.

Run on the box from the repo root:
    python docs/results/scout-kv/v0/intrinsic.py --capture-dir /workspace/kv/capture \
        --runs runs/scout_kv/v0 --out /workspace/kv/v0/intrinsic.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.data.kv_capture import GLOBAL_LAYERS, load_kv_capture
from src.eval.ann_difficulty import compute, summary
from src.eval.kv_gate import synthetic_h

N_ROWS = 20000
REAL_BOOKS = list(range(12, 20))


def measure(x, layer, source, out):
    t = time.time()
    rec = {"layer": layer, "source": source, "n_rows": int(x.shape[0])}
    rec |= summary(
        compute(x, k=100, k_hub=10, nlist=256, max_rows=N_ROWS, seed=42, metric="l2")
    )
    rec["seconds"] = round(time.time() - t, 1)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists; refusing to append duplicate records")
    for layer in GLOBAL_LAYERS:
        run_dir = a.runs / f"L{layer:02d}"
        measure(
            load_kv_capture(a.capture_dir, layer, REAL_BOOKS, n_sink=1),
            layer,
            "real",
            a.out,
        )
        best_step = torch.load(
            run_dir / "best_generator.pt", map_location="cpu", weights_only=False
        )["step"]
        names = ["best_generator.pt"] + (
            ["checkpoint_step_30000.pt"] if best_step != 30000 else []
        )
        for name in names:
            h, _, _, weights = synthetic_h(
                run_dir,
                layer,
                seed=0,
                n_keys=N_ROWS,
                n_tail=0,
                checkpoint_name=name,
                ema=True,
            )
            assert weights == "ema", weights
            source = "v0_best" if name == "best_generator.pt" else "v0_final"
            measure(h.cpu().numpy(), layer, source, a.out)


if __name__ == "__main__":
    main()
