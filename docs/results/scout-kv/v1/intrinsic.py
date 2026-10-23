"""B6 diagnostic: intrinsic difficulty statistics of h per flag, real vs v1, per layer.

The v0 version (docs/results/scout-kv/v0/intrinsic.py) pools every position.
v1 is conditioned on a body/tail flag, so each flag is measured on its own:

- real body: books 12-19, positions 1-31,743, subsampled to 20,000 rows inside
  `ann_difficulty.compute`;
- real tail: books 12-19, the last 1,024 positions of each, 8,192 rows;
- v1 body: 20,000 samples under the body flag;
- v1 tail: 8,192 samples under the tail flag (the h_query and h_id halves of
  `synthetic_h` with n_tail 4,096).

Samples are taken under EMA weights, in original coordinates. Canonical
settings as v0: k 100, k_hub 10, nlist 256, metric l2, seed 42. The final
checkpoint is measured only where it differs from best_generator.pt.

Run on the box from the repo root:
    python docs/results/scout-kv/v1/intrinsic.py --capture-dir /workspace/kv/capture \
        --runs runs/scout_kv/v1 --layers 19,43 --out /workspace/kv/v1/intrinsic_L19_43.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.data.kv_capture import GLOBAL_LAYERS, load_kv_capture
from src.eval.ann_difficulty import compute, summary
from src.eval.kv_gate import synthetic_h
from src.models.conditional import BODY, TAIL

N_BODY = 20000
N_TAIL_ROWS = 8192
N_TAIL = 1024
REAL_BOOKS = list(range(12, 20))


def measure(x, layer, source, flag, out):
    t = time.time()
    rec = {"layer": layer, "source": source, "flag": flag, "n_rows": int(x.shape[0])}
    rec |= summary(
        compute(x, k=100, k_hub=10, nlist=256, max_rows=N_BODY, seed=42, metric="l2")
    )
    rec["seconds"] = round(time.time() - t, 1)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--layers", default=",".join(str(x) for x in GLOBAL_LAYERS))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists; refusing to append duplicate records")
    layers = [int(x) for x in a.layers.split(",")]
    for layer in layers:
        if layer not in GLOBAL_LAYERS:
            raise ValueError(f"{layer} is not a global layer {GLOBAL_LAYERS}")
        x, flags = load_kv_capture(
            a.capture_dir, layer, REAL_BOOKS, n_sink=1, n_tail=N_TAIL
        )
        measure(x[flags == BODY], layer, "real", "body", a.out)
        measure(x[flags == TAIL], layer, "real", "tail", a.out)
        del x, flags
        run_dir = a.runs / f"L{layer:02d}"
        best_step = torch.load(
            run_dir / "best_generator.pt", map_location="cpu", weights_only=False
        )["step"]
        names = ["best_generator.pt"] + (
            ["checkpoint_step_30000.pt"] if best_step != 30000 else []
        )
        for name in names:
            kind = "v1_best" if name == "best_generator.pt" else "v1_final"
            h_body, _, _, w = synthetic_h(
                run_dir,
                layer,
                seed=0,
                n_keys=N_BODY,
                n_tail=0,
                checkpoint_name=name,
                ema=True,
            )
            assert w == "ema", w
            measure(h_body.cpu().numpy(), layer, kind, "body", a.out)
            _, h_q, h_id, w = synthetic_h(
                run_dir,
                layer,
                seed=0,
                n_keys=1,
                n_tail=N_TAIL_ROWS // 2,
                checkpoint_name=name,
                ema=True,
            )
            assert w == "ema", w
            h_tail = torch.cat([h_q, h_id]).cpu().numpy()
            measure(h_tail, layer, kind, "tail", a.out)


if __name__ == "__main__":
    main()
