"""Control for locality.py's lag curve: is consecutive similarity a book effect
or a position effect shared by every book?

The v2 probe's position-only arm (C, tokens independent given position) has a
lag curve as high as real h, so position alone can produce it. This measures,
on real h of books 12-19 (body positions 1-31,743, centred with the pooled
mean, as locality.py): the median cosine of (book b, t) with (book b, t+lag)
and with (book b', t+lag) for b' != b, over 50,000 random (b, b', t) per lag.

    python docs/results/scout-kv/v2-probe/cross_book_lag.py --capture-dir /workspace/kv/capture \
        --out /workspace/kv/v2probe/cross_book_lag.json [--layers 19,27]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.kv_capture import GLOBAL_LAYERS, load_kv_capture
from src.models.conditional import BODY

BOOKS = list(range(12, 20))
LAGS = (0, 1, 16, 256, 1024, 4096)
N = 50000
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--layers", default=",".join(str(x) for x in GLOBAL_LAYERS))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    res = {"books": BOOKS, "n_pairs": N, "seed": SEED, "layers": []}
    for layer in (int(v) for v in a.layers.split(",")):
        x, flags = load_kv_capture(a.capture_dir, layer, BOOKS, n_sink=1, n_tail=1024)
        hb = x[flags == BODY]
        del x
        per = hb.shape[0] // len(BOOKS)
        hb = hb - hb.mean(0, keepdims=True)
        hb /= np.linalg.norm(hb, axis=1, keepdims=True)
        hb = hb.reshape(len(BOOKS), per, -1)
        rng = np.random.default_rng(SEED)
        rec = {"layer": layer, "same_book": {}, "cross_book": {}}
        for lag in LAGS:
            b = rng.integers(0, len(BOOKS), N)
            b2 = (b + rng.integers(1, len(BOOKS), N)) % len(BOOKS)
            t = rng.integers(0, per - lag, N)
            u = hb[b, t]
            rec["same_book"][str(lag)] = float(np.median((u * hb[b, t + lag]).sum(1))) if lag else 1.0
            rec["cross_book"][str(lag)] = float(np.median((u * hb[b2, t + lag]).sum(1)))
        res["layers"].append(rec)
        print(layer, "same", [round(rec["same_book"][str(k)], 3) for k in LAGS],
              "cross", [round(rec["cross_book"][str(k)], 3) for k in LAGS], flush=True)
        del hb
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
