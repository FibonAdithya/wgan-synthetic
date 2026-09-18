"""Real-side noise floor for SIFT: ten disjoint 20,000-row draws of sift_1m.

The GloVe and NYTimes gates are bands around the mean of several draws of the
real corpus. The scripts that produced those draws are not in the tree; this
one is, so SIFT's can be reproduced from a pinned commit.

Draws are disjoint slices of one seed-42 permutation, L2-normalised row-wise
(`eda_report --preprocess l2`, the SIFT default), measured under the locked
canonical conditions in gates/sift.yaml. The output keys match
docs/datasets/nytimes_noise_floor.json.

    python scripts/sift_real_noise_floor.py \
        --real-path /workspace/data-cache/sift_1m.npy \
        --out docs/datasets/sift_noise_floor.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.eval import ann_difficulty
from src.eval.eda.series import maybe_l2_normalize
from src.eval.noise_floor import GATE_STATISTICS, summarize_spread

N, K, K_HUB, NLIST, SEED = 20000, 100, 10, 256, 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def draws(x: np.ndarray, count: int, n: int, seed: int) -> list[np.ndarray]:
    """`count` disjoint `n`-row slices of one permutation of `x`."""
    if count * n > x.shape[0]:
        raise ValueError(
            f"{count} disjoint {n}-row draws need {count * n} rows, have {x.shape[0]}"
        )
    order = np.random.default_rng(seed).permutation(x.shape[0])
    return [x[np.sort(order[i * n : (i + 1) * n])] for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10)
    args = parser.parse_args()

    x = np.load(args.real_path, mmap_mode="r")
    per_seed = []
    for i, d in enumerate(draws(x, args.draws, N, SEED)):
        d = maybe_l2_normalize(np.asarray(d, dtype=np.float32), "l2")
        s = ann_difficulty.summary(
            ann_difficulty.compute(
                d, k=K, k_hub=K_HUB, nlist=NLIST, max_rows=0, seed=SEED
            )
        )
        per_seed.append(
            {
                **{name: s[name] for name in GATE_STATISTICS},
                "lid_discarded_queries": s["lid_discarded_queries"],
            }
        )
        print(f"draw {i}: {per_seed[-1]}", flush=True)

    result = {
        "n": N,
        "k": K,
        "k_hub": K_HUB,
        "nlist": NLIST,
        "metric": "l2",
        "preprocess": "l2",
        "seeds": args.draws,
        "draws": f"disjoint {N}-row draws of one seed-{SEED} permutation",
        "source": str(args.real_path),
        "source_sha256": sha256(args.real_path),
        "rows_available": int(x.shape[0]),
        "per_seed": per_seed,
        "spread": {
            name: summarize_spread([p[name] for p in per_seed])
            for name in GATE_STATISTICS
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["spread"], indent=1))


if __name__ == "__main__":
    main()
