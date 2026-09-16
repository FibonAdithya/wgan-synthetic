"""Rebuild the cleaned NYTimes corpus that every `nytimes` job measures against.

The GPU box was rebuilt and `/workspace/data-cache` was wiped along with it,
so `nytimes_250k_l2_clean.npy` -- the file every `nytimes_v*_seed42_job.sh`
script reads as `$CLEAN` -- no longer exists there, and no script in this
repo produced it. This one does.

The definition of "cleaned" comes from `docs/datasets/nytimes.md` (search
for "cleaned"): drop exact-zero rows, drop exact duplicate rows, then
L2-normalise the survivors. `docs/datasets/nytimes_row_filters.json`'s
`drop_zero_and_duplicates` entry records the row count that definition has
to reproduce -- 217,854 -- which is also what the previous artifact's file
size implies (223,082,624 bytes = 217,854 * 256 * 4 + 128 header).

    python scripts/make_nytimes_clean.py \\
        --input data/nytimes_250k.npy --output data/nytimes_250k_l2_clean.npy
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CleanStats:
    """Row counts from one `clean_nytimes_rows` call, for reporting."""

    rows_in: int
    zero_dropped: int
    duplicates_dropped: int
    rows_out: int


def clean_nytimes_rows(x: np.ndarray) -> tuple[np.ndarray, CleanStats]:
    """Drop exact-zero rows and exact duplicate rows, then L2-normalise.

    Order matters and follows `docs/datasets/nytimes.md`: zero rows are
    dropped first, so they cannot masquerade as duplicates of each other;
    duplicates are then found on the raw, pre-normalisation rows, because
    upstream norms spread 0.90-1.10 and normalising first would erase real
    differences before they can be checked; the first occurrence of each
    duplicate group is kept, in input order, so the result is deterministic.
    Normalisation is the last step, applied only to the survivors.
    """
    rows_in = x.shape[0]

    nonzero_mask = np.any(x != 0, axis=1)
    zero_dropped = rows_in - int(nonzero_mask.sum())
    survivors = x[nonzero_mask]

    _, first_occurrence = np.unique(survivors, axis=0, return_index=True)
    keep_index = np.sort(first_occurrence)
    duplicates_dropped = survivors.shape[0] - keep_index.shape[0]
    deduped = survivors[keep_index]

    norm = np.linalg.norm(deduped, axis=1, keepdims=True)
    normalized = (deduped / np.clip(norm, 1.0e-8, None)).astype(np.float32, copy=False)

    stats = CleanStats(
        rows_in=rows_in,
        zero_dropped=zero_dropped,
        duplicates_dropped=duplicates_dropped,
        rows_out=normalized.shape[0],
    )
    return normalized, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/nytimes_250k.npy"),
        help="Raw NYTimes descriptor .npy to clean.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/nytimes_250k_l2_clean.npy"),
        help="Where to write the cleaned, L2-normalised float32 .npy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x = np.load(args.input)
    cleaned, stats = clean_nytimes_rows(x)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, cleaned)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()

    print(f"rows in:          {stats.rows_in}")
    print(f"zero rows dropped: {stats.zero_dropped}")
    print(f"duplicates dropped: {stats.duplicates_dropped}")
    print(f"rows out:         {stats.rows_out}")
    print(f"output sha256:    {digest}")


if __name__ == "__main__":
    main()
