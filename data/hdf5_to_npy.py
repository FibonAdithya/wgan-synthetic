#!/usr/bin/env python3
"""Convert an ANN-Benchmarks sift-128-euclidean.hdf5 into the .npy contract.

The `train` dataset in that file is the SIFT1M base set (1,000,000 x 128,
float32) -- the same vectors as sift_base.fvecs, just a faster mirror. Output
matches the contract in data/README.md: shape [N, 128], dtype float32.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to .hdf5")
    parser.add_argument("--output", type=Path, required=True, help="Path to .npy")
    parser.add_argument("--key", type=str, default="train", help="HDF5 dataset key.")
    parser.add_argument(
        "--num-vectors",
        type=int,
        default=0,
        help="Take only the first N vectors (0 = all).",
    )
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        if args.key not in f:
            raise KeyError(f"Key {args.key!r} not in {args.input}; found {list(f.keys())}")
        dset = f[args.key]
        vectors = dset[: args.num_vectors] if args.num_vectors > 0 else dset[:]

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape {vectors.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, vectors)
    print(
        f"Saved {vectors.shape[0]} vectors (dim={vectors.shape[1]}, "
        f"{vectors.nbytes / 1e6:.1f} MB) to {args.output}"
    )


if __name__ == "__main__":
    main()
