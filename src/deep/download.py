"""Fetch the DEEP image descriptor set and cut reproducible subsets from it.

Writes .npy deliberately: the existing loader in src/data/dataset.py
reads .npy and .fvecs, so emitting .npy means neither the loader nor the
trainer needs to learn about HDF5.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from urllib.request import urlopen

import h5py
import numpy as np

DEEP_URL = "http://ann-benchmarks.com/deep-image-96-angular.hdf5"
DESCRIPTOR_DIM = 96


def fetch(
    url: str,
    dest: Path,
    *,
    chunk_bytes: int = 1 << 20,
    poll_seconds: float = 5.0,
    timeout_seconds: float = 3600.0,
) -> Path:
    """Download `url` to `dest` atomically and single-flight. Returns `dest`.

    Two properties matter because this cache is shared with other agents on the
    box:

    Atomic -- the body is written to a sibling .part file and os.replace()d
    into position, so a concurrent reader sees either no file or a complete
    one, never a truncated download.

    Single-flight -- the .part file doubles as a lock, created O_EXCL. A second
    caller arriving mid-download waits for the first to finish rather than
    starting its own 4GB fetch. The timeout bounds the wait, so a crashed
    downloader that left a stale .part behind surfaces as an error instead of
    hanging forever; clear it by deleting the .part file.

    An existing destination is left alone -- the file is large and immutable.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        waited = 0.0
        while not dest.exists():
            if waited >= timeout_seconds:
                raise TimeoutError(
                    f"Timed out after {waited:.0f}s waiting for an in-flight "
                    f"download of {dest}. If no other process is fetching it, "
                    f"delete {tmp} and retry."
                )
            time.sleep(poll_seconds)
            waited += poll_seconds
        return dest

    try:
        with os.fdopen(fd, "wb") as handle, urlopen(url) as response:
            while True:
                chunk = response.read(chunk_bytes)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def subset(
    hdf5_path: Path, out_path: Path, *, num_rows: int, seed: int = 42
) -> Path:
    """Write a random `num_rows`-row sample of the train split to `out_path`.

    Rows are drawn without replacement and returned in sorted index order,
    which h5py requires for fancy indexing and which also makes the read
    sequential rather than random over a 4GB file.
    """
    hdf5_path = Path(hdf5_path)
    out_path = Path(out_path)
    with h5py.File(hdf5_path, "r") as f:
        train = f["train"]
        total = train.shape[0]
        take = min(num_rows, total)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total, size=take, replace=False))
        rows = train[idx, :]
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, rows)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", type=str, default=DEEP_URL)
    parser.add_argument(
        "--cache-path",
        type=str,
        default="/workspace/data-cache/deep-image-96-angular.hdf5",
        help="Where the shared read-only HDF5 lives. Downloaded once.",
    )
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[250_000, 1_000_000],
        help="Subset sizes to write, as data/deep96_<n>.npy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = fetch(args.url, Path(args.cache_path))
    print(f"hdf5: {cache}")
    for rows in args.rows:
        label = f"{rows // 1000}k" if rows < 1_000_000 else f"{rows // 1_000_000}m"
        out = subset(
            cache,
            Path(args.out_dir) / f"deep96_{label}.npy",
            num_rows=rows,
            seed=args.seed,
        )
        print(f"subset: {out} {np.load(out, mmap_mode='r').shape}")


if __name__ == "__main__":
    main()
