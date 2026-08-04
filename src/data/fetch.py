"""Fetch ANN benchmark descriptor sets and cut reproducible subsets from them.

Writes .npy deliberately: the loader in src/data/dataset.py reads .npy and
.fvecs, so emitting .npy means neither the loader nor the trainer needs to
learn about HDF5.

All six families are taken from the ann-benchmarks HDF5 mirrors so this module
handles one container format. Sets obtained by other routes -- corpus-texmex
.fvecs, say -- are read directly by load_descriptors and do not come through
here.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
from urllib.request import urlopen

import h5py
import numpy as np

BASE_URL = "http://ann-benchmarks.com"


@dataclass(frozen=True)
class Source:
    """One benchmark family: where it comes from and what shape it has.

    `metric` is the distance the real corpus is searched under, and is the
    value that lands in a dataset config's `data.metric`. It is not a
    preprocessing instruction -- l2_normalize is set independently.
    """

    name: str
    url: str
    dim: int
    metric: str
    hdf5_key: str = "train"
    default_rows: Tuple[int, ...] = (250_000, 1_000_000)


def _ann_benchmarks(name: str, slug: str, dim: int, metric: str) -> Source:
    return Source(name=name, url=f"{BASE_URL}/{slug}.hdf5", dim=dim, metric=metric)


SOURCES: Dict[str, Source] = {
    "sift": _ann_benchmarks("sift", "sift-128-euclidean", 128, "l2"),
    "gist": _ann_benchmarks("gist", "gist-960-euclidean", 960, "l2"),
    "deep": _ann_benchmarks("deep", "deep-image-96-angular", 96, "angular"),
    "glove": _ann_benchmarks("glove", "glove-100-angular", 100, "angular"),
    "nytimes": _ann_benchmarks("nytimes", "nytimes-256-angular", 256, "angular"),
    "openai": _ann_benchmarks(
        "openai", "dbpedia-openai-1000k-angular", 1536, "angular"
    ),
}


def subset_name(dataset: str, rows: int) -> str:
    """Stable on-disk stem, e.g. sift_250k or deep_1m."""
    if rows >= 1_000_000:
        label = f"{rows // 1_000_000}m"
    else:
        label = f"{rows // 1000}k"
    return f"{dataset}_{label}"


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
    hdf5_path: Path,
    out_path: Path,
    *,
    num_rows: int,
    seed: int = 42,
    key: str = "train",
) -> Path:
    """Write a random `num_rows`-row sample of the `key` split to `out_path`.

    Rows are drawn without replacement and returned in sorted index order,
    which h5py requires for fancy indexing and which also makes the read
    sequential rather than random over a multi-gigabyte file.
    """
    hdf5_path = Path(hdf5_path)
    out_path = Path(out_path)
    with h5py.File(hdf5_path, "r") as f:
        split = f[key]
        total = split.shape[0]
        take = min(num_rows, total)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total, size=take, replace=False))
        rows = split[idx, :]
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, rows)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=sorted(SOURCES),
        help="Which benchmark family to fetch.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/workspace/data-cache",
        help="Where the shared read-only HDF5 files live. Each is downloaded once.",
    )
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=None,
        help="Subset sizes to write. Defaults to the source's default_rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = SOURCES[args.dataset]
    cache = fetch(source.url, Path(args.cache_dir) / Path(source.url).name)
    print(f"hdf5: {cache}")
    for rows in args.rows or source.default_rows:
        out = subset(
            cache,
            Path(args.out_dir) / f"{subset_name(source.name, rows)}.npy",
            num_rows=rows,
            seed=args.seed,
            key=source.hdf5_key,
        )
        shape = np.load(out, mmap_mode="r").shape
        print(f"subset: {out} {shape}")
        if shape[1] != source.dim:
            raise ValueError(
                f"{source.name}: expected dim {source.dim}, file has {shape[1]}. "
                "The registry entry and the upstream file disagree."
            )


if __name__ == "__main__":
    main()
