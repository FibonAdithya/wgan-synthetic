"""Fetch ANN benchmark descriptor sets and cut reproducible subsets from them.

Writes .npy deliberately: the loader in src/data/dataset.py reads .npy and
.fvecs, so emitting .npy means neither the loader nor the trainer needs to
learn about HDF5.

Five of the six families are taken from the ann-benchmarks HDF5 mirrors. The
sixth, openai, is not published as an HDF5 at all -- upstream generates it
from a HuggingFace dataset at benchmark time -- so it is read from that
dataset's parquet shards instead. Both routes converge on the same seeded
random sample, so a subset means the same thing whichever container it came
from. Sets obtained by other routes -- corpus-texmex .fvecs, say -- are read
directly by load_descriptors and do not come through here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BASE_URL = "http://ann-benchmarks.com"

# Where a HuggingFace dataset's parquet shards are listed. Returns a JSON
# array of URLs with stable, hash-free names (.../train/0.parquet upward).
# Preferred over the repository's own filenames, which embed content hashes
# that would have to be scraped and that change on any re-upload.
HF_PARQUET_INDEX = "https://huggingface.co/api/datasets/{repo}/parquet/default/train"

# ann-benchmarks.com sits behind Cloudflare, which 403s the default
# "Python-urllib/x.y" User-Agent as a bot-blocking heuristic -- every family
# below is unreachable without this. A plain browser-like UA is enough to
# pass; no other headers are required.
USER_AGENT = "Mozilla/5.0 (compatible; wgan-synthetic-fetch/1.0)"


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
    default_rows: tuple[int, ...] = (250_000, 1_000_000)


@dataclass(frozen=True)
class ParquetSource:
    """One family that upstream generates rather than hosts.

    ann-benchmarks names `dbpedia-openai-*-angular` as datasets, but never
    publishes an HDF5 for them: `ann_benchmarks/datasets.py` builds them on
    demand from the HuggingFace dataset below. There is therefore no mirror
    to download, and the registry entry that pointed at one 404'd from the
    day it was written.

    `column` is the field holding the embedding; the other columns in the
    dataset are the source text and its identifiers, which nothing here
    wants.
    """

    name: str
    repo: str
    column: str
    dim: int
    metric: str
    # Only the 250k subset. openai's v0 names openai_250k.npy and the gate's
    # canonical N is 20,000, so a 1M subset would cost 6GB to go unread.
    default_rows: tuple[int, ...] = (250_000,)


def _ann_benchmarks(name: str, slug: str, dim: int, metric: str) -> Source:
    return Source(name=name, url=f"{BASE_URL}/{slug}.hdf5", dim=dim, metric=metric)


SOURCES: dict[str, Source | ParquetSource] = {
    "sift": _ann_benchmarks("sift", "sift-128-euclidean", 128, "l2"),
    "gist": _ann_benchmarks("gist", "gist-960-euclidean", 960, "l2"),
    "deep": _ann_benchmarks("deep", "deep-image-96-angular", 96, "angular"),
    "glove": _ann_benchmarks("glove", "glove-100-angular", 100, "angular"),
    "nytimes": _ann_benchmarks("nytimes", "nytimes-256-angular", 256, "angular"),
    "openai": ParquetSource(
        name="openai",
        repo="KShivendu/dbpedia-entities-openai-1M",
        column="openai",
        dim=1536,
        metric="angular",
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
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with os.fdopen(fd, "wb") as handle, urlopen(request) as response:
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


def shard_urls(repo: str) -> list[str]:
    """List a HuggingFace dataset's train-split parquet shards, in order."""
    request = Request(
        HF_PARQUET_INDEX.format(repo=repo), headers={"User-Agent": USER_AGENT}
    )
    with urlopen(request) as response:
        urls = json.load(response)
    if not urls:
        raise ValueError(
            f"{repo}: the parquet index listed no shards. The dataset may have "
            "been renamed, made private, or had its default config changed."
        )
    return list(urls)


def fetch_shards(source: ParquetSource, cache_dir: Path) -> list[Path]:
    """Download every parquet shard for `source`, returning them in order.

    Each shard goes through fetch(), so the atomicity and single-flight
    properties documented there hold per shard: two agents fetching this
    family at once share the download rather than doubling it.
    """
    dest_dir = Path(cache_dir) / source.repo.replace("/", "__")
    return [
        fetch(url, dest_dir / f"{i}.parquet")
        for i, url in enumerate(shard_urls(source.repo))
    ]


def _dense(column: pa.ChunkedArray) -> np.ndarray:
    """Flatten a parquet list-of-float column into a dense [rows, dim] array.

    Goes through ListArray.flatten() rather than to_pylist(): flatten()
    respects each chunk's offsets and hands back the values buffer, which
    reshapes for free. to_pylist() would materialise 59 million Python
    floats per shard.
    """
    parts = []
    for chunk in column.chunks:
        values = chunk.flatten().to_numpy(zero_copy_only=False)
        parts.append(values.reshape(len(chunk), -1))
    return np.concatenate(parts).astype(np.float32, copy=False)


def subset_parquet(
    shards: list[Path],
    out_path: Path,
    *,
    num_rows: int,
    seed: int = 42,
    column: str = "openai",
) -> Path:
    """Write a random `num_rows`-row sample drawn across every shard.

    Mirrors subset(): rows are drawn without replacement over the whole
    corpus and land in sorted index order. Sampling the corpus rather than
    reading a prefix of it is the point -- the shards are in dataset order,
    and DBpedia entities are not shuffled, so the first few shards are a
    topically skewed corpus rather than a smaller version of this one.

    Row counts come from each shard's parquet footer, which is metadata
    rather than data, so the pass that plans the sample reads almost
    nothing. Only shards a drawn row lands in are opened, one at a time.
    """
    out_path = Path(out_path)
    counts = [pq.ParquetFile(shard).metadata.num_rows for shard in shards]
    bounds = np.cumsum([0, *counts])
    total = int(bounds[-1])

    take = min(num_rows, total)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=take, replace=False))

    blocks = []
    for shard, start, stop in zip(shards, bounds[:-1], bounds[1:], strict=True):
        wanted = idx[(idx >= start) & (idx < stop)] - start
        if wanted.size == 0:
            continue
        table = pq.ParquetFile(shard).read(columns=[column])
        blocks.append(_dense(table.column(column))[wanted])

    rows = np.ascontiguousarray(np.concatenate(blocks), dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, rows)
    return out_path


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
        default="data/cache",
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
    parquet = isinstance(source, ParquetSource)

    if parquet:
        shards = fetch_shards(source, Path(args.cache_dir))
        print(f"parquet: {len(shards)} shards for {source.repo}")
    else:
        cache = fetch(source.url, Path(args.cache_dir) / Path(source.url).name)
        print(f"hdf5: {cache}")

    for rows in args.rows or source.default_rows:
        out_path = Path(args.out_dir) / f"{subset_name(source.name, rows)}.npy"
        if parquet:
            out = subset_parquet(
                shards,
                out_path,
                num_rows=rows,
                seed=args.seed,
                column=source.column,
            )
        else:
            out = subset(
                cache,
                out_path,
                num_rows=rows,
                seed=args.seed,
                key=source.hdf5_key,
            )
        shape = np.load(out, mmap_mode="r").shape
        print(f"subset: {out} {shape}")
        if shape[0] < rows:
            print(
                f"note: requested {rows} rows but the corpus only has "
                f"{shape[0]}; {out} holds {shape[0]} rows, not {rows}."
            )
        if shape[1] != source.dim:
            raise ValueError(
                f"{source.name}: expected dim {source.dim}, file has {shape[1]}. "
                "The registry entry and the upstream file disagree."
            )


if __name__ == "__main__":
    main()
