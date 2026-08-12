"""Materialize the corpora, query sets and ground truth the grid runs over.

Everything lands on the unit sphere. This is the decision the whole benchmark
turns on and it is not cosmetic: every SIFT config sets
`preprocess.l2_normalize`, so generators emit unit-norm vectors, and
`src.data.dataset.invert_preprocess` deliberately does not undo that -- the
norm is discarded and the information is gone. Meanwhile `data/sift_1m.npy`
is raw SIFT with norms in the hundreds. Building indexes over both as they sit
on disk would measure the scale difference rather than the corpora.

The cost is real and belongs in the report, not just here: normalizing real
SIFT discards its norm distribution, which is itself part of SIFT's search
difficulty. These figures describe normalized SIFT and are not comparable with
published SIFT1M results -- which, per invariant 3, they never were.

Each corpus is written to the work directory once and reused. Drawing seven
million vectors and running seven exact-kNN passes is the deterministic and
expensive half of the job; a crash inside the grid must not re-pay it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from src.data.dataset import load_descriptors
from src.device import resolve_device
from src.eval.ann_benchmark.groundtruth import exact_neighbours
from src.eval.ann_benchmark.indexes import IndexAdapter
from src.eval.compare_variants import (
    CHECKPOINT_NAME,
    RUN_CONFIG_NAME,
    Variant,
    invert_samples,
    load_preprocess_state,
    variant_seed,
)
from src.eval.evaluate_distribution import load_generator
from src.train.train_wgan_gp import sample_generator

EPS = 1.0e-8
HDF5_QUERY_KEY = "test"


@dataclass(frozen=True)
class Corpus:
    """One corpus, its queries, and its exact neighbours -- all on disk."""

    name: str
    vectors_path: Path
    queries_path: Path
    truth_distances_path: Path
    truth_ids_path: Path
    num_vectors: int
    num_queries: int
    dim: int


def normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows, leaving a zero row finite rather than NaN.

    Matches `src.data.dataset.apply_preprocess` for a config with
    `center: false, whiten: false, l2_normalize: true` -- which is every SIFT
    config -- so real and synthetic land in one space by construction rather
    than by coincidence.
    """
    out = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.clip(norm, EPS, None)).astype(np.float32)


def corpus_seed(base_seed: int, name: str) -> int:
    """Seed for a variant's corpus draw, independent of run order.

    Delegates to `compare_variants.variant_seed`, which already derives a
    per-name seed via sha256 over `(base_seed, name)`; salting the digest
    input with a `"corpus:"` prefix is the only thing added here, and it is
    what keeps this draw from ever coinciding with `query_seed`'s.
    """
    return variant_seed(base_seed, f"corpus:{name}")


def query_seed(base_seed: int, name: str) -> int:
    """Seed for a variant's query draw.

    Salted differently from `corpus_seed` (`"query:"` vs `"corpus:"`) so the
    two draws cannot coincide. If they did, every query would be an exact
    member of the index and recall would read 1.0 for every configuration of
    every algorithm -- a failure that produces a perfectly plausible-looking
    table.
    """
    return variant_seed(base_seed, f"query:{name}")


def read_hdf5_queries(cache_dir: Path, num_queries: int) -> np.ndarray:
    """Read SIFT's own query set out of the cached ann-benchmarks HDF5.

    `src.data.fetch` reads only the `train` key and never writes the queries
    to disk, so this reaches into the cache the fetcher already populated
    rather than adding a download or changing the fetcher.
    """
    cache_dir = Path(cache_dir)
    candidates = sorted(cache_dir.glob("*.hdf5"))
    if not candidates:
        raise FileNotFoundError(
            f"no .hdf5 in {cache_dir}. That cache is populated by "
            "`python -m src.data.fetch sift`; pass --cache-dir if it lives "
            "elsewhere on this box."
        )
    with h5py.File(candidates[0], "r") as handle:
        if HDF5_QUERY_KEY not in handle:
            raise KeyError(
                f"{candidates[0]} has no {HDF5_QUERY_KEY!r} dataset; found "
                f"{sorted(handle.keys())}. The real query set is what the "
                "'real' corpus is searched with."
            )
        data = handle[HDF5_QUERY_KEY][:num_queries]
    return np.asarray(data, dtype=np.float32)


def _write_truth(
    corpus_dir: Path,
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    adapter: IndexAdapter | None,
) -> tuple[Path, Path]:
    distances, ids = exact_neighbours(vectors, queries, k, adapter=adapter)
    distances_path = corpus_dir / "truth_distances.npy"
    ids_path = corpus_dir / "truth_ids.npy"
    np.save(distances_path, distances.astype(np.float32))
    np.save(ids_path, ids.astype(np.int64))
    return distances_path, ids_path


def _corpus_from_dir(name: str, corpus_dir: Path) -> Corpus:
    vectors_path = corpus_dir / "vectors.npy"
    queries_path = corpus_dir / "queries.npy"
    vectors = np.load(vectors_path, mmap_mode="r")
    queries = np.load(queries_path, mmap_mode="r")
    return Corpus(
        name=name,
        vectors_path=vectors_path,
        queries_path=queries_path,
        truth_distances_path=corpus_dir / "truth_distances.npy",
        truth_ids_path=corpus_dir / "truth_ids.npy",
        num_vectors=int(vectors.shape[0]),
        num_queries=int(queries.shape[0]),
        dim=int(vectors.shape[1]),
    )


def _is_complete(corpus_dir: Path, num_vectors: int, num_queries: int, k: int) -> bool:
    """True only when a cached corpus exists *and* matches what was asked for.

    Existence alone is not enough: a work directory reused across a smoke
    test (say, --num-vectors 20000) and the real run (1,000,000) would
    otherwise serve the smoke corpus back to the real run with no error and
    no warning -- every number in the table silently wrong while the table
    itself looks entirely normal. Shapes are read via `mmap_mode="r"`, which
    touches only the header, so validating them costs nothing beyond a stat.
    """
    names = ("vectors.npy", "queries.npy", "truth_distances.npy", "truth_ids.npy")
    if not all((corpus_dir / n).exists() for n in names):
        return False
    try:
        vectors = np.load(corpus_dir / "vectors.npy", mmap_mode="r")
        queries = np.load(corpus_dir / "queries.npy", mmap_mode="r")
        truth_distances = np.load(corpus_dir / "truth_distances.npy", mmap_mode="r")
        truth_ids = np.load(corpus_dir / "truth_ids.npy", mmap_mode="r")
    except (OSError, ValueError):
        # A partially written or corrupt cache is a miss, not a crash.
        return False
    return (
        vectors.shape[0] == num_vectors
        and queries.shape[0] == num_queries
        and truth_distances.shape == (num_queries, k)
        and truth_ids.shape == (num_queries, k)
    )


def materialize_real(
    *,
    real_path: Path,
    cache_dir: Path,
    work_dir: Path,
    num_vectors: int,
    num_queries: int,
    k: int,
    adapter: IndexAdapter | None = None,
) -> Corpus:
    """The real corpus, normalized, with SIFT's own query set."""
    corpus_dir = Path(work_dir) / "real"
    if _is_complete(corpus_dir, num_vectors, num_queries, k):
        return _corpus_from_dir("real", corpus_dir)

    corpus_dir.mkdir(parents=True, exist_ok=True)
    vectors = normalize(load_descriptors(Path(real_path))[:num_vectors])
    queries = normalize(read_hdf5_queries(Path(cache_dir), num_queries))
    np.save(corpus_dir / "vectors.npy", vectors)
    np.save(corpus_dir / "queries.npy", queries)
    _write_truth(corpus_dir, vectors, queries, k, adapter)
    return _corpus_from_dir("real", corpus_dir)


def _draw(
    variant: Variant, root: Path, count: int, batch_size: int, seed: int
) -> np.ndarray:
    """Draw `count` vectors from a variant's best checkpoint.

    The checkpoint is rebuilt against its own `run_config.yaml`, never against
    the config checked into `configs/` -- `generator_type` is not recorded in
    the checkpoint, so the run config is the only thing that knows which
    architecture these weights belong to (invariant 4).
    """
    run_dir = Path(root) / variant.run_dir
    config = yaml.safe_load((run_dir / RUN_CONFIG_NAME).read_text(encoding="utf-8"))
    device = resolve_device(config["device"])
    generator = load_generator(config, run_dir / CHECKPOINT_NAME, device)
    torch.manual_seed(seed)
    drawn = sample_generator(
        generator,
        num_samples=count,
        latent_dim=int(config["model"]["latent_dim"]),
        batch_size=batch_size,
        device=device,
    )
    return invert_samples(drawn, load_preprocess_state(run_dir))


def materialize_variant(
    variant: Variant,
    *,
    root: Path,
    work_dir: Path,
    num_vectors: int,
    num_queries: int,
    k: int,
    batch_size: int,
    seed: int,
    adapter: IndexAdapter | None = None,
) -> Corpus:
    """One synthetic corpus plus a disjoint query draw from the same generator.

    Queries come from a second draw under a different seed rather than from a
    holdout of the corpus. That mirrors how SIFT's query set relates to its
    base set -- same distribution, different sample -- so each corpus is
    searched the way it would actually be used.
    """
    corpus_dir = Path(work_dir) / variant.name
    if _is_complete(corpus_dir, num_vectors, num_queries, k):
        return _corpus_from_dir(variant.name, corpus_dir)

    corpus_dir.mkdir(parents=True, exist_ok=True)
    vectors = normalize(
        _draw(variant, root, num_vectors, batch_size, corpus_seed(seed, variant.name))
    )
    queries = normalize(
        _draw(variant, root, num_queries, batch_size, query_seed(seed, variant.name))
    )
    np.save(corpus_dir / "vectors.npy", vectors)
    np.save(corpus_dir / "queries.npy", queries)
    _write_truth(corpus_dir, vectors, queries, k, adapter)
    return _corpus_from_dir(variant.name, corpus_dir)
