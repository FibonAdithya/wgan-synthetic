"""Tests for corpus materialization.

The property that matters most here is that real and synthetic end up in the
same space. Generators emit unit-norm vectors because every SIFT config sets
l2_normalize, and invert_preprocess cannot undo that -- the norm is gone.
data/sift_1m.npy is raw SIFT with norms in the hundreds. Indexing both as
they sit on disk would measure the scale gap rather than the corpora.
"""

import h5py
import numpy as np
import pytest

from src.eval.ann_benchmark import corpora, indexes


def test_normalize_puts_every_row_on_the_unit_sphere():
    x = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = corpora.normalize(x)
    assert np.linalg.norm(out, axis=1) == pytest.approx([1.0, 1.0])


def test_normalize_preserves_direction():
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    out = corpora.normalize(x)
    assert out[0] == pytest.approx([0.6, 0.8])


def test_normalize_leaves_a_zero_row_finite():
    # A zero row has no direction. It must not become NaN and poison every
    # distance computed against it.
    out = corpora.normalize(np.zeros((1, 3), dtype=np.float32))
    assert np.all(np.isfinite(out))


def test_normalize_is_idempotent():
    x = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    once = corpora.normalize(x)
    assert corpora.normalize(once) == pytest.approx(once)


def test_read_hdf5_queries_reads_the_test_key(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    expected = np.arange(20, dtype=np.float32).reshape(5, 4)
    with h5py.File(cache / "sift-128-euclidean.hdf5", "w") as f:
        f.create_dataset("train", data=np.zeros((3, 4), dtype=np.float32))
        f.create_dataset("test", data=expected)

    got = corpora.read_hdf5_queries(cache, num_queries=5)
    assert got.shape == (5, 4)
    assert got == pytest.approx(expected)


def test_read_hdf5_queries_clamps_to_what_exists(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    with h5py.File(cache / "sift-128-euclidean.hdf5", "w") as f:
        f.create_dataset("test", data=np.zeros((3, 4), dtype=np.float32))
    assert corpora.read_hdf5_queries(cache, num_queries=99).shape[0] == 3


def test_read_hdf5_queries_names_the_directory_when_no_file_is_there(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(FileNotFoundError, match=str(cache)):
        corpora.read_hdf5_queries(cache, num_queries=5)


def test_read_hdf5_queries_names_the_key_when_it_is_absent(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    with h5py.File(cache / "sift-128-euclidean.hdf5", "w") as f:
        f.create_dataset("train", data=np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(KeyError, match="test"):
        corpora.read_hdf5_queries(cache, num_queries=5)


def test_query_seed_differs_from_corpus_seed():
    # The query draw must not reproduce the corpus draw, or every query would
    # be an exact member of the index and recall would read as 1.0 everywhere.
    assert corpora.query_seed(42, "v2") != corpora.corpus_seed(42, "v2")


def test_seeds_are_stable_across_calls():
    assert corpora.query_seed(42, "v2") == corpora.query_seed(42, "v2")


def test_seeds_depend_on_the_variant_name_not_on_call_order():
    assert corpora.corpus_seed(42, "v0") != corpora.corpus_seed(42, "v2")


def test_materialize_real_normalizes_and_caches(tmp_path):
    raw = np.array([[3.0, 4.0], [0.0, 5.0], [5.0, 0.0]], dtype=np.float32)
    real_path = tmp_path / "real.npy"
    np.save(real_path, raw)

    cache = tmp_path / "cache"
    cache.mkdir()
    with h5py.File(cache / "sift-128-euclidean.hdf5", "w") as f:
        f.create_dataset("test", data=np.array([[1.0, 1.0]], dtype=np.float32))

    work = tmp_path / "work"
    corpus = corpora.materialize_real(
        real_path=real_path,
        cache_dir=cache,
        work_dir=work,
        num_vectors=3,
        num_queries=1,
        k=2,
        # The default is the GPU brute-force index; there is no GPU under
        # pytest, so ground truth comes from the numpy stand-in.
        adapter=indexes.NumpyFlatAdapter(),
    )

    vectors = np.load(corpus.vectors_path)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0, 1.0])
    queries = np.load(corpus.queries_path)
    assert np.linalg.norm(queries, axis=1) == pytest.approx([1.0])
    assert np.load(corpus.truth_ids_path).shape == (1, 2)
    assert np.load(corpus.truth_distances_path).shape == (1, 2)


def test_materialize_real_reuses_the_cache_on_a_second_call(tmp_path):
    raw = np.array([[3.0, 4.0], [0.0, 5.0], [5.0, 0.0]], dtype=np.float32)
    real_path = tmp_path / "real.npy"
    np.save(real_path, raw)
    cache = tmp_path / "cache"
    cache.mkdir()
    with h5py.File(cache / "sift-128-euclidean.hdf5", "w") as f:
        f.create_dataset("test", data=np.array([[1.0, 1.0]], dtype=np.float32))
    work = tmp_path / "work"

    kwargs = dict(
        real_path=real_path,
        cache_dir=cache,
        work_dir=work,
        num_vectors=3,
        num_queries=1,
        k=2,
        adapter=indexes.NumpyFlatAdapter(),
    )
    first = corpora.materialize_real(**kwargs)
    stamp = first.vectors_path.stat().st_mtime_ns

    # Deleting the source proves the second call did not re-read it.
    real_path.unlink()
    second = corpora.materialize_real(**kwargs)
    assert second.vectors_path.stat().st_mtime_ns == stamp
