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
from src.eval.compare_variants import Variant


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


def test_materialize_real_rematerializes_when_the_cache_shape_does_not_match(
    tmp_path,
):
    # Reproduces the exact hazard the brief warns about: a work dir populated
    # by a small smoke run (e.g. --num-vectors 20000) must not be silently
    # served back to a later call asking for a different size (the real run,
    # --num-vectors 1000000). Existence-only caching would return the smoke
    # corpus with no error and no warning, and every number downstream would
    # be quietly wrong.
    raw = np.array(
        [[3.0, 4.0], [0.0, 5.0], [5.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        dtype=np.float32,
    )
    real_path = tmp_path / "real.npy"
    np.save(real_path, raw)
    cache = tmp_path / "cache"
    cache.mkdir()
    with h5py.File(cache / "sift-128-euclidean.hdf5", "w") as f:
        f.create_dataset(
            "test",
            data=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        )
    work = tmp_path / "work"

    small = corpora.materialize_real(
        real_path=real_path,
        cache_dir=cache,
        work_dir=work,
        num_vectors=3,
        num_queries=1,
        k=2,
        adapter=indexes.NumpyFlatAdapter(),
    )
    assert small.num_vectors == 3
    assert small.num_queries == 1

    # A second call against the same work_dir but a different requested size
    # must rematerialize, not serve the stale 3/1 cache back as if it were
    # the 5/3 corpus.
    big = corpora.materialize_real(
        real_path=real_path,
        cache_dir=cache,
        work_dir=work,
        num_vectors=5,
        num_queries=3,
        k=2,
        adapter=indexes.NumpyFlatAdapter(),
    )
    assert big.num_vectors == 5
    assert big.num_queries == 3
    vectors = np.load(big.vectors_path)
    assert vectors.shape == (5, 2)
    queries = np.load(big.queries_path)
    assert queries.shape == (3, 2)
    assert np.load(big.truth_ids_path).shape == (3, 2)
    assert np.load(big.truth_distances_path).shape == (3, 2)


def test_materialize_variant_threads_corpus_and_query_seeds_correctly(
    tmp_path, monkeypatch
):
    # This is the property the whole task exists to guarantee: the vectors
    # draw must use corpus_seed and the queries draw must use query_seed. If
    # those two arguments were ever swapped, every query would land inside
    # its own index and recall would read 1.0 everywhere -- silently, with
    # no error and a perfectly plausible-looking table. Monkeypatching _draw
    # pins the wiring directly rather than inferring it from output shapes.
    calls = []

    def fake_draw(variant, root, count, batch_size, seed):
        calls.append((count, seed))
        rng = np.random.default_rng(seed)
        return rng.normal(size=(count, 4)).astype(np.float32)

    monkeypatch.setattr(corpora, "_draw", fake_draw)

    variant = Variant(name="v2", config_path="configs/v2.yaml", run_dir="runs/v2")
    work = tmp_path / "work"
    corpus = corpora.materialize_variant(
        variant,
        root=tmp_path,
        work_dir=work,
        num_vectors=5,
        num_queries=3,
        k=2,
        batch_size=8,
        seed=42,
        adapter=indexes.NumpyFlatAdapter(),
    )

    expected_corpus_seed = corpora.corpus_seed(42, "v2")
    expected_query_seed = corpora.query_seed(42, "v2")
    assert expected_corpus_seed != expected_query_seed
    assert calls == [(5, expected_corpus_seed), (3, expected_query_seed)]
    assert corpus.num_vectors == 5
    assert corpus.num_queries == 3
