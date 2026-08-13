"""Tests for corpus materialization.

The property that matters most here is that real and synthetic end up in the
same space. Generators emit unit-norm vectors because every SIFT config sets
l2_normalize, and invert_preprocess cannot undo that -- the norm is gone.
data/sift_1m.npy is raw SIFT with norms in the hundreds. Indexing both as
they sit on disk would measure the scale gap rather than the corpora.
"""

import json

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


def test_read_hdf5_queries_picks_sift_not_whatever_sorts_first(tmp_path):
    # Reproduces the failure found on the GPU box: a cache holding several
    # ann-benchmarks families at once, where "deep-image..." sorts before
    # "sift...". Taking candidates[0] would search real SIFT with DEEP-image
    # queries -- both files carry a `test` key, so nothing about existence or
    # key-presence would catch it.
    cache = tmp_path / "cache"
    cache.mkdir()
    deep_queries = np.zeros((2, 96), dtype=np.float32)
    sift_queries = np.ones((2, 128), dtype=np.float32)
    with h5py.File(cache / "deep-image-96-angular.hdf5", "w") as f:
        f.create_dataset("test", data=deep_queries)
    with h5py.File(cache / "glove-100-angular.hdf5", "w") as f:
        f.create_dataset("test", data=np.full((2, 100), 2.0, dtype=np.float32))
    with h5py.File(cache / "sift-128-euclidean.hdf5", "w") as f:
        f.create_dataset("test", data=sift_queries)

    got = corpora.read_hdf5_queries(cache, num_queries=2)
    assert got.shape == (2, 128)
    assert got == pytest.approx(sift_queries)


def test_read_hdf5_queries_raises_on_an_ambiguous_sift_match(tmp_path):
    # Two candidates both matching the SIFT name hint must not be resolved by
    # sort order either -- that is exactly the bug being fixed, one level in.
    cache = tmp_path / "cache"
    cache.mkdir()
    with h5py.File(cache / "sift-128-euclidean.hdf5", "w") as f:
        f.create_dataset("test", data=np.zeros((2, 128), dtype=np.float32))
    with h5py.File(cache / "sift-small-128-euclidean.hdf5", "w") as f:
        f.create_dataset("test", data=np.zeros((2, 128), dtype=np.float32))

    with pytest.raises(ValueError, match="ambiguous"):
        corpora.read_hdf5_queries(cache, num_queries=2)


def test_read_hdf5_queries_explicit_path_overrides_selection(tmp_path):
    # An explicit hdf5_path is the caller's word: it must be used even when
    # it does not match the SIFT name hint and even when other candidates
    # (including ones that would themselves be ambiguous) are present.
    cache = tmp_path / "cache"
    cache.mkdir()
    named_anything = cache / "not-named-like-sift-at-all.hdf5"
    expected = np.full((2, 128), 7.0, dtype=np.float32)
    with h5py.File(named_anything, "w") as f:
        f.create_dataset("test", data=expected)
    with h5py.File(cache / "sift-a-128-euclidean.hdf5", "w") as f:
        f.create_dataset("test", data=np.zeros((2, 128), dtype=np.float32))
    with h5py.File(cache / "sift-b-128-euclidean.hdf5", "w") as f:
        f.create_dataset("test", data=np.zeros((2, 128), dtype=np.float32))

    got = corpora.read_hdf5_queries(cache, num_queries=2, hdf5_path=named_anything)
    assert got == pytest.approx(expected)


def test_materialize_real_raises_when_the_hdf5_dimension_does_not_match(tmp_path):
    # End-to-end version of the same hazard: the wrong-family HDF5 must be
    # caught inside materialize_real, naming the file, rather than surfacing
    # later as an opaque dimension mismatch with no file attached to it.
    raw = np.array([[3.0, 4.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32)
    real_path = tmp_path / "real.npy"
    np.save(real_path, raw)

    cache = tmp_path / "cache"
    cache.mkdir()
    wrong_dim_hdf5 = cache / "deep-image-96-angular.hdf5"
    with h5py.File(wrong_dim_hdf5, "w") as f:
        f.create_dataset("test", data=np.zeros((1, 96), dtype=np.float32))

    work = tmp_path / "work"
    with pytest.raises(ValueError, match=str(wrong_dim_hdf5)):
        corpora.materialize_real(
            real_path=real_path,
            cache_dir=cache,
            work_dir=work,
            num_vectors=2,
            num_queries=1,
            k=1,
            hdf5_path=wrong_dim_hdf5,
            adapter=indexes.NumpyFlatAdapter(),
        )


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


# --- manifest.json: a same-shape cache hit must still respect --seed,
# --batch-size, --real-path and --real-hdf5-path. Shape-only validation
# (above) already caught a different-size cache being served back; it did
# not catch a same-size cache holding a *different draw*, which is what
# these pin down.


def test_materialize_variant_writes_a_manifest_recording_its_inputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        corpora,
        "_draw",
        lambda variant, root, count, batch_size, seed: np.random.default_rng(seed)
        .normal(size=(count, 4))
        .astype(np.float32),
    )
    variant = Variant(name="v2", config_path="configs/v2.yaml", run_dir="runs/v2")
    work = tmp_path / "work"
    corpora.materialize_variant(
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
    manifest = json.loads((work / "v2" / "manifest.json").read_text())
    assert manifest["seed"] == 42
    assert manifest["batch_size"] == 8
    assert manifest["num_vectors"] == 5
    assert manifest["num_queries"] == 3
    assert manifest["k"] == 2
    assert manifest["dim"] == 4


def test_materialize_variant_rematerializes_when_the_seed_changes(
    tmp_path, monkeypatch
):
    # Same shape, different --seed: existence- or shape-only caching would
    # silently serve the seed=42 draw back to a run that asked for seed=99.
    monkeypatch.setattr(
        corpora,
        "_draw",
        lambda variant, root, count, batch_size, seed: np.random.default_rng(seed)
        .normal(size=(count, 4))
        .astype(np.float32),
    )
    variant = Variant(name="v2", config_path="configs/v2.yaml", run_dir="runs/v2")
    work = tmp_path / "work"
    kwargs = dict(
        root=tmp_path,
        work_dir=work,
        num_vectors=5,
        num_queries=3,
        k=2,
        batch_size=8,
        adapter=indexes.NumpyFlatAdapter(),
    )
    first = corpora.materialize_variant(variant, seed=42, **kwargs)
    first_vectors = np.load(first.vectors_path).copy()

    second = corpora.materialize_variant(variant, seed=99, **kwargs)
    second_vectors = np.load(second.vectors_path)

    assert first.num_vectors == second.num_vectors == 5
    assert not np.array_equal(first_vectors, second_vectors)


def test_materialize_variant_rematerializes_when_batch_size_changes(
    tmp_path, monkeypatch
):
    calls = []

    def fake_draw(variant, root, count, batch_size, seed):
        calls.append(batch_size)
        return np.random.default_rng(seed).normal(size=(count, 4)).astype(np.float32)

    monkeypatch.setattr(corpora, "_draw", fake_draw)
    variant = Variant(name="v2", config_path="configs/v2.yaml", run_dir="runs/v2")
    work = tmp_path / "work"
    kwargs = dict(
        variant=variant,
        root=tmp_path,
        work_dir=work,
        num_vectors=5,
        num_queries=3,
        k=2,
        seed=42,
        adapter=indexes.NumpyFlatAdapter(),
    )
    corpora.materialize_variant(batch_size=8, **kwargs)
    corpora.materialize_variant(batch_size=99, **kwargs)

    # Each materialize_variant call draws twice (vectors, then queries), so
    # a real draw at each batch_size shows up as two calls; a cache hit on
    # the second materialize_variant call would leave only the first pair.
    assert calls == [8, 8, 99, 99]


def test_materialize_real_writes_a_manifest_recording_its_inputs(tmp_path):
    raw = np.array([[3.0, 4.0], [0.0, 5.0], [5.0, 0.0]], dtype=np.float32)
    real_path = tmp_path / "real.npy"
    np.save(real_path, raw)
    cache = tmp_path / "cache"
    cache.mkdir()
    with h5py.File(cache / "sift-128-euclidean.hdf5", "w") as f:
        f.create_dataset("test", data=np.array([[1.0, 1.0]], dtype=np.float32))
    work = tmp_path / "work"

    corpora.materialize_real(
        real_path=real_path,
        cache_dir=cache,
        work_dir=work,
        num_vectors=3,
        num_queries=1,
        k=2,
        adapter=indexes.NumpyFlatAdapter(),
    )
    manifest = json.loads((work / "real" / "manifest.json").read_text())
    assert manifest["num_vectors"] == 3
    assert manifest["num_queries"] == 1
    assert manifest["k"] == 2
    assert manifest["dim"] == 2
    assert str(real_path) in manifest["source"]


def test_materialize_real_rematerializes_when_the_real_path_changes(tmp_path):
    # Same requested shape, a different source file: a shape-only cache
    # check cannot tell these apart, but the data is different every time.
    raw_a = np.array([[3.0, 4.0], [0.0, 5.0], [5.0, 0.0]], dtype=np.float32)
    raw_b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    path_a = tmp_path / "a.npy"
    path_b = tmp_path / "b.npy"
    np.save(path_a, raw_a)
    np.save(path_b, raw_b)
    cache = tmp_path / "cache"
    cache.mkdir()
    with h5py.File(cache / "sift-128-euclidean.hdf5", "w") as f:
        f.create_dataset("test", data=np.array([[1.0, 1.0]], dtype=np.float32))
    work = tmp_path / "work"
    kwargs = dict(
        cache_dir=cache,
        work_dir=work,
        num_vectors=3,
        num_queries=1,
        k=2,
        adapter=indexes.NumpyFlatAdapter(),
    )

    first = corpora.materialize_real(real_path=path_a, **kwargs)
    first_vectors = np.load(first.vectors_path).copy()

    second = corpora.materialize_real(real_path=path_b, **kwargs)
    second_vectors = np.load(second.vectors_path)

    assert not np.array_equal(first_vectors, second_vectors)


def test_materialize_real_rematerializes_when_the_hdf5_path_changes(tmp_path):
    # Same real_path, same requested shape, a different --real-hdf5-path:
    # the query set changes even though the corpus vectors would not.
    raw = np.array([[3.0, 4.0], [0.0, 5.0], [5.0, 0.0]], dtype=np.float32)
    real_path = tmp_path / "real.npy"
    np.save(real_path, raw)
    cache = tmp_path / "cache"
    cache.mkdir()
    hdf5_a = cache / "sift-a-128-euclidean.hdf5"
    hdf5_b = cache / "sift-b-128-euclidean.hdf5"
    with h5py.File(hdf5_a, "w") as f:
        f.create_dataset("test", data=np.array([[1.0, 0.0]], dtype=np.float32))
    with h5py.File(hdf5_b, "w") as f:
        f.create_dataset("test", data=np.array([[0.0, 1.0]], dtype=np.float32))
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

    first = corpora.materialize_real(hdf5_path=hdf5_a, **kwargs)
    first_queries = np.load(first.queries_path).copy()

    second = corpora.materialize_real(hdf5_path=hdf5_b, **kwargs)
    second_queries = np.load(second.queries_path)

    assert not np.array_equal(first_queries, second_queries)
