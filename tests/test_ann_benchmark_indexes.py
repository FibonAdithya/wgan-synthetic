"""Tests for the ANN index adapter boundary.

These run on a CPU-only box with no cuVS installed, which is the property
being tested: importing this module must not import cuVS. The adapters'
device code is a thin edge covered by the box run, not by pytest.
"""

import sys
import types

import numpy as np
import pytest

from src.eval.ann_benchmark import indexes


def test_module_imports_without_cuvs():
    # Importing the module is the assertion. If any cuVS import moved to
    # module scope this test fails at collection on every CPU-only machine.
    assert indexes.ADAPTER_NAMES == ("flat", "ivf_flat", "ivf_pq", "cagra")


def test_build_adapters_returns_requested_adapters_in_order():
    got = indexes.build_adapters(["cagra", "flat"])
    assert [a.name for a in got] == ["cagra", "flat"]


def test_build_adapters_rejects_unknown_name():
    with pytest.raises(ValueError, match="hnsw"):
        indexes.build_adapters(["hnsw"])


def test_flat_adapter_has_no_swept_parameter():
    (flat,) = indexes.build_adapters(["flat"])
    assert flat.sweep_params() == (None,)
    assert flat.param_name == ""


def test_ivf_adapters_sweep_n_probes():
    ivf_flat, ivf_pq = indexes.build_adapters(["ivf_flat", "ivf_pq"])
    assert ivf_flat.param_name == "n_probes"
    assert ivf_pq.param_name == "n_probes"
    assert ivf_flat.sweep_params() == (1, 2, 4, 8, 16, 32, 64, 128, 256)


def test_cagra_sweeps_itopk_size():
    (cagra,) = indexes.build_adapters(["cagra"])
    assert cagra.param_name == "itopk_size"
    assert cagra.sweep_params() == (32, 64, 128, 256, 512)


def test_describe_records_the_fixed_build_parameters():
    ivf_flat, ivf_pq, cagra = indexes.build_adapters(["ivf_flat", "ivf_pq", "cagra"])
    assert ivf_flat.describe()["n_lists"] == 4096
    assert ivf_pq.describe()["pq_dim"] == 64
    assert ivf_pq.describe()["pq_bits"] == 8
    assert cagra.describe()["graph_degree"] == 64
    assert cagra.describe()["intermediate_graph_degree"] == 128


def test_adapters_report_a_missing_cuvs_with_an_install_command():
    # Skipped on the GPU box, where cuVS is installed and the build succeeds.
    # `make check` also runs there during Task 9, and a test that can only
    # pass on one of the two machines is a test that will be deleted.
    try:
        import cuvs  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("cuVS is installed; the missing-dependency path cannot run")

    (flat,) = indexes.build_adapters(["flat"])
    with pytest.raises(RuntimeError, match="pip install cuvs-cu13"):
        flat.build(np.zeros((4, 2), dtype=np.float32))
    # The CLI preflight raises the same message, so a missing dependency is
    # reported before an hour of corpus materialization rather than after.
    with pytest.raises(RuntimeError, match="pip install cuvs-cu13"):
        indexes.require_device_stack()


def test_built_index_carries_a_vram_figure_slot():
    # None off-device rather than absent, so report.py never has to branch on
    # whether the field exists.
    built = indexes.NumpyFlatAdapter().build(np.eye(2, dtype=np.float32))
    assert built.peak_vram_bytes is None
    # NumpyFlatAdapter owns its data on the host: no device buffer to keep
    # alive, so `dataset` stays at its default rather than being populated.
    assert built.dataset is None


def test_built_index_keeps_the_device_dataset_alive(monkeypatch):
    """Regression test for a use-after-free.

    cuVS does not copy or take ownership of the dataset passed to `build()`
    -- the index stores a pointer into that device buffer. If `BuiltIndex`
    does not keep its own reference, the local `device_vectors` in each
    adapter's `build()` goes out of scope when the method returns, cupy's
    refcounting GC frees the block, the allocator hands it to the next
    allocation, and every later `search()` silently reads recycled memory:
    no exception, just wrong, plausible-looking results (reproduced on the
    GPU box, see task-3-report.md). This pins `BuiltIndex.dataset is` the
    exact object cuVS's `build()` received, for all four cuVS adapters, using
    fake `cuvs`/`cupy` modules so it runs on a CPU-only box.
    """

    class FakeDeviceArray:
        """Stands in for a cupy ndarray; only its identity matters here."""

    class FakeResources:
        def sync(self):
            pass

    captured: dict[str, object] = {}

    def make_neighbors_module(mod_name, *, dataset_first):
        mod = types.ModuleType(f"cuvs.neighbors.{mod_name}")

        class Handle:
            pass

        if dataset_first:
            # brute_force.build(dataset, metric=..., ...)

            def build(dataset, metric="sqeuclidean", **kwargs):
                captured[mod_name] = dataset
                return Handle()
        else:
            # ivf_flat/ivf_pq/cagra .build(index_params, dataset, ...)

            def build(index_params, dataset, **kwargs):
                captured[mod_name] = dataset
                return Handle()

        class IndexParams:
            def __init__(self, **kwargs):
                pass

        class SearchParams:
            def __init__(self, **kwargs):
                pass

        mod.build = build
        mod.IndexParams = IndexParams
        mod.SearchParams = SearchParams
        return mod

    fake_cupy = types.ModuleType("cupy")
    fake_cupy.asarray = lambda x: FakeDeviceArray()
    fake_cupy.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(memGetInfo=lambda: (1_000, 2_000))
    )

    fake_common = types.ModuleType("cuvs.common")
    fake_common.Resources = FakeResources

    fake_neighbors = types.ModuleType("cuvs.neighbors")
    fake_neighbors.brute_force = make_neighbors_module(
        "brute_force", dataset_first=True
    )
    fake_neighbors.ivf_flat = make_neighbors_module("ivf_flat", dataset_first=False)
    fake_neighbors.ivf_pq = make_neighbors_module("ivf_pq", dataset_first=False)
    fake_neighbors.cagra = make_neighbors_module("cagra", dataset_first=False)

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cuvs", types.ModuleType("cuvs"))
    monkeypatch.setitem(sys.modules, "cuvs.common", fake_common)
    monkeypatch.setitem(sys.modules, "cuvs.neighbors", fake_neighbors)

    vectors = np.eye(4, dtype=np.float32)
    for adapter_cls, captured_key in [
        (indexes.FlatAdapter, "brute_force"),
        (indexes.IvfFlatAdapter, "ivf_flat"),
        (indexes.IvfPqAdapter, "ivf_pq"),
        (indexes.CagraAdapter, "cagra"),
    ]:
        built = adapter_cls().build(vectors)
        assert built.dataset is not None
        # The exact object the fake `build()` received as its dataset must
        # still be reachable from `BuiltIndex`. If `dataset` is dropped (or
        # someone "tidies up" the field), this becomes a plain identity
        # mismatch here instead of a silent corruption on the GPU box.
        assert built.dataset is captured[captured_key]


def test_numpy_adapter_is_a_working_stand_in_for_the_runner():
    # The fake used by the runner tests lives here so both sides agree on the
    # interface. It is exact brute force in numpy over squared L2.
    adapter = indexes.NumpyFlatAdapter()
    vectors = np.eye(4, dtype=np.float32)
    built = adapter.build(vectors)
    assert built.train_seconds >= 0.0
    assert built.index_bytes_estimated == vectors.nbytes

    dist, ids = adapter.search(built, vectors[:2], k=2, param=None)
    assert dist.shape == (2, 2)
    assert ids.shape == (2, 2)
    # Each query is a row of the index, so its own row is the nearest at 0.
    assert dist[:, 0] == pytest.approx([0.0, 0.0])
    assert list(ids[:, 0]) == [0, 1]


def test_exact_neighbours_matches_a_hand_computed_answer():
    from src.eval.ann_benchmark import groundtruth

    vectors = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 0.0]], dtype=np.float32
    )
    queries = np.array([[0.0, 0.0]], dtype=np.float32)
    dist, ids = groundtruth.exact_neighbours(
        vectors, queries, k=3, adapter=indexes.NumpyFlatAdapter()
    )
    assert list(ids[0]) == [0, 1, 2]
    assert dist[0] == pytest.approx([0.0, 1.0, 1.0])


def test_exact_neighbours_rejects_k_larger_than_the_corpus():
    from src.eval.ann_benchmark import groundtruth

    vectors = np.zeros((3, 2), dtype=np.float32)
    queries = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="k=5"):
        groundtruth.exact_neighbours(
            vectors, queries, k=5, adapter=indexes.NumpyFlatAdapter()
        )


def test_exact_neighbours_rejects_mismatched_dimensions():
    from src.eval.ann_benchmark import groundtruth

    vectors = np.zeros((3, 4), dtype=np.float32)
    queries = np.zeros((1, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="dimension mismatch"):
        groundtruth.exact_neighbours(
            vectors, queries, k=1, adapter=indexes.NumpyFlatAdapter()
        )


def test_gpu_and_numpy_exact_neighbours_agree():
    # The GPU brute-force path and the numpy fallback must agree exactly on
    # ties and ordering: `recall_at_k` compares found distances directly
    # against whichever one produced ground truth. Skipped where cuVS is not
    # installed (the CPU-only `make check` box); runs for real on the GPU box.
    try:
        import cuvs  # noqa: F401
    except ImportError:
        pytest.skip("cuVS is not installed; the GPU path cannot run")

    from src.eval.ann_benchmark import groundtruth

    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((50, 8)).astype(np.float32)
    queries = rng.standard_normal((5, 8)).astype(np.float32)

    gpu_dist, gpu_ids = groundtruth.exact_neighbours(
        vectors, queries, k=10, adapter=indexes.FlatAdapter()
    )
    numpy_dist, numpy_ids = groundtruth.exact_neighbours(
        vectors, queries, k=10, adapter=indexes.NumpyFlatAdapter()
    )
    assert list(gpu_ids.flatten()) == list(numpy_ids.flatten())
    assert gpu_dist == pytest.approx(numpy_dist, abs=1e-4)
