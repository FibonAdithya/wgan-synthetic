"""GPU ANN index adapters -- the single boundary where cuVS is named.

`runner.py` drives everything through `IndexAdapter`, so the grid loop has no
device dependency and is drivable end-to-end by `NumpyFlatAdapter` in tests.

Every cuVS import is inside a method body, never at module scope. This module
must import on a CPU-only box with no cuVS, because `make check` runs there.

All distances are squared L2 ("sqeuclidean"). Corpora are L2-normalized, where
that is monotone in cosine, so the ordering is the one the project's `angular`
metric would give.

Argument order matches the measured cuVS 26.08.01 API, not the guessed one:
`ivf_flat.search`/`ivf_pq.search`/`cagra.search` take `(search_params, index,
queries, k, ...)` -- params before the index -- while `brute_force.search`
takes `(index, queries, k, ...)`, index first. `brute_force.build` takes the
metric directly; the other three take it on `IndexParams`. See
`docs/superpowers/plans/2026-08-12-ann-gpu-benchmark-probe.md`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

METRIC = "sqeuclidean"

IVF_N_LISTS = 4096
IVF_N_PROBES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PQ_DIM = 64
PQ_BITS = 8
CAGRA_GRAPH_DEGREE = 64
CAGRA_INTERMEDIATE_GRAPH_DEGREE = 128
CAGRA_ITOPK_SIZE = (32, 64, 128, 256, 512)

INSTALL_HINT = (
    "cuVS is not installed. It is deliberately absent from requirements.txt: "
    "it is CUDA-13-only and would break the CPU-only install CI runs. On the "
    "GPU box install it with:\n"
    "    pip install cuvs-cu13 cupy-cuda13x --extra-index-url "
    "https://pypi.nvidia.com"
)


@dataclass(frozen=True)
class BuiltIndex:
    """One built index plus what building it cost.

    `dataset` holds the device array the index was built from, for cuVS
    adapters. cuVS does not copy or take ownership of the dataset passed to
    `build()` -- the index stores a pointer into that buffer. If nothing
    outside the index keeps a Python reference to it, cupy's refcounting GC
    frees the block as soon as `build()` returns, the allocator hands that
    memory to the next allocation, and every later `search()` reads whatever
    now lives there: no exception, just silently wrong, plausible-looking
    results. Keeping `dataset` here for the `BuiltIndex`'s whole lifetime is
    what prevents that. `None` for `NumpyFlatAdapter`, which owns its data on
    the host and has no such lifetime hazard.
    """

    handle: object
    train_seconds: float
    add_seconds: float
    index_bytes: int
    peak_vram_bytes: int | None = None
    dataset: object | None = None


def require_device_stack() -> None:
    """Preflight check: raise unless cuVS and cupy are importable.

    Called by the CLI before any corpus is materialized. Without it the first
    failure would land after seven 1M draws and seven exact-kNN passes -- most
    of an hour spent to discover a missing pip install.
    """
    _require_cuvs()


def _require_cuvs():
    """Import cuVS, or raise with the command that installs it."""
    try:
        import cupy
        from cuvs.common import Resources
    except ImportError as exc:  # pragma: no cover - box-side path
        raise RuntimeError(f"{INSTALL_HINT}\n\noriginal error: {exc}") from exc
    return cupy, Resources


class IndexAdapter:
    """Interface every index in the grid presents to the runner.

    `sync()` is the fence. cuVS calls are asynchronous, so the runner calls it
    immediately before starting a clock and immediately before stopping one;
    without it every timing measures the launch queue instead of the work.

    Timing itself -- including the discarded warmup search that precedes each
    cell's timed repeats -- is owned by the runner, not by this class: `search`
    just runs one query batch and returns, so calling it once and discarding
    the result before the timed loop is exactly a normal call.
    """

    name: str = ""
    param_name: str = ""

    def sweep_params(self) -> tuple[int | None, ...]:
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        raise NotImplementedError

    def build(self, vectors: np.ndarray) -> BuiltIndex:
        raise NotImplementedError

    def search(
        self,
        built: BuiltIndex,
        queries: np.ndarray,
        k: int,
        param: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def sync(self) -> None:
        raise NotImplementedError


class _CuvsAdapter(IndexAdapter):
    """Shared cuVS plumbing: resources, host/device transfer, fencing."""

    def __init__(self) -> None:
        self._resources = None
        self._cupy = None

    def _res(self):
        if self._resources is None:
            self._cupy, resources_cls = _require_cuvs()
            self._resources = resources_cls()
        return self._resources

    def sync(self) -> None:
        self._res().sync()

    def _device_used_bytes(self) -> int:
        """Device memory in use, right now, across the whole card.

        `torch.cuda.max_memory_allocated` cannot see this: cuVS allocates
        through RMM, not through torch's caching allocator, so torch's counter
        reads near zero while an index is holding gigabytes. Asking the driver
        is the only figure that covers both. It is card-wide rather than
        process-local, which is why the runner reports the *delta* across a
        build rather than the absolute value.
        """
        self._res()
        free, total = self._cupy.cuda.runtime.memGetInfo()
        return int(total - free)

    def _to_device(self, x: np.ndarray):
        self._res()
        return self._cupy.asarray(np.ascontiguousarray(x, dtype=np.float32))

    def _to_host(self, x) -> np.ndarray:
        return self._cupy.asnumpy(x)


class FlatAdapter(_CuvsAdapter):
    """Exact GPU brute force: the recall-1.0 ceiling, and the ground truth."""

    name = "flat"
    param_name = ""

    def sweep_params(self) -> tuple[int | None, ...]:
        return (None,)

    def describe(self) -> dict[str, object]:
        return {"metric": METRIC}

    def build(self, vectors: np.ndarray) -> BuiltIndex:
        # `_device_used_bytes` calls `_res()`, which raises the friendly
        # RuntimeError if cuVS is missing -- so it must run before the
        # `cuvs.neighbors` import, or a CPU-only box would see a bare
        # ModuleNotFoundError instead.
        before = self._device_used_bytes()
        from cuvs.neighbors import brute_force

        device_vectors = self._to_device(vectors)
        self.sync()
        started = time.perf_counter()
        handle = brute_force.build(device_vectors, metric=METRIC)
        self.sync()
        elapsed = time.perf_counter() - started
        # Brute force has no training phase; the whole cost is ingesting the
        # vectors, which is reported as `add` so the two IVF indexes' train
        # column stays meaningful against it.
        return BuiltIndex(
            handle=handle,
            train_seconds=0.0,
            add_seconds=elapsed,
            index_bytes=int(vectors.nbytes),
            peak_vram_bytes=max(self._device_used_bytes() - before, 0),
            # See BuiltIndex.dataset: the handle points into this buffer.
            dataset=device_vectors,
        )

    def search(self, built, queries, k, param):
        self._res()
        from cuvs.neighbors import brute_force

        device_queries = self._to_device(queries)
        # brute_force.search takes the index first, unlike the other three.
        distances, neighbours = brute_force.search(built.handle, device_queries, k)
        return self._to_host(distances), self._to_host(neighbours)


class IvfFlatAdapter(_CuvsAdapter):
    name = "ivf_flat"
    param_name = "n_probes"

    def sweep_params(self) -> tuple[int | None, ...]:
        return IVF_N_PROBES

    def describe(self) -> dict[str, object]:
        return {"n_lists": IVF_N_LISTS, "metric": METRIC}

    def build(self, vectors: np.ndarray) -> BuiltIndex:
        before = self._device_used_bytes()
        from cuvs.neighbors import ivf_flat

        device_vectors = self._to_device(vectors)
        params = ivf_flat.IndexParams(n_lists=IVF_N_LISTS, metric=METRIC)
        self.sync()
        started = time.perf_counter()
        handle = ivf_flat.build(params, device_vectors)
        self.sync()
        elapsed = time.perf_counter() - started
        # cuVS builds the coarse quantizer and adds the vectors in one call,
        # so the split cannot be observed from here; the whole cost is
        # reported as `train` and `add` is zero.
        return BuiltIndex(
            handle=handle,
            train_seconds=elapsed,
            add_seconds=0.0,
            index_bytes=int(vectors.nbytes),
            peak_vram_bytes=max(self._device_used_bytes() - before, 0),
            # See BuiltIndex.dataset: the handle points into this buffer.
            dataset=device_vectors,
        )

    def search(self, built, queries, k, param):
        self._res()
        from cuvs.neighbors import ivf_flat

        device_queries = self._to_device(queries)
        search_params = ivf_flat.SearchParams(n_probes=int(param))
        # search takes (search_params, index, queries, k, ...).
        distances, neighbours = ivf_flat.search(
            search_params, built.handle, device_queries, k
        )
        return self._to_host(distances), self._to_host(neighbours)


class IvfPqAdapter(_CuvsAdapter):
    name = "ivf_pq"
    param_name = "n_probes"

    def sweep_params(self) -> tuple[int | None, ...]:
        return IVF_N_PROBES

    def describe(self) -> dict[str, object]:
        return {
            "n_lists": IVF_N_LISTS,
            "pq_dim": PQ_DIM,
            "pq_bits": PQ_BITS,
            "metric": METRIC,
        }

    def build(self, vectors: np.ndarray) -> BuiltIndex:
        before = self._device_used_bytes()
        from cuvs.neighbors import ivf_pq

        device_vectors = self._to_device(vectors)
        params = ivf_pq.IndexParams(
            n_lists=IVF_N_LISTS,
            pq_dim=PQ_DIM,
            pq_bits=PQ_BITS,
            metric=METRIC,
        )
        self.sync()
        started = time.perf_counter()
        handle = ivf_pq.build(params, device_vectors)
        self.sync()
        elapsed = time.perf_counter() - started
        # Compressed: one PQ_BITS-bit code per PQ_DIM subspace per vector.
        codes = vectors.shape[0] * PQ_DIM * PQ_BITS // 8
        return BuiltIndex(
            handle=handle,
            train_seconds=elapsed,
            add_seconds=0.0,
            index_bytes=int(codes),
            peak_vram_bytes=max(self._device_used_bytes() - before, 0),
            # See BuiltIndex.dataset: the handle points into this buffer.
            dataset=device_vectors,
        )

    def search(self, built, queries, k, param):
        self._res()
        from cuvs.neighbors import ivf_pq

        device_queries = self._to_device(queries)
        search_params = ivf_pq.SearchParams(n_probes=int(param))
        # search takes (search_params, index, queries, k, ...).
        distances, neighbours = ivf_pq.search(
            search_params, built.handle, device_queries, k
        )
        return self._to_host(distances), self._to_host(neighbours)


class CagraAdapter(_CuvsAdapter):
    name = "cagra"
    param_name = "itopk_size"

    def sweep_params(self) -> tuple[int | None, ...]:
        return CAGRA_ITOPK_SIZE

    def describe(self) -> dict[str, object]:
        return {
            "graph_degree": CAGRA_GRAPH_DEGREE,
            "intermediate_graph_degree": CAGRA_INTERMEDIATE_GRAPH_DEGREE,
            "metric": METRIC,
        }

    def build(self, vectors: np.ndarray) -> BuiltIndex:
        before = self._device_used_bytes()
        from cuvs.neighbors import cagra

        device_vectors = self._to_device(vectors)
        params = cagra.IndexParams(
            graph_degree=CAGRA_GRAPH_DEGREE,
            intermediate_graph_degree=CAGRA_INTERMEDIATE_GRAPH_DEGREE,
            metric=METRIC,
        )
        self.sync()
        started = time.perf_counter()
        handle = cagra.build(params, device_vectors)
        self.sync()
        elapsed = time.perf_counter() - started
        # Vectors plus a graph_degree-wide uint32 adjacency row per vector.
        graph = vectors.shape[0] * CAGRA_GRAPH_DEGREE * 4
        return BuiltIndex(
            handle=handle,
            train_seconds=elapsed,
            add_seconds=0.0,
            index_bytes=int(vectors.nbytes + graph),
            peak_vram_bytes=max(self._device_used_bytes() - before, 0),
            # See BuiltIndex.dataset: the handle points into this buffer.
            dataset=device_vectors,
        )

    def search(self, built, queries, k, param):
        self._res()
        from cuvs.neighbors import cagra

        device_queries = self._to_device(queries)
        search_params = cagra.SearchParams(itopk_size=int(param))
        # search takes (search_params, index, queries, k, ...).
        distances, neighbours = cagra.search(
            search_params, built.handle, device_queries, k
        )
        return self._to_host(distances), self._to_host(neighbours)


class NumpyFlatAdapter(IndexAdapter):
    """Exact brute force in numpy -- the runner's stand-in under pytest.

    Lives beside the real adapters rather than in the test file so both sides
    of the boundary are defined in one place: if the interface changes, this
    breaks in the same commit.
    """

    name = "numpy_flat"
    param_name = ""

    def sweep_params(self) -> tuple[int | None, ...]:
        return (None,)

    def describe(self) -> dict[str, object]:
        return {"metric": METRIC}

    def build(self, vectors: np.ndarray) -> BuiltIndex:
        started = time.perf_counter()
        stored = np.ascontiguousarray(vectors, dtype=np.float32)
        return BuiltIndex(
            handle=stored,
            train_seconds=time.perf_counter() - started,
            add_seconds=0.0,
            index_bytes=int(stored.nbytes),
        )

    def search(self, built, queries, k, param):
        stored = built.handle
        diff = queries[:, None, :] - stored[None, :, :]
        squared = np.einsum("qnd,qnd->qn", diff, diff)
        order = np.argsort(squared, axis=1, kind="stable")[:, :k]
        rows = np.arange(queries.shape[0])[:, None]
        return squared[rows, order].astype(np.float32), order.astype(np.int64)

    def sync(self) -> None:
        return None


_ADAPTERS: dict[str, type[IndexAdapter]] = {
    "flat": FlatAdapter,
    "ivf_flat": IvfFlatAdapter,
    "ivf_pq": IvfPqAdapter,
    "cagra": CagraAdapter,
}

ADAPTER_NAMES: tuple[str, ...] = tuple(_ADAPTERS)


def build_adapters(names: Sequence[str]) -> tuple[IndexAdapter, ...]:
    """Instantiate adapters by name, preserving the caller's order."""
    unknown = [n for n in names if n not in _ADAPTERS]
    if unknown:
        raise ValueError(
            f"unknown index name(s): {', '.join(unknown)}. "
            f"Known: {', '.join(ADAPTER_NAMES)}"
        )
    return tuple(_ADAPTERS[n]() for n in names)
