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

# Tiling for the torch brute-force baselines. The score tile is
# query_chunk x corpus_tile float32 = 537 MB at these values, which is what
# makes the pair a real choice: the full 10,000 x 1,000,000 distance matrix
# would be 40 GB, five times the card, so brute force here is necessarily a
# fused scan rather than one matmul. Held as constants so the box run and the
# CPU tests exercise the same code path at different sizes.
TORCH_QUERY_CHUNK = 2048
TORCH_CORPUS_TILE = 65536

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
    # An analytic estimate from the vectors/codes/graph an adapter *knows* it
    # allocated, not a measured device allocation -- cuVS exposes no API to
    # ask an index its real footprint. It omits whatever structure each
    # adapter's `build()` doesn't account for (IVF's coarse-quantizer
    # centroids, PQ's codebook tables); see the comment at each computation
    # for exactly what is and is not counted. `peak_vram_bytes` below is the
    # measured figure and does not have this gap, but it is card-wide rather
    # than per-index (see `_device_used_bytes`).
    index_bytes_estimated: int
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
            # Brute force has no structure beyond the vectors themselves, so
            # this estimate has nothing to omit.
            index_bytes_estimated=int(vectors.nbytes),
            peak_vram_bytes=max(self._device_used_bytes() - before, 0),
            # See BuiltIndex.dataset: the handle points into this buffer.
            dataset=device_vectors,
        )

    def search(self, built, queries, k, param):
        # `_res()` looks redundant here -- `build()` already created the
        # resources handle -- but it is not: it is what populates
        # `self._cupy` before `_to_device` below reads it, and it is what
        # turns a missing cuVS into `INSTALL_HINT`'s friendly RuntimeError
        # rather than a bare ModuleNotFoundError from the import beneath it.
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
            # Estimate: the raw vectors only. Excludes the n_lists coarse-
            # quantizer centroids and cuVS's inverted-list bookkeeping --
            # both real device allocations this does not measure.
            index_bytes_estimated=int(vectors.nbytes),
            peak_vram_bytes=max(self._device_used_bytes() - before, 0),
            # See BuiltIndex.dataset: the handle points into this buffer.
            dataset=device_vectors,
        )

    def search(self, built, queries, k, param):
        # See FlatAdapter.search: not redundant with build()'s resources --
        # it populates self._cupy before _to_device below reads it, and it
        # is what turns a missing cuVS into INSTALL_HINT's RuntimeError.
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
        # Estimate: the codes only. Excludes the PQ codebook tables (one
        # per subspace) and the n_lists coarse-quantizer centroids -- both
        # real device allocations this does not measure.
        codes = vectors.shape[0] * PQ_DIM * PQ_BITS // 8
        return BuiltIndex(
            handle=handle,
            train_seconds=elapsed,
            add_seconds=0.0,
            index_bytes_estimated=int(codes),
            peak_vram_bytes=max(self._device_used_bytes() - before, 0),
            # See BuiltIndex.dataset: the handle points into this buffer.
            dataset=device_vectors,
        )

    def search(self, built, queries, k, param):
        # See FlatAdapter.search: not redundant with build()'s resources --
        # it populates self._cupy before _to_device below reads it, and it
        # is what turns a missing cuVS into INSTALL_HINT's RuntimeError.
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
        # Estimate: vectors plus a graph_degree-wide uint32 adjacency row per
        # vector. No coarse quantizer or codebook to omit here, unlike the
        # two IVF adapters, but it is still an analytic figure, not a
        # measured allocation.
        graph = vectors.shape[0] * CAGRA_GRAPH_DEGREE * 4
        return BuiltIndex(
            handle=handle,
            train_seconds=elapsed,
            add_seconds=0.0,
            index_bytes_estimated=int(vectors.nbytes + graph),
            peak_vram_bytes=max(self._device_used_bytes() - before, 0),
            # See BuiltIndex.dataset: the handle points into this buffer.
            dataset=device_vectors,
        )

    def search(self, built, queries, k, param):
        # See FlatAdapter.search: not redundant with build()'s resources --
        # it populates self._cupy before _to_device below reads it, and it
        # is what turns a missing cuVS into INSTALL_HINT's RuntimeError.
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
            index_bytes_estimated=int(stored.nbytes),
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


class _TorchFlatAdapter(IndexAdapter):
    """Exact brute force in torch: tiled matmul plus top-k.

    A second opinion on `FlatAdapter`, not a replacement. The published run
    measured cuVS brute force at 7,996 QPS on 1M x 128, which is ~14% of the
    card's FP32 peak; decomposing that batch shows only ~170 ms of its
    1,251 ms is the distance GEMM, so roughly 86% is top-k selection over
    10^10 candidate distances. That makes "is brute force actually optimized
    here?" a measurable question rather than a rhetorical one, and it decides
    how much of ANN's advantage at 1M scale is real.

    Three precisions are registered so the answer separates two causes: a
    different tiling strategy (`torch_flat`, FP32, TF32 explicitly off) from
    the tensor cores cuVS may not be using (`torch_flat_tf32`,
    `torch_flat_fp16`). Reduced precision is scored by the same recall path
    as every other index, so if it costs exactness the grid reports that as a
    recall below 1.0 rather than hiding it in a faster number.

    Unlike the cuVS adapters this runs under `make check`: torch is in
    requirements.txt and installs CPU-only, so `device="cpu"` with small
    tiles exercises the real tiling and merge logic in pytest.
    """

    param_name = ""
    precision = "fp32"
    allow_tf32 = False
    compute_dtype_name = "float32"

    def __init__(
        self,
        *,
        device: str | None = None,
        query_chunk: int = TORCH_QUERY_CHUNK,
        corpus_tile: int = TORCH_CORPUS_TILE,
    ) -> None:
        self._device_override = device
        self._query_chunk = int(query_chunk)
        self._corpus_tile = int(corpus_tile)
        self._torch_module = None

    def _torch(self):
        """Import torch lazily, mirroring how cuVS is kept off module scope."""
        if self._torch_module is None:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - torch is pinned
                raise RuntimeError(
                    "torch is required for the brute-force baselines; it is "
                    "pinned in requirements.txt"
                ) from exc
            self._torch_module = torch
        return self._torch_module

    def _device(self):
        torch = self._torch()
        if self._device_override is not None:
            return torch.device(self._device_override)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _apply_backend_flags(self) -> None:
        """Pin the TF32 setting this adapter is named for.

        Set on every build and search rather than once at construction: the
        flag is global process state, so an adapter that assumed its value
        would silently inherit whichever torch adapter ran before it and
        report one precision's throughput under another's name.
        """
        torch = self._torch()
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32

    def sweep_params(self) -> tuple[int | None, ...]:
        return (None,)

    def describe(self) -> dict[str, object]:
        return {
            "metric": METRIC,
            "precision": self.precision,
            "query_chunk": self._query_chunk,
            "corpus_tile": self._corpus_tile,
        }

    def _device_used_bytes(self, device) -> int | None:
        torch = self._torch()
        if device.type != "cuda":
            return None
        free, total = torch.cuda.mem_get_info()
        return int(total - free)

    def build(self, vectors: np.ndarray) -> BuiltIndex:
        torch = self._torch()
        device = self._device()
        self._apply_backend_flags()
        before = self._device_used_bytes(device)

        self.sync()
        started = time.perf_counter()
        stored = torch.as_tensor(
            np.ascontiguousarray(vectors, dtype=np.float32), device=device
        )
        # Squared norms stay FP32 whatever the compute dtype. They are a
        # per-vector constant added once per distance, so computing them in
        # half precision would spend accuracy on every distance in the grid
        # and buy no throughput at all.
        norms = (stored.float() ** 2).sum(dim=1)
        compute_dtype = getattr(torch, self.compute_dtype_name)
        if stored.dtype == compute_dtype:
            compute = stored
        else:
            compute = stored.to(compute_dtype)
            # Drop the FP32 copy: at 1M x 128 it is 512 MB of card that the
            # half-precision path exists to avoid spending.
            del stored
        self.sync()
        elapsed = time.perf_counter() - started

        after = self._device_used_bytes(device)
        peak = None if before is None or after is None else max(after - before, 0)
        return BuiltIndex(
            handle={"vectors": compute, "norms": norms},
            # No training phase, as with cuVS brute force: the whole cost is
            # ingesting the vectors, reported as `add` so the IVF indexes'
            # train column stays comparable.
            train_seconds=0.0,
            add_seconds=elapsed,
            index_bytes_estimated=int(
                compute.element_size() * compute.numel()
                + norms.element_size() * norms.numel()
            ),
            peak_vram_bytes=peak,
            dataset=compute,
        )

    def search(self, built, queries, k, param):
        torch = self._torch()
        self._apply_backend_flags()
        stored = built.handle["vectors"]
        norms = built.handle["norms"]
        device = stored.device

        host_queries = np.ascontiguousarray(queries, dtype=np.float32)
        all_q = torch.as_tensor(host_queries, device=device)
        query_norms = (all_q**2).sum(dim=1)
        compute_q = all_q.to(stored.dtype)

        num_queries = all_q.shape[0]
        num_vectors = stored.shape[0]
        out_distances = torch.empty(
            (num_queries, k), dtype=torch.float32, device=device
        )
        out_ids = torch.empty((num_queries, k), dtype=torch.int64, device=device)

        for q_start in range(0, num_queries, self._query_chunk):
            q_end = min(q_start + self._query_chunk, num_queries)
            best_distances = None
            best_ids = None
            for c_start in range(0, num_vectors, self._corpus_tile):
                c_end = min(c_start + self._corpus_tile, num_vectors)
                # ||q||^2 is a per-query constant, so it is left out of the
                # ranking and added back at the end -- it cannot change the
                # ordering within a query, and omitting it keeps the tile in
                # one fused expression.
                scores = compute_q[q_start:q_end] @ stored[c_start:c_end].T
                scores = scores.float().mul_(-2.0).add_(norms[c_start:c_end])

                # A tile can only offer as many candidates as it holds, so k
                # above the tile size has to accumulate across tiles rather
                # than truncate at the first one's supply.
                take = min(k, c_end - c_start)
                tile_distances, tile_ids = torch.topk(
                    scores, take, dim=1, largest=False, sorted=True
                )
                tile_ids = tile_ids + c_start
                if best_distances is None:
                    best_distances, best_ids = tile_distances, tile_ids
                    continue
                merged_distances = torch.cat([best_distances, tile_distances], dim=1)
                merged_ids = torch.cat([best_ids, tile_ids], dim=1)
                keep = min(k, merged_distances.shape[1])
                best_distances, order = torch.topk(
                    merged_distances, keep, dim=1, largest=False, sorted=True
                )
                best_ids = torch.gather(merged_ids, 1, order)

            out_distances[q_start:q_end] = best_distances + (
                query_norms[q_start:q_end].unsqueeze(1)
            )
            out_ids[q_start:q_end] = best_ids

        return (
            out_distances.cpu().numpy().astype(np.float32),
            out_ids.cpu().numpy().astype(np.int64),
        )

    def sync(self) -> None:
        torch = self._torch()
        if self._device().type == "cuda":
            torch.cuda.synchronize()


class TorchFlatAdapter(_TorchFlatAdapter):
    """FP32, TF32 off: the like-for-like comparison against cuVS `flat`."""

    name = "torch_flat"
    precision = "fp32"
    allow_tf32 = False
    compute_dtype_name = "float32"


class TorchFlatTf32Adapter(_TorchFlatAdapter):
    """FP32 inputs through TF32 tensor cores."""

    name = "torch_flat_tf32"
    precision = "tf32"
    allow_tf32 = True
    compute_dtype_name = "float32"


class TorchFlatFp16Adapter(_TorchFlatAdapter):
    """FP16 storage and matmul, FP32 accumulate and FP32 norms."""

    name = "torch_flat_fp16"
    precision = "fp16"
    allow_tf32 = False
    compute_dtype_name = "float16"


_ADAPTERS: dict[str, type[IndexAdapter]] = {
    "flat": FlatAdapter,
    "ivf_flat": IvfFlatAdapter,
    "ivf_pq": IvfPqAdapter,
    "cagra": CagraAdapter,
    "torch_flat": TorchFlatAdapter,
    "torch_flat_tf32": TorchFlatTf32Adapter,
    "torch_flat_fp16": TorchFlatFp16Adapter,
}

ADAPTER_NAMES: tuple[str, ...] = tuple(_ADAPTERS)

# The grid the published artifact reports. The torch baselines are opt-in via
# `--indexes`: they answer whether cuVS brute force is near the hardware's
# limit, which is a different question from how the variant ladder searches,
# and adding a probe should not silently change the shipped table's shape.
DEFAULT_INDEX_NAMES: tuple[str, ...] = ("flat", "ivf_flat", "ivf_pq", "cagra")


def adapter_class(name: str) -> type[IndexAdapter]:
    """The adapter class registered under `name`, for callers needing kwargs."""
    if name not in _ADAPTERS:
        raise ValueError(
            f"unknown index name: {name}. Known: {', '.join(ADAPTER_NAMES)}"
        )
    return _ADAPTERS[name]


def build_adapters(names: Sequence[str]) -> tuple[IndexAdapter, ...]:
    """Instantiate adapters by name, preserving the caller's order."""
    unknown = [n for n in names if n not in _ADAPTERS]
    if unknown:
        raise ValueError(
            f"unknown index name(s): {', '.join(unknown)}. "
            f"Known: {', '.join(ADAPTER_NAMES)}"
        )
    return tuple(_ADAPTERS[n]() for n in names)
