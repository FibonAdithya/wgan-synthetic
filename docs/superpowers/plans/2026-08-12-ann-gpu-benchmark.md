> **AI-generated working note.** Written by Claude during development and kept
> for provenance. It is not the source of truth: where this file disagrees with
> `PROJECT_DOCUMENTATION.md`, the latter wins.

# GPU ANN-Algorithm Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure index build time, recall@10 and QPS for real SIFT and each trained SIFT variant, using GPU ANN indexes, and publish the table.

**Architecture:** A new package `src/eval/ann_benchmark/` with a hard boundary at `indexes.py`: every cuVS call lives behind an adapter interface, so `runner.py` never names a cuVS type and the whole grid loop is drivable by a fake adapter on a CPU-only box. Corpora, query sets and ground truth are materialized once to a work directory and cached, because that half of the job is deterministic and expensive.

**Tech Stack:** Python 3.12, numpy, torch (generator sampling only), h5py (real query set), plotly (HTML report), cuVS (GPU indexes, box-side only).

**Design spec:** `docs/superpowers/specs/2026-08-12-ann-gpu-benchmark-design.md`

## Global Constraints

- **Python 3.12.** `make check` (ruff lint, ruff format check, pytest) must stay green after every task.
- **`make check` is CPU-only and dataset-free.** No test may require a GPU, cuVS, `data/`, or `runs/`.
- **cuVS is NOT added to `requirements.txt`.** It is CUDA-12-only and installs from NVIDIA's index; pinning it breaks the CPU-only install CI runs. Every module that touches cuVS must import it *inside a function*, never at module scope.
- **Ruff config:** `select = ["E", "F", "I", "W", "UP"]`, `ignore = ["E501"]`, `line-length = 88`, `known-first-party = ["src"]`. Run `make format` only on files you touched — never repo-wide.
- **Search metric is squared L2** (`"sqeuclidean"` in cuVS). All corpora are L2-normalized, where squared L2 is monotone in cosine. Never mix squared and unsquared distances in one comparison.
- **k = 10** for both ground truth and every search.
- **Target recall for the headline number is 0.90.** A curve that never reaches it reports `None`, never an extrapolation.
- **Every timed GPU region is fenced** with an adapter `sync()` call before the clock starts and before it stops. An unfenced region times the launch queue, not the work.
- Commit after every task. Branch is `benchmark-algos`; do not commit to `main`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/eval/ann_benchmark/__init__.py` | Package marker; re-exports nothing (keeps cuVS out of import paths). |
| `src/eval/ann_benchmark/metrics.py` | `recall_at_k`, `qps`, `summarize`, `qps_at_recall`. Pure numpy, no device, no I/O. |
| `src/eval/ann_benchmark/indexes.py` | `BuiltIndex`, the adapter interface, four cuVS adapters, `build_adapters`. Sole cuVS boundary for indexing. |
| `src/eval/ann_benchmark/groundtruth.py` | `exact_neighbours` — GPU brute force with a numpy fallback. |
| `src/eval/ann_benchmark/corpora.py` | `Corpus`, `materialize_corpora` — load/draw, L2-normalize, cache. |
| `src/eval/ann_benchmark/runner.py` | `BuildRecord`, `SearchRecord`, `run_grid` — the loop, all timing, failed-cell handling, incremental JSON. |
| `src/eval/ann_benchmark/report.py` | `write_json`, `write_markdown`, `write_html`. |
| `src/eval/ann_benchmark/cli.py` | `python -m src.eval.ann_benchmark` entry point. |
| `src/eval/ann_benchmark/__main__.py` | Delegates to `cli.main`. |
| `tests/test_ann_benchmark_metrics.py` | Task 2 |
| `tests/test_ann_benchmark_indexes.py` | Task 3 |
| `tests/test_ann_benchmark_corpora.py` | Task 5 |
| `tests/test_ann_benchmark_runner.py` | Task 6 |
| `tests/test_ann_benchmark_report.py` | Task 7 |

---

### Task 1: Probe the box before writing code against a guessed API

This task writes no product code. It exists because tasks 3, 4 and 8 are written against a cuVS API surface and a set of run directories that have not been confirmed to exist. Confirming them costs minutes; discovering them wrong after six modules are written costs a day.

**Files:**
- Create: `docs/superpowers/plans/2026-08-12-ann-gpu-benchmark-probe.md`

- [ ] **Step 1: Check SSH reaches the box at all**

Per the environment notes, outbound SSH is blocked except on port 443, and the sandbox must be disabled for `ssh`.

Run: `ssh -T git@github.com` first to confirm the transport, then `ssh tig-gpu 'hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv'`

Expected: a GPU name, total memory and driver version. If the driver is CUDA 11, stop and report — cuVS needs CUDA 12 and the whole plan changes.

- [ ] **Step 2: Confirm the corpora and checkpoints are on the box**

Run:
```bash
ssh tig-gpu 'cd ~/wgan-synthetic && ls -la data/sift_1m.npy data/cache/*.hdf5 && for d in runs/long_baseline runs/x100k_ema_only runs/x100k_improved runs/x100k_sparse_clamp4 runs/sift_gan_v3 runs/x100k_structured; do printf "%s: " "$d"; ls "$d/best_generator.pt" 2>/dev/null || echo MISSING; done'
```

Expected: `data/sift_1m.npy` present, one `.hdf5` in `data/cache/`, and a verdict per run directory. Record which of `v3` and `v4` actually exist — that decides whether the table has 5, 6 or 7 rows.

- [ ] **Step 3: Confirm the real query set is in the HDF5**

Run:
```bash
ssh tig-gpu 'cd ~/wgan-synthetic && python -c "
import h5py, glob
p = glob.glob(\"data/cache/*.hdf5\")[0]
with h5py.File(p) as f:
    print(p, {k: f[k].shape for k in f.keys()})
"'
```

Expected: a `test` key of shape `(10000, 128)`. The plan reads queries from it; if it is absent, Task 5 falls back to a held-out slice of `sift_1m.npy` and the spec's query protocol must be amended.

- [ ] **Step 4: Install cuVS and record the real API surface**

Run:
```bash
ssh tig-gpu 'cd ~/wgan-synthetic && pip install cuvs-cu12 cupy-cuda12x --extra-index-url https://pypi.nvidia.com 2>&1 | tail -5'
ssh tig-gpu 'cd ~/wgan-synthetic && python -c "
import cuvs, inspect
from cuvs.neighbors import cagra, ivf_flat, ivf_pq, brute_force
print(\"cuvs\", cuvs.__version__)
for m, n in ((ivf_flat,\"ivf_flat\"),(ivf_pq,\"ivf_pq\"),(cagra,\"cagra\")):
    print(n, \"IndexParams\", inspect.signature(m.IndexParams.__init__))
    print(n, \"SearchParams\", inspect.signature(m.SearchParams.__init__))
    print(n, \"build\", inspect.signature(m.build))
    print(n, \"search\", inspect.signature(m.search))
print(\"brute_force.build\", inspect.signature(brute_force.build))
print(\"brute_force.search\", inspect.signature(brute_force.search))
from cuvs.common import Resources
print(\"Resources\", [x for x in dir(Resources) if not x.startswith(\"_\")])
"'
```

Expected: real signatures. **Task 3 is written against the signatures this step prints, not against the ones assumed in this plan.** In particular confirm the parameter names `n_lists`, `n_probes`, `pq_dim`, `pq_bits`, `graph_degree`, `intermediate_graph_degree`, `itopk_size`, the `metric="sqeuclidean"` spelling, and how a stream is synchronized (`Resources().sync()` or equivalent).

- [ ] **Step 5: Write the findings down and commit**

Write `docs/superpowers/plans/2026-08-12-ann-gpu-benchmark-probe.md` recording, verbatim: GPU model and driver, which run directories exist, the HDF5 key shapes, the cuVS version, and the printed signatures. Then:

```bash
git add docs/superpowers/plans/2026-08-12-ann-gpu-benchmark-probe.md
git commit -m "docs: record GPU box probe for the ANN benchmark"
```

- [ ] **Step 6: Stop and report any mismatch**

If `v3`/`v4` runs are missing, or the cuVS signatures differ from this plan's assumptions, report both before starting Task 2. Adjust Tasks 3 and 5 to match reality rather than adjusting reality to match the plan.

---

### Task 2: `metrics.py` — recall, QPS and interpolation

**Files:**
- Create: `src/eval/ann_benchmark/__init__.py`, `src/eval/ann_benchmark/metrics.py`
- Test: `tests/test_ann_benchmark_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `recall_at_k(found_distances: np.ndarray, truth_distances: np.ndarray, *, eps: float = 1e-6) -> float`
  - `qps(num_queries: int, seconds: float) -> float`
  - `summarize(values: Sequence[float]) -> dict[str, float]` → keys `min`, `median`, `p95`
  - `qps_at_recall(points: Sequence[tuple[float, float]], target: float) -> float | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ann_benchmark_metrics.py`:

```python
"""Unit tests for the ANN-benchmark scoring helpers.

Recall here is distance-based rather than id-based, and these tests pin that
down: SIFT descriptors sit on a lattice where exact ties are common, so an
index returning a different-but-equidistant neighbour has not missed anything.
"""

import numpy as np
import pytest

from src.eval.ann_benchmark import metrics


def test_recall_is_one_when_distances_match_ground_truth():
    truth = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    found = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert metrics.recall_at_k(found, truth) == pytest.approx(1.0)


def test_recall_counts_ties_as_hits():
    # Every true neighbour sits at distance 5.0. An index returning three
    # different points that are also at 5.0 has missed nothing, even though
    # not one id matches.
    truth = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    found = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    assert metrics.recall_at_k(found, truth) == pytest.approx(1.0)


def test_recall_is_fraction_within_the_kth_true_distance():
    truth = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    found = np.array([[1.0, 2.0, 9.0]], dtype=np.float32)
    assert metrics.recall_at_k(found, truth) == pytest.approx(2.0 / 3.0)


def test_recall_averages_over_queries():
    truth = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
    found = np.array([[1.0, 2.0], [1.0, 9.0]], dtype=np.float32)
    assert metrics.recall_at_k(found, truth) == pytest.approx(0.75)


def test_recall_rejects_mismatched_shapes():
    truth = np.zeros((2, 3), dtype=np.float32)
    found = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        metrics.recall_at_k(found, truth)


def test_qps_is_queries_over_seconds():
    assert metrics.qps(1000, 0.5) == pytest.approx(2000.0)


def test_qps_rejects_non_positive_time():
    with pytest.raises(ValueError, match="positive"):
        metrics.qps(1000, 0.0)


def test_summarize_reports_min_median_p95():
    out = metrics.summarize([1.0, 2.0, 3.0, 4.0])
    assert out["min"] == pytest.approx(1.0)
    assert out["median"] == pytest.approx(2.5)
    assert set(out) == {"min", "median", "p95"}


def test_summarize_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        metrics.summarize([])


def test_qps_at_recall_interpolates_between_bracketing_points():
    # Geometric midpoint of 100 and 400 is 200, because the interpolation is
    # linear in log(qps) -- QPS spans orders of magnitude across a sweep.
    points = [(0.80, 400.0), (0.95, 100.0)]
    got = metrics.qps_at_recall(points, 0.875)
    assert got == pytest.approx(200.0)


def test_qps_at_recall_returns_none_when_target_unreachable():
    points = [(0.10, 900.0), (0.55, 300.0)]
    assert metrics.qps_at_recall(points, 0.90) is None


def test_qps_at_recall_returns_fastest_point_when_all_exceed_target():
    points = [(0.95, 500.0), (0.99, 100.0)]
    assert metrics.qps_at_recall(points, 0.90) == pytest.approx(500.0)


def test_qps_at_recall_returns_none_for_empty_curve():
    assert metrics.qps_at_recall([], 0.90) is None


def test_qps_at_recall_is_order_independent():
    ascending = [(0.80, 400.0), (0.95, 100.0)]
    descending = list(reversed(ascending))
    assert metrics.qps_at_recall(descending, 0.875) == pytest.approx(
        metrics.qps_at_recall(ascending, 0.875)
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ann_benchmark_metrics.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'src.eval.ann_benchmark'`.

- [ ] **Step 3: Write the implementation**

Create `src/eval/ann_benchmark/__init__.py`:

```python
"""GPU ANN-algorithm benchmark: index build time, recall and QPS.

Deliberately exports nothing. Every cuVS import in this package is inside a
function body, and re-exporting from here would undo that by dragging the
device modules into any `import src.eval.ann_benchmark` on a CPU-only box.
"""
```

Create `src/eval/ann_benchmark/metrics.py`:

```python
"""Scoring for the ANN benchmark: recall, throughput, and curve interpolation.

Pure numpy. Nothing here touches a device, a file, or an index, which is what
makes the whole scoring path testable on a CPU-only box with no cuVS.

Recall is measured against ground-truth *distances*, not ids. SIFT descriptors
are quantized onto a lattice, so exact ties and true duplicates are common and
dominate the top of any neighbour list -- see `docs/datasets/sift.md`. Under
id-based recall an index that returns a different but exactly equidistant
point is scored as a miss, which would understate every corpus in this
benchmark and understate the real one most of all.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def recall_at_k(
    found_distances: np.ndarray,
    truth_distances: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Fraction of returned neighbours that are as close as the true k-th.

    Both arrays are `(num_queries, k)`, sorted ascending, and must be in the
    same distance space -- squared-L2 throughout this package. `eps` is a
    relative tolerance absorbing float error in the tie comparison; it is not
    an accuracy knob.
    """
    found = np.asarray(found_distances, dtype=np.float64)
    truth = np.asarray(truth_distances, dtype=np.float64)
    if found.shape != truth.shape:
        raise ValueError(
            f"found and truth distances must have the same shape, got "
            f"{found.shape} and {truth.shape}"
        )
    if found.ndim != 2 or found.shape[1] == 0:
        raise ValueError(f"expected a (num_queries, k) array, got {found.shape}")

    threshold = truth[:, -1:] * (1.0 + eps)
    return float(np.mean(found <= threshold))


def qps(num_queries: int, seconds: float) -> float:
    """Queries per second."""
    if seconds <= 0.0:
        raise ValueError(f"elapsed time must be positive, got {seconds}")
    return float(num_queries) / float(seconds)


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Min, median and p95 of repeated timings.

    Same three figures `src/sample/benchmark.py` reports, for the same reason:
    min is the machine's ceiling, median the typical case, p95 what a deadline
    should be budgeted against.
    """
    if len(values) == 0:
        raise ValueError("summarize needs at least one value")
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def qps_at_recall(
    points: Sequence[tuple[float, float]], target: float
) -> float | None:
    """QPS where a (recall, qps) curve crosses `target` recall.

    Returns None when the curve never reaches the target. That is a real
    result -- an index that cannot hit 0.90 on a corpus is the most
    interesting thing the table can say -- so it is reported rather than
    replaced with the nearest point or an extrapolation.

    Interpolation is linear in log(qps) because QPS spans orders of magnitude
    across a sweep while recall does not.
    """
    ordered = sorted((float(r), float(q)) for r, q in points)
    if not ordered:
        return None
    if ordered[-1][0] < target:
        return None
    if ordered[0][0] >= target:
        # Every measured point already clears the target; the fastest of them
        # is the lowest-recall one. Extrapolating past it would invent a
        # configuration that was never run.
        return ordered[0][1]

    for (r0, q0), (r1, q1) in zip(ordered, ordered[1:]):
        if r0 < target <= r1:
            if r1 == r0:
                return q1
            frac = (target - r0) / (r1 - r0)
            if q0 <= 0.0 or q1 <= 0.0:
                return float(q0 + frac * (q1 - q0))
            return float(np.exp(np.log(q0) + frac * (np.log(q1) - np.log(q0))))
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ann_benchmark_metrics.py -v`
Expected: 14 passed.

- [ ] **Step 5: Run the full gate**

Run: `make check`
Expected: ruff clean, whole suite green. If ruff format complains, run `make format` limited to the two new files.

- [ ] **Step 6: Commit**

```bash
git add src/eval/ann_benchmark/__init__.py src/eval/ann_benchmark/metrics.py tests/test_ann_benchmark_metrics.py
git commit -m "feat: add ANN benchmark scoring helpers

Recall is distance-based rather than id-based. SIFT sits on a quantized
lattice where exact ties are common, so an index returning a different but
equidistant neighbour has not missed anything; id-based recall would score
that as a miss and understate every corpus."
```

---

### Task 3: `indexes.py` — the adapter boundary

This is the only module that knows cuVS exists for indexing. Everything downstream talks to the interface.

**Files:**
- Create: `src/eval/ann_benchmark/indexes.py`
- Test: `tests/test_ann_benchmark_indexes.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `@dataclass(frozen=True) BuiltIndex(handle: object, train_seconds: float, add_seconds: float, index_bytes: int, peak_vram_bytes: int | None = None)`
  - `class IndexAdapter` with attributes `name: str`, `param_name: str`, and methods `sweep_params() -> tuple[int | None, ...]`, `build(vectors: np.ndarray) -> BuiltIndex`, `search(built: BuiltIndex, queries: np.ndarray, k: int, param: int | None) -> tuple[np.ndarray, np.ndarray]`, `sync() -> None`, `describe() -> dict[str, object]`
  - `ADAPTER_NAMES: tuple[str, ...]` = `("flat", "ivf_flat", "ivf_pq", "cagra")`
  - `build_adapters(names: Sequence[str]) -> tuple[IndexAdapter, ...]`
  - `require_device_stack() -> None` — CLI preflight; raises `RuntimeError` carrying the install command
  - `NumpyFlatAdapter` — exact brute force in numpy, the runner's test stand-in

`search` returns `(distances, neighbour_ids)`, both `(num_queries, k)`, distances in squared-L2, sorted ascending.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ann_benchmark_indexes.py`:

```python
"""Tests for the ANN index adapter boundary.

These run on a CPU-only box with no cuVS installed, which is the property
being tested: importing this module must not import cuVS. The adapters'
device code is a thin edge covered by the box run, not by pytest.
"""

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
    ivf_flat, ivf_pq, cagra = indexes.build_adapters(
        ["ivf_flat", "ivf_pq", "cagra"]
    )
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
    with pytest.raises(RuntimeError, match="pip install cuvs-cu12"):
        flat.build(np.zeros((4, 2), dtype=np.float32))
    # The CLI preflight raises the same message, so a missing dependency is
    # reported before an hour of corpus materialization rather than after.
    with pytest.raises(RuntimeError, match="pip install cuvs-cu12"):
        indexes.require_device_stack()


def test_built_index_carries_a_vram_figure_slot():
    # None off-device rather than absent, so report.py never has to branch on
    # whether the field exists.
    built = indexes.NumpyFlatAdapter().build(np.eye(2, dtype=np.float32))
    assert built.peak_vram_bytes is None


def test_numpy_adapter_is_a_working_stand_in_for_the_runner():
    # The fake used by the runner tests lives here so both sides agree on the
    # interface. It is exact brute force in numpy over squared L2.
    adapter = indexes.NumpyFlatAdapter()
    vectors = np.eye(4, dtype=np.float32)
    built = adapter.build(vectors)
    assert built.train_seconds >= 0.0
    assert built.index_bytes == vectors.nbytes

    dist, ids = adapter.search(built, vectors[:2], k=2, param=None)
    assert dist.shape == (2, 2)
    assert ids.shape == (2, 2)
    # Each query is a row of the index, so its own row is the nearest at 0.
    assert dist[:, 0] == pytest.approx([0.0, 0.0])
    assert list(ids[:, 0]) == [0, 1]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ann_benchmark_indexes.py -v`
Expected: `ModuleNotFoundError: No module named 'src.eval.ann_benchmark.indexes'`.

- [ ] **Step 3: Write the implementation**

Create `src/eval/ann_benchmark/indexes.py`. **Before writing the cuVS bodies, reconcile every parameter name below against the signatures Task 1 Step 4 printed.** The structure is correct regardless; the keyword spellings are the part that moves between cuVS releases.

```python
"""GPU ANN index adapters -- the single boundary where cuVS is named.

`runner.py` drives everything through `IndexAdapter`, so the grid loop has no
device dependency and is drivable end-to-end by `NumpyFlatAdapter` in tests.

Every cuVS import is inside a method body, never at module scope. This module
must import on a CPU-only box with no cuVS, because `make check` runs there.

All distances are squared L2 ("sqeuclidean"). Corpora are L2-normalized, where
that is monotone in cosine, so the ordering is the one the project's `angular`
metric would give.
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
    "it is CUDA-12-only and would break the CPU-only install CI runs. On the "
    "GPU box install it with:\n"
    "    pip install cuvs-cu12 cupy-cuda12x --extra-index-url https://pypi.nvidia.com"
)


@dataclass(frozen=True)
class BuiltIndex:
    """One built index plus what building it cost."""

    handle: object
    train_seconds: float
    add_seconds: float
    index_bytes: int
    peak_vram_bytes: int | None = None


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
        from cuvs.neighbors import brute_force

        before = self._device_used_bytes()
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
        )

    def search(self, built, queries, k, param):
        from cuvs.neighbors import brute_force

        device_queries = self._to_device(queries)
        distances, neighbours = brute_force.search(
            built.handle, device_queries, k
        )
        return self._to_host(distances), self._to_host(neighbours)


class IvfFlatAdapter(_CuvsAdapter):
    name = "ivf_flat"
    param_name = "n_probes"

    def sweep_params(self) -> tuple[int | None, ...]:
        return IVF_N_PROBES

    def describe(self) -> dict[str, object]:
        return {"n_lists": IVF_N_LISTS, "metric": METRIC}

    def build(self, vectors: np.ndarray) -> BuiltIndex:
        from cuvs.neighbors import ivf_flat

        before = self._device_used_bytes()
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
        )

    def search(self, built, queries, k, param):
        from cuvs.neighbors import ivf_flat

        device_queries = self._to_device(queries)
        search_params = ivf_flat.SearchParams(n_probes=int(param))
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
        from cuvs.neighbors import ivf_pq

        before = self._device_used_bytes()
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
        )

    def search(self, built, queries, k, param):
        from cuvs.neighbors import ivf_pq

        device_queries = self._to_device(queries)
        search_params = ivf_pq.SearchParams(n_probes=int(param))
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
        from cuvs.neighbors import cagra

        before = self._device_used_bytes()
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
        )

    def search(self, built, queries, k, param):
        from cuvs.neighbors import cagra

        device_queries = self._to_device(queries)
        search_params = cagra.SearchParams(itopk_size=int(param))
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ann_benchmark_indexes.py -v`
Expected: 10 passed (one skipped instead of passed if cuVS happens to be installed).

- [ ] **Step 5: Run the full gate**

Run: `make check`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/eval/ann_benchmark/indexes.py tests/test_ann_benchmark_indexes.py
git commit -m "feat: add cuVS index adapters behind a device-free interface

Every cuVS import sits inside a method body so the module imports on a
CPU-only box, which is where make check runs. NumpyFlatAdapter lives beside
the real adapters so a change to the interface breaks both sides in one
commit."
```

---

### Task 4: `groundtruth.py` — exact neighbours

**Files:**
- Create: `src/eval/ann_benchmark/groundtruth.py`
- Test: covered by `tests/test_ann_benchmark_indexes.py` extension below.

**Interfaces:**
- Consumes: `indexes.IndexAdapter`, `indexes.FlatAdapter`, `indexes.NumpyFlatAdapter`.
- Produces: `exact_neighbours(vectors: np.ndarray, queries: np.ndarray, k: int, *, adapter: IndexAdapter | None = None) -> tuple[np.ndarray, np.ndarray]` returning `(distances, ids)`, both `(num_queries, k)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ann_benchmark_indexes.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ann_benchmark_indexes.py -k exact_neighbours -v`
Expected: `ModuleNotFoundError: No module named 'src.eval.ann_benchmark.groundtruth'`.

- [ ] **Step 3: Write the implementation**

Create `src/eval/ann_benchmark/groundtruth.py`:

```python
"""Exact k-nearest neighbours, used as the recall reference.

Computed on GPU with the same brute-force index the benchmark reports as its
recall-1.0 row, so this costs nothing beyond a row that was being measured
anyway.

Queries are *not* excluded from the corpus by index here, unlike
`src.eval.ann_difficulty.knn`. That module self-queries a corpus against
itself, where a point is its own nearest neighbour and must be dropped. Here
the query set is disjoint from the corpus by construction -- a fresh
generator draw, or SIFT's own held-out query set -- so there is no self to
exclude, and dropping a column would silently discard a real neighbour.
"""

from __future__ import annotations

import numpy as np

from src.eval.ann_benchmark.indexes import FlatAdapter, IndexAdapter


def exact_neighbours(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    *,
    adapter: IndexAdapter | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact `(distances, ids)` for every query, in squared L2.

    `adapter` defaults to the GPU brute-force index; tests pass
    `NumpyFlatAdapter` to run the same code path without a device.
    """
    if vectors.ndim != 2 or queries.ndim != 2:
        raise ValueError(
            f"expected 2-D arrays, got {vectors.shape} and {queries.shape}"
        )
    if vectors.shape[1] != queries.shape[1]:
        raise ValueError(
            f"dimension mismatch: corpus is {vectors.shape[1]}-d, queries are "
            f"{queries.shape[1]}-d"
        )
    if k > vectors.shape[0]:
        raise ValueError(
            f"k={k} exceeds the corpus size {vectors.shape[0]}; there are not "
            "that many neighbours to find"
        )

    index = adapter if adapter is not None else FlatAdapter()
    built = index.build(vectors)
    index.sync()
    distances, ids = index.search(built, queries, k, None)
    index.sync()
    return distances, ids
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ann_benchmark_indexes.py -v`
Expected: 12 passed.

- [ ] **Step 5: Run the full gate and commit**

```bash
make check
git add src/eval/ann_benchmark/groundtruth.py tests/test_ann_benchmark_indexes.py
git commit -m "feat: add exact-neighbour ground truth for the ANN benchmark

Does not self-exclude the query row the way ann_difficulty.knn does: the
query sets here are disjoint from their corpus by construction, so dropping
a column would discard a real neighbour."
```

---

### Task 5: `corpora.py` — materialize and normalize

**Files:**
- Create: `src/eval/ann_benchmark/corpora.py`
- Test: `tests/test_ann_benchmark_corpora.py`

**Interfaces:**
- Consumes: `compare_variants.{Variant, CHECKPOINT_NAME, RUN_CONFIG_NAME, invert_samples, load_preprocess_state}`, `dataset.load_descriptors`, `device.resolve_device`, `evaluate_distribution.load_generator`, `train_wgan_gp.sample_generator`, `groundtruth.exact_neighbours`, `indexes.IndexAdapter`.
- Produces:
  - `@dataclass(frozen=True) Corpus(name: str, vectors_path: Path, queries_path: Path, truth_distances_path: Path, truth_ids_path: Path, num_vectors: int, num_queries: int, dim: int)`
  - `normalize(x: np.ndarray) -> np.ndarray`
  - `corpus_seed(base_seed: int, name: str) -> int`, `query_seed(base_seed: int, name: str) -> int`
  - `read_hdf5_queries(cache_dir: Path, num_queries: int) -> np.ndarray`
  - `materialize_real(*, real_path, cache_dir, work_dir, num_vectors, num_queries, k, adapter=None) -> Corpus`
  - `materialize_variant(variant, *, root, work_dir, num_vectors, num_queries, k, batch_size, seed, adapter=None) -> Corpus`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ann_benchmark_corpora.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ann_benchmark_corpora.py -v`
Expected: `ModuleNotFoundError: No module named 'src.eval.ann_benchmark.corpora'`.

- [ ] **Step 3: Write the implementation**

Create `src/eval/ann_benchmark/corpora.py`:

```python
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

import hashlib
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


def _seed(base: int, name: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{name}".encode()).digest()
    return (base + int.from_bytes(digest[:4], "big")) % (2**31)


def corpus_seed(base_seed: int, name: str) -> int:
    """Seed for a variant's corpus draw, independent of run order."""
    return _seed(base_seed, name, "corpus")


def query_seed(base_seed: int, name: str) -> int:
    """Seed for a variant's query draw.

    Salted differently from `corpus_seed` so the two draws cannot coincide.
    If they did, every query would be an exact member of the index and recall
    would read 1.0 for every configuration of every algorithm -- a failure
    that produces a perfectly plausible-looking table.
    """
    return _seed(base_seed, name, "query")


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


def _is_complete(corpus_dir: Path) -> bool:
    return all(
        (corpus_dir / n).exists()
        for n in ("vectors.npy", "queries.npy", "truth_distances.npy", "truth_ids.npy")
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
    if _is_complete(corpus_dir):
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
    if _is_complete(corpus_dir):
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ann_benchmark_corpora.py -v`
Expected: 13 passed.

- [ ] **Step 5: Run the full gate and commit**

```bash
make check
git add src/eval/ann_benchmark/corpora.py tests/test_ann_benchmark_corpora.py
git commit -m "feat: materialize normalized corpora for the ANN benchmark

Real and synthetic both land on the unit sphere. Generators emit unit-norm
vectors and invert_preprocess cannot undo that, while sift_1m.npy is raw
SIFT -- indexing both as they sit on disk would measure the scale gap.

Query draws are salted separately from corpus draws. A seed collision would
put every query inside its own index and read 1.0 recall everywhere, which
would look entirely plausible in the table."
```

---

### Task 6: `runner.py` — the grid

**Files:**
- Create: `src/eval/ann_benchmark/runner.py`
- Test: `tests/test_ann_benchmark_runner.py`

**Interfaces:**
- Consumes: `metrics.{recall_at_k, qps, summarize}`, `indexes.{IndexAdapter, BuiltIndex}`, `corpora.Corpus`.
- Produces:
  - `@dataclass(frozen=True) BuildRecord(corpus, index, train_seconds, add_seconds, index_bytes, params, peak_vram_bytes, failed)`
  - `@dataclass(frozen=True) SearchRecord(corpus, index, param_name, param_value, recall, qps_min, qps_median, qps_p95, num_queries, failed)`
  - `run_grid(corpora_list, adapters, *, k, repeats, records_path) -> tuple[list[BuildRecord], list[SearchRecord]]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ann_benchmark_runner.py`:

```python
"""Tests for the benchmark grid loop.

Driven entirely by fake adapters. The runner must never name a cuVS type, and
these tests are what holds that line: they run on a CPU-only box.
"""

import json

import numpy as np
import pytest

from src.eval.ann_benchmark import corpora, indexes, runner


@pytest.fixture
def tiny_corpus(tmp_path):
    """A four-point corpus with two queries and exact ground truth on disk."""
    vectors = np.eye(4, dtype=np.float32)
    queries = np.eye(4, dtype=np.float32)[:2]
    corpus_dir = tmp_path / "tiny"
    corpus_dir.mkdir()
    np.save(corpus_dir / "vectors.npy", vectors)
    np.save(corpus_dir / "queries.npy", queries)

    adapter = indexes.NumpyFlatAdapter()
    built = adapter.build(vectors)
    dist, ids = adapter.search(built, queries, k=2, param=None)
    np.save(corpus_dir / "truth_distances.npy", dist)
    np.save(corpus_dir / "truth_ids.npy", ids)

    return corpora.Corpus(
        name="tiny",
        vectors_path=corpus_dir / "vectors.npy",
        queries_path=corpus_dir / "queries.npy",
        truth_distances_path=corpus_dir / "truth_distances.npy",
        truth_ids_path=corpus_dir / "truth_ids.npy",
        num_vectors=4,
        num_queries=2,
        dim=4,
    )


class SweepingAdapter(indexes.NumpyFlatAdapter):
    """Exact search that pretends to have a swept knob."""

    name = "sweeping"
    param_name = "n_probes"

    def sweep_params(self):
        return (1, 2)


class ExplodingAdapter(indexes.NumpyFlatAdapter):
    name = "exploding"

    def build(self, vectors):
        raise RuntimeError("out of memory")


class ExplodingSearchAdapter(indexes.NumpyFlatAdapter):
    name = "exploding_search"
    param_name = "n_probes"

    def sweep_params(self):
        return (1, 2)

    def search(self, built, queries, k, param):
        if param == 2:
            raise RuntimeError("search blew up")
        return super().search(built, queries, k, param)


def test_exact_adapter_scores_perfect_recall(tiny_corpus, tmp_path):
    builds, searches = runner.run_grid(
        [tiny_corpus],
        [indexes.NumpyFlatAdapter()],
        k=2,
        repeats=2,
        records_path=tmp_path / "records.json",
    )
    assert len(builds) == 1
    assert len(searches) == 1
    assert searches[0].recall == pytest.approx(1.0)
    assert searches[0].failed is None


def test_one_record_per_swept_parameter(tiny_corpus, tmp_path):
    _, searches = runner.run_grid(
        [tiny_corpus],
        [SweepingAdapter()],
        k=2,
        repeats=1,
        records_path=tmp_path / "records.json",
    )
    assert [s.param_value for s in searches] == [1, 2]
    assert all(s.param_name == "n_probes" for s in searches)


def test_qps_summary_has_all_three_figures(tiny_corpus, tmp_path):
    _, searches = runner.run_grid(
        [tiny_corpus],
        [indexes.NumpyFlatAdapter()],
        k=2,
        repeats=3,
        records_path=tmp_path / "records.json",
    )
    record = searches[0]
    assert record.qps_min > 0.0
    assert record.qps_median >= record.qps_min
    assert record.qps_p95 >= record.qps_min


def test_a_failed_build_is_recorded_and_the_grid_continues(tiny_corpus, tmp_path):
    builds, searches = runner.run_grid(
        [tiny_corpus],
        [ExplodingAdapter(), indexes.NumpyFlatAdapter()],
        k=2,
        repeats=1,
        records_path=tmp_path / "records.json",
    )
    assert builds[0].failed is not None
    assert "out of memory" in builds[0].failed
    # No search records for the index that never built, and the next adapter
    # still ran: one bad cell must not cost the rest of the grid.
    assert [s.index for s in searches] == ["numpy_flat"]


def test_a_failed_search_leaves_its_siblings_intact(tiny_corpus, tmp_path):
    _, searches = runner.run_grid(
        [tiny_corpus],
        [ExplodingSearchAdapter()],
        k=2,
        repeats=1,
        records_path=tmp_path / "records.json",
    )
    by_param = {s.param_value: s for s in searches}
    assert by_param[1].failed is None
    assert by_param[2].failed is not None
    assert by_param[2].recall is None


def test_records_are_written_incrementally(tiny_corpus, tmp_path):
    records_path = tmp_path / "records.json"
    runner.run_grid(
        [tiny_corpus],
        [SweepingAdapter()],
        k=2,
        repeats=1,
        records_path=records_path,
    )
    payload = json.loads(records_path.read_text())
    assert len(payload["searches"]) == 2
    assert len(payload["builds"]) == 1
    assert payload["searches"][0]["corpus"] == "tiny"


def test_build_record_carries_the_fixed_parameters(tiny_corpus, tmp_path):
    builds, _ = runner.run_grid(
        [tiny_corpus],
        [indexes.NumpyFlatAdapter()],
        k=2,
        repeats=1,
        records_path=tmp_path / "records.json",
    )
    assert builds[0].params == {"metric": "sqeuclidean"}
    assert builds[0].index_bytes == 64
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ann_benchmark_runner.py -v`
Expected: `ModuleNotFoundError: No module named 'src.eval.ann_benchmark.runner'`.

- [ ] **Step 3: Write the implementation**

Create `src/eval/ann_benchmark/runner.py`:

```python
"""The benchmark grid: every corpus against every index against every knob.

Nothing here names cuVS. The runner reaches indexes only through
`IndexAdapter`, which is what makes the whole loop -- timing, fencing, failure
handling, incremental output -- testable on a CPU-only box with fake adapters.

Timing discipline: `adapter.sync()` is called immediately before the clock
starts and immediately before it stops. cuVS calls are asynchronous, so an
unfenced region times the launch queue rather than the work and every number
in the table would be fiction. This is the single most important correctness
property in the harness.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.eval.ann_benchmark import metrics
from src.eval.ann_benchmark.corpora import Corpus
from src.eval.ann_benchmark.indexes import IndexAdapter


@dataclass(frozen=True)
class BuildRecord:
    corpus: str
    index: str
    train_seconds: float | None
    add_seconds: float | None
    index_bytes: int | None
    params: dict[str, object]
    peak_vram_bytes: int | None = None
    failed: str | None = None


@dataclass(frozen=True)
class SearchRecord:
    corpus: str
    index: str
    param_name: str
    param_value: int | None
    recall: float | None
    qps_min: float | None
    qps_median: float | None
    qps_p95: float | None
    num_queries: int
    failed: str | None = None


def _flush(
    records_path: Path,
    builds: Sequence[BuildRecord],
    searches: Sequence[SearchRecord],
) -> None:
    """Rewrite the records file after every cell.

    Rewriting rather than appending keeps the file valid JSON at all times, so
    a job killed mid-grid leaves something readable rather than a truncated
    array. The grid is a few hundred small records; the write cost is noise
    next to a CAGRA build.
    """
    payload = {
        "builds": [asdict(b) for b in builds],
        "searches": [asdict(s) for s in searches],
    }
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _time_search(
    adapter: IndexAdapter,
    built,
    queries: np.ndarray,
    k: int,
    param: int | None,
    repeats: int,
) -> tuple[np.ndarray, list[float]]:
    """Run one search point `repeats` times, returning distances and timings.

    The whole query set goes in one call. GPU indexes are throughput devices;
    issuing one query at a time would measure launch latency rather than the
    index.
    """
    seconds: list[float] = []
    distances = None
    for _ in range(repeats):
        adapter.sync()
        started = time.perf_counter()
        distances, _ = adapter.search(built, queries, k, param)
        adapter.sync()
        seconds.append(time.perf_counter() - started)
    return distances, seconds


def run_grid(
    corpora_list: Sequence[Corpus],
    adapters: Sequence[IndexAdapter],
    *,
    k: int,
    repeats: int,
    records_path: Path,
) -> tuple[list[BuildRecord], list[SearchRecord]]:
    """Build every index over every corpus and sweep its search knob."""
    records_path = Path(records_path)
    builds: list[BuildRecord] = []
    searches: list[SearchRecord] = []

    for corpus in corpora_list:
        vectors = np.load(corpus.vectors_path)
        queries = np.load(corpus.queries_path)
        truth = np.load(corpus.truth_distances_path)

        for adapter in adapters:
            try:
                built = adapter.build(vectors)
            except Exception as exc:  # noqa: BLE001 - one bad cell, not the grid
                builds.append(
                    BuildRecord(
                        corpus=corpus.name,
                        index=adapter.name,
                        train_seconds=None,
                        add_seconds=None,
                        index_bytes=None,
                        params=adapter.describe(),
                        failed=f"{type(exc).__name__}: {exc}",
                    )
                )
                _flush(records_path, builds, searches)
                continue

            builds.append(
                BuildRecord(
                    corpus=corpus.name,
                    index=adapter.name,
                    train_seconds=built.train_seconds,
                    add_seconds=built.add_seconds,
                    index_bytes=built.index_bytes,
                    params=adapter.describe(),
                    peak_vram_bytes=built.peak_vram_bytes,
                )
            )
            _flush(records_path, builds, searches)

            for param in adapter.sweep_params():
                try:
                    distances, seconds = _time_search(
                        adapter, built, queries, k, param, repeats
                    )
                    recall = metrics.recall_at_k(distances, truth)
                    throughput = [
                        metrics.qps(queries.shape[0], s) for s in seconds
                    ]
                    summary = metrics.summarize(throughput)
                    searches.append(
                        SearchRecord(
                            corpus=corpus.name,
                            index=adapter.name,
                            param_name=adapter.param_name,
                            param_value=param,
                            recall=recall,
                            qps_min=summary["min"],
                            qps_median=summary["median"],
                            qps_p95=summary["p95"],
                            num_queries=int(queries.shape[0]),
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    searches.append(
                        SearchRecord(
                            corpus=corpus.name,
                            index=adapter.name,
                            param_name=adapter.param_name,
                            param_value=param,
                            recall=None,
                            qps_min=None,
                            qps_median=None,
                            qps_p95=None,
                            num_queries=int(queries.shape[0]),
                            failed=f"{type(exc).__name__}: {exc}",
                        )
                    )
                _flush(records_path, builds, searches)

    return builds, searches
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ann_benchmark_runner.py -v`
Expected: 8 passed.

- [ ] **Step 5: Run the full gate and commit**

```bash
make check
git add src/eval/ann_benchmark/runner.py tests/test_ann_benchmark_runner.py
git commit -m "feat: add the ANN benchmark grid loop

Every timed region is fenced with adapter.sync(). cuVS calls are async, so
an unfenced region times the launch queue rather than the work.

A failed cell is recorded with its exception and the grid continues; one
OOM must not cost six corpora of completed work."
```

---

### Task 7: `report.py` — JSON, markdown, HTML

**Files:**
- Create: `src/eval/ann_benchmark/report.py`
- Test: `tests/test_ann_benchmark_report.py`

**Interfaces:**
- Consumes: `runner.{BuildRecord, SearchRecord}`, `metrics.qps_at_recall`.
- Produces:
  - `headline_rows(builds, searches, *, target_recall: float) -> list[dict[str, object]]`
  - `write_json(path, *, builds, searches, environment) -> None`
  - `write_markdown(path, rows, *, target_recall) -> None`
  - `write_html(path, builds, searches, *, target_recall) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ann_benchmark_report.py`:

```python
"""Tests for benchmark report rendering."""

import json

import pytest

from src.eval.ann_benchmark import report
from src.eval.ann_benchmark.runner import BuildRecord, SearchRecord


def _builds():
    return [
        BuildRecord(
            corpus="real",
            index="ivf_flat",
            train_seconds=12.0,
            add_seconds=0.0,
            index_bytes=1024,
            params={"n_lists": 4096},
        ),
        BuildRecord(
            corpus="v2",
            index="ivf_flat",
            train_seconds=9.0,
            add_seconds=0.0,
            index_bytes=1024,
            params={"n_lists": 4096},
        ),
    ]


def _searches():
    def rec(corpus, param, recall, q):
        return SearchRecord(
            corpus=corpus,
            index="ivf_flat",
            param_name="n_probes",
            param_value=param,
            recall=recall,
            qps_min=q * 0.9,
            qps_median=q,
            qps_p95=q * 1.1,
            num_queries=10,
        )

    return [
        rec("real", 1, 0.80, 400.0),
        rec("real", 2, 0.95, 100.0),
        rec("v2", 1, 0.50, 900.0),
        rec("v2", 2, 0.70, 500.0),
    ]


def test_headline_interpolates_qps_at_the_target_recall():
    rows = report.headline_rows(_builds(), _searches(), target_recall=0.875)
    real = next(r for r in rows if r["corpus"] == "real")
    assert real["qps_at_target"] == pytest.approx(200.0)


def test_headline_reports_none_when_the_target_is_unreachable():
    rows = report.headline_rows(_builds(), _searches(), target_recall=0.90)
    v2 = next(r for r in rows if r["corpus"] == "v2")
    assert v2["qps_at_target"] is None
    assert v2["peak_recall"] == pytest.approx(0.70)


def test_headline_carries_build_time_through():
    rows = report.headline_rows(_builds(), _searches(), target_recall=0.90)
    real = next(r for r in rows if r["corpus"] == "real")
    assert real["build_seconds"] == pytest.approx(12.0)


def test_write_json_round_trips(tmp_path):
    path = tmp_path / "out.json"
    report.write_json(
        path,
        builds=_builds(),
        searches=_searches(),
        environment={"gpu": "test-gpu"},
    )
    payload = json.loads(path.read_text())
    assert payload["environment"]["gpu"] == "test-gpu"
    assert len(payload["searches"]) == 4
    assert payload["builds"][0]["corpus"] == "real"


def test_markdown_marks_an_unreachable_target_rather_than_inventing_a_number(
    tmp_path,
):
    path = tmp_path / "out.md"
    rows = report.headline_rows(_builds(), _searches(), target_recall=0.90)
    report.write_markdown(path, rows, target_recall=0.90)
    text = path.read_text()
    assert "not reached" in text
    assert "| real |" in text
    assert "0.90" in text


def test_html_is_self_contained(tmp_path):
    path = tmp_path / "out.html"
    report.write_html(path, _builds(), _searches(), target_recall=0.90)
    text = path.read_text()
    assert text.lstrip().startswith("<")
    # Inlined plotly, not a CDN reference: the report has to be readable from
    # a checkout with no network.
    assert "cdn.plot.ly" not in text
    assert "recall" in text.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ann_benchmark_report.py -v`
Expected: `ModuleNotFoundError: No module named 'src.eval.ann_benchmark.report'`.

- [ ] **Step 3: Write the implementation**

Create `src/eval/ann_benchmark/report.py`:

```python
"""Render benchmark records as JSON, a markdown table and an HTML report.

The HTML inlines plotly rather than referencing a CDN, so the report is
readable from a checkout with no network -- which is how it will be read,
since it lands in docs/results/ and is opened from disk.

`peak_vram_bytes` reaches the JSON through `asdict` but is deliberately not a
markdown column. It is a card-wide delta rather than a per-index allocation
(see `indexes._device_used_bytes`), so a column of it beside exact per-index
byte counts would read as more precise than it is. The README quotes it as a
sizing figure instead.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.eval.ann_benchmark import metrics
from src.eval.ann_benchmark.runner import BuildRecord, SearchRecord

NOT_REACHED = "not reached"


def headline_rows(
    builds: Sequence[BuildRecord],
    searches: Sequence[SearchRecord],
    *,
    target_recall: float,
) -> list[dict[str, object]]:
    """One row per (corpus, index): build cost and QPS at the target recall."""
    build_by_cell = {(b.corpus, b.index): b for b in builds}
    rows: list[dict[str, object]] = []

    for (corpus, index), build in build_by_cell.items():
        points = [
            (s.recall, s.qps_median)
            for s in searches
            if s.corpus == corpus and s.index == index and s.recall is not None
        ]
        recalls = [r for r, _ in points]
        build_seconds = None
        if build.train_seconds is not None and build.add_seconds is not None:
            build_seconds = build.train_seconds + build.add_seconds
        rows.append(
            {
                "corpus": corpus,
                "index": index,
                "build_seconds": build_seconds,
                "index_bytes": build.index_bytes,
                "qps_at_target": metrics.qps_at_recall(points, target_recall),
                "peak_recall": max(recalls) if recalls else None,
                "failed": build.failed,
            }
        )
    return rows


def write_json(
    path: Path,
    *,
    builds: Sequence[BuildRecord],
    searches: Sequence[SearchRecord],
    environment: dict[str, object],
) -> None:
    """Every cell, unaggregated, plus what produced it."""
    payload = {
        "environment": environment,
        "builds": [asdict(b) for b in builds],
        "searches": [asdict(s) for s in searches],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fmt(value: object, digits: int = 1) -> str:
    if value is None:
        return NOT_REACHED
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def write_markdown(
    path: Path, rows: Sequence[dict[str, object]], *, target_recall: float
) -> None:
    """The headline table.

    A cell whose curve never reached the target prints "not reached" and its
    peak recall, rather than the nearest measured point. Substituting a number
    there would hide the most interesting result the table can carry.
    """
    lines = [
        f"# GPU ANN benchmark (target recall@10 = {target_recall:.2f})",
        "",
        "All corpora are L2-normalized; see the design note. These figures are",
        "not comparable with published SIFT1M results.",
        "",
        "| Corpus | Index | Build (s) | Index (MB) | "
        f"QPS @ recall {target_recall:.2f} | Peak recall |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (str(r["index"]), str(r["corpus"]))):
        megabytes = (
            None
            if row["index_bytes"] is None
            else float(row["index_bytes"]) / 1e6
        )
        lines.append(
            f"| {row['corpus']} | {row['index']} | "
            f"{_fmt(row['build_seconds'], 2)} | {_fmt(megabytes)} | "
            f"{_fmt(row['qps_at_target'])} | {_fmt(row['peak_recall'], 3)} |"
        )
    lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_html(
    path: Path,
    builds: Sequence[BuildRecord],
    searches: Sequence[SearchRecord],
    *,
    target_recall: float,
) -> None:
    """Recall-vs-QPS curves, one facet per index, one trace per corpus."""
    index_names = sorted({s.index for s in searches})
    corpus_names = sorted({s.corpus for s in searches})

    figure = make_subplots(
        rows=1,
        cols=max(len(index_names), 1),
        subplot_titles=index_names or ["no data"],
        shared_yaxes=True,
    )
    for column, index in enumerate(index_names, start=1):
        for corpus in corpus_names:
            points = sorted(
                (s.recall, s.qps_median)
                for s in searches
                if s.corpus == corpus
                and s.index == index
                and s.recall is not None
            )
            if not points:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[r for r, _ in points],
                    y=[q for _, q in points],
                    mode="lines+markers",
                    name=corpus,
                    legendgroup=corpus,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        figure.add_vline(
            x=target_recall, line_dash="dot", row=1, col=column
        )
        figure.update_xaxes(title_text="recall@10", row=1, col=column)

    figure.update_yaxes(title_text="queries/second", type="log", row=1, col=1)
    figure.update_layout(
        title=(
            "GPU ANN benchmark: recall vs throughput "
            "(L2-normalized corpora; not comparable with published SIFT1M)"
        ),
        height=520,
    )

    build_lines = [
        f"{b.corpus}/{b.index}: "
        + (
            f"failed -- {b.failed}"
            if b.failed
            else f"{(b.train_seconds or 0.0) + (b.add_seconds or 0.0):.2f}s"
        )
        for b in builds
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        figure.to_html(full_html=True, include_plotlyjs=True)
        + "<h2>Index build time</h2><pre>"
        + "\n".join(build_lines)
        + "</pre>",
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ann_benchmark_report.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run the full gate and commit**

```bash
make check
git add src/eval/ann_benchmark/report.py tests/test_ann_benchmark_report.py
git commit -m "feat: render ANN benchmark records as JSON, markdown and HTML

An unreachable target recall prints 'not reached' with the peak recall
instead of the nearest measured point; substituting a number there would
hide the most interesting result the table can carry.

Plotly is inlined rather than pulled from a CDN so the report reads from a
checkout with no network."
```

---

### Task 8: `cli.py` — wire it together, and extend the manifest

**Files:**
- Create: `src/eval/ann_benchmark/cli.py`, `src/eval/ann_benchmark/__main__.py`
- Modify: `configs/eval/sift.yaml`

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv: Sequence[str] | None = None) -> None`, invoked as `python -m src.eval.ann_benchmark`.

- [ ] **Step 1: Extend the SIFT variant manifest**

`configs/eval/sift.yaml` stops at `v2` while `docs/datasets/sift.md` documents a ladder through `v4`. Append the two missing rungs, using the run directories confirmed in Task 1 Step 2. **If Task 1 found either run absent, skip that entry and note it in the commit message rather than naming a directory that does not exist.**

```yaml
  - name: v3
    config: configs/sift/v3.yaml
    run_dir: runs/sift_gan_v3
  - name: v4
    config: configs/sift/v4.yaml
    run_dir: runs/x100k_structured
```

This is a manifest edit — it names where existing runs live. It does not change what any variant number means, which would need a human (AGENTS.md, "What requires a human").

- [ ] **Step 2: Write the CLI**

Create `src/eval/ann_benchmark/cli.py`:

```python
"""Run the GPU ANN benchmark over real SIFT and each trained variant.

Example:
    python -m src.eval.ann_benchmark \
        --real-path data/sift_1m.npy \
        --work-dir runs/ann_benchmark \
        --output-dir docs/results/ann-gpu-benchmark
"""

from __future__ import annotations

import argparse
import platform
from collections.abc import Sequence
from pathlib import Path

from src.eval.ann_benchmark import corpora as corpora_mod
from src.eval.ann_benchmark import indexes, report, runner
from src.eval.compare_variants import (
    DEFAULT_MANIFEST,
    describe_missing,
    load_variants,
    resolve_variants,
)

DEFAULT_NUM_VECTORS = 1_000_000
DEFAULT_NUM_QUERIES = 10_000
DEFAULT_K = 10
DEFAULT_REPEATS = 5
DEFAULT_TARGET_RECALL = 0.90


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real-path", type=str, default="data/sift_1m.npy")
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--variants-manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--work-dir", type=str, default="runs/ann_benchmark")
    parser.add_argument(
        "--output-dir", type=str, default="docs/results/ann-gpu-benchmark"
    )
    parser.add_argument("--num-vectors", type=int, default=DEFAULT_NUM_VECTORS)
    parser.add_argument("--num-queries", type=int, default=DEFAULT_NUM_QUERIES)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-recall", type=float, default=DEFAULT_TARGET_RECALL
    )
    parser.add_argument(
        "--indexes",
        nargs="+",
        default=list(indexes.ADAPTER_NAMES),
        choices=list(indexes.ADAPTER_NAMES),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Proceed on a partial ladder instead of aborting. Checkpoints "
            "live on the training box, and a partial comparison is worth "
            "reading once you have decided it is partial on purpose."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root)
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)

    manifest = Path(args.variants_manifest)
    variants = load_variants(manifest)
    found, skipped = resolve_variants(variants, root)
    if skipped and not args.allow_missing:
        raise SystemExit(describe_missing(skipped, manifest, root))
    for variant, reason in skipped:
        print(f"skipping {variant.name}: {reason}")

    adapters = indexes.build_adapters(args.indexes)

    # Fail here rather than forty cells in. Materializing seven corpora and
    # their ground truth is most of an hour's work, and discovering the
    # missing dependency afterwards wastes all of it.
    try:
        indexes.require_device_stack()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    built_corpora = [
        corpora_mod.materialize_real(
            real_path=Path(args.real_path),
            cache_dir=Path(args.cache_dir),
            work_dir=work_dir,
            num_vectors=args.num_vectors,
            num_queries=args.num_queries,
            k=args.k,
        )
    ]
    for variant in found:
        built_corpora.append(
            corpora_mod.materialize_variant(
                variant,
                root=root,
                work_dir=work_dir,
                num_vectors=args.num_vectors,
                num_queries=args.num_queries,
                k=args.k,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        )
        print(f"materialized {variant.name}")

    builds, searches = run_and_report(
        built_corpora, adapters, args, work_dir, output_dir
    )
    print(
        f"{len(builds)} build records, {len(searches)} search records -> "
        f"{output_dir}"
    )


def run_and_report(built_corpora, adapters, args, work_dir, output_dir):
    builds, searches = runner.run_grid(
        built_corpora,
        adapters,
        k=args.k,
        repeats=args.repeats,
        records_path=work_dir / "records.json",
    )
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "num_vectors": args.num_vectors,
        "num_queries": args.num_queries,
        "k": args.k,
        "repeats": args.repeats,
        "target_recall": args.target_recall,
        "normalized": True,
    }
    report.write_json(
        output_dir / "ann_benchmark.json",
        builds=builds,
        searches=searches,
        environment=environment,
    )
    rows = report.headline_rows(
        builds, searches, target_recall=args.target_recall
    )
    report.write_markdown(
        output_dir / "ann_benchmark.md", rows, target_recall=args.target_recall
    )
    report.write_html(
        output_dir / "report.html",
        builds,
        searches,
        target_recall=args.target_recall,
    )
    return builds, searches


if __name__ == "__main__":
    main()
```

Create `src/eval/ann_benchmark/__main__.py`:

```python
from src.eval.ann_benchmark.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the CLI resolves without a GPU**

Run: `python -m src.eval.ann_benchmark --help`
Expected: the help text, listing `--indexes {flat,ivf_flat,ivf_pq,cagra}`. This is the assertion that no cuVS import leaked to module scope.

- [ ] **Step 4: Run the full gate**

Run: `make check`
Expected: green. Note `tests/test_docs_references.py` exists — if it checks that documented commands resolve, a new entry may be needed once Task 9 writes the README.

- [ ] **Step 5: Commit**

```bash
git add src/eval/ann_benchmark/cli.py src/eval/ann_benchmark/__main__.py configs/eval/sift.yaml
git commit -m "feat: add the ANN benchmark CLI and complete the SIFT manifest

configs/eval/sift.yaml stopped at v2 while docs/datasets/sift.md documents a
ladder through v4. Naming where those runs live is a manifest edit; it does
not change what any variant number means."
```

---

### Task 9: Run it on the box and publish the results

**Files:**
- Create: `docs/results/ann-gpu-benchmark/README.md` (plus the generated `ann_benchmark.json`, `ann_benchmark.md`, `report.html`, `gpuq_job_spec.json`)

- [ ] **Step 1: Smoke-test the whole path at tiny scale on the box**

Before queuing an hour-long job, prove the pipeline end to end at a size that fails fast.

```bash
ssh tig-gpu 'cd ~/wgan-synthetic && python -m src.eval.ann_benchmark \
    --num-vectors 20000 --num-queries 1000 --repeats 2 \
    --work-dir /tmp/annbench-smoke \
    --output-dir /tmp/annbench-smoke/out \
    --allow-missing'
```

Expected: a table printed and three files under `/tmp/annbench-smoke/out`. `n_lists=4096` over 20k vectors will warn about too few points per list — that is expected at smoke scale and is not a failure. If any cuVS call raises on a signature, fix `indexes.py` against the real API and re-run Task 3's tests before continuing.

- [ ] **Step 2: Invoke the gpu-jobs skill and submit the full run**

Use the `gpu-jobs` skill. Per the queue notes: do **not** declare `runs/` as an artifact, and stage data per-job because each job gets a fresh worktree. Declare the artifacts as `docs/results/ann-gpu-benchmark/`.

The job command is the CLI with defaults:

```bash
python -m src.eval.ann_benchmark \
    --real-path data/sift_1m.npy \
    --work-dir runs/ann_benchmark \
    --output-dir docs/results/ann-gpu-benchmark
```

Budget 40-60 minutes; CAGRA graph construction dominates. Detach rather than holding the session open.

- [ ] **Step 3: Retrieve the artifacts and write the README**

Write `docs/results/ann-gpu-benchmark/README.md` covering: what was run, the GPU and cuVS version from Task 1, the fixed build parameters, the swept ranges, the peak-VRAM figures out of `ann_benchmark.json` (noting they are card-wide deltas, not per-index allocations), and — stated plainly, not buried — that **every corpus including the real one was L2-normalized before indexing, so these figures describe normalized SIFT and are not comparable with published SIFT1M results.** Note which ladder rungs are present and which were skipped.

Follow the shape of `docs/results/generation-timing/README.md`.

- [ ] **Step 4: Verify the numbers before claiming anything**

Read `ann_benchmark.md` and check each of these before writing a summary:

- Does the exact `flat` row report recall 1.0 on every corpus? If not, the ground truth and the search are in different distance spaces and everything downstream is wrong.
- Does every corpus's recall rise monotonically with the swept knob? A curve that does not is a bug, not a finding.
- Are any cells marked `failed`? Report them explicitly rather than showing the table as complete.
- Is `qps_at_target` `null` anywhere? That is a real result to state, not an omission to paper over.

- [ ] **Step 5: Run the full gate and commit**

```bash
make check
git add docs/results/ann-gpu-benchmark
git commit -m "artifacts: GPU ANN benchmark over real SIFT and the variant ladder"
```

- [ ] **Step 6: Open the PR and take it to green**

Push and open a PR against `main`. Watch the checks and fix until they pass — a PR with pending checks is not done. Report the measured findings plainly, including any failed cells and any corpus that never reached the target recall.

---

## Notes for the implementer

**The one thing most likely to produce a plausible but wrong table:** a query set that is not disjoint from its corpus. Recall reads 1.0 everywhere and nothing looks broken. `corpus_seed` and `query_seed` are salted differently to prevent it, and `tests/test_ann_benchmark_corpora.py` pins that down — do not "simplify" them into one function.

**The second:** an unfenced timed region. cuVS is asynchronous; without `adapter.sync()` on both sides of the clock you are timing kernel launches. Every QPS number would be enormous and meaningless.

**The third:** mixing squared and unsquared L2. `recall_at_k` compares found distances to true distances directly, so if ground truth is in one space and search results in the other, recall collapses without raising anything. Everything in this package is squared L2 (`"sqeuclidean"`).

**On `n_lists=4096`:** correct at 1M, wrong at smoke scale. Expect cuVS warnings during Step 1 of Task 9 and ignore them there.
