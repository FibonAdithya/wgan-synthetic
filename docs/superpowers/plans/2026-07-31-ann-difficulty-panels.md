> **AI-generated working note.** Written by Claude during development and kept
> for provenance. It is not the source of truth: where this file disagrees with
> `PROJECT_DOCUMENTATION.md`, the latter wins.

# ANN Difficulty Panels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four vector-only ANN-difficulty metrics (LID, relative contrast, hubness, IVF cell balance) to `src/eval/eda_report.py` as three new report sections, so the report can answer whether synthetic data would benchmark as hard as real SIFT.

**Architecture:** A new dependency-light module `src/eval/ann_difficulty.py` owns all metric computation as pure functions over an `(N, D)` float array, sharing one brute-force k-NN pass per set. `eda_report.py` imports it and owns only plotting and CLI wiring. The split exists so the metrics are testable without argparse or plotly — this plan also introduces the repository's first test suite.

**Tech Stack:** Python 3.12, numpy, scikit-learn (`NearestNeighbors`, `MiniBatchKMeans`), plotly, pytest.

**Spec:** `docs/superpowers/specs/2026-07-31-ann-difficulty-panels-design.md`

## Global Constraints

- **No new heavyweight dependencies.** numpy + scikit-learn only. No faiss, no hnswlib. The only new requirement is `pytest`.
- **`src/eval/ann_difficulty.py` must not import from `src/eval/eda_report.py`.** It knows nothing about plotly, argparse, or the `Series` type. The dependency runs one way only.
- **Determinism.** Every function taking randomness takes an explicit `seed`. Two runs at the same seed must produce byte-identical output. `MiniBatchKMeans` gets explicit `n_init` and `random_state`.
- **Equal N always.** Every set is truncated to the same row count before any difficulty metric is measured. LID, relative contrast and hubness all drift with sample count.
- **Defaults, copied verbatim from the spec:** `--ann-k` = 100, `--ann-hub-k` = 10, `--ann-max-rows` = 20000, `--ivf-nlist` = 256, relative-contrast target sample = 2000 rows.
- **Section notes must state that the reference is the `real` series in the same report, never a published SIFT1M value.** These panels measure a 20k L2-normalized self-queried subsample; published LID (~9.3) is full-1M, raw, real-query-set and will not reproduce.
- Run tests from the repository root with `python3 -m pytest` so `src.` imports resolve. The environment has `python3`, not `python`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/eval/ann_difficulty.py` | **Create.** All metric computation. Pure numpy/sklearn. |
| `tests/test_ann_difficulty.py` | **Create.** Unit tests with known answers for each metric. |
| `tests/test_eda_report.py` | **Create.** End-to-end wiring smoke test for the report. |
| `src/eval/eda_report.py` | **Modify.** New flags, three figure builders, section placement, `stats_table_html` None-tolerance. |
| `requirements.txt` | **Modify.** Add `pytest`. |

---

### Task 1: Test scaffolding and the Gini helper

Establishes `tests/` (the repository has none) and lands the one metric helper with a closed-form answer, so the test harness is proven before anything harder depends on it.

**Files:**
- Create: `src/eval/ann_difficulty.py`
- Create: `tests/test_ann_difficulty.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `gini(occupancy: np.ndarray) -> float`

- [ ] **Step 1: Add pytest to requirements**

Append to `requirements.txt`:

```
# Test suite for src/eval/ann_difficulty.py. Run from the repo root:
#   python3 -m pytest
pytest
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_ann_difficulty.py`:

```python
import numpy as np

from src.eval.ann_difficulty import gini


def test_gini_is_zero_for_perfectly_balanced_occupancy():
    assert gini(np.array([5, 5, 5, 5])) == 0.0


def test_gini_is_maximal_for_a_single_dominant_cluster():
    # With n cells and all mass in one, the Gini coefficient is exactly
    # (n - 1) / n. At n=4 that is 0.75.
    assert abs(gini(np.array([0, 0, 0, 20])) - 0.75) < 1e-9


def test_gini_is_zero_for_empty_occupancy():
    assert gini(np.array([0, 0, 0])) == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.ann_difficulty'`

- [ ] **Step 4: Create the module with the Gini implementation**

Create `src/eval/ann_difficulty.py`:

```python
"""ANN-difficulty metrics for real-vs-synthetic descriptor comparison.

The EDA report next door compares distributional shape -- whether a synthetic
set *looks* like SIFT. These metrics ask the other question: whether it would
*behave* like SIFT under nearest-neighbour search. A generator can match mean,
variance, pairwise-distance median and effective rank while producing a set
that is far easier or harder to search than the real thing, which makes it
useless as a benchmark stand-in.

Everything here is computed from the vectors alone; no index is built. The
metrics are the published predictors of ANN hardness: local intrinsic
dimensionality, relative contrast, hubness, and partition balance.

All values are relative instruments. They are comparable across the sets
measured in one run and are *not* comparable with published figures for
SIFT1M, which are measured on the full 1M set against the real query set
rather than on a self-queried subsample.

This module deliberately does not import from eda_report: it must stay usable
and testable without plotly or argparse.
"""

from __future__ import annotations

import numpy as np


def gini(occupancy: np.ndarray) -> float:
    """Gini coefficient of a cluster-occupancy vector.

    0.0 means every cell holds the same number of points; (n-1)/n means one
    cell holds everything. Reads as how lopsided an IVF partition would be,
    which drives how many cells a query has to probe.
    """
    x = np.sort(np.asarray(occupancy, dtype=np.float64))
    total = x.sum()
    if total <= 0.0:
        return 0.0
    n = x.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float(2.0 * np.sum(index * x) / (n * total) - (n + 1.0) / n)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v`
Expected: PASS, 3 passed

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/eval/ann_difficulty.py tests/test_ann_difficulty.py
git commit -m "test: add pytest scaffolding and Gini helper for ANN metrics"
```

---

### Task 2: k-NN cache with exact self-exclusion

The single expensive computation all other metrics read from. Self-exclusion is done by index rather than by dropping column 0, because exact duplicate rows tie at distance 0 and sklearn does not guarantee the query itself comes back first.

**Files:**
- Modify: `src/eval/ann_difficulty.py`
- Modify: `tests/test_ann_difficulty.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `knn(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, int]` returning `(distances, indices, k_eff)`, both arrays shaped `(n, k_eff)`, sorted ascending by distance, with each row's own index excluded.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ann_difficulty.py`:

```python
from src.eval.ann_difficulty import knn


def test_knn_excludes_the_query_itself():
    x = np.eye(6, dtype=np.float32)
    _, idx, k_eff = knn(x, k=3)
    assert k_eff == 3
    assert idx.shape == (6, 3)
    for row in range(6):
        assert row not in idx[row].tolist()


def test_knn_excludes_self_even_when_a_duplicate_ties_at_zero_distance():
    # Rows 0 and 1 are identical. Naively stripping column 0 can remove the
    # duplicate instead of the query, leaving the query in its own list.
    x = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    _, idx, _ = knn(x, k=2)
    for row in range(4):
        assert row not in idx[row].tolist()


def test_knn_clamps_k_to_available_neighbours():
    x = np.eye(4, dtype=np.float32)
    dist, idx, k_eff = knn(x, k=100)
    assert k_eff == 3
    assert dist.shape == (4, 3)
    assert idx.shape == (4, 3)


def test_knn_returns_distances_in_ascending_order():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 8)).astype(np.float32)
    dist, _, _ = knn(x, k=5)
    assert np.all(np.diff(dist, axis=1) >= -1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v -k knn`
Expected: FAIL — `ImportError: cannot import name 'knn'`

- [ ] **Step 3: Implement the k-NN cache**

Add the sklearn import to the top of `src/eval/ann_difficulty.py`:

```python
from typing import Dict, Optional, Tuple

from sklearn.neighbors import NearestNeighbors
```

Append to `src/eval/ann_difficulty.py`:

```python
def knn(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Nearest neighbours of every row among the *other* rows.

    Returns (distances, indices, k_eff). k is clamped to n-1 when the set is
    smaller than requested, and k_eff reports what was actually used.

    Self-exclusion is by index, not by dropping the first column. Exact
    duplicate rows tie with the query at distance 0 and sklearn does not
    promise the query sorts first, so column-dropping can silently leave a
    point in its own neighbour list -- which would drag its k-occurrence up
    and its LID down.
    """
    n = x.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 rows to compute neighbours, got {n}")
    k_eff = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="brute").fit(x)
    dist, idx = nn.kneighbors(x)

    rows = np.arange(n)[:, None]
    keep = idx != rows
    # A row whose own index did not come back has k_eff+1 keepers; drop its
    # farthest so every row yields exactly k_eff.
    surplus = keep.sum(axis=1) > k_eff
    if np.any(surplus):
        last_true = (keep.shape[1] - 1) - np.argmax(keep[:, ::-1], axis=1)
        keep[surplus, last_true[surplus]] = False

    selected = np.where(keep)
    return (
        dist[selected].reshape(n, k_eff),
        idx[selected].reshape(n, k_eff),
        k_eff,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/ann_difficulty.py tests/test_ann_difficulty.py
git commit -m "feat(eval): k-NN cache with exact self-exclusion for ANN metrics"
```

---

### Task 3: Local intrinsic dimensionality

**Files:**
- Modify: `src/eval/ann_difficulty.py`
- Modify: `tests/test_ann_difficulty.py`

**Interfaces:**
- Consumes: `knn` from Task 2.
- Produces: `survivor_mask(dist: np.ndarray) -> np.ndarray` (bool, shape `(n,)`) and `lid_mle(dist: np.ndarray) -> np.ndarray` (float, one value per row of the array passed in).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ann_difficulty.py`:

```python
from src.eval.ann_difficulty import lid_mle, survivor_mask


def _uniform_in_ball(n, d, seed):
    """Sample uniformly inside the unit d-ball. LID of such a set equals d."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, d))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = rng.random(size=(n, 1)) ** (1.0 / d)
    return (direction * radius).astype(np.float32)


def test_lid_recovers_the_generating_dimension():
    x = _uniform_in_ball(20000, 4, seed=0)
    dist, _, _ = knn(x, k=100)
    estimate = float(np.median(lid_mle(dist[survivor_mask(dist)])))
    # The Hill estimator is biased and its bias grows with d/n, so this is a
    # deliberately loose 20% band around the true value of 4.
    assert 3.2 < estimate < 4.8


def test_lid_rises_with_the_generating_dimension():
    low = _uniform_in_ball(20000, 4, seed=1)
    high = _uniform_in_ball(20000, 12, seed=1)
    d_low, _, _ = knn(low, k=100)
    d_high, _, _ = knn(high, k=100)
    lid_low = float(np.median(lid_mle(d_low[survivor_mask(d_low)])))
    lid_high = float(np.median(lid_mle(d_high[survivor_mask(d_high)])))
    assert lid_high > lid_low


def test_survivor_mask_rejects_queries_with_a_zero_nearest_distance():
    dist = np.array([[0.0, 1.0], [0.5, 1.0], [0.0, 2.0]])
    assert survivor_mask(dist).tolist() == [False, True, False]


def test_lid_is_finite_for_every_surviving_query_when_duplicates_exist():
    base = _uniform_in_ball(2000, 4, seed=2)
    x = np.vstack([base, base[:200]])  # 200 exact duplicate rows
    dist, _, _ = knn(x, k=50)
    values = lid_mle(dist[survivor_mask(dist)])
    assert values.size > 0
    assert np.all(np.isfinite(values))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v -k "lid or survivor"`
Expected: FAIL — `ImportError: cannot import name 'lid_mle'`

- [ ] **Step 3: Implement LID**

Append to `src/eval/ann_difficulty.py`:

```python
def survivor_mask(dist: np.ndarray) -> np.ndarray:
    """Queries whose nearest neighbour is strictly farther than zero.

    A query sitting on an exact duplicate has r_1 = 0, which sends the LID
    estimator to a degenerate value. Those queries are dropped rather than
    clamped: clamping invents a number, dropping just declines to answer.
    The count is reported so the bias stays visible -- duplicates are exactly
    the low-LID region, so discarding them nudges the estimate upward.
    """
    return dist[:, 0] > 0.0


def lid_mle(dist: np.ndarray) -> np.ndarray:
    """Hill / Amsaleg maximum-likelihood local intrinsic dimensionality.

        LID(q) = -[ (1/k) * sum_i log(r_i / r_k) ]^-1

    Pass only rows selected by survivor_mask. The i=k term contributes zero
    and is kept so the divisor is k, matching the standard MLE form.

    Higher means locally higher-dimensional, which means harder to search:
    this is the strongest single published predictor of ANN difficulty.
    """
    if dist.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    r_k = dist[:, -1:]
    ratio = np.clip(dist / r_k, 1.0e-12, 1.0)
    return -1.0 / np.mean(np.log(ratio), axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/ann_difficulty.py tests/test_ann_difficulty.py
git commit -m "feat(eval): LID estimator with duplicate-query exclusion"
```

---

### Task 4: Relative contrast

**Files:**
- Modify: `src/eval/ann_difficulty.py`
- Modify: `tests/test_ann_difficulty.py`

**Interfaces:**
- Consumes: `knn` from Task 2.
- Produces: `relative_contrast(x: np.ndarray, dist: np.ndarray, seed: int, num_targets: int = 2000) -> np.ndarray`, one value per row of `x`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ann_difficulty.py`:

```python
from src.eval.ann_difficulty import relative_contrast


def test_relative_contrast_falls_as_dimension_rises():
    # Distances concentrate in high dimensions, so the gap between the mean
    # distance and the nearest distance shrinks and search gets harder.
    rng = np.random.default_rng(3)
    low = rng.normal(size=(3000, 2)).astype(np.float32)
    high = rng.normal(size=(3000, 64)).astype(np.float32)
    d_low, _, _ = knn(low, k=10)
    d_high, _, _ = knn(high, k=10)
    rc_low = float(np.median(relative_contrast(low, d_low, seed=0)))
    rc_high = float(np.median(relative_contrast(high, d_high, seed=0)))
    assert rc_high < rc_low


def test_relative_contrast_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(1000, 8)).astype(np.float32)
    dist, _, _ = knn(x, k=10)
    first = relative_contrast(x, dist, seed=7)
    second = relative_contrast(x, dist, seed=7)
    assert np.array_equal(first, second)


def test_relative_contrast_returns_one_value_per_row():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(500, 8)).astype(np.float32)
    dist, _, _ = knn(x, k=10)
    assert relative_contrast(x, dist, seed=0).shape == (500,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v -k relative_contrast`
Expected: FAIL — `ImportError: cannot import name 'relative_contrast'`

- [ ] **Step 3: Implement relative contrast**

Append to `src/eval/ann_difficulty.py`:

```python
def relative_contrast(
    x: np.ndarray,
    dist: np.ndarray,
    seed: int,
    num_targets: int = 2000,
) -> np.ndarray:
    """Mean distance to a fixed target sample, divided by nearest distance.

    The classic Indyk-Motwani hardness measure. A value near 1 means the
    nearest neighbour is barely closer than an arbitrary point, so an index
    has almost nothing to exploit.

    One target sample is drawn per set and shared by every query, so the
    numerator is measured against a fixed reference rather than a per-query
    one. Apply survivor_mask to the result before summarising; rows whose
    nearest distance is zero divide to infinity here.
    """
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    targets = x[np.sort(rng.choice(n, size=min(num_targets, n), replace=False))]

    # Expanded-square distances in chunks. The broadcast form would allocate
    # chunk x targets x dim floats, which is gigabytes at these sizes.
    target_sq = np.einsum("ij,ij->i", targets, targets)
    mean_distance = np.empty(n, dtype=np.float64)
    chunk = 2048
    for start in range(0, n, chunk):
        block = x[start : start + chunk]
        block_sq = np.einsum("ij,ij->i", block, block)
        d2 = block_sq[:, None] + target_sq[None, :] - 2.0 * (block @ targets.T)
        mean_distance[start : start + chunk] = np.sqrt(np.maximum(d2, 0.0)).mean(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        return mean_distance / dist[:, 0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v`
Expected: PASS, 14 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/ann_difficulty.py tests/test_ann_difficulty.py
git commit -m "feat(eval): relative contrast against a fixed target sample"
```

---

### Task 5: Hubness

**Files:**
- Modify: `src/eval/ann_difficulty.py`
- Modify: `tests/test_ann_difficulty.py`

**Interfaces:**
- Consumes: `knn` from Task 2.
- Produces: `k_occurrence(idx: np.ndarray, n: int, k_hub: int) -> np.ndarray` (int counts, shape `(n,)`) and `hubness_skew(counts: np.ndarray) -> float`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ann_difficulty.py`:

```python
from src.eval.ann_difficulty import hubness_skew, k_occurrence


def test_k_occurrence_conserves_total_count():
    # Every one of the n queries contributes exactly k_hub list entries, so
    # the counts must total n * k_hub. This catches off-by-one slips in the
    # index bookkeeping.
    rng = np.random.default_rng(6)
    x = rng.normal(size=(400, 8)).astype(np.float32)
    _, idx, _ = knn(x, k=20)
    counts = k_occurrence(idx, n=400, k_hub=10)
    assert counts.sum() == 400 * 10
    assert counts.shape == (400,)


def test_hubness_skew_is_higher_when_a_hub_is_planted():
    rng = np.random.default_rng(7)
    shell = rng.normal(size=(1500, 6)).astype(np.float32)
    shell /= np.linalg.norm(shell, axis=1, keepdims=True)
    # A single point at the centre is close to everything on the shell, so it
    # lands in a disproportionate share of neighbour lists.
    planted = np.vstack([shell, np.zeros((1, 6), dtype=np.float32)])

    _, idx_plain, _ = knn(shell, k=20)
    _, idx_planted, _ = knn(planted, k=20)
    skew_plain = hubness_skew(k_occurrence(idx_plain, shell.shape[0], 10))
    skew_planted = hubness_skew(k_occurrence(idx_planted, planted.shape[0], 10))
    assert skew_planted > skew_plain


def test_hubness_skew_is_zero_for_a_flat_count_distribution():
    assert hubness_skew(np.array([4, 4, 4, 4])) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v -k "occurrence or hubness"`
Expected: FAIL — `ImportError: cannot import name 'k_occurrence'`

- [ ] **Step 3: Implement hubness**

Append to `src/eval/ann_difficulty.py`:

```python
def k_occurrence(idx: np.ndarray, n: int, k_hub: int) -> np.ndarray:
    """How often each point appears in other points' neighbour lists.

    Reuses the leading columns of the k-NN cache, so k_hub must not exceed
    the k the cache was built with. Ten is the convention in the hubness
    literature and is what the report passes.
    """
    if k_hub > idx.shape[1]:
        raise ValueError(f"k_hub={k_hub} exceeds cached neighbours {idx.shape[1]}")
    return np.bincount(idx[:, :k_hub].ravel(), minlength=n)


def hubness_skew(counts: np.ndarray) -> float:
    """Skewness of the k-occurrence distribution.

    Zero means every point is drawn on about equally. Large positive values
    mean a few hubs dominate the neighbour lists, which is what degrades
    graph indexes like HNSW -- searches funnel into the hubs and stall. A
    generator has no direct training pressure to reproduce this, so it is a
    property worth checking explicitly.
    """
    x = np.asarray(counts, dtype=np.float64)
    spread = x.std()
    if spread <= 0.0:
        return 0.0
    return float(np.mean(((x - x.mean()) / spread) ** 3))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v`
Expected: PASS, 17 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/ann_difficulty.py tests/test_ann_difficulty.py
git commit -m "feat(eval): k-occurrence and hubness skew"
```

---

### Task 6: IVF cell balance

**Files:**
- Modify: `src/eval/ann_difficulty.py`
- Modify: `tests/test_ann_difficulty.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `cell_occupancy(x: np.ndarray, nlist: int, seed: int) -> Tuple[np.ndarray, int]` returning `(occupancy sorted ascending, nlist_eff)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ann_difficulty.py`:

```python
from src.eval.ann_difficulty import cell_occupancy


def test_cell_occupancy_totals_the_row_count_and_sorts_ascending():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(600, 8)).astype(np.float32)
    occupancy, nlist_eff = cell_occupancy(x, nlist=16, seed=0)
    assert occupancy.sum() == 600
    assert occupancy.shape == (nlist_eff,)
    assert np.all(np.diff(occupancy) >= 0)


def test_cell_occupancy_clamps_nlist_to_half_the_row_count():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(40, 4)).astype(np.float32)
    _, nlist_eff = cell_occupancy(x, nlist=256, seed=0)
    assert nlist_eff == 20


def test_cell_occupancy_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(600, 8)).astype(np.float32)
    first, _ = cell_occupancy(x, nlist=16, seed=3)
    second, _ = cell_occupancy(x, nlist=16, seed=3)
    assert np.array_equal(first, second)


def test_well_separated_blobs_partition_more_evenly_than_one_dense_lump():
    rng = np.random.default_rng(11)
    centres = rng.normal(size=(8, 6)).astype(np.float32) * 30.0
    blobs = np.repeat(centres, 100, axis=0) + rng.normal(size=(800, 6)).astype(np.float32)
    lump = rng.normal(size=(800, 6)).astype(np.float32)
    blob_occupancy, _ = cell_occupancy(blobs, nlist=8, seed=0)
    lump_occupancy, _ = cell_occupancy(lump, nlist=8, seed=0)
    assert gini(blob_occupancy) < gini(lump_occupancy)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v -k cell_occupancy`
Expected: FAIL — `ImportError: cannot import name 'cell_occupancy'`

- [ ] **Step 3: Implement cell occupancy**

Add to the imports at the top of `src/eval/ann_difficulty.py`:

```python
from sklearn.cluster import MiniBatchKMeans
```

Append to `src/eval/ann_difficulty.py`:

```python
def cell_occupancy(x: np.ndarray, nlist: int, seed: int) -> Tuple[np.ndarray, int]:
    """Points per cluster under a k-means partition, sorted ascending.

    Stands in for how an IVF index would carve up the set. Each set is
    clustered independently, because an index would be built on whichever
    set was actually shipped -- clustering real and reusing its centroids
    would measure coverage instead of balance.

    nlist is clamped to n // 2 so a small set cannot ask for more cells than
    it can meaningfully fill. Returns the clamped value alongside the counts
    so the caller can report it.
    """
    n = x.shape[0]
    nlist_eff = max(2, min(nlist, n // 2))
    kmeans = MiniBatchKMeans(
        n_clusters=nlist_eff,
        random_state=seed,
        n_init=3,
        batch_size=1024,
    )
    labels = kmeans.fit_predict(x)
    return np.sort(np.bincount(labels, minlength=nlist_eff)), nlist_eff
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v`
Expected: PASS, 21 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/ann_difficulty.py tests/test_ann_difficulty.py
git commit -m "feat(eval): IVF cell-balance occupancy via MiniBatchKMeans"
```

---

### Task 7: Assemble `AnnMetrics`, `compute` and `summary`

Ties the five metric functions into the single entry point the report calls, and handles the degenerate cases that must not crash a long run.

**Files:**
- Modify: `src/eval/ann_difficulty.py`
- Modify: `tests/test_ann_difficulty.py`

**Interfaces:**
- Consumes: `gini`, `knn`, `survivor_mask`, `lid_mle`, `relative_contrast`, `k_occurrence`, `hubness_skew`, `cell_occupancy` from Tasks 1-6.
- Produces:
  - `AnnMetrics` dataclass with fields `lid`, `relative_contrast`, `k_occurrence`, `cell_occupancy` (all `np.ndarray`), `num_rows: int`, `k: int`, `nlist: int`, `discarded_queries: int`.
  - `compute(x, *, k=100, k_hub=10, nlist=256, max_rows=20000, seed=42) -> AnnMetrics`
  - `summary(m: AnnMetrics) -> Dict[str, Optional[float]]` with exactly the keys `lid_median`, `relative_contrast_median`, `hubness_skew`, `ivf_gini`, `lid_discarded_queries`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ann_difficulty.py`:

```python
from src.eval.ann_difficulty import compute, summary


def test_compute_truncates_to_max_rows():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(5000, 8)).astype(np.float32)
    metrics = compute(x, k=20, k_hub=5, nlist=16, max_rows=1000, seed=0)
    assert metrics.num_rows == 1000
    assert metrics.k_occurrence.shape == (1000,)


def test_compute_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(1200, 8)).astype(np.float32)
    kwargs = dict(k=20, k_hub=5, nlist=16, max_rows=800, seed=5)
    first = compute(x, **kwargs)
    second = compute(x, **kwargs)
    assert np.array_equal(first.lid, second.lid)
    assert np.array_equal(first.k_occurrence, second.k_occurrence)
    assert np.array_equal(first.cell_occupancy, second.cell_occupancy)


def test_compute_counts_discarded_duplicate_queries():
    rng = np.random.default_rng(14)
    base = rng.normal(size=(600, 8)).astype(np.float32)
    x = np.vstack([base, base[:100]])
    metrics = compute(x, k=20, k_hub=5, nlist=16, max_rows=0, seed=0)
    assert metrics.discarded_queries >= 200
    assert np.all(np.isfinite(metrics.lid))
    assert np.all(np.isfinite(metrics.relative_contrast))


def test_compute_survives_a_set_that_is_entirely_duplicates():
    x = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (300, 1))
    metrics = compute(x, k=10, k_hub=5, nlist=8, max_rows=0, seed=0)
    assert metrics.lid.size == 0
    assert metrics.discarded_queries == 300
    assert summary(metrics)["lid_median"] is None


def test_summary_returns_the_agreed_keys():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(800, 8)).astype(np.float32)
    result = summary(compute(x, k=20, k_hub=5, nlist=16, max_rows=0, seed=0))
    assert set(result) == {
        "lid_median",
        "relative_contrast_median",
        "hubness_skew",
        "ivf_gini",
        "lid_discarded_queries",
    }
    assert result["lid_median"] > 0
    assert result["lid_discarded_queries"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v -k "compute or summary"`
Expected: FAIL — `ImportError: cannot import name 'compute'`

- [ ] **Step 3: Implement the assembly**

Add to the imports at the top of `src/eval/ann_difficulty.py`:

```python
from dataclasses import dataclass
```

Append to `src/eval/ann_difficulty.py`:

```python
@dataclass
class AnnMetrics:
    """Everything one set contributes to the difficulty panels.

    lid and relative_contrast are aligned: both carry one entry per query
    that survived survivor_mask, in the same order.
    """

    lid: np.ndarray
    relative_contrast: np.ndarray
    k_occurrence: np.ndarray
    cell_occupancy: np.ndarray
    num_rows: int
    k: int
    nlist: int
    discarded_queries: int


def _subsample(x: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Cut to max_rows (0 = keep all).

    Deliberately duplicated from eda_report rather than imported: this module
    must not depend on the report. It is five lines and the dependency
    direction is worth more than the sharing.
    """
    if max_rows <= 0 or x.shape[0] <= max_rows:
        return x
    rng = np.random.default_rng(seed)
    return x[np.sort(rng.choice(x.shape[0], size=max_rows, replace=False))]


def compute(
    x: np.ndarray,
    *,
    k: int = 100,
    k_hub: int = 10,
    nlist: int = 256,
    max_rows: int = 20000,
    seed: int = 42,
) -> AnnMetrics:
    """Measure every difficulty metric for one set off a single k-NN pass.

    Callers must pass the same max_rows for every set they intend to compare:
    LID, relative contrast and hubness all drift with sample count, so
    unequal N makes the overlay meaningless.
    """
    x = np.ascontiguousarray(_subsample(x, max_rows, seed), dtype=np.float32)
    n = x.shape[0]

    dist, idx, k_eff = knn(x, k)
    survivors = survivor_mask(dist)

    lid = lid_mle(dist[survivors])
    contrast = relative_contrast(x, dist, seed)[survivors]
    counts = k_occurrence(idx, n, min(k_hub, k_eff))
    occupancy, nlist_eff = cell_occupancy(x, nlist, seed)

    return AnnMetrics(
        lid=lid,
        relative_contrast=contrast,
        k_occurrence=counts,
        cell_occupancy=occupancy,
        num_rows=n,
        k=k_eff,
        nlist=nlist_eff,
        discarded_queries=int((~survivors).sum()),
    )


def summary(m: AnnMetrics) -> Dict[str, Optional[float]]:
    """Scalars for the report's statistics table and summary.json.

    lid_median and relative_contrast_median are None when every query was
    discarded, which happens only for a fully degenerate set. Callers must
    render None rather than assuming a float.
    """
    has_queries = m.lid.size > 0
    return {
        "lid_median": float(np.median(m.lid)) if has_queries else None,
        "relative_contrast_median": (
            float(np.median(m.relative_contrast)) if has_queries else None
        ),
        "hubness_skew": hubness_skew(m.k_occurrence),
        "ivf_gini": gini(m.cell_occupancy),
        "lid_discarded_queries": float(m.discarded_queries),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ann_difficulty.py -v`
Expected: PASS, 26 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/ann_difficulty.py tests/test_ann_difficulty.py
git commit -m "feat(eval): assemble AnnMetrics compute and summary entry points"
```

---

### Task 8: Wire the metrics into the report's flags and statistics table

Numbers before pictures. This task makes the new scalars appear in the existing table and `summary.json` without adding any figure, so a reviewer can confirm the values are sane before judging how they are drawn.

**Files:**
- Modify: `src/eval/eda_report.py` (imports, `parse_args`, `nn_distances`, `summary_stats`, `stats_table_html`, `main`)

**Interfaces:**
- Consumes: `compute`, `summary`, `AnnMetrics` from Task 7.
- Produces: `main` holds a `Dict[str, AnnMetrics]` keyed by series name, available to Task 9's figure builders.

- [ ] **Step 1: Add the import**

In `src/eval/eda_report.py`, below the existing `from src.data.sift1m_dataset import load_descriptors`:

```python
from src.eval import ann_difficulty
```

- [ ] **Step 2: Add the four flags**

In `parse_args`, immediately after the existing `--knn` argument:

```python
    parser.add_argument(
        "--ann-k",
        type=int,
        default=100,
        help="Neighbours per query for the LID and relative-contrast panels.",
    )
    parser.add_argument(
        "--ann-hub-k",
        type=int,
        default=10,
        help="Neighbour depth for the k-occurrence count behind the hubness panel.",
    )
    parser.add_argument(
        "--ann-max-rows",
        type=int,
        default=20000,
        help=(
            "Equal-N truncation for every difficulty metric, and for the "
            "within-set k-NN panel. LID, contrast and hubness all drift with "
            "sample count, so every set must be cut to the same size."
        ),
    )
    parser.add_argument(
        "--ivf-nlist",
        type=int,
        default=256,
        help="Cluster count for the IVF cell-balance panel.",
    )
```

- [ ] **Step 3: Route the hardcoded truncation through the new flag**

Change the signature of `nn_distances` so its default is supplied rather than baked in:

```python
def nn_distances(x: np.ndarray, k: int, seed: int, max_rows: int) -> np.ndarray:
```

Then update its single call site inside the "Within-set k-NN distances" section in `main` to pass `args.ann_max_rows`:

```python
                [
                    (s.name, nn_distances(s.x, args.knn, args.seed, args.ann_max_rows), s.color)
                    for s in series
                ],
```

And update the same call inside `summary_stats` (Step 5 below covers this).

- [ ] **Step 4: Make the statistics table tolerate None**

`stats_table_html` formats every cell with `:.6g`, which raises `TypeError` on the `None` that `summary` returns for a fully degenerate set. Replace the cell-building line:

```python
        cells = "".join(
            f"<td>{'n/a' if s[k] is None else format(s[k], '.6g')}</td>" for s in stats
        )
```

- [ ] **Step 5: Merge the ANN scalars into `summary_stats`**

Change the signature and the return so the new numbers join the existing table:

```python
def summary_stats(
    s: Series, knn: int, num_pairs: int, seed: int, max_rows: int, metrics
) -> Dict:
    norms = np.linalg.norm(s.x, axis=1)
    stats = {
        "name": s.name,
        "num_vectors": int(s.x.shape[0]),
        "dim": int(s.x.shape[1]),
        "value_mean": float(s.x.mean()),
        "value_std": float(s.x.std()),
        "value_min": float(s.x.min()),
        "value_max": float(s.x.max()),
        "exact_zero_fraction": float((s.x == 0.0).mean()),
        "negative_fraction": float((s.x < 0.0).mean()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "duplicate_row_fraction": float(
            1.0 - np.unique(s.x, axis=0).shape[0] / s.x.shape[0]
        ),
        "median_pairwise_distance": float(
            np.median(pairwise_distance_sample(s.x, num_pairs, seed))
        ),
        f"median_{knn}nn_distance": float(
            np.median(nn_distances(s.x, knn, seed, max_rows))
        ),
        "effective_rank": effective_rank(s.x),
    }
    stats.update(ann_difficulty.summary(metrics))
    return stats
```

- [ ] **Step 6: Compute the metrics once per series in `main`**

In `main`, replace the existing `stats = [...]` line with:

```python
    ann_metrics = {
        s.name: ann_difficulty.compute(
            s.x,
            k=args.ann_k,
            k_hub=args.ann_hub_k,
            nlist=args.ivf_nlist,
            max_rows=args.ann_max_rows,
            seed=args.seed,
        )
        for s in series
    }
    stats = [
        summary_stats(
            s, args.knn, args.num_pairs, args.seed, args.ann_max_rows, ann_metrics[s.name]
        )
        for s in series
    ]
```

- [ ] **Step 7: Record the settings in summary.json**

In the `summary` dict built near the end of `main`, add after `"seed": args.seed,`:

```python
        "ann_settings": {
            "k": args.ann_k,
            "k_hub": args.ann_hub_k,
            "max_rows": args.ann_max_rows,
            "nlist": args.ivf_nlist,
        },
```

- [ ] **Step 8: Verify the report still runs end to end**

```bash
python3 - <<'PY'
import numpy as np
rng = np.random.default_rng(0)
np.save("/tmp/ann_real.npy", rng.normal(size=(3000, 128)).astype(np.float32))
np.save("/tmp/ann_fake.npy", rng.normal(size=(3000, 128)).astype(np.float32))
PY
python3 -m src.eval.eda_report \
    --real-path /tmp/ann_real.npy \
    --synthetic-path fake=/tmp/ann_fake.npy \
    --output-dir /tmp/ann_check \
    --ann-max-rows 1500 --ann-k 30 --ivf-nlist 32 \
    --no-png --plotlyjs cdn
python3 -c "
import json
s = json.load(open('/tmp/ann_check/summary.json'))
print(json.dumps(s['ann_settings'], indent=2))
for row in s['stats']:
    print(row['name'], row['lid_median'], row['hubness_skew'], row['ivf_gini'])
"
```

Expected: the command completes, `ann_settings` prints the four values, and each series prints a positive `lid_median`. For two independent Gaussian sets the LID medians should land close to each other — this is a wiring check, not a fidelity check.

- [ ] **Step 9: Commit**

```bash
git add src/eval/eda_report.py
git commit -m "feat(eval): surface ANN difficulty scalars in the report table"
```

---

### Task 9: The three difficulty figures

**Files:**
- Modify: `src/eval/eda_report.py` (three figure builders, section assembly in `main`)
- Create: `tests/test_eda_report.py`

**Interfaces:**
- Consumes: `ann_metrics` dict from Task 8, `Series` and `overlay_hist_fig` from the existing module.
- Produces: nothing consumed downstream — this is the last task.

- [ ] **Step 1: Write the failing wiring test**

Create `tests/test_eda_report.py`:

```python
import json
import sys

import numpy as np
import pytest

from src.eval import eda_report


def _write_set(path, rows, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    np.save(path, x)


def test_report_writes_html_and_summary_with_ann_sections(tmp_path, monkeypatch):
    real = tmp_path / "real.npy"
    fake = tmp_path / "fake.npy"
    _write_set(real, 400, seed=0)
    _write_set(fake, 400, seed=1)
    out = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eda_report",
            "--real-path", str(real),
            "--synthetic-path", f"fake={fake}",
            "--output-dir", str(out),
            "--ann-max-rows", "300",
            "--ann-k", "20",
            "--ann-hub-k", "5",
            "--ivf-nlist", "8",
            "--max-vectors", "400",
            "--num-pairs", "2000",
            "--no-png",
            "--plotlyjs", "cdn",
        ],
    )
    eda_report.main()

    html = (out / "eda_report.html").read_text(encoding="utf-8")
    assert "Local intrinsic dimensionality" in html
    assert "Hubness" in html
    assert "IVF cell balance" in html

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["ann_settings"]["k"] == 20
    for row in summary["stats"]:
        assert row["lid_median"] > 0
        assert "hubness_skew" in row
        assert "ivf_gini" in row
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_eda_report.py -v`
Expected: FAIL on `assert "Local intrinsic dimensionality" in html` — Task 8 added the numbers but no sections carry those titles yet.

- [ ] **Step 3: Add the difficulty-profile figure**

Append to the plotting-helpers block of `src/eval/eda_report.py`, after `fig_dim_divergence`:

```python
def fig_ann_profile(
    series: Sequence[Series], metrics: Dict[str, "ann_difficulty.AnnMetrics"], bins: int
) -> go.Figure:
    """LID and relative contrast side by side, overlaid across sets.

    Both read off the same surviving queries, so a set that shifts left on
    LID and right on contrast is unambiguously easier to search than real --
    not an artefact of different query subsets.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "local intrinsic dimensionality",
            "relative contrast",
        ),
    )
    for col, attr in ((1, "lid"), (2, "relative_contrast")):
        values = [getattr(metrics[s.name], attr) for s in series]
        populated = [v for v in values if v.size]
        if not populated:
            continue
        edges = shared_edges(populated, bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for s in series:
            v = getattr(metrics[s.name], attr)
            if not v.size:
                continue
            hist, _ = np.histogram(v, bins=edges, density=True)
            fig.add_bar(
                x=centers,
                y=hist,
                name=s.name,
                legendgroup=s.name,
                showlegend=(col == 1),
                marker_color=s.color,
                opacity=0.55,
                row=1,
                col=col,
            )
    fig.update_layout(
        title="ANN difficulty profile",
        barmode="overlay",
        bargap=0.0,
        template="plotly_white",
        height=440,
    )
    return fig
```

- [ ] **Step 4: Add the IVF balance figure**

Append immediately after `fig_ann_profile`:

```python
def fig_ivf_balance(
    series: Sequence[Series], metrics: Dict[str, "ann_difficulty.AnnMetrics"]
) -> go.Figure:
    """Lorenz curve of cluster occupancy: how lopsided an IVF partition is.

    The diagonal is a perfectly even split. Bowing below it means a few
    cells hold most of the points, so a query has to probe more of them to
    reach the same recall.
    """
    fig = go.Figure()
    fig.add_scatter(
        x=[0.0, 1.0],
        y=[0.0, 1.0],
        name="perfect balance",
        line=dict(color="#a0aec0", dash="dash"),
    )
    for s in series:
        occupancy = metrics[s.name].cell_occupancy
        fig.add_scatter(
            x=np.arange(1, occupancy.size + 1) / occupancy.size,
            y=np.cumsum(occupancy) / occupancy.sum(),
            name=s.name,
            line=dict(color=s.color),
        )
    fig.update_layout(
        title="IVF cell balance",
        xaxis_title="fraction of cells (emptiest first)",
        yaxis_title="cumulative fraction of points",
        template="plotly_white",
        height=440,
    )
    return fig
```

- [ ] **Step 5: Insert the three sections ahead of the existing ones**

In `main`, the difficulty sections lead the report. Insert this immediately after `sections: List[Tuple[str, str, go.Figure]] = []` and **before** the existing `sections.append(("Pooled value distribution", ...))`:

```python
    ann_note_suffix = (
        " Compare against the <code>real</code> series in this report only. "
        "These numbers come from a self-queried subsample, so they are not "
        "comparable with published SIFT1M figures."
    )
    first_metrics = ann_metrics[series[0].name]
    sections.append(
        (
            "Local intrinsic dimensionality",
            "How locally high-dimensional the neighbourhood of a typical query "
            "is, and the strongest single predictor of how hard an index will "
            "find this data. A synthetic set landing well below real is easier "
            "to search and would understate any index's difficulty; well above "
            "and it overstates it. Relative contrast sits alongside: values "
            "near 1 mean the nearest neighbour is barely closer than an "
            "arbitrary point, leaving an index little to exploit."
            f" Measured on {first_metrics.num_rows} rows at k={first_metrics.k}."
            + ann_note_suffix,
            fig_ann_profile(series, ann_metrics, args.bins),
        )
    )
    sections.append(
        (
            "Hubness",
            "How often each point turns up in other points' neighbour lists. A "
            "long right tail means a few hubs dominate, which is what stalls "
            "graph indexes like HNSW. A generator gets no direct training "
            "pressure to reproduce this, so matching it is genuine evidence "
            "rather than a fitted artefact." + ann_note_suffix,
            overlay_hist_fig(
                [
                    (s.name, ann_metrics[s.name].k_occurrence.astype(np.float64), s.color)
                    for s in series
                ],
                args.bins,
                f"k-occurrence at k={args.ann_hub_k} (log density)",
                "times appearing in a neighbour list",
                log_y=True,
            ),
        )
    )
    sections.append(
        (
            "IVF cell balance",
            "How evenly k-means would partition each set, which drives how many "
            "cells an IVF query has to probe. Each set is clustered on its own, "
            "because an index would be built on whichever set you shipped."
            f" nlist={first_metrics.nlist}." + ann_note_suffix,
            fig_ivf_balance(series, ann_metrics),
        )
    )
```

- [ ] **Step 6: Run the wiring test to verify it passes**

Run: `python3 -m pytest tests/test_eda_report.py -v`
Expected: PASS, 1 passed

- [ ] **Step 7: Run the whole suite**

Run: `python3 -m pytest -v`
Expected: PASS, 27 passed

- [ ] **Step 8: Generate a report against real data and read it**

```bash
python3 -m src.eval.eda_report \
    --real-path data/sift_base.npy \
    --output-dir runs/eda/ann_check \
    --no-png --plotlyjs cdn
python3 -c "
import json
row = json.load(open('runs/eda/ann_check/summary.json'))['stats'][0]
print({k: row[k] for k in ('lid_median','relative_contrast_median','hubness_skew','ivf_gini','lid_discarded_queries')})
"
```

Expected: completes, and real SIFT reports a positive `hubness_skew` and a non-zero `ivf_gini`. Record the `lid_median` — it is the reference every later comparison reads against. If `data/sift_base.npy` is absent, substitute whichever real descriptor file is on hand and note the substitution.

- [ ] **Step 9: Commit**

```bash
git add src/eval/eda_report.py tests/test_eda_report.py
git commit -m "feat(eval): ANN difficulty, hubness and IVF balance report panels"
```

---

## Self-Review

**Spec coverage:** LID → Task 3; relative contrast → Task 4; hubness → Task 5; IVF cell balance → Tasks 1 (Gini) and 6; `AnnMetrics`/`compute`/`summary` with the five agreed keys → Task 7; four CLI flags and the `nn_distances:186` unification → Task 8; three figures and leading placement → Task 9; all eight spec tests → Tasks 1-7 and 9. The spec's failure-mode table is covered: `N <= k` clamp in Task 2, `nlist` clamp in Task 6, all-discarded path in Task 7, duplicate exclusion in Task 3, `MiniBatchKMeans` determinism in Task 6.

**Gap found and closed:** the spec did not mention that `stats_table_html` formats every cell with `:.6g` and would raise `TypeError` on the `None` that `summary` returns for a degenerate set. Added as Task 8 Step 4.

**Second gap found and closed:** the spec's section notes requirement ("reference is the real series, never the literature") needed to appear in actual note text; added as `ann_note_suffix` in Task 9 Step 5.

**Type consistency:** `compute`/`summary`/`AnnMetrics` field names in Task 7 match every call site in Tasks 8 and 9 (`lid`, `relative_contrast`, `k_occurrence`, `cell_occupancy`, `num_rows`, `k`, `nlist`, `discarded_queries`). `knn` returns a 3-tuple everywhere it is called. `cell_occupancy` returns a 2-tuple in Tasks 6 and 7. `nn_distances` gains its fourth parameter in Task 8 Step 3 and every call site is updated in Steps 3 and 5.
