# GloVe Hub-Statistic Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure which hub statistic, at which N, can carry a gate band for GloVe, and commit the evidence — so issue #29 can be settled by a number instead of an argument, and GloVe `v1` becomes judgeable.

**Architecture:** A torch neighbour backend is added to `src/eval/ann_difficulty.py` behind an opt-in `backend=` argument so the GPU can do the k-NN work, with the self-exclusion logic extracted into one helper both backends share. Two candidate hub statistics are added to the same module but deliberately kept out of `summary()`. A new CLI, `src/eval/hub_stability.py`, draws repeated subsamples at several N, measures six statistics off one k-NN pass per draw, and writes a JSON holding every raw draw plus a mechanically-computed verdict under a rule fixed in advance.

**Tech Stack:** Python 3.12, numpy 2.5.1, torch 2.13.0, scikit-learn 1.9.0, pytest, ruff. Jobs run on the shared GPU box through `gpuq`.

## Scope

This plan covers **phase 1 of the spec**: the measurement tooling, the runs, the committed artifacts, and the write-up. It ends at the pivot — bringing the verdict to a human.

The spec names three possible phase-2 shapes and which one runs is decided by the committed JSON. Writing all three out now would mean writing three implementations knowing two get deleted, and the expensive one (splitting canonical N) touches `AGENTS.md` invariant 3 and deserves its own plan written against real numbers. **Phase 2 gets its own plan after the pivot.**

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-13-glove-hub-statistic-stability-design.md`. It is authoritative on the rule and the grid.
- **The rule is pre-registered and must not be edited after seeing results.** Stable: real-side `range_pct_of_mean` ≤ **10.0**. Discriminating: `|mean(real) − mean(v0 seeds)|` ≥ **1.0 ×** the real-side range (`max − min`).
- **`knn`'s default backend stays `"sklearn"`.** Every committed figure in `docs/datasets/` came from it.
- **Do not add the candidate statistics to `ann_difficulty.summary()`.** That changes `summary.json` for every family; it belongs to phase 2, for the winner only.
- **Every band in `gates/glove.yaml` stays null.** This plan does not touch that file.
- **No model is trained.** The only GPU work is sampling from existing checkpoints and neighbour search.
- **`make check` (ruff lint + ruff format-check + pytest) must pass before every commit.**
- **Python interpreter:** worktrees have no `.venv`. Use `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python`.
- **`gpuq` is at `/venv/main/bin/gpuq` on the box and is not on the ssh PATH.** Always pin `--commit "$(git rev-parse HEAD)"`. Neighbour-search and sampling jobs go in `--lane gpu`; never declare `runs/` as an `--artifact`.

## File Structure

| File | Responsibility |
|---|---|
| `src/eval/ann_difficulty.py` (modify) | Gains `_exclude_self()`, a `"torch"` k-NN backend, a `backend=` passthrough on `compute()`, and the two candidate statistics `hubness_gini()` / `hub_share_top1pct()`. Stays free of plotly and argparse, as its module docstring requires. |
| `src/eval/hub_stability.py` (create) | The sweep: draw allocation, per-draw measurement, spread aggregation, rule evaluation, CLI. Importable functions plus a thin `main()`, mirroring `src/eval/noise_floor.py`. |
| `tests/test_ann_difficulty.py` (modify) | Backend equivalence, self-exclusion under duplicates, and the two new statistics. |
| `tests/test_hub_stability.py` (create) | Draw allocation, determinism, rule evaluation including exact boundaries, and a harness smoke test. |
| `docs/datasets/glove_hub_stability.json` (create, from a run) | GloVe results: every raw draw, spreads, verdicts. |
| `docs/datasets/deep_hub_stability.json` (create, from a run) | DEEP results: condition-1 evidence only. |
| `docs/datasets/glove.md` (modify) | A new results section, and corrections to the two existing hubness sections so the page does not contradict itself. |

---

### Task 1: Share the self-exclusion logic between backends

The current `knn()` does self-exclusion by index, not by dropping the first column, and its docstring explains why: exact duplicate rows tie with the query at distance 0, sklearn does not promise the query sorts first, and column-dropping would silently leave a point in its own neighbour list — dragging its k-occurrence up and its LID down. A second backend must not get a second copy of that reasoning. Extract it first, with no behaviour change, so Task 2 has one place to plug into.

**Files:**
- Modify: `src/eval/ann_difficulty.py:48-81`
- Test: `tests/test_ann_difficulty.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `_exclude_self(dist: np.ndarray, idx: np.ndarray, k_eff: int) -> tuple[np.ndarray, np.ndarray]` — takes the `(n, k_eff+1)` raw neighbour arrays and returns `(n, k_eff)` arrays with each row's own index removed.

- [ ] **Step 1: Write the failing test**

`tests/test_ann_difficulty.py` currently imports bare symbols (`from src.eval.ann_difficulty import cell_occupancy, compute, ...`) and does not import pytest. The new tests reach for private helpers and `pytest.approx`, so add these two lines to its imports first:

```python
import pytest

from src.eval import ann_difficulty
```

Keep the existing bare-symbol import block as it is — the pre-existing tests use it.

Then add:

```python
def test_exclude_self_drops_the_query_from_its_own_row():
    # Row 1 came back as its own nearest neighbour, which is what the
    # +1 column exists to absorb.
    dist = np.array([[0.0, 1.0, 2.0], [0.0, 1.5, 2.5]])
    idx = np.array([[0, 1, 2], [1, 0, 2]])

    kept_dist, kept_idx = ann_difficulty._exclude_self(dist, idx, 2)

    np.testing.assert_array_equal(kept_idx, [[1, 2], [0, 2]])
    np.testing.assert_allclose(kept_dist, [[1.0, 2.0], [1.5, 2.5]])


def test_exclude_self_drops_the_farthest_when_the_query_did_not_come_back():
    # Row 0's own index is absent, so it has three keepers for two slots
    # and the farthest must go -- not an arbitrary one.
    dist = np.array([[0.5, 1.0, 2.0]])
    idx = np.array([[7, 8, 9]])

    kept_dist, kept_idx = ann_difficulty._exclude_self(dist, idx, 2)

    np.testing.assert_array_equal(kept_idx, [[7, 8]])
    np.testing.assert_allclose(kept_dist, [[0.5, 1.0]])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_ann_difficulty.py -k exclude_self -v`
Expected: FAIL with `AttributeError: module 'src.eval.ann_difficulty' has no attribute '_exclude_self'`

- [ ] **Step 3: Extract the helper**

In `src/eval/ann_difficulty.py`, add above `knn`:

```python
def _exclude_self(
    dist: np.ndarray, idx: np.ndarray, k_eff: int
) -> tuple[np.ndarray, np.ndarray]:
    """Drop each row's own index from its neighbour list.

    Self-exclusion is by index, not by dropping the first column. Exact
    duplicate rows tie with the query at distance 0 and neither backend
    promises the query sorts first, so column-dropping can silently leave a
    point in its own neighbour list -- which would drag its k-occurrence up
    and its LID down.

    Callers pass the raw (n, k_eff + 1) arrays. A row whose own index did not
    come back has k_eff + 1 keepers; its farthest is dropped so every row
    yields exactly k_eff.
    """
    n = idx.shape[0]
    rows = np.arange(n)[:, None]
    keep = idx != rows
    surplus = keep.sum(axis=1) > k_eff
    if np.any(surplus):
        last_true = (keep.shape[1] - 1) - np.argmax(keep[:, ::-1], axis=1)
        keep[surplus, last_true[surplus]] = False

    selected = np.where(keep)
    return dist[selected].reshape(n, k_eff), idx[selected].reshape(n, k_eff)
```

Then replace the tail of `knn` (everything from `rows = np.arange(n)[:, None]` to the `return`) with:

```python
    dist, idx = _exclude_self(dist, idx, k_eff)
    return dist, idx, k_eff
```

- [ ] **Step 4: Run the full ann_difficulty suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_ann_difficulty.py -v`
Expected: PASS, including every pre-existing test — this step changed no behaviour.

- [ ] **Step 5: Commit**

```bash
git add src/eval/ann_difficulty.py tests/test_ann_difficulty.py
git commit -m "refactor(eval): extract knn's self-exclusion into a shared helper"
```

---

### Task 2: A torch neighbour backend

**Files:**
- Modify: `src/eval/ann_difficulty.py` (`knn`, `compute`)
- Test: `tests/test_ann_difficulty.py`

**Interfaces:**
- Consumes: `_exclude_self(dist, idx, k_eff)` from Task 1.
- Produces:
  - `knn(x, k, *, backend: str = "sklearn", chunk_rows: int = 1024) -> tuple[np.ndarray, np.ndarray, int]` — unchanged return contract.
  - `compute(x, *, k=100, k_hub=10, nlist=256, max_rows=20000, seed=42, backend="sklearn", chunk_rows=1024) -> AnnMetrics`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ann_difficulty.py`:

```python
def test_torch_backend_returns_the_same_neighbours_as_sklearn():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 16)).astype(np.float32)

    d_sk, i_sk, k_sk = ann_difficulty.knn(x, 10)
    d_t, i_t, k_t = ann_difficulty.knn(x, 10, backend="torch")

    assert k_t == k_sk
    np.testing.assert_allclose(d_t, d_sk, rtol=1e-4, atol=1e-4)
    # Indices are allowed to differ only where the two candidates are
    # indistinguishable at that tolerance -- a genuine tie, not a wrong
    # neighbour.
    differing = i_t != i_sk
    assert np.all(np.abs(d_t[differing] - d_sk[differing]) <= 1e-4)


def test_torch_backend_excludes_self_when_every_row_is_a_duplicate():
    # Every distance is 0.0, so nothing about the sort order can be relied
    # on. This is the case the index-based exclusion exists for.
    x = np.zeros((6, 4), dtype=np.float32)

    _, idx, k_eff = ann_difficulty.knn(x, 3, backend="torch")

    assert k_eff == 3
    assert not np.any(idx == np.arange(6)[:, None])


def test_torch_backend_agrees_with_sklearn_on_every_summary_statistic():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((500, 12)).astype(np.float32)

    sk = ann_difficulty.summary(
        ann_difficulty.compute(x, k=20, k_hub=5, nlist=8, max_rows=0)
    )
    torch_ = ann_difficulty.summary(
        ann_difficulty.compute(x, k=20, k_hub=5, nlist=8, max_rows=0, backend="torch")
    )

    for name in ("lid_median", "relative_contrast_median", "hubness_skew", "ivf_gini"):
        assert sk[name] == pytest.approx(torch_[name], rel=1e-4, abs=1e-6), name


def test_knn_rejects_an_unknown_backend():
    x = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="backend"):
        ann_difficulty.knn(x, 2, backend="faiss")


def test_torch_backend_is_unaffected_by_chunk_width():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((257, 8)).astype(np.float32)

    wide = ann_difficulty.knn(x, 5, backend="torch", chunk_rows=1024)
    narrow = ann_difficulty.knn(x, 5, backend="torch", chunk_rows=7)

    np.testing.assert_array_equal(wide[1], narrow[1])
    np.testing.assert_allclose(wide[0], narrow[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_ann_difficulty.py -k "torch or unknown_backend" -v`
Expected: FAIL with `TypeError: knn() got an unexpected keyword argument 'backend'`

- [ ] **Step 3: Implement the backend**

In `src/eval/ann_difficulty.py`, add the module-level import beside the existing ones:

```python
import torch
```

Add above `knn`:

```python
def _knn_torch(
    x: np.ndarray, want: int, chunk_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force neighbours on the GPU when there is one, else on CPU torch.

    Chunked over query rows because the distance block is the memory cost:
    chunk_rows x n floats. At the default 1024 and n = 250,000 that is 1 GB,
    which sits comfortably beside a 100 MB corpus on an 8 GB card.

    Returns the raw (n, want) arrays including each row's own index; the
    caller drops it via `_exclude_self`.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
    n = corpus.shape[0]

    dist = np.empty((n, want), dtype=np.float64)
    idx = np.empty((n, want), dtype=np.int64)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        block = torch.cdist(corpus[start:stop], corpus)
        values, columns = torch.topk(block, want, dim=1, largest=False, sorted=True)
        dist[start:stop] = values.to(torch.float64).cpu().numpy()
        idx[start:stop] = columns.cpu().numpy()
    return dist, idx
```

Replace `knn`'s signature and dispatch. The new signature and body head:

```python
def knn(
    x: np.ndarray, k: int, *, backend: str = "sklearn", chunk_rows: int = 1024
) -> tuple[np.ndarray, np.ndarray, int]:
```

and, inside, replace the two `NearestNeighbors` lines with:

```python
    if backend == "sklearn":
        nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="brute").fit(x)
        dist, idx = nn.kneighbors(x)
    elif backend == "torch":
        dist, idx = _knn_torch(x, k_eff + 1, chunk_rows)
    else:
        raise ValueError(f"unknown knn backend: {backend!r}")
```

Extend `knn`'s docstring with:

```
    `backend` selects the neighbour search. "sklearn" is the default and is
    what every figure committed under docs/datasets/ was measured with;
    "torch" runs the same search on the GPU when one is available. The two
    agree to the tolerance asserted in tests/test_ann_difficulty.py --
    sklearn's brute euclidean uses the ||x||^2 + ||y||^2 - 2xy expansion,
    which is the numerically worse form, so the difference is not all in
    torch's column.
```

Then thread the arguments through `compute`, whose signature becomes:

```python
def compute(
    x: np.ndarray,
    *,
    k: int = 100,
    k_hub: int = 10,
    nlist: int = 256,
    max_rows: int = 20000,
    seed: int = 42,
    backend: str = "sklearn",
    chunk_rows: int = 1024,
) -> AnnMetrics:
```

with its `knn` call becoming:

```python
    dist, idx, k_eff = knn(x, k, backend=backend, chunk_rows=chunk_rows)
```

- [ ] **Step 4: Run the tests**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_ann_difficulty.py -v`
Expected: PASS, all of them. These run on CPU torch, so no card is needed.

- [ ] **Step 5: Commit**

```bash
git add src/eval/ann_difficulty.py tests/test_ann_difficulty.py
git commit -m "feat(eval): opt-in torch backend for the neighbour search"
```

---

### Task 3: The two candidate hub statistics

Both read the same k-occurrence counts `hubness_skew` reads, so they cost nothing beyond the neighbour search. Neither goes into `summary()` — see Global Constraints.

**Files:**
- Modify: `src/eval/ann_difficulty.py` (add after `hubness_skew`)
- Test: `tests/test_ann_difficulty.py`

**Interfaces:**
- Consumes: the existing `gini(occupancy) -> float`.
- Produces: `hubness_gini(counts: np.ndarray) -> float`, `hub_share_top1pct(counts: np.ndarray) -> float`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ann_difficulty.py`:

```python
def test_hubness_gini_is_zero_when_every_point_is_drawn_on_equally():
    counts = np.full(100, 7)
    assert ann_difficulty.hubness_gini(counts) == pytest.approx(0.0, abs=1e-12)


def test_hubness_gini_approaches_one_when_a_single_hub_takes_everything():
    counts = np.zeros(100)
    counts[0] = 100.0
    # (n - 1) / n for n = 100.
    assert ann_difficulty.hubness_gini(counts) == pytest.approx(0.99)


def test_hub_share_top1pct_is_one_percent_under_a_flat_distribution():
    counts = np.full(100, 7)
    assert ann_difficulty.hub_share_top1pct(counts) == pytest.approx(0.01)


def test_hub_share_top1pct_is_one_when_a_single_hub_takes_everything():
    counts = np.zeros(100)
    counts[0] = 100.0
    assert ann_difficulty.hub_share_top1pct(counts) == pytest.approx(1.0)


def test_hub_share_top1pct_rounds_the_top_slice_up_to_at_least_one_point():
    # 1% of 10 points is 0.1; taking zero points would make the statistic
    # meaningless on small sets.
    counts = np.array([5, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert ann_difficulty.hub_share_top1pct(counts) == pytest.approx(5.0 / 14.0)


def test_hub_statistics_are_zero_on_an_empty_neighbour_budget():
    # Every count zero means no neighbour slots were handed out at all.
    counts = np.zeros(50)
    assert ann_difficulty.hubness_gini(counts) == 0.0
    assert ann_difficulty.hub_share_top1pct(counts) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_ann_difficulty.py -k "hubness_gini or hub_share or hub_statistics" -v`
Expected: FAIL with `AttributeError: module 'src.eval.ann_difficulty' has no attribute 'hubness_gini'`

- [ ] **Step 3: Implement them**

Add `import math` beside the existing imports, then add after `hubness_skew` in `src/eval/ann_difficulty.py`:

```python
def hubness_gini(counts: np.ndarray) -> float:
    """Gini coefficient of the k-occurrence distribution.

    A candidate replacement for `hubness_skew`, which is a third moment and
    is therefore set by whichever handful of tail hubs happened to land in
    the draw -- 108% of its own mean across eight draws of real GloVe. The
    Gini reads the whole distribution rather than its tail, so no small set
    of points can move it far.

    0.0 means every point is drawn on equally; (n - 1) / n means one point
    takes every neighbour slot. Same helper the IVF cell balance uses, which
    is deliberate: two lopsidedness measures should not disagree about what
    lopsided means.
    """
    return gini(np.asarray(counts, dtype=np.float64))


def hub_share_top1pct(counts: np.ndarray) -> float:
    """Fraction of all neighbour slots taken by the top 1% of points.

    The other candidate replacement for `hubness_skew`, and the one that
    reads most directly as the thing that hurts a graph index: how much of
    the neighbour traffic funnels into hubs. Bounded in [0.01, 1] -- 0.01
    when every point is drawn on equally, 1.0 when the top 1% take
    everything.

    The top slice rounds up, so it is never empty on a small set.
    """
    x = np.asarray(counts, dtype=np.float64)
    total = x.sum()
    if total <= 0.0:
        return 0.0
    top = max(1, math.ceil(0.01 * x.size))
    return float(np.sort(x)[-top:].sum() / total)
```

- [ ] **Step 4: Run the tests**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_ann_difficulty.py -v`
Expected: PASS

- [ ] **Step 5: Confirm summary() was left alone**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -c "
from src.eval import ann_difficulty
import numpy as np
m = ann_difficulty.compute(np.random.default_rng(0).standard_normal((200, 8)).astype(np.float32), k=10, k_hub=5, nlist=4, max_rows=0)
print(sorted(ann_difficulty.summary(m)))
"`
Expected: exactly `['hubness_skew', 'ivf_gini', 'lid_discarded_queries', 'lid_median', 'relative_contrast_median']` — the two new statistics must **not** appear.

- [ ] **Step 6: Commit**

```bash
git add src/eval/ann_difficulty.py tests/test_ann_difficulty.py
git commit -m "feat(eval): two candidate hub statistics that are not third moments"
```

---

### Task 4: Draw allocation

Sixteen draws of N rows are disjoint only when `draws * N` fits inside the pool. Above that they share rows, and a shared-row spread understates the true subsample spread — which biases the sweep toward finding statistics stable at large N, the direction that would wrongly favour raising canonical N. The allocator must therefore report which case it is in, not just hand back indices.

**Files:**
- Create: `src/eval/hub_stability.py`
- Test: `tests/test_hub_stability.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `HubStabilityError(Exception)`
  - `allocate_draws(pool_size: int, n: int, draws: int, seed: int) -> tuple[list[np.ndarray], bool]` — returns the row-index arrays (each sorted ascending, length `n`) and whether they are pairwise disjoint.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hub_stability.py`:

```python
import numpy as np
import pytest

from src.eval import hub_stability


def test_draws_are_disjoint_when_the_pool_can_afford_it():
    draws, disjoint = hub_stability.allocate_draws(1000, 100, 10, seed=42)

    assert disjoint is True
    assert len(draws) == 10
    assert all(d.shape == (100,) for d in draws)
    combined = np.concatenate(draws)
    assert combined.size == np.unique(combined).size


def test_draws_overlap_and_say_so_when_the_pool_cannot():
    draws, disjoint = hub_stability.allocate_draws(1000, 400, 10, seed=42)

    assert disjoint is False
    assert len(draws) == 10
    # Each draw is still internally without replacement.
    assert all(np.unique(d).size == 400 for d in draws)


def test_the_exact_boundary_where_the_pool_is_used_up_is_still_disjoint():
    _, disjoint = hub_stability.allocate_draws(1000, 100, 10, seed=42)
    assert disjoint is True
    _, one_more = hub_stability.allocate_draws(999, 100, 10, seed=42)
    assert one_more is False


def test_draw_indices_are_sorted_so_the_subsample_preserves_corpus_order():
    draws, _ = hub_stability.allocate_draws(1000, 100, 3, seed=7)
    for d in draws:
        np.testing.assert_array_equal(d, np.sort(d))


def test_allocation_is_reproducible_under_the_same_seed():
    first, _ = hub_stability.allocate_draws(1000, 100, 4, seed=11)
    second, _ = hub_stability.allocate_draws(1000, 100, 4, seed=11)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_a_draw_larger_than_the_pool_is_an_error():
    with pytest.raises(hub_stability.HubStabilityError, match="pool"):
        hub_stability.allocate_draws(50, 100, 2, seed=42)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_hub_stability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.hub_stability'`

- [ ] **Step 3: Create the module with the allocator**

Create `src/eval/hub_stability.py`:

```python
"""Which hub statistic can carry a gate band, and at what N.

`docs/datasets/glove.md` names hubness skew as the statistic GloVe is most
likely to fail and the most informative one when it does, and then shows that
at the locked canonical N it measures the draw rather than the corpus: eight
20,000-row draws of the real corpus span 108% of the mean. Issue #29 lists
four fixes and says choosing between them needs a measurement. This is that
measurement.

The rule that decides is pre-registered in
`docs/superpowers/specs/2026-08-13-glove-hub-statistic-stability-design.md`
and is applied here by `evaluate_rule`, so the verdict lands in the committed
artifact rather than in a reader's summary of a table.

Like `src/eval/noise_floor.py`, this module stays importable without plotly
and without the report.
"""

from __future__ import annotations

import numpy as np


class HubStabilityError(Exception):
    """The sweep could not run as asked -- bad grid, bad corpus, bad series."""


def allocate_draws(
    pool_size: int, n: int, draws: int, seed: int
) -> tuple[list[np.ndarray], bool]:
    """Row indices for `draws` subsamples of `n` rows, and whether they overlap.

    Disjoint draws are the ones worth having: they are independent samples of
    the corpus, so their spread is the subsample noise. When `draws * n`
    exceeds the pool they cannot all be disjoint, and the spread across
    overlapping draws is a *lower bound* on the true subsample spread -- the
    draws share rows, so they agree with each other more than independent
    draws would. Callers must record the returned flag: a statistic that looks
    stable only in the overlapping regime has not been shown to be stable.
    """
    if n > pool_size:
        raise HubStabilityError(
            f"a draw of {n} rows does not fit a pool of {pool_size}"
        )
    if draws < 2:
        raise HubStabilityError(f"a spread needs at least two draws, got {draws}")

    rng = np.random.default_rng(seed)
    if draws * n <= pool_size:
        order = rng.permutation(pool_size)
        return [np.sort(order[i * n : (i + 1) * n]) for i in range(draws)], True

    return [
        np.sort(rng.choice(pool_size, size=n, replace=False)) for _ in range(draws)
    ], False
```

- [ ] **Step 4: Run the tests**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_hub_stability.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/hub_stability.py tests/test_hub_stability.py
git commit -m "feat(eval): draw allocation that reports whether draws overlap"
```

---

### Task 5: Measure the six statistics for one draw

**Files:**
- Modify: `src/eval/hub_stability.py`
- Test: `tests/test_hub_stability.py`

**Interfaces:**
- Consumes: `ann_difficulty.compute`, `ann_difficulty.summary`, `ann_difficulty.hubness_gini`, `ann_difficulty.hub_share_top1pct` (Tasks 2 and 3).
- Produces:
  - `STATISTICS: tuple[str, ...]` — the six names, in report order.
  - `measure_draw(x: np.ndarray, *, k: int, k_hub: int, nlist: int, seed: int, backend: str, chunk_rows: int) -> dict[str, float]` — one value per name in `STATISTICS`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_hub_stability.py`:

```python
def _draw(rows: int = 300, dim: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, dim)).astype(np.float32)


def test_measure_draw_returns_every_statistic_as_a_finite_number():
    values = hub_stability.measure_draw(
        _draw(), k=10, k_hub=5, nlist=4, seed=42, backend="sklearn", chunk_rows=1024
    )

    assert sorted(values) == sorted(hub_stability.STATISTICS)
    assert all(np.isfinite(v) for v in values.values())


def test_measure_draw_measures_every_row_it_is_given():
    # max_rows must be disabled inside: the caller has already drawn the
    # rows, and a second subsample would silently shrink the draw.
    big = hub_stability.measure_draw(
        _draw(rows=400), k=10, k_hub=5, nlist=4, seed=42,
        backend="sklearn", chunk_rows=1024,
    )
    small = hub_stability.measure_draw(
        _draw(rows=400)[:200], k=10, k_hub=5, nlist=4, seed=42,
        backend="sklearn", chunk_rows=1024,
    )
    assert big["lid_median"] != small["lid_median"]


def test_measure_draw_is_deterministic():
    kwargs = dict(k=10, k_hub=5, nlist=4, seed=42, backend="sklearn", chunk_rows=1024)
    first = hub_stability.measure_draw(_draw(), **kwargs)
    second = hub_stability.measure_draw(_draw(), **kwargs)
    assert first == second


def test_measure_draw_agrees_between_backends():
    kwargs = dict(k=10, k_hub=5, nlist=4, seed=42, chunk_rows=1024)
    sk = hub_stability.measure_draw(_draw(), backend="sklearn", **kwargs)
    torch_ = hub_stability.measure_draw(_draw(), backend="torch", **kwargs)
    for name in hub_stability.STATISTICS:
        assert sk[name] == pytest.approx(torch_[name], rel=1e-4, abs=1e-6), name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_hub_stability.py -k measure_draw -v`
Expected: FAIL with `AttributeError: module 'src.eval.hub_stability' has no attribute 'STATISTICS'`

- [ ] **Step 3: Implement it**

Add to `src/eval/hub_stability.py`, importing the metrics module at the top:

```python
from src.eval import ann_difficulty
```

and:

```python
# Report order: the four incumbents first, then the two candidates. The
# incumbents are carried along as a control -- they re-measure the committed
# eight-draw table at more draws and at larger N for the price of the k-NN
# pass that was happening anyway.
STATISTICS = (
    "lid_median",
    "relative_contrast_median",
    "hubness_skew",
    "ivf_gini",
    "hubness_gini",
    "hub_share_top1pct",
)


def measure_draw(
    x: np.ndarray,
    *,
    k: int,
    k_hub: int,
    nlist: int,
    seed: int,
    backend: str,
    chunk_rows: int,
) -> dict[str, float]:
    """Every statistic for one already-drawn subsample, off a single k-NN pass.

    `max_rows=0` is not optional: the caller drew these rows deliberately, and
    letting `compute` subsample again would measure a smaller set than the one
    the grid says was measured.
    """
    metrics = ann_difficulty.compute(
        x,
        k=k,
        k_hub=k_hub,
        nlist=nlist,
        max_rows=0,
        seed=seed,
        backend=backend,
        chunk_rows=chunk_rows,
    )
    reported = ann_difficulty.summary(metrics)

    for name in ("lid_median", "relative_contrast_median"):
        if reported[name] is None:
            raise HubStabilityError(
                f"{name} was not measurable on this draw: every query was "
                "discarded, which means the draw is degenerate"
            )

    return {
        "lid_median": float(reported["lid_median"]),
        "relative_contrast_median": float(reported["relative_contrast_median"]),
        "hubness_skew": float(reported["hubness_skew"]),
        "ivf_gini": float(reported["ivf_gini"]),
        "hubness_gini": ann_difficulty.hubness_gini(metrics.k_occurrence),
        "hub_share_top1pct": ann_difficulty.hub_share_top1pct(metrics.k_occurrence),
    }
```

- [ ] **Step 4: Run the tests**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_hub_stability.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/hub_stability.py tests/test_hub_stability.py
git commit -m "feat(eval): six hub statistics from one neighbour pass per draw"
```

---

### Task 6: The pre-registered rule, in code

The rule is fixed in the spec and must be mechanical here, so nobody applies it by eye. Two conditions, and one consequence of the overlap flag from Task 4.

**Files:**
- Modify: `src/eval/hub_stability.py`
- Test: `tests/test_hub_stability.py`

**Interfaces:**
- Consumes: `noise_floor.summarize_spread(values) -> dict` with keys `mean`, `std`, `min`, `max`, `range_pct_of_mean`, `cv_pct`.
- Produces:
  - `STABLE_MAX_RANGE_PCT = 10.0`, `MIN_SEPARATION_IN_RANGES = 1.0`
  - `evaluate_rule(real_spread: dict, synthetic_mean: float | None, *, draws_disjoint: bool) -> dict` with keys `stable`, `range_pct_of_mean`, `separation_in_ranges`, `discriminating`, `draws_disjoint`, `verdict`.

Verdicts: with a synthetic mean, `"qualified"` / `"provisional"` / `"rejected"`; without one (a stability-only corpus such as DEEP), `"stable"` / `"unstable"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_hub_stability.py`:

```python
def _spread(mean: float, low: float, high: float) -> dict:
    return {
        "mean": mean,
        "std": 0.0,
        "min": low,
        "max": high,
        "range_pct_of_mean": (high - low) / mean * 100.0,
        "cv_pct": 0.0,
    }


def test_a_stable_and_discriminating_statistic_qualifies():
    # 5% range, and the synthetic mean sits two ranges away.
    verdict = hub_stability.evaluate_rule(
        _spread(1.0, 0.975, 1.025), synthetic_mean=1.1, draws_disjoint=True
    )
    assert verdict["stable"] is True
    assert verdict["discriminating"] is True
    assert verdict["verdict"] == "qualified"


def test_hubness_skews_measured_instability_is_rejected():
    # The real numbers from docs/datasets/glove_noise_floor.json.
    verdict = hub_stability.evaluate_rule(
        _spread(4.4976, 3.4630, 8.3308), synthetic_mean=1.695891, draws_disjoint=True
    )
    assert verdict["stable"] is False
    assert verdict["verdict"] == "rejected"


def test_a_stable_statistic_that_cannot_separate_is_rejected():
    verdict = hub_stability.evaluate_rule(
        _spread(1.0, 0.95, 1.05), synthetic_mean=1.02, draws_disjoint=True
    )
    assert verdict["stable"] is True
    assert verdict["discriminating"] is False
    assert verdict["verdict"] == "rejected"


def test_overlapping_draws_downgrade_a_pass_to_provisional():
    verdict = hub_stability.evaluate_rule(
        _spread(1.0, 0.975, 1.025), synthetic_mean=1.1, draws_disjoint=False
    )
    assert verdict["verdict"] == "provisional"


def test_exactly_ten_percent_range_is_stable_because_the_bound_is_inclusive():
    verdict = hub_stability.evaluate_rule(
        _spread(1.0, 0.95, 1.05), synthetic_mean=2.0, draws_disjoint=True
    )
    assert verdict["range_pct_of_mean"] == pytest.approx(10.0)
    assert verdict["stable"] is True
    assert verdict["verdict"] == "qualified"


def test_exactly_one_range_of_separation_discriminates():
    spread = _spread(1.0, 0.975, 1.025)
    verdict = hub_stability.evaluate_rule(
        spread, synthetic_mean=1.05, draws_disjoint=True
    )
    assert verdict["separation_in_ranges"] == pytest.approx(1.0)
    assert verdict["discriminating"] is True
    assert verdict["verdict"] == "qualified"


def test_a_corpus_with_no_synthetic_series_gets_a_stability_only_verdict():
    verdict = hub_stability.evaluate_rule(
        _spread(1.0, 0.975, 1.025), synthetic_mean=None, draws_disjoint=True
    )
    assert verdict["separation_in_ranges"] is None
    assert verdict["discriminating"] is None
    assert verdict["verdict"] == "stable"


def test_a_zero_range_cannot_be_divided_into_and_does_not_discriminate():
    verdict = hub_stability.evaluate_rule(
        _spread(1.0, 1.0, 1.0), synthetic_mean=5.0, draws_disjoint=True
    )
    assert verdict["separation_in_ranges"] is None
    assert verdict["verdict"] == "rejected"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_hub_stability.py -k "qualif or reject or provisional or exactly or stability_only or zero_range or instability" -v`
Expected: FAIL with `AttributeError: module 'src.eval.hub_stability' has no attribute 'evaluate_rule'`

- [ ] **Step 3: Implement it**

Add to `src/eval/hub_stability.py`:

```python
# Pre-registered in the design spec before any number existed, and not to be
# edited afterwards. Both constants come from precedent in the tree: GloVe's
# three usable statistics sit at 0.32-3.68% range-of-mean against hubness
# skew's 108.2%, and docs/datasets/sift.md already bolds "noise exceeds
# signal" below 1x.
STABLE_MAX_RANGE_PCT = 10.0
MIN_SEPARATION_IN_RANGES = 1.0


def evaluate_rule(
    real_spread: dict,
    synthetic_mean: float | None,
    *,
    draws_disjoint: bool,
) -> dict:
    """Apply the pre-registered rule to one (statistic, N) cell.

    Two conditions. Stable: the real-side range is at most
    STABLE_MAX_RANGE_PCT of the real-side mean. Discriminating: the synthetic
    mean sits at least MIN_SEPARATION_IN_RANGES real-side ranges away, so a
    band drawn around real would reject that generator. Both bounds are
    inclusive.

    Overlapping draws downgrade a pass to "provisional" rather than granting
    it. Their spread is a lower bound on the true subsample spread, so a cell
    that passes only there has not been shown to pass. Since draws are
    disjoint at every N the pool can afford, this is exactly the spec's rule
    that a statistic qualifying only at the largest N is provisional.

    A corpus measured without a synthetic series -- DEEP, here -- can only be
    judged on condition 1, and gets "stable" or "unstable" instead. It is
    evidence about whether an instability generalises, not a vote on GloVe's
    gate.
    """
    range_pct = float(real_spread["range_pct_of_mean"])
    stable = range_pct <= STABLE_MAX_RANGE_PCT

    if synthetic_mean is None:
        return {
            "stable": stable,
            "range_pct_of_mean": range_pct,
            "separation_in_ranges": None,
            "discriminating": None,
            "draws_disjoint": draws_disjoint,
            "verdict": "stable" if stable else "unstable",
        }

    real_range = float(real_spread["max"]) - float(real_spread["min"])
    if real_range == 0.0:
        # Every draw returned the same number. That is not a spread, and
        # dividing by it would report infinite separation from a measurement
        # that has not shown it can vary at all.
        separation = None
        discriminating = False
    else:
        separation = abs(float(real_spread["mean"]) - synthetic_mean) / real_range
        discriminating = separation >= MIN_SEPARATION_IN_RANGES

    if not (stable and discriminating):
        verdict = "rejected"
    elif draws_disjoint:
        verdict = "qualified"
    else:
        verdict = "provisional"

    return {
        "stable": stable,
        "range_pct_of_mean": range_pct,
        "separation_in_ranges": separation,
        "discriminating": discriminating,
        "draws_disjoint": draws_disjoint,
        "verdict": verdict,
    }
```

- [ ] **Step 4: Run the tests**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_hub_stability.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/hub_stability.py tests/test_hub_stability.py
git commit -m "feat(eval): the pre-registered rule, applied mechanically"
```

---

### Task 7: The sweep and its CLI

**Files:**
- Modify: `src/eval/hub_stability.py`
- Test: `tests/test_hub_stability.py`

**Interfaces:**
- Consumes: `allocate_draws`, `measure_draw`, `evaluate_rule`, `STATISTICS` (Tasks 4–6); `noise_floor.summarize_spread`.
- Produces:
  - `sweep(real: np.ndarray, synthetic: dict[str, np.ndarray], *, ns: Sequence[int], draws: int, k: int, k_hub: int, nlist: int, seed: int, backend: str, chunk_rows: int) -> dict`
  - `parse_args() -> argparse.Namespace`, `main() -> None`

Each synthetic series contributes **one** draw per N, not `draws` — mirroring the existing v0 noise floor, where each seed is one measurement. Their mean feeds condition 2; their spread is reported alongside as the training-seed floor at that N.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_hub_stability.py`:

```python
def _sweep_kwargs(**overrides):
    kwargs = dict(
        ns=[60],
        draws=3,
        k=8,
        k_hub=4,
        nlist=4,
        seed=42,
        backend="sklearn",
        chunk_rows=1024,
    )
    kwargs.update(overrides)
    return kwargs


def test_sweep_reports_one_cell_per_n_with_every_raw_draw():
    real = _draw(rows=400, seed=1)

    result = hub_stability.sweep(real, {}, **_sweep_kwargs(ns=[60, 100]))

    assert [c["n"] for c in result["cells"]] == [60, 100]
    for cell in result["cells"]:
        assert len(cell["real"]["per_draw"]) == 3
        assert sorted(cell["real"]["spread"]) == sorted(hub_stability.STATISTICS)


def test_sweep_records_the_pool_and_whether_the_draws_were_disjoint():
    real = _draw(rows=400, seed=1)

    result = hub_stability.sweep(real, {}, **_sweep_kwargs(ns=[60, 200]))

    assert result["pool_rows"] == 400
    disjoint = {c["n"]: c["draws_disjoint"] for c in result["cells"]}
    assert disjoint[60] is True  # 3 x 60 = 180 <= 400
    assert disjoint[200] is False  # 3 x 200 = 600 > 400
    assert result["cells"][0]["pool_to_n"] == pytest.approx(400 / 60)


def test_sweep_without_synthetic_series_gives_stability_only_verdicts():
    result = hub_stability.sweep(_draw(rows=400, seed=1), {}, **_sweep_kwargs())

    verdicts = result["cells"][0]["verdicts"]
    assert sorted(verdicts) == sorted(hub_stability.STATISTICS)
    assert all(v["verdict"] in {"stable", "unstable"} for v in verdicts.values())


def test_sweep_with_synthetic_series_measures_each_once_and_judges_the_mean():
    real = _draw(rows=400, seed=1)
    synthetic = {
        "v0_seed42": _draw(rows=400, seed=2),
        "v0_seed43": _draw(rows=400, seed=3),
    }

    result = hub_stability.sweep(real, synthetic, **_sweep_kwargs())

    cell = result["cells"][0]
    assert sorted(cell["synthetic"]["per_series"]) == ["v0_seed42", "v0_seed43"]
    assert sorted(cell["synthetic"]["mean"]) == sorted(hub_stability.STATISTICS)
    assert all(
        v["verdict"] in {"qualified", "provisional", "rejected"}
        for v in cell["verdicts"].values()
    )


def test_sweep_refuses_an_n_larger_than_the_corpus():
    with pytest.raises(hub_stability.HubStabilityError, match="pool"):
        hub_stability.sweep(_draw(rows=100, seed=1), {}, **_sweep_kwargs(ns=[500]))


def test_sweep_is_reproducible():
    real = _draw(rows=400, seed=1)
    first = hub_stability.sweep(real, {}, **_sweep_kwargs())
    second = hub_stability.sweep(real, {}, **_sweep_kwargs())
    assert first == second
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_hub_stability.py -k sweep -v`
Expected: FAIL with `AttributeError: module 'src.eval.hub_stability' has no attribute 'sweep'`

- [ ] **Step 3: Implement the sweep**

Add the remaining imports at the top of `src/eval/hub_stability.py`:

```python
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from src.eval.noise_floor import summarize_spread
```

and add:

```python
def sweep(
    real: np.ndarray,
    synthetic: dict[str, np.ndarray],
    *,
    ns: Sequence[int],
    draws: int,
    k: int,
    k_hub: int,
    nlist: int,
    seed: int,
    backend: str,
    chunk_rows: int,
) -> dict:
    """Measure every statistic across repeated draws, at every N in the grid.

    The real corpus gets `draws` subsamples per N -- that is the subsample
    noise the gate band is judged against. Each synthetic series gets exactly
    one, mirroring the v0 noise floor where each training seed is one
    measurement; their mean feeds condition 2 and their spread is reported
    beside it as the training-seed floor at that N.
    """
    measure = {
        "k": k,
        "k_hub": k_hub,
        "nlist": nlist,
        "seed": seed,
        "backend": backend,
        "chunk_rows": chunk_rows,
    }
    cells = []

    for n in ns:
        indices, disjoint = allocate_draws(real.shape[0], n, draws, seed)
        per_draw = [measure_draw(real[rows], **measure) for rows in indices]
        real_spread = {
            name: summarize_spread([d[name] for d in per_draw])
            for name in STATISTICS
        }

        per_series: dict[str, dict[str, float]] = {}
        for label in sorted(synthetic):
            series = synthetic[label]
            rows, _ = allocate_draws(series.shape[0], n, 2, seed)
            per_series[label] = measure_draw(series[rows[0]], **measure)

        synthetic_mean: dict[str, float] | None = None
        synthetic_spread: dict[str, dict[str, float]] | None = None
        if len(per_series) >= 2:
            synthetic_mean = {
                name: float(np.mean([v[name] for v in per_series.values()]))
                for name in STATISTICS
            }
            synthetic_spread = {
                name: summarize_spread([v[name] for v in per_series.values()])
                for name in STATISTICS
            }
        elif len(per_series) == 1:
            only = next(iter(per_series.values()))
            synthetic_mean = {name: only[name] for name in STATISTICS}

        cells.append(
            {
                "n": n,
                "draws": draws,
                "draws_disjoint": disjoint,
                "pool_to_n": real.shape[0] / n,
                "real": {"per_draw": per_draw, "spread": real_spread},
                "synthetic": {
                    "series": sorted(per_series),
                    "per_series": per_series,
                    "mean": synthetic_mean,
                    "spread": synthetic_spread,
                },
                "verdicts": {
                    name: evaluate_rule(
                        real_spread[name],
                        None if synthetic_mean is None else synthetic_mean[name],
                        draws_disjoint=disjoint,
                    )
                    for name in STATISTICS
                },
            }
        )

    return {
        "pool_rows": int(real.shape[0]),
        "conditions": measure,
        "rule": {
            "stable_max_range_pct": STABLE_MAX_RANGE_PCT,
            "min_separation_in_ranges": MIN_SEPARATION_IN_RANGES,
        },
        "statistics": list(STATISTICS),
        "cells": cells,
    }
```

- [ ] **Step 4: Run the sweep tests**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_hub_stability.py -v`
Expected: PASS

- [ ] **Step 5: Add the CLI**

Append to `src/eval/hub_stability.py`:

```python
def _load_series(specs: Sequence[str] | None) -> dict[str, np.ndarray]:
    """Parse repeatable LABEL=PATH arguments into loaded arrays."""
    loaded: dict[str, np.ndarray] = {}
    for spec in specs or []:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            raise HubStabilityError(
                f"--synthetic-path wants LABEL=PATH, got {spec!r}"
            )
        if label in loaded:
            raise HubStabilityError(f"--synthetic-path {label!r} given twice")
        loaded[label] = np.load(path)
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--real-path", type=str, required=True, help="Real corpus .npy to draw from."
    )
    parser.add_argument(
        "--synthetic-path",
        type=str,
        action="append",
        metavar="LABEL=PATH",
        help=(
            "Repeatable. One per generator seed. Each is measured once per N; "
            "their mean is what the rule's second condition judges. Omit "
            "entirely to measure real-side stability alone."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        action="append",
        required=True,
        dest="ns",
        help="Repeatable. Subsample size to measure at.",
    )
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--k-hub", type=int, default=10)
    parser.add_argument("--nlist", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        type=str,
        default="sklearn",
        choices=("sklearn", "torch"),
        help=(
            "Neighbour search. Default sklearn, which every committed figure "
            "was measured with; torch uses the GPU when there is one."
        ),
    )
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument(
        "--output", type=str, default=None, help="Also write the JSON here."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        real = np.load(args.real_path)
        synthetic = _load_series(args.synthetic_path)
        result = sweep(
            real,
            synthetic,
            ns=args.ns,
            draws=args.draws,
            k=args.k,
            k_hub=args.k_hub,
            nlist=args.nlist,
            seed=args.seed,
            backend=args.backend,
            chunk_rows=args.chunk_rows,
        )
    except (HubStabilityError, OSError, ValueError) as exc:
        # stderr, so stdout stays parseable as JSON or empty, never half a
        # report -- the same contract noise_floor.py keeps.
        print(f"hub_stability: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    result["real_path"] = args.real_path
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write the CLI smoke test**

Add to `tests/test_hub_stability.py`:

```python
def test_cli_writes_a_json_that_holds_its_own_evidence(tmp_path, monkeypatch, capsys):
    real_path = tmp_path / "real.npy"
    np.save(real_path, _draw(rows=400, seed=1))
    output = tmp_path / "out.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "hub_stability",
            "--real-path", str(real_path),
            "--n", "60",
            "--draws", "3",
            "--k", "8",
            "--k-hub", "4",
            "--nlist", "4",
            "--output", str(output),
        ],
    )
    hub_stability.main()

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["real_path"] == str(real_path)
    assert written["rule"]["stable_max_range_pct"] == 10.0
    assert len(written["cells"][0]["real"]["per_draw"]) == 3
    assert json.loads(capsys.readouterr().out) == written


def test_cli_rejects_a_malformed_synthetic_path(tmp_path, monkeypatch):
    real_path = tmp_path / "real.npy"
    np.save(real_path, _draw(rows=200, seed=1))

    monkeypatch.setattr(
        "sys.argv",
        [
            "hub_stability",
            "--real-path", str(real_path),
            "--synthetic-path", "no-equals-sign",
            "--n", "50", "--draws", "2", "--k", "8", "--k-hub", "4", "--nlist", "4",
        ],
    )
    with pytest.raises(SystemExit):
        hub_stability.main()
```

Add `import json` to the test file's imports.

- [ ] **Step 7: Run everything and lint**

Run: `cd /home/fibonadithya/.herdr/worktrees/wgan-synthetic/glove-v1-design && make check`
Expected: ruff clean, full suite passes.

- [ ] **Step 8: Commit**

```bash
git add src/eval/hub_stability.py tests/test_hub_stability.py
git commit -m "feat(eval): sweep hub-statistic stability across N and commit the verdict"
```

---

### Task 8: Re-sample the five v0 checkpoints to 250k vectors

Condition 2 needs `v0` at every N in the grid, and `v0`'s committed samples are 50,000 vectors per seed. This is a **new measurement of `v0`**, not the one behind `docs/datasets/glove_v0_noise_floor.json`.

**Files:**
- No repository files change. This task produces run artifacts on the box.

**Interfaces:**
- Consumes: `runs/glove/v0_seed{42..46}/best_generator.pt`, plus `configs/glove/v0_seed{42..46}.yaml` from the pinned checkout — the same pairing `docs/datasets/glove.md#noise-floor` documents.
- Produces: `runs/glove/v0_seed{42..46}/samples_250k.npy`, 250,000 × 100 float32 each.

- [ ] **Step 1: Confirm the checkpoints are on the box**

```bash
ssh tig-gpu 'ls -la ~/wgan-synthetic/runs/glove/v0_seed4{2,3,4,5,6}/best_generator.pt'
```

Expected: five files. If any is missing, stop — the sweep cannot judge condition 2 and the finding is that `v0`'s artifacts did not survive (see issue #36).

- [ ] **Step 2: Push the branch so the runner can check out this commit**

```bash
git push -u origin glove-v1-design
```

- [ ] **Step 3: Submit the sampling job**

```bash
ssh tig-gpu '/venv/main/bin/gpuq submit --project wgan-synthetic \
  --commit "'"$(git rev-parse HEAD)"'" --branch glove-v1-design --lane gpu \
  --artifact docs/datasets/.keep \
  -- bash -c "for seed in 42 43 44 45 46; do \
      python -m src.sample.generate \
        --checkpoint runs/glove/v0_seed\${seed}/best_generator.pt \
        --config configs/glove/v0_seed\${seed}.yaml \
        --num-samples 250000 --seed 42 \
        --output-path runs/glove/v0_seed\${seed}/samples_250k.npy; \
    done"'
```

Note: `runs/` is never declared as an artifact — the samples stay on the box and are read by the next job, which runs in the same checkout tree.

- [ ] **Step 4: Wait for it**

```bash
ssh tig-gpu '/venv/main/bin/gpuq wait <id>'
```

Expected: exit 0. On failure, `gpuq show <id>` carries the stderr tail.

- [ ] **Step 5: Check the shapes**

```bash
ssh tig-gpu 'python -c "
import numpy as np
for s in range(42, 47):
    a = np.load(f\"runs/glove/v0_seed{s}/samples_250k.npy\")
    print(s, a.shape, a.dtype)
"'
```

Expected: five lines reading `(250000, 100) float32`.

- [ ] **Step 6: No commit**

Nothing in the repository changed. Record the job id in the eventual PR description so the artifacts can be traced.

---

### Task 9: Run the provenance cell and check it against the committed floor

Before any sweep number is believed, the new code must reproduce the measurement already in the tree. This runs the exact conditions `docs/datasets/glove_noise_floor.json` was measured under — N=20,000 drawn from `glove_250k.npy` — and its ranges must sit inside the committed ones.

**Files:**
- Create: `docs/datasets/glove_hub_stability_provenance.json`

- [ ] **Step 1: Submit the provenance job**

```bash
ssh tig-gpu '/venv/main/bin/gpuq submit --project wgan-synthetic \
  --commit "'"$(git rev-parse HEAD)"'" --branch glove-v1-design --lane gpu \
  --artifact docs/datasets/glove_hub_stability_provenance.json \
  -- python -m src.eval.hub_stability \
      --real-path data/glove_250k.npy \
      --n 20000 --draws 16 \
      --k 100 --k-hub 10 --nlist 256 --seed 42 \
      --backend torch \
      --output docs/datasets/glove_hub_stability_provenance.json'
```

- [ ] **Step 2: Wait, then read the four incumbent spreads**

```bash
ssh tig-gpu '/venv/main/bin/gpuq wait <id>'
```

- [ ] **Step 3: Compare against the committed floor**

The committed eight-draw figures in `docs/datasets/glove_noise_floor.json` are:

| Statistic | mean | min | max |
|---|---|---|---|
| LID median | 35.1238 | 35.0318 | 35.2086 |
| Relative contrast | 1.38951 | 1.38754 | 1.39201 |
| Hubness skew | 4.4976 | 3.4630 | 8.3308 |
| IVF Gini | 0.59324 | 0.58157 | 0.60339 |

**Pass condition:** each of the four new means sits inside the committed min–max range for that statistic. Sixteen draws will not reproduce eight draws' extremes exactly, and the ranges are expected to widen slightly with more draws — the mean is what must agree.

**If it does not pass, stop.** The discrepancy is the finding, and it means either the torch backend or the draw allocation differs from what produced the committed numbers. Report it before running anything larger.

- [ ] **Step 4: Commit the artifact**

```bash
git add docs/datasets/glove_hub_stability_provenance.json
git commit -m "measure: the new sweep reproduces the committed GloVe noise floor"
```

---

### Task 10: Run the full grid

**Files:**
- Create: `docs/datasets/glove_hub_stability.json`, `docs/datasets/deep_hub_stability.json`

- [ ] **Step 1: Submit the GloVe sweep**

```bash
ssh tig-gpu '/venv/main/bin/gpuq submit --project wgan-synthetic \
  --commit "'"$(git rev-parse HEAD)"'" --branch glove-v1-design --lane gpu \
  --artifact docs/datasets/glove_hub_stability.json \
  -- python -m src.eval.hub_stability \
      --real-path data/glove_1m.npy \
      --synthetic-path v0_seed42=runs/glove/v0_seed42/samples_250k.npy \
      --synthetic-path v0_seed43=runs/glove/v0_seed43/samples_250k.npy \
      --synthetic-path v0_seed44=runs/glove/v0_seed44/samples_250k.npy \
      --synthetic-path v0_seed45=runs/glove/v0_seed45/samples_250k.npy \
      --synthetic-path v0_seed46=runs/glove/v0_seed46/samples_250k.npy \
      --n 20000 --n 50000 --n 100000 --n 250000 \
      --draws 16 --k 100 --k-hub 10 --nlist 256 --seed 42 \
      --backend torch \
      --output docs/datasets/glove_hub_stability.json'
```

- [ ] **Step 2: Submit the DEEP sweep**

DEEP carries condition 1 only — no synthetic series, so no `--synthetic-path`.

```bash
ssh tig-gpu '/venv/main/bin/gpuq submit --project wgan-synthetic \
  --commit "'"$(git rev-parse HEAD)"'" --branch glove-v1-design --lane gpu \
  --artifact docs/datasets/deep_hub_stability.json \
  -- python -m src.eval.hub_stability \
      --real-path data/deep_1m.npy \
      --n 20000 --n 50000 --n 100000 --n 250000 \
      --draws 16 --k 100 --k-hub 10 --nlist 256 --seed 42 \
      --backend torch \
      --output docs/datasets/deep_hub_stability.json'
```

- [ ] **Step 3: Wait for both**

```bash
ssh tig-gpu '/venv/main/bin/gpuq wait <glove-id>; /venv/main/bin/gpuq wait <deep-id>'
```

Expected: both exit 0. The 250,000-row cells dominate; the pair should finish inside two hours.

- [ ] **Step 4: Read out the verdict table**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -c "
import json
for family in ('glove', 'deep'):
    data = json.load(open(f'docs/datasets/{family}_hub_stability.json'))
    print(family)
    for cell in data['cells']:
        for name in ('hubness_skew', 'hubness_gini', 'hub_share_top1pct'):
            v = cell['verdicts'][name]
            print(f\"  n={cell['n']:>6} {name:<18} \"
                  f\"range={v['range_pct_of_mean']:.2f}% \"
                  f\"sep={v['separation_in_ranges']} -> {v['verdict']}\")
"
```

- [ ] **Step 5: Commit both artifacts**

```bash
git add docs/datasets/glove_hub_stability.json docs/datasets/deep_hub_stability.json
git commit -m "measure: hub-statistic stability across N for glove and deep"
```

---

### Task 11: Write up the result in `docs/datasets/glove.md`

The page currently contains two sections asserting things about hubness that the sweep either confirms or overturns. Leaving them beside a new section that disagrees would make the page contradict itself, which `AGENTS.md` treats as worse than saying nothing.

**Files:**
- Modify: `docs/datasets/glove.md`

- [ ] **Step 1: Add the results section**

Insert a `## Hub statistic stability` section after `## Noise floor`, containing:

- The grid, in one sentence: two corpora, four N, 16 draws, `--backend torch`, measured at commit `<sha>`.
- A table with one row per (statistic, N) for the three hub statistics on GloVe: `range_pct_of_mean`, `separation_in_ranges`, `draws_disjoint`, verdict.
- The DEEP table beside it, condition 1 only, with a sentence saying it is evidence about generality rather than a vote on GloVe's gate.
- The overlap caveat, stated plainly: at N=100,000 and 250,000 the sixteen draws do not fit a 1M pool, so their spread is a lower bound and any pass there is `provisional`.
- The provenance note: the N=20,000 cell drawn from `glove_250k.npy` reproduced the committed eight-draw means, and `docs/datasets/glove_hub_stability_provenance.json` is the check.
- A note that the `v0` figures here come from a fresh 250,000-vector sampling of the same five checkpoints, not from the 50,000-vector samples behind `glove_v0_noise_floor.json`, and whether the two agree at N=20,000.
- The rule, restated with both constants, and the sentence that it was fixed before the sweep ran, citing the spec.

- [ ] **Step 2: Correct the two existing hubness sections**

`### Hubness skew is below the noise floor at this N` currently ends by saying the choice between fixes "needs a human. See the `## Gate` section." Replace that closing with a pointer to the new section and what it measured. Do not delete the eight-draw table: it is the measurement the new one is checked against.

`### Hubness skew is coarse, not useless` argues from the 3.463–8.331 versus 1.535–1.798 ranges. Whatever the sweep found, that argument is now either confirmed at more draws or superseded. Rewrite its conclusion to match the committed numbers and cross-reference the new section.

- [ ] **Step 3: Check the references resolve**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_docs_references.py -v`
Expected: PASS. Every path named in the new prose must exist.

- [ ] **Step 4: Full check**

Run: `cd /home/fibonadithya/.herdr/worktrees/wgan-synthetic/glove-v1-design && make check`
Expected: clean.

- [ ] **Step 5: Confirm no band was set**

Run: `grep -c "min: null" gates/glove.yaml`
Expected: `4` — every band still unset. This plan does not gate anything.

- [ ] **Step 6: Commit**

```bash
git add docs/datasets/glove.md
git commit -m "docs(glove): report which hub statistic can carry a band"
```

---

### Task 12: Open the PR and bring the verdict to a human

**Files:**
- No repository files change.

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin glove-v1-design
gh pr create --base glove-gan-v1 \
  --title "Measure which hub statistic GloVe can be gated on (#29)" \
  --body "$(cat <<'EOF'
`v0` misses all four gate statistics, and the one this family is most likely
to fail cannot currently be read: at the locked canonical N, hubness skew
measures the draw rather than the corpus. Issue #29 lists four fixes and says
choosing between them needs a measurement. This is that measurement.

The rule that decides was pre-registered in the design spec before any number
existed, and is applied in code by `hub_stability.evaluate_rule`, so the
verdict is in the committed JSON rather than in a reading of a table.

No band is set and no model is trained.

Based on `glove-gan-v1` (#46), which this needs for the v0 noise floor.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Watch the checks to green**

```bash
gh pr checks --watch
```

A PR is not done until its checks are green. Fix what fails and push again.

- [ ] **Step 3: Bring the pivot to the human**

Report the verdict table and which of the spec's three phase-2 shapes it selects under the pre-registered tie-break:

1. A candidate `qualified` at N=20,000 → adopt it, canonical conditions untouched.
2. Nothing qualified at 20,000, something `qualified` higher → split canonical N. Note that `provisional` is not `qualified`, and a provisional-only result selects shape 3, not shape 2.
3. Nothing qualified anywhere → gate GloVe on three statistics, as a measured decision.

**Do not implement phase 2 from this plan.** It gets its own plan, written against the real numbers.

---

## Self-Review

**Spec coverage.** The pre-registered rule → Task 6. The DEEP condition-1-only clarification → Task 6 (`synthetic_mean=None`) and Task 10 Step 2. The six statistics → Tasks 3 and 5. The grid and the 1M pools → Task 10. The provenance cell → Task 9. Sixteen draws → Task 10. The overlap bias and `provisional` → Tasks 4, 6 and 11. `v0` re-sampling → Task 8. The torch backend and `_exclude_self` → Tasks 1 and 2. Candidates kept out of `summary()` → Task 3 Step 5. The harness → Tasks 4–7. Both output JSONs → Task 10. All five tests named in the spec → Tasks 1–7. Success criteria → Tasks 9 (provenance), 10 (artifacts), 11 (docs, `make check`, bands still null). Phase 2 → deliberately out of scope, stated in `## Scope` and Task 12.

**Naming consistency.** `backend` and `chunk_rows` keep the same names and defaults across `knn`, `compute`, `measure_draw`, `sweep` and the CLI. `STATISTICS` is the single source of the six names and every aggregation iterates it. `range_pct_of_mean`, `mean`, `min`, `max` are `summarize_spread`'s own keys, not new ones. `draws_disjoint` is the same key in the allocator's return, the cell, and the verdict.

**Known gap, accepted.** Task 11's prose cannot be written verbatim in advance because it reports numbers that do not exist yet. Its steps specify exactly which claims each section must make and which committed file backs each one, which is as far as a plan can honestly go.
