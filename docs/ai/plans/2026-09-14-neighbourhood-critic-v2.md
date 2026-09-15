# Neighbourhood Critic, shared change + v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the WGAN-GP critic per-row within-batch neighbourhood features so it can see local dimension, and train NYTimes `v2` with it.

**Architecture:** A new `NeighbourhoodCritic` in `src/models/critic.py` computes each row's sorted, floored k-nearest within-batch distances, turns them into a log-ratio profile, and feeds `[x, profile]` to the existing MLP. A `build_critic` factory selects it by `model.critic_type`; the trainer's one construction site uses the factory. Two supporting changes: `data.preprocess.drop_zero_rows` removes exact-zero rows at load, and the gate logs a near-duplicate fraction plus the linear-skip generator's trunk/skip energies per evaluation.

**Tech Stack:** Python 3, PyTorch, numpy, pytest, ruff. Run everything with the project venv: locally `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python` (set `PYTHON=` for `make`), on the box `/opt/venvs/wgan-synthetic/bin/python`.

**Spec:** `docs/ai/specs/2026-09-14-neighbourhood-critic-design.md` (sections "Shared" and "Approach 1").

## Global Constraints

- `Critic` (the per-vector class) does not change; `tests/test_critic.py` keeps passing unchanged.
- `gradient_penalty` in `src/train/train_wgan_gp.py` does not change in behaviour; only its docstring and type hint.
- Neighbour computation runs in float32 with autocast disabled; distances use the expanded-square form with `clamp(min=0)`, never `torch.cdist`; self is excluded by index (an `inf` mask), never by dropping the nearest column.
- The floor is applied to *squared* distances as `clamp(min=floor**2)` before `sqrt`, so a zero distance never reaches `sqrt` (its gradient there is infinite).
- Defaults: `critic_type: per_vector`, `critic_k: 20`, `critic_distance_floor: 0.01`, `drop_zero_rows: false`. Every existing config trains exactly as before.
- Duplicates are kept. Only exact-zero rows are dropped, and only when the flag is on.
- Commit after every task with explicit paths (`git add <paths>`; never `-A`/`.`). Run `make check` (ruff + pytest) before each commit; the repo gate is `make check`, not `pytest` alone.
- `docs/ai/specs/*` is not edited. Deviations are recorded here under "Deviations".
- Never push `wgan-synthetic` until the tests are green; the gpuq runner needs the commit on the remote, so Task 10 pushes once, after Task 9.

## Deviations from the spec

None for approach 1. (The `v2b` plan carves the real bank out of the training split instead of index-excluding; see that plan.)

## File map

| file | change |
|---|---|
| `src/data/dataset.py` | `PreprocessConfig.drop_zero_rows`, `PreprocessState.dropped_zero_rows`, drop in `build_training_data` |
| `src/models/critic.py` | `DEFAULT_K`, `DEFAULT_DISTANCE_FLOOR`, `_masked_squared_distances`, `neighbourhood_distances`, `profile_features`, `NeighbourhoodCritic`, `CRITIC_TYPES`, `build_critic` |
| `src/models/generator.py` | `LinearSkipGenerator.component_energies` |
| `src/eval/ann_difficulty.py` | `AnnMetrics.nearest_distance` |
| `src/train/selection.py` | `near_duplicate_fraction` in `gate_statistics` and `LOGGED` |
| `src/train/train_wgan_gp.py` | factory at the construction site, `dropped_zero_rows` in metadata, floor passed to the gate, energies per eval |
| `tests/test_dataset_zero_rows.py` | new |
| `tests/test_neighbourhood_critic.py` | new |
| `tests/test_critic_factory.py` | new |
| `tests/test_selection.py`, `tests/test_train_smoke.py`, `tests/test_generator.py` | extended |
| `configs/nytimes/v2.yaml`, `configs/nytimes/v2_seed42.yaml`, `scripts/nytimes_v2_seed42_job.sh` | new |
| `PROJECT_DOCUMENTATION.md`, `docs/datasets/nytimes.md` | documented |

---

### Task 1: `drop_zero_rows` at load

**Files:**
- Modify: `src/data/dataset.py` (`PreprocessConfig` at line 46, `PreprocessState` at line 63, `build_training_data` at line 217)
- Modify: `src/train/train_wgan_gp.py:507-516` (`run_meta["data"]`)
- Test: `tests/test_dataset_zero_rows.py` (new)

**Interfaces:**
- Produces: `PreprocessConfig(drop_zero_rows: bool = False)`; `PreprocessState.dropped_zero_rows: int`; `build_training_data(...)` unchanged signature, drops rows before the split when the flag is on; `run_metadata.json["data"]["dropped_zero_rows"]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dataset_zero_rows.py
"""`data.preprocess.drop_zero_rows` removes exact-zero rows before the
train/holdout split, so both sides and the gate reference are clean."""

import numpy as np
import pytest

from src.data.dataset import PreprocessConfig, PreprocessState, build_training_data


def _corpus(tmp_path, n=10, dim=4, zero_rows=(3, 7)):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    for i in zero_rows:
        x[i] = 0.0
    path = tmp_path / "corpus.npy"
    np.save(path, x)
    return path


def _build(path, drop):
    return build_training_data(
        descriptor_path=str(path),
        file_format="npy",
        descriptor_dim=4,
        holdout_fraction=0.2,
        preprocess_cfg=PreprocessConfig(drop_zero_rows=drop),
        seed=0,
    )


def test_flag_defaults_to_false():
    assert PreprocessConfig().drop_zero_rows is False


def test_drop_zero_rows_removes_exactly_the_zero_rows_before_the_split(tmp_path):
    x_train, x_holdout, state = _build(_corpus(tmp_path), drop=True)
    assert x_train.shape[0] + x_holdout.shape[0] == 8
    assert state.dropped_zero_rows == 2
    # Nothing left on either side has zero norm.
    assert (np.linalg.norm(x_train, axis=1) > 0).all()
    assert (np.linalg.norm(x_holdout, axis=1) > 0).all()


def test_flag_off_keeps_every_row_and_reports_zero(tmp_path):
    x_train, x_holdout, state = _build(_corpus(tmp_path), drop=False)
    assert x_train.shape[0] + x_holdout.shape[0] == 10
    assert state.dropped_zero_rows == 0


def test_dropped_count_survives_serialisation_round_trip():
    state = PreprocessState(
        descriptor_dim=4, config=PreprocessConfig(drop_zero_rows=True), dropped_zero_rows=2
    )
    payload = state.to_serializable()
    assert payload["dropped_zero_rows"] == 2
    assert payload["config"]["drop_zero_rows"] is True
    restored = PreprocessState.from_serializable(payload)
    assert restored.dropped_zero_rows == 2
    assert restored.config.drop_zero_rows is True


def test_old_payload_without_the_field_loads_as_zero():
    payload = {
        "descriptor_dim": 4,
        "config": {"center": False, "whiten": False, "l2_normalize": True},
        "mean": None,
        "whitening_matrix": None,
    }
    assert PreprocessState.from_serializable(payload).dropped_zero_rows == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `PY=/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python; $PY -m pytest tests/test_dataset_zero_rows.py -v`
Expected: FAIL, `TypeError: PreprocessConfig.__init__() got an unexpected keyword argument 'drop_zero_rows'`.

- [ ] **Step 3: Implement**

In `src/data/dataset.py`, `PreprocessConfig` gains a field after `metric`:

```python
@dataclass
class PreprocessConfig:
    center: bool = False
    whiten: bool = False
    l2_normalize: bool = True
    eps: float = 1.0e-8
    metric: str = "l2"
    # Drop rows with exact L2 norm zero before the train/holdout split. Off by
    # default so every existing config trains on exactly the rows it did.
    # A zero row is an empty document after preprocessing: the gate already
    # drops it, the generator cannot emit one, and to a neighbourhood critic
    # it is a row at L2 exactly 1.0 from everything -- a shortcut with nothing
    # to emulate. Exact duplicates are NOT dropped: they are part of the
    # search target (docs/ai/specs/2026-09-14-neighbourhood-critic-design.md).
    drop_zero_rows: bool = False
```

`PreprocessState` gains a field with a default and carries it through both serialisers:

```python
@dataclass
class PreprocessState:
    descriptor_dim: int
    config: PreprocessConfig
    mean: np.ndarray | None = None
    whitening_matrix: np.ndarray | None = None
    # How many exact-zero rows build_training_data removed (0 unless
    # config.drop_zero_rows). Recorded so run_metadata.json says how many
    # rows the run actually trained on.
    dropped_zero_rows: int = 0
```

`from_serializable` adds `dropped_zero_rows=int(payload.get("dropped_zero_rows", 0)),` to the `cls(...)` call. `to_serializable` needs no change: `asdict` already includes the new int.

In `build_training_data`, after the dim check and before the split:

```python
    if x.shape[1] != descriptor_dim:
        raise ValueError(f"Expected descriptor dim {descriptor_dim}, got {x.shape[1]}")

    dropped_zero_rows = 0
    if preprocess_cfg.drop_zero_rows:
        keep = np.linalg.norm(x, axis=1) > 0.0
        dropped_zero_rows = int(np.count_nonzero(~keep))
        if dropped_zero_rows:
            x = x[keep]

    x_train_raw, x_holdout_raw = train_holdout_split(
        x, holdout_fraction=holdout_fraction, seed=seed
    )
    state = _fit_preprocess_state(
        x_train=x_train_raw, descriptor_dim=descriptor_dim, cfg=preprocess_cfg
    )
    state.dropped_zero_rows = dropped_zero_rows
```

In `src/train/train_wgan_gp.py`, the `run_meta` dict's `"data"` block gains one entry:

```python
        "data": {
            "num_train": int(x_train.shape[0]),
            "num_holdout": int(x_holdout.shape[0]),
            "descriptor_dim": descriptor_dim,
            "dropped_zero_rows": int(preprocess_state.dropped_zero_rows),
        },
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_dataset_zero_rows.py tests/test_dataset_metric.py tests/test_invert_preprocess.py tests/test_train_smoke.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset.py src/train/train_wgan_gp.py tests/test_dataset_zero_rows.py
git commit -m "feat(data): drop_zero_rows preprocess flag, off by default, counted in run metadata"
```

---

### Task 2: `neighbourhood_distances` and `profile_features`

**Files:**
- Modify: `src/models/critic.py`
- Test: `tests/test_neighbourhood_critic.py` (new)

**Interfaces:**
- Produces:
  - `DEFAULT_K: int = 20`, `DEFAULT_DISTANCE_FLOOR: float = 0.01`
  - `_masked_squared_distances(x: Tensor, bank: Tensor | None, self_index: Tensor | None) -> Tensor` of shape `(n, m)`, float32, `inf` where excluded.
  - `neighbourhood_distances(x: Tensor, k: int, floor: float, bank: Tensor | None = None, self_index: Tensor | None = None) -> Tensor` of shape `(n, k)`, sorted ascending, every entry `>= floor` (up to float32 rounding of `sqrt(floor**2)`).
  - `profile_features(r: Tensor) -> Tensor` of shape `(n, k)`: `log(r_i / r_k)` for `i < k`, then `log r_k`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_neighbourhood_critic.py
"""The within-batch neighbourhood features and the critic that consumes them.

Every test names the mutation it catches in its docstring. The per-vector
`Critic` is untouched and keeps its own tests in test_critic.py.
"""

import math

import numpy as np
import pytest
import torch

from src.models.critic import (
    DEFAULT_DISTANCE_FLOOR,
    DEFAULT_K,
    neighbourhood_distances,
    profile_features,
)


def _unit_batch(seed: int, n: int, dim: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return torch.from_numpy(x)


def test_defaults_are_the_spec_values():
    assert DEFAULT_K == 20
    assert DEFAULT_DISTANCE_FLOOR == 0.01


def test_distances_have_shape_n_by_k_and_are_sorted_ascending():
    """Catches a wrong topk axis or an unsorted topk."""
    x = _unit_batch(0, n=12, dim=5)
    r = neighbourhood_distances(x, k=4, floor=0.01)
    assert r.shape == (12, 4)
    assert (r[:, 1:] >= r[:, :-1]).all()


def test_distances_match_brute_force_on_a_batch_without_ties():
    """Catches an expanded-square sign error or a lost self-mask column."""
    x = _unit_batch(1, n=9, dim=6)
    r = neighbourhood_distances(x, k=3, floor=1e-6)
    full = torch.cdist(x, x)
    full.fill_diagonal_(float("inf"))
    expected, _ = torch.topk(full, 3, dim=1, largest=False, sorted=True)
    torch.testing.assert_close(r, expected, atol=1e-5, rtol=1e-5)


def test_self_is_excluded_by_index():
    """Catches a missing diagonal mask: every row's own distance is 0, which
    would read as the floor."""
    x = 2.0 * torch.eye(6)  # pairwise distance 2*sqrt(2), well above 0.5
    r = neighbourhood_distances(x, k=2, floor=0.01)
    assert r.min() > 0.5


def _batch_with_an_exact_copy(seed: int, n: int, dim: int) -> torch.Tensor:
    """Rows 0 and 1 are the same one-hot vector. One-hot, not a random unit
    row: the expanded-square distance of a random copy rounds to ~1e-7, not
    0, so a test built on it would pass even without the clamp. For a
    one-hot pair sq + sq - 2*dot is exactly 1 + 1 - 2 = 0."""
    x = _unit_batch(seed, n, dim)
    x[0] = 0.0
    x[0, 0] = 1.0
    x[1] = x[0]
    return x


def test_exact_copies_read_as_the_floor_not_zero_or_nan():
    """Catches a lost clamp (would be 0, then log gives -inf) and a clamp
    applied after sqrt (gradient at sqrt(0) is infinite)."""
    x = _batch_with_an_exact_copy(2, n=8, dim=5)
    r = neighbourhood_distances(x, k=2, floor=0.01)
    assert r[0, 0].item() == pytest.approx(0.01, abs=1e-8)
    assert r[1, 0].item() == pytest.approx(0.01, abs=1e-8)
    assert torch.isfinite(profile_features(r)).all()


def test_floor_gradient_is_finite_through_a_copy():
    """Catches sqrt-before-clamp: d sqrt(0)/dx is inf and poisons the batch."""
    x = _batch_with_an_exact_copy(3, n=8, dim=5).requires_grad_(True)
    r = neighbourhood_distances(x, k=2, floor=0.01)
    r.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_floor_does_not_touch_distances_above_it():
    """Catches a clamp on the wrong side (min vs max)."""
    x = _unit_batch(4, n=10, dim=8)
    r_low = neighbourhood_distances(x, k=3, floor=1e-6)
    r_high = neighbourhood_distances(x, k=3, floor=0.01)
    torch.testing.assert_close(r_low, r_high)


def test_within_batch_requires_more_than_k_rows():
    """Catches silent truncation of k, which would change the feature width."""
    x = _unit_batch(5, n=4, dim=3)
    with pytest.raises(ValueError, match="k"):
        neighbourhood_distances(x, k=4, floor=0.01)


def test_bank_neighbours_come_from_the_bank_not_the_batch():
    """Catches a bank argument that is accepted and ignored."""
    x = _unit_batch(6, n=3, dim=4)
    bank = _unit_batch(7, n=20, dim=4)
    r = neighbourhood_distances(x, k=5, floor=1e-6, bank=bank)
    expected, _ = torch.topk(torch.cdist(x, bank), 5, dim=1, largest=False, sorted=True)
    assert r.shape == (3, 5)
    torch.testing.assert_close(r, expected, atol=1e-5, rtol=1e-5)


def test_self_index_excludes_a_row_from_a_bank_that_contains_it():
    """Catches self_index accepted and ignored: with the bank equal to the
    batch, every row would see itself at distance 0."""
    x = 2.0 * torch.eye(6)
    r = neighbourhood_distances(
        x, k=2, floor=0.01, bank=x, self_index=torch.arange(6)
    )
    assert r.min() > 0.5


def test_self_index_minus_one_means_not_in_bank():
    """Catches a mask applied at index -1 (which numpy/torch read as the last
    row) instead of being skipped."""
    x = 2.0 * torch.eye(6)
    bank = x.clone()
    self_index = torch.full((6,), -1, dtype=torch.long)
    r = neighbourhood_distances(x, k=1, floor=0.01, bank=bank, self_index=self_index)
    # No exclusion: each row finds its own copy in the bank at the floor.
    assert r.max().item() == pytest.approx(0.01, abs=1e-8)


def test_profile_is_log_ratios_then_log_scale():
    """Catches a wrong divisor (r_1 instead of r_k) or a dropped scale entry."""
    r = torch.tensor([[1.0, 2.0, 4.0]])
    phi = profile_features(r)
    expected = torch.tensor([[math.log(0.25), math.log(0.5), math.log(4.0)]])
    torch.testing.assert_close(phi, expected)
    assert phi.shape == (1, 3)


def test_profile_runs_in_float32_under_autocast():
    """Catches the profile inheriting fp16 from an enclosing autocast region,
    where the expanded-square form cancels catastrophically."""
    x = _unit_batch(8, n=16, dim=8)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        r = neighbourhood_distances(x, k=3, floor=0.01)
    assert r.dtype == torch.float32
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_neighbourhood_critic.py -v`
Expected: FAIL at import, `ImportError: cannot import name 'DEFAULT_DISTANCE_FLOOR'`.

- [ ] **Step 3: Implement**

Append to `src/models/critic.py` (keep the existing `Critic` class exactly as it is; add `from torch import Tensor` to the imports):

```python
# Spec: docs/ai/specs/2026-09-14-neighbourhood-critic-design.md.
# k=20 separates the collapsed sheet from the corpus per row at 0.998 and
# keeps the real-vs-Gaussian ordering; the floor sits inside the measured
# gap between exact copies (~1e-8 after rounding) and the nearest genuine
# pair (first percentile 0.90), so it changes the feature of a copy and of
# nothing else.
DEFAULT_K = 20
DEFAULT_DISTANCE_FLOOR = 0.01


def _masked_squared_distances(
    x: Tensor, bank: Tensor | None, self_index: Tensor | None
) -> Tensor:
    """Squared L2 distances from each row of `x` to each row of `bank`
    (default: `x` itself), float32, with excluded entries set to +inf.

    Expanded squares with `clamp(min=0)` rather than `torch.cdist`: cdist's
    gradient is undefined at distance zero, and a single pair of exact
    copies -- present in NYTimes, and produced by a collapsing generator --
    would poison the whole backward pass. Self-exclusion is by index (an
    additive inf mask, so no in-place write on a tensor autograd saved), never
    by dropping the nearest column: an exact copy ties with the query at zero
    and would otherwise be dropped in its place.

    `self_index[i]` is the bank row that *is* `x[i]`, or -1 for "not in the
    bank". Only used when `bank` is given; within-batch the diagonal is the
    self index.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        q = x.float()
        b = q if bank is None else bank.float()
        sq_q = (q * q).sum(dim=1)
        sq_b = sq_q if bank is None else (b * b).sum(dim=1)
        d2 = sq_q[:, None] + sq_b[None, :] - 2.0 * (q @ b.T)
        d2 = d2.clamp(min=0.0)
        mask = torch.zeros_like(d2)
        if bank is None:
            mask.fill_diagonal_(float("inf"))
        elif self_index is not None:
            present = self_index >= 0
            rows = torch.arange(q.shape[0], device=q.device)[present]
            mask[rows, self_index[present]] = float("inf")
        return d2 + mask


def neighbourhood_distances(
    x: Tensor,
    k: int,
    floor: float,
    bank: Tensor | None = None,
    self_index: Tensor | None = None,
) -> Tensor:
    """Sorted distances from each row of `x` to its `k` nearest rows of
    `bank` (default: the other rows of `x`), each floored at `floor`.

    The floor is applied to the squared distances before the sqrt, so a
    zero distance never reaches sqrt (whose gradient there is infinite) and
    an exact copy reads as exactly `floor`, a bounded "tight pair" feature a
    continuous generator can match by producing near-copies.

    Raises when fewer than `k` candidates exist rather than truncating: the
    feature width is the critic's input width and must not change silently.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if floor <= 0.0:
        raise ValueError(f"floor must be positive, got {floor}")
    n_candidates = x.shape[0] - 1 if bank is None else bank.shape[0]
    if n_candidates < k:
        raise ValueError(
            f"need at least k={k} neighbour candidates per row, got {n_candidates} "
            f"({'within-batch, batch of ' + str(x.shape[0]) if bank is None else 'bank'})"
        )
    with torch.autocast(device_type=x.device.type, enabled=False):
        d2 = _masked_squared_distances(x, bank, self_index)
        r2, _ = torch.topk(d2, k, dim=1, largest=False, sorted=True)
        r2 = r2.clamp(min=float(floor) ** 2)
        return r2.sqrt()


def profile_features(r: Tensor) -> Tensor:
    """`[log(r_1/r_k), ..., log(r_{k-1}/r_k), log r_k]` per row: the Hill
    estimator's sufficient statistic (shape) plus the neighbourhood scale.
    `r` must be positive everywhere, which `neighbourhood_distances` ensures.
    """
    r_k = r[:, -1:]
    return torch.cat([torch.log(r[:, :-1] / r_k), torch.log(r_k)], dim=1)
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_neighbourhood_critic.py tests/test_critic.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/critic.py tests/test_neighbourhood_critic.py
git commit -m "feat(critic): floored within-batch neighbour distances and the log-ratio profile"
```

---

### Task 3: `NeighbourhoodCritic`

**Files:**
- Modify: `src/models/critic.py`
- Test: `tests/test_neighbourhood_critic.py`

**Interfaces:**
- Consumes: `neighbourhood_distances`, `profile_features` (Task 2); `Critic`.
- Produces: `NeighbourhoodCritic(input_dim: int, hidden_dims: Iterable[int], k: int = DEFAULT_K, distance_floor: float = DEFAULT_DISTANCE_FLOOR, negative_slope: float = 0.2)` with `forward(x: (n, D)) -> (n,)`, `features(x) -> (n, k)`, attributes `k`, `distance_floor`, submodule `mlp: Critic` whose `input_dim` is `D + k`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_neighbourhood_critic.py`; extend the import block with `NeighbourhoodCritic` and add `from src.train.train_wgan_gp import gradient_penalty`)

```python
def test_critic_emits_one_score_per_row():
    """Catches a wrong squeeze or reduction axis."""
    critic = NeighbourhoodCritic(input_dim=5, hidden_dims=[8, 4], k=3)
    assert critic(_unit_batch(10, n=7, dim=5)).shape == (7,)


def test_critic_input_width_is_dim_plus_k():
    """Catches the profile computed but not concatenated."""
    critic = NeighbourhoodCritic(input_dim=5, hidden_dims=[8], k=3)
    first = next(m for m in critic.mlp.net if isinstance(m, torch.nn.Linear))
    assert first.in_features == 5 + 3


def test_features_have_width_k():
    critic = NeighbourhoodCritic(input_dim=5, hidden_dims=[8], k=4)
    assert critic.features(_unit_batch(11, n=9, dim=5)).shape == (9, 4)


def test_scores_depend_on_the_rest_of_the_batch():
    """The mirror image of test_critic.py's independence test. Catches the
    class silently degenerating to per-vector (features not wired).

    k = n - 1 so every row's profile uses every other row; moving one row
    then changes every other row's r_k or a ratio."""
    torch.manual_seed(0)
    critic = NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=7)
    x = _unit_batch(12, n=8, dim=4)
    base = critic(x)
    moved = x.clone()
    moved[7] = moved[7] + 1.0
    assert (critic(moved)[:7] - base[:7]).abs().max() > 1e-6


def test_permuting_the_batch_permutes_the_scores():
    """Catches a topk/gather index bug pairing a row with another's profile."""
    torch.manual_seed(0)
    critic = NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=3)
    x = _unit_batch(13, n=10, dim=4)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(critic(x[perm]), critic(x)[perm], atol=1e-5, rtol=1e-5)


def test_gradient_penalty_is_finite_and_double_backward_works():
    """Catches a non-differentiable op in the profile path: the penalty
    needs d(grad norm)/d(params), i.e. create_graph=True through topk,
    clamp, sqrt and log."""
    torch.manual_seed(0)
    critic = NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=3)
    real = _unit_batch(14, n=8, dim=4)
    fake = _unit_batch(15, n=8, dim=4)
    gp = gradient_penalty(critic, real, fake, device=torch.device("cpu"))
    assert torch.isfinite(gp)
    gp.backward()
    # The head's bias never gets gradient from the penalty (it does not
    # affect d score / d input), on any critic. Check the first layer, which
    # the profile feeds, and that whatever grads exist are finite.
    first = next(m for m in critic.mlp.net if isinstance(m, torch.nn.Linear))
    assert first.weight.grad is not None and torch.isfinite(first.weight.grad).all()
    assert first.weight.grad[:, 4:].abs().sum() > 0  # the profile columns
    assert all(torch.isfinite(p.grad).all() for p in critic.parameters() if p.grad is not None)


def test_gradient_penalty_reaches_neighbour_rows():
    """With a batch-dependent critic the per-row gradient is of the summed
    batch score, so it includes how row i moves other rows' features. Catches
    a forward that detaches the profile."""
    torch.manual_seed(0)
    critic = NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=7)
    x = _unit_batch(16, n=8, dim=4).requires_grad_(True)
    critic(x)[0].backward()  # only row 0's score...
    # ...yet other rows get gradient, because they are row 0's neighbours.
    assert x.grad[1:].abs().sum() > 0


def test_rejects_k_below_two_and_nonpositive_floor():
    """k=1 leaves no ratio entries; the profile would be scale only."""
    with pytest.raises(ValueError, match="k"):
        NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=1)
    with pytest.raises(ValueError, match="floor"):
        NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=3, distance_floor=0.0)


def test_forward_refuses_a_batch_no_larger_than_k():
    critic = NeighbourhoodCritic(input_dim=4, hidden_dims=[6], k=5)
    with pytest.raises(ValueError, match="k"):
        critic(_unit_batch(17, n=5, dim=4))
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_neighbourhood_critic.py -v`
Expected: the new tests FAIL with `ImportError: cannot import name 'NeighbourhoodCritic'`.

- [ ] **Step 3: Implement** (append to `src/models/critic.py`)

```python
class NeighbourhoodCritic(nn.Module):
    """The per-vector MLP on `[x_i, phi_i]`, where `phi_i` is row i's
    within-batch neighbourhood profile (`profile_features` of
    `neighbourhood_distances`).

    Batch-dependent by design: a per-vector critic cannot see local
    dimension, so a low-rank sheet with the right covariance envelope is,
    to it, the corpus (NYTimes v1 collapsed to LID 5 with the Wasserstein
    estimate under 0.06 throughout). Real rows are profiled among real
    batch-mates and fake among fake, so the within-batch bias in the
    distances is identical on both sides and cancels.

    Under `gradient_penalty` the per-row gradient is therefore of the
    batch's *summed* score, which includes how row i moves every other
    row's features. That is the intended Lipschitz constraint for a
    minibatch-dependent critic; do not "fix" it back to per-row.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        k: int = DEFAULT_K,
        distance_floor: float = DEFAULT_DISTANCE_FLOOR,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        if k < 2:
            raise ValueError(f"critic_k must be at least 2 (one ratio entry), got {k}")
        if distance_floor <= 0.0:
            raise ValueError(f"critic_distance_floor must be positive, got {distance_floor}")
        self.k = int(k)
        self.distance_floor = float(distance_floor)
        self.mlp = Critic(
            input_dim=input_dim + self.k,
            hidden_dims=hidden_dims,
            negative_slope=negative_slope,
        )

    def features(self, x: Tensor) -> Tensor:
        r = neighbourhood_distances(x, self.k, self.distance_floor)
        return profile_features(r)

    def forward(self, x: Tensor) -> Tensor:
        phi = self.features(x).to(x.dtype)
        return self.mlp(torch.cat([x, phi], dim=1))
```

Also in `src/train/train_wgan_gp.py`, loosen the type hints and record the formulation. `gradient_penalty`'s signature becomes `critic: nn.Module` (import `nn` from torch if not already) and it gains a docstring:

```python
def gradient_penalty(
    critic: nn.Module, real: Tensor, fake: Tensor, device: torch.device
) -> Tensor:
    """WGAN-GP penalty on row-wise interpolates of `real` and `fake`.

    The gradient is of the *summed* critic output with respect to each
    interpolated row. For a per-vector critic that is each row's own score
    gradient. For a batch-dependent critic (`NeighbourhoodCritic` and its
    relatives) it also includes how row i moves every other row's
    neighbourhood features; the penalty then bounds the Lipschitz constant
    of the summed batch score, one row at a time. That is the intended
    formulation for a minibatch-dependent critic, and the interpolated batch
    mixing real and fake neighbourhoods is fine: the penalty is about the
    critic's smoothness, not the population the batch came from.
    """
```

`save_checkpoint`'s `critic: Critic` hint also becomes `nn.Module`.

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_neighbourhood_critic.py tests/test_critic.py tests/test_train_smoke.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/critic.py src/train/train_wgan_gp.py tests/test_neighbourhood_critic.py
git commit -m "feat(critic): NeighbourhoodCritic, the per-vector MLP on [x, within-batch profile]"
```

---

### Task 4: `build_critic` factory and the trainer construction site

**Files:**
- Modify: `src/models/critic.py`
- Modify: `src/train/train_wgan_gp.py:25` (import) and `:454-458` (construction)
- Test: `tests/test_critic_factory.py` (new), `tests/test_train_smoke.py`

**Interfaces:**
- Produces: `CRITIC_TYPES = ("per_vector", "neighbourhood")`; `build_critic(model_cfg: Mapping[str, Any], input_dim: int) -> nn.Module`.
- Later plans (`v2b`, `v2c`) add their type string to `CRITIC_TYPES` and a branch to `build_critic`; nothing else in this task changes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_critic_factory.py
"""`build_critic` mirrors `build_generator`: the config's `critic_type`
picks the class, and an unknown string fails loudly rather than silently
training the per-vector critic."""

import pytest
import torch

from src.models.critic import (
    CRITIC_TYPES,
    Critic,
    NeighbourhoodCritic,
    build_critic,
)

BASE_CFG = {"critic_hidden_dims": [16, 8], "negative_slope": 0.2}


def test_the_documented_types():
    assert CRITIC_TYPES == ("per_vector", "neighbourhood")


def test_missing_critic_type_defaults_to_the_per_vector_critic():
    critic = build_critic(dict(BASE_CFG), input_dim=12)
    assert type(critic) is Critic
    first = next(m for m in critic.net if isinstance(m, torch.nn.Linear))
    assert first.in_features == 12


def test_explicit_per_vector():
    assert type(build_critic(dict(BASE_CFG, critic_type="per_vector"), input_dim=12)) is Critic


def test_neighbourhood_reads_k_and_floor_from_the_config():
    cfg = dict(BASE_CFG, critic_type="neighbourhood", critic_k=7, critic_distance_floor=0.02)
    critic = build_critic(cfg, input_dim=12)
    assert isinstance(critic, NeighbourhoodCritic)
    assert critic.k == 7
    assert critic.distance_floor == 0.02


def test_neighbourhood_defaults_k_20_and_floor_0_01():
    critic = build_critic(dict(BASE_CFG, critic_type="neighbourhood"), input_dim=12)
    assert (critic.k, critic.distance_floor) == (20, 0.01)


def test_negative_slope_reaches_both_classes():
    for kind in CRITIC_TYPES:
        critic = build_critic(dict(BASE_CFG, critic_type=kind, negative_slope=0.31), input_dim=6)
        net = critic.net if isinstance(critic, Critic) else critic.mlp.net
        slopes = {m.negative_slope for m in net if isinstance(m, torch.nn.LeakyReLU)}
        assert slopes == {0.31}, kind


def test_unknown_type_raises_naming_the_value():
    with pytest.raises(ValueError, match="vibes"):
        build_critic(dict(BASE_CFG, critic_type="vibes"), input_dim=12)


def test_state_dicts_do_not_cross_load():
    """A resume must not silently load per-vector weights into the
    neighbourhood critic or vice versa."""
    a = build_critic(dict(BASE_CFG, critic_type="per_vector"), input_dim=12)
    b = build_critic(dict(BASE_CFG, critic_type="neighbourhood", critic_k=4), input_dim=12)
    with pytest.raises(RuntimeError):
        a.load_state_dict(b.state_dict())
    with pytest.raises(RuntimeError):
        b.load_state_dict(a.state_dict())
```

And in `tests/test_train_smoke.py`, add after `test_mlp_config_without_generator_type_still_trains`:

```python
def test_neighbourhood_critic_trains_and_its_checkpoint_reloads(tmp_path):
    """Catches the trainer not using the factory (the config would be
    ignored) and a state-dict key mismatch on resume."""
    from src.models.critic import NeighbourhoodCritic, build_critic

    cfg = make_config(tmp_path, "mlp")
    cfg["model"]["critic_type"] = "neighbourhood"
    cfg["model"]["critic_k"] = 5  # batch_size is 32; k must be below it
    ckpt_path, meta = train(cfg)
    assert ckpt_path.exists()
    for entry in meta["metrics"]:
        assert math.isfinite(entry["d_loss"]) and math.isfinite(entry["gp"])
    saved = torch.load(ckpt_path, weights_only=False)
    rebuilt = build_critic(cfg["model"], input_dim=16)
    assert isinstance(rebuilt, NeighbourhoodCritic)
    rebuilt.load_state_dict(saved["critic_state_dict"])


def test_run_metadata_records_dropped_zero_rows(tmp_path):
    cfg = make_config(tmp_path, "mlp")
    cfg["data"]["preprocess"]["drop_zero_rows"] = True
    _, meta = train(cfg)
    # Synthetic Gaussian data has no zero rows; the key must still be there.
    assert meta["data"]["dropped_zero_rows"] == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_critic_factory.py tests/test_train_smoke.py -k "factory or neighbourhood_critic or dropped_zero" -v`
Expected: FAIL, `ImportError: cannot import name 'CRITIC_TYPES'`; the smoke test fails because the trainer builds `Critic` regardless (the `isinstance` assertion).

- [ ] **Step 3: Implement**

Append to `src/models/critic.py` (add `from collections.abc import Iterable, Mapping` and `from typing import Any` to the imports):

```python
CRITIC_TYPES = ("per_vector", "neighbourhood")


def build_critic(model_cfg: Mapping[str, Any], input_dim: int) -> nn.Module:
    """Build the configured critic, defaulting to the per-vector `Critic`.

    Mirrors `build_generator`: `critic_type` selects the class, the common
    keys mean the same for every class, and an unknown value fails here
    rather than silently training the default.
    """
    kind = str(model_cfg.get("critic_type", "per_vector"))
    common = {
        "hidden_dims": model_cfg["critic_hidden_dims"],
        "negative_slope": float(model_cfg["negative_slope"]),
    }
    if kind == "per_vector":
        return Critic(input_dim=input_dim, **common)
    if kind == "neighbourhood":
        return NeighbourhoodCritic(
            input_dim=input_dim,
            k=int(model_cfg.get("critic_k", DEFAULT_K)),
            distance_floor=float(
                model_cfg.get("critic_distance_floor", DEFAULT_DISTANCE_FLOOR)
            ),
            **common,
        )
    raise ValueError(f"Unknown critic_type: {kind!r}; expected one of {CRITIC_TYPES}")
```

In `src/train/train_wgan_gp.py`, change the import at line 25 from `from src.models.critic import Critic` to `from src.models.critic import build_critic`, and replace the construction:

```python
    generator = build_generator(model_cfg, output_dim=descriptor_dim).to(device)
    critic = build_critic(model_cfg, input_dim=descriptor_dim).to(device)
```

Any remaining `Critic` type hints in the trainer become `nn.Module` (Task 3 already did `gradient_penalty` and `save_checkpoint`; grep for `Critic` to confirm none remain).

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_critic_factory.py tests/test_train_smoke.py tests/test_neighbourhood_critic.py tests/test_critic.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/critic.py src/train/train_wgan_gp.py tests/test_critic_factory.py tests/test_train_smoke.py
git commit -m "feat(critic): build_critic factory; trainer selects the critic by model.critic_type"
```

---

### Task 5: near-duplicate fraction in the gate

**Files:**
- Modify: `src/eval/ann_difficulty.py:246-261` (`AnnMetrics`) and `:312-321` (`compute` return)
- Modify: `src/train/selection.py` (`LOGGED`, `gate_statistics`)
- Modify: `src/train/train_wgan_gp.py` (both `gate_statistics` calls, at `:586` and `:742`)
- Test: `tests/test_selection.py`

**Interfaces:**
- Produces: `AnnMetrics.nearest_distance: np.ndarray | None` (one entry per measured row, before the survivor mask); `gate_statistics(x, *, metric, seed, distance_floor: float = DEFAULT_DISTANCE_FLOOR)` with the new logged key `near_duplicate_fraction`; eval entries `gate_real_near_duplicate_fraction` / `gate_fake_near_duplicate_fraction`.

- [ ] **Step 1: Write the failing tests** (in `tests/test_selection.py`)

Update the existing key-set assertion in `test_gate_statistics_returns_the_four_statistics_and_the_discard_count` to include `"near_duplicate_fraction"`. Then add:

```python
def test_gate_statistics_logs_the_fraction_of_rows_within_the_floor_of_a_neighbour():
    """Catches the diagnostic reading a survivor-masked column (copies are
    exactly the rows the mask drops) or the wrong column of dist."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((200, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    for i in range(1, 6):
        x[i] = x[0]  # rows 0..5 each have a neighbour at distance 0: 6 of 200
    stats = gate_statistics(x, metric="angular", seed=0)
    assert stats["near_duplicate_fraction"] == pytest.approx(0.03)


def test_near_duplicate_fraction_uses_the_given_floor():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((200, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    # Random unit vectors in 16-D sit far above 0.01 from each other, and
    # far below 3.0 (the diameter of the unit sphere is 2).
    assert gate_statistics(x, metric="angular", seed=0)["near_duplicate_fraction"] == 0.0
    assert (
        gate_statistics(x, metric="angular", seed=0, distance_floor=3.0)[
            "near_duplicate_fraction"
        ]
        == 1.0
    )
```

And extend the loop in `test_gate_statistics_drops_exact_zero_rows_and_reports_the_counts` to include `"near_duplicate_fraction"` in the keys that must be equal with and without zero rows.

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_selection.py -v`
Expected: FAIL, `KeyError: 'near_duplicate_fraction'` and the key-set mismatch.

- [ ] **Step 3: Implement**

`src/eval/ann_difficulty.py`, `AnnMetrics` gains a trailing optional field so the two test fixtures that construct it by hand keep working:

```python
    discarded_queries: int
    # Distance from each measured row to its nearest other row, one entry
    # per row, before survivor_mask. Optional so hand-built fixtures that
    # predate it still construct; compute() always fills it.
    nearest_distance: np.ndarray | None = None
```

and `compute` returns it: add `nearest_distance=dist[:, 0].copy(),` to the `AnnMetrics(...)` call.

`src/train/selection.py`: import the floor and log the fraction.

```python
from src.eval import ann_difficulty
from src.models.critic import DEFAULT_DISTANCE_FLOOR
```

`LOGGED` gains `"near_duplicate_fraction"` (after `"ivf_gini"`). `gate_statistics` gains the keyword and the line:

```python
def gate_statistics(
    x: np.ndarray,
    *,
    metric: str,
    seed: int,
    distance_floor: float = DEFAULT_DISTANCE_FLOOR,
) -> dict[str, float | int | None]:
    """...(existing docstring)...

    `near_duplicate_fraction` is the share of measured rows whose nearest
    other row is within `distance_floor`, the same constant the
    neighbourhood critic floors its distances at. It is logged, not
    scored: it says whether a run learned to make near-copies. Measured on
    the holdout, so the real-side figure is far below the corpus's 14%
    (a copy is only counted if its twin is also in the holdout).
    """
    ...
    s = ann_difficulty.summary(m)
    s["zero_rows"] = zero_rows
    s["measured_rows"] = int(measured.shape[0])
    s["near_duplicate_fraction"] = (
        float(np.mean(m.nearest_distance <= distance_floor))
        if m.nearest_distance is not None and m.nearest_distance.size
        else None
    )
    return {k: s[k] for k in LOGGED}
```

`src/train/train_wgan_gp.py`: read the floor once next to the other model keys and pass it to both calls.

```python
    from src.models.critic import DEFAULT_DISTANCE_FLOOR  # at the top with the other imports
    ...
    gate_distance_floor = float(
        model_cfg.get("critic_distance_floor", DEFAULT_DISTANCE_FLOOR)
    )
```

Both `gate_statistics(x_holdout, metric=metric, seed=seed)` and `gate_statistics(fake_holdout, metric=metric, seed=seed)` gain `distance_floor=gate_distance_floor`.

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_selection.py tests/test_ann_difficulty.py tests/test_eda_notes.py tests/test_eda_run.py tests/test_train_smoke.py -v`
Expected: all PASS (the gate smoke test's key loop still passes: it iterates the four names it lists, and the new key is extra).

- [ ] **Step 5: Commit**

```bash
git add src/eval/ann_difficulty.py src/train/selection.py src/train/train_wgan_gp.py tests/test_selection.py
git commit -m "feat(gate): log near_duplicate_fraction at the critic's distance floor, real and fake"
```

---

### Task 6: trunk/skip energies per evaluation

**Files:**
- Modify: `src/models/generator.py` (`LinearSkipGenerator`, forward at `:96-97`)
- Modify: `src/train/train_wgan_gp.py` (eval block at `:731-745`)
- Test: `tests/test_generator.py`, `tests/test_train_smoke.py`

**Interfaces:**
- Produces: `LinearSkipGenerator.component_energies(z: Tensor) -> dict[str, float]` with keys `trunk_energy`, `skip_energy`, `skip_share`, `trunk_skip_abs_cos`; the same four keys on every eval entry of a `linear_skip` run.

- [ ] **Step 1: Write the failing tests**

In `tests/test_generator.py` (it already imports `LinearSkipGenerator`; add `import torch` if absent):

```python
def test_component_energies_read_the_trunk_and_skip_terms_separately():
    """Catches the two terms swapped, or the share computed against the
    normalised output instead of the pre-normalisation sum."""
    torch.manual_seed(0)
    gen = LinearSkipGenerator(
        latent_dim=8, output_dim=6, hidden_dims=[8], negative_slope=0.2, skip_dim=4
    )
    z = torch.randn(256, 8)
    with torch.no_grad():
        gen.skip.weight.zero_()
    e = gen.component_energies(z)
    assert set(e) == {"trunk_energy", "skip_energy", "skip_share", "trunk_skip_abs_cos"}
    assert e["skip_energy"] == 0.0
    assert e["skip_share"] == 0.0
    assert e["trunk_energy"] > 0.0

    with torch.no_grad():
        torch.nn.init.orthogonal_(gen.skip.weight)
        for p in gen.trunk.parameters():
            p.zero_()
    e = gen.component_energies(z)
    assert e["trunk_energy"] == 0.0
    assert e["skip_share"] == 1.0
    # Orthogonal 6x4 W on 4 unit-variance latents: E||Wz||^2 = 4.
    assert e["skip_energy"] == pytest.approx(4.0, rel=0.2)
```

In `tests/test_train_smoke.py`: add `"linear_skip"` to the `test_training_loop_runs` parametrize list, teach `make_config` to set `skip_dim` for it, and add a test:

```python
    if generator_type is not None:
        cfg["model"]["generator_type"] = generator_type
        if generator_type == "structured_gated":
            cfg["model"]["layout"] = [2, 2, 4]
        if generator_type == "linear_skip":
            cfg["model"]["skip_dim"] = 4  # latent_dim is 8: 4 trunk + 4 skip
```

```python
def test_linear_skip_evals_log_the_trunk_skip_balance(tmp_path):
    """Catches the energies not reaching the eval entry, or reaching it for
    the wrong generator type."""
    _, meta = train(make_config(tmp_path, "linear_skip"))
    for e in meta["eval"]:
        assert 0.0 <= e["skip_share"] <= 1.0
        assert math.isfinite(e["trunk_energy"]) and math.isfinite(e["skip_energy"])
    _, meta_mlp = train(make_config(tmp_path, "mlp"))
    assert "skip_share" not in meta_mlp["eval"][0]
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_generator.py tests/test_train_smoke.py -k "component_energies or linear_skip" -v`
Expected: FAIL, `AttributeError: 'LinearSkipGenerator' object has no attribute 'component_energies'` and `KeyError: 'skip_share'`.

- [ ] **Step 3: Implement**

In `LinearSkipGenerator` (after `forward`):

```python
    @torch.no_grad()
    def component_energies(self, z: Tensor) -> dict[str, float]:
        """Mean squared norm of the trunk and skip terms on `z`, the skip
        term's share of the pre-normalisation output energy, and the mean
        |cos| between the two terms.

        These are the quantities docs/datasets/nytimes.md measured on v1's
        checkpoints after the fact: the trunk's energy grew 38x while the
        skip's held near skip_dim, and the per-vector critic could not see
        the balance move. Logged per evaluation so a run shows the drift
        as it happens.
        """
        t = self.trunk_latent_dim
        trunk = self.trunk(z[:, :t]).float()
        skip = self.skip(z[:, t:]).float()
        trunk_energy = float((trunk * trunk).sum(dim=1).mean())
        skip_energy = float((skip * skip).sum(dim=1).mean())
        total = trunk_energy + skip_energy
        cos = torch.nn.functional.cosine_similarity(trunk, skip, dim=1, eps=1e-12)
        return {
            "trunk_energy": trunk_energy,
            "skip_energy": skip_energy,
            "skip_share": skip_energy / total if total > 0.0 else 0.0,
            "trunk_skip_abs_cos": float(cos.abs().mean()),
        }
```

In the trainer: import `LinearSkipGenerator` from `src.models.generator`; before the training loop (next to `real_gate`), draw a fixed probe from a dedicated generator so the run's own RNG stream is untouched:

```python
    # Fixed latents for the linear-skip balance readout, drawn from a
    # separate generator so the training stream is exactly what it was.
    energy_probe = None
    if isinstance(generator, LinearSkipGenerator):
        probe_rng = torch.Generator(device=device.type)
        probe_rng.manual_seed(seed)
        energy_probe = torch.randn(4096, latent_dim, generator=probe_rng, device=device)
```

Inside the eval block, right after `stats.update(collapse_stats(fake_holdout))`:

```python
                if energy_probe is not None:
                    stats.update(generator.component_energies(energy_probe))
```

(Inside the `ema_weights` context, so the energies describe the same weights the sample and the gate describe.)

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_generator.py tests/test_train_smoke.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/generator.py src/train/train_wgan_gp.py tests/test_generator.py tests/test_train_smoke.py
git commit -m "feat(train): log linear-skip trunk/skip energies and share on every evaluation"
```

---

### Task 7: the `v2` configs and job script

**Files:**
- Create: `configs/nytimes/v2.yaml`, `configs/nytimes/v2_seed42.yaml`, `scripts/nytimes_v2_seed42_job.sh`
- Test: `tests/test_nytimes_configs.py` (new)

**Interfaces:**
- Consumes: `critic_type: neighbourhood`, `critic_k`, `critic_distance_floor` (Task 4), `drop_zero_rows` (Task 1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_nytimes_configs.py
"""v2 is v1 plus exactly two stated changes. Pin them, so a drift in any
other key is caught before it burns 40 GPU-minutes."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent / "configs" / "nytimes"


def _load(name):
    with (ROOT / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


V2_DELTA = {
    "model.critic_type": "neighbourhood",
    "model.critic_k": 20,
    "model.critic_distance_floor": 0.01,
    "data.preprocess.drop_zero_rows": True,
}


def test_v2_is_v1_plus_the_two_stated_changes():
    v1 = _flatten(_load("v1.yaml"))
    v2 = _flatten(_load("v2.yaml"))
    for key, value in V2_DELTA.items():
        assert v2.pop(key) == value, key
    assert v2.pop("output_dir") == "runs/nytimes/v2"
    v1.pop("output_dir")
    assert v2 == v1


def test_v2_seed42_is_v2_with_an_absolute_real_path_and_its_own_output_dir():
    v2 = _flatten(_load("v2.yaml"))
    inst = _flatten(_load("v2_seed42.yaml"))
    assert inst.pop("output_dir") == "runs/nytimes/v2_seed42"
    assert inst.pop("data.real_path").startswith("/workspace/")
    v2.pop("output_dir")
    v2.pop("data.real_path")
    assert inst == v2


def test_v2_job_script_runs_the_v2_seed42_config():
    script = (ROOT.parent.parent / "scripts" / "nytimes_v2_seed42_job.sh").read_text()
    assert "configs/nytimes/v2_seed42.yaml" in script
    assert "runs/nytimes/v2_seed42" in script
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_nytimes_configs.py -v`
Expected: FAIL, `FileNotFoundError` on `v2.yaml`.

- [ ] **Step 3: Write the configs and the script**

`configs/nytimes/v2.yaml`: copy `configs/nytimes/v1.yaml`, replace the header comment, and make the two changes:

```yaml
# NYTIMES v2 -- v1 plus a neighbourhood-aware critic. v1 kept the output full
# rank but its trunk-to-skip balance drifted monotonically and the per-vector
# critic could not see it (Wasserstein < 0.06 while LID went 40 -> 5). The
# critic here scores each row together with its within-batch k-NN log-ratio
# profile, so local dimension is part of what it discriminates. Two changes
# from v1, both stated:
#
#   1. model.critic_type: neighbourhood (critic_k 20, critic_distance_floor
#      0.01, written out so the file says what ran)
#   2. data.preprocess.drop_zero_rows: true -- exact-zero rows are an empty
#      document, the gate already drops them, and to this critic they are a
#      shortcut. Exact duplicates are KEPT: they are part of the corpus.
#
# See docs/ai/specs/2026-09-14-neighbourhood-critic-design.md.
seed: 42
device: auto
output_dir: runs/nytimes/v2

data:
  real_path: data/nytimes_250k.npy
  format: npy
  metric: angular
  descriptor_dim: 256
  holdout_fraction: 0.05
  synthetic_if_missing: false
  synthetic_num_vectors: 100000
  preprocess:
    center: false
    whiten: false
    l2_normalize: true
    drop_zero_rows: true

model:
  # 256 trunk + 256 skip. The generator splits the latent itself.
  latent_dim: 512
  generator_hidden_dims: [512, 1024, 1024]
  critic_hidden_dims: [1024, 512, 256]
  negative_slope: 0.2
  generator_type: linear_skip
  skip_dim: 256
  skip_init: orthogonal
  skip_init_gain: 1.0
  critic_type: neighbourhood
  critic_k: 20
  critic_distance_floor: 0.01

training:
  batch_size: 512
  num_gen_steps: 30000
  n_critic: 3
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  betas: [0.0, 0.9]
  lambda_gp: 5.0
  distance_reg_alpha: 0.0
  distance_reg_max_points: 256
  num_workers: 0
  amp: false
  log_every: 250
  eval_every: 1000
  save_every: 2000
  select_on: gate
```

Before writing, diff the `seed`/`training` blocks against `v1.yaml` to be sure they are byte-identical; the test enforces it.

`configs/nytimes/v2_seed42.yaml`: the same file with this header and the two instrument differences (`output_dir: runs/nytimes/v2_seed42`, `real_path: /workspace/data-cache/nytimes_250k.npy`):

```yaml
# NYTimes v2, run 42. Identical to configs/nytimes/v2.yaml in every training
# hyperparameter. Two things differ, listed so a reader can confirm nothing
# else moved:
#
#   1. output_dir  runs/nytimes/v2 -> runs/nytimes/v2_seed42
#   2. real_path   data/nytimes_250k.npy -> an absolute path (see below)
#
# Why this file exists rather than an edit to v2.yaml: a rung is a historical
# record of a run that happened; this config is the measurement instrument,
# the same pattern as configs/nytimes/v1_seed42.yaml.
#
# Why an absolute real_path: the gpuq runner executes each job in a fresh
# detached git worktree cut from the pinned commit, and data/ is gitignored.
# The file it names is `python -m src.data.fetch nytimes` output at seed 42,
# sha256 ba8e8d512cc465e983661cbbabba64ead8251c3f313de86c3b1c3abe56c03428,
# the same bytes as a local data/nytimes_250k.npy. The trainer drops its
# 197 exact-zero rows at load (drop_zero_rows) and keeps its duplicates.
```

`scripts/nytimes_v2_seed42_job.sh`:

```bash
#!/usr/bin/env bash
# One gpuq job: train NYTimes v2 (seed 42), sample it, and measure it against
# the cleaned real corpus.
#
#   gpuq submit --project wgan-synthetic --commit <sha> --branch nytimes-eda \
#     --lane gpu --timeout-s 10800 -- bash scripts/nytimes_v2_seed42_job.sh
#
# runs/ is gitignored, so nothing here is declared as a --artifact; the run
# directory is copied to /workspace/nytimes-v2 at the end instead, the way
# the v1 job did. Box-specific by construction, like the config it runs.
set -euo pipefail
P=${WGAN_PYTHON:-/opt/venvs/wgan-synthetic/bin/python}
RUN=runs/nytimes/v2_seed42
REAL=/workspace/data-cache/nytimes_250k.npy
CLEAN=/workspace/data-cache/nytimes_250k_l2_clean.npy
KEEP=/workspace/nytimes-v2/v2_seed42
CANON="--ann-max-rows 20000 --ann-k 100 --ann-hub-k 10 --metric angular"

test -f "$REAL" && test -f "$CLEAN"
"$P" -m src.train.train_wgan_gp --config configs/nytimes/v2_seed42.yaml
"$P" -m src.sample.generate --checkpoint "$RUN/best_generator.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_best.npy"
"$P" -m src.sample.generate --checkpoint "$RUN/checkpoint_step_30000.pt" --config "$RUN/run_config.yaml" \
  --num-samples 50000 --seed 42 --output-path "$RUN/synthetic_50k_step30000.npy"
# shellcheck disable=SC2086
"$P" -m src.eval.eda_report --real-path "$CLEAN" \
  --synthetic-path "v2_best=$RUN/synthetic_50k_best.npy" \
  --synthetic-path "v2_step30000=$RUN/synthetic_50k_step30000.npy" \
  --output-dir "$RUN/eda_clean" $CANON --no-png --plotlyjs cdn
mkdir -p "$KEEP" && cp -r "$RUN"/. "$KEEP"/ && ls -la "$KEEP"
```

`chmod +x scripts/nytimes_v2_seed42_job.sh`.

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_nytimes_configs.py -v`
Expected: all PASS. If `test_v2_is_v1_plus_the_two_stated_changes` fails on a key other than the four, the copy drifted: fix the config, not the test.

- [ ] **Step 5: Commit**

```bash
git add configs/nytimes/v2.yaml configs/nytimes/v2_seed42.yaml scripts/nytimes_v2_seed42_job.sh tests/test_nytimes_configs.py
git commit -m "configs+scripts(nytimes): v2 rung, v1 plus the neighbourhood critic and drop_zero_rows"
```

---

### Task 8: documentation

**Files:**
- Modify: `PROJECT_DOCUMENTATION.md` (config-key table near line 372; after the `### generator_type` section at line 322)
- Modify: `docs/datasets/nytimes.md` (ladder table at line 199)

- [ ] **Step 1: `PROJECT_DOCUMENTATION.md`**

After the `### generator_type` section, add:

```markdown
### `critic_type`

The critic axis in the `model` config block, built by `build_critic` in
`src/models/critic.py`. Two values: `per_vector` (default; the `Critic` MLP
scoring one row at a time, every config before NYTimes v2) and
`neighbourhood` (`NeighbourhoodCritic`: the same MLP on each row
concatenated with its within-batch neighbourhood profile, `critic_k`
entries of sorted, floored k-NN distances expressed as `log(r_i / r_k)`
plus `log r_k`).

Why: a per-vector critic cannot see local dimension, so a low-rank sheet
with the right covariance envelope is, to it, the corpus. NYTimes v1
collapsed to LID 5 with the Wasserstein estimate under 0.06 throughout.
The neighbourhood critic makes the k-NN profile part of what is
discriminated. Under it, `gradient_penalty` bounds the gradient of the
batch's *summed* score per row, which is the intended constraint for a
batch-dependent critic (its docstring says so).

| Config key | Default | Meaning |
|---|---|---|
| `model.critic_type` | `per_vector` | Which critic class. |
| `model.critic_k` | `20` | Neighbour depth; the profile has `critic_k` entries. Must be below `training.batch_size`. |
| `model.critic_distance_floor` | `0.01` | Every neighbour distance the critic reads is clamped from below here, so an exact copy reads as a bounded "tight pair" rather than `-inf`. The gate's `near_duplicate_fraction` uses the same constant. |
| `data.preprocess.drop_zero_rows` | `false` | Drop exact-zero rows at load, before the train/holdout split; count in `run_metadata.json` under `data.dropped_zero_rows`. Duplicates are never dropped. |

Checkpoints do not record `critic_type` either; like the generator, the
critic is rebuilt from `run_config.yaml`, and the per-vector and
neighbourhood state dicts do not cross-load.
```

In the "Generator regularizers" paragraph, add after the sentence about `distance_reg` and `lid_reg` being logged: "`linear_skip` runs also log `trunk_energy`, `skip_energy`, `skip_share` and `trunk_skip_abs_cos` on every evaluation entry (`LinearSkipGenerator.component_energies`), and every `select_on: gate` run logs `gate_real_near_duplicate_fraction` / `gate_fake_near_duplicate_fraction`."

- [ ] **Step 2: `docs/datasets/nytimes.md` ladder row**

Append to the ladder table:

```markdown
| `v2` | + neighbourhood-aware critic (`critic_type: neighbourhood`, k 20, floor 0.01) and `drop_zero_rows: true`; duplicates kept | `configs/nytimes/v2.yaml`; box instrument `configs/nytimes/v2_seed42.yaml` | `runs/nytimes/v2_seed42` (box: `/workspace/nytimes-v2/v2_seed42`) | planned -- spec `docs/ai/specs/2026-09-14-neighbourhood-critic-design.md` |
```

- [ ] **Step 3: Run the docs checks and the full gate**

Run: `make check PYTHON=$PY`
Expected: ruff clean, all tests pass. `tests/test_docs_references.py` checks that symbols cited in `PROJECT_DOCUMENTATION.md` exist: `build_critic`, `NeighbourhoodCritic`, `Critic`, `LinearSkipGenerator.component_energies` all do by now.

- [ ] **Step 4: Commit**

```bash
git add PROJECT_DOCUMENTATION.md docs/datasets/nytimes.md
git commit -m "docs: critic_type, the neighbourhood critic's keys, and the nytimes v2 ladder row"
```

---

### Task 9: mutation checks

**Files:** none kept; each mutation is reverted.

The spec requires three mutations to be shown to fail their test. Record the three commands and their failing output in the PR body (or the final report if no PR).

- [ ] **Step 1: remove the diagonal mask**

In `_masked_squared_distances`, comment out `mask.fill_diagonal_(float("inf"))`. Run `$PY -m pytest tests/test_neighbourhood_critic.py -k self_is_excluded -v`. Expected: FAIL (`r.min()` is 0.01). Restore with `git checkout src/models/critic.py`.

- [ ] **Step 2: remove the clamp**

In `neighbourhood_distances`, replace `r2 = r2.clamp(min=float(floor) ** 2)` with `pass`. Run `$PY -m pytest tests/test_neighbourhood_critic.py -k "exact_copies or floor_gradient" -v`. Expected: both FAIL: the one-hot copy's distance is exactly 0.0 instead of 0.01, and `sqrt(0)` puts inf/NaN in the gradient. (The tests use a one-hot copy precisely so this mutation is caught deterministically; a random-unit copy rounds to ~1e-7 and would slip through.) Restore.

- [ ] **Step 3: drop the concatenation**

In `NeighbourhoodCritic.forward`, return `self.mlp(torch.cat([x, torch.zeros_like(phi)], dim=1))`. Run `$PY -m pytest tests/test_neighbourhood_critic.py -k depend_on_the_rest -v`. Expected: FAIL. Restore.

- [ ] **Step 4: confirm the tree is clean and green**

Run: `git status --short` (empty) and `make check PYTHON=$PY` (green). Paste both outputs into the report.

---

### Task 10: run `v2` on the box

**Files:** none in the repo until the results come back.

Follow the `gpu-jobs` skill for queue mechanics. Everything below is what the `v1` run did, with `v2` names.

- [ ] **Step 1: push the branch** (the runner cuts a worktree from the pinned commit; it must be on the remote)

```bash
timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=8 -T git@github.com   # probe first
git push origin nytimes-eda        # fall back to the HTTPS remote if SSH hangs >30 s
SHA=$(git rev-parse HEAD)
```

- [ ] **Step 2: probe the card, then submit**

```bash
ssh tig-gpu nvidia-smi            # must list the RTX 3060; if it errors, STOP (device: auto would train on CPU)
ssh tig-gpu "/opt/gpuq/venv/bin/gpuq submit --project wgan-synthetic --commit $SHA --branch nytimes-eda --lane gpu --timeout-s 10800 -- bash scripts/nytimes_v2_seed42_job.sh"
```

Record the job id. Expected wall time: about 40 minutes (v1 took 38; the profile adds under 5%).

- [ ] **Step 3: poll, do not block**

```bash
ssh tig-gpu "/opt/gpuq/venv/bin/gpuq show <job-id>"
```

Poll every 10 minutes with a `Monitor`/background loop, not a foreground sleep. On completion, the job's stdout carries the JSON eval lines; confirm `resumed_from_step: 0`, no `gate_error`, and `data.dropped_zero_rows` around 197.

- [ ] **Step 4: bring back only what is irreproducible** (the transfer rules in the global instructions)

```bash
ssh tig-gpu 'cd /workspace/nytimes-v2/v2_seed42 && ls -l run_metadata.json run_config.yaml eda_clean/summary.json && sha256sum run_metadata.json run_config.yaml eda_clean/summary.json'
mkdir -p docs/results/nytimes-v2-seed42
ssh tig-gpu 'cd /workspace/nytimes-v2/v2_seed42 && tar czf - run_metadata.json run_config.yaml eda_clean/summary.json' | tar xzf - -C docs/results/nytimes-v2-seed42/
(cd docs/results/nytimes-v2-seed42 && sha256sum run_metadata.json run_config.yaml eda_clean/summary.json)   # must match the remote hashes line for line
mv docs/results/nytimes-v2-seed42/eda_clean/summary.json docs/results/nytimes-v2-seed42/eda_clean_summary.json && rmdir docs/results/nytimes-v2-seed42/eda_clean
```

Checkpoints and samples stay on the box, as `v1`'s did.

- [ ] **Step 5: the claim table, then the page**

Before writing prose, build the claim table (`| claim | value | MEASURED / ESTIMATED | source | when |`) from `eda_clean_summary.json` (the four gate statistics at canonical conditions, selected and final checkpoints), `run_metadata.json` (`skip_share` trace, `selection_score` at the selected step and its neighbours, `gate_*_near_duplicate_fraction`, `dropped_zero_rows`, wall time from the job record). Then write `## v2, measured` in `docs/datasets/nytimes.md` in the shape of `## v1, measured`, update the ladder row from "planned" to the outcome, and state against the spec's bar: pass or miss per statistic, whether `skip_share` drifted monotonically, and the 2x-neighbour check on the selection score.

- [ ] **Step 6: commit the results locally**

```bash
git add docs/results/nytimes-v2-seed42 docs/datasets/nytimes.md
git commit -m "docs(nytimes): record the v2 seed-42 run"
```

Do not push the results commit until the numbers on the page have been read back against the JSON once more.

---

## Self-review

- **Spec coverage.** Corpus decision: Task 1. Floor and helpers: Task 2. Critic interface, factory, GP formulation: Tasks 3, 4. Numerics: Task 2 (`_masked_squared_distances`). Duplication diagnostic: Task 5. Success bar's energy trace: Task 6. The rung and job: Tasks 7, 10. Tests table: every row has a test in Tasks 1 to 6 (shape; permutation; dependence; copies; self-exclusion; double backward; factory default/dispatch/unknown; independence kept on `Critic` and mirrored; smoke + reload; `drop_zero_rows` count; `near_duplicate_fraction == 0.03`). Mutation checks: Task 9. Docs: Task 8.
- **Placeholders.** None: every step carries its code or command.
- **Type consistency.** `neighbourhood_distances(x, k, floor, bank=None, self_index=None)` is used with those names in Tasks 2 and 3 and in the `v2b` plan. `gate_statistics(..., distance_floor=)` in Task 5 and the trainer. `component_energies` keys match between Task 6's method, its test, and Task 8's docs.
