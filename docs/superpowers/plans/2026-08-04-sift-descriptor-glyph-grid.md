# SIFT Descriptor Glyph Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render individual SIFT descriptors as 4x4 grids of 8-ray orientation stars, so two rows of real descriptors can be eyeballed against one row per trained GAN variant.

**Architecture:** A pure NumPy geometry module (`src/eval/descriptor_glyph.py`) converts a 128-vector into ray endpoint coordinates and knows nothing about I/O or plotting. A thin CLI (`src/eval/plot_descriptor_grid.py`) loads real vectors, samples each variant's checkpoint via the existing `compare_variants` helpers, and assembles a Plotly figure. The split exists because the risky part — the index and angle convention — must be pinned by tests on a machine that has neither the dataset nor any checkpoints.

**Tech Stack:** Python 3, NumPy, PyTorch, Plotly (+ kaleido for PNG), PyYAML, pytest.

## Global Constraints

- Python interpreter for all commands: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python`. There is no `.venv` inside the worktree.
- Run every command from the worktree root: `/home/fibonadithya/TIG/wgan-synthetic/.claude/worktrees/sift-visualisation`.
- **Do not add dependencies.** matplotlib is deliberately not in `requirements.txt`; use Plotly only.
- This machine has no `data/` vectors and no `runs/` checkpoints — they live on tig-gpu. Every test must construct its own fixtures under `tmp_path`.
- Descriptor index convention, used verbatim everywhere: `index = (row * 4 + col) * 8 + orientation_bin`.
- Constants: `CELL_ROWS = CELL_COLS = 4`, `ORIENTATION_BINS = 8`, `DESCRIPTOR_DIM = 128`.
- Ray length rule: `length = min(abs(value) * scale, 1.0) * pitch / 2`. `scale` maps a value to a fraction of a half-cell; the clip at `1.0` means a ray never crosses into a neighbouring cell.
- Angle rule: `bin j -> j * 45 degrees`, counter-clockwise from `+x`. Row 0 is at the **top**, so cell row `r` sits at `y = (1.5 - r) * pitch` relative to the glyph centre.
- Negative bins are **never clamped**. They are routed to separate output arrays so they can be drawn in a warning colour. v0/v1/v1_5 use the unactivated `MLPGenerator` and will produce them; hiding that would defeat the figure.
- Gap marker inside coordinate arrays is `np.nan`, not `None`.

**Reference spec:** `docs/superpowers/specs/2026-08-04-sift-descriptor-glyph-grid-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/eval/descriptor_glyph.py` (create) | Pure geometry: 128-vector to `(4,4,8)`, shared scale, ray endpoint arrays. No I/O, no Plotly. |
| `tests/test_descriptor_glyph.py` (create) | Pins the index convention, angles, clipping, negative routing, edge cases. NumPy only. |
| `tests/conftest.py` (create) | `make_run_dir` and `write_tiny_gated_run` fixtures, shared by two test modules. |
| `tests/test_compare_variants.py` (modify) | Drop the two module-private helpers, consume the fixtures. Behaviour unchanged. |
| `src/eval/plot_descriptor_grid.py` (create) | CLI: load, guard, sample, assemble figure, write HTML/PNG. |
| `tests/test_plot_descriptor_grid.py` (create) | Integration against `tmp_path`, including the refuse-loudly paths. |

---

## Task 1: Descriptor reshape and index convention

**Files:**
- Create: `src/eval/descriptor_glyph.py`
- Test: `tests/test_descriptor_glyph.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CELL_ROWS`, `CELL_COLS`, `ORIENTATION_BINS`, `DESCRIPTOR_DIM` (ints); `descriptor_to_cells(vec: np.ndarray) -> np.ndarray` returning shape `(4, 4, 8)` float64.

- [ ] **Step 1: Write the failing test**

Create `tests/test_descriptor_glyph.py`:

```python
import numpy as np
import pytest

from src.eval import descriptor_glyph as dg


def test_constants_multiply_to_the_descriptor_dim():
    assert dg.CELL_ROWS * dg.CELL_COLS * dg.ORIENTATION_BINS == dg.DESCRIPTOR_DIM
    assert dg.DESCRIPTOR_DIM == 128


def test_index_convention_is_cell_major():
    """index = (row * 4 + col) * 8 + bin, pinned exactly."""
    cells = dg.descriptor_to_cells(np.arange(128, dtype=np.float64))
    assert cells.shape == (4, 4, 8)
    for row in range(4):
        for col in range(4):
            for bin_ in range(8):
                expected = (row * 4 + col) * 8 + bin_
                assert cells[row, col, bin_] == expected


@pytest.mark.parametrize("length", [0, 127, 129, 256])
def test_wrong_length_raises(length):
    with pytest.raises(ValueError, match="128"):
        dg.descriptor_to_cells(np.zeros(length))


def test_two_dimensional_input_raises():
    with pytest.raises(ValueError, match="128"):
        dg.descriptor_to_cells(np.zeros((2, 128)))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_descriptor_glyph.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.descriptor_glyph'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/eval/descriptor_glyph.py`:

```python
"""Geometry for drawing one SIFT descriptor as a 4x4 grid of 8-ray stars.

A SIFT descriptor is a 4x4 grid of spatial cells, each holding an 8-bin
histogram of gradient orientations (Lowe 2004). Real descriptors are sparse
and spiky because most image patches contain edges; a generator that matches
the marginals but produces even, bushy stars is failing in a way no aggregate
panel in `eda_report` will show.

The glyph is a diagram of the descriptor, not of the image patch it came
from -- SIFT is lossy and orientation-normalised, and inverting it is out of
scope.

Convention caveat: the row/col scan order and the zero direction of the
orientation bins differ between SIFT implementations (VLFeat, OpenCV, Lowe's
original binary), and we do not know which extractor produced SIFT1M. The
glyph may be rotated or transposed relative to true patch geometry. That does
not affect the comparison this module exists for: the same convention is
applied to real and generated vectors alike. It would only matter for a claim
about underlying image content, which we do not make.

Pure NumPy by design: no I/O and no Plotly import, so the index and angle
conventions can be pinned by tests on a machine holding neither the dataset
nor any checkpoint.
"""

from __future__ import annotations

import numpy as np

CELL_ROWS = 4
CELL_COLS = 4
ORIENTATION_BINS = 8
DESCRIPTOR_DIM = CELL_ROWS * CELL_COLS * ORIENTATION_BINS  # 128


def descriptor_to_cells(vec: np.ndarray) -> np.ndarray:
    """Reshape a descriptor to (4, 4, 8) indexed [row][col][orientation_bin].

    The flat layout is `index = (row * 4 + col) * 8 + bin`, which is exactly
    C-order for this shape, so the reshape is the convention.
    """
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape != (DESCRIPTOR_DIM,):
        raise ValueError(
            f"expected a flat descriptor of {DESCRIPTOR_DIM} values, "
            f"got array of shape {arr.shape}"
        )
    return arr.reshape(CELL_ROWS, CELL_COLS, ORIENTATION_BINS)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_descriptor_glyph.py -v
```

Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/eval/descriptor_glyph.py tests/test_descriptor_glyph.py
git commit -m "feat(eval): descriptor-to-cells reshape with the index convention pinned"
```

---

## Task 2: Shared ray scale

**Files:**
- Modify: `src/eval/descriptor_glyph.py`
- Test: `tests/test_descriptor_glyph.py`

**Interfaces:**
- Consumes: `DESCRIPTOR_DIM` from Task 1.
- Produces: `shared_scale(descriptors: np.ndarray, percentile: float = 99.0) -> float`. Returns the factor mapping a bin value to a fraction of a half-cell; `0.0` for empty or all-zero input.

Why a shared scale rather than per-glyph normalisation: normalising each glyph to fill its own box would rescale a near-flat generated descriptor to look as structured as a real one, which is the exact comparison the figure exists to make. The 99th percentile rather than the max keeps one outlier bin from shrinking everything else to invisibility.

**The percentile is taken over non-zero magnitudes only.** Real SIFT descriptors are sparse, and over the raw values the 99th percentile can land inside the run of zeros — returning a scale of `0.0` and drawing an empty figure. Filtering first also makes the reference mean "a typical *meaningful* bin", which is what the ray length should be relative to.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_descriptor_glyph.py`:

```python
def test_shared_scale_maps_the_percentile_value_to_one():
    """A bin at the reference percentile draws a full half-cell ray."""
    data = np.full((10, 128), 0.5)
    scale = dg.shared_scale(data)
    assert 0.5 * scale == pytest.approx(1.0)


def test_shared_scale_ignores_a_single_outlier():
    """One huge spike must not shrink every other ray into invisibility."""
    typical = np.full((100, 128), 0.5)
    with_spike = typical.copy()
    with_spike[0, 0] = 1000.0
    assert dg.shared_scale(with_spike) == pytest.approx(
        dg.shared_scale(typical), rel=0.05
    )


def test_shared_scale_uses_magnitude_of_negatives():
    positive = np.full((4, 128), 0.5)
    negative = np.full((4, 128), -0.5)
    assert dg.shared_scale(negative) == pytest.approx(dg.shared_scale(positive))


def test_shared_scale_ignores_zeros_on_sparse_input():
    """Real SIFT is sparse; a percentile over the raw values would land in
    the run of zeros and scale every ray to nothing."""
    sparse = np.zeros((4, 128))
    sparse[:, :3] = 0.5
    assert 0.5 * dg.shared_scale(sparse) == pytest.approx(1.0)


def test_shared_scale_of_all_zeros_is_zero():
    assert dg.shared_scale(np.zeros((4, 128))) == 0.0


def test_shared_scale_of_empty_input_is_zero():
    assert dg.shared_scale(np.zeros((0, 128))) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_descriptor_glyph.py -k shared_scale -v
```

Expected: FAIL — `AttributeError: module 'src.eval.descriptor_glyph' has no attribute 'shared_scale'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/eval/descriptor_glyph.py`:

```python
def shared_scale(descriptors: np.ndarray, percentile: float = 99.0) -> float:
    """Ray-length factor shared by every glyph in a figure.

    Computed from the given percentile of `|value|` across every descriptor
    that will be plotted -- real and generated together -- so rows stay
    honestly comparable. A value at the percentile maps to `1.0`, i.e. a ray
    filling the half-cell; larger values are clipped by `glyph_segments`.

    Zeros are excluded before the percentile is taken. Real SIFT descriptors
    are sparse, and over the raw values the percentile can fall inside the run
    of zeros, returning 0.0 and drawing an empty figure. Excluding them also
    makes the reference mean "a typical meaningful bin", which is what a ray
    length should be relative to.

    Returns 0.0 for empty or all-zero input, which draws no rays rather than
    dividing by zero.
    """
    arr = np.abs(np.asarray(descriptors, dtype=np.float64))
    nonzero = arr[arr > 0.0]
    if nonzero.size == 0:
        return 0.0
    reference = float(np.percentile(nonzero, percentile))
    if reference <= 0.0:
        return 0.0
    return 1.0 / reference
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_descriptor_glyph.py -v
```

Expected: PASS, 13 tests.

- [ ] **Step 5: Commit**

```bash
git add src/eval/descriptor_glyph.py tests/test_descriptor_glyph.py
git commit -m "feat(eval): shared percentile ray scale for the glyph grid"
```

---

## Task 3: Ray endpoint geometry

**Files:**
- Modify: `src/eval/descriptor_glyph.py`
- Test: `tests/test_descriptor_glyph.py`

**Interfaces:**
- Consumes: `descriptor_to_cells`, constants from Task 1; `shared_scale` from Task 2.
- Produces:
  ```python
  glyph_segments(
      cells: np.ndarray,             # (4, 4, 8)
      origin: tuple[float, float],   # centre of the whole 4x4 glyph
      pitch: float,                  # centre-to-centre cell spacing
      scale: float,                  # from shared_scale
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
  ```
  Returns `(pos_x, pos_y, neg_x, neg_y)`, float64 arrays of NaN-separated ray endpoints. Each ray contributes three entries: cell centre, tip, `np.nan`. Zero-length rays are omitted entirely.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_descriptor_glyph.py`:

```python
def _one_hot(row, col, bin_, value=1.0):
    vec = np.zeros(128)
    vec[(row * 4 + col) * 8 + bin_] = value
    return dg.descriptor_to_cells(vec)


def _segments(xs, ys):
    """Split NaN-separated coordinates into a list of (start, end) pairs."""
    pairs = []
    for i in range(0, len(xs), 3):
        assert np.isnan(xs[i + 2]) and np.isnan(ys[i + 2])
        pairs.append(((xs[i], ys[i]), (xs[i + 1], ys[i + 1])))
    return pairs


def test_one_hot_bin_draws_exactly_one_ray():
    cells = _one_hot(row=1, col=2, bin_=0)
    px, py, nx, ny = dg.glyph_segments(cells, origin=(0.0, 0.0), pitch=2.0, scale=1.0)
    assert len(nx) == 0 and len(ny) == 0
    pairs = _segments(px, py)
    assert len(pairs) == 1
    (start, end) = pairs[0]
    # cell (1, 2): x = (2 - 1.5) * 2 = 1.0, y = (1.5 - 1) * 2 = 1.0
    assert start == pytest.approx((1.0, 1.0))
    # bin 0 points along +x; length = min(1.0 * 1.0, 1.0) * 2.0 / 2 = 1.0
    assert end == pytest.approx((2.0, 1.0))


@pytest.mark.parametrize(
    "bin_, unit",
    [(0, (1.0, 0.0)), (1, (0.7071, 0.7071)), (2, (0.0, 1.0)), (6, (0.0, -1.0))],
)
def test_bin_angles_are_45_degrees_counter_clockwise_from_x(bin_, unit):
    cells = _one_hot(row=0, col=0, bin_=bin_)
    px, py, _, _ = dg.glyph_segments(cells, origin=(0.0, 0.0), pitch=2.0, scale=1.0)
    (start, end) = _segments(px, py)[0]
    assert (end[0] - start[0], end[1] - start[1]) == pytest.approx(unit, abs=1e-3)


def test_row_zero_is_at_the_top():
    top = dg.glyph_segments(_one_hot(0, 0, 0), (0.0, 0.0), 2.0, 1.0)[1][0]
    bottom = dg.glyph_segments(_one_hot(3, 0, 0), (0.0, 0.0), 2.0, 1.0)[1][0]
    assert top > bottom


def test_origin_is_the_glyph_centre():
    """Cells straddle the origin symmetrically: (0,0) and (3,3) mirror."""
    a = dg.glyph_segments(_one_hot(0, 0, 0), (0.0, 0.0), 2.0, 1.0)
    b = dg.glyph_segments(_one_hot(3, 3, 0), (0.0, 0.0), 2.0, 1.0)
    assert a[0][0] == pytest.approx(-b[0][0])
    assert a[1][0] == pytest.approx(-b[1][0])


def test_origin_offset_translates_every_coordinate():
    at_zero = dg.glyph_segments(_one_hot(1, 2, 0), (0.0, 0.0), 2.0, 1.0)
    shifted = dg.glyph_segments(_one_hot(1, 2, 0), (10.0, -5.0), 2.0, 1.0)
    assert shifted[0][0] == pytest.approx(at_zero[0][0] + 10.0)
    assert shifted[1][0] == pytest.approx(at_zero[1][0] - 5.0)


def test_negative_bin_goes_to_the_negative_arrays_at_full_magnitude():
    """Negatives are impossible for a gradient histogram, so they are shown,
    not clamped -- this is what distinguishes the MLP variants from v2."""
    cells = _one_hot(row=1, col=2, bin_=0, value=-1.0)
    px, py, nx, ny = dg.glyph_segments(cells, (0.0, 0.0), pitch=2.0, scale=1.0)
    assert len(px) == 0 and len(py) == 0
    (start, end) = _segments(nx, ny)[0]
    assert start == pytest.approx((1.0, 1.0))
    assert end == pytest.approx((2.0, 1.0))


def test_mixed_signs_are_split_across_both_outputs():
    vec = np.zeros(128)
    vec[(1 * 4 + 2) * 8 + 0] = 1.0
    vec[(1 * 4 + 2) * 8 + 4] = -1.0
    px, py, nx, ny = dg.glyph_segments(
        dg.descriptor_to_cells(vec), (0.0, 0.0), 2.0, 1.0
    )
    assert len(_segments(px, py)) == 1
    assert len(_segments(nx, ny)) == 1


def test_ray_length_is_linear_in_scale_below_the_clip():
    cells = _one_hot(0, 0, 0, value=0.25)
    short = dg.glyph_segments(cells, (0.0, 0.0), 2.0, scale=1.0)[0]
    long = dg.glyph_segments(cells, (0.0, 0.0), 2.0, scale=2.0)[0]
    assert (long[1] - long[0]) == pytest.approx(2.0 * (short[1] - short[0]))


def test_ray_is_clipped_to_half_a_cell():
    """A bin far above the percentile must not bleed into its neighbour."""
    cells = _one_hot(0, 0, 0, value=1000.0)
    px, _, _, _ = dg.glyph_segments(cells, (0.0, 0.0), pitch=2.0, scale=1.0)
    assert (px[1] - px[0]) == pytest.approx(1.0)  # pitch / 2


def test_zero_vector_draws_nothing():
    cells = dg.descriptor_to_cells(np.zeros(128))
    px, py, nx, ny = dg.glyph_segments(cells, (0.0, 0.0), 2.0, 1.0)
    assert len(px) == len(py) == len(nx) == len(ny) == 0


def test_zero_scale_draws_nothing():
    cells = dg.descriptor_to_cells(np.ones(128))
    px, _, nx, _ = dg.glyph_segments(cells, (0.0, 0.0), 2.0, scale=0.0)
    assert len(px) == 0 and len(nx) == 0


def test_wrong_cell_shape_raises():
    with pytest.raises(ValueError, match="4, 4, 8"):
        dg.glyph_segments(np.zeros((4, 4)), (0.0, 0.0), 2.0, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_descriptor_glyph.py -k glyph_segments -v
```

Expected: FAIL — `AttributeError: module 'src.eval.descriptor_glyph' has no attribute 'glyph_segments'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/eval/descriptor_glyph.py`:

```python
def glyph_segments(
    cells: np.ndarray,
    origin: tuple,
    pitch: float,
    scale: float,
):
    """Ray endpoints for one glyph, split by sign of the bin value.

    Returns `(pos_x, pos_y, neg_x, neg_y)` as NaN-separated coordinate arrays
    ready to hand to a Plotly line trace: each ray contributes cell centre,
    tip, then NaN. Zero-length rays are omitted, so an all-zero descriptor
    yields empty arrays.

    Negative bins land in `neg_*` at magnitude `|value|` rather than being
    clamped to zero. Real SIFT bins are gradient-magnitude histogram counts
    and cannot be negative, so a negative bin is an impossible value, not a
    small distributional error; the caller draws these in a warning colour.
    """
    cells = np.asarray(cells, dtype=np.float64)
    if cells.shape != (CELL_ROWS, CELL_COLS, ORIENTATION_BINS):
        raise ValueError(
            f"expected cells of shape (4, 4, 8), got {cells.shape}"
        )

    origin_x, origin_y = float(origin[0]), float(origin[1])
    max_length = pitch / 2.0
    angles = np.arange(ORIENTATION_BINS) * (2.0 * np.pi / ORIENTATION_BINS)
    unit_x = np.cos(angles)
    unit_y = np.sin(angles)

    pos_x, pos_y, neg_x, neg_y = [], [], [], []
    for row in range(CELL_ROWS):
        # Row 0 at the top, and cells straddle the origin, so the grid reads
        # like an image and `origin` is the centre of the whole glyph.
        centre_y = origin_y + ((CELL_ROWS - 1) / 2.0 - row) * pitch
        for col in range(CELL_COLS):
            centre_x = origin_x + (col - (CELL_COLS - 1) / 2.0) * pitch
            for bin_ in range(ORIENTATION_BINS):
                value = cells[row, col, bin_]
                length = min(abs(value) * scale, 1.0) * max_length
                if length <= 0.0:
                    continue
                xs, ys = (pos_x, pos_y) if value > 0 else (neg_x, neg_y)
                xs.extend((centre_x, centre_x + unit_x[bin_] * length, np.nan))
                ys.extend((centre_y, centre_y + unit_y[bin_] * length, np.nan))

    return (
        np.array(pos_x, dtype=np.float64),
        np.array(pos_y, dtype=np.float64),
        np.array(neg_x, dtype=np.float64),
        np.array(neg_y, dtype=np.float64),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_descriptor_glyph.py -v
```

Expected: PASS, 28 tests.

- [ ] **Step 5: Commit**

```bash
git add src/eval/descriptor_glyph.py tests/test_descriptor_glyph.py
git commit -m "feat(eval): glyph ray geometry with negatives shown, not clamped"
```

---

## Task 4: Promote checkpoint fixtures to conftest

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/test_compare_variants.py` (delete lines 28-35 and 86-124; update 7 call sites)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: two pytest fixtures.
  - `make_run_dir` — callable `(root: Path, name: str, with_checkpoint: bool = True, with_config: bool = True) -> Path`.
  - `write_tiny_gated_run` — callable `(tmp_path: Path, name: str = "tiny_gated", descriptor_dim: int = 8) -> tuple[compare_variants.Variant, int]`, writing a real `save_checkpoint` + `run_config.yaml` pair.

`descriptor_dim` is a **new parameter**, defaulting to 8 so existing tests are unaffected; Task 6 passes 128 because the glyph mapping is only defined at that width.

This is a pure refactor: `test_compare_variants.py` must pass unchanged afterwards.

- [ ] **Step 1: Create the conftest with both fixtures**

Create `tests/conftest.py`:

```python
"""Fixtures shared by the eval test modules.

`write_tiny_gated_run` builds a real checkpoint rather than a stub because
`load_generator` rebuilds the architecture from `run_config.yaml` and then
loads a state dict into it -- a fake file would only exercise the path
lookup, not the round trip.
"""

import pytest
import torch
import yaml

from src.eval import compare_variants as cv
from src.models.critic import Critic
from src.models.generator import build_generator
from src.train.train_wgan_gp import save_checkpoint


@pytest.fixture
def make_run_dir():
    """Create a run directory with placeholder artifacts for resolve tests."""

    def _make(root, name, with_checkpoint=True, with_config=True):
        d = root / name
        d.mkdir(parents=True)
        if with_config:
            (d / "run_config.yaml").write_text("model: {}\n")
        if with_checkpoint:
            (d / "best_generator.pt").write_bytes(b"")
        return d

    return _make


@pytest.fixture
def write_tiny_gated_run():
    """Write a real save_checkpoint + run_config pair for a tiny gated model."""

    def _write(tmp_path, name="tiny_gated", descriptor_dim=8):
        model_cfg = {
            "latent_dim": 4,
            "generator_hidden_dims": [6],
            "negative_slope": 0.2,
            "generator_type": "gated",
            "gate_temperature": 0.5,
            "logit_clamp": 4.0,
        }

        generator = build_generator(model_cfg, output_dim=descriptor_dim)
        critic = Critic(
            input_dim=descriptor_dim, hidden_dims=[6], negative_slope=0.2
        )
        optim_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
        optim_d = torch.optim.Adam(critic.parameters(), lr=1e-4)

        run_dir = tmp_path / "runs" / name
        save_checkpoint(
            generator,
            critic,
            optim_g,
            optim_d,
            out_dir=run_dir,
            step=1,
            best=True,
            generator_weights="live",
        )

        run_config = {
            "device": "cpu",
            "model": model_cfg,
            "data": {"descriptor_dim": descriptor_dim},
        }
        (run_dir / "run_config.yaml").write_text(yaml.safe_dump(run_config))

        variant = cv.Variant(name, "configs/sift_gan_v2.yaml", f"runs/{name}")
        return variant, descriptor_dim

    return _write
```

- [ ] **Step 2: Delete the helpers from `test_compare_variants.py`**

Delete the `_make_run_dir` function (currently lines 28-35) and the `_write_tiny_gated_run` function (currently lines 86-124) in their entirety.

Then remove imports that become unused. After deletion, `Critic`, `build_generator` and `save_checkpoint` are no longer referenced, so delete these three lines:

```python
from src.models.critic import Critic
from src.models.generator import build_generator
from src.train.train_wgan_gp import save_checkpoint
```

Leave `numpy`, `torch`, `yaml`, `argparse`, `sys`, `Path`, `cv` and `eda_report` imports in place — the remaining tests still use them.

- [ ] **Step 3: Update the seven call sites to take the fixtures**

Add the fixture name as a parameter to each test function, and drop the leading underscore at each call. The seven call sites, with their enclosing test:

| Test function | Change |
|---|---|
| `test_resolve_skips_variants_with_no_checkpoint(tmp_path)` | add `make_run_dir` param; 2 calls |
| `test_resolve_skips_variants_with_no_run_config(tmp_path)` | add `make_run_dir` param; 1 call |
| `test_resolve_finds_everything_when_present(tmp_path)` | add `make_run_dir` param; 2 calls |
| `test_generate_samples_round_trips_a_real_gated_checkpoint(tmp_path)` | add `write_tiny_gated_run` param; 1 call |
| `test_generate_samples_does_not_depend_on_preceding_variants(tmp_path)` | add `write_tiny_gated_run` param; 1 call |

For example, the first becomes:

```python
def test_resolve_skips_variants_with_no_checkpoint(tmp_path, make_run_dir):
    make_run_dir(tmp_path / "runs", "a")
    make_run_dir(tmp_path / "runs", "b", with_checkpoint=False)
```

Note `test_resolve_reports_a_missing_run_dir` does **not** call either helper — leave it alone.

- [ ] **Step 4: Run the full suite to verify nothing regressed**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/ -v
```

Expected: PASS, same test count as before the refactor. If any test errors with `fixture 'make_run_dir' not found`, a call site was updated without adding the parameter.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_compare_variants.py
git commit -m "refactor(tests): promote run-dir fixtures to conftest"
```

---

## Task 5: CLI rendering the two real rows

**Files:**
- Create: `src/eval/plot_descriptor_grid.py`
- Test: `tests/test_plot_descriptor_grid.py`

**Interfaces:**
- Consumes: `DESCRIPTOR_DIM`, `descriptor_to_cells`, `glyph_segments`, `shared_scale` (Tasks 1-3).
- Produces:
  - `CELL_PITCH = 1.0`, `GLYPH_PITCH = 5.0` (floats), `REAL_COLORS`, `VARIANT_COLORS` (tuples of str), `NEGATIVE_COLOR` (str).
  - `l2_normalize(x: np.ndarray) -> np.ndarray`
  - `pick_real_rows(real: np.ndarray, num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]`
  - `build_figure(rows: list) -> go.Figure` where `rows` is `list[tuple[str, np.ndarray, str]]` of `(label, vectors, colour)`
  - `write_report(fig, out_dir: Path, plotlyjs_mode: str, write_png: bool) -> Path`
  - `parse_args() -> argparse.Namespace`, `run(args) -> Path`, `main() -> None`

This task delivers a working CLI that renders `real-a` and `real-b`. Task 6 adds variant rows. Splitting here matters because this half is fully exercisable on a machine with no checkpoints.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plot_descriptor_grid.py`:

```python
import argparse

import numpy as np
import pytest

from src.eval import plot_descriptor_grid as pdg


def _write_real(tmp_path, n=64, dim=128, seed=0):
    """Sparse non-negative vectors, standing in for real SIFT descriptors."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, dim)).astype(np.float32)
    x[x < 0.8] = 0.0
    x[:, 0] = 1.0  # guarantee no all-zero row
    path = tmp_path / "real.npy"
    np.save(path, x)
    return path


def _args(tmp_path, **overrides):
    base = dict(
        real_path=str(_write_real(tmp_path)),
        real_format="auto",
        output_dir=str(tmp_path / "out"),
        root=str(tmp_path),
        num_samples=4,
        seed=42,
        no_png=True,
        plotlyjs="cdn",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_l2_normalize_gives_unit_rows():
    x = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    out = pdg.l2_normalize(x)
    assert np.linalg.norm(out, axis=1) == pytest.approx([1.0, 1.0])


def test_l2_normalize_leaves_a_zero_row_finite():
    out = pdg.l2_normalize(np.zeros((1, 4), dtype=np.float32))
    assert np.all(np.isfinite(out))


def test_pick_real_rows_returns_two_disjoint_rows():
    real = np.arange(40 * 128, dtype=np.float32).reshape(40, 128)
    row_a, row_b = pdg.pick_real_rows(real, num_samples=5, seed=1)
    assert row_a.shape == (5, 128) and row_b.shape == (5, 128)
    seen = {tuple(v) for v in row_a} | {tuple(v) for v in row_b}
    assert len(seen) == 10


def test_pick_real_rows_is_seed_reproducible():
    real = np.arange(40 * 128, dtype=np.float32).reshape(40, 128)
    first = pdg.pick_real_rows(real, 5, seed=7)[0]
    second = pdg.pick_real_rows(real, 5, seed=7)[0]
    assert np.array_equal(first, second)


def test_pick_real_rows_rejects_too_few_vectors():
    real = np.zeros((9, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="at least 10"):
        pdg.pick_real_rows(real, num_samples=5, seed=1)


def test_run_writes_html_with_the_two_real_rows(tmp_path):
    out = pdg.run(_args(tmp_path))
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "real-a" in text and "real-b" in text


def test_run_rejects_a_non_128_dimensional_dataset(tmp_path):
    path = _write_real(tmp_path, dim=64)
    with pytest.raises(ValueError, match="128"):
        pdg.run(_args(tmp_path, real_path=str(path)))


def test_run_rejects_a_missing_real_path(tmp_path):
    """Nothing to compare against, so this is a hard error, not a skip."""
    with pytest.raises((FileNotFoundError, ValueError)):
        pdg.run(_args(tmp_path, real_path=str(tmp_path / "absent.npy")))


def test_build_figure_puts_negative_rays_in_their_own_trace():
    vecs = np.zeros((1, 128), dtype=np.float32)
    vecs[0, 0] = 1.0
    vecs[0, 8] = -1.0
    fig = pdg.build_figure([("row", vecs, "#000000")])
    names = [t.name for t in fig.data]
    assert "negative" in names
    negative = next(t for t in fig.data if t.name == "negative")
    assert len(negative.x) == 3  # one ray: centre, tip, NaN


def test_build_figure_omits_the_negative_trace_when_all_bins_are_positive():
    vecs = np.abs(np.random.default_rng(0).random((2, 128))).astype(np.float32)
    fig = pdg.build_figure([("row", vecs, "#000000")])
    assert "negative" not in [t.name for t in fig.data]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_plot_descriptor_grid.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.plot_descriptor_grid'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/eval/plot_descriptor_grid.py`:

```python
"""Render real and generated SIFT descriptors as a grid of orientation glyphs.

Every other panel in `eda_report` is an aggregate over tens of thousands of
vectors. All of them can look healthy while the generator produces
descriptors that are structurally wrong, because a matched marginal says
nothing about whether the 128 numbers form a plausible gradient histogram.
This figure shows individual descriptors instead.

Two rows of real descriptors are drawn, not one. Without a sense of how much
two real descriptors differ from each other, a variant row below them is just
a vibe; the real-a/real-b gap is the baseline the rest is read against.

Samples are drawn at random under a fixed seed, never selected.

Example:
    python -m src.eval.plot_descriptor_grid \
        --real-path data/sift_base.npy \
        --output-dir runs/descriptor_grid
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go

from src.data.sift1m_dataset import load_descriptors
from src.eval import eda_report
from src.eval.descriptor_glyph import (
    DESCRIPTOR_DIM,
    descriptor_to_cells,
    glyph_segments,
    shared_scale,
)

CELL_PITCH = 1.0
# Roughly one glyph width (4 * CELL_PITCH) plus a gutter, so rows read as
# discrete descriptors rather than one continuous texture.
GLYPH_PITCH = 5.0

REAL_COLORS = ("#1f77b4", "#17becf")
VARIANT_COLORS = ("#ff7f0e", "#2ca02c", "#9467bd", "#8c564b")
NEGATIVE_COLOR = "#d62728"

REPORT_NAME = "descriptor_grid.html"


def l2_normalize(x: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    """Scale rows to unit norm, matching the training preprocessing."""
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def pick_real_rows(
    real: np.ndarray, num_samples: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw two disjoint random rows of real descriptors."""
    needed = 2 * num_samples
    if real.shape[0] < needed:
        raise ValueError(
            f"need at least {needed} real vectors for two rows of "
            f"{num_samples}, got {real.shape[0]}"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(real.shape[0], size=needed, replace=False)
    return real[idx[:num_samples]], real[idx[num_samples:]]


def build_figure(rows: List[Tuple[str, np.ndarray, str]]) -> go.Figure:
    """Assemble the glyph grid.

    One positive-ray trace per row so each gets its own colour and legend
    entry, plus a single shared trace for negative rays across all rows --
    those mark impossible values and should read as one alarming category,
    not as a per-row detail.
    """
    scale = shared_scale(np.concatenate([vecs for _, vecs, _ in rows], axis=0))

    fig = go.Figure()
    neg_x: List[np.ndarray] = []
    neg_y: List[np.ndarray] = []

    for row_index, (label, vecs, color) in enumerate(rows):
        pos_x: List[np.ndarray] = []
        pos_y: List[np.ndarray] = []
        for col_index in range(vecs.shape[0]):
            cells = descriptor_to_cells(vecs[col_index])
            origin = (col_index * GLYPH_PITCH, -row_index * GLYPH_PITCH)
            gx, gy, nx, ny = glyph_segments(cells, origin, CELL_PITCH, scale)
            pos_x.append(gx)
            pos_y.append(gy)
            neg_x.append(nx)
            neg_y.append(ny)
        fig.add_scatter(
            x=np.concatenate(pos_x) if pos_x else np.array([]),
            y=np.concatenate(pos_y) if pos_y else np.array([]),
            mode="lines",
            name=label,
            line=dict(color=color, width=1.4),
            hoverinfo="skip",
        )
        fig.add_annotation(
            x=-GLYPH_PITCH * 0.7,
            y=-row_index * GLYPH_PITCH,
            text=label,
            showarrow=False,
            xanchor="right",
            font=dict(size=13),
        )

    stacked_neg_x = np.concatenate(neg_x) if neg_x else np.array([])
    if stacked_neg_x.size:
        fig.add_scatter(
            x=stacked_neg_x,
            y=np.concatenate(neg_y),
            mode="lines",
            name="negative",
            line=dict(color=NEGATIVE_COLOR, width=1.8),
            hoverinfo="skip",
        )

    # Rule separating the two real rows from the variant rows below them.
    if len(rows) > 2:
        fig.add_hline(
            y=-1.5 * GLYPH_PITCH, line=dict(color="#999999", width=1, dash="dot")
        )

    axis = dict(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(
        title="SIFT descriptor glyphs: real vs generated",
        xaxis=axis,
        yaxis=dict(**axis, scaleanchor="x", scaleratio=1),
        height=180 * len(rows) + 120,
        plot_bgcolor="white",
        margin=dict(l=90, r=20, t=60, b=20),
    )
    return fig


def write_report(
    fig: go.Figure, out_dir: Path, plotlyjs_mode: str, write_png: bool
) -> Path:
    """Write the HTML report, and optionally a static PNG beside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    head = eda_report.plotlyjs_head(plotlyjs_mode, out_dir)
    body = fig.to_html(full_html=False, include_plotlyjs=False)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>SIFT descriptor glyph grid</title>"
        f"{head}</head><body>{body}</body></html>"
    )
    path = out_dir / REPORT_NAME
    path.write_text(html, encoding="utf-8")
    if write_png:
        eda_report.export_pngs([("descriptor grid", "", fig)], out_dir)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument(
        "--real-format", type=str, default="auto", choices=["auto", "npy", "fvecs"]
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repo root that variant config and run paths resolve against.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Descriptors per row. Each row is an independent random draw.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument(
        "--plotlyjs", type=str, default="inline", choices=["inline", "cdn", "directory"]
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    real = load_descriptors(Path(args.real_path), args.real_format)
    if real.shape[1] != DESCRIPTOR_DIM:
        raise ValueError(
            f"the glyph mapping is only defined for {DESCRIPTOR_DIM}-dimensional "
            f"descriptors; {args.real_path} holds {real.shape[1]}-dimensional ones"
        )
    real = l2_normalize(real)
    row_a, row_b = pick_real_rows(real, args.num_samples, args.seed)

    rows: List[Tuple[str, np.ndarray, str]] = [
        ("real-a", row_a, REAL_COLORS[0]),
        ("real-b", row_b, REAL_COLORS[1]),
    ]

    fig = build_figure(rows)
    return write_report(fig, Path(args.output_dir), args.plotlyjs, not args.no_png)


def main() -> None:
    print(run(parse_args()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_plot_descriptor_grid.py -v
```

Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add src/eval/plot_descriptor_grid.py tests/test_plot_descriptor_grid.py
git commit -m "feat(eval): glyph grid CLI rendering the two real reference rows"
```

---

## Task 6: Variant rows and the preprocessing guard

**Files:**
- Modify: `src/eval/plot_descriptor_grid.py`
- Test: `tests/test_plot_descriptor_grid.py`

**Interfaces:**
- Consumes: everything from Task 5; `write_tiny_gated_run` fixture from Task 4; `compare_variants.resolve_variants`, `.variant_seed`, `.VARIANTS`, `.CHECKPOINT_NAME`, `.RUN_CONFIG_NAME`.
- Produces:
  - `check_preprocess(config: dict, name: str) -> None` — raises `ValueError` if the run enabled centering or whitening.
  - `variant_rows(root: Path, num_samples: int, seed: int) -> list[tuple[str, np.ndarray, str]]`
  - `run` extended to append variant rows.

Why the guard: `center: false, whiten: false` in all four current configs is the only reason dimension *k* still maps to a specific cell and orientation bin. Centering is a constant offset and whitening a dense linear mix; under either, a glyph becomes a picture of mixed bins while still looking entirely plausible. Refusing beats drawing a silent lie.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_plot_descriptor_grid.py`:

```python
import yaml

from src.eval import compare_variants as cv


def test_check_preprocess_accepts_the_current_config_shape():
    config = {"data": {"preprocess": {"center": False, "whiten": False,
                                      "l2_normalize": True}}}
    pdg.check_preprocess(config, "v2")  # must not raise


def test_check_preprocess_accepts_a_missing_preprocess_block():
    """Absent keys mean the dataclass defaults, which are both False."""
    pdg.check_preprocess({"data": {}}, "v2")
    pdg.check_preprocess({}, "v2")


@pytest.mark.parametrize("flag", ["center", "whiten"])
def test_check_preprocess_refuses_centering_or_whitening(flag):
    config = {"data": {"preprocess": {flag: True}}}
    with pytest.raises(ValueError, match=flag):
        pdg.check_preprocess(config, "v2")


def test_variant_row_renders_from_a_real_checkpoint(
    tmp_path, write_tiny_gated_run, monkeypatch
):
    variant, _ = write_tiny_gated_run(tmp_path, name="v2", descriptor_dim=128)
    monkeypatch.setattr(cv, "VARIANTS", (variant,))
    out = pdg.run(_args(tmp_path))
    text = out.read_text(encoding="utf-8")
    assert "real-a" in text and "v2" in text


def test_variant_row_is_seed_reproducible(tmp_path, write_tiny_gated_run, monkeypatch):
    """GatedGenerator samples gate noise in eval() too, so this needs a seed."""
    variant, _ = write_tiny_gated_run(tmp_path, name="v2", descriptor_dim=128)
    monkeypatch.setattr(cv, "VARIANTS", (variant,))
    first = pdg.variant_rows(tmp_path, num_samples=4, seed=42)[0][1]
    second = pdg.variant_rows(tmp_path, num_samples=4, seed=42)[0][1]
    assert np.array_equal(first, second)


def test_missing_checkpoint_is_skipped_and_the_poster_still_renders(
    tmp_path, capsys, monkeypatch
):
    absent = cv.Variant("ghost", "configs/sift_gan_v2.yaml", "runs/ghost")
    monkeypatch.setattr(cv, "VARIANTS", (absent,))
    out = pdg.run(_args(tmp_path))
    assert out.exists()
    assert "ghost" in capsys.readouterr().out


def test_whitened_run_is_refused(tmp_path, write_tiny_gated_run, monkeypatch):
    variant, _ = write_tiny_gated_run(tmp_path, name="v2", descriptor_dim=128)
    run_config_path = tmp_path / "runs" / "v2" / "run_config.yaml"
    config = yaml.safe_load(run_config_path.read_text())
    config["data"]["preprocess"] = {"center": False, "whiten": True}
    run_config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(cv, "VARIANTS", (variant,))
    with pytest.raises(ValueError, match="whiten"):
        pdg.run(_args(tmp_path))


def test_variant_generating_the_wrong_width_is_refused(
    tmp_path, write_tiny_gated_run, monkeypatch
):
    variant, _ = write_tiny_gated_run(tmp_path, name="v2", descriptor_dim=8)
    monkeypatch.setattr(cv, "VARIANTS", (variant,))
    with pytest.raises(ValueError, match="128"):
        pdg.run(_args(tmp_path))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_plot_descriptor_grid.py -k "preprocess or variant or checkpoint or whitened" -v
```

Expected: FAIL — `AttributeError: module 'src.eval.plot_descriptor_grid' has no attribute 'check_preprocess'`.

- [ ] **Step 3: Write the implementation**

In `src/eval/plot_descriptor_grid.py`, extend the imports:

```python
import torch
import yaml

from src.eval import compare_variants as cv
from src.eval.evaluate_distribution import get_device, load_generator
from src.train.train_wgan_gp import sample_generator
```

Add these two functions above `parse_args`:

```python
def check_preprocess(config: dict, name: str) -> None:
    """Refuse a run whose preprocessing destroys the bin-to-dimension map.

    `center: false, whiten: false` across all four current variant configs is
    the only reason dimension k still means "cell i, orientation bin j".
    Centering shifts by a constant and whitening applies a dense linear mix;
    under either, the glyph becomes a picture of mixed bins that still looks
    entirely plausible. Better to refuse than to draw a silent lie.

    A missing preprocess block means the `PreprocessConfig` defaults, both of
    which are False.
    """
    preprocess = (config.get("data") or {}).get("preprocess") or {}
    enabled = [key for key in ("center", "whiten") if bool(preprocess.get(key, False))]
    if enabled:
        raise ValueError(
            f"variant {name} was trained with {' and '.join(enabled)} enabled, so "
            "its dimensions no longer map to (cell, orientation bin) and the "
            "glyph would be meaningless. Refusing to plot it."
        )


def variant_rows(
    root: Path, num_samples: int, seed: int
) -> List[Tuple[str, np.ndarray, str]]:
    """Sample every resolvable variant checkpoint into one row each.

    A variant whose artifacts are not on this machine is skipped with a
    message rather than aborting: checkpoints usually live on the training
    box, and a partial poster is still worth reading.
    """
    found, skipped = cv.resolve_variants(cv.VARIANTS, root)
    for variant, reason in skipped:
        print(f"skipping {variant.name}: {reason}")
    if not found:
        print("no variant checkpoints resolved; rendering the real rows only")

    rows: List[Tuple[str, np.ndarray, str]] = []
    for index, variant in enumerate(found):
        run_dir = root / variant.run_dir
        config = yaml.safe_load(
            (run_dir / cv.RUN_CONFIG_NAME).read_text(encoding="utf-8")
        )
        check_preprocess(config, variant.name)
        device = get_device(config["device"])
        generator = load_generator(config, run_dir / cv.CHECKPOINT_NAME, device)
        # GatedGenerator samples its gate in eval() mode too, so the seed is
        # what makes a row reproducible. Keying off the variant name keeps a
        # row identical whether or not other checkpoints are on this machine.
        torch.manual_seed(cv.variant_seed(seed, variant.name))
        samples = sample_generator(
            generator,
            num_samples=num_samples,
            latent_dim=int(config["model"]["latent_dim"]),
            batch_size=num_samples,
            device=device,
        )
        if samples.shape[1] != DESCRIPTOR_DIM:
            raise ValueError(
                f"variant {variant.name} generates {samples.shape[1]}-dimensional "
                f"vectors; the glyph mapping needs {DESCRIPTOR_DIM}"
            )
        rows.append(
            (variant.name, samples, VARIANT_COLORS[index % len(VARIANT_COLORS)])
        )
    return rows
```

Then in `run`, replace the line `fig = build_figure(rows)` with:

```python
    rows.extend(variant_rows(Path(args.root), args.num_samples, args.seed))

    fig = build_figure(rows)
```

- [ ] **Step 4: Run the full suite to verify it passes**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/ -v
```

Expected: PASS. `tests/test_plot_descriptor_grid.py` now has 19 tests; everything else is unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/eval/plot_descriptor_grid.py tests/test_plot_descriptor_grid.py
git commit -m "feat(eval): variant rows for the glyph grid, refusing whitened runs"
```

---

## Task 7: Document the tool

**Files:**
- Modify: `PROJECT_DOCUMENTATION.md`
- Modify: `data/README.md`

**Interfaces:**
- Consumes: the finished CLI from Task 6.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Read the surrounding sections to match their style**

```bash
grep -n "compare_variants\|plot_embedding_clusters" PROJECT_DOCUMENTATION.md
```

Read the section describing the other eval entry points, and follow its formatting exactly — heading level, whether commands are fenced, how flags are listed.

- [ ] **Step 2: Add a section for the glyph grid**

Add an entry alongside the other eval tools covering:

- What it produces: `descriptor_grid.html` (plus `png/` unless `--no-png`), two real reference rows above one row per resolvable variant.
- The invocation:

  ```bash
  python -m src.eval.plot_descriptor_grid \
      --real-path data/sift_base.npy \
      --output-dir runs/descriptor_grid
  ```

- How to read it: real SIFT is sparse and spiky, most cells dominated by one or two directions. Even, bushy stars mean the generator matched the marginals without the structure. **Red rays are negative bins — impossible for a gradient histogram**, and expected from v0/v1/v1_5, which use the unactivated MLP generator.
- The real-a/real-b pair is the baseline for how much natural variation to expect.
- It refuses to run against a variant trained with centering or whitening, because those break the dimension-to-bin mapping the glyph depends on.

- [ ] **Step 3: Note the constraint in the data contract**

In `data/README.md`, under the preprocessing contract section, add a sentence recording that `src/eval/plot_descriptor_grid.py` depends on `center: false` and `whiten: false`, since it interprets dimension `(row * 4 + col) * 8 + bin` as a specific spatial cell and orientation bin.

- [ ] **Step 4: Verify the docs are accurate**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m src.eval.plot_descriptor_grid --help
```

Confirm every flag named in the docs appears in the help output with the same default.

- [ ] **Step 5: Commit**

```bash
git add PROJECT_DOCUMENTATION.md data/README.md
git commit -m "docs: describe the descriptor glyph grid and its preprocessing constraint"
```

---

## Final Verification

- [ ] **Full suite passes**

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/ -v
```

- [ ] **The CLI runs end to end on synthetic data** (no dataset on this machine)

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python - <<'PY'
import numpy as np
rng = np.random.default_rng(0)
x = rng.random((64, 128)).astype(np.float32)
x[x < 0.8] = 0.0
np.save("/tmp/fake_sift.npy", x)
PY

/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m src.eval.plot_descriptor_grid \
    --real-path /tmp/fake_sift.npy \
    --output-dir /tmp/glyph_out \
    --no-png
```

Expected: prints the path to `descriptor_grid.html`, plus a skip message per variant (no `runs/` on this machine). Open the HTML and confirm two rows of 8 glyphs render, each a 4x4 grid of stars.

- [ ] **Real run on tig-gpu** — the figure this was built for, against the actual dataset and checkpoints. Confirm v0/v1/v1_5 rows show red negative rays and v2 does not.

---

## Notes for the implementer

- **Do not clamp negatives.** Several steps would be shorter if `glyph_segments` took `abs()` and forgot the sign. That single change would hide the clearest defect the figure exists to show.
- **Do not add a numeric score.** Ranking the variants is out of scope; that is what the existing `eda_report` panels do.
- **Do not add a panel to `eda_report.py`.** At 1017 lines it is already triple the size of any other module, and its panel contract is aggregates over ~50k vectors, not eight individual ones.
- The `origin` argument to `glyph_segments` is the centre of the whole 4x4 glyph, not its top-left corner. Tests pin this; do not "fix" it to a corner.
