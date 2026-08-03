# DEEP Image WGAN Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a three-rung WGAN-GP variant ladder that synthesizes `deep-image-96-angular`-like 96-D descriptors, judged by ANN-difficulty parity against real DEEP.

**Architecture:** A parallel `src/deep/` package supplies data acquisition, an inverse-preprocessing transform, a covariance-spectrum regularizer, an inverting sampler, and a comparison report. It *calls* the existing `train(config)` entry point at `src/train/train_wgan_gp.py:299` rather than reimplementing the training loop. Only two existing files change, both additively.

**Tech Stack:** Python 3, PyTorch, NumPy, h5py, PyYAML, pytest, plotly (via the existing eval suite).

## Global Constraints

- **Do not change SIFT behaviour.** No edits to `configs/sift_gan_*.yaml`, to `VARIANTS` in `src/eval/compare_variants.py`, or to any SIFT logic. The existing 118 tests are the regression guard and must stay green after every task.
- **Two permitted touches to shared files:** an optional `spectrum_reg_alpha` term in `src/train/train_wgan_gp.py`, and a DEEP section in `data/README.md`. Nothing else outside `src/deep/`, `configs/deep_gan_*.yaml`, and `tests/`.
- **`spectrum_reg_alpha` defaults to `0.0`** so every existing config behaves identically.
- **Descriptor dimension is 96.** Latent dim stays 128.
- **Fixed across all rungs:** `latent_dim: 128`, generator `[512, 1024, 1024]`, critic `[1024, 512, 256]`, `batch_size: 512`, `n_critic: 3`, `lambda_gp: 5.0`, `ema_decay: 0.999`, `distance_reg_alpha: 0.0`, `num_gen_steps: 30000`, `seed: 42`. Only each rung's stated delta varies.
- **Test imports go at the top of the test file**, not interleaved between test groups (`FOLLOWUPS.md` item 3).
- **Never touch `/workspace/wgan-synthetic` on `tig-gpu`.** That is a shared checkout. This work lives in `/workspace/deep-gan/`.
- **Run tests with the venv at the main repo root:** `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest`. Worktrees have no `.venv`.

---

### Task 1: DEEP data acquisition

**Files:**
- Create: `src/deep/__init__.py`, `src/deep/download.py`
- Modify: `requirements.txt`
- Test: `tests/test_deep_download.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DEEP_URL: str` — the ann-benchmarks HDF5 URL.
  - `fetch(url: str, dest: Path, *, chunk_bytes: int = 1 << 20, poll_seconds: float = 5.0, timeout_seconds: float = 3600.0) -> Path` — atomic, single-flight download; returns `dest`.
  - `subset(hdf5_path: Path, out_path: Path, *, num_rows: int, seed: int = 42) -> Path` — writes a `[num_rows, 96]` float32 `.npy`; returns `out_path`.

- [ ] **Step 1: Add h5py to requirements**

In `requirements.txt`, after the `scipy` line, add:

```
# HDF5 reader for the DEEP dataset, used only by src/deep/download.py.
h5py
```

Then install it: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/pip install h5py`

- [ ] **Step 2: Write the failing tests**

Create `tests/test_deep_download.py`:

```python
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.deep.download import DEEP_URL, fetch, subset


@pytest.fixture
def fake_hdf5(tmp_path: Path) -> Path:
    """A miniature stand-in for deep-image-96-angular.hdf5."""
    path = tmp_path / "fake.hdf5"
    rng = np.random.default_rng(0)
    train = rng.normal(size=(500, 96)).astype(np.float32)
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=train[:10])
    return path


def test_subset_writes_requested_shape_and_dtype(fake_hdf5: Path, tmp_path: Path):
    out = subset(fake_hdf5, tmp_path / "sub.npy", num_rows=100)
    arr = np.load(out)
    assert arr.shape == (100, 96)
    assert arr.dtype == np.float32


def test_subset_is_deterministic_under_the_same_seed(fake_hdf5: Path, tmp_path: Path):
    a = np.load(subset(fake_hdf5, tmp_path / "a.npy", num_rows=50, seed=7))
    b = np.load(subset(fake_hdf5, tmp_path / "b.npy", num_rows=50, seed=7))
    np.testing.assert_array_equal(a, b)


def test_subset_differs_under_a_different_seed(fake_hdf5: Path, tmp_path: Path):
    a = np.load(subset(fake_hdf5, tmp_path / "a.npy", num_rows=50, seed=7))
    b = np.load(subset(fake_hdf5, tmp_path / "b.npy", num_rows=50, seed=8))
    assert not np.array_equal(a, b)


def test_subset_takes_everything_when_num_rows_exceeds_the_file(
    fake_hdf5: Path, tmp_path: Path
):
    arr = np.load(subset(fake_hdf5, tmp_path / "all.npy", num_rows=10_000))
    assert arr.shape == (500, 96)


def test_fetch_leaves_no_partial_file_when_the_download_fails(
    tmp_path: Path, monkeypatch
):
    """A crashed fetch must not leave a truncated file a reader could load."""
    dest = tmp_path / "deep.hdf5"

    class Boom(Exception):
        pass

    def exploding_urlopen(*args, **kwargs):
        raise Boom("network down")

    monkeypatch.setattr("src.deep.download.urlopen", exploding_urlopen)
    with pytest.raises(Boom):
        fetch("http://example.invalid/x.hdf5", dest)
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_fetch_skips_the_download_when_the_destination_already_exists(
    tmp_path: Path, monkeypatch
):
    dest = tmp_path / "deep.hdf5"
    dest.write_bytes(b"already here")

    def exploding_urlopen(*args, **kwargs):
        raise AssertionError("fetch must not download over an existing file")

    monkeypatch.setattr("src.deep.download.urlopen", exploding_urlopen)
    assert fetch("http://example.invalid/x.hdf5", dest) == dest
    assert dest.read_bytes() == b"already here"


def test_fetch_waits_for_a_concurrent_downloader_instead_of_duplicating_it(
    tmp_path: Path, monkeypatch
):
    """Another agent sharing the cache may already be pulling the 4GB file.

    A held .part file means a fetch is in flight; the second caller waits for
    the result rather than starting a second 4GB download.
    """
    dest = tmp_path / "deep.hdf5"
    lock = dest.with_suffix(dest.suffix + ".part")
    lock.write_bytes(b"")  # simulate the in-flight download

    def exploding_urlopen(*args, **kwargs):
        raise AssertionError("must not download while another fetch holds the lock")

    monkeypatch.setattr("src.deep.download.urlopen", exploding_urlopen)

    def finish_the_other_download(_seconds):
        dest.write_bytes(b"complete")
        lock.unlink()

    monkeypatch.setattr("src.deep.download.time.sleep", finish_the_other_download)
    assert fetch("http://example.invalid/x.hdf5", dest) == dest
    assert dest.read_bytes() == b"complete"


def test_fetch_gives_up_on_a_stalled_concurrent_downloader(tmp_path: Path, monkeypatch):
    dest = tmp_path / "deep.hdf5"
    dest.with_suffix(dest.suffix + ".part").write_bytes(b"")
    monkeypatch.setattr("src.deep.download.time.sleep", lambda _s: None)
    with pytest.raises(TimeoutError, match="in-flight download"):
        fetch("http://example.invalid/x.hdf5", dest, timeout_seconds=0.0)


def test_deep_url_points_at_the_angular_variant():
    assert DEEP_URL.endswith("deep-image-96-angular.hdf5")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_download.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.deep'`

- [ ] **Step 4: Write the implementation**

Create `src/deep/__init__.py` as an empty file.

Create `src/deep/download.py`:

```python
"""Fetch the DEEP image descriptor set and cut reproducible subsets from it.

Writes .npy deliberately: the existing loader in src/data/sift1m_dataset.py
reads .npy and .fvecs, so emitting .npy means neither the loader nor the
trainer needs to learn about HDF5.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from urllib.request import urlopen

import h5py
import numpy as np

DEEP_URL = "http://ann-benchmarks.com/deep-image-96-angular.hdf5"
DESCRIPTOR_DIM = 96


def fetch(
    url: str,
    dest: Path,
    *,
    chunk_bytes: int = 1 << 20,
    poll_seconds: float = 5.0,
    timeout_seconds: float = 3600.0,
) -> Path:
    """Download `url` to `dest` atomically and single-flight. Returns `dest`.

    Two properties matter because this cache is shared with other agents on the
    box:

    Atomic -- the body is written to a sibling .part file and os.replace()d
    into position, so a concurrent reader sees either no file or a complete
    one, never a truncated download.

    Single-flight -- the .part file doubles as a lock, created O_EXCL. A second
    caller arriving mid-download waits for the first to finish rather than
    starting its own 4GB fetch. The timeout bounds the wait, so a crashed
    downloader that left a stale .part behind surfaces as an error instead of
    hanging forever; clear it by deleting the .part file.

    An existing destination is left alone -- the file is large and immutable.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        waited = 0.0
        while not dest.exists():
            if waited >= timeout_seconds:
                raise TimeoutError(
                    f"Timed out after {waited:.0f}s waiting for an in-flight "
                    f"download of {dest}. If no other process is fetching it, "
                    f"delete {tmp} and retry."
                )
            time.sleep(poll_seconds)
            waited += poll_seconds
        return dest

    try:
        with os.fdopen(fd, "wb") as handle, urlopen(url) as response:
            while True:
                chunk = response.read(chunk_bytes)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def subset(
    hdf5_path: Path, out_path: Path, *, num_rows: int, seed: int = 42
) -> Path:
    """Write a random `num_rows`-row sample of the train split to `out_path`.

    Rows are drawn without replacement and returned in sorted index order,
    which h5py requires for fancy indexing and which also makes the read
    sequential rather than random over a 4GB file.
    """
    hdf5_path = Path(hdf5_path)
    out_path = Path(out_path)
    with h5py.File(hdf5_path, "r") as f:
        train = f["train"]
        total = train.shape[0]
        take = min(num_rows, total)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total, size=take, replace=False))
        rows = train[idx, :]
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, rows)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", type=str, default=DEEP_URL)
    parser.add_argument(
        "--cache-path",
        type=str,
        default="/workspace/data-cache/deep-image-96-angular.hdf5",
        help="Where the shared read-only HDF5 lives. Downloaded once.",
    )
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[250_000, 1_000_000],
        help="Subset sizes to write, as data/deep96_<n>.npy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = fetch(args.url, Path(args.cache_path))
    print(f"hdf5: {cache}")
    for rows in args.rows:
        label = f"{rows // 1000}k" if rows < 1_000_000 else f"{rows // 1_000_000}m"
        out = subset(
            cache,
            Path(args.out_dir) / f"deep96_{label}.npy",
            num_rows=rows,
            seed=args.seed,
        )
        print(f"subset: {out} {np.load(out, mmap_mode='r').shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_download.py -v`
Expected: PASS, 9 tests

- [ ] **Step 6: Run the full suite to confirm no regression**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: 127 passed

- [ ] **Step 7: Commit**

```bash
git add requirements.txt src/deep/__init__.py src/deep/download.py tests/test_deep_download.py
git commit -m "feat(deep): fetch and subset the DEEP image descriptor set"
```

---

### Task 2: Inverse preprocessing

**Files:**
- Create: `src/deep/dataset.py`
- Test: `tests/test_deep_dataset.py`

**Interfaces:**
- Consumes: `PreprocessConfig`, `PreprocessState`, `apply_preprocess`, `_fit_preprocess_state` from `src.data.sift1m_dataset`.
- Produces: `invert_preprocess(x: np.ndarray, state: PreprocessState) -> np.ndarray` — inverts centering and whitening. Used by Task 4 (`src/deep/sample.py`).

**Why this exists:** `deep_v2` trains in whitened space. Without an inverse, its samples would be written in whitened coordinates and silently compared against real DEEP in original coordinates.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_deep_dataset.py`:

```python
import numpy as np
import pytest

from src.data.sift1m_dataset import (
    PreprocessConfig,
    _fit_preprocess_state,
    apply_preprocess,
)
from src.deep.dataset import invert_preprocess


def _sample(n: int = 400, d: int = 96, seed: int = 0) -> np.ndarray:
    """Anisotropic data, so whitening is a non-trivial transform."""
    rng = np.random.default_rng(seed)
    scale = np.linspace(1.0, 0.05, d).astype(np.float32)
    return (rng.normal(size=(n, d)) * scale).astype(np.float32)


@pytest.mark.parametrize(
    "center,whiten",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_invert_round_trips_apply_without_l2(center: bool, whiten: bool):
    x = _sample()
    cfg = PreprocessConfig(center=center, whiten=whiten, l2_normalize=False)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=x.shape[1], cfg=cfg)
    round_tripped = invert_preprocess(apply_preprocess(x, state), state)
    np.testing.assert_allclose(round_tripped, x, rtol=1e-3, atol=1e-3)


def test_invert_returns_float32():
    x = _sample()
    cfg = PreprocessConfig(center=True, whiten=True, l2_normalize=False)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=x.shape[1], cfg=cfg)
    assert invert_preprocess(apply_preprocess(x, state), state).dtype == np.float32


def test_invert_moves_whitened_data_off_the_identity_covariance():
    """Guards against a no-op inverse silently passing the round-trip test."""
    x = _sample()
    cfg = PreprocessConfig(center=True, whiten=True, l2_normalize=False)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=x.shape[1], cfg=cfg)
    whitened = apply_preprocess(x, state)
    # Whitened data has near-unit variance in every direction; the original
    # does not, because _sample builds in a decaying scale.
    assert whitened.var(axis=0).ptp() < 0.5
    assert invert_preprocess(whitened, state).var(axis=0).ptp() > 0.5


def test_invert_is_a_no_op_when_no_transform_was_fitted():
    x = _sample()
    cfg = PreprocessConfig(center=False, whiten=False, l2_normalize=False)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=x.shape[1], cfg=cfg)
    np.testing.assert_array_equal(invert_preprocess(x, state), x)


def test_l2_normalization_is_documented_as_not_invertible():
    """apply_preprocess discards vector norms; invert cannot restore them.

    The inverse undoes centering and whitening only. This test pins that
    contract so a future change does not quietly claim a full inverse.
    """
    x = _sample()
    cfg = PreprocessConfig(center=False, whiten=False, l2_normalize=True)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=x.shape[1], cfg=cfg)
    normalized = apply_preprocess(x, state)
    recovered = invert_preprocess(normalized, state)
    np.testing.assert_allclose(recovered, normalized, rtol=1e-6, atol=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.deep.dataset'`

- [ ] **Step 3: Write the implementation**

Create `src/deep/dataset.py`:

```python
"""The inverse of the preprocessing transform in src/data/sift1m_dataset.py.

DEEP's v2 rung trains in PCA-whitened space, so its samples have to be mapped
back to the original coordinates before anything compares them against real
DEEP. The forward transform is applied in the order center -> whiten ->
l2_normalize; the inverse undoes whiten then center.

L2 normalization is deliberately NOT inverted: it discards each vector's norm,
so the information needed to undo it is gone. This is not a limitation in
practice -- DEEP vectors are unit-norm to begin with, and the comparison is
angular.
"""
from __future__ import annotations

import numpy as np

from src.data.sift1m_dataset import PreprocessState


def invert_preprocess(x: np.ndarray, state: PreprocessState) -> np.ndarray:
    """Undo centering and whitening, in reverse order of application.

    The whitening matrix is symmetric (u @ diag(1/sqrt(s)) @ u.T over a
    symmetric covariance), so its inverse is likewise symmetric and is
    obtained with a plain matrix inverse. pinv rather than inv, because the
    eps-regularized covariance can still be near-singular on the tail
    dimensions of a PCA-derived set like DEEP.
    """
    out = np.asarray(x, dtype=np.float32)
    if state.whitening_matrix is not None:
        out = out @ np.linalg.pinv(state.whitening_matrix).astype(np.float32)
    if state.mean is not None:
        out = out + state.mean
    return np.ascontiguousarray(out, dtype=np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_dataset.py -v`
Expected: PASS, 8 tests (5 test functions, one parametrized 4 ways)

- [ ] **Step 5: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: 135 passed

- [ ] **Step 6: Commit**

```bash
git add src/deep/dataset.py tests/test_deep_dataset.py
git commit -m "feat(deep): invert centering and whitening for the v2 sampler"
```

---

### Task 3: Covariance-spectrum regularizer

**Files:**
- Create: `src/deep/spectrum.py`
- Test: `tests/test_deep_spectrum.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `spectrum_distance(real: Tensor, fake: Tensor, *, eps: float = 1e-8) -> Tensor` — scalar, non-negative, differentiable w.r.t. `fake`. Consumed by Task 4's trainer hook.

**What it measures:** the L1 gap between the sorted eigenvalue spectra of the two batches' covariance matrices, each normalized by its own trace. Normalizing by trace makes it a comparison of *shape* — how variance is distributed across directions — rather than of overall scale, which the unit-norm constraint already fixes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_deep_spectrum.py`:

```python
import torch

from src.deep.spectrum import spectrum_distance


def _anisotropic(n: int, d: int, decay: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scale = torch.linspace(1.0, decay, d)
    return torch.randn(n, d, generator=g) * scale


def test_distance_is_near_zero_for_identical_batches():
    x = _anisotropic(256, 32, 0.1, seed=0)
    assert spectrum_distance(x, x).item() < 1e-5


def test_distance_is_positive_for_mismatched_spectra():
    real = _anisotropic(256, 32, 0.02, seed=0)
    fake = _anisotropic(256, 32, 1.0, seed=1)
    assert spectrum_distance(real, fake).item() > 0.01


def test_distance_grows_as_the_spectra_diverge():
    real = _anisotropic(256, 32, 0.05, seed=0)
    near = _anisotropic(256, 32, 0.10, seed=1)
    far = _anisotropic(256, 32, 1.00, seed=1)
    assert spectrum_distance(real, near) < spectrum_distance(real, far)


def test_distance_is_invariant_to_overall_scale():
    """Trace normalization means a rescaled batch has the same spectrum shape."""
    real = _anisotropic(256, 32, 0.1, seed=0)
    fake = _anisotropic(256, 32, 0.1, seed=1)
    base = spectrum_distance(real, fake)
    scaled = spectrum_distance(real, fake * 10.0)
    torch.testing.assert_close(base, scaled, rtol=1e-4, atol=1e-4)


def test_distance_is_differentiable_with_respect_to_fake():
    real = _anisotropic(128, 16, 0.05, seed=0)
    fake = _anisotropic(128, 16, 0.5, seed=1).requires_grad_(True)
    spectrum_distance(real, fake).backward()
    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()
    assert fake.grad.abs().sum() > 0


def test_distance_returns_a_scalar():
    x = _anisotropic(64, 8, 0.1, seed=0)
    assert spectrum_distance(x, x).shape == torch.Size([])


def test_distance_is_finite_for_a_rank_deficient_batch():
    """Fewer rows than dimensions makes the covariance singular; eigvalsh must
    still return finite values rather than NaN."""
    real = _anisotropic(8, 32, 0.1, seed=0)
    fake = _anisotropic(8, 32, 0.5, seed=1)
    assert torch.isfinite(spectrum_distance(real, fake)).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_spectrum.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.deep.spectrum'`

- [ ] **Step 3: Write the implementation**

Create `src/deep/spectrum.py`:

```python
"""Match the covariance eigenvalue spectrum of generated batches to real ones.

DEEP descriptors are PCA-compressed CNN embeddings, so their variance decays
sharply and unevenly across directions. That decay is the property most
specific to the dataset, and the WGAN critic does not reliably enforce it --
the same failure mode that motivated the pairwise-distance regularizer on the
SIFT track.

The penalty compares the *shape* of the two spectra: each is divided by its own
trace before comparison, so it measures how variance is distributed across
directions rather than how much there is in total. Overall scale is already
pinned by the unit-norm constraint.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _normalized_spectrum(x: Tensor, eps: float) -> Tensor:
    """Sorted eigenvalues of x's covariance, scaled to sum to one."""
    centered = x - x.mean(dim=0, keepdim=True)
    n = max(centered.shape[0] - 1, 1)
    cov = (centered.T @ centered) / n
    # eigvalsh, not eigvals: the covariance is symmetric, and the symmetric
    # solver is both cheaper and free of the complex-valued output that would
    # otherwise have to be discarded. Clamp because a singular covariance --
    # any batch with fewer rows than dimensions -- yields eigenvalues that are
    # zero up to float error, and float error can put them slightly negative.
    eigenvalues = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    return eigenvalues / (eigenvalues.sum() + eps)


def spectrum_distance(real: Tensor, fake: Tensor, *, eps: float = 1.0e-8) -> Tensor:
    """Mean absolute gap between the two trace-normalized spectra.

    Non-negative, zero when the spectra match, and differentiable with respect
    to `fake`. Returns a scalar.
    """
    real_spectrum = _normalized_spectrum(real.detach(), eps)
    fake_spectrum = _normalized_spectrum(fake, eps)
    return torch.abs(real_spectrum - fake_spectrum).mean()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_spectrum.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/deep/spectrum.py tests/test_deep_spectrum.py
git commit -m "feat(deep): covariance-spectrum regularizer"
```

---

### Task 4: Wire the regularizer into the trainer

**Files:**
- Modify: `src/train/train_wgan_gp.py` (imports near line 24; config reads near line 373; generator step at lines 432-449)
- Test: `tests/test_deep_train_hook.py`

**Interfaces:**
- Consumes: `spectrum_distance` from `src.deep.spectrum`.
- Produces: config key `training.spectrum_reg_alpha` (float, default `0.0`) and a `spectrum_reg` entry in the per-step log dict.

**This is one of the two permitted touches to a shared file.** It must be additive: with `spectrum_reg_alpha` absent or `0.0`, the loop must behave exactly as it does today.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_deep_train_hook.py`:

```python
import numpy as np
import pytest
import torch

from src.train.train_wgan_gp import train


def _config(tmp_path, alpha: float) -> dict:
    """A tiny but real training run: 96-D, a handful of steps, CPU."""
    return {
        "seed": 42,
        "device": "cpu",
        "output_dir": str(tmp_path / "run"),
        "data": {
            "real_path": None,
            "format": "npy",
            "descriptor_dim": 96,
            "holdout_fraction": 0.2,
            "synthetic_if_missing": True,
            "synthetic_num_vectors": 512,
            "preprocess": {"center": False, "whiten": False, "l2_normalize": True},
        },
        "model": {
            "latent_dim": 16,
            "generator_hidden_dims": [32],
            "critic_hidden_dims": [32],
            "negative_slope": 0.2,
            "generator_type": "mlp",
        },
        "training": {
            "batch_size": 32,
            "num_gen_steps": 3,
            "n_critic": 1,
            "lr_g": 1.0e-4,
            "lr_d": 1.0e-4,
            "betas": [0.0, 0.9],
            "lambda_gp": 5.0,
            "ema_decay": 0.0,
            "distance_reg_alpha": 0.0,
            "spectrum_reg_alpha": alpha,
            "num_workers": 0,
            "amp": False,
            "log_every": 1,
            "eval_every": 100,
            "save_every": 100,
        },
    }


def test_spectrum_reg_is_logged_as_zero_when_disabled(tmp_path):
    _, meta = train(_config(tmp_path, alpha=0.0))
    assert all(m["spectrum_reg"] == 0.0 for m in meta["metrics"])


def test_spectrum_reg_is_positive_when_enabled(tmp_path):
    _, meta = train(_config(tmp_path, alpha=0.1))
    assert any(m["spectrum_reg"] > 0.0 for m in meta["metrics"])


def test_missing_spectrum_reg_alpha_defaults_to_disabled(tmp_path):
    """Every existing SIFT config omits this key and must keep working."""
    config = _config(tmp_path, alpha=0.0)
    del config["training"]["spectrum_reg_alpha"]
    _, meta = train(config)
    assert all(m["spectrum_reg"] == 0.0 for m in meta["metrics"])


def test_enabling_the_regularizer_changes_the_generator(tmp_path):
    """Proves the term reaches the generator's gradients, not just the log."""
    off_path, _ = train(_config(tmp_path / "off", alpha=0.0))
    on_path, _ = train(_config(tmp_path / "on", alpha=5.0))
    off = torch.load(off_path, map_location="cpu")["generator_state_dict"]
    on = torch.load(on_path, map_location="cpu")["generator_state_dict"]
    assert any(not torch.equal(off[k], on[k]) for k in off)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_train_hook.py -v`
Expected: FAIL — `KeyError: 'spectrum_reg'` on the metrics dict

- [ ] **Step 3: Add the import**

In `src/train/train_wgan_gp.py`, alongside the existing model imports near line 24, add:

```python
from src.deep.spectrum import spectrum_distance
```

- [ ] **Step 4: Read the config value**

Immediately after the `distance_reg_max_points` line (near line 374), add:

```python
    spectrum_reg_alpha = float(train_cfg.get("spectrum_reg_alpha", 0.0))
```

- [ ] **Step 5: Add the loss term**

In the generator step, replace this block (lines 436-448):

```python
            adv_loss = -critic(fake).mean()
            if distance_reg_alpha > 0.0:
                real_for_reg = real_batch.to(device)
                dist_real = batch_pairwise_distance_mean(
                    real_for_reg, max_points=distance_reg_max_points
                )
                dist_fake = batch_pairwise_distance_mean(
                    fake, max_points=distance_reg_max_points
                )
                distance_reg = torch.abs(dist_real - dist_fake)
                g_loss = adv_loss + distance_reg_alpha * distance_reg
            else:
                distance_reg = torch.zeros((), device=device, dtype=fake.dtype)
                g_loss = adv_loss
```

with:

```python
            adv_loss = -critic(fake).mean()
            g_loss = adv_loss
            if distance_reg_alpha > 0.0:
                real_for_reg = real_batch.to(device)
                dist_real = batch_pairwise_distance_mean(
                    real_for_reg, max_points=distance_reg_max_points
                )
                dist_fake = batch_pairwise_distance_mean(
                    fake, max_points=distance_reg_max_points
                )
                distance_reg = torch.abs(dist_real - dist_fake)
                g_loss = g_loss + distance_reg_alpha * distance_reg
            else:
                distance_reg = torch.zeros((), device=device, dtype=fake.dtype)
            if spectrum_reg_alpha > 0.0:
                spectrum_reg = spectrum_distance(real_batch.to(device), fake)
                g_loss = g_loss + spectrum_reg_alpha * spectrum_reg
            else:
                spectrum_reg = torch.zeros((), device=device, dtype=fake.dtype)
```

- [ ] **Step 6: Log the value**

In the per-step log dict at `src/train/train_wgan_gp.py:459-469`, add one line directly after `"distance_reg": float(distance_reg.item()),` (line 467):

```python
                "spectrum_reg": float(spectrum_reg.item()),
```

The dict is appended to `run_meta["metrics"]` on line 469, which is what the tests read.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_train_hook.py -v`
Expected: PASS, 4 tests

- [ ] **Step 8: Run the full suite — this is the regression gate**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: 146 passed. Any SIFT test failure here means the touch was not additive; fix before committing.

- [ ] **Step 9: Commit**

```bash
git add src/train/train_wgan_gp.py tests/test_deep_train_hook.py
git commit -m "feat(train): optional spectrum regularizer, off by default"
```

---

### Task 5: Inverting sampler

**Files:**
- Create: `src/deep/sample.py`
- Test: `tests/test_deep_sample.py`

**Interfaces:**
- Consumes: `invert_preprocess` (Task 2); `PreprocessState.from_serializable` from `src.data.sift1m_dataset`; `build_generator` from `src.models.generator`; `sample_generator`, `get_device` from `src.train.train_wgan_gp`.
- Produces:
  - `load_preprocess_state(run_dir: Path) -> PreprocessState` — reads `run_metadata.json`.
  - `sample_variant(run_dir: Path, num_samples: int, *, batch_size: int = 4096, seed: int = 42, checkpoint_name: str = "best_generator.pt") -> np.ndarray` — returns `[num_samples, 96]` float32 in **original** coordinates. Consumed by Task 7.

**Why it can't reuse `src/sample/generate.py`:** that script L2-normalizes unconditionally (`:64`) and never inverts the transform, so under a whitening config it would emit vectors in whitened space.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_deep_sample.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.sift1m_dataset import (
    PreprocessConfig,
    _fit_preprocess_state,
)
from src.deep.sample import load_preprocess_state, sample_variant


def _write_run(tmp_path: Path, *, whiten: bool) -> Path:
    """Build a run directory shaped exactly like one train() writes."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    dim, latent = 96, 16

    config = {
        "device": "cpu",
        "model": {
            "latent_dim": latent,
            "generator_hidden_dims": [32],
            "critic_hidden_dims": [32],
            "negative_slope": 0.2,
            "generator_type": "mlp",
        },
        "data": {"descriptor_dim": dim},
    }
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    from src.models.generator import build_generator

    generator = build_generator(config["model"], output_dim=dim)
    torch.save({"generator_state_dict": generator.state_dict()},
               run_dir / "best_generator.pt")

    rng = np.random.default_rng(0)
    scale = np.linspace(1.0, 0.05, dim).astype(np.float32)
    x = (rng.normal(size=(400, dim)) * scale).astype(np.float32)
    cfg = PreprocessConfig(center=True, whiten=whiten, l2_normalize=True)
    state = _fit_preprocess_state(x_train=x, descriptor_dim=dim, cfg=cfg)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"preprocess_state": state.to_serializable()}), encoding="utf-8"
    )
    return run_dir


def test_load_preprocess_state_round_trips_from_run_metadata(tmp_path: Path):
    state = load_preprocess_state(_write_run(tmp_path, whiten=True))
    assert state.descriptor_dim == 96
    assert state.mean is not None
    assert state.whitening_matrix is not None


def test_sample_variant_returns_the_requested_shape_and_dtype(tmp_path: Path):
    x = sample_variant(_write_run(tmp_path, whiten=False), num_samples=64, batch_size=32)
    assert x.shape == (64, 96)
    assert x.dtype == np.float32


def test_sample_variant_is_deterministic_under_the_same_seed(tmp_path: Path):
    run_dir = _write_run(tmp_path, whiten=False)
    a = sample_variant(run_dir, num_samples=32, seed=7)
    b = sample_variant(run_dir, num_samples=32, seed=7)
    np.testing.assert_array_equal(a, b)


def test_whitened_run_output_is_not_left_in_whitened_space(tmp_path: Path):
    """The whole reason this module exists.

    A whitened run's raw generator output has near-flat per-dimension variance.
    After inversion the anisotropy of the fitted state must be restored, so the
    spread of per-dimension variance is visibly larger.
    """
    inverted = sample_variant(_write_run(tmp_path, whiten=True), num_samples=512)
    unwhitened = sample_variant(_write_run(tmp_path, whiten=False), num_samples=512)
    assert inverted.var(axis=0).ptp() > unwhitened.var(axis=0).ptp()


def test_sample_variant_errors_clearly_when_run_metadata_is_missing(tmp_path: Path):
    run_dir = _write_run(tmp_path, whiten=False)
    (run_dir / "run_metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="run_metadata.json"):
        sample_variant(run_dir, num_samples=8)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_sample.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.deep.sample'`

- [ ] **Step 3: Write the implementation**

Create `src/deep/sample.py`:

```python
"""Sample a deep variant's checkpoint back into original DEEP coordinates.

src/sample/generate.py cannot be reused here: it L2-normalizes unconditionally
and never inverts the preprocessing transform, so a whitened run would silently
emit vectors in whitened space and be compared against real DEEP in original
space.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.sift1m_dataset import PreprocessState
from src.deep.dataset import invert_preprocess
from src.models.generator import build_generator
from src.train.train_wgan_gp import get_device, sample_generator

CHECKPOINT_NAME = "best_generator.pt"
RUN_CONFIG_NAME = "run_config.yaml"
RUN_METADATA_NAME = "run_metadata.json"


def load_preprocess_state(run_dir: Path) -> PreprocessState:
    """Read the transform train() fitted, so sampling can undo it."""
    path = Path(run_dir) / RUN_METADATA_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No {RUN_METADATA_NAME} in {run_dir}. It records the preprocessing "
            "state, without which samples cannot be returned to original "
            "coordinates. Copy it from the training box alongside the checkpoint."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PreprocessState.from_serializable(payload["preprocess_state"])


def sample_variant(
    run_dir: Path,
    num_samples: int,
    *,
    batch_size: int = 4096,
    seed: int = 42,
    checkpoint_name: str = CHECKPOINT_NAME,
) -> np.ndarray:
    """Draw `num_samples` vectors in original DEEP coordinates."""
    run_dir = Path(run_dir)
    state = load_preprocess_state(run_dir)
    config = yaml.safe_load((run_dir / RUN_CONFIG_NAME).read_text(encoding="utf-8"))

    device = get_device(config["device"])
    model_cfg = config["model"]
    descriptor_dim = int(config["data"]["descriptor_dim"])

    generator = build_generator(model_cfg, output_dim=descriptor_dim).to(device)
    checkpoint = torch.load(run_dir / checkpoint_name, map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    torch.manual_seed(seed)
    x = sample_generator(
        generator,
        num_samples=num_samples,
        latent_dim=int(model_cfg["latent_dim"]),
        batch_size=batch_size,
        device=device,
    )
    return invert_preprocess(x, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x = sample_variant(
        Path(args.run_dir),
        args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, x)
    print(f"Saved {x.shape[0]} vectors to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_sample.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/deep/sample.py tests/test_deep_sample.py
git commit -m "feat(deep): sampler that inverts preprocessing"
```

---

### Task 6: Variant configs

**Files:**
- Create: `configs/deep_gan_v0.yaml`, `configs/deep_gan_v1.yaml`, `configs/deep_gan_v2.yaml`
- Modify: `data/README.md`
- Test: `tests/test_deep_configs.py`

**Interfaces:**
- Consumes: `training.spectrum_reg_alpha` (Task 4).
- Produces: three config files. Task 7's variant table references them by path; Task 8 trains from them.

**The ladder invariant:** each rung differs from the previous by exactly one key. The test enforces this, because it is the property that makes the comparison report interpretable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_deep_configs.py`:

```python
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

CONFIG_DIR = Path("configs")
LADDER = ["deep_gan_v0", "deep_gan_v1", "deep_gan_v2"]


def _load(name: str) -> Dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


@pytest.mark.parametrize("name", LADDER)
def test_every_rung_targets_96_dimensions(name: str):
    assert _load(name)["data"]["descriptor_dim"] == 96


@pytest.mark.parametrize("name", LADDER)
def test_every_rung_shares_the_fixed_hyperparameters(name: str):
    config = _load(name)
    assert config["seed"] == 42
    assert config["model"]["latent_dim"] == 128
    assert config["model"]["generator_hidden_dims"] == [512, 1024, 1024]
    assert config["model"]["critic_hidden_dims"] == [1024, 512, 256]
    assert config["model"]["generator_type"] == "mlp"
    assert config["training"]["batch_size"] == 512
    assert config["training"]["n_critic"] == 3
    assert config["training"]["lambda_gp"] == 5.0
    assert config["training"]["ema_decay"] == 0.999
    assert config["training"]["distance_reg_alpha"] == 0.0
    assert config["training"]["num_gen_steps"] == 30000


@pytest.mark.parametrize(
    "lower,upper,expected_delta",
    [
        ("deep_gan_v0", "deep_gan_v1", "training.spectrum_reg_alpha"),
        ("deep_gan_v1", "deep_gan_v2", "data.preprocess.whiten"),
    ],
)
def test_each_rung_is_exactly_one_change_from_the_previous(
    lower: str, upper: str, expected_delta: str
):
    """The ladder is only interpretable if one thing varies at a time."""
    a, b = _flatten(_load(lower)), _flatten(_load(upper))
    differing = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert differing - {"output_dir"} == {expected_delta}


def test_v0_disables_the_spectrum_regularizer():
    assert _load("deep_gan_v0")["training"]["spectrum_reg_alpha"] == 0.0


def test_v1_enables_the_spectrum_regularizer():
    assert _load("deep_gan_v1")["training"]["spectrum_reg_alpha"] == 0.1


def test_only_v2_whitens():
    assert _load("deep_gan_v0")["data"]["preprocess"]["whiten"] is False
    assert _load("deep_gan_v1")["data"]["preprocess"]["whiten"] is False
    assert _load("deep_gan_v2")["data"]["preprocess"]["whiten"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_configs.py -v`
Expected: FAIL — `FileNotFoundError: configs/deep_gan_v0.yaml`

- [ ] **Step 3: Write `configs/deep_gan_v0.yaml`**

```yaml
# Deep variant v0 -- the baseline rung of the DEEP image ladder.
#
# Derived from configs/sift_gan_v1.yaml: the SIFT rung that has generator EMA
# but not the pairwise-distance regularizer. Those settings are carried over
# rather than re-ablated, because the SIFT track already answered them.
#
# No spherical generator is needed. normalize_l2 is already applied wherever
# the generator's output is consumed (train_wgan_gp.py:408, :435, :292), and
# it sits in the autograd graph, so a plain MLP is already trained against the
# unit sphere.
seed: 42
device: auto
output_dir: runs/deep_gan_v0

data:
  real_path: data/deep96_1m.npy
  format: npy
  descriptor_dim: 96
  holdout_fraction: 0.05
  synthetic_if_missing: false
  synthetic_num_vectors: 100000
  preprocess:
    center: false
    whiten: false
    l2_normalize: true

model:
  latent_dim: 128
  generator_hidden_dims: [512, 1024, 1024]
  critic_hidden_dims: [1024, 512, 256]
  negative_slope: 0.2
  generator_type: mlp

training:
  batch_size: 512
  num_gen_steps: 30000
  n_critic: 3
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  betas: [0.0, 0.9]
  lambda_gp: 5.0
  ema_decay: 0.999
  distance_reg_alpha: 0.0
  distance_reg_max_points: 256
  spectrum_reg_alpha: 0.0
  num_workers: 0
  amp: false
  log_every: 250
  eval_every: 1000
  save_every: 2000
```

- [ ] **Step 4: Write `configs/deep_gan_v1.yaml`**

Identical to v0 except `output_dir` and the single delta:

```yaml
# Deep variant v1 -- v0 plus the covariance-spectrum regularizer, which pushes
# the generator to reproduce DEEP's PCA variance decay. That decay is the
# property most specific to a PCA-compressed embedding set, and the critic does
# not reliably enforce it. Sole delta from v0: training.spectrum_reg_alpha.
seed: 42
device: auto
output_dir: runs/deep_gan_v1

data:
  real_path: data/deep96_1m.npy
  format: npy
  descriptor_dim: 96
  holdout_fraction: 0.05
  synthetic_if_missing: false
  synthetic_num_vectors: 100000
  preprocess:
    center: false
    whiten: false
    l2_normalize: true

model:
  latent_dim: 128
  generator_hidden_dims: [512, 1024, 1024]
  critic_hidden_dims: [1024, 512, 256]
  negative_slope: 0.2
  generator_type: mlp

training:
  batch_size: 512
  num_gen_steps: 30000
  n_critic: 3
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  betas: [0.0, 0.9]
  lambda_gp: 5.0
  ema_decay: 0.999
  distance_reg_alpha: 0.0
  distance_reg_max_points: 256
  spectrum_reg_alpha: 0.1
  num_workers: 0
  amp: false
  log_every: 250
  eval_every: 1000
  save_every: 2000
```

- [ ] **Step 5: Write `configs/deep_gan_v2.yaml`**

Identical to v1 except `output_dir` and the single delta:

```yaml
# Deep variant v2 -- v1 plus PCA whitening of the training space. Where v1
# pushes the variance decay in with a penalty, v2 makes it exact by
# construction: the generator learns an isotropic target and the anisotropy
# lives in a fixed linear map, inverted at sample time by src/deep/sample.py.
# Sole delta from v1: data.preprocess.whiten.
#
# This rung REQUIRES src/deep/sample.py. src/sample/generate.py does not invert
# the transform and would emit vectors in whitened space.
seed: 42
device: auto
output_dir: runs/deep_gan_v2

data:
  real_path: data/deep96_1m.npy
  format: npy
  descriptor_dim: 96
  holdout_fraction: 0.05
  synthetic_if_missing: false
  synthetic_num_vectors: 100000
  preprocess:
    center: false
    whiten: true
    l2_normalize: true

model:
  latent_dim: 128
  generator_hidden_dims: [512, 1024, 1024]
  critic_hidden_dims: [1024, 512, 256]
  negative_slope: 0.2
  generator_type: mlp

training:
  batch_size: 512
  num_gen_steps: 30000
  n_critic: 3
  lr_g: 1.0e-4
  lr_d: 1.0e-4
  betas: [0.0, 0.9]
  lambda_gp: 5.0
  ema_decay: 0.999
  distance_reg_alpha: 0.0
  distance_reg_max_points: 256
  spectrum_reg_alpha: 0.1
  num_workers: 0
  amp: false
  log_every: 250
  eval_every: 1000
  save_every: 2000
```

- [ ] **Step 6: Document the DEEP data contract**

Append to `data/README.md`:

```markdown
## DEEP image descriptors

The deep track expects 96-dimensional descriptors from
`deep-image-96-angular`, as `.npy` with shape `[N, 96]`, float32.

Fetch and subset them with:

    python -m src.deep.download --out-dir data

That downloads `deep-image-96-angular.hdf5` once into a shared cache and writes
`data/deep96_250k.npy` (pipeline smoke runs) and `data/deep96_1m.npy` (the real
runs). Neither the HDF5 nor the subsets are committed.

DEEP vectors arrive already L2-normalized, dense, and signed — the opposite of
SIFT on every count. The `deep_gan_v2` config additionally PCA-whitens; its
samples must be drawn with `src.deep.sample`, which inverts that transform.
`src.sample.generate` does not, and would return vectors in whitened space.
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_configs.py -v`
Expected: PASS, 12 tests

- [ ] **Step 8: Commit**

```bash
git add configs/deep_gan_v0.yaml configs/deep_gan_v1.yaml configs/deep_gan_v2.yaml data/README.md tests/test_deep_configs.py
git commit -m "feat(deep): three-rung variant ladder configs"
```

---

### Task 7: Comparison report

**Files:**
- Create: `src/deep/report.py`
- Test: `tests/test_deep_report.py`

**Interfaces:**
- Consumes: `sample_variant` (Task 5); the configs (Task 6); `eda_report.run` and its `ANN_K_DEFAULT` / `ANN_HUB_K_DEFAULT` / `ANN_MAX_ROWS_DEFAULT` / `IVF_NLIST_DEFAULT` constants; `Variant`, `resolve_variants`, `variant_seed` from `src.eval.compare_variants`.
- Produces: `DEEP_VARIANTS: Tuple[Variant, ...]`, `generate_samples(...) -> Path`, `build_report_args(...) -> argparse.Namespace`, `main()`.

**Reuse note:** `Variant`, `resolve_variants`, and `variant_seed` are imported from `compare_variants`, not copied. They are dataset-agnostic — a name, a config path, a run dir, and a name-keyed seed. Only the variant *table* and the sampling call differ, because deep sampling must invert preprocessing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_deep_report.py`:

```python
import argparse
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.eval import compare_variants, eda_report
from src.deep.report import DEEP_VARIANTS, build_report_args


def test_deep_variants_cover_the_whole_ladder():
    assert [v.name for v in DEEP_VARIANTS] == ["v0", "v1", "v2"]


def test_deep_variant_configs_all_exist():
    for variant in DEEP_VARIANTS:
        assert Path(variant.config_path).exists(), variant.config_path


def test_deep_variants_do_not_collide_with_sift_run_dirs():
    """A deep run must never be read out of a SIFT run directory."""
    sift_dirs = {v.run_dir for v in compare_variants.VARIANTS}
    assert not sift_dirs & {v.run_dir for v in DEEP_VARIANTS}


def test_report_args_match_eda_report_fields():
    """Field-for-field parity with eda_report.parse_args is load-bearing.

    If eda_report gains a required argument and this Namespace is not updated,
    sampling hundreds of thousands of vectors succeeds before the mismatch
    surfaces as a runtime AttributeError. Mirrors the SIFT test of the same
    name in tests/test_compare_variants.py.
    """
    args = argparse.Namespace(
        real_path="real.npy",
        real_format="npy",
        output_dir="out",
        max_vectors=100,
        num_pairs=100,
        knn=5,
        ann_k=eda_report.ANN_K_DEFAULT,
        ann_hub_k=eda_report.ANN_HUB_K_DEFAULT,
        ann_max_rows=eda_report.ANN_MAX_ROWS_DEFAULT,
        ivf_nlist=eda_report.IVF_NLIST_DEFAULT,
        bins=80,
        top_divergent=16,
        seed=42,
        no_png=True,
        plotlyjs="inline",
    )
    produced = vars(build_report_args(args, ["v0=samples/v0.npy"]))

    source = inspect.getsource(eda_report.parse_args)
    expected = {
        line.split('"')[1].lstrip("-").replace("-", "_")
        for line in source.splitlines()
        if "add_argument(" in line and '"--' in line
    }
    assert expected - set(produced) == set()


def test_report_args_pass_through_the_ann_settings():
    args = argparse.Namespace(
        real_path="real.npy", real_format="npy", output_dir="out",
        max_vectors=100, num_pairs=100, knn=5,
        ann_k=17, ann_hub_k=3, ann_max_rows=999, ivf_nlist=8,
        bins=80, top_divergent=16, seed=42, no_png=True, plotlyjs="inline",
    )
    produced = build_report_args(args, ["v0=samples/v0.npy"])
    assert produced.ann_k == 17
    assert produced.ann_hub_k == 3
    assert produced.ann_max_rows == 999
    assert produced.ivf_nlist == 8
    assert produced.synthetic_path == ["v0=samples/v0.npy"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.deep.report'`

- [ ] **Step 3: Write the implementation**

Create `src/deep/report.py`:

```python
"""Overlay every trained deep variant on real DEEP in one EDA report.

Usage:

    python -m src.deep.report \
        --real-path data/deep96_1m.npy \
        --output-dir runs/eda_deep

The variant table and the sampling call are the only things that differ from
src/eval/compare_variants.py. Variant, resolve_variants and variant_seed are
imported from there rather than copied: they are dataset-agnostic. Sampling
cannot be shared, because deep samples must be returned to original
coordinates by src/deep/sample.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.deep.sample import sample_variant
from src.eval import eda_report
from src.eval.compare_variants import Variant, resolve_variants, variant_seed

# Ordered so the report legend reads as a progression. Each entry is one
# config delta from the previous.
DEEP_VARIANTS: Tuple[Variant, ...] = (
    Variant("v0", "configs/deep_gan_v0.yaml", "runs/deep_gan_v0"),
    Variant("v1", "configs/deep_gan_v1.yaml", "runs/deep_gan_v1"),
    Variant("v2", "configs/deep_gan_v2.yaml", "runs/deep_gan_v2"),
)


def generate_samples(
    variant: Variant,
    root: Path,
    num_samples: int,
    batch_size: int,
    out_dir: Path,
    seed: int,
) -> Path:
    """Sample a deep variant to an .npy in original coordinates."""
    x = sample_variant(
        root / variant.run_dir,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=variant_seed(seed, variant.name),
    )
    out_path = out_dir / f"{variant.name}.npy"
    np.save(out_path, x)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument(
        "--real-format", type=str, default="auto", choices=["auto", "npy", "fvecs"]
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-vectors", type=int, default=50000)
    parser.add_argument("--num-pairs", type=int, default=200000)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--ann-k", type=int, default=eda_report.ANN_K_DEFAULT)
    parser.add_argument("--ann-hub-k", type=int, default=eda_report.ANN_HUB_K_DEFAULT)
    parser.add_argument(
        "--ann-max-rows", type=int, default=eda_report.ANN_MAX_ROWS_DEFAULT
    )
    parser.add_argument("--ivf-nlist", type=int, default=eda_report.IVF_NLIST_DEFAULT)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--top-divergent", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument(
        "--plotlyjs", type=str, default="inline", choices=["inline", "cdn", "directory"]
    )
    return parser.parse_args()


def build_report_args(
    args: argparse.Namespace, specs: List[str]
) -> argparse.Namespace:
    """Build the Namespace eda_report.run expects from our own parsed args.

    preprocess="l2" because both sides are already unit-norm: real DEEP ships
    that way, and every variant's samples come off normalize_l2.
    """
    return argparse.Namespace(
        real_path=args.real_path,
        real_format=args.real_format,
        synthetic_path=specs,
        synthetic_format="npy",
        output_dir=args.output_dir,
        preprocess="l2",
        max_vectors=args.max_vectors,
        num_pairs=args.num_pairs,
        knn=args.knn,
        ann_k=args.ann_k,
        ann_hub_k=args.ann_hub_k,
        ann_max_rows=args.ann_max_rows,
        ivf_nlist=args.ivf_nlist,
        bins=args.bins,
        top_divergent=args.top_divergent,
        seed=args.seed,
        no_png=args.no_png,
        plotlyjs=args.plotlyjs,
    )


def main() -> None:
    args = parse_args()
    num_samples = args.num_samples if args.num_samples is not None else args.max_vectors
    root = Path(args.root)
    out_dir = Path(args.output_dir)

    found, skipped = resolve_variants(DEEP_VARIANTS, root)
    for variant, reason in skipped:
        print(f"skipping {variant.name}: {reason}")
    if not found:
        raise SystemExit(
            "No deep variant has both a checkpoint and a run config on this "
            "machine. Copy them from the training box, or pass --root at the "
            "tree holding them."
        )

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    specs = []
    for variant in found:
        print(f"sampling {variant.name} from {variant.run_dir}")
        path = generate_samples(
            variant, root, num_samples, args.batch_size, samples_dir, seed=args.seed
        )
        specs.append(f"{variant.name}={path}")

    report_path = eda_report.run(build_report_args(args, specs))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_deep_report.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: 163 passed

- [ ] **Step 6: Commit**

```bash
git add src/deep/report.py tests/test_deep_report.py
git commit -m "feat(deep): ANN-difficulty comparison report for the deep ladder"
```

---

### Task 8: Provision the GPU box and smoke-run the ladder

**Files:**
- Create: `scripts/deep_gpu_setup.sh`, `scripts/deep_train_service.sh`
- Test: manual verification against `tig-gpu` (this task provisions infrastructure; its correctness is observable only on the box)

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: `/workspace/deep-gan/` on `tig-gpu` containing this branch, the DEEP subsets, and a `deepgan_<variant>` supervisor program per rung.

**Isolation constraints — all four are mandatory:**
1. Never read or write `/workspace/wgan-synthetic`; it is a shared checkout.
2. Cap VRAM per process at 25% (~2 GB, roughly 20x the measured 71 MiB peak).
3. Run rungs sequentially, never concurrently.
4. Namespace supervisor programs `deepgan_*`.

- [ ] **Step 1: Write the setup script**

Create `scripts/deep_gpu_setup.sh`:

```bash
#!/usr/bin/env bash
# Provision /workspace/deep-gan on tig-gpu from a git bundle pushed up from a
# local worktree. Never touches /workspace/wgan-synthetic, which is a shared
# checkout other agents may be using.
set -euo pipefail

REMOTE="${REMOTE:-tig-gpu}"
WORK_DIR="/workspace/deep-gan"
CACHE_DIR="/workspace/data-cache"
BUNDLE="/tmp/deep-gan.bundle"
BRANCH="$(git branch --show-current)"

echo "==> bundling ${BRANCH}"
git bundle create "${BUNDLE}" "${BRANCH}"
scp -q "${BUNDLE}" "${REMOTE}:/tmp/deep-gan.bundle"

echo "==> unpacking on ${REMOTE}"
ssh "${REMOTE}" bash -s <<REMOTE_SCRIPT
set -euo pipefail
if [ -d "${WORK_DIR}/.git" ]; then
    cd "${WORK_DIR}"
    git fetch /tmp/deep-gan.bundle "${BRANCH}:refs/heads/${BRANCH}" --force
    git checkout --force "${BRANCH}"
else
    git clone -b "${BRANCH}" /tmp/deep-gan.bundle "${WORK_DIR}"
fi
cd "${WORK_DIR}"
/venv/main/bin/pip install -q -r requirements.txt
mkdir -p "${CACHE_DIR}"
REMOTE_SCRIPT

echo "==> fetching DEEP data (shared cache, downloads only once)"
ssh "${REMOTE}" "cd ${WORK_DIR} && /venv/main/bin/python -m src.deep.download \
    --cache-path ${CACHE_DIR}/deep-image-96-angular.hdf5 --out-dir data"

echo "==> done: ${REMOTE}:${WORK_DIR}"
```

Make it executable: `chmod +x scripts/deep_gpu_setup.sh`

- [ ] **Step 2: Write the training service script**

Create `scripts/deep_train_service.sh`:

```bash
#!/usr/bin/env bash
# Train one deep variant under supervisor on tig-gpu.
#
# Run as a supervisor program, not a bare background process: a loose
# `python ... &` dies with the shell and its logs never reach the portal.
# Invoked as: deep_train_service.sh <variant>
set -euo pipefail

VARIANT="${1:?usage: deep_train_service.sh <v0|v1|v2>}"
WORK_DIR="/workspace/deep-gan"

cd "${WORK_DIR}"

# Courtesy preflight: other agents share this GPU. Report what is already
# resident so a heavy neighbour is visible in the log; the cap below is what
# actually protects them.
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# Return freed blocks instead of holding a fragmented reserve.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec /venv/main/bin/python - "${VARIANT}" <<'PYTHON'
import sys

import torch
import yaml

from src.train.train_wgan_gp import train

variant = sys.argv[1]

# Hard ceiling at 25% of the card, ~20x the measured 71 MiB peak. A runaway
# allocation then fails this job rather than another agent's.
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.25)

with open(f"configs/deep_gan_{variant}.yaml", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

checkpoint, meta = train(config)
print(f"done: {checkpoint}")
print(f"final metrics: {meta['metrics'][-1] if meta['metrics'] else 'none'}")
PYTHON
```

Make it executable: `chmod +x scripts/deep_train_service.sh`

- [ ] **Step 3: Commit the scripts**

```bash
git add scripts/deep_gpu_setup.sh scripts/deep_train_service.sh
git commit -m "chore(deep): GPU provisioning and isolated training service"
```

- [ ] **Step 4: Provision the box**

Run: `REMOTE=tig-gpu ./scripts/deep_gpu_setup.sh`

Expected: the bundle unpacks to `/workspace/deep-gan`, requirements install, and the download prints an `hdf5:` line plus two `subset:` lines ending in `(250000, 96)` and `(1000000, 96)`.

This step downloads ~4 GB. Confirm `/workspace/wgan-synthetic` was untouched:

```bash
ssh tig-gpu 'cd /workspace/wgan-synthetic && git status --short && git log --oneline -1'
```

Expected: identical to before provisioning.

- [ ] **Step 5: Smoke-run each rung on the 250k subset**

For each of `v0`, `v1`, `v2`, run a 200-step version to prove the pipeline end to end before spending an hour on it:

```bash
ssh tig-gpu 'cd /workspace/deep-gan && /venv/main/bin/python - <<PY
import yaml
from src.train.train_wgan_gp import train
for variant in ["v0", "v1", "v2"]:
    with open(f"configs/deep_gan_{variant}.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["real_path"] = "data/deep96_250k.npy"
    cfg["training"]["num_gen_steps"] = 200
    cfg["training"]["eval_every"] = 100
    cfg["training"]["save_every"] = 200
    cfg["output_dir"] = f"runs/smoke_{variant}"
    ckpt, meta = train(cfg)
    print(variant, ckpt, meta["metrics"][-1])
PY'
```

Expected: three checkpoints written; `v0` logs `spectrum_reg == 0.0`, `v1` and `v2` log it positive.

- [ ] **Step 6: Smoke-test the report path**

```bash
ssh tig-gpu 'cd /workspace/deep-gan && /venv/main/bin/python -m src.deep.report \
    --real-path data/deep96_250k.npy --output-dir runs/smoke_report \
    --root . --max-vectors 5000 --no-png' 
```

This reads `runs/deep_gan_*`, which do not exist yet, so expect three `skipping` lines and the `SystemExit`. That is the correct behaviour and confirms `resolve_variants` is wired. To exercise the full path, re-run with the smoke run dirs temporarily symlinked:

```bash
ssh tig-gpu 'cd /workspace/deep-gan && for v in v0 v1 v2; do ln -sfn "$PWD/runs/smoke_$v" "runs/deep_gan_$v"; done && \
    /venv/main/bin/python -m src.deep.report --real-path data/deep96_250k.npy \
    --output-dir runs/smoke_report --root . --max-vectors 5000 --no-png && \
    for v in v0 v1 v2; do rm -f "runs/deep_gan_$v"; done'
```

Expected: `report: runs/smoke_report/report.html`. The symlinks are removed so the real runs write to clean directories.

---

### Task 9: Train the ladder and produce the report

**Files:**
- Create: `docs/deep_ladder_results.md`
- Test: the ANN-difficulty report is the deliverable

**Interfaces:**
- Consumes: everything from Tasks 1-8.
- Produces: three trained checkpoints pulled back locally, and a written comparison.

**Persistence constraint:** `workspace_is_volume=false` on `tig-gpu` — nothing there survives a recycle or destroy. Pull each checkpoint down as its rung finishes, not at the end.

- [ ] **Step 1: Register the supervisor programs**

```bash
ssh tig-gpu 'for v in v0 v1 v2; do cat > /etc/supervisor/conf.d/deepgan_$v.conf <<CONF
[program:deepgan_$v]
environment=PROC_NAME="%(program_name)s"
command=/workspace/deep-gan/scripts/deep_train_service.sh $v
autostart=false
autorestart=false
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
CONF
done
supervisorctl reread && supervisorctl update && supervisorctl status | grep deepgan'
```

Expected: three `deepgan_v0/v1/v2` programs in `STOPPED` state. `autostart=false` and `autorestart=false` are deliberate — rungs are started one at a time by hand, and a crashed 30k-step run should be diagnosed, not silently restarted.

- [ ] **Step 2: Train v0**

```bash
ssh tig-gpu 'supervisorctl start deepgan_v0'
```

Poll until it exits (~24 minutes):

```bash
ssh tig-gpu 'supervisorctl status deepgan_v0; tail -5 /var/log/portal/deepgan_v0.log'
```

Expected: exits with a `done: runs/deep_gan_v0/best_generator.pt` line.

- [ ] **Step 3: Pull v0's artifacts down immediately**

```bash
mkdir -p runs/deep_gan_v0
scp tig-gpu:/workspace/deep-gan/runs/deep_gan_v0/{best_generator.pt,run_config.yaml,run_metadata.json} runs/deep_gan_v0/
```

Do not defer this. The box has no volume.

- [ ] **Step 4: Train v1 and pull it down**

```bash
ssh tig-gpu 'supervisorctl start deepgan_v1'
```

Wait for exit, then:

```bash
mkdir -p runs/deep_gan_v1
scp tig-gpu:/workspace/deep-gan/runs/deep_gan_v1/{best_generator.pt,run_config.yaml,run_metadata.json} runs/deep_gan_v1/
```

- [ ] **Step 5: Train v2 and pull it down**

```bash
ssh tig-gpu 'supervisorctl start deepgan_v2'
```

Wait for exit, then:

```bash
mkdir -p runs/deep_gan_v2
scp tig-gpu:/workspace/deep-gan/runs/deep_gan_v2/{best_generator.pt,run_config.yaml,run_metadata.json} runs/deep_gan_v2/
```

- [ ] **Step 6: Build the comparison report**

Run locally, against the real 1M subset (fetch it down first if not present):

```bash
scp tig-gpu:/workspace/deep-gan/data/deep96_1m.npy data/
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m src.deep.report \
    --real-path data/deep96_1m.npy \
    --output-dir runs/eda_deep \
    --no-png
```

Expected: `report: runs/eda_deep/report.html`, with all three variants present and no `skipping` lines.

- [ ] **Step 7: Write up the results**

Create `docs/deep_ladder_results.md` recording, for real DEEP and each of `v0`/`v1`/`v2`, the four scalars from `ann_difficulty.summary`: `lid_median`, `relative_contrast_median`, `hubness_skew`, `ivf_gini`. Read them from `runs/eda_deep/summary.json`.

State plainly which rung lands closest to real DEEP on the three primary metrics (LID, hubness skew, IVF gini), and say so even if the answer is `v0` — a ladder where the added machinery does not help is a real result, and the spec anticipates it.

- [ ] **Step 8: Tear down the supervisor programs**

Leaving them registered would confuse the next agent on the box:

```bash
ssh tig-gpu 'supervisorctl stop deepgan_v0 deepgan_v1 deepgan_v2 2>/dev/null; \
    rm -f /etc/supervisor/conf.d/deepgan_*.conf && supervisorctl reread && supervisorctl update'
```

- [ ] **Step 9: Commit**

```bash
git add docs/deep_ladder_results.md
git commit -m "docs(deep): ANN-difficulty results for the deep variant ladder"
```

---

## Verification

After Task 9, confirm all of the following before claiming completion:

- [ ] `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q` — all tests pass, including the original 118 SIFT tests.
- [ ] `git diff main --stat -- configs/sift_gan_v0.yaml configs/sift_gan_v1.yaml configs/sift_gan_v1_5.yaml configs/sift_gan_v2.yaml` is empty.
- [ ] `git diff main -- src/eval/compare_variants.py` is empty.
- [ ] `runs/eda_deep/report.html` exists and shows all three variants.
- [ ] `ssh tig-gpu 'ls /etc/supervisor/conf.d/deepgan_*.conf'` returns nothing.
- [ ] `ssh tig-gpu 'cd /workspace/wgan-synthetic && git status --short'` is clean and unchanged.
