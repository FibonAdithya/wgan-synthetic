# Run Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a 100k-step training run safe to launch on a shared, ephemeral GPU box — explicit device claiming, an exclusive lock keyed to the physical card, a memory cap, and a resume path that survives preemption.

**Architecture:** One shared device resolver replaces three duplicated copies, gaining a `strict` mode that refuses to let `device: auto` silently grab `cuda:0`. A separate `fcntl.flock`-based module keys an exclusive lock on the GPU's UUID rather than its index. The lock is taken in `main()`, not `train()`, so the existing test suite — which calls `train()` directly — is unaffected. Checkpoints gain the three fields resume needs but does not currently save.

**Tech Stack:** Python 3.12, PyTorch 2.x, pytest, `fcntl` (POSIX advisory locking).

## Global Constraints

- Phase 1 of 3 from `docs/superpowers/specs/2026-08-03-structured-gate-generator-design.md`. The generator and regularizer are separate plans and are **not** in scope here.
- Every task in this plan is testable on CPU. No task requires a GPU.
- Sampling and eval keep today's permissive `auto`; only training becomes strict.
- Existing behaviour must not change when the new config keys are absent. `tests/test_train_smoke.py` must pass untouched at every commit.
- Run tests with the main-repo interpreter: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest`. Worktrees have no `.venv`.
- `flock` is advisory and host-local. It coordinates cooperating processes on one machine and does nothing across hosts.

## File Structure

| File | Responsibility |
|---|---|
| `src/device.py` (create) | Resolve a config `device` string to a `torch.device`. Shared by training, sampling and eval. |
| `src/train/gpu_lock.py` (create) | Key and hold an exclusive lock on a physical GPU; report the holder on contention. Training-only. |
| `src/train/train_wgan_gp.py` (modify) | Use the shared resolver, cap memory, record preflight, persist resume state, accept `--resume`. |
| `src/sample/generate.py` (modify) | Use the shared resolver. Delete its local copy. |
| `src/eval/evaluate_distribution.py` (modify) | Use the shared resolver. Delete its local copy. |
| `tests/test_device.py` (create) | Resolver behaviour, including strict-mode refusal. |
| `tests/test_gpu_lock.py` (create) | Lock acquisition, contention, timeout, holder reporting. |
| `tests/test_resume.py` (create) | Checkpoint round-trip and resumed-run continuation. |

**Note on placement:** the spec proposed `src/train/device.py`. This plan puts the resolver at `src/device.py` instead, because `src/eval/` and `src/sample/` must import it and should not depend on `src/train/`. The repo already treats dependency direction as load-bearing — `src/eval/ann_difficulty.py` documents refusing to import `eda_report` for exactly this reason. `gpu_lock.py` stays under `src/train/` since only training uses it.

---

### Task 1: Shared device resolver with strict training mode

**Files:**
- Create: `src/device.py`
- Create: `tests/test_device.py`
- Modify: `src/train/train_wgan_gp.py:34-41` (delete `get_device`, import the shared one)
- Modify: `src/sample/generate.py:14-21` (delete `get_device`, import the shared one)
- Modify: `src/eval/evaluate_distribution.py:35-42` (delete `get_device`, import the shared one)

**Interfaces:**
- Consumes: nothing.
- Produces: `resolve_device(device_cfg: str, *, strict: bool = False) -> torch.device` and `class DeviceClaimError(RuntimeError)`, both in `src.device`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_device.py`:

```python
import pytest
import torch

from src.device import DeviceClaimError, resolve_device


def test_explicit_device_is_returned_verbatim():
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_falls_back_to_cpu_without_accelerators(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def test_auto_picks_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")


def test_strict_auto_refuses_to_guess_a_gpu(monkeypatch):
    # The failure this exists to prevent: two agents both running `auto` on a
    # shared box silently land on cuda:0 and contend.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(DeviceClaimError):
        resolve_device("auto", strict=True)


def test_strict_accepts_cuda_visible_devices_as_a_deliberate_claim(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert resolve_device("auto", strict=True) == torch.device("cuda")


def test_strict_accepts_an_explicit_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert resolve_device("cuda:1", strict=True) == torch.device("cuda:1")


def test_strict_is_irrelevant_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto", strict=True) == torch.device("cpu")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_device.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.device'`

- [ ] **Step 3: Write the implementation**

Create `src/device.py`:

```python
from __future__ import annotations

import os

import torch


class DeviceClaimError(RuntimeError):
    """Raised when a training run declines to guess which GPU it may use."""


def resolve_device(device_cfg: str, *, strict: bool = False) -> torch.device:
    """Resolve a config `device` string to a torch device.

    `strict` is for long-running training on a shared box. Plain `auto`
    resolves to a bare `cuda`, i.e. `cuda:0`, so two agents both running
    `auto` land on the same card and contend -- silently, until one of them
    fails to allocate hours in. Under `strict`, `auto` is only accepted when
    the process has been pinned by `CUDA_VISIBLE_DEVICES`; anything else must
    name its device.

    Sampling and eval deliberately do not pass `strict`: they are short and
    read-only, and making them refuse to start would be pure friction.
    """
    if device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        if strict and not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise DeviceClaimError(
                "device: auto will not claim a GPU for training on a shared "
                "box. Name the device explicitly in the config (e.g. "
                "device: cuda:0), or pin the process with CUDA_VISIBLE_DEVICES."
            )
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_device.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Replace the three duplicated copies**

In `src/train/train_wgan_gp.py`, delete the `get_device` function (lines 34-41) and add to the import block near `from src.data.sift1m_dataset import load_descriptors`:

```python
from src.device import resolve_device
```

Then change the call site at line ~302 from:

```python
    device = get_device(config["device"])
```

to:

```python
    device = resolve_device(config["device"], strict=True)
```

In `src/sample/generate.py`, delete `get_device` (lines 14-21), add `from src.device import resolve_device`, and change its call site to `resolve_device(device_cfg)` — no `strict`.

In `src/eval/evaluate_distribution.py`, delete `get_device` (lines 35-42), add `from src.device import resolve_device`, and change its call site to `resolve_device(device_cfg)` — no `strict`.

- [ ] **Step 6: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: PASS. The count rises by 7 from the 118 baseline to 125. `tests/test_train_smoke.py` must still pass — it runs on CPU, where `strict` is inert.

- [ ] **Step 7: Commit**

```bash
git add src/device.py tests/test_device.py src/train/train_wgan_gp.py src/sample/generate.py src/eval/evaluate_distribution.py
git commit -m "refactor: one device resolver, with a strict mode for training

Three duplicated get_device copies collapse into src/device.py. Training
now refuses 'auto' when CUDA is present and nothing has pinned the
process, so two agents on a shared box cannot both silently claim cuda:0.
Sampling and eval keep the permissive behaviour."
```

---

### Task 2: GPU lock keyed on the physical card

**Files:**
- Create: `src/train/gpu_lock.py`
- Create: `tests/test_gpu_lock.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `class GpuBusyError(RuntimeError)`
  - `gpu_lock_key(device: torch.device) -> Optional[str]` — `None` for non-CUDA devices.
  - `claim_gpu(device: torch.device, *, run_dir: Path, timeout_s: float = 0.0, poll_s: float = 5.0) -> ContextManager[Optional[Path]]`
  - `_claim_key(key: str, *, run_dir: Path, timeout_s: float, poll_s: float) -> ContextManager[Path]` — the testable core, exercised directly by tests since CI has no GPU.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gpu_lock.py`:

```python
import os
import time
from pathlib import Path

import pytest
import torch

from src.train.gpu_lock import GpuBusyError, _claim_key, claim_gpu, gpu_lock_key


def test_non_cuda_device_needs_no_lock(tmp_path):
    assert gpu_lock_key(torch.device("cpu")) is None
    with claim_gpu(torch.device("cpu"), run_dir=tmp_path) as held:
        assert held is None


def test_sequential_claims_of_the_same_key_both_succeed(tmp_path, monkeypatch):
    monkeypatch.setenv("WGAN_GPU_LOCK_DIR", str(tmp_path))
    with _claim_key("fake-uuid", run_dir=tmp_path, timeout_s=0.0, poll_s=0.01):
        pass
    with _claim_key("fake-uuid", run_dir=tmp_path, timeout_s=0.0, poll_s=0.01):
        pass


def test_second_concurrent_claim_fails_and_names_the_holder(tmp_path, monkeypatch):
    monkeypatch.setenv("WGAN_GPU_LOCK_DIR", str(tmp_path))
    run_a = tmp_path / "run_a"
    with _claim_key("fake-uuid", run_dir=run_a, timeout_s=0.0, poll_s=0.01):
        with pytest.raises(GpuBusyError) as excinfo:
            with _claim_key("fake-uuid", run_dir=tmp_path / "run_b",
                            timeout_s=0.0, poll_s=0.01):
                pass
    # A refusal that does not say who is holding the card is useless to an
    # operator deciding whether to wait or kill.
    message = str(excinfo.value)
    assert "run_a" in message
    assert str(os.getpid()) in message


def test_different_keys_do_not_block_each_other(tmp_path, monkeypatch):
    monkeypatch.setenv("WGAN_GPU_LOCK_DIR", str(tmp_path))
    with _claim_key("uuid-one", run_dir=tmp_path, timeout_s=0.0, poll_s=0.01):
        with _claim_key("uuid-two", run_dir=tmp_path, timeout_s=0.0, poll_s=0.01):
            pass


def test_timeout_gives_up_rather_than_waiting_forever(tmp_path, monkeypatch):
    monkeypatch.setenv("WGAN_GPU_LOCK_DIR", str(tmp_path))
    with _claim_key("fake-uuid", run_dir=tmp_path, timeout_s=0.0, poll_s=0.01):
        started = time.monotonic()
        with pytest.raises(GpuBusyError):
            with _claim_key("fake-uuid", run_dir=tmp_path,
                            timeout_s=0.3, poll_s=0.05):
                pass
        waited = time.monotonic() - started
    assert 0.25 <= waited < 3.0


def test_lock_is_released_even_when_the_body_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("WGAN_GPU_LOCK_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        with _claim_key("fake-uuid", run_dir=tmp_path, timeout_s=0.0, poll_s=0.01):
            raise ValueError("boom")
    with _claim_key("fake-uuid", run_dir=tmp_path, timeout_s=0.0, poll_s=0.01):
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_gpu_lock.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.train.gpu_lock'`

- [ ] **Step 3: Write the implementation**

Create `src/train/gpu_lock.py`:

```python
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import torch


class GpuBusyError(RuntimeError):
    """Raised when another process already holds the requested GPU."""


def _lock_dir() -> Path:
    return Path(os.environ.get("WGAN_GPU_LOCK_DIR", "/tmp"))


def gpu_lock_key(device: torch.device) -> Optional[str]:
    """Stable identifier for the physical card behind `device`.

    Keyed on the GPU UUID, not the index. Two processes with different
    CUDA_VISIBLE_DEVICES mappings both see their card as index 0, so an
    index-keyed lock would hand them *different* locks for the *same*
    physical GPU -- isolation that looks correct and is not.

    Returns None for non-CUDA devices, which need no lock.
    """
    if device.type != "cuda":
        return None
    props = torch.cuda.get_device_properties(device)
    uuid = getattr(props, "uuid", None)
    if uuid is not None:
        return str(uuid).replace("/", "_")
    # Older torch builds do not expose .uuid. Index-keyed is weaker, but a
    # weak lock beats none; the name keeps it readable in the filename.
    index = device.index if device.index is not None else torch.cuda.current_device()
    return f"{props.name}-{index}".replace(" ", "_").replace("/", "_")


@contextmanager
def _claim_key(
    key: str,
    *,
    run_dir: Path,
    timeout_s: float,
    poll_s: float,
) -> Iterator[Path]:
    """Hold an exclusive advisory lock named by `key`.

    Split out from `claim_gpu` so it is testable without a GPU.

    flock is advisory and host-local: it coordinates cooperating processes on
    one machine and does nothing across hosts or against a process that does
    not take the lock. That matches the threat -- other agents running this
    same codebase.
    """
    path = _lock_dir() / f"wgan-gpu-{key}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.monotonic() >= deadline:
                handle.seek(0)
                holder = handle.read().strip() or "(holder wrote no metadata)"
                handle.close()
                raise GpuBusyError(
                    f"GPU lock {path} is held by: {holder}. "
                    f"Waited {timeout_s:.0f}s. Raise "
                    f"training.gpu_lock_timeout_s to queue for longer."
                ) from None
            time.sleep(poll_s)
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "run_dir": str(run_dir),
                    "started": datetime.now(timezone.utc).isoformat(),
                }
            )
        )
        handle.flush()
        yield path
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def claim_gpu(
    device: torch.device,
    *,
    run_dir: Path,
    timeout_s: float = 0.0,
    poll_s: float = 5.0,
) -> Iterator[Optional[Path]]:
    """Hold the GPU behind `device` for the duration of the block."""
    key = gpu_lock_key(device)
    if key is None:
        yield None
        return
    with _claim_key(key, run_dir=run_dir, timeout_s=timeout_s, poll_s=poll_s) as path:
        yield path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_gpu_lock.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/train/gpu_lock.py tests/test_gpu_lock.py
git commit -m "feat(train): exclusive GPU lock keyed on the card's UUID

Keyed on UUID rather than index: two processes with different
CUDA_VISIBLE_DEVICES mappings both see their card as index 0, so an
index-keyed lock would give them different locks for the same GPU.
Waits with a timeout, then refuses and names the holding PID and run
directory."
```

---

### Task 3: Claim the GPU, cap memory, and record preflight state

**Files:**
- Modify: `src/train/train_wgan_gp.py` — add `gpu_preflight`, cap memory inside `train`, take the lock in `main`

**Interfaces:**
- Consumes: `resolve_device` (Task 1), `claim_gpu` and `gpu_lock_key` (Task 2).
- Produces: `gpu_preflight(device: torch.device) -> Dict[str, object]`, and a `run_metadata.json` that gains a `gpu` key.

**Why the lock goes in `main()` and not `train()`:** `tests/test_train_smoke.py` calls `train()` directly five times. Locking inside `train()` would make the suite serialise on a lock it has no reason to care about, and would need the whole 130-line body reindented into a `with`. Locking in `main()` covers every real launch — training is only ever started through the CLI — while leaving tests and the existing structure alone.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_device.py`:

```python
from src.train.train_wgan_gp import gpu_preflight


def test_preflight_reports_the_device_on_cpu():
    meta = gpu_preflight(torch.device("cpu"))
    assert meta["device"] == "cpu"
    # No CUDA fields invented on a CPU box.
    assert "memory_total_bytes" not in meta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_device.py::test_preflight_reports_the_device_on_cpu -v`
Expected: FAIL — `ImportError: cannot import name 'gpu_preflight'`

- [ ] **Step 3: Add `gpu_preflight` to `src/train/train_wgan_gp.py`**

Place it directly after the imports, near where `get_device` used to live:

```python
def gpu_preflight(device: torch.device) -> Dict[str, object]:
    """Snapshot of the card at launch, for `run_metadata.json`.

    Costs nothing and turns "the run died at step 60k" into an answerable
    question -- specifically, whether it was already sharing the card.

    The spec also asked for a list of other compute processes on the card.
    Torch exposes no such API (it needs NVML), and adding a dependency for
    forensics is not worth it: `memory_free_bytes` well below
    `memory_total_bytes` at launch already says someone else is resident,
    which is the only part that changes a decision.
    """
    meta: Dict[str, object] = {"device": str(device)}
    if device.type != "cuda":
        return meta
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    meta.update(
        {
            "name": props.name,
            "uuid": gpu_lock_key(device),
            "memory_free_bytes": int(free),
            "memory_total_bytes": int(total),
        }
    )
    return meta
```

Add to the import block:

```python
from src.train.gpu_lock import claim_gpu, gpu_lock_key
```

- [ ] **Step 4: Cap memory and record preflight inside `train`**

Immediately after `device = resolve_device(config["device"], strict=True)` (line ~302), insert:

```python
    memory_fraction = float(config["training"].get("gpu_memory_fraction", 0.9))
    if device.type == "cuda" and 0.0 < memory_fraction < 1.0:
        # Belt and braces: if the lock is bypassed, a run degrades instead of
        # taking the whole card down with it.
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
```

`run_meta` is created at line ~376 and serialised at line ~522. Add the
preflight snapshot immediately before serialisation, just above:

```python
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
```

the line:

```python
    run_meta["gpu"] = gpu_preflight(device)
```

- [ ] **Step 5: Take the lock in `main`**

Replace `main()` with:

```python
def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    device = resolve_device(config["device"], strict=True)
    train_cfg = config["training"]
    with claim_gpu(
        device,
        run_dir=Path(config["output_dir"]),
        timeout_s=float(train_cfg.get("gpu_lock_timeout_s", 1800.0)),
    ):
        best_ckpt, _ = train(config)
    print(f"Training complete. Best checkpoint: {best_ckpt}")
```

- [ ] **Step 6: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: PASS, 126 tests. `tests/test_train_smoke.py` unaffected — it calls `train()`, which takes no lock.

- [ ] **Step 7: Commit**

```bash
git add src/train/train_wgan_gp.py tests/test_device.py
git commit -m "feat(train): claim the GPU, cap memory, record preflight state

The lock is taken in main() rather than train() so the smoke tests, which
call train() directly, neither serialise on it nor need reindenting.
gpu_memory_fraction defaults to 0.9 so a bypassed lock degrades a run
instead of taking the card down; run_metadata.json now records which card
the run got and how much of it was already in use."
```

---

### Task 4: Persist the three fields resume needs

**Files:**
- Modify: `src/train/train_wgan_gp.py:237-268` (`save_checkpoint`)
- Create: `tests/test_resume.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `save_checkpoint(..., ema_params: Dict[str, Tensor] | None = None, ema_step: int = 0, best_cov: float = float("inf"))`. Checkpoints gain keys `ema_params`, `ema_step`, `best_cov`.

**Context:** `save_checkpoint` already writes `step`, `generator_state_dict`, `critic_state_dict`, `optim_g_state_dict` and `optim_d_state_dict` — most of what a resume needs. Three pieces of live training state are missing. `ema_params` matters most: with `ema_decay: 0.999` a resume that loses the shadow silently restarts a 1000-step average, and `best_generator.pt` is selected from EMA weights, so the damage is invisible until the final artifact is worse than it should be.

- [ ] **Step 1: Write the failing test**

Create `tests/test_resume.py`:

```python
import torch

from src.models.critic import Critic
from src.models.generator import Generator
from src.train.train_wgan_gp import save_checkpoint


def _tiny_setup():
    generator = Generator(latent_dim=4, output_dim=6, hidden_dims=[8])
    critic = Critic(input_dim=6, hidden_dims=[8])
    optim_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optim_d = torch.optim.Adam(critic.parameters(), lr=1e-4)
    return generator, critic, optim_g, optim_d


def test_checkpoint_carries_the_ema_shadow_and_best_cov(tmp_path):
    generator, critic, optim_g, optim_d = _tiny_setup()
    ema = {name: p.detach().clone() for name, p in generator.named_parameters()}

    save_checkpoint(
        generator, critic, optim_g, optim_d, tmp_path, step=500,
        ema_params=ema, ema_step=500, best_cov=0.25,
    )

    ckpt = torch.load(tmp_path / "checkpoint_step_500.pt", weights_only=False)
    assert ckpt["ema_step"] == 500
    assert ckpt["best_cov"] == 0.25
    assert set(ckpt["ema_params"]) == set(ema)
    for name, tensor in ema.items():
        assert torch.allclose(ckpt["ema_params"][name], tensor)


def test_checkpoint_without_ema_records_an_empty_shadow(tmp_path):
    generator, critic, optim_g, optim_d = _tiny_setup()
    save_checkpoint(generator, critic, optim_g, optim_d, tmp_path, step=10)
    ckpt = torch.load(tmp_path / "checkpoint_step_10.pt", weights_only=False)
    assert ckpt["ema_params"] == {}
    assert ckpt["ema_step"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_resume.py -v`
Expected: FAIL — `TypeError: save_checkpoint() got an unexpected keyword argument 'ema_params'`

- [ ] **Step 3: Extend `save_checkpoint`**

Add the three parameters to the signature and the dict. The full replacement body:

```python
def save_checkpoint(
    generator: nn.Module,
    critic: Critic,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    out_dir: Path,
    step: int,
    best: bool = False,
    generator_weights: str = "live",
    ema_params: Optional[Dict[str, Tensor]] = None,
    ema_step: int = 0,
    best_cov: float = float("inf"),
) -> None:
    """Write a checkpoint.

    `generator_weights` records which parameters `generator_state_dict` holds:
    "live" (the currently optimised parameters) or "ema" (the bias-corrected
    EMA swapped in for evaluation). Everything else in the file -- critic and
    both optimiser states -- is always live.

    `ema_params`, `ema_step` and `best_cov` are the live training state a
    resume needs and the model files do not carry. The EMA shadow matters
    most: at decay 0.999 a resume that loses it silently restarts a
    thousand-step average, and since best_generator.pt is chosen from EMA
    weights the damage only shows up in the final artifact.
    """
    if generator_weights not in ("live", "ema"):
        raise ValueError(f"generator_weights must be 'live' or 'ema', got {generator_weights!r}")
    ckpt = {
        "step": step,
        "generator_weights": generator_weights,
        "generator_state_dict": generator.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optim_g_state_dict": optim_g.state_dict(),
        "optim_d_state_dict": optim_d.state_dict(),
        "ema_params": {k: v.detach().cpu() for k, v in (ema_params or {}).items()},
        "ema_step": int(ema_step),
        "best_cov": float(best_cov),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_dir / f"checkpoint_step_{step}.pt")
    if best:
        torch.save(ckpt, out_dir / "best_generator.pt")
```

Confirm `Optional` is imported at the top of the file; add it to the `typing` import if not.

- [ ] **Step 4: Pass the new state at both call sites**

At the `best` call site (line ~498) and the periodic one (line ~511), add the three arguments:

```python
                        ema_params=ema_params,
                        ema_step=ema_step,
                        best_cov=best_cov,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_resume.py -q`
Expected: PASS (2 tests)

- [ ] **Step 6: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: PASS, 128 tests.

- [ ] **Step 7: Commit**

```bash
git add src/train/train_wgan_gp.py tests/test_resume.py
git commit -m "feat(train): checkpoint the EMA shadow, ema_step and best_cov

Checkpoints already carried both optimiser states; these three were the
remaining gap between what is saved and what a resume needs. The EMA
shadow is the important one -- at decay 0.999 a resume that drops it
restarts a thousand-step average, and best_generator.pt is selected from
EMA weights, so the loss is invisible until the final artifact."
```

---

### Task 5: Resume a run from a checkpoint

**Files:**
- Modify: `src/train/train_wgan_gp.py` — `train` signature, restore block, loop bounds, `parse_args`, `main`
- Modify: `tests/test_resume.py` — add the end-to-end test

**Interfaces:**
- Consumes: the checkpoint format from Task 4.
- Produces: `train(config: Dict, resume: Optional[str] = None) -> Tuple[Path, Dict]` and a `--resume` CLI flag.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_resume.py`:

```python
import pytest

from src.train.train_wgan_gp import train


def _smoke_config(tmp_path, num_gen_steps, save_every):
    return {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "run"),
        "data": {
            # None plus synthetic_if_missing is how tests/test_train_smoke.py
            # drives training without the 512MB dataset.
            "real_path": None,
            "format": "npy",
            "descriptor_dim": 8,
            "holdout_fraction": 0.2,
            "synthetic_if_missing": True,
            "synthetic_num_vectors": 256,
            "preprocess": {"center": False, "whiten": False, "l2_normalize": True},
        },
        "model": {
            "latent_dim": 4,
            "generator_hidden_dims": [8],
            "critic_hidden_dims": [8],
            "negative_slope": 0.2,
        },
        "training": {
            "batch_size": 16, "num_gen_steps": num_gen_steps, "n_critic": 1,
            "lr_g": 1e-4, "lr_d": 1e-4, "betas": [0.0, 0.9], "lambda_gp": 5.0,
            "ema_decay": 0.9, "num_workers": 0, "distance_reg_alpha": 0.0,
            "distance_reg_max_points": 16, "amp": False,
            "log_every": 100, "eval_every": 100, "save_every": save_every,
        },
    }


def test_resuming_continues_from_the_saved_step(tmp_path):
    cfg = _smoke_config(tmp_path, num_gen_steps=4, save_every=2)
    train(cfg)
    ckpt_path = tmp_path / "run" / "checkpoint_step_4.pt"
    assert ckpt_path.exists()

    cfg_more = _smoke_config(tmp_path, num_gen_steps=6, save_every=2)
    _, meta = train(cfg_more, resume=str(ckpt_path))

    # A resumed run must not redo the first four steps.
    assert meta["resumed_from_step"] == 4
    assert (tmp_path / "run" / "checkpoint_step_6.pt").exists()


def test_resuming_restores_the_ema_shadow(tmp_path):
    cfg = _smoke_config(tmp_path, num_gen_steps=4, save_every=2)
    train(cfg)
    ckpt = torch.load(tmp_path / "run" / "checkpoint_step_4.pt", weights_only=False)
    assert ckpt["ema_step"] == 4
    assert ckpt["ema_params"], "EMA shadow must be non-empty at ema_decay 0.9"


def test_resuming_past_the_step_budget_is_rejected(tmp_path):
    cfg = _smoke_config(tmp_path, num_gen_steps=4, save_every=2)
    train(cfg)
    ckpt_path = tmp_path / "run" / "checkpoint_step_4.pt"
    # Asking to resume into a budget already exhausted is a config mistake,
    # not a no-op: silently doing nothing would look like a successful run.
    with pytest.raises(ValueError, match="already at or past"):
        train(_smoke_config(tmp_path, num_gen_steps=4, save_every=2),
              resume=str(ckpt_path))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_resume.py -v`
Expected: FAIL — `TypeError: train() got an unexpected keyword argument 'resume'`

- [ ] **Step 3: Add the resume path to `train`**

Change the signature:

```python
def train(config: Dict, resume: Optional[str] = None) -> Tuple[Path, Dict]:
```

After `best_cov = float("inf")` (line ~388) and before the training loop, insert:

```python
    start_step = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator_state_dict"])
        critic.load_state_dict(ckpt["critic_state_dict"])
        optim_g.load_state_dict(ckpt["optim_g_state_dict"])
        optim_d.load_state_dict(ckpt["optim_d_state_dict"])
        start_step = int(ckpt["step"])
        ema_step = int(ckpt.get("ema_step", 0))
        best_cov = float(ckpt.get("best_cov", float("inf")))
        if use_ema:
            saved = ckpt.get("ema_params") or {}
            # An EMA-enabled resume from a checkpoint written without a shadow
            # would restart the average silently, which is the exact failure
            # persisting it was meant to prevent. Refuse instead.
            if not saved:
                raise ValueError(
                    f"{resume} carries no EMA shadow but ema_decay is "
                    f"{ema_decay}; resuming would silently restart the average"
                )
            ema_params = {k: v.to(device) for k, v in saved.items()}
        if start_step >= num_gen_steps:
            raise ValueError(
                f"checkpoint is at step {start_step}, already at or past "
                f"num_gen_steps={num_gen_steps}; raise the budget to continue"
            )
```

Change the loop bound from `for step in range(1, num_gen_steps + 1):` to:

```python
    for step in range(start_step + 1, num_gen_steps + 1):
```

Where `run_meta` is assembled, add:

```python
    run_meta["resumed_from_step"] = start_step
```

- [ ] **Step 4: Add the CLI flag**

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WGAN-GP for SIFT1M-like descriptors.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to continue from. The config's num_gen_steps is the "
             "target total, not an additional budget.",
    )
    return parser.parse_args()
```

and in `main`, change `train(config)` to `train(config, resume=args.resume)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest tests/test_resume.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Run the full suite**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: PASS, 131 tests.

- [ ] **Step 7: Commit**

```bash
git add src/train/train_wgan_gp.py tests/test_resume.py
git commit -m "feat(train): resume a run from a checkpoint

num_gen_steps is the target total rather than an extra budget, so a
resumed run stops where the original would have. Refuses to resume an
EMA-enabled run from a checkpoint with no shadow, and refuses a
checkpoint already past the budget -- both would otherwise look like
successful no-op runs."
```

---

### Task 6: Document the infrastructure

**Files:**
- Modify: `PROJECT_DOCUMENTATION.md` — extend the "Device behavior" section
- Modify: `FOLLOWUPS.md` — record the off-box sync gap

**Context:** `PROJECT_DOCUMENTATION.md` is the human source of truth and its "Device behavior" section currently describes the auto-selection this plan changed. Leaving it stale would contradict the code.

Off-box sync is specified but **not implemented by this plan**. The spec calls for syncing `summary.json`, logs and checkpoints off the ephemeral box. That is an operational concern with no natural home in this codebase and no test that would prove it works, so it is recorded as a follow-up rather than half-built here.

- [ ] **Step 1: Replace the "Device behavior" section in `PROJECT_DOCUMENTATION.md`**

```markdown
## Device behavior

Resolved by `src/device.py` (`resolve_device`), shared by training, sampling
and eval.

Order for `device: auto`:

1. CUDA (if available)
2. MPS (Apple Metal, if available)
3. CPU fallback

Training passes `strict=True`, which **rejects `auto` when CUDA is present
and `CUDA_VISIBLE_DEVICES` is unset**. Plain `auto` resolves to a bare
`cuda`, i.e. `cuda:0`, so on a shared box two runs silently land on the same
card. Name the device in the config (`device: cuda:0`) or pin the process.
Sampling and eval stay permissive -- they are short and read-only.

### GPU claiming

`src/train/gpu_lock.py` takes an exclusive `flock` for the duration of a run,
keyed on the card's **UUID** rather than its index, since two processes with
different `CUDA_VISIBLE_DEVICES` mappings both see their card as index 0. The
lock is acquired in `main()`, so it covers every CLI launch but not direct
`train()` calls from tests.

| Config key | Default | Meaning |
|---|---|---|
| `training.gpu_lock_timeout_s` | `1800` | Seconds to queue for a busy card before giving up. |
| `training.gpu_memory_fraction` | `0.9` | Per-process cap, so a bypassed lock degrades a run rather than taking the card down. |

`flock` is advisory and host-local: it coordinates cooperating processes on
one machine, and does nothing across hosts or against a process that does not
take the lock.

`run_metadata.json` records a `gpu` block with the card's name, UUID and free
and total memory at launch.

### Resume

`--resume <checkpoint>` continues a run. `num_gen_steps` is the target
**total**, not an additional budget. Checkpoints carry both optimiser states,
the EMA shadow, `ema_step` and `best_cov`. Resuming an EMA-enabled run from a
checkpoint without a shadow is refused rather than silently restarting the
average.
```

- [ ] **Step 2: Add the follow-up**

Append to `FOLLOWUPS.md`:

```markdown
## Run infrastructure

### Off-box sync is specified but not implemented

`docs/superpowers/specs/2026-08-03-structured-gate-generator-design.md` calls
for syncing run artifacts off the vast.ai box, whose `workspace_is_volume` is
`false` -- nothing survives recycle or destroy. Resume protects against
contention and preemption, but only if checkpoints leave the machine.

Still needed: sync `summary.json`, logs and `run_metadata.json` on the
`eval_every` cadence, and pull `best_generator.pt` after each arm. Deferred
because it is an operational concern with no natural home in this codebase
and no test that would prove it works. A shell script beside
`data/sample_sift1m_100k.sh` is the likely shape.
```

- [ ] **Step 3: Verify the suite still passes**

Run: `/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q`
Expected: PASS, 131 tests. Documentation-only changes.

- [ ] **Step 4: Commit**

```bash
git add PROJECT_DOCUMENTATION.md FOLLOWUPS.md
git commit -m "docs: device claiming, GPU lock and resume

Records off-box sync as an explicit gap rather than leaving the spec's
durability story looking complete."
```

---

## Verification

After Task 6, confirm the whole phase:

```bash
/home/fibonadithya/TIG/wgan-synthetic/.venv/bin/python -m pytest -q
```

Expected: 131 passed, up from the 118 baseline.

Then confirm the strict-mode refusal actually fires on hardware, which no CPU
test can prove:

```bash
ssh tig-gpu 'cd /workspace/wgan-synthetic && \
  /venv/main/bin/python -m src.train.train_wgan_gp --config configs/sift_gan_v2.yaml'
```

Expected: exits immediately with `DeviceClaimError`, because every existing
config sets `device: auto`. **This is the correct result** — it is the whole
point of the task. Then confirm the lock by setting `device: cuda:0` in a
scratch config and launching two runs at once: the second must refuse with the
first's PID and run directory in the message.

Note that box's checkout is roughly 15 commits stale on
`experiment/wgan-improvements` with uncommitted changes to
`train_wgan_gp.py`, and other agents share it. Establish ownership before
refreshing it.

## Not in this plan

- **`StructuredGateGenerator`** — phase 2, its own plan.
- **The log-ratio regularizer** — phase 3, its own plan. Inert at
  `lid_reg_alpha: 0.0`, so it can land while v3 is running.
- **Off-box sync** — recorded in `FOLLOWUPS.md` (Task 6).
- **Config files for v3/v4** — they belong with the phases that need them.
