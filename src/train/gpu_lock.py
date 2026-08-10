from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import torch


class GpuBusyError(RuntimeError):
    """Raised when another process already holds the requested GPU."""


def _lock_dir() -> Path:
    return Path(os.environ.get("WGAN_GPU_LOCK_DIR", "/tmp"))


def gpu_lock_key(device: torch.device) -> str | None:
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
    t0 = time.monotonic()
    deadline = t0 + max(0.0, timeout_s)
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.monotonic() >= deadline:
                elapsed = time.monotonic() - t0
                handle.seek(0)
                holder = handle.read().strip() or "(holder wrote no metadata)"
                handle.close()
                raise GpuBusyError(
                    f"GPU lock {path} is held by: {holder}. "
                    f"Waited {elapsed:.0f}s. Raise "
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
                    "started": datetime.now(UTC).isoformat(),
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
) -> Iterator[Path | None]:
    """Hold the GPU behind `device` for the duration of the block."""
    key = gpu_lock_key(device)
    if key is None:
        yield None
        return
    with _claim_key(key, run_dir=run_dir, timeout_s=timeout_s, poll_s=poll_s) as path:
        yield path
