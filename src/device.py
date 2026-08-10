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
