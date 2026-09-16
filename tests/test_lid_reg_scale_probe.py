"""The sizing probe must run against this project's configs and stay where it
is told.

Every NYTimes config says `device: auto`, which `torch.device` cannot parse --
only `src.device.resolve_device` handles it. And a probe that picks its device
from the config alone takes the GPU even when submitted to the cpu lane, which
is the collision the queue exists to prevent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_SPEC = importlib.util.spec_from_file_location(
    "lid_reg_scale_probe",
    Path(__file__).resolve().parents[1] / "tools" / "probes" / "lid_reg_scale_probe.py",
)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def test_auto_config_resolves_without_raising():
    """Catches torch.device(cfg["device"]) coming back: it raises on "auto"."""
    got = probe.resolve_probe_device({"device": "auto"}, None)
    assert isinstance(got, torch.device)


def test_delegates_to_resolve_device(monkeypatch):
    """Catches a fix that special-cases the string "auto" in the probe instead
    of delegating, and catches passing strict=True.

    Comparing return values cannot catch either: on a CPU-only host -- which
    is what CI is -- resolve_device("auto") and a hand-rolled `if auto: cpu`
    both return cpu, and strict never fires because it sits behind a
    cuda.is_available() check. Only observing the call discriminates.
    """
    seen = {}

    def fake(device_cfg, **kwargs):
        seen["cfg"] = device_cfg
        seen["kwargs"] = kwargs
        return torch.device("cpu")

    monkeypatch.setattr(probe, "resolve_device", fake)
    probe.resolve_probe_device({"device": "auto"}, None)
    assert seen["cfg"] == "auto"
    assert seen["kwargs"].get("strict", False) is False


def test_device_argument_overrides_the_config():
    """Catches an override that is parsed but ignored, which would put a
    cpu-lane job on the card."""
    got = probe.resolve_probe_device({"device": "cuda:0"}, "cpu")
    assert got == torch.device("cpu")


def test_config_is_used_when_no_override_is_given():
    """Catches an override that always wins, which would ignore the config."""
    got = probe.resolve_probe_device({"device": "cpu"}, None)
    assert got == torch.device("cpu")
