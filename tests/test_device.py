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
