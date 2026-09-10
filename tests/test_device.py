import pytest
import torch

from src.device import DeviceClaimError, resolve_device
from src.train.train_wgan_gp import gpu_preflight


def test_explicit_device_is_returned_verbatim():
    assert resolve_device("cpu") == torch.device("cpu")


def test_explicit_device_beats_an_available_accelerator(monkeypatch):
    # An explicit config must win over autodetection, not merely agree with it
    # on a box that has no accelerator to detect.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_falls_back_to_cpu_without_accelerators(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def test_auto_picks_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")


def test_auto_prefers_cuda_over_mps_when_both_are_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
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


def test_preflight_reports_the_device_on_cpu():
    meta = gpu_preflight(torch.device("cpu"))
    assert meta["device"] == "cpu"
    # No CUDA fields invented on a CPU box.
    assert "memory_total_bytes" not in meta


def test_cuda_device_index_keeps_an_explicit_index():
    from src.device import cuda_device_index

    assert cuda_device_index(torch.device("cuda:1")) == 1


def test_cuda_device_index_resolves_a_bare_cuda_to_the_current_card(monkeypatch):
    # torch >= 2.13 refuses a bare `cuda` in set_per_process_memory_fraction
    # ("Expected a torch.device with a specified index or an integer"), which
    # is exactly what resolve_device("auto") returns, so the index has to be
    # filled in before the call.
    from src.device import cuda_device_index

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert cuda_device_index(torch.device("cuda")) == 3


def test_cuda_device_index_refuses_a_non_cuda_device():
    from src.device import cuda_device_index

    with pytest.raises(ValueError):
        cuda_device_index(torch.device("cpu"))
