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
