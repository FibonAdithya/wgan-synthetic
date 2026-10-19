import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import PreprocessConfig, build_training_data
from src.data.kv_capture import (
    h_path,
    load_kv_capture,
    read_safetensor,
    write_bf16_safetensor,
)

FIX = Path(__file__).parent / "fixtures" / "kv"


def test_reads_bf16_written_by_the_real_library():
    """Catches a wrong bf16 decode (byte order, shift, or treating it as f16)."""
    x = read_safetensor(FIX / "tiny_bf16.safetensors", "h")
    assert x.dtype == np.float32
    np.testing.assert_array_equal(
        x, [[1.5, -2.0, 3.140625], [0.0, 0.0009765625, -256.0]]
    )


def test_reads_f32_and_f16_by_name_from_a_multi_tensor_file():
    """Catches using the first tensor's offsets for every name, or no f16 path."""
    np.testing.assert_array_equal(
        read_safetensor(FIX / "tiny_multi.safetensors", "k_proj"),
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )
    np.testing.assert_array_equal(
        read_safetensor(FIX / "tiny_multi.safetensors", "q_proj"), [[0.5, -1.0]]
    )


def test_missing_tensor_name_raises():
    """Catches silently returning another tensor."""
    with pytest.raises(KeyError, match="v_proj"):
        read_safetensor(FIX / "tiny_multi.safetensors", "v_proj")


def test_unsupported_dtype_raises(tmp_path):
    """Catches decoding I64 (or anything else) as if it were float."""
    header = json.dumps(
        {"h": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}}
    ).encode()
    p = tmp_path / "bad.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
    with pytest.raises(ValueError, match="I64"):
        read_safetensor(p, "h")


def _write_book(root, book, layer, arr):
    """Writes a BF16 file the way the box does, via the reader's own inverse.
    Only used for layout tests; decoding is pinned by the fixtures above."""
    p = h_path(root, book, layer)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_bf16_safetensor(p, "h", arr)


def test_load_kv_capture_drops_the_sink_per_book_and_keeps_book_order(tmp_path):
    """Catches dropping the sink only once, or concatenating books out of order."""
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    b = 100 + np.arange(12, dtype=np.float32).reshape(4, 3)
    _write_book(tmp_path, 0, 3, a)
    _write_book(tmp_path, 5, 3, b)
    x = load_kv_capture(tmp_path, 3, [5, 0], n_sink=1)
    np.testing.assert_array_equal(x, np.vstack([b[1:], a[1:]]))


def test_build_training_data_kv_capture_branch(tmp_path):
    """Catches the trainer path ignoring kv_layer / kv_books."""
    rng = np.random.default_rng(0)
    for bk in (0, 1):
        _write_book(tmp_path, bk, 7, rng.normal(size=(50, 4)).astype(np.float32))
    xt, xh, _ = build_training_data(
        str(tmp_path),
        "kv_capture",
        4,
        0.1,
        PreprocessConfig(l2_normalize=False),
        0,
        kv_layer=7,
        kv_books=[0, 1],
        kv_n_sink=1,
    )
    assert xt.shape[0] + xh.shape[0] == 98


def test_write_then_read_round_trips_bf16_exact_values(tmp_path):
    """The reader is pinned by the real-library fixture, so a round trip here
    checks the writer: catches truncation instead of round-to-nearest-even."""
    exact = np.array([[1.5, -2.0, 3.140625]], np.float32)
    write_bf16_safetensor(tmp_path / "a.safetensors", "h", exact)
    np.testing.assert_array_equal(
        read_safetensor(tmp_path / "a.safetensors", "h"), exact
    )
    # 1 + 2**-8 is exactly halfway between bf16 neighbours 1.0 and 1 + 2**-7: ties to even -> 1.0;
    # 1 + 3 * 2**-8 is halfway between 1 + 2**-7 and 1 + 2**-6: ties to even -> 1 + 2**-6.
    tie = np.array([[1 + 2**-8, 1 + 3 * 2**-8]], np.float32)
    write_bf16_safetensor(tmp_path / "b.safetensors", "h", tie)
    np.testing.assert_array_equal(
        read_safetensor(tmp_path / "b.safetensors", "h"), [[1.0, 1 + 2**-6]]
    )
