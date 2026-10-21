"""Held-out-book tail control (docs/ai/plans/2026-10-21-scout-kv-held-out.md)."""

import json
import os
import struct

import numpy as np
import pytest
import torch

from src.kv.corpus import mixed_real_corpus
from src.kv.transforms import head_keys, head_queries

T, D, NT = 400, 16, 16


def _books(n, t=T, d=D, seed=0):
    rng = np.random.default_rng(seed)
    return [
        torch.from_numpy(rng.normal(size=(t, d)).astype(np.float32)) for _ in range(n)
    ]


def _weights(d=D):
    rng = np.random.default_rng(99)
    wk = torch.from_numpy(rng.normal(size=(2 * 128, d)).astype(np.float32))
    wq = torch.from_numpy(rng.normal(size=(2 * 5 * 128, d)).astype(np.float32))
    return wk, wq


def _corpus(ref, qb, seed=3, g=1):
    from src.kv.controls import held_out_tail_corpus

    wk, wq = _weights()
    return held_out_tail_corpus(ref, qb, wk, wq, g, n_sink=1, n_tail=NT, seed=seed)


# --- Task 1: held_out_tail_corpus ---


def test_held_out_corpus_keeps_reference_keys():
    ref, qb = _books(4, seed=0), _books(2, seed=1)
    wk, wq = _weights()
    want = mixed_real_corpus(ref, wk, wq, 1, n_sink=1, n_tail=NT, seed=3)
    c = _corpus(ref, qb)
    assert np.array_equal(c.keys, want.keys)
    assert np.array_equal(c.key_book, want.key_book)
    assert np.array_equal(c.key_pos, want.key_pos)


def test_held_out_corpus_queries_come_from_query_book_tails():
    ref, qb = _books(4, seed=0), _books(2, seed=1)
    wk, wq = _weights()
    c = _corpus(ref, qb)
    for j, h in enumerate(qb):
        rows = np.flatnonzero(c.q_id_book == 4 + j)
        assert len(rows) == NT // 2
        pos = c.q_id_pos[rows]
        assert len(set(pos.tolist())) == len(pos)
        assert pos.min() >= T - NT and pos.max() < T
        ht = h[torch.as_tensor(pos)]
        assert np.array_equal(c.q_id[rows], head_keys(ht, wk, 1).numpy())
        qrows = (rows[:, None] * 5 + np.arange(5)).ravel()
        assert np.array_equal(c.q_ood[qrows], head_queries(ht, wq, 1, 5).numpy())
    assert set(c.q_id_book.tolist()) == {4, 5}
    assert np.array_equal(c.q_book, np.repeat(c.q_id_book, 5))
    assert np.array_equal(c.q_pos, np.repeat(c.q_id_pos, 5))
    assert c.q_book.min() == 4
    one = _corpus(ref, qb[:1])
    assert set(one.q_id_pos.tolist()) == set(range(T - NT, T))


def test_held_out_corpus_seeds_and_errors():
    ref, qb = _books(4, seed=0), _books(2, seed=1)
    assert np.array_equal(_corpus(ref, qb).q_id_pos, _corpus(ref, qb).q_id_pos)
    assert not np.array_equal(
        _corpus(ref, qb, seed=3).q_id_pos, _corpus(ref, qb, seed=4).q_id_pos
    )
    with pytest.raises(ValueError):
        _corpus(ref, [])
    with pytest.raises(ValueError):
        _corpus(ref, [qb[0], _books(1, t=T - 1, seed=2)[0]])
    with pytest.raises(ValueError):
        _corpus(ref, _books(3, seed=1))


# --- Task 2: measure_held_out ---


def _write_multi(path, tensors):
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        b = np.ascontiguousarray(arr, "<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [off, off + len(b)],
        }
        blobs.append(b)
        off += len(b)
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def _capture(tmp_path):
    from src.data.kv_capture import h_path, write_bf16_safetensor

    rng = np.random.default_rng(0)
    cap = tmp_path / "cap"
    for b in range(20):
        p = h_path(cap, b, 3)
        p.parent.mkdir(parents=True)
        write_bf16_safetensor(p, "h", rng.normal(size=(T, D)).astype(np.float32))
    (cap / "weights").mkdir()
    _write_multi(
        cap / "weights" / "layer03.safetensors",
        {
            "k_proj": rng.normal(size=(2 * 128, D)).astype(np.float32),
            "q_proj": rng.normal(size=(2 * 5 * 128, D)).astype(np.float32),
        },
    )
    return cap


_IDX = dict(n_tail=NT, nlist=4, hnsw_m=8, ef_construction=16, threads=1)
_KW = dict(layers=[3], seed=0, query_books=[0, 1], **_IDX)


def _recs(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_measure_held_out_records(tmp_path):
    from src.eval import kv_gate
    from src.kv.controls import measure_held_out

    cap, out = _capture(tmp_path), tmp_path / "out"
    measure_held_out(cap, out_dir=out, **_KW)
    assert sorted(p.name for p in out.iterdir()) == [
        "held_out_tail.jsonl",
        "rebuild.jsonl",
    ]
    ref_books = list(range(12, 20))
    got = {n: _recs(out / f"{n}.jsonl") for n in ("rebuild", "held_out_tail")}
    for name, recs in got.items():
        assert [(r["layer"], r["kv_head"]) for r in recs] == [(3, 0), (3, 1)]
        for r in recs:
            assert r["books"] == ref_books and r["source"] == "control"
            assert r["control"] == name
            assert (r["n_keys"], r["n_q_ood"], r["n_q_id"]) == (
                8 * ((T - NT - 1) // 8),
                NT * 5,
                NT,
            )
            want_q = ref_books if name == "rebuild" else [0, 1]
            assert r["query_books"] == want_q
    assert all(r["own_book_share_top10"] == 0.0 for r in got["held_out_tail"])
    real = tmp_path / "real.jsonl"
    kv_gate.measure_real(
        cap, books=ref_books, layers=[3], seed=0, out=real, n_sink=1, **_IDX
    )
    skip = {"source", "seconds", "control", "query_books"}
    for a, b in zip(got["rebuild"], _recs(real), strict=True):
        assert {k: v for k, v in a.items() if k not in skip} == {
            k: v for k, v in b.items() if k not in skip
        }


def test_measure_held_out_rejects_bad_arguments(tmp_path):
    from src.kv.controls import measure_held_out

    cap = _capture(tmp_path)
    for i, bad in enumerate(
        (
            {"query_books": [0, 12]},
            {"query_books": []},
            {"controls": ["rebuild", "nope"]},
        )
    ):
        fresh = tmp_path / f"bad{i}"
        with pytest.raises(ValueError):
            measure_held_out(cap, out_dir=fresh, **{**_KW, **bad})
        assert not fresh.exists()
    out = tmp_path / "out"
    measure_held_out(cap, out_dir=out, **_KW)
    before = {p.name: os.stat(p).st_mtime_ns for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        measure_held_out(cap, out_dir=out, **_KW)
    assert before == {p.name: os.stat(p).st_mtime_ns for p in out.iterdir()}
