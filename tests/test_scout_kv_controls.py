import json
import os
import struct

import numpy as np
import pytest

from src.kv.controls import (
    equalise_norms,
    fit_gaussian,
    measure_controls,
    pooled_split,
    sample_gaussian,
)


def _books(n_books=4, t=50, d=8, seed=0):
    rng = np.random.default_rng(seed)
    books = []
    for _ in range(n_books):
        b = rng.normal(size=(t, d)).astype(np.float32)
        b[0] = 1e6  # sentinel for the sink position
        books.append(b)
    return books


def test_pooled_split_rows_sizes_and_disjointness():
    books = _books()
    pool = np.concatenate([b[1:] for b in books])
    out = pooled_split(books, 100, 20, seed=3)
    hk, hq, hi, ik, iq, ii = out
    assert (len(hk), len(hq), len(hi)) == (100, 20, 20)
    for h, i in ((hk, ik), (hq, iq), (hi, ii)):
        assert np.array_equal(np.asarray(h), pool[i])
        assert not (np.asarray(h) == 1e6).any()
    assert len(set(ik) | set(iq) | set(ii)) == 140
    assert len(pool) == 196


def test_pooled_split_n_sink_and_seed_and_too_small():
    books = _books()
    a = pooled_split(books, 100, 20, seed=3)
    b = pooled_split(books, 100, 20, seed=3)
    c = pooled_split(books, 100, 20, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[3], c[3])
    pool2 = np.concatenate([bk[2:] for bk in books])
    _, _, _, ik, _, _ = pooled_split(books, 100, 20, seed=3, n_sink=2)
    assert ik.max() < len(pool2)
    with pytest.raises(ValueError):
        pooled_split(books, 150, 30, seed=0)  # needs 210 > 196


def test_equalise_norms_to_the_joint_mean():
    rng = np.random.default_rng(0)
    hk = rng.normal(size=(30, 6)).astype(np.float32)
    hq = 3 * rng.normal(size=(10, 6)).astype(np.float32)
    hi = 0.5 * rng.normal(size=(10, 6)).astype(np.float32)
    mean = np.concatenate([np.linalg.norm(x, axis=1) for x in (hk, hq, hi)]).mean()
    outs = equalise_norms(hk, hq, hi)
    for x, y in zip((hk, hq, hi), outs, strict=True):
        y = np.asarray(y)
        assert np.allclose(np.linalg.norm(y, axis=1), mean, rtol=1e-5)
        cos = (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
        assert (cos > 1 - 1e-6).all()


def _rank4(n, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 4)) * np.sqrt([0.25, 1.0, 4.0, 9.0])
    return np.concatenate([z, z[:, :1] + z[:, 1:2], z[:, 2:3] - 0.5 * z[:, 3:4]], 1)


def test_gaussian_fit_and_sample_match_moments():
    x = _rank4(5000, 0) + 2.0
    fit = fit_gaussian([x])
    s = np.asarray(sample_gaussian(fit, 200_000, seed=1), dtype=np.float64)
    assert not np.isnan(s).any()
    mu, cov = x.mean(0), np.cov(x.T, bias=True)
    cs = np.cov(s.T)
    assert np.allclose(np.diag(cs), np.diag(cov), rtol=0.02)
    assert np.abs(cs - cov).max() < 0.02 * np.diag(cov).max()
    assert np.allclose(s.mean(0), mu, atol=0.02 * np.sqrt(np.diag(cov).max()))
    again = sample_gaussian(fit, 50, seed=1)
    assert np.array_equal(
        np.asarray(again), np.asarray(sample_gaussian(fit, 50, seed=1))
    )
    assert not np.array_equal(
        np.asarray(again), np.asarray(sample_gaussian(fit, 50, seed=2))
    )


def test_fit_gaussian_chunked_equals_single_pass():
    x = _rank4(1000, 5) + 1.0
    one = fit_gaussian([x])
    two = fit_gaussian([x[:300], x[300:]])
    assert np.allclose(one["mean"], two["mean"], atol=1e-10, rtol=0)
    assert np.allclose(one["cov"], two["cov"], atol=1e-10, rtol=0)


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


def test_measure_controls_end_to_end_on_a_tiny_capture(tmp_path):
    from src.data.kv_capture import h_path, write_bf16_safetensor

    rng = np.random.default_rng(0)
    cap = tmp_path / "cap"
    for b in range(20):
        p = h_path(cap, b, 3)
        p.parent.mkdir(parents=True)
        write_bf16_safetensor(p, "h", rng.normal(size=(64, 16)).astype(np.float32))
    (cap / "weights").mkdir()
    _write_multi(
        cap / "weights" / "layer03.safetensors",
        {
            "k_proj": rng.normal(size=(2 * 128, 16)).astype(np.float32),
            "q_proj": rng.normal(size=(2 * 5 * 128, 16)).astype(np.float32),
        },
    )
    out = tmp_path / "out"
    names = ["pooled_real", "norm_equalised", "gaussian"]
    kw = dict(
        layers=[3],
        controls=names,
        seed=0,
        out_dir=out,
        threads=1,
        train_books=range(12),
        ref_books=range(12, 20),
        n_keys=200,
        n_tail=16,
        nlist=4,
        hnsw_m=8,
        ef_construction=16,
    )
    measure_controls(cap, **kw)
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{n}.jsonl" for n in names)
    for n in names:
        recs = [
            json.loads(line) for line in (out / f"{n}.jsonl").read_text().splitlines()
        ]
        assert len(recs) == 2
        for r in recs:
            assert r["source"] == n and r["control"] == n
            assert (r["n_keys"], r["n_q_id"], r["n_q_ood"]) == (200, 16, 80)
            want = list(range(0, 12)) if n == "gaussian" else list(range(12, 20))
            assert r["books"] == want
    before = {p.name: os.stat(p).st_mtime_ns for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        measure_controls(cap, **kw)
    assert before == {p.name: os.stat(p).st_mtime_ns for p in out.iterdir()}
