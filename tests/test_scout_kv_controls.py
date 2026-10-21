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
    s = np.asarray(
        sample_gaussian(fit, 200_000, seed=1, device="cpu"), dtype=np.float64
    )
    assert not np.isnan(s).any()
    mu, cov = x.mean(0), np.cov(x.T, bias=True)
    cs = np.cov(s.T)
    assert np.allclose(np.diag(cs), np.diag(cov), rtol=0.02)
    assert np.abs(cs - cov).max() < 0.02 * np.diag(cov).max()
    assert np.allclose(s.mean(0), mu, atol=0.02 * np.sqrt(np.diag(cov).max()))
    again = sample_gaussian(fit, 50, seed=1, device="cpu")
    assert np.array_equal(
        np.asarray(again),
        np.asarray(sample_gaussian(fit, 50, seed=1, device="cpu")),
    )
    assert not np.array_equal(
        np.asarray(again),
        np.asarray(sample_gaussian(fit, 50, seed=2, device="cpu")),
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


# --- tail_split (cross-book plan, Task 1) ---


def _id_books(ids, t=64, d=6, seed=0):
    """Books whose rows encode (book id, position) in the first two columns."""
    rng = np.random.default_rng(seed)
    out = []
    for i in ids:
        b = rng.normal(size=(t, d)).astype(np.float32)
        b[:, 0] = i
        b[:, 1] = np.arange(t)
        out.append(b)
    return out


def _tail(kb=None, qb=None, n_keys=160, n_tail=16, seed=0, **kw):
    from src.kv.controls import tail_split

    kb = _id_books(range(4)) if kb is None else kb
    qb = _id_books(range(4, 8), seed=1) if qb is None else qb
    return [np.asarray(x) for x in tail_split(kb, qb, n_keys, n_tail, seed, **kw)]


def test_tail_split_positions_and_books():
    hk, hq, hi, kp, qp = _tail()
    assert (len(hk), len(hq), len(hi)) == (160, 16, 16)
    assert set(hk[:, 0]) <= {0, 1, 2, 3} and set(hq[:, 0]) <= {4, 5, 6, 7}
    assert hk[:, 1].min() >= 1 and hk[:, 1].max() < 48
    assert hq[:, 1].min() >= 48 and hq[:, 1].max() < 64
    assert np.array_equal(np.bincount(hk[:, 0].astype(int)), [40] * 4)
    assert np.array_equal(np.bincount(hq[:, 0].astype(int))[4:], [4] * 4)
    assert len({tuple(r) for r in hk[:, :2]}) == 160
    assert len({tuple(r) for r in hq[:, :2]}) == 16
    assert np.array_equal(kp[:, 0], hk[:, 0]) and np.array_equal(kp[:, 1], hk[:, 1])
    assert np.array_equal(qp[:, 0] + 4, hq[:, 0]) and np.array_equal(qp[:, 1], hq[:, 1])


def test_tail_split_id_is_query_rows():
    _, hq, hi, _, _ = _tail()
    assert np.array_equal(hq, hi)


def test_tail_split_key_rows_ignore_query_books():
    # A different number of query books changes how many draws the tail takes,
    # so a stream shared with the keys would move the key rows.
    a = _tail()
    b = _tail(qb=_id_books([4], seed=1))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[3], b[3])


def test_tail_split_seeds_and_errors():
    a, b, c = _tail(seed=0), _tail(seed=0), _tail(seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[3], c[3]) and not np.array_equal(a[4], c[4])
    with pytest.raises(ValueError):
        _tail(qb=_id_books(range(4, 8), t=63))
    for kw in (dict(n_keys=161), dict(n_tail=18), dict(n_keys=4 * 48)):
        with pytest.raises(ValueError):
            _tail(**kw)
    full = _tail(n_keys=4 * 47)
    assert len({tuple(r) for r in full[0][:, :2]}) == 188
    whole = _tail(qb=_id_books([4], seed=1))  # one query book takes the whole tail
    assert sorted(whole[1][:, 1]) == list(range(48, 64))


# --- tail controls in measure_controls (cross-book plan, Task 2) ---


def _id_capture(tmp_path, t=64, d=16):
    """Tiny capture whose book b has h[t] = [b, t, noise...] (bf16-exact columns)."""
    from src.data.kv_capture import h_path, write_bf16_safetensor

    rng = np.random.default_rng(0)
    cap = tmp_path / "cap"
    for b in range(20):
        h = rng.normal(size=(t, d)).astype(np.float32)
        h[:, 0], h[:, 1] = b, np.arange(t)
        p = h_path(cap, b, 3)
        p.parent.mkdir(parents=True)
        write_bf16_safetensor(p, "h", h)
    (cap / "weights").mkdir()
    _write_multi(
        cap / "weights" / "layer03.safetensors",
        {
            "k_proj": rng.normal(size=(2 * 128, d)).astype(np.float32),
            "q_proj": rng.normal(size=(2 * 5 * 128, d)).astype(np.float32),
        },
    )
    return cap


_TAIL_KW = dict(
    layers=[3],
    controls=["same_book_tail", "cross_book_tail"],
    seed=0,
    threads=1,
    ref_books=range(12, 20),
    key_books=(13, 15, 17, 19),
    n_keys=160,
    n_tail=16,
    nlist=4,
    hnsw_m=8,
    ef_construction=16,
)


def test_measure_controls_tail_controls_feed_the_right_rows(tmp_path, monkeypatch):
    from src.eval import kv_gate

    seen = {}

    def spy(hk, hq, hi, capture_dir, layer, *, source, extra, **kw):
        seen[source] = (
            np.asarray(hk.cpu()),
            np.asarray(hq.cpu()),
            np.asarray(hi.cpu()),
            extra,
        )
        return []

    monkeypatch.setattr(kv_gate, "measure_h_layer", spy)
    measure_controls(_id_capture(tmp_path), out_dir=tmp_path / "out", **_TAIL_KW)
    sk, sq, si, s_extra = seen["same_book_tail"]
    ck, cq, ci, c_extra = seen["cross_book_tail"]
    assert set(sk[:, 0]) == {13, 15, 17, 19} and set(sq[:, 0]) == {13, 15, 17, 19}
    assert set(ck[:, 0]) == {13, 15, 17, 19} and set(cq[:, 0]) == {12, 14, 16, 18}
    assert sq[:, 1].min() >= 48 and cq[:, 1].min() >= 48 and sk[:, 1].max() < 48
    assert np.array_equal(sk, ck)
    assert np.array_equal(si, sq) and np.array_equal(ci, cq)
    assert s_extra == {
        "control": "same_book_tail",
        "books": [13, 15, 17, 19],
        "query_books": [13, 15, 17, 19],
    }
    assert c_extra == {
        "control": "cross_book_tail",
        "books": [13, 15, 17, 19],
        "query_books": [12, 14, 16, 18],
    }


def test_measure_controls_tail_controls_records(tmp_path):
    cap, out = _id_capture(tmp_path), tmp_path / "out"
    measure_controls(cap, out_dir=out, **_TAIL_KW)
    names = ["cross_book_tail", "same_book_tail"]
    assert sorted(p.name for p in out.iterdir()) == [f"{n}.jsonl" for n in names]
    for n in names:
        recs = [json.loads(x) for x in (out / f"{n}.jsonl").read_text().splitlines()]
        assert len(recs) == 2
        for r in recs:
            assert r["source"] == n and r["control"] == n
            assert (r["n_keys"], r["n_q_id"], r["n_q_ood"]) == (160, 16, 80)
            assert r["books"] == [13, 15, 17, 19]
            want = [12, 14, 16, 18] if n == "cross_book_tail" else [13, 15, 17, 19]
            assert r["query_books"] == want
    with pytest.raises(FileExistsError):
        measure_controls(cap, out_dir=out, **_TAIL_KW)
    for bad in ((12, 99), range(12, 20)):
        fresh = tmp_path / f"bad{len(list(bad))}"
        with pytest.raises(ValueError):
            measure_controls(cap, out_dir=fresh, **{**_TAIL_KW, "key_books": bad})
        assert not fresh.exists()
