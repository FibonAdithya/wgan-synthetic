import json

import numpy as np
import pytest
from tests.test_kv_corpus import _setup

from src.eval import kv_gate
from src.kv.corpus import Corpus, mixed_real_corpus, synthetic_corpus

EDGES = [2**i for i in range(16)]


def _bins(**shares):
    """15-bin vector from {"lo": share}, keyed by the bin's lower edge."""
    v = np.zeros(15)
    for lo, s in shares.items():
        v[EDGES.index(int(lo[1:]))] = s
    return v


def _hand_corpus():
    """N1/N2 fixture in D = 2. Book 0 keys at positions 1-20, book 1 at 101-120.
    Both OOD and both key-side queries have top-10 = book-0 positions 11-20."""
    pos0, pos1 = np.arange(1, 21), np.arange(101, 121)
    score0 = np.where(pos0 >= 11, 100.0 + pos0, pos0.astype(float))
    score = np.concatenate([score0, np.full(20, 0.5)])
    keys = np.stack([score, np.zeros(40)], 1).astype(np.float32)
    q = np.array([[1.0, 0.0], [1.0, 0.0]], np.float32)
    return Corpus(
        keys=keys,
        q_ood=q,
        q_id=q.copy(),
        key_book=np.repeat([0, 1], 20),
        q_book=np.array([0, 1]),
        key_pos=np.concatenate([pos0, pos1]),
        q_pos=np.array([40, 130]),
        q_id_pos=np.array([45, 125]),
        q_id_book=np.array([0, 1]),
    )


def test_hand_built_corpus_gives_the_known_ood_statistics():
    """N1. Catches other-book slots counted in the lag histogram (0.5 in
    [16, 32) and 0.5 in [64, 128)), a baseline over all keys instead of
    own-book keys (negative lags for Q1), and the lag sign flipped."""
    from src.kv.neighbours import neighbour_positions

    r = neighbour_positions(_hand_corpus())
    assert r["edges"] == EDGES
    o = r["ood"]
    assert o["own_book_share"] == 0.5
    np.testing.assert_allclose(o["nbr_lag_share"], _bins(e16=1.0))
    np.testing.assert_allclose(
        o["base_lag_share"], _bins(e8=0.15, e16=0.65, e32=0.20), atol=1e-12
    )
    np.testing.assert_allclose(o["nbr_pos_share"], _bins(e8=0.5, e16=0.5))
    np.testing.assert_allclose(
        o["key_pos_share"],
        _bins(e1=1, e2=2, e4=4, e8=8, e16=5, e64=20) / 40,
    )
    assert o["share_pos_lt_16"] == 0.5
    assert o["base_share_pos_lt_16"] == 0.375
    assert o["share_lag_le_1024"] == 1.0
    assert o["base_share_lag_le_1024"] == 1.0
    json.dumps(r)


def test_key_side_queries_use_their_own_positions():
    """N2. Catches the `id` path reading the OOD positions (lags 20-29, all in
    [16, 32))."""
    from src.kv.neighbours import neighbour_positions

    i = neighbour_positions(_hand_corpus())["id"]
    assert i["own_book_share"] == 0.5
    np.testing.assert_allclose(i["nbr_lag_share"], _bins(e16=0.7, e32=0.3))


def test_no_same_book_slots_gives_none_not_zero():
    """A query set whose top-10s are all other-book has no lag statistics.
    Catches a truthiness test that turns a 0 count into a division error or a
    fabricated share."""
    from src.kv.neighbours import neighbour_positions

    c = _hand_corpus()
    c.q_book, c.q_pos = np.array([1, 1]), np.array([130, 130])
    o = neighbour_positions(c)["ood"]
    assert o["own_book_share"] == 0.0
    assert o["nbr_lag_share"] is None and o["share_lag_le_1024"] is None
    json.dumps(o)


def _hub_corpus():
    rng = np.random.default_rng(0)
    keys = np.zeros((150, 8), np.float32)
    keys[:, 1:] = rng.normal(size=(150, 7))
    keys[7] = 0.0
    keys[7, 0] = 1000.0
    q = np.zeros((50, 8), np.float32)
    q[:, 0] = 1.0
    q[:, 1:] = rng.normal(size=(50, 7))
    return Corpus(
        keys=keys,
        q_ood=q,
        q_id=q.copy(),
        key_book=np.zeros(150, int),
        q_book=np.zeros(50, int),
        key_pos=1 + np.arange(150),
        q_pos=np.full(50, 200),
        q_id_pos=np.full(50, 200),
        q_id_book=np.zeros(50, int),
    )


def test_hubs_are_counted_over_slots_and_located_by_position():
    """N3. Catches a share over queries instead of slots (1.0), n_hub taken
    from the query count, and the median taken over all keys."""
    from src.kv.neighbours import neighbour_positions

    c = _hub_corpus()
    gt = np.argpartition(-(c.q_ood @ c.keys.T), 9, axis=1)[:, :10]
    counts = np.bincount(gt.ravel(), minlength=150)
    # Fixture precondition: key 7 is the only key in every top-10.
    assert counts[7] == 50 and (np.delete(counts, 7) < 50).all()
    r = neighbour_positions(c)
    assert r["n_hub"] == 1
    assert r["ood"]["hub_slot_share"] == 0.1
    assert r["ood"]["hub_median_pos"] == 8
    assert r["ood"]["key_median_pos"] == 75.5


def _random_corpus(n_keys, n_q, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n_q, 4)).astype(np.float32)
    return Corpus(
        keys=rng.normal(size=(n_keys, 4)).astype(np.float32),
        q_ood=q,
        q_id=q.copy(),
        key_book=np.zeros(n_keys, int),
        q_book=np.zeros(n_q, int),
        key_pos=1 + np.arange(n_keys),
        q_pos=np.full(n_q, n_keys + 1),
        q_id_pos=np.full(n_q, n_keys + 1),
        q_id_book=np.zeros(n_q, int),
    )


def test_n_hub_follows_the_key_count_not_the_query_count():
    """N3 cannot tell the two apart (150 // 100 and max(1, 50 // 100) are both
    1). Catches n_hub taken from the query count."""
    from src.kv.neighbours import neighbour_positions

    assert neighbour_positions(_random_corpus(300, 10))["n_hub"] == 3
    assert neighbour_positions(_random_corpus(50, 300))["n_hub"] == 1


def test_synthetic_corpus_is_refused():
    """N4. A synthetic corpus has no positions (all -1)."""
    from src.kv.neighbours import neighbour_positions

    hs, wk, wq = _setup(1)
    h = hs[0]
    c = synthetic_corpus(h[:40], h[40:48], h[48:56], wk, wq, 0)
    with pytest.raises(ValueError):
        neighbour_positions(c)


def test_ood_own_book_share_matches_the_gate():
    """N5. Catches k other than 10, and the `id` queries used in place of `ood`."""
    from src.kv.neighbours import neighbour_positions

    hs, wk, wq = _setup(2)
    c = mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=0)
    assert c.keys.shape[0] == 46 and c.q_ood.shape[0] == 80
    got = neighbour_positions(c)["ood"]["own_book_share"]
    assert got == kv_gate._own_book_share_top10(c)


def _write_multi(path, tensors):
    """F32 multi-tensor safetensors, the layout of capture/weights/layerXX.
    Copied from tests/test_kv_gate.py."""
    import struct

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


def _tiny_capture(root):
    """2 books, layer 3, T = 400, D = 16, 2 KV heads (as in
    test_measure_real_end_to_end_on_a_tiny_capture)."""
    from src.data.kv_capture import h_path, write_bf16_safetensor

    rng = np.random.default_rng(0)
    for b in (0, 1):
        p = h_path(root, b, 3)
        p.parent.mkdir(parents=True)
        write_bf16_safetensor(p, "h", rng.normal(size=(400, 16)).astype(np.float32))
    (root / "weights").mkdir()
    _write_multi(
        root / "weights" / "layer03.safetensors",
        {
            "k_proj": rng.normal(size=(2 * 128, 16)).astype(np.float32),
            "q_proj": rng.normal(size=(2 * 5 * 128, 16)).astype(np.float32),
        },
    )


def test_measure_neighbours_pairs_with_measure_real(tmp_path):
    """S1. Catches a seed formula that differs from measure_real's (seed 3, so
    dropping the 7919 * seed term fails), and an overwrite of `out`."""
    from src.kv.neighbours import measure_neighbours

    _tiny_capture(tmp_path)
    real, nbr = tmp_path / "real.jsonl", tmp_path / "nbr.jsonl"
    kv_gate.measure_real(
        tmp_path,
        books=[0, 1],
        layers=[3],
        seed=3,
        out=real,
        n_sink=1,
        n_tail=16,
        nlist=4,
        hnsw_m=8,
        ef_construction=16,
        threads=1,
    )
    measure_neighbours(tmp_path, [0, 1], [3], 3, nbr, n_sink=1, n_tail=16)
    reals = {r["kv_head"]: r for r in map(json.loads, real.read_text().splitlines())}
    nbrs = [json.loads(line) for line in nbr.read_text().splitlines()]
    assert sorted(r["kv_head"] for r in nbrs) == [0, 1]
    for r in nbrs:
        assert r["layer"] == 3 and r["source"] == "real" and r["seed"] == 3
        assert r["books"] == [0, 1] and r["k"] == 10
        assert r["n_keys"] == reals[r["kv_head"]]["n_keys"]
        assert r["ood"]["own_book_share"] == reals[r["kv_head"]]["own_book_share_top10"]
    with pytest.raises(FileExistsError):
        measure_neighbours(tmp_path, [0, 1], [3], 3, nbr, n_sink=1, n_tail=16)


def test_corpus_seed_is_the_gate_formula():
    from src.kv.corpus import corpus_seed

    assert corpus_seed(7, 3, 2) == 1009 * 7 + 3 + 7919 * 2
    assert isinstance(corpus_seed(7, 3, 2), int)
