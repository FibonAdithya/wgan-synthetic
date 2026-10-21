"""Held-out-book tail control (docs/ai/plans/2026-10-21-scout-kv-held-out.md)."""


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
