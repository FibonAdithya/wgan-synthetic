import numpy as np
import pytest
import torch

from src.kv.corpus import mixed_real_corpus, real_sizes, synthetic_corpus

D, T = 16, 64


def _setup(n_books, seed=0):
    rng = np.random.default_rng(seed)
    # Encode (book, position) in h so selected rows are identifiable after projection:
    # with W_k = identity on the first 2 dims, key[:, :2] == (book, position).
    hs = []
    for b in range(n_books):
        h = np.zeros((T, D), np.float32)
        h[:, 0], h[:, 1] = b, np.arange(T)
        h[:, 2:] = rng.normal(size=(T, D - 2))
        hs.append(torch.from_numpy(h))
    wk = torch.zeros(2 * 128, D)
    wk[0, 0] = wk[1, 1] = 1.0
    wq = torch.as_tensor(rng.normal(size=(2 * 5 * 128, D)), dtype=torch.float32)
    return hs, wk, wq


def test_real_sizes_eight_books_matches_the_spec():
    """Catches the 4-book count (31,740) leaking into the 8-book reference."""
    assert real_sizes(32768, 8, 1, 1024) == (31736, 1024, 1024)


def test_mixed_corpus_never_indexes_sink_or_tail_and_balances_books():
    """Catches indexing position 0 or the tail, or unequal per-book shares."""
    hs, wk, wq = _setup(4)
    c = mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=3)
    book, pos = c.keys[:, 0], c.keys[:, 1]
    assert pos.min() >= 1 and pos.max() < T - 16
    assert np.bincount(book.astype(int)).tolist() == [(T - 16 - 1) // 4] * 4
    assert (c.key_book == book).all()


def test_single_book_indexes_exactly_positions_1_to_before_tail():
    """With one book every indexable position is drawn, so the set is exact.
    Catches indexing position 0 or any tail position, deterministically."""
    hs, wk, wq = _setup(1)
    c = mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=0)
    assert sorted(c.keys[:, 1].astype(int).tolist()) == list(range(1, T - 16))


def test_mixed_corpus_queries_come_from_tail_five_per_h():
    """Catches queries from indexed positions, or a wrong query count."""
    hs, wk, wq = _setup(4)
    c = mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=3)
    assert c.q_ood.shape == (16 * 5, 128) and c.q_id.shape == (16, 128)
    assert (c.q_id[:, 1] >= T - 16).all()
    assert np.bincount(c.q_book).tolist() == [4 * 5] * 4


def test_mixed_corpus_is_seed_reproducible_and_seed_sensitive():
    """Catches unseeded sampling (flaky gate) or a seed that is ignored."""
    hs, wk, wq = _setup(4)
    a = mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=1)
    b = mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=1)
    c = mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=2)
    np.testing.assert_array_equal(a.keys, b.keys)
    assert not np.array_equal(a.keys, c.keys)


def test_unequal_book_lengths_raise():
    hs, wk, wq = _setup(2)
    hs[1] = hs[1][:-1]
    with pytest.raises(ValueError, match="length"):
        mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=0)


def test_tail_not_divisible_by_books_raises():
    hs, wk, wq = _setup(3)
    with pytest.raises(ValueError, match="divisible"):
        mixed_real_corpus(hs, wk, wq, 0, n_sink=1, n_tail=16, seed=0)


def test_synthetic_corpus_shapes():
    """Catches keys and key-side queries drawn from the same h (they must be independent)."""
    hs, wk, wq = _setup(1)
    h = hs[0]
    c = synthetic_corpus(h[:40], h[40:48], h[48:56], wk, wq, 0)
    assert (
        c.keys.shape == (40, 128)
        and c.q_ood.shape == (40, 128)
        and c.q_id.shape == (8, 128)
    )
    np.testing.assert_array_equal(c.q_id[:, 1], np.arange(48, 56))
