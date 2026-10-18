"""Real (mixed-book) and synthetic KV corpora for one (layer, KV head)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from src.kv.transforms import head_keys, head_queries, n_rep


@dataclass
class Corpus:
    keys: np.ndarray
    q_ood: np.ndarray
    q_id: np.ndarray
    key_book: np.ndarray
    q_book: np.ndarray


def real_sizes(T: int, n_books: int, n_sink: int, n_tail: int) -> tuple[int, int, int]:
    return n_books * ((T - n_tail - n_sink) // n_books), n_tail, n_tail


def _np(t):
    return t.detach().float().cpu().numpy()


def mixed_real_corpus(
    h_books: Sequence[torch.Tensor], wk, wq, g, *, n_sink, n_tail, seed
) -> Corpus:
    B = len(h_books)
    T = h_books[0].shape[0]
    if any(h.shape[0] != T for h in h_books):
        raise ValueError(f"books differ in length: {[h.shape[0] for h in h_books]}")
    if n_tail % B:
        raise ValueError(f"n_tail {n_tail} is not divisible by {B} books")
    Tk = T - n_tail
    per, tail_per = (Tk - n_sink) // B, n_tail // B
    r = n_rep(wq, wk)
    rng = np.random.default_rng(seed)
    ks, qs, ids, kb, qb = [], [], [], [], []
    for b, h in enumerate(h_books):
        kpos = torch.as_tensor(
            np.sort(rng.choice(np.arange(n_sink, Tk), per, replace=False))
        )
        tpos = torch.as_tensor(
            np.sort(rng.choice(np.arange(Tk, T), tail_per, replace=False))
        )
        hk, ht = h[kpos.to(h.device)], h[tpos.to(h.device)]
        ks.append(_np(head_keys(hk, wk, g)))
        qs.append(_np(head_queries(ht, wq, g, r)))
        ids.append(_np(head_keys(ht, wk, g)))
        kb.append(np.full(per, b))
        qb.append(np.full(tail_per * r, b))
    return Corpus(
        np.vstack(ks),
        np.vstack(qs),
        np.vstack(ids),
        np.concatenate(kb),
        np.concatenate(qb),
    )


def synthetic_corpus(h_keys, h_query, h_id, wk, wq, g) -> Corpus:
    r = n_rep(wq, wk)
    return Corpus(
        _np(head_keys(h_keys, wk, g)),
        _np(head_queries(h_query, wq, g, r)),
        _np(head_keys(h_id, wk, g)),
        np.full(len(h_keys), -1),
        np.full(len(h_query) * r, -1),
    )
