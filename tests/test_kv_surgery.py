"""Tests for src.kv.surgery: real h with one property removed (a scout_kv diagnostic)."""

import json

import numpy as np
import pytest
import torch
from tests.test_kv_neighbours import _tiny_capture

from src.data.kv_capture import load_kv_capture
from src.eval import kv_gate
from src.kv import surgery

T, D, NS, NT, BLK = 40, 6, 1, 8, 8


def _books(n=2, seed=0, t=T, d=D):
    """Correlated, heavy-tailed books: column 1 follows column 0, column 2 is cubed."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, t, d, generator=g)
    x[..., 1] = 2 * x[..., 0] + 0.1 * x[..., 1]
    x[..., 2] = x[..., 2] ** 3
    return [b.clone() for b in x]


def _z(books, pca):
    return [(b - pca.mean) @ pca.basis for b in books]


def test_block_ids_split_body_from_the_sink_and_keep_the_tail_whole():
    """Catches a tail starting one row early or late, body blocks counted from position 0
    instead of the first non-sink row, and the sink given a block."""
    ids = surgery.block_ids(T, NS, NT, BLK).tolist()
    assert ids[0] == -1
    # body positions 1..31: (p - 1) // 8 -> blocks 0..3, the last holding 7 rows
    assert ids[1:9] == [0] * 8 and ids[25:32] == [3] * 7
    assert ids[32:] == [4] * 8


def test_fit_pca_gives_an_orthonormal_basis_with_descending_variance():
    """Catches the sink rows included in the fit, ascending eigenvalue order (eigh's
    default), and an unnormalised or transposed basis."""
    books = _books()
    pca = surgery.fit_pca(books, NS)
    rows = torch.cat([b[NS:] for b in books]).double()
    torch.testing.assert_close(pca.mean.double(), rows.mean(0), atol=1e-5, rtol=0)
    eye = pca.basis.T @ pca.basis
    torch.testing.assert_close(eye, torch.eye(D), atol=1e-5, rtol=0)
    assert torch.all(pca.var[:-1] >= pca.var[1:])
    z = (rows - pca.mean.double()) @ pca.basis.double()
    torch.testing.assert_close(
        z.var(0, unbiased=False), pca.var.double(), atol=1e-4, rtol=1e-4
    )


def test_shuffle_keeping_every_component_returns_the_books():
    """Catches the rotation back done with the basis instead of its transpose."""
    books = _books()
    pca = surgery.fit_pca(books, NS)
    out = surgery.shuffle_components(
        books, pca, keep=D, n_sink=NS, n_tail=NT, seed=0, block=BLK
    )
    for a, b in zip(out, books, strict=True):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_shuffle_keeps_each_blocks_values_and_the_top_components():
    """Catches shuffling across blocks or books (the per-block sorted values change),
    shuffling the kept components, and touching the sink row."""
    books = _books()
    pca = surgery.fit_pca(books, NS)
    keep = 2
    out = surgery.shuffle_components(
        books, pca, keep=keep, n_sink=NS, n_tail=NT, seed=0, block=BLK
    )
    ids = surgery.block_ids(T, NS, NT, BLK)
    for zi, zo, bi, bo in zip(_z(books, pca), _z(out, pca), books, out, strict=True):
        torch.testing.assert_close(bo[:NS], bi[:NS], atol=0, rtol=0)
        torch.testing.assert_close(zo[:, :keep], zi[:, :keep], atol=1e-4, rtol=1e-4)
        for blk in range(int(ids.max()) + 1):
            rows = ids == blk
            torch.testing.assert_close(
                zo[rows, keep:].sort(0).values,
                zi[rows, keep:].sort(0).values,
                atol=1e-4,
                rtol=1e-4,
            )


def test_shuffle_moves_each_component_independently():
    """Catches one permutation applied to every component (a row shuffle), which keeps
    every row intact and so removes nothing: the share of rows whose (z0, z1) pair
    survives must be far below 1."""
    books = _books(t=200)
    pca = surgery.fit_pca(books, NS)
    out = surgery.shuffle_components(
        books, pca, keep=0, n_sink=NS, n_tail=NT, seed=0, block=64
    )
    kept = []
    for zi, zo in zip(_z(books, pca), _z(out, pca), strict=True):
        pairs = {tuple(np.round(r, 3)) for r in zi[NS:, :2].numpy()}
        kept += [tuple(np.round(r, 3)) in pairs for r in zo[NS:, :2].numpy()]
    assert np.mean(kept) < 0.2


def test_shuffle_is_seeded():
    """Catches an unseeded generator (two runs differ) or a seed that is ignored."""
    books = _books()
    pca = surgery.fit_pca(books, NS)
    kw = dict(keep=0, n_sink=NS, n_tail=NT, block=BLK)
    a = surgery.shuffle_components(books, pca, seed=1, **kw)
    b = surgery.shuffle_components(books, pca, seed=1, **kw)
    c = surgery.shuffle_components(books, pca, seed=2, **kw)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert not all(torch.equal(x, y) for x, y in zip(a, c, strict=True))


def test_gaussianise_keeps_ranks_and_moments_and_removes_heavy_tails():
    """Catches a no-op (the heavy-tailed component keeps its kurtosis), ranks taken per
    book instead of over the pooled rows, a broken rank order, a wrong scale, and a
    changed sink row."""
    books = _books(t=200)
    pca = surgery.fit_pca(books, NS)
    out = surgery.gaussianise_components(books, pca, n_sink=NS, chunk=4)
    zi = torch.cat([z[NS:] for z in _z(books, pca)])
    zo = torch.cat([z[NS:] for z in _z(out, pca)])
    for b, o in zip(books, out, strict=True):
        torch.testing.assert_close(o[:NS], b[:NS], atol=0, rtol=0)
    for j in range(D):
        assert torch.equal(zi[:, j].argsort(), zo[:, j].argsort())
    torch.testing.assert_close(zo.mean(0), zi.mean(0), atol=1e-3, rtol=0)
    torch.testing.assert_close(zo.std(0), zi.std(0), atol=1e-3, rtol=1e-3)

    def kurt(z):
        c = z - z.mean(0)
        return (c**4).mean(0) / (c**2).mean(0) ** 2 - 3

    assert kurt(zi).max() > 3
    assert kurt(zo).abs().max() < 0.5


def test_gaussian_books_match_the_fit_and_replace_every_row_but_the_sink():
    """Catches the sink replaced, input rows passed through, the variance applied as a
    standard deviation (or not at all), and an unseeded draw."""
    books = _books(n=2, t=4001, d=4)
    pca = surgery.fit_pca(books, NS)
    out = surgery.gaussian_books(books, pca, n_sink=NS, seed=0)
    again = surgery.gaussian_books(books, pca, n_sink=NS, seed=0)
    assert all(torch.equal(a, b) for a, b in zip(out, again, strict=True))
    for b, o in zip(books, out, strict=True):
        assert o.shape == b.shape
        torch.testing.assert_close(o[:NS], b[:NS], atol=0, rtol=0)
        assert not torch.isclose(o[NS:], b[NS:]).all(1).any()
    rows = torch.cat([o[NS:] for o in out]).double()
    cov = torch.cov(rows.T, correction=0)
    want = (pca.basis.double() * pca.var.double()) @ pca.basis.double().T
    torch.testing.assert_close(cov, want, atol=0.15 * want.abs().max().item(), rtol=0)


def test_transform_dispatches_every_set_and_rounds_to_bf16():
    """Catches a set name with no transform, `real` changing the books, an unknown name
    accepted, and an output the capture format could not hold."""
    books = [b.bfloat16().float() for b in _books()]
    pca = surgery.fit_pca(books, NS)
    kw = dict(n_sink=NS, n_tail=NT, seed=0, block=BLK)
    real = surgery.transform("real", books, pca, **kw)
    assert all(torch.equal(a, b) for a, b in zip(real, books, strict=True))
    for name in surgery.SETS:
        out = surgery.transform(name, books, pca, **kw)
        assert len(out) == len(books)
        for o in out:
            assert torch.equal(o, o.bfloat16().float())
    with pytest.raises(ValueError, match="unknown set"):
        surgery.transform("nope", books, pca, **kw)


def test_keep_counts_cover_the_shuffle_sets():
    """Catches a shuffle set whose kept-component count drifts from its name."""
    assert surgery.KEEP == {"s0": 0, "s32": 32, "s256": 256}
    assert set(surgery.KEEP) < set(surgery.SETS)


def test_transform_seeds_differ_per_layer_seed_and_set():
    """Catches a transform seed that ignores one of its three inputs."""
    seeds = {
        surgery.transform_seed(layer, seed, name)
        for layer in (11, 15, 43)
        for seed in (0, 1)
        for name in surgery.SETS
    }
    assert len(seeds) == 3 * 2 * len(surgery.SETS)


def test_flat_rows_match_load_kv_capture(tmp_path):
    """Catches the intrinsic rows or flags misaligned with the capture loader (sink not
    dropped, tail flags on the wrong rows)."""
    _tiny_capture(tmp_path)
    books = kv_gate.load_books(tmp_path, 3, [0, 1])
    x, flags = surgery.flat_rows(books, n_sink=1, n_tail=16)
    want_x, want_flags = load_kv_capture(tmp_path, 3, [0, 1], n_sink=1, n_tail=16)
    np.testing.assert_array_equal(x, want_x)
    np.testing.assert_array_equal(flags, want_flags)


SMALL = dict(n_sink=1, n_tail=16, nlist=4, hnsw_m=8, ef_construction=16, threads=1)


def _run(tmp_path, out, sets, seeds=(0,)):
    surgery.run_surgery(
        tmp_path,
        books=[0, 1],
        layers=[3],
        sets=sets,
        seeds=seeds,
        out_dir=out,
        block=64,
        intrinsic_kw=dict(k=10, k_hub=5, nlist=4, max_rows=2000),
        **SMALL,
    )


def test_run_surgery_real_set_reproduces_measure_real(tmp_path):
    """Catches the real set measured through a different split or seed than the
    reference: its gate fields equal measure_real's for the same books and seed."""
    _tiny_capture(tmp_path)
    ref = tmp_path / "ref.jsonl"
    kv_gate.measure_real(tmp_path, books=[0, 1], layers=[3], seed=0, out=ref, **SMALL)
    out = tmp_path / "surgery"
    _run(tmp_path, out, ["real"])
    want = [json.loads(line) for line in ref.read_text().splitlines()]
    got = [json.loads(line) for line in (out / "gate.jsonl").read_text().splitlines()]
    assert len(got) == len(want) == 2
    for w, g in zip(want, got, strict=True):
        for key in (
            "kv_head",
            "n_keys",
            "own_book_share_top10",
            "plain_ood_k10",
            "plain_id_k128",
        ):
            assert g[key] == w[key], key
        assert g["source"] == "real" and g["seed"] == 0
        assert g["nbr_n_distinct_keys"] >= 1


def test_run_surgery_writes_every_set_once_and_resumes(tmp_path):
    """Catches a set skipped, intrinsic lines missing a flag, and a rerun that appends
    duplicates instead of skipping finished (layer, set, seed) units, including a unit
    whose gate records were written but whose intrinsic lines were not."""
    _tiny_capture(tmp_path)
    out = tmp_path / "surgery"
    sets = list(surgery.SETS)
    _run(tmp_path, out, sets[:2])
    intr_path = out / "intrinsic.jsonl"
    lines = intr_path.read_text().splitlines()
    intr_path.write_text(
        "".join(line + "\n" for line in lines if json.loads(line)["source"] != sets[0])
    )
    _run(tmp_path, out, sets)
    gate = [json.loads(line) for line in (out / "gate.jsonl").read_text().splitlines()]
    intr = [
        json.loads(line) for line in (out / "intrinsic.jsonl").read_text().splitlines()
    ]
    assert sorted((r["source"], r["kv_head"]) for r in gate) == sorted(
        (s, g) for s in sets for g in (0, 1)
    )
    assert sorted((r["source"], r["flag"]) for r in intr) == sorted(
        (s, f) for s in sets for f in ("body", "tail")
    )
    assert all(r["layer"] == 3 and r["seed"] == 0 for r in gate + intr)


def test_measure_neighbours_records_the_given_source(tmp_path):
    """Catches the source label hard-coded to `real` when a v2 capture is measured."""
    from src.kv.neighbours import measure_neighbours

    _tiny_capture(tmp_path)
    out = tmp_path / "n.jsonl"
    measure_neighbours(
        tmp_path, [0, 1], [3], 0, out, n_sink=1, n_tail=16, source="v2_s42"
    )
    assert {json.loads(line)["source"] for line in out.read_text().splitlines()} == {
        "v2_s42"
    }
