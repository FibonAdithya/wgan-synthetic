import json
import re
import subprocess
import sys

import numpy as np
import pytest

from src.eval import kv_gate
from src.eval.check_gate import EXIT_FAIL, EXIT_PASS, EXIT_UNSET

GATE = kv_gate.GATES_FILE  # gates/scout_kv.yaml
RESULTS = GATE.parents[1] / "docs/results/scout-kv"


def _rec(L, g, reached=True, rec10=0.8, ivf_o=0.2, ivf_i=0.1, n_keys=31736):
    def part(r, reach, iv, ivr):
        return {
            "hnsw_reached": reach,
            "hnsw_recall_op": r,
            "ivf_scan_frac": iv,
            "ivf_reached": iv is not None,
            "ivf_recall_op": ivr,
            "hnsw_ndis": 100.0,
        }

    return {
        "layer": L,
        "kv_head": g,
        "n_keys": n_keys,
        "n_q_ood": 5120,
        "n_q_id": 1024,
        "n_sink": 1,
        "n_tail": 1024,
        "conditions": {
            "nlist": 256,
            "hnsw_m": 48,
            "ef_construction": 500,
            "target": 0.9,
        },
        "plain_ood_k10": part(rec10, reached, ivf_o, 0.8),
        "plain_id_k10": part(1.0, True, ivf_i, 0.9),
        "plain_ood_k128": part(rec10, reached, ivf_o, 0.8),
        "plain_id_k128": part(1.0, True, ivf_i, 0.9),
    }


def _full(**kw):
    return [_rec(L, g, **kw) for L in kv_gate.GLOBAL_LAYERS for g in range(8)]


def test_pooled_statistics_values():
    """Catches mean/median mix-ups and the ratio direction (ood / id, not id / ood)."""
    R = _full()
    for r in R[:24]:
        r["plain_ood_k10"]["hnsw_reached"] = False
    s = kv_gate.pooled_statistics(R)
    assert s["hnsw_reach_k10"] == pytest.approx(72 / 96)
    assert s["ivf_ratio_k128"] == pytest.approx(2.0)
    assert s["hnsw_recall_k10_ef176_id"] == 1.0


def test_ratio_skips_unreached_and_is_none_when_all_unreached():
    """Catches dividing None, or a falsy-zero guard dropping a legitimate 0.0 scan."""
    R = _full()
    for r in R:
        r["plain_ood_k128"]["ivf_scan_frac"] = None
    assert kv_gate.pooled_statistics(R)["ivf_ratio_k128"] is None
    R = _full(ivf_o=0.0)
    assert kv_gate.pooled_statistics(R)["ivf_ratio_k128"] == 0.0


@pytest.mark.parametrize(
    "part, field, statistic",
    [
        ("plain_ood_k10", "hnsw_recall_op", "hnsw_recall_k10_ef176"),
        ("plain_ood_k128", "ivf_recall_op", "ivf_recall_k128_np24"),
        ("plain_id_k10", "hnsw_recall_op", "hnsw_recall_k10_ef176_id"),
    ],
)
def test_null_recall_fails_its_statistic_instead_of_crashing(part, field, statistic):
    """Catches pystats.median raising TypeError on a None, which would crash
    `check` with a traceback instead of returning a fail verdict."""
    R = _full()
    R[0][part][field] = None
    rep = kv_gate.evaluate(R, kv_gate.load_kv_gate(GATE))
    by_name = {s["statistic"]: s for s in rep["statistics"]}
    assert rep["verdict"] == "fail"
    assert by_name[statistic]["status"] == "fail"
    assert by_name[statistic]["value"] is None


def test_report_counts_the_corpora_the_ivf_ratio_pools():
    """Catches the ratio silently pooling fewer than 96 corpora."""
    R = _full()
    for r in R[:10]:
        r["plain_ood_k128"]["ivf_scan_frac"] = None
    R[10]["plain_id_k128"]["ivf_scan_frac"] = 0.0
    rep = kv_gate.evaluate(R, kv_gate.load_kv_gate(GATE))
    assert rep["ivf_ratio_k128_corpora"] == 85


def _gate_with(**bands):
    gate = kv_gate.load_kv_gate(GATE)
    for k, v in bands.items():
        gate["statistics"][k] = v
    return gate


def test_one_null_band_makes_the_verdict_unset():
    """Catches a null band being skipped instead of holding the verdict at unset."""
    gate = _gate_with(**{n: {"min": -1e9, "max": 1e9} for n in kv_gate.STATISTICS})
    gate["statistics"]["ivf_ratio_k128"] = None
    rep = kv_gate.evaluate(_full(), gate)
    assert rep["verdict"] == "unset" and rep["exit_code"] == EXIT_UNSET


def _committed(name):
    path = RESULTS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.mark.parametrize("seed", range(5))
def test_shipped_gate_passes_every_real_draw(seed):
    """Catches a shipped band too narrow for the real reference's own spread."""
    rep = kv_gate.evaluate(
        _committed(f"reference/seed{seed}"), kv_gate.load_kv_gate(GATE)
    )
    assert rep["verdict"] == "pass", rep["reasons"]


@pytest.mark.parametrize("checkpoint", ["best", "final"])
def test_shipped_gate_fails_v0_on_four_statistics(checkpoint):
    """Catches a shipped band loose enough to admit v0, or one wide enough to stop
    rejecting it on a statistic it misses (only ivf_recall_k128_np24 should pass)."""
    rep = kv_gate.evaluate(_committed(f"v0/{checkpoint}"), kv_gate.load_kv_gate(GATE))
    failed = {s["statistic"] for s in rep["statistics"] if s["status"] == "fail"}
    assert rep["verdict"] == "fail"
    assert failed == set(kv_gate.STATISTICS) - {"ivf_recall_k128_np24"}


def test_set_band_passes_inside_and_fails_outside():
    """Catches an inverted comparison."""
    gate = _gate_with(**{n: {"min": -1e9, "max": 1e9} for n in kv_gate.STATISTICS})
    gate["statistics"]["hnsw_reach_k10"] = {"min": 0.5, "max": 0.6}
    R = _full()
    for r in R[:40]:
        r["plain_ood_k10"]["hnsw_reached"] = False  # reach = 56/96 = 0.583
    assert kv_gate.evaluate(R, gate)["verdict"] == "pass"
    for r in R[:60]:
        r["plain_ood_k10"]["hnsw_reached"] = False  # 36/96 = 0.375
    assert kv_gate.evaluate(R, gate)["verdict"] == "fail"


@pytest.mark.parametrize("mutate", ["key_count", "missing", "duplicate", "nlist"])
def test_condition_mismatch_fails(mutate):
    """Catches comparing a measurement taken under other conditions."""
    R = _full()
    if mutate == "key_count":
        R[0]["n_keys"] = 31740
    elif mutate == "missing":
        R.pop()
    elif mutate == "duplicate":
        R.append(
            dict(R[0])
        )  # 97 records, full coverage: only a duplicate/count check catches it
    else:
        R[3]["conditions"]["nlist"] = 1024
    rep = kv_gate.evaluate(R, kv_gate.load_kv_gate(GATE))
    assert rep["verdict"] == "fail" and rep["reasons"]


def _cli_check(records, *extra):
    return subprocess.run(
        [sys.executable, "-m", "src.eval.kv_gate", "check", "--records", str(records)]
        + list(extra),
        capture_output=True,
        text=True,
    )


def test_cli_check_exit_codes(tmp_path):
    """Catches the CLI exiting 0 on a fail or an unset gate."""
    out = _cli_check(RESULTS / "reference/seed0.jsonl")
    assert out.returncode == EXIT_PASS, out.stderr
    out = _cli_check(RESULTS / "v0/best.jsonl")
    assert out.returncode == EXIT_FAIL, out.stderr
    unset = tmp_path / "unset.yaml"
    unset.write_text(
        re.sub(r"(?m)^(  \w+):\n(    \w+: .*\n)+", r"\1: null\n", GATE.read_text())
    )
    out = _cli_check(RESULTS / "reference/seed0.jsonl", "--gate-file", str(unset))
    assert out.returncode == EXIT_UNSET, out.stderr
    assert json.loads(out.stdout)["verdict"] == "unset"


def _write_multi(path, tensors):
    """F32 multi-tensor safetensors, the layout of capture/weights/layerXX."""
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


def test_measure_real_end_to_end_on_a_tiny_capture(tmp_path):
    """Catches measure-real not iterating KV heads, mis-sizing corpora, or not
    recording conditions. Tiny sizes: 2 books, 1 layer, T=400, D=16. T must give
    more than 128 keys, because profile_corpus takes k=128 ground truth."""
    from src.data.kv_capture import h_path, write_bf16_safetensor

    rng = np.random.default_rng(0)
    for b in (0, 1):
        p = h_path(tmp_path, b, 3)
        p.parent.mkdir(parents=True)
        write_bf16_safetensor(p, "h", rng.normal(size=(400, 16)).astype(np.float32))
    (tmp_path / "weights").mkdir()
    _write_multi(
        tmp_path / "weights" / "layer03.safetensors",
        {
            "k_proj": rng.normal(size=(2 * 128, 16)).astype(np.float32),
            "q_proj": rng.normal(size=(2 * 5 * 128, 16)).astype(np.float32),
        },
    )
    out = tmp_path / "r.jsonl"
    kv_gate.measure_real(
        tmp_path,
        books=[0, 1],
        layers=[3],
        seed=0,
        out=out,
        n_sink=1,
        n_tail=16,
        nlist=4,
        hnsw_m=8,
        ef_construction=16,
        threads=1,
    )
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted((r["layer"], r["kv_head"]) for r in recs) == [(3, 0), (3, 1)]
    for r in recs:
        assert (
            r["n_keys"] == 2 * ((400 - 16 - 1) // 2)
            and r["n_q_ood"] == 16 * 5
            and r["n_q_id"] == 16
        )
        assert (
            r["conditions"]["nlist"] == 4
            and r["books"] == [0, 1]
            and r["source"] == "real"
        )
        assert 0.0 <= r["own_book_share_top10"] <= 1.0
    with pytest.raises(FileExistsError):
        kv_gate.measure_real(
            tmp_path,
            books=[0, 1],
            layers=[3],
            seed=0,
            out=out,
            n_sink=1,
            n_tail=16,
            nlist=4,
            hnsw_m=8,
            ef_construction=16,
            threads=1,
        )


def test_load_books_keeps_position_zero(tmp_path):
    """Catches measure_real loading books through load_kv_capture, which already
    drops the sink: mixed_real_corpus would then drop a second position, and the
    key count can't show it (31,742 // 8 == 31,743 // 8)."""
    from src.data.kv_capture import h_path, write_bf16_safetensor

    a = np.arange(48, dtype=np.float32).reshape(12, 4)
    p = h_path(tmp_path, 0, 3)
    p.parent.mkdir(parents=True)
    write_bf16_safetensor(p, "h", a)
    (b,) = kv_gate.load_books(tmp_path, 3, [0])
    assert b.shape == (12, 4)
    np.testing.assert_array_equal(np.asarray(b)[0], a[0])


def test_measure_run_layer_on_a_tiny_trained_run(tmp_path):
    """Catches measure_run_layer mis-sizing the synthetic corpora, skipping KV
    heads, or not recording the conditions."""
    from tests.test_train_smoke import make_config

    from src.train.train_wgan_gp import train

    cfg = make_config(tmp_path, "mlp")
    cfg["data"]["preprocess"] = {"center": True, "whiten": False, "l2_normalize": False}
    cfg["model"]["output_normalize"] = False
    cfg["training"]["num_gen_steps"] = 2
    ckpt, _ = train(cfg)
    run_dir = ckpt.parent

    cap = tmp_path / "capture"
    (cap / "weights").mkdir(parents=True)
    rng = np.random.default_rng(0)
    _write_multi(
        cap / "weights" / "layer03.safetensors",
        {
            "k_proj": rng.normal(size=(2 * 128, 16)).astype(np.float32),
            "q_proj": rng.normal(size=(2 * 5 * 128, 16)).astype(np.float32),
        },
    )
    recs = kv_gate.measure_run_layer(
        run_dir,
        cap,
        3,
        seed=0,
        n_keys=200,
        n_tail=16,
        nlist=4,
        hnsw_m=8,
        ef_construction=16,
    )
    assert [r["kv_head"] for r in recs] == [0, 1]
    for r in recs:
        assert r["layer"] == 3 and r["source"] == "synthetic"
        assert r["n_keys"] == 200 and r["n_q_ood"] == 16 * 5 and r["n_q_id"] == 16
        assert r["conditions"]["nlist"] == 4 and r["conditions"]["hnsw_m"] == 8
        assert "plain_ood_k128" in r and "plain_id_k10" in r
    assert all(r["n_sink"] == 1 for r in recs)
    assert all(
        r["checkpoint"] == "best_generator.pt" and r["generator_weights"] == "ema"
        for r in recs
    )


def _tiny_run(tmp_path):
    from tests.test_train_smoke import make_config

    from src.train.train_wgan_gp import train

    cfg = make_config(tmp_path, "mlp")
    cfg["data"]["preprocess"] = {"center": True, "whiten": False, "l2_normalize": False}
    cfg["model"]["output_normalize"] = False
    cfg["training"]["num_gen_steps"] = 4
    ckpt, _ = train(cfg)
    run_dir = ckpt.parent
    cap = tmp_path / "capture"
    (cap / "weights").mkdir(parents=True)
    rng = np.random.default_rng(0)
    wk = rng.normal(size=(2 * 128, 16)).astype(np.float32)
    _write_multi(
        cap / "weights" / "layer03.safetensors",
        {
            "k_proj": wk,
            "q_proj": rng.normal(size=(2 * 5 * 128, 16)).astype(np.float32),
        },
    )
    return run_dir, cap, wk


_TINY = dict(seed=0, n_keys=200, n_tail=16, nlist=4, hnsw_m=8, ef_construction=16)


def _spy_keys(monkeypatch):
    seen = []
    real = kv_gate.profile_corpus

    def spy(keys, *a, **k):
        seen.append(np.array(keys))
        return real(keys, *a, **k)

    monkeypatch.setattr(kv_gate, "profile_corpus", spy)
    return seen


def test_measure_run_layer_refuses_a_centered_run_without_metadata(tmp_path):
    """Catches un-centering silently becoming an identity when
    run_metadata.json is missing."""
    run_dir, cap, _ = _tiny_run(tmp_path)
    (run_dir / "run_metadata.json").unlink()
    with pytest.raises(ValueError, match="run_metadata.json"):
        kv_gate.measure_run_layer(run_dir, cap, 3, **_TINY)


def test_measure_run_layer_keys_are_in_original_coordinates(tmp_path, monkeypatch):
    """Catches a deleted invert_samples call: keys must equal W_k[0] @ h for
    h = generator samples plus the training mean, recomputed independently."""
    import torch
    import yaml

    from src.eval.compare_variants import invert_samples, load_preprocess_state
    from src.eval.evaluate_distribution import load_generator
    from src.train.train_wgan_gp import sample_generator

    run_dir, cap, wk = _tiny_run(tmp_path)
    seen = _spy_keys(monkeypatch)
    kv_gate.measure_run_layer(run_dir, cap, 3, **_TINY)

    cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text())
    # Draw on the device the gate uses: torch.randn streams differ between CPU
    # and CUDA for the same seed, so a CPU draw cannot match a CUDA measurement.
    dev = kv_gate._device()
    gen = load_generator(cfg, run_dir / "best_generator.pt", dev)
    torch.manual_seed(7919 * 0 + 3)
    x = sample_generator(
        gen,
        200 + 2 * 16,
        int(cfg["model"]["latent_dim"]),
        int(cfg["training"]["batch_size"]),
        dev,
        normalize=False,
    )
    state = load_preprocess_state(run_dir)
    assert state.mean is not None and np.abs(state.mean).max() > 1e-3
    h = np.asarray(invert_samples(x, state), np.float32)
    expected = h[:200] @ wk[:128].T
    np.testing.assert_allclose(seen[0], expected, rtol=1e-4, atol=1e-4)


def test_synthetic_h_splits_are_distinct_rows(tmp_path):
    """Catches h_id aliasing h_query (or h_keys): the three splits must be
    disjoint slices of one sample."""
    run_dir, _, _ = _tiny_run(tmp_path)
    hk, hq, hi, _ = kv_gate.synthetic_h(run_dir, 3, seed=0, n_keys=200, n_tail=16)
    assert (len(hk), len(hq), len(hi)) == (200, 16, 16)
    assert not np.allclose(hq.cpu().numpy(), hi.cpu().numpy())
    assert not np.allclose(hk[:16].cpu().numpy(), hq.cpu().numpy())


def test_measure_run_layer_ema_changes_the_keys(tmp_path, monkeypatch):
    """Catches the ema flag being ignored: on a live checkpoint, EMA weights
    give different keys, and the record says which weights were measured."""
    run_dir, cap, _ = _tiny_run(tmp_path)
    live = "checkpoint_step_4.pt"
    seen = _spy_keys(monkeypatch)
    r0 = kv_gate.measure_run_layer(run_dir, cap, 3, checkpoint_name=live, **_TINY)
    r1 = kv_gate.measure_run_layer(
        run_dir, cap, 3, checkpoint_name=live, ema=True, **_TINY
    )
    assert not np.allclose(seen[0], seen[2])
    assert {r["generator_weights"] for r in r0} == {"live"}
    assert {r["generator_weights"] for r in r1} == {"ema"}
    assert all(r["checkpoint"] == live for r in r1)


def test_measure_run_layer_ema_on_an_ema_checkpoint_is_a_no_op(tmp_path, monkeypatch):
    """best_generator.pt already holds EMA weights; ema=True must not apply
    the average a second time."""
    run_dir, cap, _ = _tiny_run(tmp_path)
    seen = _spy_keys(monkeypatch)
    kv_gate.measure_run_layer(run_dir, cap, 3, **_TINY)
    kv_gate.measure_run_layer(run_dir, cap, 3, ema=True, **_TINY)
    np.testing.assert_array_equal(seen[0], seen[2])


def test_measure_run_layer_ema_without_ema_params_raises(tmp_path):
    import torch

    run_dir, cap, _ = _tiny_run(tmp_path)
    p = run_dir / "checkpoint_step_4.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    ck["ema_params"] = {}
    torch.save(ck, p)
    with pytest.raises(ValueError, match="ema_params"):
        kv_gate.measure_run_layer(
            run_dir, cap, 3, checkpoint_name=p.name, ema=True, **_TINY
        )


def _spy_profiles(monkeypatch):
    """Record (keys, q_ood, q_id, seed) for every profile_corpus call."""
    seen = []
    real = kv_gate.profile_corpus

    def spy(keys, q_ood, q_id, **k):
        seen.append((np.array(keys), np.array(q_ood), np.array(q_id), k["seed"]))
        return real(keys, q_ood, q_id, **k)

    monkeypatch.setattr(kv_gate, "profile_corpus", spy)
    return seen


def _fixed_h(n_keys=200, n_tail=16):
    import torch

    g = torch.Generator().manual_seed(123)
    return (
        torch.randn(n_keys, 16, generator=g),
        torch.randn(n_tail, 16, generator=g),
        torch.randn(n_tail, 16, generator=g),
    )


def test_measure_h_layer_profiles_each_head_of_the_supplied_h(tmp_path, monkeypatch):
    """Catches a changed seed formula or head order, keys/queries built from the
    wrong tensor (h_query and h_id swapped), source or extra ignored, and extra
    merged before the fixed fields."""
    from src.kv.transforms import head_keys, head_queries, load_layer_weights

    _, cap, _ = _tiny_run(tmp_path)
    h_keys, h_query, h_id = _fixed_h()
    seen = _spy_profiles(monkeypatch)
    recs = kv_gate.measure_h_layer(
        h_keys,
        h_query,
        h_id,
        cap,
        3,
        source="pooled_real",
        seed=2,
        nlist=4,
        hnsw_m=8,
        ef_construction=16,
        extra={"control": "x", "books": [12, 13]},
    )
    w = load_layer_weights(cap, 3)
    wk, wq = w["k_proj"].float(), w["q_proj"].float()
    assert len(seen) == 2
    assert [s[3] for s in seen] == [1009 * 3 + g + 7919 * 2 for g in (0, 1)]
    for g, (keys, q_ood, q_id, _) in enumerate(seen):
        np.testing.assert_allclose(keys, head_keys(h_keys, wk, g).numpy(), rtol=1e-6)
        np.testing.assert_allclose(
            q_ood, head_queries(h_query, wq, g, 5).numpy(), rtol=1e-6
        )
        np.testing.assert_allclose(q_id, head_keys(h_id, wk, g).numpy(), rtol=1e-6)
    assert [r["kv_head"] for r in recs] == [0, 1]
    for r in recs:
        assert r["source"] == "pooled_real" and r["control"] == "x"
        assert r["books"] == [12, 13] and r["n_tail"] == len(h_query)


def test_measure_run_layer_records_keep_the_committed_fields(tmp_path):
    """Catches a field dropped, renamed or reordered by the refactor, which
    would make new v0 measurements incomparable with the committed ones."""
    from pathlib import Path

    run_dir, cap, _ = _tiny_run(tmp_path)
    first = Path(__file__).resolve().parents[1] / "docs/results/scout-kv/v0/best.jsonl"
    committed = json.loads(first.read_text(encoding="utf-8").splitlines()[0])
    recs = kv_gate.measure_run_layer(run_dir, cap, 3, **_TINY)
    assert set(recs[0]) == set(committed)
    assert list(recs[0]) == list(committed)


def test_measure_h_layer_refuses_mismatched_query_counts(tmp_path):
    _, cap, _ = _tiny_run(tmp_path)
    h_keys, h_query, h_id = _fixed_h()
    with pytest.raises(ValueError, match="h_id"):
        kv_gate.measure_h_layer(
            h_keys,
            h_query,
            h_id[:-1],
            cap,
            3,
            source="pooled_real",
            seed=0,
            nlist=4,
            hnsw_m=8,
            ef_construction=16,
        )


# --- measure-run --layers and compare-layers --------------------------------


def _out_layers(out):
    return [json.loads(line)["layer"] for line in out.read_text().splitlines() if line]


def test_measure_run_layers_measures_only_the_requested_layers(tmp_path):
    """Catches `layers` being ignored, so every supplied run dir is measured."""
    run_dir, cap, _ = _tiny_run(tmp_path)
    out = tmp_path / "o.jsonl"
    kv_gate.measure_run(
        cap, {3: run_dir, 7: run_dir}, 0, out, layers=[3],
        n_keys=200, n_tail=16, nlist=4, hnsw_m=8, ef_construction=16,
    )  # fmt: skip
    recs = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert [(r["layer"], r["kv_head"]) for r in recs] == [(3, 0), (3, 1)]


def test_measure_run_layers_missing_run_dir_raises_before_writing(tmp_path):
    """Catches a missing run dir silently skipped, and validation that runs after
    layer 3 has already been written."""
    run_dir, cap, _ = _tiny_run(tmp_path)
    out = tmp_path / "o.jsonl"
    with pytest.raises(ValueError, match="7"):
        kv_gate.measure_run(cap, {3: run_dir}, 0, out, layers=[3, 7])
    assert not out.exists()


def test_measure_run_layers_rejects_a_non_global_layer(tmp_path):
    """Catches a non-global layer being accepted. A run dir is supplied for 4 so
    the missing-run-dir check cannot raise instead."""
    run_dir, cap, _ = _tiny_run(tmp_path)
    out = tmp_path / "o.jsonl"
    with pytest.raises(ValueError, match="4.*not a global layer"):
        kv_gate.measure_run(cap, {3: run_dir, 4: run_dir}, 0, out, layers=[4])
    assert not out.exists()


def test_measure_run_layers_rejects_duplicates_and_empty(tmp_path):
    """Catches a repeated layer measured twice (duplicate records) and an empty
    list that writes nothing and exits cleanly."""
    run_dir, cap, _ = _tiny_run(tmp_path)
    out = tmp_path / "o.jsonl"
    with pytest.raises(ValueError, match="more than once"):
        kv_gate.measure_run(cap, {3: run_dir}, 0, out, layers=[3, 3])
    with pytest.raises(ValueError, match="empty"):
        kv_gate.measure_run(cap, {3: run_dir}, 0, out, layers=[])
    assert not out.exists()


def test_measure_run_without_layers_still_needs_every_global_layer(tmp_path):
    """Catches the default changing to 'whatever was supplied'."""
    run_dir, cap, _ = _tiny_run(tmp_path)
    with pytest.raises(ValueError, match="every global layer"):
        kv_gate.measure_run(cap, {3: run_dir}, 0, tmp_path / "o.jsonl")


def test_check_refuses_the_committed_three_layer_subset():
    """Catches a subset passing `check`."""
    recs = [r for r in _committed("v0/best") if r["layer"] in (11, 19, 43)]
    assert len(recs) == 24
    rep = kv_gate.evaluate(recs, kv_gate.load_kv_gate(GATE))
    assert rep["verdict"] == "fail"
    assert any("need 96 records" in r and "(3, 0)" in r for r in rep["reasons"])


def _refs():
    return [_committed(f"reference/seed{s}") for s in range(5)]


def test_compare_layers_on_committed_v0_statuses():
    """Catches swapped above/below and pooling over all layers instead of per layer."""
    res = kv_gate.compare_layers(_committed("v0/best"), _refs())
    assert res["n_reference_sets"] == 5
    assert sorted(res["layers"]) == list(kv_gate.GLOBAL_LAYERS)
    assert all(v["n_records"] == 8 for v in res["layers"].values())
    want = {
        (11, "hnsw_reach_k10"): "above",
        (11, "ivf_ratio_k128"): "below",
        (19, "hnsw_recall_k10_ef176_id"): "below",
        (43, "ivf_ratio_k128"): "below",
        (43, "hnsw_recall_k10_ef176"): "above",
        (11, "ivf_recall_k128_np24"): "above",
        (19, "ivf_recall_k128_np24"): "below",
    }
    for (L, stat), status in want.items():
        assert res["layers"][L]["statistics"][stat]["status"] == status, (L, stat)
    e = res["layers"][11]["statistics"]["hnsw_reach_k10"]
    assert set(e) >= {"min", "max", "mean", "value", "status"}


def test_compare_layers_bounds_are_inclusive_and_none_is_none():
    """Catches exclusive bounds on either side, and a None compared as a number."""
    refs = [_full(rec10=0.8), _full(rec10=0.9)]
    for v in (0.9, 0.8):
        res = kv_gate.compare_layers(_full(rec10=v), refs)
        e = res["layers"][3]["statistics"]["hnsw_recall_k10_ef176"]
        assert e["status"] == "inside", v
    res = kv_gate.compare_layers(_full(rec10=0.95), refs)
    assert res["layers"][3]["statistics"]["hnsw_recall_k10_ef176"]["status"] == "above"
    res = kv_gate.compare_layers(_full(rec10=0.75), refs)
    assert res["layers"][3]["statistics"]["hnsw_recall_k10_ef176"]["status"] == "below"
    res = kv_gate.compare_layers(_full(ivf_o=None, ivf_i=None), refs)
    e = res["layers"][3]["statistics"]["ivf_ratio_k128"]
    assert e["value"] is None and e["status"] == "none"


def test_cli_compare_layers_resolves_default_references_from_the_repo_root(tmp_path):
    """Catches default references resolved against the working directory."""
    import os

    env = dict(os.environ, PYTHONPATH=str(GATE.parents[1]))
    p = subprocess.run(
        [
            sys.executable, "-m", "src.eval.kv_gate", "compare-layers",
            "--records", str(RESULTS / "v0/best.jsonl"),
        ],  # fmt: skip
        capture_output=True, text=True, cwd=tmp_path, env=env,
    )  # fmt: skip
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert out["n_reference_sets"] == 5
    assert sorted(out["layers"], key=int) == [str(L) for L in kv_gate.GLOBAL_LAYERS]


def test_compare_layers_reports_condition_mismatches():
    """Catches condition mismatches being silently ignored by compare-layers."""
    canonical = kv_gate.load_kv_gate(kv_gate.GATES_FILE)["canonical"]
    refs = [_full()]
    ok = kv_gate.compare_layers(_full(), refs, canonical)
    assert ok["condition_mismatches"] == []
    R = _full()
    R[4]["n_keys"] = 100
    bad = kv_gate.compare_layers(R, refs, canonical)
    assert bad["condition_mismatches"]
    assert any("n_keys" in m for m in bad["condition_mismatches"])


def test_compare_layers_counts_reference_sets_per_layer():
    """Catches the per-layer count being copied from the global count."""
    full = _full()
    partial = [r for r in _full() if r["layer"] != 3]
    rep = kv_gate.compare_layers(_full(), [full, partial])
    assert rep["n_reference_sets"] == 2
    assert rep["layers"][3]["n_reference_sets"] == 1
    assert all(v["n_reference_sets"] == 2 for k, v in rep["layers"].items() if k != 3)


def _tiny_flagged_run(tmp_path):
    """A trained flagged run plus layer-3 weights shaped like _tiny_run's."""
    from tests.test_train_flagged import flagged_config

    from src.train.train_wgan_gp import train

    ckpt, _ = train(flagged_config(tmp_path))
    run_dir = ckpt.parent
    cap = tmp_path / "wcapture"
    (cap / "weights").mkdir(parents=True)
    rng = np.random.default_rng(0)
    _write_multi(
        cap / "weights" / "layer03.safetensors",
        {
            "k_proj": rng.normal(size=(2 * 128, 16)).astype(np.float32),
            "q_proj": rng.normal(size=(2 * 5 * 128, 16)).astype(np.float32),
        },
    )
    return run_dir, cap


def _inline_draw(run_dir, layer, seed, parts):
    """Seed, then draw each (flag, rows) part in order with gen(z, flags), written
    out here so a bug in sample_generator's flag handling cannot hide. Returns the
    inverted rows of each part, on the gate's device."""
    import torch
    import yaml

    from src.eval.compare_variants import invert_samples, load_preprocess_state
    from src.eval.evaluate_distribution import load_generator

    cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text())
    dev = kv_gate._device()
    gen = load_generator(cfg, run_dir / "best_generator.pt", dev)
    latent = int(cfg["model"]["latent_dim"])
    bs = int(cfg["training"]["batch_size"])
    state = load_preprocess_state(run_dir)
    torch.manual_seed(7919 * seed + layer)
    out = []
    with torch.no_grad():
        for flag, n in parts:
            xs = []
            for start in range(0, n, bs):
                cur = min(bs, n - start)
                z = torch.randn(cur, latent, device=dev)
                flags = torch.full((cur,), flag, dtype=torch.long, device=dev)
                xs.append(gen(z, flags).cpu().numpy().astype(np.float32))
            out.append(
                np.asarray(invert_samples(np.concatenate(xs), state), np.float32)
            )
    return out


def test_flagged_run_keys_come_from_body_and_queries_from_tail(tmp_path):
    """Catches the sampler drawing keys and queries under the same flag (spec: the test
    that matters most). Recomputes the draw with an inline loop. Checking keys alone
    would not catch queries sampled with flag=BODY, so the query and id rows are
    compared too, against both the tail draw and a body draw."""
    from src.models.conditional import BODY, TAIL

    run_dir, _ = _tiny_flagged_run(tmp_path)
    n_keys, n_tail = 40, 8
    hk, hq, hi, _ = kv_gate.synthetic_h(
        run_dir, 3, seed=0, n_keys=n_keys, n_tail=n_tail
    )
    hk, hq, hi = (t.cpu().numpy() for t in (hk, hq, hi))
    keys, tail = _inline_draw(run_dir, 3, 0, [(BODY, n_keys), (TAIL, 2 * n_tail)])
    np.testing.assert_allclose(hk, keys, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(hq, tail[:n_tail], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(hi, tail[n_tail:], rtol=1e-5, atol=1e-5)
    (tail_keys,) = _inline_draw(run_dir, 3, 0, [(TAIL, n_keys)])
    assert not np.allclose(hk, tail_keys, atol=1e-4)
    _, body_tail = _inline_draw(run_dir, 3, 0, [(BODY, n_keys), (BODY, 2 * n_tail)])
    assert not np.allclose(hq, body_tail[:n_tail], atol=1e-4)
    assert not np.allclose(hi, body_tail[n_tail:], atol=1e-4)


def test_unflagged_run_sampling_is_unchanged(tmp_path):
    """Catches the v0 path changing (an extra RNG draw, a reordered split, a flag
    argument leaking in): synthetic_h on _tiny_run equals the pre-v1 formula written
    out inline. It does not call sample_generator, so a change to that function
    cannot move the expected value with it. The committed v0 best.jsonl/final.jsonl
    tests read files and never re-sample, so they do not guard this."""
    import torch
    import yaml

    from src.eval.compare_variants import invert_samples, load_preprocess_state
    from src.eval.evaluate_distribution import load_generator

    run_dir, _, _ = _tiny_run(tmp_path)
    n_keys, n_tail = 200, 16
    hk, hq, hi, _ = kv_gate.synthetic_h(
        run_dir, 3, seed=0, n_keys=n_keys, n_tail=n_tail
    )
    cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text())
    assert cfg["model"].get("n_flags") is None
    dev = kv_gate._device()
    gen = load_generator(cfg, run_dir / "best_generator.pt", dev)
    latent = int(cfg["model"]["latent_dim"])
    bs = int(cfg["training"]["batch_size"])
    total = n_keys + 2 * n_tail
    torch.manual_seed(7919 * 0 + 3)
    xs = []
    with torch.no_grad():
        for start in range(0, total, bs):
            cur = min(bs, total - start)
            z = torch.randn(cur, latent, device=dev)
            xs.append(gen(z).cpu().numpy().astype(np.float32))
    h = np.asarray(
        invert_samples(np.concatenate(xs), load_preprocess_state(run_dir)), np.float32
    )
    np.testing.assert_allclose(hk.cpu().numpy(), h[:n_keys], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        hq.cpu().numpy(), h[n_keys : n_keys + n_tail], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        hi.cpu().numpy(), h[n_keys + n_tail :], rtol=1e-5, atol=1e-5
    )


def test_generate_refuses_a_flagged_run(tmp_path, monkeypatch):
    """Catches src.sample.generate emitting unconditioned samples from a v1 checkpoint:
    the flagged generator needs flags, so main() must raise instead of writing a file."""
    from src.sample import generate

    run_dir, _ = _tiny_flagged_run(tmp_path)
    out = tmp_path / "samples.npy"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--checkpoint", str(run_dir / "best_generator.pt"),
            "--config", str(run_dir / "run_config.yaml"),
            "--num-samples", "8",
            "--output-path", str(out),
        ],
    )  # fmt: skip
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'flags'"
    ):
        generate.main()
    assert not out.exists()
