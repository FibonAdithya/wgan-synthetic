import json
import subprocess
import sys

import numpy as np
import pytest

from src.eval import kv_gate
from src.eval.check_gate import EXIT_UNSET

GATE = kv_gate.GATES_FILE  # gates/scout_kv.yaml


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


def test_shipped_gate_is_unset():
    """Catches a band being set in the shipped file (a human decision)."""
    gate = kv_gate.load_kv_gate(GATE)
    assert all(v is None for v in gate["statistics"].values())
    assert kv_gate.evaluate(_full(), gate)["verdict"] == "unset"


def _gate_with(**bands):
    gate = kv_gate.load_kv_gate(GATE)
    for k, v in bands.items():
        gate["statistics"][k] = v
    return gate


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


def test_cli_check_exit_codes(tmp_path):
    """Catches the CLI exiting 0 on an unset gate."""
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _full()))
    out = subprocess.run(
        [sys.executable, "-m", "src.eval.kv_gate", "check", "--records", str(p)],
        capture_output=True,
        text=True,
    )
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
    gen = load_generator(cfg, run_dir / "best_generator.pt", torch.device("cpu"))
    torch.manual_seed(7919 * 0 + 3)
    x = sample_generator(
        gen,
        200 + 2 * 16,
        int(cfg["model"]["latent_dim"]),
        int(cfg["training"]["batch_size"]),
        torch.device("cpu"),
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
