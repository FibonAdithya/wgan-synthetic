"""scout_kv gate: IVF/HNSW search difficulty pooled over all 96 corpora.

Unlike the other families, scout_kv is not gated on the four intrinsic
statistics. It is gated on how hard the 96 (global layer, KV head) key corpora
are to search with IVFFlat and an HNSW graph on the Xbox reduction, using the
queries the model would really issue. See docs/datasets/scout_kv.md.

    python -m src.eval.kv_gate measure-real --capture-dir D --books 12-19 --out real.jsonl
    python -m src.eval.kv_gate measure-run  --capture-dir D --run-dir 3=runs/a ... --out run.jsonl
    python -m src.eval.kv_gate check --records run.jsonl

`check` prints the JSON report on stdout and exits 0 (pass), 1 (fail) or
2 (unset: at least one band is null).
"""

from __future__ import annotations

import argparse
import json
import statistics as pystats
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.kv_capture import GLOBAL_LAYERS, h_path, read_safetensor
from src.eval import check_gate
from src.eval.check_gate import EXIT_FAIL, EXIT_PASS, EXIT_UNSET, GateError
from src.kv.corpus import mixed_real_corpus, real_sizes, synthetic_corpus
from src.kv.search_profile import conditions, profile_corpus
from src.kv.transforms import HEAD_DIM, load_layer_weights

STATISTICS = (
    "hnsw_reach_k10",
    "hnsw_recall_k10_ef176",
    "ivf_ratio_k128",
    "ivf_recall_k128_np24",
    "hnsw_recall_k10_ef176_id",
)
CANONICAL_KEYS = (
    "n_corpora",
    "n_keys",
    "n_q_ood",
    "n_q_id",
    "n_sink",
    "n_tail",
    "nlist",
    "hnsw_m",
    "ef_construction",
    "target",
)
N_KV_HEADS = 8
GATES_FILE = check_gate.GATES_DIR / "scout_kv.yaml"
TARGET = 0.9


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pooled_statistics(records: list[dict]) -> dict[str, float | None]:
    ratios = []
    for r in records:
        o = r["plain_ood_k128"]["ivf_scan_frac"]
        i = r["plain_id_k128"]["ivf_scan_frac"]
        # `is not None`, not truthiness: 0.0 is a legitimate scan fraction. A
        # zero denominator has no ratio, so that record does not qualify.
        if o is not None and i is not None and i != 0:
            ratios.append(o / i)
    return {
        "hnsw_reach_k10": float(
            np.mean([bool(r["plain_ood_k10"]["hnsw_reached"]) for r in records])
        ),
        "hnsw_recall_k10_ef176": float(
            pystats.median(r["plain_ood_k10"]["hnsw_recall_op"] for r in records)
        ),
        "ivf_ratio_k128": float(pystats.median(ratios)) if ratios else None,
        "ivf_recall_k128_np24": float(
            pystats.median(r["plain_ood_k128"]["ivf_recall_op"] for r in records)
        ),
        "hnsw_recall_k10_ef176_id": float(
            pystats.median(r["plain_id_k10"]["hnsw_recall_op"] for r in records)
        ),
    }


def load_books(capture_dir, layer, books) -> list[torch.Tensor]:
    """Full-length h per book, position 0 included (the sink is dropped once,
    inside mixed_real_corpus)."""
    return [
        torch.from_numpy(read_safetensor(h_path(Path(capture_dir), b, layer), "h"))
        for b in books
    ]


def load_kv_gate(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise GateError(f"gate file not found: {path}")
    gate = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(gate, dict):
        raise GateError(f"gate file is not a mapping: {path}")
    canonical = gate.get("canonical")
    if not isinstance(canonical, dict) or set(canonical) != set(CANONICAL_KEYS):
        raise GateError(
            f"canonical in {path} must have exactly: {', '.join(CANONICAL_KEYS)}"
        )
    stats = gate.get("statistics")
    if not isinstance(stats, dict) or set(stats) != set(STATISTICS):
        raise GateError(
            f"statistics in {path} must be exactly: {', '.join(STATISTICS)}"
        )
    for name in STATISTICS:
        check_gate.validate_band(name, stats[name], path)
    return gate


def _condition_failures(records: list[dict], canonical: dict) -> list[str]:
    reasons = []
    expected = {(L, g) for L in GLOBAL_LAYERS for g in range(N_KV_HEADS)}
    got = [(r.get("layer"), r.get("kv_head")) for r in records]
    if (
        len(records) != canonical["n_corpora"]
        or len(set(got)) != len(got)
        or set(got) != expected
    ):
        dup = sorted({k for k in got if got.count(k) > 1})
        missing = sorted(expected - set(got))
        reasons.append(
            f"need {canonical['n_corpora']} records with distinct (layer, kv_head) covering "
            f"GLOBAL_LAYERS x 8; got {len(records)}, duplicates {dup}, missing {missing}"
        )
    for r in records:
        who = f"record (layer {r.get('layer')}, kv_head {r.get('kv_head')})"
        for key in ("n_keys", "n_q_ood", "n_q_id", "n_sink", "n_tail"):
            if r.get(key) != canonical[key]:
                reasons.append(
                    f"{who}: {key} is {r.get(key)!r}, canonical {canonical[key]!r}"
                )
        cond = r.get("conditions") or {}
        for key in ("nlist", "hnsw_m", "ef_construction", "target"):
            if cond.get(key) != canonical[key]:
                reasons.append(
                    f"{who}: conditions.{key} is {cond.get(key)!r}, canonical {canonical[key]!r}"
                )
    return reasons


def evaluate(records: list[dict], gate: dict) -> dict:
    reasons = _condition_failures(records, gate["canonical"])
    pooled = pooled_statistics(records) if records else {n: None for n in STATISTICS}
    results = [
        check_gate.check_statistic(n, pooled, gate["statistics"][n]) for n in STATISTICS
    ]
    failures = [s["statistic"] for s in results if s["status"] == "fail"]
    unset = [s["statistic"] for s in results if s["status"] == "unset"]
    if failures:
        reasons.append(f"outside band or unusable: {', '.join(failures)}")
    if reasons:
        verdict = "fail"
    elif unset:
        verdict = "unset"
        reasons.append(f"no band set for: {', '.join(unset)}")
    else:
        verdict = "pass"
    code = {"pass": EXIT_PASS, "fail": EXIT_FAIL, "unset": EXIT_UNSET}[verdict]
    return {
        "verdict": verdict,
        "exit_code": code,
        "dataset": gate.get("dataset"),
        "n_records": len(records),
        "statistics": results,
        "reasons": reasons,
    }


def _own_book_share_top10(c) -> float:
    """Share of each OOD query's exact top-10 keys drawn from its own book."""
    sims = (
        np.ascontiguousarray(c.q_ood, np.float32)
        @ np.ascontiguousarray(c.keys, np.float32).T
    )
    gt = np.argpartition(-sims, 9, axis=1)[:, :10]
    return float((c.key_book[gt] == c.q_book[:, None]).mean())


def _append(out: Path, rec: dict) -> None:
    with Path(out).open("a") as f:
        f.write(json.dumps(rec) + "\n")


def measure_real(
    capture_dir,
    books,
    layers,
    seed,
    out,
    *,
    n_sink=1,
    n_tail=1024,
    nlist=256,
    hnsw_m=48,
    ef_construction=500,
    threads=15,
) -> None:
    import faiss

    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to append duplicate records")
    faiss.omp_set_num_threads(threads)
    dev = _device()
    for layer in layers:
        h_books = [h.to(dev).float() for h in load_books(capture_dir, layer, books)]
        w = load_layer_weights(capture_dir, layer)
        wk, wq = w["k_proj"].to(dev).float(), w["q_proj"].to(dev).float()
        n_keys, n_q, n_id = real_sizes(h_books[0].shape[0], len(books), n_sink, n_tail)
        for g in range(wk.shape[0] // HEAD_DIM):
            t0 = time.time()
            s = 1009 * layer + g + 7919 * seed
            c = mixed_real_corpus(
                h_books, wk, wq, g, n_sink=n_sink, n_tail=n_tail, seed=s
            )
            prof = profile_corpus(
                c.keys,
                c.q_ood,
                c.q_id,
                nlist=nlist,
                hnsw_m=hnsw_m,
                ef_construction=ef_construction,
                seed=s,
                target=TARGET,
            )
            rec = {
                "layer": layer,
                "kv_head": g,
                "source": "real",
                "seed": seed,
                "books": list(books),
                "n_keys": len(c.keys),
                "n_q_ood": len(c.q_ood),
                "n_q_id": len(c.q_id),
                "n_sink": n_sink,
                "n_tail": n_tail,
                "conditions": conditions(nlist, hnsw_m, ef_construction, TARGET),
                "own_book_share_top10": _own_book_share_top10(c),
                **prof,
                "seconds": round(time.time() - t0, 1),
            }
            assert (rec["n_keys"], rec["n_q_ood"], rec["n_q_id"]) == (
                n_keys,
                n_q * (wq.shape[0] // wk.shape[0]),
                n_id,
            )
            _append(out, rec)
            print(layer, g, rec["seconds"], "s", flush=True)


def synthetic_h(
    run_dir,
    layer,
    *,
    seed,
    n_keys,
    n_tail,
    checkpoint_name="best_generator.pt",
    ema=False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Sample h from a trained run, in original coordinates, split into
    (h_keys, h_query, h_id, generator_weights).

    `generator_weights` is "live" or "ema": what was actually measured. ema=True
    on a checkpoint that already holds EMA weights (best_generator.pt) uses it
    as is; on a live one it swaps in the stored EMA shadow with the training
    run's decay and step.
    """
    import contextlib

    from src.eval.compare_variants import (
        _inversion_blocker,
        invert_samples,
        load_preprocess_state,
    )
    from src.eval.evaluate_distribution import load_generator
    from src.train.train_wgan_gp import ema_weights, output_normalize, sample_generator

    run_dir = Path(run_dir)
    blocker = _inversion_blocker(run_dir / "run_config.yaml", run_dir)
    if blocker is not None:
        raise ValueError(blocker)
    cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    dev = _device()
    ckpt_path = run_dir / checkpoint_name
    gen = load_generator(cfg, ckpt_path, dev)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    weights = ckpt.get("generator_weights", "live")
    ctx = contextlib.nullcontext()
    if ema and weights != "ema":
        ema_params = ckpt.get("ema_params") or {}
        if not ema_params:
            raise ValueError(f"{ckpt_path} has no ema_params; cannot apply EMA")
        ctx = ema_weights(
            gen,
            {k: v.to(dev) for k, v in ema_params.items()},
            decay=float(cfg["training"].get("ema_decay", 0.0)),
            ema_step=int(ckpt.get("ema_step", 0)),
        )
        weights = "ema"
    torch.manual_seed(7919 * seed + layer)
    with ctx:
        x = sample_generator(
            gen,
            n_keys + 2 * n_tail,
            int(cfg["model"]["latent_dim"]),
            int(cfg["training"]["batch_size"]),
            dev,
            normalize=output_normalize(cfg["model"]),
        )
    x = invert_samples(x, load_preprocess_state(run_dir))
    h = torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(dev)
    return h[:n_keys], h[n_keys : n_keys + n_tail], h[n_keys + n_tail :], weights


def measure_run_layer(
    run_dir,
    capture_dir,
    layer,
    *,
    seed,
    n_keys,
    n_tail,
    nlist,
    hnsw_m,
    ef_construction,
    checkpoint_name="best_generator.pt",
    ema=False,
) -> list[dict]:
    """Profile one layer of a trained run: one record per KV head."""
    run_dir = Path(run_dir)
    dev = _device()
    h_keys, h_query, h_id, weights = synthetic_h(
        run_dir,
        layer,
        seed=seed,
        n_keys=n_keys,
        n_tail=n_tail,
        checkpoint_name=checkpoint_name,
        ema=ema,
    )
    w = load_layer_weights(capture_dir, layer)
    wk, wq = w["k_proj"].to(dev).float(), w["q_proj"].to(dev).float()
    recs = []
    for g in range(wk.shape[0] // HEAD_DIM):
        t0 = time.time()
        s = 1009 * layer + g + 7919 * seed
        c = synthetic_corpus(h_keys, h_query, h_id, wk, wq, g)
        prof = profile_corpus(
            c.keys,
            c.q_ood,
            c.q_id,
            nlist=nlist,
            hnsw_m=hnsw_m,
            ef_construction=ef_construction,
            seed=s,
            target=TARGET,
        )
        recs.append(
            {
                "layer": layer,
                "kv_head": g,
                "source": "synthetic",
                "seed": seed,
                "books": [],
                "run_dir": str(run_dir),
                "n_keys": len(c.keys),
                "n_q_ood": len(c.q_ood),
                "n_q_id": len(c.q_id),
                # Synthetic corpora have no sink; n_sink records the reference
                # condition they are compared under, not a property of h.
                "n_sink": 1,
                "n_tail": n_tail,
                "checkpoint": checkpoint_name,
                "generator_weights": weights,
                "conditions": conditions(nlist, hnsw_m, ef_construction, TARGET),
                **prof,
                "seconds": round(time.time() - t0, 1),
            }
        )
    return recs


def measure_run(
    capture_dir,
    run_dirs: dict[int, Path],
    seed,
    out,
    *,
    n_keys=31736,
    n_tail=1024,
    nlist=256,
    hnsw_m=48,
    ef_construction=500,
    threads=15,
    checkpoint_name="best_generator.pt",
    ema=False,
) -> None:
    import faiss

    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to append duplicate records")
    missing = [L for L in GLOBAL_LAYERS if L not in run_dirs]
    if missing:
        raise ValueError(
            f"measure-run needs a run dir for every global layer; missing {missing}"
        )
    faiss.omp_set_num_threads(threads)
    for layer in GLOBAL_LAYERS:
        for rec in measure_run_layer(
            run_dirs[layer],
            capture_dir,
            layer,
            seed=seed,
            n_keys=n_keys,
            n_tail=n_tail,
            nlist=nlist,
            hnsw_m=hnsw_m,
            ef_construction=ef_construction,
            checkpoint_name=checkpoint_name,
            ema=ema,
        ):
            _append(out, rec)
            print(layer, rec["kv_head"], rec["seconds"], "s", flush=True)


def parse_books(text: str) -> list[int]:
    if "-" in text and "," not in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(t) for t in text.split(",")]


def _add_index_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-tail", type=int, default=1024)
    p.add_argument("--nlist", type=int, default=256)
    p.add_argument("--hnsw-m", type=int, default=48)
    p.add_argument("--ef-construction", type=int, default=500)
    p.add_argument("--threads", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("measure-real")
    pr.add_argument("--capture-dir", type=Path, required=True)
    pr.add_argument("--books", type=parse_books, default=parse_books("12-19"))
    pr.add_argument(
        "--layers",
        type=lambda t: [int(x) for x in t.split(",")],
        default=list(GLOBAL_LAYERS),
    )
    pr.add_argument("--n-sink", type=int, default=1)
    pr.add_argument("--out", type=Path, required=True)
    _add_index_args(pr)
    pn = sub.add_parser("measure-run")
    pn.add_argument("--capture-dir", type=Path, required=True)
    pn.add_argument("--run-dir", action="append", required=True, metavar="LAYER=PATH")
    pn.add_argument("--checkpoint-name", default="best_generator.pt")
    pn.add_argument(
        "--ema",
        action="store_true",
        help="measure the checkpoint under its stored EMA weights",
    )
    pn.add_argument("--n-keys", type=int, default=31736)
    pn.add_argument("--out", type=Path, required=True)
    _add_index_args(pn)
    pc = sub.add_parser("check")
    pc.add_argument("--records", type=Path, required=True)
    pc.add_argument("--gate-file", type=Path, default=GATES_FILE)
    a = ap.parse_args(argv)

    if a.cmd == "measure-real":
        measure_real(
            a.capture_dir,
            a.books,
            a.layers,
            a.seed,
            a.out,
            n_sink=a.n_sink,
            n_tail=a.n_tail,
            nlist=a.nlist,
            hnsw_m=a.hnsw_m,
            ef_construction=a.ef_construction,
            threads=a.threads,
        )
        return 0
    if a.cmd == "measure-run":
        pairs = dict(p.split("=", 1) for p in a.run_dir)
        measure_run(
            a.capture_dir,
            {int(k): Path(v) for k, v in pairs.items()},
            a.seed,
            a.out,
            n_keys=a.n_keys,
            n_tail=a.n_tail,
            nlist=a.nlist,
            hnsw_m=a.hnsw_m,
            ef_construction=a.ef_construction,
            threads=a.threads,
            checkpoint_name=a.checkpoint_name,
            ema=a.ema,
        )
        return 0
    try:
        gate = load_kv_gate(a.gate_file)
        text = a.records.read_text(encoding="utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (GateError, OSError, json.JSONDecodeError) as exc:
        print(f"kv_gate: {exc}", file=sys.stderr)
        return EXIT_FAIL
    report = evaluate(records, gate)
    print(json.dumps(report, indent=2))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
