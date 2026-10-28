"""Judge scout_kv v2 (book-latent) on book clustering and vector spread.

The rules are in docs/ai/specs/2026-10-28-scout-kv-v2-book-latent-design.md,
"How v2 is judged" and "Pilot, then the full run":

- port check: at L19 and L27 the v2 code must reproduce the throwaway probe
  (same-book share within 0.1 of the probe's two seeds, body LID within 1.5).
- criterion 1: |same-book share - real| <= 0.15, per layer and seed.
- criterion 2: body LID >= v1's final-checkpoint body LID - 6, per layer and seed.
- aim: criterion 1 at >= 10 of the 12 layers (both seeds each) and criterion 2
  at all 12 (both seeds). Only judged when all 12 layers are given; a pilot
  gets per-layer results and no aim.

Comparisons carry EPS so a value exactly on a threshold passes
(0.65 - 0.5 is 0.15000000000000002).

    python -m src.kv.book_judge port --runs-dir D --out port.json
    python -m src.kv.book_judge verdict --runs-dir D --v1-seed1 F [--v1-seed1 F ...] \\
        [--layers 3,19,27] --out verdict.json

`--runs-dir` holds `L{LL}_s{seed}/locality.json` and `L{LL}_s{seed}/intrinsic.jsonl`
(LL zero-padded). Pure Python on committed JSON/JSONL.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBE_RUNS = REPO_ROOT / "docs/results/scout-kv/v2-probe/runs"
DEFAULT_REAL_LOCALITY = REPO_ROOT / "docs/results/scout-kv/locality/locality.json"
DEFAULT_V1_SEED42 = REPO_ROOT / "docs/results/scout-kv/v1/full/intrinsic.jsonl"

SAME_BOOK_TOL = 0.15
LID_DROP_MAX = 6.0
PORT_SAME_BOOK_TOL = 0.1
PORT_LID_TOL = 1.5
MIN_C1_LAYERS = 10
EPS = 1e-9
SEEDS = (42, 1)
V1_SEED1_LAYERS = (19, 27, 35, 43)
ALL_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
PORT_LAYERS = (19, 27)
# Probe run directory per port layer (L19 has a 25% tail run, L27 only the 3% one).
PROBE_RUN_NAMES = {19: "Bflag25", 27: "Bflag"}


def same_book_share(locality_path: Path, layer: int) -> float:
    """`h_l2.same_book` of the entry for `layer`; KeyError if the layer is absent."""
    data = json.loads(Path(locality_path).read_text())
    for entry in data["layers"]:
        if entry["layer"] == layer:
            return float(entry["h_l2"]["same_book"])
    raise KeyError(f"layer {layer} not in {locality_path}")


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def body_lid(intrinsic_path: Path, layer: int, source: str | None = None) -> float:
    """`lid_median` of the one body row at `layer` (and `source`, when given)."""
    hits = [
        r
        for r in _rows(intrinsic_path)
        if r["layer"] == layer
        and r["flag"] == "body"
        and (source is None or r["source"] == source)
    ]
    if len(hits) != 1:
        raise ValueError(
            f"{intrinsic_path}: expected one body row for layer {layer}"
            f" source {source!r}, found {len(hits)}"
        )
    return float(hits[0]["lid_median"])


def _has_body(path: Path, layer: int, source: str) -> bool:
    return any(
        r["layer"] == layer and r["flag"] == "body" and r["source"] == source
        for r in _rows(path)
    )


def v1_final_lid(intrinsic_path: Path, layer: int) -> float:
    """v1's final-checkpoint body LID: the `v1_final` row, else `v1_best`.

    v1's intrinsic.py writes `v1_final` only when the best checkpoint is not step 30,000.
    """
    if _has_body(intrinsic_path, layer, "v1_final"):
        return body_lid(intrinsic_path, layer, "v1_final")
    return body_lid(intrinsic_path, layer, "v1_best")


def port_ranges(probe_runs: Path) -> dict[int, dict[str, tuple[float, float]]]:
    """Port-check windows from the committed probe runs, over the two seeds."""
    out: dict[int, dict[str, tuple[float, float]]] = {}
    for layer, name in PROBE_RUN_NAMES.items():
        sb, lid = [], []
        for seed in SEEDS:
            d = Path(probe_runs) / f"L{layer}" / f"{name}_s{seed}"
            sb.append(same_book_share(d / "locality.json", layer))
            lid.append(body_lid(d / "intrinsic.jsonl", layer))
        out[layer] = {
            "same_book": (min(sb) - PORT_SAME_BOOK_TOL, max(sb) + PORT_SAME_BOOK_TOL),
            "body_lid": (min(lid) - PORT_LID_TOL, max(lid) + PORT_LID_TOL),
        }
    return out


def _inside(x: float, window: tuple[float, float]) -> bool:
    return window[0] - EPS <= x <= window[1] + EPS


def port_check(v2: dict[tuple[int, int], dict[str, float]], ranges) -> dict:
    """Does v2 at L19 and L27, both seeds, land inside the probe windows?"""
    runs = {}
    for layer in PORT_LAYERS:
        for seed in SEEDS:
            if (layer, seed) not in v2:
                raise ValueError(f"port check needs layer {layer} seed {seed}")
            m = v2[(layer, seed)]
            runs[(layer, seed)] = {
                "same_book_ok": _inside(m["same_book"], ranges[layer]["same_book"]),
                "lid_ok": _inside(m["body_lid"], ranges[layer]["body_lid"]),
            }
    passed = all(r["same_book_ok"] and r["lid_ok"] for r in runs.values())
    return {"passed": passed, "runs": runs}


def verdict(
    v2: dict[tuple[int, int], dict[str, float]],
    real_same_book: dict[int, float],
    v1_lid: dict[tuple[int, int], float],
    layers: Sequence[int],
) -> dict:
    """Criteria 1 and 2 per (layer, seed) and per layer; `aim` only for all 12 layers."""
    runs = {}
    for layer in layers:
        for seed in SEEDS:
            if (layer, seed) not in v2:
                raise ValueError(f"v2 run missing for layer {layer} seed {seed}")
            m = v2[(layer, seed)]
            real = real_same_book[layer]
            v1 = v1_lid[(layer, seed)]
            runs[(layer, seed)] = {
                "same_book": m["same_book"],
                "real_same_book": real,
                "c1": abs(m["same_book"] - real) <= SAME_BOOK_TOL + EPS,
                "body_lid": m["body_lid"],
                "v1_lid": v1,
                "c2": m["body_lid"] >= v1 - LID_DROP_MAX - EPS,
            }
    per_layer = {
        layer: {
            "c1_both": all(runs[(layer, s)]["c1"] for s in SEEDS),
            "c2_both": all(runs[(layer, s)]["c2"] for s in SEEDS),
        }
        for layer in layers
    }
    aim = None
    if set(layers) == set(ALL_LAYERS):
        n_c1 = sum(1 for p in per_layer.values() if p["c1_both"])
        aim = n_c1 >= MIN_C1_LAYERS and all(p["c2_both"] for p in per_layer.values())
    return {"aim": aim, "layers": per_layer, "runs": runs}


def v1_lid_table(
    seed42_path: Path, seed1_paths: Sequence[Path], layers: Sequence[int]
) -> dict[tuple[int, int], float]:
    """v1 final LID per (layer, seed): seed 1 from its own file at V1_SEED1_LAYERS, else seed 42."""
    table: dict[tuple[int, int], float] = {}
    for layer in layers:
        table[(layer, 42)] = v1_final_lid(seed42_path, layer)
        if layer in V1_SEED1_LAYERS:
            holders = [
                p
                for p in seed1_paths
                if _has_body(p, layer, "v1_final") or _has_body(p, layer, "v1_best")
            ]
            if not holders:
                raise ValueError(
                    f"v1 seed 1 layer {layer} is in none of the given files"
                )
            table[(layer, 1)] = v1_final_lid(holders[0], layer)
        else:
            table[(layer, 1)] = table[(layer, 42)]
    return table


def load_v2(
    runs_dir: Path, layers: Sequence[int]
) -> dict[tuple[int, int], dict[str, float]]:
    """Read `L{LL}_s{seed}/{locality.json,intrinsic.jsonl}` for each layer and seed."""
    out = {}
    for layer in layers:
        for seed in SEEDS:
            d = Path(runs_dir) / f"L{layer:02d}_s{seed}"
            if not d.is_dir():
                raise ValueError(f"missing v2 run directory {d}")
            out[(layer, seed)] = {
                "same_book": same_book_share(d / "locality.json", layer),
                "body_lid": body_lid(d / "intrinsic.jsonl", layer, f"v2_s{seed}"),
            }
    return out


def _jsonable(result: dict) -> dict:
    out = dict(result)
    out["runs"] = {f"L{L:02d}_s{s}": v for (L, s), v in result["runs"].items()}
    if "layers" in result:
        out["layers"] = {str(L): v for L, v in result["layers"].items()}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("port", help="L19/L27 port check against the probe")
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--probe-runs", type=Path, default=DEFAULT_PROBE_RUNS)
    p.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verdict", help="criteria 1 and 2 and the aim")
    v.add_argument("--runs-dir", type=Path, required=True)
    v.add_argument("--v1-seed1", type=Path, action="append", default=[])
    v.add_argument("--v1-seed42", type=Path, default=DEFAULT_V1_SEED42)
    v.add_argument("--real-locality", type=Path, default=DEFAULT_REAL_LOCALITY)
    v.add_argument("--layers", default=",".join(str(x) for x in ALL_LAYERS))
    v.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    if a.cmd == "port":
        res = port_check(load_v2(a.runs_dir, PORT_LAYERS), port_ranges(a.probe_runs))
        res = _jsonable(res)
        res["ranges"] = {str(k): v for k, v in port_ranges(a.probe_runs).items()}
    else:
        layers = tuple(int(x) for x in a.layers.split(","))
        real = {L: same_book_share(a.real_locality, L) for L in layers}
        v1 = v1_lid_table(a.v1_seed42, a.v1_seed1, layers)
        res = _jsonable(verdict(load_v2(a.runs_dir, layers), real, v1, layers))
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: res[k] for k in ("passed", "aim") if k in res}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
