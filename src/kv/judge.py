"""Judge scout_kv v1 against v0 and the real reference with the "moves toward real" rule.

The rule is the one in docs/ai/specs/2026-10-22-scout-kv-v1-position-flag-design.md,
"How v1 is judged":

- distance: how far a value lies outside its target interval; 0 inside.
- spread: max minus min of the statistic over the five reference draws, at the
  same level (pooled or per layer).
- a statistic moves toward real if v1 is inside the target, or v1's distance
  is smaller than v0's by more than the spread. A move that closes less than
  twice the spread is labelled borderline; it still counts.

`pilot` judges layers 19 and 43 per layer, with seeds paired (v1 at sampling
seed s against v0 at seed s). At L19 the target interval is widened to include
the training books' value. `full` judges the pooled statistics against the
bands in gates/scout_kv.yaml.

    python -m src.kv.judge pilot --v1 0=v1_s0.jsonl --v1 1=v1_s1.jsonl \\
        --v0 0=v0_s0.jsonl --v0 1=v0_s1.jsonl --out pilot_best.json
    python -m src.kv.judge full --v1 v1.jsonl --v0 0=v0_s0.jsonl --v0 1=v0_s1.jsonl \\
        --out full_best.json

Each call judges one checkpoint kind (all best or all final). Pure Python on
the jsonl records written by `kv_gate measure-real` / `measure-run`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval import check_gate
from src.eval.kv_gate import STATISTICS, load_kv_gate, pooled_statistics

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCES = [
    REPO_ROOT / f"docs/results/scout-kv/reference/seed{s}.jsonl" for s in range(5)
]
DEFAULT_TRAIN_BOOKS = (
    REPO_ROOT / "docs/results/scout-kv/controls/train_books_real.jsonl"
)
DEFAULT_GATE_FILE = REPO_ROOT / "gates/scout_kv.yaml"

PILOT_LAYERS = (19, 43)
PILOT_STATISTICS = ("ivf_ratio_k128", "hnsw_reach_k10", "hnsw_recall_k10_ef176")
KEY_SIDE = "hnsw_recall_k10_ef176_id"
# The statistic v0 already passes; the full run requires it to stay in its band.
STAYS_INSIDE = "ivf_recall_k128_np24"
# The layer whose target interval is widened to the training books' value
# (owner's decision, 2026-10-22).
WIDENED_LAYER = 19


def distance(value: float, lo: float | None, hi: float | None) -> float:
    """0 inside [lo, hi] (inclusive), else the gap to the nearer edge. A None
    bound is open."""
    if lo is not None and value < lo:
        return lo - value
    if hi is not None and value > hi:
        return value - hi
    return 0.0


def moves_toward(
    v1: float, v0: float, lo: float | None, hi: float | None, spread: float
) -> bool:
    d1 = distance(v1, lo, hi)
    return d1 == 0 or d1 < distance(v0, lo, hi) - spread


def is_borderline(
    v1: float, v0: float, lo: float | None, hi: float | None, spread: float
) -> bool:
    """Moves, is not inside, and closes less than twice the spread."""
    d1 = distance(v1, lo, hi)
    d0 = distance(v0, lo, hi)
    return moves_toward(v1, v0, lo, hi, spread) and d1 > 0 and d0 - d1 < 2 * spread


def read_records(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _layer_statistics(records: list[dict], layer: int, what: str) -> dict:
    recs = [r for r in records if r["layer"] == layer]
    if not recs:
        raise ValueError(f"{what} has no records at layer {layer}")
    return pooled_statistics(recs)


def _required(value: float | None, name: str, layer: int | str, what: str) -> float:
    """A None in a reference draw, v0 or the training books means the inputs are
    broken; refuse to judge rather than skip the statistic."""
    if value is None:
        raise ValueError(f"{what}: {name} is None at layer {layer}")
    return value


def train_books_targets(source: Path | list[dict], layer: int) -> dict[str, float]:
    """Pooled statistics of the training books' real records at one layer.
    `source` is the jsonl path or its records already read; pilot_verdict
    passes records, the CLI and tests pass the path."""
    if isinstance(source, list):
        records, what = source, "train books"
    else:
        records, what = read_records(source), f"train books {source}"
    stats = _layer_statistics(records, layer, what)
    return {n: _required(stats[n], n, layer, what) for n in STATISTICS}


def _check_seed_labels(by_seed: dict[int, list[dict]], what: str) -> None:
    """Every record's own `seed` field must match the seed it was passed under,
    or the pairing would compare the wrong draws."""
    for s, recs in by_seed.items():
        found = sorted({r["seed"] for r in recs if "seed" in r})
        if found and found != [s]:
            raise ValueError(f"{what} passed as seed {s} holds records of seed {found}")


def _check_one_checkpoint(records: list[dict], what: str) -> None:
    """One checkpoint file and one weight set (EMA or raw) across all records,
    so best and final, or --ema and plain sampling, are never judged together."""
    kinds = sorted(
        {(r.get("checkpoint"), r.get("generator_weights")) for r in records},
        key=repr,
    )
    if len(kinds) > 1:
        raise ValueError(
            f"{what} mixes checkpoint kinds (checkpoint, generator_weights) "
            f"{kinds}; judge one kind at a time"
        )


def _corpora(records: list[dict], layers=None) -> set[tuple[int, int]]:
    return {
        (r["layer"], r["kv_head"])
        for r in records
        if layers is None or r["layer"] in layers
    }


def _check_same_corpora(
    base: list[dict], base_what: str, others: dict[str, list[dict]], layers=None
) -> None:
    """Every input must cover the same (layer, kv_head) corpora as `base`, or
    pooled values and the v0 seed spread would be taken over different sets."""
    want = _corpora(base, layers)
    for what, recs in others.items():
        got = _corpora(recs, layers)
        if got != want:
            raise ValueError(
                f"{what} covers different corpora from {base_what}: "
                f"missing {sorted(want - got)}, extra {sorted(got - want)}"
            )


def _reference_range(ref_stats: list[dict], name: str, layer) -> tuple[float, float]:
    vals = [
        _required(s[name], name, layer, f"reference draw {i}")
        for i, s in enumerate(ref_stats)
    ]
    return min(vals), max(vals)


def _cell(v1, v0, lo, hi, spread) -> dict:
    """One v1-against-v0 comparison with every intermediate value."""
    d0 = distance(v0, lo, hi)
    if v1 is None:
        return {
            "v1": None,
            "v0": v0,
            "v1_distance": None,
            "v0_distance": d0,
            "gap_closed": None,
            "moves": False,
            "borderline": False,
        }
    d1 = distance(v1, lo, hi)
    return {
        "v1": v1,
        "v0": v0,
        "v1_distance": d1,
        "v0_distance": d0,
        "gap_closed": d0 - d1,
        "moves": moves_toward(v1, v0, lo, hi, spread),
        "borderline": is_borderline(v1, v0, lo, hi, spread),
    }


def pilot_verdict(
    v1_by_seed: dict[int, list[dict]],
    v0_by_seed: dict[int, list[dict]],
    references: list[list[dict]],
    train_books: list[dict],
) -> dict:
    """Pilot go/no-go at layers 19 and 43 for one checkpoint kind.

    A statistic moves only if it moves at every seed, v1 at seed s against v0
    at seed s. A statistic is borderline if it moves and is borderline at any
    seed. Key-side recall must not fall below v0's by more than v0's spread
    across the seeds at that layer. `go` is the AND of all of these over both
    layers."""
    seeds = sorted(v1_by_seed)
    # The spec judges the pilot "at both sampling seeds"; with one seed the
    # key-side allowance (v0's seed spread) is 0 by construction.
    if len(seeds) < 2:
        raise ValueError(f"the pilot needs at least two sampling seeds; got {seeds}")
    if set(seeds) != set(v0_by_seed):
        raise ValueError(
            f"v1 seeds {seeds} and v0 seeds {sorted(v0_by_seed)} must be the same"
        )
    if not references:
        raise ValueError("no reference draws")
    _check_seed_labels(v1_by_seed, "v1")
    _check_seed_labels(v0_by_seed, "v0")
    _check_one_checkpoint([r for s in seeds for r in v1_by_seed[s]], "v1")
    _check_one_checkpoint([r for s in seeds for r in v0_by_seed[s]], "v0")
    _check_same_corpora(
        v1_by_seed[seeds[0]],
        f"v1 seed {seeds[0]}",
        {
            **{f"v1 seed {s}": v1_by_seed[s] for s in seeds[1:]},
            **{f"v0 seed {s}": v0_by_seed[s] for s in seeds},
        },
        layers=PILOT_LAYERS,
    )
    train = train_books_targets(train_books, WIDENED_LAYER)

    go = True
    layers = {}
    for layer in PILOT_LAYERS:
        ref_stats = [
            _layer_statistics(ref, layer, f"reference draw {i}")
            for i, ref in enumerate(references)
        ]
        v0_stats = {
            s: _layer_statistics(v0_by_seed[s], layer, f"v0 seed {s}") for s in seeds
        }
        v1_stats = {
            s: _layer_statistics(v1_by_seed[s], layer, f"v1 seed {s}") for s in seeds
        }

        stats = {}
        for name in PILOT_STATISTICS:
            ref_lo, ref_hi = _reference_range(ref_stats, name, layer)
            spread = ref_hi - ref_lo
            lo, hi = ref_lo, ref_hi
            tb = None
            if layer == WIDENED_LAYER:
                tb = train[name]
                lo, hi = min(ref_lo, tb), max(ref_hi, tb)
            cells = {}
            for s in seeds:
                v0 = _required(v0_stats[s][name], name, layer, f"v0 seed {s}")
                cells[s] = _cell(v1_stats[s][name], v0, lo, hi, spread)
            moves = all(c["moves"] for c in cells.values())
            stats[name] = {
                "reference_min": ref_lo,
                "reference_max": ref_hi,
                "spread": spread,
                "train_books": tb,
                "target_min": lo,
                "target_max": hi,
                "seeds": cells,
                "moves": moves,
                "borderline": moves and any(c["borderline"] for c in cells.values()),
            }
            go = go and moves

        v0_key = {
            s: _required(v0_stats[s][KEY_SIDE], KEY_SIDE, layer, f"v0 seed {s}")
            for s in seeds
        }
        allowance = max(v0_key.values()) - min(v0_key.values())
        key_cells = {}
        for s in seeds:
            v1 = v1_stats[s][KEY_SIDE]
            floor = v0_key[s] - allowance
            key_cells[s] = {
                "v1": v1,
                "v0": v0_key[s],
                "floor": floor,
                "not_lower": v1 is not None and v1 >= floor,
            }
        not_lower = all(c["not_lower"] for c in key_cells.values())
        go = go and not_lower
        layers[layer] = {
            "statistics": stats,
            "key_side": {
                "statistic": KEY_SIDE,
                "v0_spread": allowance,
                "seeds": key_cells,
                "not_lower": not_lower,
            },
        }
    return {"kind": "pilot", "seeds": seeds, "layers": layers, "go": go}


def full_verdict(
    v1: list[dict],
    v0_by_seed: dict[int, list[dict]],
    references: list[list[dict]],
    bands: dict,
) -> dict:
    """Pooled verdict over every statistic, target = the gate band, v1 against
    v0 at seed 0. The spread is the pooled reference spread, except key-side
    recall, whose allowance is v0's pooled max - min across `v0_by_seed`.
    `ivf_recall_k128_np24` must stay inside its band; every other statistic
    must move toward it."""
    # The key-side allowance is v0's spread across seeds; one seed makes it 0,
    # the degenerate case the spec rules out.
    if len(v0_by_seed) < 2:
        raise ValueError(
            f"full verdict needs v0 at two or more seeds; got {sorted(v0_by_seed)}"
        )
    if 0 not in v0_by_seed:
        raise ValueError(f"full verdict needs v0 at seed 0; got {sorted(v0_by_seed)}")
    if not references:
        raise ValueError("no reference draws")
    if not v1:
        raise ValueError("v1 has no records")
    _check_seed_labels(v0_by_seed, "v0")
    # v1 is compared with v0 at seed 0, so it must be the seed-0 measurement.
    _check_seed_labels({0: v1}, "v1")
    _check_one_checkpoint(v1, "v1")
    _check_one_checkpoint([r for recs in v0_by_seed.values() for r in recs], "v0")
    _check_same_corpora(
        v1, "v1", {f"v0 seed {s}": recs for s, recs in sorted(v0_by_seed.items())}
    )

    v1_stats = pooled_statistics(v1)
    v0_stats = {s: pooled_statistics(recs) for s, recs in sorted(v0_by_seed.items())}
    ref_stats = [pooled_statistics(ref) for ref in references]

    stats = {}
    for name in STATISTICS:
        band = bands.get(name)
        lo, hi = check_gate.band_bounds(band)
        if lo is None and hi is None:
            raise ValueError(f"band for {name} is unset; nothing to judge against")
        ref_lo, ref_hi = _reference_range(ref_stats, name, "pooled")
        v0_all = {
            s: _required(st[name], name, "pooled", f"v0 seed {s}")
            for s, st in v0_stats.items()
        }
        if name == KEY_SIDE:
            spread = max(v0_all.values()) - min(v0_all.values())
            spread_source = "v0 seeds"
        else:
            spread = ref_hi - ref_lo
            spread_source = "reference"
        cell = _cell(v1_stats[name], v0_all[0], lo, hi, spread)
        band_check = check_gate.check_statistic(name, v1_stats, band)
        band_pass = band_check["status"] == "pass"
        requirement = "inside" if name == STAYS_INSIDE else "toward"
        stats[name] = {
            **cell,
            "target_min": lo,
            "target_max": hi,
            "reference_min": ref_lo,
            "reference_max": ref_hi,
            "spread": spread,
            "spread_source": spread_source,
            "v0_by_seed": v0_all,
            "band_pass": band_pass,
            "requirement": requirement,
            "meets": band_pass if requirement == "inside" else cell["moves"],
        }
    return {
        "kind": "full",
        "v0_seeds": sorted(v0_by_seed),
        "statistics": stats,
        "all_bands_pass": all(c["band_pass"] for c in stats.values()),
        "all_meet": all(c["meets"] for c in stats.values()),
    }


def _seed_path(text: str) -> tuple[int, Path]:
    seed, sep, path = text.partition("=")
    if not sep or not path:
        raise argparse.ArgumentTypeError(f"expected SEED=PATH, got {text!r}")
    try:
        return int(seed), Path(path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"seed is not an integer in {text!r}") from exc


def _seed_map(pairs: list[tuple[int, Path]], flag: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for seed, path in pairs:
        if seed in out:
            raise ValueError(f"{flag} names seed {seed} more than once")
        out[seed] = path
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pilot")
    pp.add_argument("--v1", type=_seed_path, action="append", required=True)
    pp.add_argument("--train-books", type=Path, default=DEFAULT_TRAIN_BOOKS)
    pf = sub.add_parser("full")
    pf.add_argument("--v1", type=Path, required=True)
    pf.add_argument("--gate-file", type=Path, default=DEFAULT_GATE_FILE)
    for p in (pp, pf):
        p.add_argument("--v0", type=_seed_path, action="append", required=True)
        p.add_argument("--references", type=Path, nargs="+", default=DEFAULT_REFERENCES)
        p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    v0_paths = _seed_map(args.v0, "--v0")
    v0 = {s: read_records(p) for s, p in v0_paths.items()}
    references = [read_records(p) for p in args.references]
    inputs = {
        "v0": {str(s): str(p) for s, p in sorted(v0_paths.items())},
        "references": [str(p) for p in args.references],
    }
    if args.cmd == "pilot":
        v1_paths = _seed_map(args.v1, "--v1")
        v1 = {s: read_records(p) for s, p in v1_paths.items()}
        verdict = pilot_verdict(v1, v0, references, read_records(args.train_books))
        inputs["v1"] = {str(s): str(p) for s, p in sorted(v1_paths.items())}
        inputs["train_books"] = str(args.train_books)
        summary = f"go: {verdict['go']}"
    else:
        bands = load_kv_gate(args.gate_file)["statistics"]
        verdict = full_verdict(read_records(args.v1), v0, references, bands)
        inputs["v1"] = str(args.v1)
        inputs["gate_file"] = str(args.gate_file)
        summary = (
            f"all_meet: {verdict['all_meet']}  "
            f"all_bands_pass: {verdict['all_bands_pass']}"
        )
    verdict["inputs"] = inputs
    args.out.write_text(json.dumps(verdict, sort_keys=True, indent=2) + "\n", "utf-8")
    print(f"{summary}  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
