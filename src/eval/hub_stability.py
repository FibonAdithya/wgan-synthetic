"""Which hub statistic can carry a gate band, and at what N.

`docs/datasets/glove.md` names hubness skew as the statistic GloVe is most
likely to fail and the most informative one when it does, and then shows that
at the locked canonical N it measures the draw rather than the corpus: eight
20,000-row draws of the real corpus span 108% of the mean. Issue #29 lists
four fixes and says choosing between them needs a measurement. This is that
measurement.

The rule that decides is pre-registered in
`docs/superpowers/specs/2026-08-13-glove-hub-statistic-stability-design.md`
and is applied here by `evaluate_rule`, so the verdict lands in the committed
artifact rather than in a reader's summary of a table.

Like `src/eval/noise_floor.py`, this module stays importable without plotly
and without the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.eval import ann_difficulty
from src.eval.eda.series import maybe_l2_normalize
from src.eval.noise_floor import summarize_spread


class HubStabilityError(Exception):
    """The sweep could not run as asked -- bad grid, bad corpus, bad series."""


def allocate_draws(
    pool_size: int, n: int, draws: int, seed: int
) -> tuple[list[np.ndarray], bool]:
    """Row indices for `draws` subsamples of `n` rows, and whether they are disjoint.

    Disjoint draws are the ones worth having: they are independent samples of
    the corpus, so their spread is the subsample noise. When `draws * n`
    exceeds the pool they cannot all be disjoint, and the spread across
    overlapping draws is a *lower bound* on the true subsample spread -- the
    draws share rows, so they agree with each other more than independent
    draws would. Callers must record the returned flag: a statistic that looks
    stable only in the overlapping regime has not been shown to be stable.
    """
    if n > pool_size:
        raise HubStabilityError(
            f"a draw of {n} rows does not fit a pool of {pool_size}"
        )
    if draws < 2:
        raise HubStabilityError(f"a spread needs at least two draws, got {draws}")

    rng = np.random.default_rng(seed)
    if draws * n <= pool_size:
        order = rng.permutation(pool_size)
        return [np.sort(order[i * n : (i + 1) * n]) for i in range(draws)], True

    return [
        np.sort(rng.choice(pool_size, size=n, replace=False)) for _ in range(draws)
    ], False


# Report order: the four incumbents first, then the two candidates. The
# incumbents are carried along as a control -- they re-measure the committed
# eight-draw table at more draws and at larger N for the price of the k-NN
# pass that was happening anyway.
STATISTICS = (
    "lid_median",
    "relative_contrast_median",
    "hubness_skew",
    "ivf_gini",
    "hubness_gini",
    "hub_share_top1pct",
)


# Pre-registered in the design spec before any number existed, and not to be
# edited afterwards. Both constants come from precedent in the tree: GloVe's
# three usable statistics sit at 0.32-3.68% range-of-mean against hubness
# skew's 108.2%, and docs/datasets/sift.md already bolds "noise exceeds
# signal" below 1x.
STABLE_MAX_RANGE_PCT = 10.0
MIN_SEPARATION_IN_RANGES = 1.0

# Floating-point representation tolerance for inclusive boundary checks.
# Not part of the pre-registered rule, but necessary to enforce that the
# rule's inclusive bounds are truly inclusive when computed from fixtures.
# Concretely: _spread(1.0, 0.95, 1.05)["range_pct_of_mean"] evaluates to
# 10.000000000000009 (IEEE 754: 1.05 - 0.95 is not exactly 0.1), so a
# literal "<=" would reject the very boundary the rule calls inclusive.
# Applied to the stability comparison only -- the discrimination side needs
# no tolerance because its boundary already satisfies a plain ">=". At 1e-9
# against a bound of 10.0, this cannot move any verdict at the scale of the
# spreads measured in this study (0.32% to 108%).
BOUNDARY_EPSILON = 1e-9


def evaluate_rule(
    real_spread: dict,
    synthetic_mean: float | None,
    *,
    draws_disjoint: bool,
) -> dict:
    """Apply the pre-registered rule to one (statistic, N) cell.

    Two conditions. Stable: the real-side range is at most
    STABLE_MAX_RANGE_PCT of the real-side mean. Discriminating: the synthetic
    mean sits at least MIN_SEPARATION_IN_RANGES real-side ranges away, so a
    band drawn around real would reject that generator. Both bounds are
    inclusive, enforced via BOUNDARY_EPSILON to handle IEEE 754 rounding
    in the stability comparison (where _spread(1.0, 0.95, 1.05) evaluates
    to 10.000000000000009, so a literal "<=" would reject the true boundary).

    Overlapping draws downgrade a pass to "provisional" rather than granting
    it. Their spread is a lower bound on the true subsample spread, so a cell
    that passes only there has not been shown to pass. Since draws are
    disjoint at every N the pool can afford, this is exactly the spec's rule
    that a statistic qualifying only at the largest N is provisional.

    A corpus measured without a synthetic series -- DEEP, here -- can only be
    judged on condition 1, and gets "stable" or "unstable" instead. It is
    evidence about whether an instability generalises, not a vote on GloVe's
    gate.
    """
    range_pct = float(real_spread["range_pct_of_mean"])
    stable = range_pct <= STABLE_MAX_RANGE_PCT + BOUNDARY_EPSILON

    if synthetic_mean is None:
        return {
            "stable": stable,
            "range_pct_of_mean": range_pct,
            "separation_in_ranges": None,
            "discriminating": None,
            "draws_disjoint": draws_disjoint,
            "verdict": "stable" if stable else "unstable",
        }

    real_range = float(real_spread["max"]) - float(real_spread["min"])
    if real_range == 0.0:
        # Every draw returned the same number. That is not a spread, and
        # dividing by it would report infinite separation from a measurement
        # that has not shown it can vary at all.
        separation = None
        discriminating = False
    else:
        separation = abs(float(real_spread["mean"]) - synthetic_mean) / real_range
        discriminating = separation >= MIN_SEPARATION_IN_RANGES

    if not (stable and discriminating):
        verdict = "rejected"
    elif draws_disjoint:
        verdict = "qualified"
    else:
        verdict = "provisional"

    return {
        "stable": stable,
        "range_pct_of_mean": range_pct,
        "separation_in_ranges": separation,
        "discriminating": discriminating,
        "draws_disjoint": draws_disjoint,
        "verdict": verdict,
    }


def measure_draw(
    x: np.ndarray,
    *,
    k: int,
    k_hub: int,
    nlist: int,
    seed: int,
    backend: str,
    chunk_rows: int,
) -> dict[str, float]:
    """Every statistic for one already-drawn subsample, off a single k-NN pass.

    `max_rows=0` is not optional: the caller drew these rows deliberately, and
    letting `compute` subsample again would measure a smaller set than the one
    the grid says was measured.
    """
    metrics = ann_difficulty.compute(
        x,
        k=k,
        k_hub=k_hub,
        nlist=nlist,
        max_rows=0,
        seed=seed,
        backend=backend,
        chunk_rows=chunk_rows,
    )
    reported = ann_difficulty.summary(metrics)

    for name in ("lid_median", "relative_contrast_median"):
        if reported[name] is None:
            raise HubStabilityError(
                f"{name} was not measurable on this draw: every query was "
                "discarded, which means the draw is degenerate"
            )

    return {
        "lid_median": float(reported["lid_median"]),
        "relative_contrast_median": float(reported["relative_contrast_median"]),
        "hubness_skew": float(reported["hubness_skew"]),
        "ivf_gini": float(reported["ivf_gini"]),
        "hubness_gini": ann_difficulty.hubness_gini(metrics.k_occurrence),
        "hub_share_top1pct": ann_difficulty.hub_share_top1pct(metrics.k_occurrence),
    }


def sweep(
    real: np.ndarray,
    synthetic: dict[str, np.ndarray],
    *,
    ns: Sequence[int],
    draws: int,
    k: int,
    k_hub: int,
    nlist: int,
    seed: int,
    backend: str,
    chunk_rows: int,
    preprocess: str,
) -> dict:
    """Measure every statistic across repeated draws, at every N in the grid.

    The real corpus gets `draws` subsamples per N -- that is the subsample
    noise the gate band is judged against. Each synthetic series gets exactly
    one, mirroring the v0 noise floor where each training seed is one
    measurement; their mean feeds condition 2 and their spread is reported
    beside it as the training-seed floor at that N.

    `preprocess` is not a convenience knob. The corpora these families load
    from disk are raw -- `glove_250k.npy` has row norms spanning roughly
    2.17 to 11.33 -- while the generator samples this study compares them
    against already come out at exactly unit norm. Measuring both sides as
    stored would compare a raw corpus against a normalised one, and every
    statistic here is scale-sensitive enough that the comparison would read
    as a generator deficit when it is really a units mismatch. Every figure
    committed under `docs/datasets/` for these families was measured at
    `preprocess="l2"`, so that is the default here too; `"none"` stays
    reachable for anyone who explicitly wants the as-stored numbers, and
    whichever mode was used is recorded in the returned `conditions` block
    so a downstream reader never has to guess.
    """
    real = maybe_l2_normalize(real, preprocess)
    synthetic = {
        label: maybe_l2_normalize(x, preprocess) for label, x in synthetic.items()
    }

    measure = {
        "k": k,
        "k_hub": k_hub,
        "nlist": nlist,
        "seed": seed,
        "backend": backend,
        "chunk_rows": chunk_rows,
    }
    conditions = {**measure, "preprocess": preprocess}
    cells = []

    for n in ns:
        indices, disjoint = allocate_draws(real.shape[0], n, draws, seed)
        per_draw = [measure_draw(real[rows], **measure) for rows in indices]
        real_spread = {
            name: summarize_spread([d[name] for d in per_draw]) for name in STATISTICS
        }

        per_series: dict[str, dict[str, float]] = {}
        for label in sorted(synthetic):
            series = synthetic[label]
            # Only rows[0] is used -- one draw per synthetic series, mirroring
            # the v0 noise floor where each training seed is one measurement.
            # `draws=2` is requested (not 1) purely to clear allocate_draws's
            # `draws < 2` guard; the disjoint flag it returns is discarded
            # because a single draw has nothing to be disjoint from. Do not
            # "simplify" this to draws=1 -- it would raise HubStabilityError.
            rows, _ = allocate_draws(series.shape[0], n, 2, seed)
            per_series[label] = measure_draw(series[rows[0]], **measure)

        synthetic_mean: dict[str, float] | None = None
        synthetic_spread: dict[str, dict[str, float]] | None = None
        if len(per_series) >= 2:
            synthetic_spread = {
                name: summarize_spread([v[name] for v in per_series.values()])
                for name in STATISTICS
            }
            # Read the mean off the spread block rather than recomputing it
            # with np.mean: summarize_spread uses statistics.fmean, and the
            # two differ at float-precision (largest observed delta 3.6e-15).
            # One series, one value -- there is no spread block, so that
            # path below still computes its own mean.
            synthetic_mean = {
                name: synthetic_spread[name]["mean"] for name in STATISTICS
            }
        elif len(per_series) == 1:
            only = next(iter(per_series.values()))
            synthetic_mean = {name: only[name] for name in STATISTICS}

        cells.append(
            {
                "n": n,
                "draws": draws,
                "draws_disjoint": disjoint,
                "pool_to_n": real.shape[0] / n,
                "real": {"per_draw": per_draw, "spread": real_spread},
                "synthetic": {
                    "series": sorted(per_series),
                    "per_series": per_series,
                    "mean": synthetic_mean,
                    "spread": synthetic_spread,
                },
                "verdicts": {
                    name: evaluate_rule(
                        real_spread[name],
                        None if synthetic_mean is None else synthetic_mean[name],
                        draws_disjoint=disjoint,
                    )
                    for name in STATISTICS
                },
            }
        )

    return {
        "pool_rows": int(real.shape[0]),
        "conditions": conditions,
        "rule": {
            "stable_max_range_pct": STABLE_MAX_RANGE_PCT,
            "min_separation_in_ranges": MIN_SEPARATION_IN_RANGES,
        },
        "statistics": list(STATISTICS),
        "cells": cells,
    }


def _load_series(specs: Sequence[str] | None) -> dict[str, np.ndarray]:
    """Parse repeatable LABEL=PATH arguments into loaded arrays."""
    loaded: dict[str, np.ndarray] = {}
    for spec in specs or []:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            raise HubStabilityError(f"--synthetic-path wants LABEL=PATH, got {spec!r}")
        if label in loaded:
            raise HubStabilityError(f"--synthetic-path {label!r} given twice")
        loaded[label] = np.load(path)
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--real-path", type=str, required=True, help="Real corpus .npy to draw from."
    )
    parser.add_argument(
        "--synthetic-path",
        type=str,
        action="append",
        metavar="LABEL=PATH",
        help=(
            "Repeatable. One per generator seed. Each is measured once per N; "
            "their mean is what the rule's second condition judges. Omit "
            "entirely to measure real-side stability alone."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        action="append",
        required=True,
        dest="ns",
        help="Repeatable. Subsample size to measure at.",
    )
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--k-hub", type=int, default=10)
    parser.add_argument("--nlist", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        type=str,
        default="sklearn",
        choices=("sklearn", "torch"),
        help=(
            "Neighbour search. Default sklearn, which every committed figure "
            "was measured with; torch uses the GPU when there is one."
        ),
    )
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument(
        "--preprocess",
        type=str,
        default="l2",
        choices=("l2", "none"),
        help=(
            "Row normalisation applied to every series before measuring. "
            "Default l2, which is what every figure committed under "
            "docs/datasets/ for these families was measured at. Passing "
            "'none' measures the vectors as stored, which is not comparable "
            "with those figures."
        ),
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Also write the JSON here."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        real = np.load(args.real_path)
        synthetic = _load_series(args.synthetic_path)
        result = sweep(
            real,
            synthetic,
            ns=args.ns,
            draws=args.draws,
            k=args.k,
            k_hub=args.k_hub,
            nlist=args.nlist,
            seed=args.seed,
            backend=args.backend,
            chunk_rows=args.chunk_rows,
            preprocess=args.preprocess,
        )
    except (HubStabilityError, OSError, ValueError, ZeroDivisionError) as exc:
        # stderr, so stdout stays parseable as JSON or empty, never half a
        # report -- the same contract noise_floor.py keeps.
        print(f"hub_stability: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    result["real_path"] = args.real_path
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
