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

import numpy as np

from src.eval import ann_difficulty


class HubStabilityError(Exception):
    """The sweep could not run as asked -- bad grid, bad corpus, bad series."""


def allocate_draws(
    pool_size: int, n: int, draws: int, seed: int
) -> tuple[list[np.ndarray], bool]:
    """Row indices for `draws` subsamples of `n` rows, and whether they overlap.

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
    inclusive.

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
    # Add small tolerance for floating point precision
    stable = range_pct <= STABLE_MAX_RANGE_PCT + 1e-9

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
        # Add small tolerance for floating point precision
        discriminating = separation >= MIN_SEPARATION_IN_RANGES - 1e-9

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
