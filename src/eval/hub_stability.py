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
