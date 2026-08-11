"""Prose fragments the ANN panels compute rather than hard-code.

A panel's fixed prose lives with the panel in `panels.py`. What lives here is
the part that depends on the run: which measurement conditions each series
was actually measured under, and whether any series contributed no queries at
all. Both exist so a reader cannot mistake one series' numbers for all of
them.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.eval import ann_difficulty
from src.eval.eda.series import Series


def ann_condition_note(
    series: Sequence[Series],
    ann_metrics: dict[str, ann_difficulty.AnnMetrics],
    attrs: tuple[tuple[str, str], ...],
) -> str:
    """State the actual per-series ANN measurement conditions for `attrs`.

    `attrs` is a sequence of (AnnMetrics field name, display label) pairs,
    e.g. (("num_rows", "rows"), ("k", "k")). When every series in this run
    was measured under the same conditions, one summary sentence is enough.
    When they differ -- e.g. a series with fewer rows than --ann-max-rows
    gets num_rows, k or nlist clamped -- a reader must not be able to mistake
    one series' numbers for all of them, so each series' actual values are
    spelled out instead.
    """
    per_series = {
        s.name: tuple(getattr(ann_metrics[s.name], field) for field, _ in attrs)
        for s in series
    }
    if len(set(per_series.values())) == 1:
        values = next(iter(per_series.values()))
        parts = ", ".join(f"{label}={v}" for (_, label), v in zip(attrs, values))
        return f" Measured with {parts} for every series."
    per_series_text = "; ".join(
        f"{name} ("
        + ", ".join(f"{label}={v}" for (_, label), v in zip(attrs, values))
        + ")"
        for name, values in per_series.items()
    )
    return (
        " Measurement conditions differ across series (a series with fewer "
        f"rows than requested has k and/or nlist clamped): {per_series_text}."
    )


def ann_discarded_note(
    series: Sequence[Series],
    ann_metrics: dict[str, ann_difficulty.AnnMetrics],
) -> str:
    """Call out any series that contributed no queries at all, and why.

    `summary` returns None for `lid_median` and `relative_contrast_median`
    when every query was discarded, and the panel simply has no trace for
    that series. That is the honest answer, but on its own it renders as a
    silent `n/a`. The two ways to get there are a set of exact duplicates
    (every query has r_1 == 0) and `k == 1` -- either passed via `--ann-k 1`
    or clamped there by `knn` for a two-row series -- where r_1 and r_k are
    the same column, so `survivor_mask`'s r_1 < r_k can never hold.

    Only the LID/contrast panels need this: hubness and IVF balance are
    computed over every row regardless of which queries survived.
    """
    affected = []
    for s in series:
        m = ann_metrics[s.name]
        if m.num_rows == 0 or m.discarded_queries != m.num_rows:
            continue
        # k < 2 is checked first: at k == 1 the mask cannot pass whatever the
        # data looks like, so it explains the whole series on its own.
        reason = (
            "measured at k=1, where the nearest and the k-th neighbour are "
            "the same point, so no query can pass the estimator's r_1 < r_k "
            "test"
            if m.k < 2
            else "every query sits on an exact duplicate"
        )
        affected.append(f"{s.name} ({reason})")
    if not affected:
        return ""
    return (
        f" <b>No surviving queries for {'; '.join(affected)}</b>. Both panels "
        "report n/a for those series rather than a number, and draw no trace "
        "for them."
    )


# Appended to all three ANN panel notes. Invariant 3 in AGENTS.md: these are
# self-queried subsample figures with no absolute meaning, and a reader who
# checks them against published SIFT1M numbers is drawing a false conclusion.
ANN_NOTE_SUFFIX = (
    " Compare against the <code>real</code> series in this report only. "
    "These numbers come from a self-queried subsample, so they are not "
    "comparable with published SIFT1M figures."
)
