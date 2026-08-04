from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def batch_log_ratio_profile(
    x: Tensor,
    k: int,
    max_points: int = 0,
    eps: float = 1.0e-12,
) -> Optional[Tensor]:
    """Mean ``log(r_i / r_k)`` over a batch's within-batch neighbours.

    Returns one entry per ``i = 1 .. k-1``, or ``None`` when the batch is too
    small or every query hits a degenerate case.

    This is the sufficient statistic the Hill estimator reduces to a scalar
    (``LID = -1 / mean_i(p_i)``), so matching it moves LID -- but it is
    bounded and smooth where LID's ``-1/x`` blows up as the mean log-ratio
    approaches zero, and it is a vector, so it constrains the *shape* of the
    neighbourhood rather than only its scale.

    The distances are within-batch and therefore much larger than true k-NN
    distances in the full dataset. That bias is identical on the real and fake
    sides and cancels in the difference -- the same equal-N discipline the ANN
    report enforces.

    Two degenerate cases are dropped rather than clamped, matching
    ``src.eval.ann_difficulty.survivor_mask``: ``r_1 == 0`` (the query sits on
    a duplicate) and ``r_1 == r_k`` (every neighbour ties). Clamping would
    invent a number; dropping declines to answer.
    """
    n = x.shape[0]
    if max_points > 1 and n > max_points:
        idx = torch.randperm(n, device=x.device)[:max_points]
        x = x[idx]
        n = max_points
    k_eff = min(int(k), n - 1)
    if k_eff < 2:
        return None

    # Expanded squares with a clamped sqrt rather than torch.cdist: cdist's
    # gradient is undefined at distance zero, so a single pair of duplicate
    # rows -- exactly what a collapsing generator produces -- would poison the
    # entire backward pass, including for rows that survive the filter below.
    sq = (x * x).sum(dim=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    d2 = d2.clamp(min=0.0)
    # Exclude each row from its own neighbour list by index, not by dropping
    # the nearest column: an exact duplicate ties with the query at distance
    # zero and would otherwise be dropped in its place.
    d2 = d2 + torch.diag(torch.full((n,), float("inf"), device=x.device, dtype=x.dtype))

    # Select and filter on SQUARED distances, before any epsilon clamp.
    # Ordering is identical, and it keeps the degenerate tests exact: a pair of
    # duplicate rows has d2 == 0, so `r2_1 > 0` drops it. Clamping first would
    # turn that 0 into eps, the row would survive, and its log-ratio -- a large
    # negative number rather than -inf -- would quietly dominate the mean.
    r2, _ = torch.topk(d2, k_eff, dim=1, largest=False, sorted=True)
    survivors = (r2[:, 0] > 0.0) & (r2[:, 0] < r2[:, -1])
    if not bool(survivors.any()):
        return None

    # The clamp below is only gradient insurance for near-duplicates; every
    # surviving row already has a strictly positive nearest distance.
    kept = r2[survivors].clamp(min=eps).sqrt()
    ratio = (kept[:, :-1] / kept[:, -1:]).clamp(min=eps, max=1.0)
    return torch.log(ratio).mean(dim=0)
