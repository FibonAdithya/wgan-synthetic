from __future__ import annotations

import torch
from torch import Tensor


def batch_log_ratio_profile(
    x: Tensor,
    k: int,
    max_points: int = 0,
    eps: float = 1.0e-12,
) -> Tensor | None:
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


class LogRatioTarget:
    """EMA of the real batches' log-ratio profile.

    The real distribution is fixed, so a fresh per-minibatch estimate only
    adds variance to the gradient. Averaging costs nothing and is much
    quieter. At decay 0.99 the average settles within about a hundred steps,
    which is why it is deliberately *not* checkpointed: unlike the generator
    weight EMA at decay 0.999, losing it on resume costs a brief transient
    rather than a thousand-step average.

    That argument depends on the decay, so here is the condition that ends it:
    **if this decay is ever raised toward 0.999, the target has to start being
    checkpointed.** A resumed run would otherwise rebuild a thousand-step
    average from scratch and regularize against a target that is still
    settling. There is no knob for this today -- `train_wgan_gp` constructs
    the target with the default and never passes a decay, while its siblings
    `lid_reg_alpha`, `lid_reg_k` and `lid_reg_max_points` all come from the
    config -- so the tripwire is on adding one, or on changing the default.
    """

    def __init__(self, decay: float = 0.99):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.value: Tensor | None = None

    def update(self, profile: Tensor) -> Tensor:
        observed = profile.detach()
        # A short final batch clamps k_eff and shortens the profile; a stale
        # target of the wrong length would broadcast silently.
        if self.value is None or self.value.shape != observed.shape:
            self.value = observed.clone()
        else:
            self.value.mul_(self.decay).add_(observed, alpha=1.0 - self.decay)
        return self.value


def log_ratio_penalty(
    fake: Tensor,
    real: Tensor,
    k: int,
    max_points: int,
    target: LogRatioTarget,
) -> Tensor:
    """L1 gap between the fake profile and the EMA of the real one.

    Zero when either side is degenerate -- an all-duplicate or all-tied batch
    yields no usable queries, and a penalty invented from nothing would push
    the generator in an arbitrary direction at exactly the moment it is
    collapsing.
    """
    zero = torch.zeros((), device=fake.device, dtype=fake.dtype)
    fake_profile = batch_log_ratio_profile(fake, k=k, max_points=max_points)
    if fake_profile is None:
        return zero
    with torch.no_grad():
        real_profile = batch_log_ratio_profile(real, k=k, max_points=max_points)
    if real_profile is None:
        return zero
    real_profile = real_profile.to(fake_profile.dtype)
    # Check the shape match against fake_profile BEFORE updating the EMA: if
    # this were done after, an odd-length real_profile would already have
    # reset the accumulated average (LogRatioTarget.update resets on any
    # shape change) by the time we discover the mismatch and bail out,
    # discarding the average for nothing. Comparing first means a mismatched
    # batch never touches the target at all.
    if real_profile.shape != fake_profile.shape:
        return zero
    reference = target.update(real_profile)
    return (fake_profile - reference).abs().sum()
