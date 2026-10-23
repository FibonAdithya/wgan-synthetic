"""Batches with a fixed body/tail mix for flag-conditioned training."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from src.data.dataset import NumpyTensorDataset
from src.models.conditional import BODY, TAIL


class FlagBatchStream:
    """Endless stream of `(rows, flags)` batches with a fixed tail share.

    Each batch holds `batch_size - n_tail_rows` body rows followed by
    `n_tail_rows = round(tail_share * batch_size)` tail rows. Body and tail
    rows come from separate shuffling loaders (seeds `seed` and `seed + 1`),
    each restarted when it runs out. Rows are float32 and flags are int64,
    both on CPU; the caller moves them to the device.
    """

    def __init__(
        self,
        x: np.ndarray,
        flags: np.ndarray,
        *,
        batch_size: int,
        tail_share: float,
        seed: int,
        num_workers: int,
        pin_memory: bool,
    ):
        # Function-level import: train_wgan_gp imports this module.
        from src.train.train_wgan_gp import build_dataloader

        n_tail = round(tail_share * batch_size)
        if n_tail < 1 or n_tail >= batch_size:
            raise ValueError(
                f"tail_share={tail_share} at batch_size={batch_size} gives "
                f"{n_tail} tail rows; need between 1 and {batch_size - 1}"
            )
        n_body = batch_size - n_tail
        flags = np.asarray(flags)
        body_x = x[flags == BODY]
        tail_x = x[flags == TAIL]
        for name, rows, need in (("body", body_x, n_body), ("tail", tail_x, n_tail)):
            if len(rows) < need:
                raise ValueError(
                    f"{name} has {len(rows)} rows, fewer than its batch size {need}"
                )
        self.n_body, self.n_tail = n_body, n_tail
        self._kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
        self._loaders = [
            build_dataloader(
                NumpyTensorDataset(body_x), batch_size=n_body, seed=seed, **self._kwargs
            ),
            build_dataloader(
                NumpyTensorDataset(tail_x),
                batch_size=n_tail,
                seed=seed + 1,
                **self._kwargs,
            ),
        ]
        self._iters = [iter(loader) for loader in self._loaders]

    def _draw(self, i: int) -> Tensor:
        try:
            return next(self._iters[i])
        except StopIteration:
            self._iters[i] = iter(self._loaders[i])
            return next(self._iters[i])

    def next(self) -> tuple[Tensor, Tensor]:
        body, tail = self._draw(0), self._draw(1)
        rows = torch.cat([body, tail]).float()
        flags = torch.cat(
            [
                torch.full((self.n_body,), BODY, dtype=torch.long),
                torch.full((self.n_tail,), TAIL, dtype=torch.long),
            ]
        )
        return rows, flags
