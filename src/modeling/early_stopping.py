"""Shared early-stopping bookkeeping for the V1-V4 trainers.

All four trainers (V1/V2 SSL, V3 flow, V4 localization head) stop on a
validation metric that should be minimised, snapshot the best weights as they
go, and optionally restore them at the end. This module holds the two pieces
they had each copied: the CPU state-dict snapshot and the patience counter.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

import torch

S = TypeVar("S")


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached CPU clone of ``module``'s state dict.

    Used instead of ``copy.deepcopy`` for the "best so far" snapshot. deepcopy
    routes a state dict through the pickling path and allocates fresh CPU
    tensors each time, which fragments system RAM across the hundreds of best
    updates a long run produces. This single comprehension lets the previous
    snapshot's tensors be reclaimed as soon as the binding is overwritten.
    """
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


class EarlyStopping(Generic[S]):
    """Track the lowest validation value seen and the snapshot taken there.

    Call :meth:`update` once per epoch with the validation value and a callable
    that produces a snapshot of the current weights. The value has to improve on
    the running best by more than ``min_delta`` to count; ``update`` returns
    ``True`` once ``patience`` consecutive epochs pass without such an
    improvement, which the caller uses to break out of its training loop. The
    snapshot is kept opaque, so a trainer can hand back a single module's state
    dict or a dict of several (e.g. flow plus pooling head).
    """

    def __init__(self, patience: int, min_delta: float = 0.0,
                 initial: S | None = None) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_snapshot: S | None = initial
        self._num_bad = 0

    def update(self, value: float, snapshot_fn: Callable[[], S]) -> bool:
        """Record ``value``; refresh the snapshot if it improved. Returns whether
        patience has run out and training should stop."""
        if value < self.best - self.min_delta:
            self.best = value
            self.best_snapshot = snapshot_fn()
            self._num_bad = 0
        else:
            self._num_bad += 1
        return self._num_bad >= self.patience
