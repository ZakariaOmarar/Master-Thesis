"""Shared V4 localization metric helpers.

The canonical V4 localization accuracy is the **event-aggregated** 3-D MAE: each
recording's per-knock predictions are averaged into a single position estimate,
the error is taken against that recording's ground truth, and the result is
meaned over recordings.  This is the deployment-faithful number (a fault event
fires several knocks; you localize the event once by pooling them) and is
variance-reduced relative to scoring every knock independently.

`event_aggregated_mae` is the single source of truth for that aggregation so the
trainer headline, the V3-gated CV metric (`v4_cv_common.gated_fold_mae`), and the
`full_run` Stage 5b holdout metric all group knocks identically.  The grouping
key — ``f"{Path(s.source_dir).name}/{s.recording_id}"`` — must never diverge from
the trainer's `val_recording_breakdown` key.

This module depends only on numpy + pathlib (samples are duck-typed: anything
with ``.source_dir`` and ``.recording_id``), so it imports nothing from the
trainer or orchestration packages and can be imported by all of them without a
cycle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def event_aggregated_mae(
    predictions: np.ndarray,
    targets: np.ndarray,
    samples: list,
) -> tuple[float, np.ndarray, int]:
    """Event-aggregated (per-recording) 3-D mean absolute error.

    Groups the per-knock predictions by ``f"{Path(s.source_dir).name}/
    {s.recording_id}"``, averages each recording's predictions into one estimate,
    takes the 3-D Euclidean error against that recording's ground truth (all
    knocks of a recording share the GT, so the group's first target is used —
    identical to the trainer's ``val_recording_breakdown``), and means over
    recordings.

    Args:
      predictions: ``(n, 3)`` per-knock predicted xyz, aligned 1:1 with samples.
      targets:     ``(n, 3)`` per-knock target xyz, aligned 1:1 with samples.
      samples:     length-``n`` list of objects exposing ``source_dir`` and
        ``recording_id`` (e.g. ``V4Sample``), in the same order as predictions.

    Returns:
      ``(mae, agg_errs, n_recordings)`` where ``agg_errs`` is the
      ``(n_recordings,)`` array of per-recording errors (the sample a CI would
      bootstrap over) and ``mae`` is their mean.  Returns
      ``(nan, empty, 0)`` when there are no samples or the lengths disagree
      (so callers degrade rather than raise).
    """
    pred = np.asarray(predictions, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)
    n = pred.shape[0]
    if n == 0 or tgt.shape[0] != n or len(samples) != n:
        return float("nan"), np.zeros((0,), dtype=np.float64), 0

    keys = [f"{Path(s.source_dir).name}/{s.recording_id}" for s in samples]
    errs: list[float] = []
    for k in sorted(set(keys)):
        mask = np.fromiter((vk == k for vk in keys), dtype=bool, count=n)
        if not mask.any():
            continue
        pred_mean = pred[mask].mean(axis=0)
        tgt_k = tgt[mask][0]  # all knocks of a recording share the GT
        errs.append(float(np.linalg.norm(pred_mean - tgt_k)))

    agg = np.asarray(errs, dtype=np.float64)
    mae = float(agg.mean()) if agg.size else float("nan")
    return mae, agg, int(agg.size)


__all__ = ["event_aggregated_mae"]
