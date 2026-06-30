"""V4 multi-window temporal smoothing — inference-time aggregation across
consecutive alert windows.

Motivation: V4 trains and predicts per window, but a single anomaly event
typically lights up several **consecutive** alert windows.  Averaging the
per-window (x, y, z) predictions across the windows of one alert burst
sharpens the spatial estimate by ≈ √n_windows for white per-window noise.

Two aggregation modes:

  - ``"mean"``    : arithmetic mean of `(x, y, z)` predictions across the burst.
  - ``"weighted"``: V3-anomaly-score-weighted mean — windows with higher
                    anomaly scores contribute more.  Useful when the burst
                    starts and ends with marginal alert windows.

The smoother is **inference-time only** and operates on already-trained V4
checkpoints.  No re-training is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class AlertBurst:
    """One contiguous run of alert windows from a single recording."""

    recording_id: str
    dataset_id: str
    window_indices: list[int]  # indices into the per-window prediction stream
    per_window_xyz: np.ndarray  # (n_windows_in_burst, 3)
    per_window_score: np.ndarray  # (n_windows_in_burst,) V3 score
    aggregated_xyz: np.ndarray  # (3,) burst-level estimate

    @property
    def n_windows(self) -> int:
        return len(self.window_indices)


def detect_alert_bursts(
    alert_mask: np.ndarray,
    *,
    min_burst_size: int = 1,
    max_gap: int = 0,
) -> list[tuple[int, int]]:
    """Find contiguous runs of `True` in `alert_mask`.

    `max_gap = 0` requires strict contiguity; `max_gap > 0` merges bursts
    separated by up to `max_gap` non-alert windows (to bridge V3 misses
    inside a longer event).  Bursts shorter than `min_burst_size` are
    dropped.

    Returns a list of `(start_idx, end_idx)` half-open intervals.
    """
    alert_mask = np.asarray(alert_mask, dtype=bool)
    if alert_mask.size == 0:
        return []
    bursts: list[tuple[int, int]] = []
    i = 0
    n = alert_mask.size
    while i < n:
        if not alert_mask[i]:
            i += 1
            continue
        start = i
        end = i + 1
        while end < n:
            if alert_mask[end]:
                end += 1
                continue
            if max_gap > 0:
                # peek ahead up to `max_gap` non-alerts; if next alert exists
                # within the gap window, extend.
                j = end + 1
                gap_len = 1
                while j < n and not alert_mask[j] and gap_len < max_gap:
                    j += 1
                    gap_len += 1
                if j < n and alert_mask[j]:
                    end = j + 1
                    continue
            break
        if end - start >= min_burst_size:
            bursts.append((start, end))
        i = end
    return bursts


def smooth_predictions_over_bursts(
    per_window_xyz: np.ndarray,
    alert_mask: np.ndarray,
    per_window_score: np.ndarray | None = None,
    *,
    aggregation: Literal["mean", "weighted"] = "mean",
    min_burst_size: int = 2,
    max_gap: int = 1,
    recording_id: str = "",
    dataset_id: str = "",
) -> tuple[np.ndarray, list[AlertBurst]]:
    """Aggregate per-window V4 predictions into per-burst estimates.

    Returns ``(smoothed_xyz, bursts)``:
      - ``smoothed_xyz`` is shape ``(n_alert_windows, 3)``: each alert
        window inherits its parent burst's aggregated prediction;
        non-alert windows do not appear (gating still applies).
      - ``bursts`` exposes the burst structure for downstream evaluation.

    Single-window bursts (when `min_burst_size = 1`) inherit their own
    prediction unchanged — equivalent to skipping smoothing.
    """
    per_window_xyz = np.asarray(per_window_xyz, dtype=np.float64)
    alert_mask = np.asarray(alert_mask, dtype=bool)
    if per_window_score is None:
        per_window_score = np.ones(alert_mask.shape, dtype=np.float64)
    else:
        per_window_score = np.asarray(per_window_score, dtype=np.float64)

    burst_intervals = detect_alert_bursts(
        alert_mask, min_burst_size=min_burst_size, max_gap=max_gap
    )
    out_rows: list[np.ndarray] = []
    bursts: list[AlertBurst] = []
    for start, end in burst_intervals:
        idxs = list(range(start, end))
        bxyz = per_window_xyz[idxs]
        bscore = per_window_score[idxs]
        if aggregation == "mean":
            agg = bxyz.mean(axis=0)
        elif aggregation == "weighted":
            w = bscore - bscore.min() + 1e-6  # shift to positive
            agg = (bxyz * w[:, None]).sum(axis=0) / w.sum()
        else:
            raise ValueError(f"unknown aggregation {aggregation!r}")
        bursts.append(
            AlertBurst(
                recording_id=recording_id,
                dataset_id=dataset_id,
                window_indices=idxs,
                per_window_xyz=bxyz.astype(np.float32),
                per_window_score=bscore.astype(np.float32),
                aggregated_xyz=agg.astype(np.float32),
            )
        )
        out_rows.append(np.broadcast_to(agg, (end - start, 3)).copy())
    if out_rows:
        smoothed = np.concatenate(out_rows, axis=0).astype(np.float32)
    else:
        smoothed = np.zeros((0, 3), dtype=np.float32)
    return smoothed, bursts


def evaluate_burst_localization(
    bursts: list[AlertBurst],
    ground_truth_xyz: np.ndarray,
) -> dict:
    """3-D MAE / 95th percentile of `aggregated_xyz` against a single
    ground-truth target shared by all bursts of one recording.

    Inputs:
      - bursts: list of `AlertBurst` (typically one recording's bursts)
      - ground_truth_xyz: (3,) — the recording-level spatial label.

    Returns ``{n_bursts, mean_error_m, p95_error_m, per_burst_errors}``.
    """
    if not bursts:
        return {
            "n_bursts": 0,
            "mean_error_m": float("nan"),
            "p95_error_m": float("nan"),
            "per_burst_errors": [],
        }
    gt = np.asarray(ground_truth_xyz, dtype=np.float64)
    errs = np.array(
        [float(np.linalg.norm(b.aggregated_xyz - gt)) for b in bursts]
    )
    return {
        "n_bursts": int(len(bursts)),
        "mean_error_m": float(errs.mean()),
        "p95_error_m": float(np.percentile(errs, 95)) if errs.size > 0 else float("nan"),
        "per_burst_errors": errs.tolist(),
    }


__all__ = [
    "AlertBurst",
    "detect_alert_bursts",
    "evaluate_burst_localization",
    "smooth_predictions_over_bursts",
]
