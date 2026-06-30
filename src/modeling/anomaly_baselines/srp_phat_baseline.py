"""V0 classical SRP-PHAT localization baseline (RQ3 reference number).

Thin wrapper around `src/modeling/localization/classical.py`'s `gcc_phat` +
`srp_phat_3d` primitives.  Runs on any `TestDatasetSegment` whose
`mic_positions` are known and whose spatial ground truth is available.

Spatial-label resolution:
  - D2 RandomFault recordings: `s.spatial_label` is the parsed `pos_(x,y,z)_*`
    coordinate from the folder name (already converted to meters by the loader).
  - D3 `hit_between_<a>_<b>_speed<n>` recordings: the loader's `spatial_label`
    is None for these; we recover the ground truth here as the midpoint of the
    two mic positions named in the folder.
  - All other recordings have no spatial ground truth and are skipped.

Reported metric: 3-D Euclidean error (mean / 95th percentile across recordings).
This is the V0 reference number that V4 must beat in the RQ3 ablation table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ...ingestion.positions import PositionRegistry
from ...ingestion.test_dataset_loader import (
    TestDatasetLoader,
    TestDatasetSegment,
)
from ..localization.classical import gcc_phat, srp_phat_3d

_HIT_RE = re.compile(r"^hit_between_(?P<a>[A-Za-z0-9]+)_(?P<b>[A-Za-z0-9]+)_speed\d+$")


@dataclass(frozen=True)
class SRPConfig:
    """Hyperparameters for the V0 classical SRP-PHAT baseline.

    Defaults are tuned for the ~ 10 cm bench-top prototype scale used in
    D2/D3/D4 (mic positions span 0–11 cm; spatial labels span ~ 50 cm).
    `grid_step_m = 1 cm` is the smallest resolution that still finishes
    in seconds per recording on CPU; a smaller step would over-resolve
    the SRP peak which is itself bounded above by the array's spatial
    aliasing limit.
    """

    c_air: float = 343.0  # speed of sound, m/s
    window_seconds: float = 1.0
    grid_step_m: float = 0.01  # 1 cm grid resolution (was 5 cm)
    grid_margin_m: float = 0.20  # 20 cm padding so the off-array spatial
                                 # labels (D2/D4 reach 40 cm) are inside the
                                 # search volume even though the mic bbox
                                 # itself is only ~ 11 cm wide.


# ---------------------------------------------------------------------------
# Pairwise GCC-PHAT stack (generic over any Nm)
# ---------------------------------------------------------------------------


def _compute_gcc_stack(
    mic_data: np.ndarray,
    fs: float,
    mic_xyz: np.ndarray,
    cfg: SRPConfig,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Average GCC-PHAT over a centred analysis window for every mic pair."""
    n_mics = int(mic_data.shape[0])
    pairs = [(i, j) for i in range(n_mics) for j in range(i + 1, n_mics)]

    max_dist = 0.0
    for i, j in pairs:
        d = float(np.linalg.norm(mic_xyz[i] - mic_xyz[j]))
        max_dist = max(max_dist, d)
    max_delay_samples = max(1, int(np.ceil(max_dist / cfg.c_air * fs)))

    n_samples = int(mic_data.shape[1])
    win = max(1, min(n_samples, int(round(fs * cfg.window_seconds))))
    start = (n_samples - win) // 2
    seg = mic_data[:, start : start + win].astype(np.float64)

    L = 2 * max_delay_samples + 1
    stack = np.zeros((len(pairs), L), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        stack[k] = gcc_phat(seg[i], seg[j], max_delay_samples)
    return stack, pairs


# ---------------------------------------------------------------------------
# SRP-PHAT predictor
# ---------------------------------------------------------------------------


def predict_srp_phat(
    mic_data: np.ndarray,
    fs: float,
    mic_xyz: np.ndarray,
    cfg: SRPConfig | None = None,
) -> np.ndarray:
    """Argmax of the SRP-PHAT power volume over a bbox-padded search grid."""
    cfg = cfg or SRPConfig()
    if mic_xyz.shape[0] != mic_data.shape[0]:
        raise ValueError("mic_xyz and mic_data row counts must match")

    stack, pairs = _compute_gcc_stack(mic_data, fs, mic_xyz, cfg)

    bbox_lo = mic_xyz.min(axis=0) - cfg.grid_margin_m
    bbox_hi = mic_xyz.max(axis=0) + cfg.grid_margin_m
    grid_x = np.arange(bbox_lo[0], bbox_hi[0] + cfg.grid_step_m, cfg.grid_step_m)
    grid_y = np.arange(bbox_lo[1], bbox_hi[1] + cfg.grid_step_m, cfg.grid_step_m)
    grid_z = np.arange(bbox_lo[2], bbox_hi[2] + cfg.grid_step_m, cfg.grid_step_m)

    power = srp_phat_3d(
        stack,
        mic_xyz.astype(np.float64),
        grid_x.astype(np.float64),
        grid_y.astype(np.float64),
        grid_z.astype(np.float64),
        fs=float(fs),
        c=cfg.c_air,
        mic_pairs=pairs,
    )
    idx = np.unravel_index(int(np.argmax(power)), power.shape)
    return np.array(
        [grid_x[idx[0]], grid_y[idx[1]], grid_z[idx[2]]], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Spatial-label resolution + per-segment evaluation
# ---------------------------------------------------------------------------


def _resolve_ground_truth(
    s: TestDatasetSegment, registry: PositionRegistry
) -> np.ndarray | None:
    if s.spatial_label is not None:
        return np.asarray(s.spatial_label, dtype=np.float64)
    if s.dataset_id != "d3":
        return None
    m = _HIT_RE.match(s.recording_id)
    if m is None:
        return None
    try:
        pa = registry.lookup_mic(m.group("a"))
        pb = registry.lookup_mic(m.group("b"))
    except KeyError:
        return None
    return ((pa + pb) / 2.0).astype(np.float64)


def evaluate_srp_phat(
    loader: TestDatasetLoader,
    cfg: SRPConfig | None = None,
) -> list[dict]:
    """Predict + score SRP-PHAT on every loader segment that has ground truth."""
    cfg = cfg or SRPConfig()
    out: list[dict] = []
    for s in loader.list_segments():
        gt = _resolve_ground_truth(s, loader.registry)
        if gt is None:
            continue
        pred = predict_srp_phat(
            s.segment.mic_data,
            float(s.segment.mic_sample_rate),
            s.mic_positions,
            cfg=cfg,
        )
        err = float(np.linalg.norm(pred - gt))
        out.append(
            {
                "dataset_id": s.dataset_id,
                "recording_id": s.recording_id,
                "ground_truth_xyz": gt,
                "predicted_xyz": pred,
                "error_m": err,
                "spatial_label_source": (
                    "folder" if s.spatial_label is not None else "mic_pair_midpoint"
                ),
            }
        )
    return out


def summarise(records: list[dict]) -> dict:
    """Mean / 95th-percentile / per-recording error summary."""
    if not records:
        return {"n_recordings": 0, "mean_error_m": 0.0, "p95_error_m": 0.0}
    errs = np.array([r["error_m"] for r in records], dtype=np.float64)
    return {
        "n_recordings": int(errs.shape[0]),
        "mean_error_m": float(errs.mean()),
        "p95_error_m": float(np.percentile(errs, 95)),
        "median_error_m": float(np.median(errs)),
    }


__all__ = [
    "SRPConfig",
    "evaluate_srp_phat",
    "predict_srp_phat",
    "summarise",
]
