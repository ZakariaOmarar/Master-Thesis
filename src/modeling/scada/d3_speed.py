"""V5.1 — D3 speed conditioning for the V4 localization head.

Supports the RQ4a study: re-run V4 on D3 with `s_t` set to {none, one-hot
speed, c_t-only}.  This module produces the **one-hot speed** SCADA tensor that V4 consumes via
its `scada_dim` slot.  D3's three speed buckets (`speed1` / `speed2` /
`speed3`) come straight from the loader's `op_condition` field (label scheme
`d3_speed_with_hit`).  D1/D2 segments don't have a meaningful speed bucket —
the helper returns `None` for them, so V4 will fill the slot with zeros.

Why one-hot rather than the raw RPM:
  - D3 only has three nominal speeds (no continuous RPM telemetry alongside
    the WAVs).  A one-hot is the most honest representation.
  - V5.2 will mine continuous SCADA channels from real Illwerke data; the
    plan keeps the two pieces separate.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ...ingestion.test_dataset_loader import TestDatasetSegment

D3_SPEED_BUCKETS = ("speed1", "speed2", "speed3")
D3_SCADA_DIM = len(D3_SPEED_BUCKETS)


def d3_speed_one_hot(op_condition: str | None) -> np.ndarray | None:
    """Return a 3-D one-hot for the D3 speed bucket, or `None` if unrecognised.

    `op_condition` for D3 is one of `speed1`, `speed2`, `speed3` (set by the
    loader for both healthy speed-* recordings and `hit_between_*_speedN`).
    """
    if op_condition is None:
        return None
    op = op_condition.strip().lower()
    if op not in D3_SPEED_BUCKETS:
        return None
    out = np.zeros(D3_SCADA_DIM, dtype=np.float32)
    out[D3_SPEED_BUCKETS.index(op)] = 1.0
    return out


def d3_speed_lookup(
    segments: Iterable[TestDatasetSegment],
) -> dict[str, np.ndarray]:
    """Build a `{recording_id: one_hot}` dict for any segment whose
    `op_condition` matches one of `D3_SPEED_BUCKETS`.

    Originally D3-only; now also covers D4 (whose `op_condition` field is
    populated with `speed{N}` for both healthy speed-bucket recordings and
    the spatial-labeled RandomFault subfolders).  Segments whose
    `op_condition` doesn't match a known bucket are skipped, so D1 and D2
    naturally fall through.
    """
    out: dict[str, np.ndarray] = {}
    for s in segments:
        oh = d3_speed_one_hot(s.op_condition)
        if oh is not None:
            out[s.recording_id] = oh
    return out


__all__ = ["D3_SCADA_DIM", "D3_SPEED_BUCKETS", "d3_speed_lookup", "d3_speed_one_hot"]
