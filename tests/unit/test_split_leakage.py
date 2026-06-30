"""Leakage-regression tests for the train/val/threshold-fit/holdout splits.

The scientific validity of the whole pipeline rests on three separations that
are currently *asserted in docstrings*:

  * V4 random split is at the RECORDING level — no recording's windows span
    train and val (`_split_samples_by_recording`).
  * V4 spatial holdout is at the POSITION level — no position's windows span
    train and holdout (`split_samples_by_position`); this is what makes the
    headline "localise an unseen position" claim honest.
  * V3 threshold percentiles are fit on a cohort DISJOINT from the reportable
    val cohort.  The V3 trainer realizes this as a nested
    `_split_segments_by_recording`, then HARD-ERRORS if either side collapses.

These tests fail loudly if a future refactor reintroduces window-level leakage
or removes the disjointness guarantee.  They are pure (no data, no models).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.modeling.context.v2_ssl import _split_segments_by_recording
from src.modeling.localization.v4_trainer import (
    V4Sample,
    _position_key,
    _split_samples_by_recording,
    split_samples_by_position,
)


def _mk_v4_sample(recording_id: str, pos, *, source_dir: str = "d", dataset_id: str = "d3") -> V4Sample:
    """Minimal V4Sample — only the split-relevant fields carry real values."""
    return V4Sample(
        srp_volume=np.zeros((2, 2, 2), dtype=np.float32),
        tdoa_tokens=np.zeros((0, 8), dtype=np.float32),
        context=np.zeros(4, dtype=np.float32),
        x_for_v3=np.zeros(4, dtype=np.float32),
        target_xyz=np.asarray(pos, dtype=np.float32),
        scada=None,
        mode_label=None,
        recording_id=recording_id,
        source_dir=source_dir,
        dataset_id=dataset_id,
    )


class _StubSeg:
    """Duck-typed `_PairedSegment` exposing only what the split reads for
    multi-recording modes (no `_time_split_paired_segment` path is taken)."""

    def __init__(self, dataset_idx: int, recording_id: str, source_dir: str, mode_label) -> None:
        self.dataset_idx = dataset_idx
        self.recording_id = recording_id
        self.source_dir = source_dir
        self.mode_label = mode_label


def _v4_rec_key(s: V4Sample) -> tuple[str, str]:
    return (s.source_dir, s.recording_id)


# ---------------------------------------------------------------------------
# V4 recording-level split
# ---------------------------------------------------------------------------


def test_v4_recording_split_train_val_disjoint_and_lossless() -> None:
    # 10 recordings × 3 windows each.
    samples = [
        _mk_v4_sample(f"r{i}", (i * 0.1, 0.0, 0.0))
        for i in range(10)
        for _ in range(3)
    ]
    train, val = _split_samples_by_recording(samples, val_ratio=0.3, seed=0)

    train_recs = {_v4_rec_key(s) for s in train}
    val_recs = {_v4_rec_key(s) for s in val}
    assert train_recs.isdisjoint(val_recs), "a recording leaked across train/val"
    assert len(train) + len(val) == len(samples), "split dropped or duplicated windows"
    # Every window of a recording must land entirely on one side.
    for rec in train_recs | val_recs:
        on_train = any(_v4_rec_key(s) == rec for s in train)
        on_val = any(_v4_rec_key(s) == rec for s in val)
        assert on_train != on_val, f"recording {rec} split across both sides"


def test_v4_recording_split_is_seed_deterministic() -> None:
    samples = [_mk_v4_sample(f"r{i}", (i * 0.1, 0.0, 0.0)) for i in range(8) for _ in range(2)]
    a = _split_samples_by_recording(samples, 0.25, seed=7)
    b = _split_samples_by_recording(samples, 0.25, seed=7)
    assert {_v4_rec_key(s) for s in a[1]} == {_v4_rec_key(s) for s in b[1]}


# ---------------------------------------------------------------------------
# V4 position-level holdout
# ---------------------------------------------------------------------------


def test_v4_position_holdout_no_position_leaks() -> None:
    positions = [(0.00, 0, 0), (0.10, 0, 0), (0.20, 0, 0), (0.30, 0, 0)]
    # Same position can appear in two recordings — that's exactly the case the
    # position split must keep together (and the recording split would leak).
    samples = []
    for pi, p in enumerate(positions):
        for rec in range(2):
            for _w in range(3):
                samples.append(_mk_v4_sample(f"r{pi}_{rec}", p))

    holdout_positions = [(0.10, 0, 0), (0.30, 0, 0)]
    train, holdout = split_samples_by_position(samples, holdout_positions)

    train_pos = {_position_key(s.target_xyz) for s in train}
    hold_pos = {_position_key(s.target_xyz) for s in holdout}
    assert train_pos.isdisjoint(hold_pos), "a position leaked across train/holdout"
    assert hold_pos == {_position_key(p) for p in holdout_positions}
    assert len(train) + len(holdout) == len(samples)


# ---------------------------------------------------------------------------
# V2/V3 recording-level split + nested threshold-fit disjointness
# ---------------------------------------------------------------------------


def _seg_key(s: _StubSeg) -> tuple[int, str, str]:
    return (s.dataset_idx, s.recording_id, s.source_dir)


def _make_segs() -> list[_StubSeg]:
    segs: list[_StubSeg] = []
    for mode in ("Pump", "Turbine", None):  # None = D3/D4 speed-bucket (unknown mode)
        for r in range(4):  # ≥2 per mode → no single-recording time-split path
            segs.append(_StubSeg(0, f"{mode}_{r}", "d", mode))
    return segs


def test_v2_recording_split_disjoint() -> None:
    train, val = _split_segments_by_recording(_make_segs(), val_ratio=0.5, seed=0)
    assert {_seg_key(s) for s in train}.isdisjoint({_seg_key(s) for s in val})
    assert train and val


def test_v3_threshold_fit_cohort_disjoint_from_reported_val() -> None:
    """Mirror the V3 trainer's nested split: held-out val is split again into a
    threshold-fit cohort and the reportable cohort, and the two must be disjoint
    by recording — this is the leakage path the V3 protocol closes."""
    _train, val = _split_segments_by_recording(_make_segs(), val_ratio=0.5, seed=0)
    val_fit, val_eval = _split_segments_by_recording(val, val_ratio=0.5, seed=1)
    assert {_seg_key(s) for s in val_fit}.isdisjoint({_seg_key(s) for s in val_eval})
    # No recording from the fit cohort reappears in the reportable cohort.
    assert val_fit and val_eval


def test_single_recording_collapses_to_empty_val() -> None:
    """Failure-behaviour contract: a single recording cannot yield a non-empty
    held-out cohort.  `_split_samples_by_recording` returns an empty val here;
    the V3 trainer's nested split turns the equivalent collapse into a hard
    RuntimeError rather than silently fitting and scoring on the same windows."""
    samples = [_mk_v4_sample("only_rec", (0.0, 0.0, 0.0)) for _ in range(5)]
    train, val = _split_samples_by_recording(samples, val_ratio=0.3, seed=0)
    assert len(val) == 0
    assert len(train) == len(samples)


def test_paired_window_split_never_shares_recording_at_extreme_ratios() -> None:
    samples = [_mk_v4_sample(f"r{i}", (i * 0.1, 0.0, 0.0)) for i in range(6) for _ in range(4)]
    for ratio in (0.1, 0.5, 0.9):
        train, val = _split_samples_by_recording(samples, val_ratio=ratio, seed=3)
        assert {_v4_rec_key(s) for s in train}.isdisjoint({_v4_rec_key(s) for s in val})


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
