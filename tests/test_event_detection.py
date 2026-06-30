"""Tests for sliding-window anomaly event detection.

Five invariants we test:

  1. **Single event recovery**: a synthetic score timeline with one
     above-threshold burst produces exactly one event whose duration
     matches the burst.
  2. **Multi-event separation with hysteresis**: two bursts separated
     by a clear gap below `low_threshold` produce two events.
  3. **Hysteresis closure**: a burst that dips between
     `low_threshold` and `high_threshold` stays open (no spurious
     event split).
  4. **Min-duration filter**: an isolated single-window above-
     threshold spike whose total event coverage falls below
     `min_duration_s` is discarded.
  5. **No-event timeline**: a flat-low timeline produces no events.

These are the standard SED (sound event detection) unit-test
invariants — same shape used by the DCASE Task 4 baselines
(Turpault et al. 2019).
"""

from __future__ import annotations

import numpy as np

from src.modeling.anomaly.event_detection import (
    AnomalyEvent,
    detect_events_from_score_timeline,
    summarise_events,
)


def _score_timeline(n: int, stride_s: float = 0.25) -> np.ndarray:
    """Return a flat-1.0 baseline timeline of length n at the given stride."""
    return np.ones(n, dtype=np.float64), np.arange(n, dtype=np.float64) * stride_s


def test_single_event_recovery() -> None:
    scores, times = _score_timeline(100, stride_s=0.25)
    # Inject a 4-window burst at indices 40..43 (t = 10.0..10.75)
    scores[40:44] = [5.0, 6.5, 6.0, 4.5]
    events = detect_events_from_score_timeline(
        scores, times,
        high_threshold=3.0, low_threshold=2.0,
        min_duration_s=0.1, window_seconds=2.0,
        recording_id="rec0", dataset_id="d4",
    )
    assert len(events) == 1
    e = events[0]
    assert isinstance(e, AnomalyEvent)
    # First above-threshold window is at t = 10.0 → t_start = 10.0 − 1.0 = 9.0
    # Last above-low window at t = 10.75 → t_end = 10.75 + 1.0 = 11.75
    assert abs(e.t_start_s - 9.0) < 1e-9
    assert abs(e.t_end_s - 11.75) < 1e-9
    assert abs(e.duration_s - 2.75) < 1e-9
    assert e.peak_score == 6.5
    assert e.peak_t_s == 10.25  # idx 41
    assert e.n_windows == 4
    assert e.recording_id == "rec0"
    assert e.dataset_id == "d4"


def test_multi_event_separation() -> None:
    scores, times = _score_timeline(100, stride_s=0.25)
    # Event 1: indices 20..22 (t = 5.0..5.5)
    scores[20:23] = [5.0, 6.0, 4.5]
    # Event 2: indices 60..62 (t = 15.0..15.5), separated by a long gap below low
    scores[60:63] = [5.0, 6.0, 4.5]
    events = detect_events_from_score_timeline(
        scores, times,
        high_threshold=3.0, low_threshold=2.0,
        min_duration_s=0.1, window_seconds=2.0,
        recording_id="rec0",
    )
    assert len(events) == 2
    assert events[0].peak_t_s == 5.25
    assert events[1].peak_t_s == 15.25
    # Intervals are disjoint.
    assert events[0].t_end_s < events[1].t_start_s


def test_hysteresis_keeps_event_open_through_small_dips() -> None:
    """A burst that dips between low and high thresholds must NOT
    split into multiple events (this is the hysteresis invariant)."""
    scores, times = _score_timeline(100, stride_s=0.25)
    # Single conceptual event with a 2.5-valued dip in the middle.
    # High = 3.0, low = 2.0; the dip is above low so the event stays open.
    scores[30:38] = [5.0, 6.0, 2.5, 2.5, 6.0, 5.0, 2.5, 6.0]
    events = detect_events_from_score_timeline(
        scores, times,
        high_threshold=3.0, low_threshold=2.0,
        min_duration_s=0.1, window_seconds=2.0,
    )
    assert len(events) == 1
    assert events[0].n_windows == 8


def test_min_duration_filter() -> None:
    """An event whose total coverage [t_start, t_end] = window_s
    (single-window burst) is kept iff `min_duration_s ≤ window_s`."""
    scores, times = _score_timeline(50, stride_s=0.25)
    scores[20] = 10.0  # single-window spike
    events_kept = detect_events_from_score_timeline(
        scores, times,
        high_threshold=3.0, low_threshold=2.0,
        min_duration_s=2.0, window_seconds=2.0,
    )
    assert len(events_kept) == 1  # window_seconds = 2.0 == min_duration_s
    events_filtered = detect_events_from_score_timeline(
        scores, times,
        high_threshold=3.0, low_threshold=2.0,
        min_duration_s=3.0,  # > single-window coverage
        window_seconds=2.0,
    )
    assert len(events_filtered) == 0


def test_flat_low_timeline_no_events() -> None:
    scores, times = _score_timeline(50, stride_s=0.25)
    events = detect_events_from_score_timeline(
        scores, times,
        high_threshold=3.0, low_threshold=2.0,
        min_duration_s=0.1, window_seconds=2.0,
    )
    assert events == []


def test_summarise_events_handles_empty() -> None:
    summary = summarise_events([])
    assert summary["n_events"] == 0
    assert np.isnan(summary["mean_duration_s"])
    assert summary["events_per_recording"] == {}


def test_summarise_events_aggregates_correctly() -> None:
    events = [
        AnomalyEvent(
            t_start_s=0.0, t_end_s=2.0, duration_s=2.0,
            peak_score=5.0, peak_t_s=1.0, mean_score=4.0, n_windows=1,
            recording_id="rA", dataset_id="d4",
        ),
        AnomalyEvent(
            t_start_s=5.0, t_end_s=8.0, duration_s=3.0,
            peak_score=7.0, peak_t_s=6.0, mean_score=5.5, n_windows=3,
            recording_id="rA", dataset_id="d4",
        ),
        AnomalyEvent(
            t_start_s=10.0, t_end_s=11.0, duration_s=1.0,
            peak_score=4.0, peak_t_s=10.5, mean_score=3.5, n_windows=2,
            recording_id="rB", dataset_id="d4",
        ),
    ]
    summary = summarise_events(events)
    assert summary["n_events"] == 3
    assert summary["mean_duration_s"] == 2.0  # (2 + 3 + 1) / 3
    assert summary["max_peak_score"] == 7.0
    assert summary["events_per_recording"] == {"rA": 2, "rB": 1}


def test_low_threshold_above_high_threshold_raises() -> None:
    import pytest
    with pytest.raises(ValueError, match="hysteresis"):
        scores, times = _score_timeline(10)
        detect_events_from_score_timeline(
            scores, times,
            high_threshold=3.0, low_threshold=5.0,
            window_seconds=2.0,
        )
