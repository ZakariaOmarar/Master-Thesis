"""Weak temporal ground-truth for sparse anomalies (RandomFault / knock).

RandomFault recordings contain anomalies (knocks) at *random, sparse* times —
the recording-level ``is_anomaly`` flag does not say WHEN. A knock is an
impulsive transient, so its onset is recoverable from the signal envelope
without any manual annotation. This module turns that impulse structure into
**weak temporal labels**: a list of ``(t_start_s, t_end_s)`` anomaly intervals
per recording.

Two consumers:
  * V4 sample assembly keeps only windows overlapping a derived interval,
    dropping the healthy stretches that would otherwise inherit the
    recording's knock position (label noise).
  * ``v3_real_anomaly_detection`` (in ``event_detection.py``) scores V3's
    detected events against these intervals → precision / recall / F1 /
    onset-timing error.

The detector generalises ``localization.v4_features.find_burst_window`` (which
returns only the single highest-energy burst) to **multiple** bursts via
iterative peak-picking with suppression, so it covers both regimes: sparse
recordings (few intervals) and densely-anomalous recordings (many intervals
that merge into broad spans).
"""

from __future__ import annotations

import numpy as np


def _smoothed_window_energy(data: np.ndarray, n_burst: int) -> np.ndarray:
    """Moving-sum of the multi-channel Hilbert envelope.

    Mirrors ``find_burst_window``'s core: mean ``|hilbert|`` across channels,
    then an O(T) moving-sum of length ``n_burst``. Returns the length
    ``T - n_burst + 1`` array whose index ``i`` is the energy of the
    sub-window ``[i, i + n_burst)``.
    """
    from scipy.signal import hilbert

    env = np.abs(hilbert(data.astype(np.float64), axis=-1)).mean(axis=0)
    csum = np.concatenate([[0.0], np.cumsum(env)])
    return csum[n_burst:] - csum[:-n_burst]


def derive_knock_intervals(
    data: np.ndarray,
    fs: float,
    *,
    burst_seconds: float = 0.10,
    max_events: int = 12,
    noise_floor_mult: float = 3.0,
    merge_gap_s: float = 0.05,
) -> list[tuple[float, float]]:
    """Derive ``(t_start_s, t_end_s)`` anomaly intervals from one signal.

    Args:
      data: ``(n_channels, T)`` raw waveform (mic or accel).
      fs: sample rate (Hz).
      burst_seconds: impulse ring-down width; sets the sub-window length.
      max_events: cap on extracted bursts (guards pathological recordings).
      noise_floor_mult: a burst is accepted only if its windowed energy
        exceeds ``noise_floor_mult × median(window_energy)`` — the median is
        the robust healthy-baseline estimate, so a recording with no impulse
        yields zero intervals (correct for the "is this even anomalous here?"
        question).
      merge_gap_s: adjacent/overlapping intervals closer than this are merged
        (so a broad continuous-anomaly span becomes one interval, not many).

    Returns:
      Sorted, merged list of ``(start_s, end_s)``. Empty when no burst clears
      the noise floor.
    """
    if data.ndim != 2 or data.shape[1] < 2:
        return []
    n_burst = max(2, int(round(burst_seconds * fs)))
    T = int(data.shape[1])
    if n_burst >= T:
        # Whole signal is shorter than one burst → treat the entire span as
        # the anomaly interval (the knock-window case).
        return [(0.0, T / fs)]

    win = _smoothed_window_energy(data, n_burst)
    if win.size == 0:
        return []
    baseline = float(np.median(win))
    # A degenerate baseline of 0 (silent stretches) would make any non-zero
    # energy "infinitely" above floor; fall back to a small fraction of the
    # peak so the threshold stays meaningful.
    threshold = max(noise_floor_mult * baseline, 1e-12 + 0.1 * float(win.max()))

    work = win.copy()
    picks: list[int] = []
    for _ in range(max_events):
        peak = int(np.argmax(work))
        if work[peak] < threshold:
            break
        picks.append(peak)
        # Suppress a ±n_burst neighbourhood so the next pick is a distinct burst.
        lo = max(0, peak - n_burst)
        hi = min(work.size, peak + n_burst)
        work[lo:hi] = -np.inf
    if not picks:
        return []

    intervals = sorted((p / fs, (p + n_burst) / fs) for p in picks)
    # Merge intervals closer than merge_gap_s.
    merged: list[tuple[float, float]] = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s - pe <= merge_gap_s:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def derive_knock_events(
    segment,
    *,
    burst_seconds: float = 0.10,
    max_events: int = 12,
    noise_floor_mult: float = 3.0,
    use_accel: bool = True,
) -> list[tuple[float, float]]:
    """Segment-level wrapper: union of mic- and accel-derived intervals.

    ``segment`` is a ``TestDatasetSegment`` (has ``.segment`` → ``DataSegment``
    with ``mic_data`` / ``accel_data`` / ``*_sample_rate``). A knock can be
    clearer acoustically or structurally depending on the rig coupling, so we
    union both modalities' intervals.

    Returns the merged ``(t_start_s, t_end_s)`` list in recording-relative
    seconds. Empty list means "no impulse found" — caller decides whether to
    treat the whole recording as anomalous (conservative) or skip it.
    """
    ds = getattr(segment, "segment", segment)
    intervals: list[tuple[float, float]] = []
    mic = getattr(ds, "mic_data", None)
    if mic is not None and mic.size:
        intervals += derive_knock_intervals(
            mic, float(ds.mic_sample_rate),
            burst_seconds=burst_seconds, max_events=max_events,
            noise_floor_mult=noise_floor_mult,
        )
    if use_accel:
        accel = getattr(ds, "accel_data", None)
        if accel is not None and accel.size and accel.shape[1] >= 2:
            # Accel rate (4-376 Hz) is far below mic; a 0.10 s burst may be
            # only a few samples, so widen the burst window for the accel
            # pass to keep the moving-sum meaningful.
            accel_burst = max(burst_seconds, 3.0 / float(ds.accel_sample_rate))
            intervals += derive_knock_intervals(
                accel, float(ds.accel_sample_rate),
                burst_seconds=accel_burst, max_events=max_events,
                noise_floor_mult=noise_floor_mult,
            )
    if not intervals:
        return []
    # Final merge across the two modalities' unioned intervals.
    intervals.sort()
    merged: list[tuple[float, float]] = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s - pe <= 0.05:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def window_overlaps_any(
    win_start_s: float, win_end_s: float, intervals: list[tuple[float, float]]
) -> bool:
    """True if ``[win_start_s, win_end_s)`` overlaps any anomaly interval."""
    for s, e in intervals:
        if win_start_s < e and s < win_end_s:
            return True
    return False


__all__ = [
    "derive_knock_events",
    "derive_knock_intervals",
    "window_overlaps_any",
]
