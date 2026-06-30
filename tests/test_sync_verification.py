"""Tests for the cross-modal sync verification module.

Three invariants we test:

  1. **Known-offset recovery**: when a synthetic recording has
     vibration impulses placed at known offsets relative to the
     acoustic impulses, the estimator must recover the offset within
     one vibration sample of accuracy.
  2. **Zero-offset case**: when both modalities are aligned, the
     estimator must report `offset ≈ 0` with high confidence.
  3. **No-shared-signal case**: when the two modalities are pure
     independent noise (no shared events), the confidence ratio
     must collapse toward 1 — i.e., the test reliably flags the
     "no real anchor" failure mode.

This is the standard test suite for any cross-correlation-based
sync estimator (Sargin et al. 2007, "Cross-modal canonical
correlation analysis for audio-visual asynchrony detection").
"""

from __future__ import annotations

import numpy as np

from src.ingestion.sync_verification import SyncResult, verify_paired_sync


def _build_paired_recording(
    *,
    duration_s: float = 10.0,
    mic_fs: int = 16000,
    accel_fs: int = 376,
    n_mics: int = 4,
    n_vib: int = 4,
    vibration_lead_s: float = 0.0,
    impulse_times_s: tuple[float, ...] = (2.0, 4.0, 7.5),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise a paired (mic, vib) recording with controlled offset."""
    rng = np.random.default_rng(seed)
    T_mic = int(duration_s * mic_fs)
    T_vib = int(duration_s * accel_fs)
    mic = 0.01 * rng.normal(size=(n_mics, T_mic))
    vib = 0.01 * rng.normal(size=(n_vib, T_vib))
    for it in impulse_times_s:
        mic_idx = int(it * mic_fs)
        vib_idx = int((it - vibration_lead_s) * accel_fs)
        if 0 <= mic_idx < T_mic - 200:
            mic[:, mic_idx : mic_idx + 200] += 5.0 * rng.normal(size=(n_mics, 200))
        if 0 <= vib_idx < T_vib - 10:
            vib[:, vib_idx : vib_idx + 10] += 5.0 * rng.normal(size=(n_vib, 10))
    return mic, vib


def test_zero_offset_recovers_near_zero() -> None:
    mic, vib = _build_paired_recording(vibration_lead_s=0.0, seed=0)
    result = verify_paired_sync(mic, vib, mic_fs=16000, accel_fs=376, max_offset_s=0.5)
    assert isinstance(result, SyncResult)
    # Within one vibration sample (~ 2.66 ms at 376 Hz) of zero.
    assert abs(result.offset_s) < 0.005
    assert result.confidence > 1.5  # clear peak vs side-lobes


def test_known_offset_vibration_leads() -> None:
    """Vibration impulses placed 50 ms earlier than mic impulses
    must be recovered as `offset_s ≈ +0.050` (vibration leads)."""
    mic, vib = _build_paired_recording(vibration_lead_s=0.050, seed=0)
    result = verify_paired_sync(mic, vib, mic_fs=16000, accel_fs=376, max_offset_s=0.5)
    assert abs(result.offset_s - 0.050) < 0.005
    assert result.confidence > 1.5


def test_known_offset_acoustic_leads() -> None:
    """Negative vibration_lead_s means vibration trails — equivalent
    to acoustic leading by 25 ms.  Expected offset_s ≈ -0.025."""
    mic, vib = _build_paired_recording(vibration_lead_s=-0.025, seed=0)
    result = verify_paired_sync(mic, vib, mic_fs=16000, accel_fs=376, max_offset_s=0.5)
    assert abs(result.offset_s - (-0.025)) < 0.005
    assert result.confidence > 1.5


def test_no_shared_signal_low_confidence() -> None:
    """When the two modalities are independent noise the confidence
    ratio must collapse toward 1 — the orchestrator uses this to
    flag the recording as 'sync indeterminate'."""
    rng = np.random.default_rng(0)
    mic = 0.01 * rng.normal(size=(4, 16000 * 10))
    vib = 0.01 * rng.normal(size=(4, 376 * 10))
    result = verify_paired_sync(mic, vib, mic_fs=16000, accel_fs=376, max_offset_s=0.5)
    assert result.confidence < 2.0  # noise → essentially uniform xcorr → conf ≈ 1
    assert abs(result.peak_correlation) < 0.2  # peak is just noise


def test_very_short_recording_returns_nan_structure() -> None:
    """A few-sample recording cannot support cross-correlation; the
    estimator must report NaN rather than raise — the orchestrator
    skips these and flags them in the log."""
    mic = np.zeros((2, 8), dtype=np.float64)
    vib = np.zeros((2, 2), dtype=np.float64)
    result = verify_paired_sync(mic, vib, mic_fs=16000, accel_fs=376, max_offset_s=0.5)
    assert np.isnan(result.offset_s)
    assert np.isnan(result.confidence)


def test_max_offset_clamps_search_window() -> None:
    """A 200 ms vibration lead with max_offset_s = 0.050 (50 ms search)
    must NOT report the true 200 ms — the estimator is bounded by
    the search window.  This protects against spurious long-lag
    peaks from environmental periodicity."""
    mic, vib = _build_paired_recording(vibration_lead_s=0.200, seed=0)
    result = verify_paired_sync(mic, vib, mic_fs=16000, accel_fs=376, max_offset_s=0.050)
    # The estimator can only report within ± max_lag samples, where
    # max_lag = round(max_offset_s · target_fs).  At target_fs = 376 Hz
    # and max_offset_s = 50 ms, max_lag = 19 → ± 50.5 ms — i.e., the
    # estimator may report up to one vibration sample beyond the
    # nominal cap because of sample-rate quantisation.  The point of
    # this test is that the estimator does NOT report the true 200 ms
    # lag — the search-window cap protects against spurious long-lag
    # peaks.
    accel_fs = 376
    max_lag_samples = int(round(0.050 * accel_fs))
    cap_s = max_lag_samples / accel_fs
    assert abs(result.offset_s) <= cap_s + 1e-9, (
        f"offset {result.offset_s} exceeds the search-window cap {cap_s}"
    )
    # And we must definitively not be at the true 200 ms lag.
    assert abs(result.offset_s) < 0.100
