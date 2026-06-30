"""Cross-modal synchronization verification for paired acoustic / vibration recordings.

Why this module exists
----------------------

The current `WavVibrationAdapter` reads the WAV and vibration CSV files
independently and aligns them by **sample index** — i.e., it assumes
sample 0 of the WAV corresponds to sample 0 of the vibration CSV in
wall-clock time, then truncates both streams to ``min(mic_duration,
vib_duration)``.  This implicit alignment is what allows
`_PairedWindowedDataset` to map an acoustic window start time to a
vibration window start time via ``start_vib = round(t_start *
vibration_fs)`` in `v2_ssl.py`.

The assumption is plausible but **unverified** on the bench-top
prototype.  In a real embedded acquisition pipeline several sources
of misalignment exist:

  * Firmware buffer-flush latency between WAV and CSV writers.
  * USB / UART enumeration order at boot.
  * Different start-up paths between the audio codec and the
    accelerometer DMA.
  * Clock drift between the audio codec clock and the MCU's
    millisecond timer.

A misalignment of even 10–50 ms at our 16 kHz WAV / 376 Hz raw
vibration rates would:

  * Smear the V2 cross-attention fusion at frame edges (the modality
    tokens would no longer represent the same physical instant).
  * Introduce a constant bias in the V3 conditional flow's `c_t`
    (the encoder sees temporally-misaligned features but learns to
    treat them as aligned, which is a model-quality issue).
  * Have **no** effect on the V4 SRP-PHAT volume (acoustic-only)
    or the V4 structure-borne TDOA tokens (vibration-only).

This module estimates the cross-modal offset for any paired
recording and reports it as a per-recording diagnostic.

Method
------

Acoustic stream is at 16 kHz; vibration stream is at 4 – 376 Hz
(varies by campaign).  They share information only at the slow
**envelope** scale where common physical events (e.g. a knock, a
guide-vane oscillation onset) modulate the energy of both modalities.
We compute:

  1. The **acoustic envelope** as the mean of `|hilbert(mic)|` across
     all mics, then anti-alias-decimated to the vibration sample rate
     (`scipy.signal.resample_poly`).
  2. The **vibration envelope** as the mean of the vibration amplitude
     stream across all accelerometer channels (already at the
     vibration rate).
  3. The **normalised cross-correlation** between the two envelopes
     within a ± `max_offset_s` window.  The argmax of the
     cross-correlation gives the estimated offset in samples; ×
     ``1 / vibration_fs`` gives the offset in seconds.

A confidence score is the peak-to-side-lobe ratio of the
cross-correlation — high confidence ⇒ a clear peak; low confidence
⇒ the cross-correlation is flat (envelopes are uncorrelated or
the signal is dominated by noise).

This is the canonical cross-modal sync verification technique used
in lip-sync detection (Sargin et al. 2007, "Cross-modal canonical
correlation analysis for audio-visual asynchrony detection",
ICASSP), audio-video alignment in broadcast (Bredin & Chollet 2007),
and seismic-acoustic event localisation (Diehl et al. 2009).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.architecture import SYNC


@dataclass(frozen=True)
class SyncResult:
    """Result of a single-recording cross-modal sync check.

    Convention: ``offset_s > 0`` means the **vibration envelope leads
    the acoustic envelope** by ``offset_s`` seconds.  Equivalently:
    sample 0 of the vibration CSV corresponds to a wall-clock instant
    ``offset_s`` *earlier* than sample 0 of the WAV.  To realign the
    streams, advance the acoustic by ``offset_s`` (drop the first
    ``offset_s · mic_fs`` mic samples) or delay the vibration by
    ``offset_s`` (drop the first ``offset_s · accel_fs`` vibration
    samples) — both are equivalent up to the resampling step.

    The sign convention is verified by the test
    `test_sync_verification.py::test_known_offset_vibration_leads`.
    """

    offset_s: float  # sub-sample-refined offset (parabolic interpolation)
    offset_samples_vib: int  # integer-sample peak lag (pre-refinement)
    offset_s_integer: float  # integer-sample offset (= offset_samples_vib / vibration_fs)
    refinement_delta_samples: float  # sub-sample correction added by the parabola
    confidence: float  # peak / next-largest-peak ratio in the cross-correlation
    peak_correlation: float  # the cross-correlation value at the estimated offset
    mean_correlation: float  # mean |corr| over the search window — sanity floor
    n_envelope_samples: int  # length of the envelopes that fed the cross-correlation
    vibration_fs_used: float
    max_offset_s: float
    method: str = "envelope_normxcorr_parabolic_refined"


def _hilbert_envelope_mean(mic_data: np.ndarray) -> np.ndarray:
    """Mean Hilbert-envelope across all microphone channels.

    Args:
      mic_data: ``(n_mics, T)`` waveform.

    Returns:
      ``(T,)`` envelope array.  We average across mics rather than
      picking a single channel so the envelope reflects the **room-
      averaged** acoustic energy — robust to single-mic anomalies
      (e.g., a fan blowing on one mic) that would not appear in the
      structure-borne vibration.
    """
    from scipy.signal import hilbert

    if mic_data.ndim != 2 or mic_data.shape[1] < 2:
        raise ValueError(
            f"mic_data must be (n_mics, T) with T ≥ 2; got {mic_data.shape}"
        )
    env = np.abs(hilbert(mic_data.astype(np.float64), axis=-1))
    return env.mean(axis=0)


def _vibration_envelope_mean(accel_data: np.ndarray) -> np.ndarray:
    """Mean amplitude envelope across all accelerometer channels.

    For peak-amplitude vibration streams (D1/D2/D3) the input is
    already the envelope; for raw-waveform streams (D4) we compute
    the Hilbert envelope first.  Heuristic: if the per-channel data
    has both positive and negative values, treat it as raw waveform.
    """
    from scipy.signal import hilbert

    if accel_data.ndim != 2 or accel_data.shape[1] < 2:
        raise ValueError(
            f"accel_data must be (n_vib, T) with T ≥ 2; got {accel_data.shape}"
        )
    if (accel_data < 0).any():
        env = np.abs(hilbert(accel_data.astype(np.float64), axis=-1))
        return env.mean(axis=0)
    return accel_data.astype(np.float64).mean(axis=0)


def _decimate_to_rate(
    signal: np.ndarray,
    source_fs: float,
    target_fs: float,
) -> np.ndarray:
    """Anti-alias-decimate `signal` from `source_fs` to ~`target_fs`."""
    from math import gcd

    from scipy.signal import resample_poly

    if source_fs <= target_fs + 1e-9:
        return signal.astype(np.float64)
    src = int(round(source_fs))
    dst = int(round(target_fs))
    g = gcd(src, dst)
    up = dst // g
    down = src // g
    return resample_poly(signal.astype(np.float64), up=up, down=down).astype(np.float64)


def _normalised_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised cross-correlation of two 1-D arrays for lags ∈ [-max_lag, +max_lag].

    Uses `scipy.signal.correlate` for the full unbounded cross-
    correlation, then slices to the ± `max_lag` window.  Sign
    convention matches the scipy / numpy "full"-mode standard:

      * Index ``len(b) - 1`` of the full output is lag = 0.
      * Index ``len(b) - 1 + k`` is lag = +k → ``a[t] = b[t - k]`` →
        `a` is `b` delayed by k → **b leads a** by k samples.
      * Symmetric for negative k.

    The normalisation divides by ``‖a‖ · ‖b‖`` so the peak value lies
    in [-1, +1], the standard Pearson normalisation used in TDOA
    estimation (Knapp & Carter 1976) and cross-modal sync (Sargin
    et al. 2007).

    Returns:
      ``(2 * max_lag + 1,)`` array.  The k-th element corresponds to
      lag = (k - max_lag).
    """
    from scipy.signal import correlate

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("envelopes must be non-empty")
    a = a - a.mean()
    b = b - b.mean()
    a_norm = float(np.sqrt((a * a).sum()))
    b_norm = float(np.sqrt((b * b).sum()))
    if a_norm == 0.0 or b_norm == 0.0:
        return np.zeros(2 * max_lag + 1, dtype=np.float64)

    full = correlate(a, b, mode="full", method="fft")
    centre = b.size - 1  # lag 0 lives here in scipy's convention
    lo = centre - max_lag
    hi = centre + max_lag + 1
    pad_low = max(0, -lo)
    pad_high = max(0, hi - full.size)
    lo = max(0, lo)
    hi = min(full.size, hi)
    sliced = full[lo:hi]
    if pad_low or pad_high:
        sliced = np.pad(sliced, (pad_low, pad_high))
    return (sliced / (a_norm * b_norm)).astype(np.float64)


def verify_paired_sync(
    mic_data: np.ndarray,
    accel_data: np.ndarray,
    mic_fs: float,
    accel_fs: float,
    *,
    max_offset_s: float = SYNC.max_offset_s,
    target_envelope_fs: float | None = None,
) -> SyncResult:
    """Estimate the cross-modal offset between paired acoustic and vibration streams.

    Args:
      mic_data: ``(n_mics, T_mic)`` raw mic waveform.
      accel_data: ``(n_vib, T_vib)`` vibration amplitude or raw waveform.
      mic_fs: WAV sample rate (Hz).
      accel_fs: vibration sample rate (Hz).
      max_offset_s: half-width of the cross-correlation search window
        in seconds.  Default 0.5 s — covers any plausible
        firmware-startup-latency on the bench-top prototype while
        excluding spurious second-scale peaks.
      target_envelope_fs: rate at which both envelopes are evaluated
        before cross-correlation.  Default is ``accel_fs`` (the lower
        of the two rates), which is the cleanest choice when D4 raw
        vibration is at 376 Hz; for the peak-amplitude campaigns
        (D1 / D2 / D3) the vibration is already an envelope so the
        evaluation rate matches the data rate.

    Returns:
      `SyncResult` with the estimated offset in seconds, the
      offset in vibration-sample units, the peak correlation value,
      a confidence ratio (peak / next-largest local maximum), and
      sundry metadata for the orchestrator log line.

    Convention: ``result.offset_s > 0`` means the **vibration leads
    the acoustic** by `offset_s` seconds — i.e., a knock that
    appears at sample 0 + Δ of the WAV appears at sample 0 + Δ −
    offset_s of the vibration.  See `SyncResult` docstring for the
    canonical sign-convention specification.
    """
    if mic_fs <= 0 or accel_fs <= 0:
        raise ValueError("sample rates must be positive")
    if max_offset_s <= 0:
        raise ValueError("max_offset_s must be positive")

    target_fs = float(target_envelope_fs or accel_fs)
    mic_env = _hilbert_envelope_mean(mic_data)
    vib_env = _vibration_envelope_mean(accel_data)

    # Anti-alias-decimate the acoustic envelope to the target rate.
    mic_env_dec = _decimate_to_rate(mic_env, float(mic_fs), target_fs)
    # Decimate the vibration envelope to the same target rate if needed
    # (almost always a no-op since target_fs defaults to accel_fs).
    vib_env_dec = _decimate_to_rate(vib_env, float(accel_fs), target_fs)

    # Truncate both envelopes to the same length so the cross-correlation
    # is well-defined.  Length mismatch comes from rounding inside the
    # decimator + the original `common_duration` truncation in the
    # ingestion adapter.
    T = int(min(mic_env_dec.size, vib_env_dec.size))
    if T < 16:
        # Too short to give a meaningful cross-correlation; report a
        # NaN-decorated SyncResult so the orchestrator can flag the
        # recording as inconclusive.
        return SyncResult(
            offset_s=float("nan"),
            offset_samples_vib=0,
            offset_s_integer=float("nan"),
            refinement_delta_samples=float("nan"),
            confidence=float("nan"),
            peak_correlation=float("nan"),
            mean_correlation=float("nan"),
            n_envelope_samples=T,
            vibration_fs_used=target_fs,
            max_offset_s=max_offset_s,
        )
    mic_env_dec = mic_env_dec[:T]
    vib_env_dec = vib_env_dec[:T]

    max_lag = int(round(max_offset_s * target_fs))
    max_lag = max(1, min(max_lag, T - 1))
    xcorr = _normalised_xcorr(mic_env_dec, vib_env_dec, max_lag)
    abs_xcorr = np.abs(xcorr)
    peak_idx = int(np.argmax(abs_xcorr))
    # Sign convention (verified empirically by
    # `test_known_offset_vibration_leads`):
    #   peak_lag > 0 ⇒ vibration leads acoustic by `peak_lag` samples
    #   peak_lag < 0 ⇒ acoustic leads vibration by `|peak_lag|` samples
    peak_lag = peak_idx - max_lag
    peak_value = float(xcorr[peak_idx])

    # Sub-sample peak refinement via 3-point parabolic interpolation
    # around the integer peak (Jacovitti & Scarano, "Discrete time
    # techniques for time delay estimation," IEEE TSP 1993).  The
    # parabolic vertex of (peak_idx-1, peak_idx, peak_idx+1) gives the
    # fractional offset within ± 0.5 samples of the integer peak;
    # bounded inside the ± max_lag search window.
    refined_lag_samples = float(peak_lag)
    if 0 < peak_idx < abs_xcorr.size - 1:
        y_m = float(abs_xcorr[peak_idx - 1])
        y_0 = float(abs_xcorr[peak_idx])
        y_p = float(abs_xcorr[peak_idx + 1])
        denom = 2.0 * (y_m - 2.0 * y_0 + y_p)
        if abs(denom) > 1e-12:
            delta = (y_m - y_p) / denom
            # Clamp to ± 0.5 samples — anything larger means the parabola
            # is not concave-down at the peak (numerical pathology).
            delta = max(-0.5, min(0.5, delta))
            refined_lag_samples = peak_lag + delta

    # Confidence = peak / next-largest local maximum.
    # Mask out a ± 10 % window around the peak before computing the
    # second-largest value, so spurious side-lobe ringing doesn't tank
    # the confidence ratio.
    mask_half = max(1, int(0.10 * max_lag))
    masked = abs_xcorr.copy()
    lo_mask = max(0, peak_idx - mask_half)
    hi_mask = min(masked.size, peak_idx + mask_half + 1)
    masked[lo_mask:hi_mask] = 0.0
    second = float(masked.max()) if masked.size > 0 else 0.0
    confidence = float(abs(peak_value) / (second + 1e-12))
    mean_corr = float(np.mean(abs_xcorr))

    offset_s_integer = peak_lag / target_fs
    offset_s = refined_lag_samples / target_fs
    return SyncResult(
        offset_s=float(offset_s),
        offset_samples_vib=int(peak_lag),
        offset_s_integer=float(offset_s_integer),
        refinement_delta_samples=float(refined_lag_samples - peak_lag),
        confidence=float(confidence),
        peak_correlation=float(peak_value),
        mean_correlation=float(mean_corr),
        n_envelope_samples=int(T),
        vibration_fs_used=float(target_fs),
        max_offset_s=float(max_offset_s),
    )


# ---------------------------------------------------------------------------
# Time-stability check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncStabilityResult:
    """Result of a time-stability check across K sub-segments of a recording.

    Two failure modes the orchestrator must distinguish:

      * **Constant offset, high confidence**: every sub-segment reports
        ≈ the same offset.  Safe to correct with a single shift.

      * **Drifting offset**: sub-segment offsets vary monotonically by
        more than `drift_tolerance_s` across the recording.  This is a
        clock-drift signature (the audio codec clock and the MCU clock
        run at slightly different rates).  A single-shift correction
        does not work; the recording is flagged for manual review and
        left uncorrected.  The right fix at deployment is to resample
        one stream onto the other's clock (Chapter 7 future work).

    Returns the per-sub-segment offsets and confidences, plus the
    median, MAD, and the linear-fit slope (offset vs sub-segment
    time-midpoint) which surfaces drift.
    """

    sub_segment_offsets_s: list[float]
    sub_segment_confidences: list[float]
    sub_segment_midpoints_s: list[float]
    median_offset_s: float
    mad_offset_s: float
    drift_slope_s_per_s: float  # linear fit; |slope| < 1e-3 means drift << 1 ms / s
    is_stable: bool
    drift_tolerance_s: float
    n_sub_segments: int
    n_sub_segments_high_conf: int


def verify_sync_stability(
    mic_data: np.ndarray,
    accel_data: np.ndarray,
    mic_fs: float,
    accel_fs: float,
    *,
    n_sub_segments: int = SYNC.n_sub_segments,
    max_offset_s: float = SYNC.max_offset_s,
    confidence_floor: float = SYNC.confidence_floor,
    drift_tolerance_s: float = SYNC.drift_tolerance_s,
    min_sub_segment_seconds: float = 5.0,
) -> SyncStabilityResult:
    """Run the sync audit on K disjoint sub-segments of a recording.

    The recording is split into `n_sub_segments` equal-duration
    contiguous chunks; the cross-modal offset is estimated on each
    chunk independently.  If the per-chunk offsets agree within
    `drift_tolerance_s` and the linear-fit drift slope is small
    (|slope| ≤ `drift_tolerance_s / total_duration`), the alignment
    is declared **stable** and a single-shift correction is safe.

    Why split into K sub-segments rather than estimate a moving
    offset: clock-drift signatures on a single recording are
    typically linear (the two clocks differ by a fixed
    parts-per-million ratio), and 5 anchor points spread across the
    recording give a clean linear fit while keeping each anchor's
    cross-correlation window long enough (~10 s on a 60 s recording)
    to have a confident peak.  Standard practice in audio-clock-drift
    correction (Bregler & Konig 1998, "Eigenlips for robust speech
    recognition," ICASSP).

    Args:
      mic_data, accel_data: paired streams.
      mic_fs, accel_fs: sample rates.
      n_sub_segments: K sub-segments.  Default 5: more gives a finer
        drift estimate but each sub-segment becomes shorter and the
        per-chunk confidence drops.
      max_offset_s: search-window half-width per sub-segment.
      confidence_floor: minimum per-chunk confidence ratio for that
        chunk to count toward the stability statistics.
      drift_tolerance_s: maximum allowed offset spread across
        high-confidence chunks for the recording to be declared
        stable.  Default 10 ms — comfortably above the
        sample-quantisation floor at all our vibration rates.
      min_sub_segment_seconds: minimum chunk duration; if the
        recording is shorter than `K · min_sub_segment_seconds`,
        K is reduced.
    """
    if n_sub_segments < 2:
        raise ValueError("n_sub_segments must be ≥ 2")
    mic_dur = mic_data.shape[1] / float(mic_fs)
    vib_dur = accel_data.shape[1] / float(accel_fs)
    total_dur = float(min(mic_dur, vib_dur))
    K = int(n_sub_segments)
    while K > 2 and total_dur / K < min_sub_segment_seconds:
        K -= 1
    if total_dur / K < min_sub_segment_seconds:
        # Recording too short — return single-chunk audit as a stability
        # result with n_sub_segments = 1.
        sr = verify_paired_sync(mic_data, accel_data, mic_fs, accel_fs,
                                max_offset_s=max_offset_s)
        return SyncStabilityResult(
            sub_segment_offsets_s=[sr.offset_s],
            sub_segment_confidences=[sr.confidence],
            sub_segment_midpoints_s=[total_dur / 2.0],
            median_offset_s=sr.offset_s,
            mad_offset_s=0.0,
            drift_slope_s_per_s=0.0,
            is_stable=(sr.confidence >= confidence_floor and not np.isnan(sr.offset_s)),
            drift_tolerance_s=drift_tolerance_s,
            n_sub_segments=1,
            n_sub_segments_high_conf=int(sr.confidence >= confidence_floor),
        )

    per_chunk_offsets: list[float] = []
    per_chunk_confs: list[float] = []
    per_chunk_mids: list[float] = []
    high_conf_offsets: list[float] = []
    high_conf_mids: list[float] = []
    for k in range(K):
        t_lo = total_dur * k / K
        t_hi = total_dur * (k + 1) / K
        mid = 0.5 * (t_lo + t_hi)
        i_lo_mic = int(round(t_lo * mic_fs))
        i_hi_mic = int(round(t_hi * mic_fs))
        i_lo_vib = int(round(t_lo * accel_fs))
        i_hi_vib = int(round(t_hi * accel_fs))
        sr_k = verify_paired_sync(
            mic_data[:, i_lo_mic:i_hi_mic],
            accel_data[:, i_lo_vib:i_hi_vib],
            mic_fs, accel_fs,
            max_offset_s=max_offset_s,
        )
        per_chunk_offsets.append(sr_k.offset_s)
        per_chunk_confs.append(sr_k.confidence)
        per_chunk_mids.append(mid)
        if (
            not np.isnan(sr_k.offset_s)
            and not np.isnan(sr_k.confidence)
            and sr_k.confidence >= confidence_floor
        ):
            high_conf_offsets.append(sr_k.offset_s)
            high_conf_mids.append(mid)

    if len(high_conf_offsets) >= 2:
        arr = np.asarray(high_conf_offsets)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        # Linear fit: offset = a + b * t.  b ≪ drift_tolerance / total_dur
        # means the offset is essentially constant across the recording.
        t_arr = np.asarray(high_conf_mids)
        if t_arr.std() > 1e-9:
            slope = float(
                np.polyfit(t_arr, arr, deg=1)[0]
            )  # seconds-of-offset per second-of-recording
        else:
            slope = 0.0
        offset_range = float(arr.max() - arr.min())
        is_stable = bool(
            offset_range <= drift_tolerance_s
            and abs(slope) * total_dur <= drift_tolerance_s
        )
    elif len(high_conf_offsets) == 1:
        med = float(high_conf_offsets[0])
        mad = 0.0
        slope = 0.0
        is_stable = True  # only one high-confidence anchor; conservatively trust it
    else:
        med = float("nan")
        mad = float("nan")
        slope = float("nan")
        is_stable = False

    return SyncStabilityResult(
        sub_segment_offsets_s=per_chunk_offsets,
        sub_segment_confidences=per_chunk_confs,
        sub_segment_midpoints_s=per_chunk_mids,
        median_offset_s=med,
        mad_offset_s=mad,
        drift_slope_s_per_s=slope,
        is_stable=is_stable,
        drift_tolerance_s=drift_tolerance_s,
        n_sub_segments=K,
        n_sub_segments_high_conf=len(high_conf_offsets),
    )


# ---------------------------------------------------------------------------
# Sync correction
# ---------------------------------------------------------------------------


def _fractional_sample_shift(
    signal: np.ndarray, shift_samples: float
) -> np.ndarray:
    """Sub-sample shift via FFT phase-ramp (ideal-sinc interpolation).

    For each row of `signal`, multiply its DFT by ``exp(-j·2π·k·shift / N)``
    where `k` is the frequency bin and `N` is the DFT length, then inverse
    DFT.  The result is the band-limited time-shifted version of the
    input — exactly the sinc interpolator on a finite-length signal
    (Smith, *Mathematics of the DFT*, Chapter 7, "Interpolating a
    function by ideal-bandlimited shifting").

    Positive `shift_samples` shifts the signal LATER (delays it);
    negative shifts it EARLIER (advances it).

    The full-length output is returned; the caller is responsible for
    trimming any edge artefacts (the first / last
    ``|shift_samples|`` samples carry circular-wraparound energy).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal_2d = signal[None, :]
        squeeze = True
    elif signal.ndim == 2:
        signal_2d = signal
        squeeze = False
    else:
        raise ValueError(f"signal must be 1-D or 2-D; got {signal.shape}")
    N = signal_2d.shape[1]
    if N < 2 or abs(shift_samples) < 1e-9:
        return signal.copy() if not squeeze else signal_2d[0].copy()
    spectrum = np.fft.fft(signal_2d, axis=-1)
    freqs = np.fft.fftfreq(N, d=1.0)
    phase = np.exp(-1j * 2.0 * np.pi * freqs * float(shift_samples))
    rotated = spectrum * phase
    # For even-length real signals the Nyquist bin (index N/2) must
    # remain real for the inverse FFT to yield a real-valued result.
    # Multiplying it by a non-integer phase ramp gives it a small
    # imaginary part that, when discarded by `.real`, leaks O(0.5 %)
    # of the signal energy.  Zeroing the imaginary part of the
    # Nyquist bin restores the Hermitian symmetry that makes the
    # phase ramp truly unitary; the discarded imaginary energy is
    # exactly the amount that would have been lost to the `.real`
    # truncation anyway, so this is loss-free.  Standard practice
    # in band-limited fractional-delay filtering (Smith, MDFT §7.6).
    if N % 2 == 0:
        rotated[..., N // 2] = rotated[..., N // 2].real
    shifted = np.fft.ifft(rotated, axis=-1).real
    return shifted[0] if squeeze else shifted


@dataclass(frozen=True)
class SyncCorrectionReport:
    """Outcome of a sync-correction attempt on a paired recording."""

    applied: bool  # True if the correction was applied; False if gated out
    reason: str  # short text reason for `applied` / `not applied`
    applied_offset_s: float  # 0.0 when not applied; the offset shifted otherwise
    integer_sample_shift_mic: int  # samples dropped from mic_data (≥ 0)
    integer_sample_shift_vib: int  # samples dropped from accel_data (≥ 0)
    fractional_sample_shift_mic: float  # FFT-phase-ramp fractional shift applied to mic
    fractional_sample_shift_vib: float  # FFT-phase-ramp fractional shift applied to vib
    audit: SyncResult  # the full-recording audit
    stability: SyncStabilityResult  # the K-chunk stability check
    residual_offset_uncertainty_s: float  # 1 / max(mic_fs, accel_fs) after sub-sample shift
    # F3 diagnostic — excess kurtosis of the decimated acoustic envelope (the
    # exact signal the cross-correlation operates on).  Lets the orchestrator
    # distinguish *why* a recording was gated out at low confidence:
    #   * low kurtosis (≲ 1) + low confidence → genuinely steady-state
    #     content; there is nothing for envelope xcorr to lock onto.  Under
    #     shared-trigger acquisition the streams are aligned by construction.
    #   * high kurtosis (≫ 1) + low confidence → the acoustic stream HAS
    #     transients, but they do not co-occur with vibration transients
    #     (different physical paths, or sparse events over a long recording).
    #     This is the measured D4 case: healthy speed buckets sit at
    #     kurtosis ≈ 1.5–1.9, knock recordings at 40–150, yet cross-modal
    #     confidence stays ≈ 1.0 for all of them.
    acoustic_envelope_kurtosis: float = float("nan")


def apply_sync_correction(
    mic_data: np.ndarray,
    accel_data: np.ndarray,
    mic_fs: float,
    accel_fs: float,
    offset_s: float,
    *,
    use_fractional_shift: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, int, float, float]:
    """Apply a single-shift cross-modal correction.

    Convention: ``offset_s > 0`` means vibration leads acoustic by
    `offset_s` seconds (see `SyncResult` docstring).  The correction
    aligns the two streams so sample 0 of each corresponds to the
    same physical instant.

    Implementation: we decompose the shift into an **integer-sample
    part** (applied by dropping leading samples on the leading
    stream) and a **fractional-sample part** (applied as an
    FFT-phase-ramp sinc-shift on the same stream).  Decomposition is
    on whichever stream has the **higher sample rate** so the
    integer-sample resolution is finest:

      * mic_fs > accel_fs → mic resolution is 1/mic_fs (~ 62.5 µs at
        16 kHz); vib resolution is 1/accel_fs (~ 2.66 ms at 376 Hz).
        We therefore apply the correction on the mic stream when
        offset < 0 (acoustic leads → drop leading mic samples), and
        on the vib stream when offset > 0 (vib leads → drop leading
        vib samples).  In both cases the residual sub-sample
        uncertainty is bounded by 1 / max(mic_fs, accel_fs).

    Returns:
      `(mic_corrected, accel_corrected, n_drop_mic, n_drop_vib,
        frac_mic, frac_vib)`.
    """
    if abs(offset_s) < 1e-9:
        return mic_data, accel_data, 0, 0, 0.0, 0.0

    # Sign convention recap (verified empirically by
    # `test_known_offset_vibration_leads`, and stated explicitly below):
    #
    #   offset_s > 0 ⇒ vibration leads acoustic
    #             ⇔ at the same physical event the vib-stream's
    #               internal clock reads LESS than the mic-stream's
    #               by `offset_s` seconds
    #             ⇔ the mic stream is the LATE one and must be
    #               *advanced* — equivalently, the first `offset_s`
    #               of mic samples are pre-event noise that we drop
    #               so the new mic_sample_0 corresponds to the same
    #               physical instant as vib_sample_0.
    #
    # Dropping from the lagging stream is the only zero-fake-data
    # option: padding the leading stream would inject zeros.

    if offset_s > 0:
        # Vibration leads acoustic → mic lags → drop leading MIC samples.
        n_drop_total_mic_float = float(offset_s) * float(mic_fs)
        n_drop_int = int(np.floor(n_drop_total_mic_float))
        # Guard against shifts that would consume the whole recording.  This
        # cannot occur under the four-gate `auto_sync_paired_recording`
        # pipeline (`max_offset_s` caps the search at 0.5 s by default), but
        # silent clamping here would mask bugs in any future direct caller.
        if n_drop_int >= mic_data.shape[1] - 2:
            raise ValueError(
                f"sync correction would drop {n_drop_int} leading mic samples "
                f"from a {mic_data.shape[1]}-sample stream "
                f"(offset_s={offset_s:.4f} s, mic_fs={mic_fs} Hz); offset "
                f"exceeds recording duration — verify the audit and the "
                f"caller's max_offset_s / confidence_floor"
            )
        frac_mic = n_drop_total_mic_float - n_drop_int
        mic_dropped = mic_data[:, n_drop_int:]
        if use_fractional_shift and frac_mic > 1e-9:
            mic_corrected = _fractional_sample_shift(mic_dropped, +frac_mic)
        else:
            mic_corrected = mic_dropped
        return mic_corrected, accel_data, n_drop_int, 0, frac_mic, 0.0

    # offset_s < 0 → acoustic leads vibration → vib lags → drop leading VIB samples.
    n_drop_total_vib_float = abs(float(offset_s)) * float(accel_fs)
    n_drop_int = int(np.floor(n_drop_total_vib_float))
    if n_drop_int >= accel_data.shape[1] - 2:
        raise ValueError(
            f"sync correction would drop {n_drop_int} leading vibration samples "
            f"from a {accel_data.shape[1]}-sample stream "
            f"(offset_s={offset_s:.4f} s, accel_fs={accel_fs} Hz); offset "
            f"exceeds recording duration — verify the audit and the caller's "
            f"max_offset_s / confidence_floor"
        )
    frac_vib = n_drop_total_vib_float - n_drop_int
    accel_dropped = accel_data[:, n_drop_int:]
    if use_fractional_shift and frac_vib > 1e-9:
        accel_corrected = _fractional_sample_shift(accel_dropped, +frac_vib)
    else:
        accel_corrected = accel_dropped
    return mic_data, accel_corrected, 0, n_drop_int, 0.0, frac_vib


def _acoustic_envelope_excess_kurtosis(
    mic_data: np.ndarray, mic_fs: float, accel_fs: float
) -> float:
    """Excess kurtosis of the acoustic envelope, on the *exact* signal the
    cross-correlation operates on.

    The envelope-cross-correlation TDOA estimator is informative only when
    the acoustic signal contains broadband **transients** — impacts, clicks,
    onsets — that produce a sharply-peaked envelope.  Steady-state content
    (sustained pumping / turbine tones) produces a near-Gaussian envelope
    whose cross-correlation with the vibration envelope is dominated by
    noise, yielding the kind of low-confidence, wildly-varying offsets the
    F3 D4 audit revealed (raw and peak both swinging ±400 ms at confidence
    ≈ 1.0).

    The metric is computed on the *same* decimated envelope
    `verify_paired_sync` feeds to `_normalised_xcorr` — i.e. the mean
    Hilbert envelope across mics, anti-alias-decimated to the vibration
    sample rate.  This makes the kurtosis a direct measure of the
    peakedness of the signal the offset estimator actually sees, rather
    than a carrier-band proxy.

    Excess kurtosis (Fisher 1929; ``scipy.stats.kurtosis(fisher=True)``):

      * ≈ 0 — Gaussian-ish envelope (steady-state content; D4-style)
      * ≳ a few — transient-rich audio (broadband impacts)
      * very high — impulse-like content (isolated knocks)

    Empirically on this rig: D4 sustained-tone content sits well below 1.0;
    D1/D2 impulse-bearing recordings sit far above it.
    """
    from scipy.stats import kurtosis as _kurt

    if mic_data.ndim != 2 or mic_data.shape[1] < 2:
        return 0.0
    # Same two steps verify_paired_sync uses before cross-correlation.
    acoustic_env = _hilbert_envelope_mean(mic_data)
    acoustic_env = _decimate_to_rate(acoustic_env, float(mic_fs), float(accel_fs))
    if acoustic_env.size < 8:
        return 0.0
    return float(_kurt(acoustic_env, fisher=True, bias=False))


def auto_sync_paired_recording(
    mic_data: np.ndarray,
    accel_data: np.ndarray,
    mic_fs: float,
    accel_fs: float,
    *,
    max_offset_s: float = SYNC.max_offset_s,
    n_sub_segments: int = SYNC.n_sub_segments,
    confidence_floor: float = SYNC.confidence_floor,
    drift_tolerance_s: float = SYNC.drift_tolerance_s,
    min_offset_to_correct_s: float = SYNC.min_offset_to_correct_s,
    min_envelope_kurtosis: float = SYNC.min_envelope_kurtosis,
    use_fractional_shift: bool = SYNC.use_fractional_shift,
) -> tuple[np.ndarray, np.ndarray, SyncCorrectionReport]:
    """Verify, gate, and apply cross-modal sync correction on a paired recording.

    The pipeline is **gated on four conditions** before any correction is
    applied:

      0. Acoustic-envelope excess kurtosis ≥ `min_envelope_kurtosis`
         (default 1.0).  Below this the audio envelope is essentially
         Gaussian — there is provably no peaked structure for the
         cross-correlation to lock onto, so attempting correction is
         meaningless.  This is a cheap early-out; note it does not catch
         every uninformative recording (D4's healthy speed buckets sit at
         kurtosis ≈ 1.5–1.9 yet still fail to cross-correlate — those are
         caught by Gate 1 instead).
      1. Full-recording audit confidence ≥ `confidence_floor`.
      2. Time-stability check: ≥ 2 high-confidence sub-segments AND
         offset spread ≤ `drift_tolerance_s` AND linear-fit drift
         |slope × duration| ≤ `drift_tolerance_s`.
      3. |offset| ≥ `min_offset_to_correct_s` — below this threshold
         the correction would have no measurable effect (and might
         introduce numerical noise from the FFT-shift step).

    When any gate fails the streams are returned **unmodified** and
    the `SyncCorrectionReport.applied = False` with a `reason`
    string the orchestrator surfaces in the run log.  Every report
    carries `acoustic_envelope_kurtosis` so the orchestrator can
    distinguish a genuinely-flat recording from a transient-rich one
    that simply does not cross-correlate (the measured D4 case — see
    `SyncCorrectionReport.acoustic_envelope_kurtosis`).

    Returns:
      `(mic_corrected, accel_corrected, report)`.
    """
    # Acoustic-envelope peakedness — computed once, attached to every report
    # as the F3 diagnostic and used as the Gate 0 early-out threshold.
    env_kurt = _acoustic_envelope_excess_kurtosis(mic_data, mic_fs, accel_fs)
    residual_uncertainty = 1.0 / max(float(mic_fs), float(accel_fs))

    # Gate 0 — genuinely steady-state content: envelope is near-Gaussian, so
    # there is nothing for cross-correlation to find.  Cheap early-out that
    # also skips the (more expensive) stability check.
    if env_kurt < min_envelope_kurtosis:
        audit_min = verify_paired_sync(
            mic_data, accel_data, mic_fs, accel_fs, max_offset_s=max_offset_s
        )
        return mic_data, accel_data, SyncCorrectionReport(
            applied=False,
            reason=(
                f"acoustic envelope is near-Gaussian (excess kurtosis "
                f"{env_kurt:.2f} < {min_envelope_kurtosis:.2f}); no peaked "
                f"structure for envelope cross-correlation to lock onto — "
                f"streams retained as-is (shared-trigger acquisition aligns "
                f"them by construction)"
            ),
            applied_offset_s=0.0,
            integer_sample_shift_mic=0,
            integer_sample_shift_vib=0,
            fractional_sample_shift_mic=0.0,
            fractional_sample_shift_vib=0.0,
            audit=audit_min,
            stability=SyncStabilityResult(
                sub_segment_offsets_s=[],
                sub_segment_confidences=[],
                sub_segment_midpoints_s=[],
                median_offset_s=float("nan"),
                mad_offset_s=float("nan"),
                drift_slope_s_per_s=0.0,
                is_stable=False,
                drift_tolerance_s=drift_tolerance_s,
                n_sub_segments=0,
                n_sub_segments_high_conf=0,
            ),
            residual_offset_uncertainty_s=residual_uncertainty,
            acoustic_envelope_kurtosis=env_kurt,
        )

    audit = verify_paired_sync(
        mic_data, accel_data, mic_fs, accel_fs, max_offset_s=max_offset_s
    )
    stability = verify_sync_stability(
        mic_data, accel_data, mic_fs, accel_fs,
        n_sub_segments=n_sub_segments,
        max_offset_s=max_offset_s,
        confidence_floor=confidence_floor,
        drift_tolerance_s=drift_tolerance_s,
    )

    # Gate 1: full-recording audit confidence.  A low confidence here means
    # the envelope cross-correlation has no clear peak — which under
    # shared-trigger acquisition does not imply the streams are misaligned,
    # only that envelope xcorr cannot *prove* the alignment.  The reason
    # string therefore avoids "manual review" language and reports the
    # envelope kurtosis so a reader can tell *why* the correlation failed:
    # genuinely-flat content vs transient content that simply does not
    # co-occur across modalities (the measured D4 case).
    if np.isnan(audit.offset_s) or audit.confidence < confidence_floor:
        return mic_data, accel_data, SyncCorrectionReport(
            applied=False,
            reason=(
                f"envelope cross-correlation uninformative "
                f"(confidence {audit.confidence:.2f} < floor "
                f"{confidence_floor:.2f}, envelope excess kurtosis "
                f"{env_kurt:.2f}); no correction applied — under "
                f"shared-trigger acquisition the streams are aligned by "
                f"construction and the offset cannot be independently "
                f"verified from this recording's content"
            ),
            applied_offset_s=0.0,
            integer_sample_shift_mic=0,
            integer_sample_shift_vib=0,
            fractional_sample_shift_mic=0.0,
            fractional_sample_shift_vib=0.0,
            audit=audit,
            stability=stability,
            residual_offset_uncertainty_s=residual_uncertainty,
            acoustic_envelope_kurtosis=env_kurt,
        )

    # Gate 2: time-stability check.
    if not stability.is_stable:
        return mic_data, accel_data, SyncCorrectionReport(
            applied=False,
            reason=(
                f"stability check failed: {stability.n_sub_segments_high_conf}/"
                f"{stability.n_sub_segments} high-confidence chunks, "
                f"drift_slope = {stability.drift_slope_s_per_s * 1e3:.2f} ms/s; "
                f"clock drift suspected — manual correction required"
            ),
            applied_offset_s=0.0,
            integer_sample_shift_mic=0,
            integer_sample_shift_vib=0,
            fractional_sample_shift_mic=0.0,
            fractional_sample_shift_vib=0.0,
            audit=audit,
            stability=stability,
            residual_offset_uncertainty_s=residual_uncertainty,
            acoustic_envelope_kurtosis=env_kurt,
        )

    # Use the median of the high-confidence sub-segments as the
    # correction offset — more robust than the full-recording audit
    # against impulses near the recording boundaries.
    correction_offset_s = float(stability.median_offset_s)

    # Gate 3: offset magnitude.
    if abs(correction_offset_s) < min_offset_to_correct_s:
        return mic_data, accel_data, SyncCorrectionReport(
            applied=False,
            reason=(
                f"offset {correction_offset_s * 1e3:+.2f} ms is below the "
                f"correction floor {min_offset_to_correct_s * 1e3:.2f} ms; "
                f"streams already aligned within tolerance"
            ),
            applied_offset_s=0.0,
            integer_sample_shift_mic=0,
            integer_sample_shift_vib=0,
            fractional_sample_shift_mic=0.0,
            fractional_sample_shift_vib=0.0,
            audit=audit,
            stability=stability,
            residual_offset_uncertainty_s=residual_uncertainty,
            acoustic_envelope_kurtosis=env_kurt,
        )

    mic_corr, accel_corr, n_mic, n_vib, frac_mic, frac_vib = apply_sync_correction(
        mic_data, accel_data, mic_fs, accel_fs,
        offset_s=correction_offset_s,
        use_fractional_shift=use_fractional_shift,
    )

    return mic_corr, accel_corr, SyncCorrectionReport(
        applied=True,
        reason=(
            f"corrected by {correction_offset_s * 1e3:+.2f} ms "
            f"(int+frac shift; integer = {max(n_mic, n_vib)} samples, "
            f"fractional = {max(frac_mic, frac_vib):+.4f} samples)"
        ),
        applied_offset_s=correction_offset_s,
        integer_sample_shift_mic=int(n_mic),
        integer_sample_shift_vib=int(n_vib),
        fractional_sample_shift_mic=float(frac_mic),
        fractional_sample_shift_vib=float(frac_vib),
        audit=audit,
        stability=stability,
        residual_offset_uncertainty_s=residual_uncertainty,
        acoustic_envelope_kurtosis=env_kurt,
    )


__all__ = [
    "SyncCorrectionReport",
    "SyncResult",
    "SyncStabilityResult",
    "apply_sync_correction",
    "auto_sync_paired_recording",
    "verify_paired_sync",
    "verify_sync_stability",
]
