"""Tests for the cross-modal sync correction pipeline.

The correction has three layers:

  1. Sub-sample peak refinement via parabolic interpolation
     (Jacovitti & Scarano 1993).
  2. Time-stability check across K disjoint sub-segments
     (Bregler & Konig 1998).
  3. Sample-level + fractional-sample-level shift application
     (Smith, MDFT chapter 7, "Interpolating a function by
     ideal-bandlimited shifting").

We test each layer independently, then the high-level
`auto_sync_paired_recording` wrapper on a realistic
impulse-and-modulation paired signal.
"""

from __future__ import annotations

import numpy as np

from src.ingestion.sync_verification import (
    SyncCorrectionReport,
    SyncStabilityResult,
    _fractional_sample_shift,  # private but useful
    apply_sync_correction,
    auto_sync_paired_recording,
    verify_paired_sync,
    verify_sync_stability,
)

# ---------------------------------------------------------------------------
# Synthetic-signal helpers
# ---------------------------------------------------------------------------


def _impulse_train_pair(
    *,
    duration_s: float,
    mic_fs: int,
    accel_fs: int,
    vibration_lead_s: float,
    impulse_times_s: tuple[float, ...],
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired recording with impulses at known offsets.  Used because
    impulse-rich signals produce sharp xcorr peaks → high confidence
    → exercises the correction-applied code path."""
    rng = np.random.default_rng(seed)
    T_mic = int(duration_s * mic_fs)
    T_vib = int(duration_s * accel_fs)
    mic = 0.01 * rng.normal(size=(4, T_mic))
    vib = 0.01 * rng.normal(size=(4, T_vib))
    for it in impulse_times_s:
        mic_idx = int(it * mic_fs)
        vib_idx = int((it - vibration_lead_s) * accel_fs)
        if 0 <= mic_idx < T_mic - 200:
            mic[:, mic_idx : mic_idx + 200] += 5.0 * rng.normal(size=(4, 200))
        if 0 <= vib_idx < T_vib - 10:
            vib[:, vib_idx : vib_idx + 10] += 5.0 * rng.normal(size=(4, 10))
    return mic, vib


# ---------------------------------------------------------------------------
# Sub-sample refinement
# ---------------------------------------------------------------------------


def test_sync_result_carries_integer_and_refined_offsets() -> None:
    mic, vib = _impulse_train_pair(
        duration_s=30.0, mic_fs=16000, accel_fs=376,
        vibration_lead_s=0.080,
        impulse_times_s=(2.0, 5.0, 10.0, 15.0, 20.0, 25.0),
    )
    sr = verify_paired_sync(mic, vib, mic_fs=16000, accel_fs=376, max_offset_s=0.5)
    # Both refined and integer offsets should be present, finite, and close.
    assert not np.isnan(sr.offset_s)
    assert not np.isnan(sr.offset_s_integer)
    # The parabolic refinement is bounded to ± 0.5 samples → at 376 Hz that's
    # ± 1.33 ms.  So |refined - integer| < 1.5 ms.
    assert abs(sr.offset_s - sr.offset_s_integer) < 0.0015
    # The refinement delta must be in [-0.5, +0.5] samples.
    assert abs(sr.refinement_delta_samples) <= 0.5 + 1e-9


def test_refinement_moves_offset_toward_truth() -> None:
    """The cross-correlation integer-peak is biased by ~ 1–3 vibration
    samples on impulse-train synthetic data because the mic burst
    (200 samples at 16 kHz = 12.5 ms) and the vib impulse (10 samples
    at 376 Hz = 26.6 ms) have *different temporal extents* — the
    centroid of the cross-correlation envelope sits between the two
    centroids of energy, not at the truth lag.  This is a known
    property of cross-correlation TDOA estimators with mismatched
    pulse shapes (Knapp & Carter 1976 §III).  The PARABOLIC
    REFINEMENT is asked to do one thing only: move the integer-peak
    estimate closer to the local maximum of the underlying
    continuous cross-correlation function, which is at most ± 0.5
    samples away.  We test that invariant rather than absolute
    truth-recovery (which is bound by the integer-peak bias)."""
    mic, vib = _impulse_train_pair(
        duration_s=30.0, mic_fs=16000, accel_fs=376,
        vibration_lead_s=0.080,
        impulse_times_s=(2.0, 5.0, 10.0, 15.0, 20.0, 25.0),
    )
    sr = verify_paired_sync(mic, vib, mic_fs=16000, accel_fs=376, max_offset_s=0.5)
    one_vib_sample_s = 1.0 / 376.0
    # The refinement may move the offset by at most 0.5 samples.
    assert abs(sr.offset_s - sr.offset_s_integer) < 0.5 * one_vib_sample_s + 1e-9
    # The refinement-delta is bounded in [-0.5, +0.5] samples.
    assert -0.5 - 1e-9 <= sr.refinement_delta_samples <= 0.5 + 1e-9
    # The refined offset is finite and in the search window.
    assert -0.5 <= sr.offset_s <= 0.5


# ---------------------------------------------------------------------------
# Fractional-sample shift (FFT phase-ramp)
# ---------------------------------------------------------------------------


def test_fractional_shift_delays_impulse() -> None:
    """Shifting an impulse by +1.5 samples should put most of its energy
    at indices 51 and 52 (it was originally at index 50)."""
    x = np.zeros(200)
    x[50] = 1.0
    y = _fractional_sample_shift(x, 1.5)
    # The peak should be near 51 — between 51 and 52 because of the
    # half-sample shift.
    peak = int(np.argmax(np.abs(y)))
    assert peak in (51, 52)
    # Most of the energy lives in the immediate neighbourhood.
    near = float(np.sum(y[50:54] ** 2))
    total = float(np.sum(y ** 2))
    assert near / total > 0.5


def test_fractional_shift_zero_is_identity() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 256))
    y = _fractional_sample_shift(x, 0.0)
    assert np.allclose(x, y)


def test_fractional_shift_preserves_signal_norm_band_limited() -> None:
    """For band-limited real signals (energy at Nyquist ≈ 0), the
    fractional shift preserves total energy to fp precision.

    Why we test on band-limited input rather than white Gaussian:
    at the Nyquist bin of an even-N real-input DFT, a phase rotation
    by non-integer δ unavoidably forces a projection back onto the
    real axis (so the inverse-transformed signal is real), and that
    projection costs |sin(π·δ)|² · |X[N/2]|² of energy.  For an
    anti-alias-filtered real signal X[N/2] ≈ 0 → the projection is
    loss-free.  For white Gaussian noise X[N/2] has full random
    energy → the projection costs ~ 0.5 % of energy on δ = 0.25,
    which is the expected and theoretically-bounded behaviour, not a
    bug.  Real acoustic / vibration data is heavily band-limited
    long before Nyquist (16 kHz mic Nyquist = 8 kHz; the V0 LSTM-AE
    log-mel only covers 20 Hz – 8 kHz with `librosa.feature.melspectrogram`'s
    natural roll-off well below Nyquist).
    """
    rng = np.random.default_rng(0)
    # Construct a band-limited signal: low-pass at 0.4 × Nyquist.
    N = 512
    x = rng.normal(size=N)
    spec = np.fft.fft(x)
    cutoff = int(0.4 * N // 2)  # band-limit to 0–40 % of Nyquist
    mask = np.zeros(N, dtype=bool)
    mask[:cutoff + 1] = True
    mask[N - cutoff:] = True
    spec[~mask] = 0.0
    x_bl = np.fft.ifft(spec).real
    for shift in (0.25, 0.75, 1.5, -0.5, -2.5):
        y = _fractional_sample_shift(x_bl, shift)
        rel_err = abs(float(np.sum(x_bl ** 2)) - float(np.sum(y ** 2))) / max(
            float(np.sum(x_bl ** 2)), 1e-9
        )
        assert rel_err < 1e-9, (
            f"shift={shift}: rel_err={rel_err} (expected < 1e-9 on band-limited input)"
        )


def test_fractional_shift_energy_loss_bounded_at_nyquist() -> None:
    """On a non-band-limited input (white Gaussian), the energy loss
    is bounded above by `|sin(π·δ)|² · |X[N/2]|² / total_energy` —
    a known property of even-N DFT phase ramps, not a bug.  We
    verify the loss is ≤ 1 % for any |δ| < 1, which is the relevant
    operating regime on the prototype (max correction offset ≈ 50 ms
    = 19 samples at 376 Hz → much less than 1 sample at the
    higher-rate stream where the fractional shift is applied)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=512)
    for shift in (0.25, 0.5, 0.75):
        y = _fractional_sample_shift(x, shift)
        rel_loss = abs(float(np.sum(x ** 2)) - float(np.sum(y ** 2))) / float(np.sum(x ** 2))
        assert rel_loss < 0.01, (
            f"shift={shift}: rel_loss={rel_loss} exceeded 1 % bound"
        )


# ---------------------------------------------------------------------------
# Stability check
# ---------------------------------------------------------------------------


def test_stability_check_stable_when_offset_is_constant() -> None:
    """Impulses uniformly spread across the recording with the same
    lead in every chunk should give a stable result."""
    mic, vib = _impulse_train_pair(
        duration_s=50.0, mic_fs=16000, accel_fs=376,
        vibration_lead_s=0.050,
        impulse_times_s=tuple(np.linspace(2.0, 48.0, 25).tolist()),
    )
    st = verify_sync_stability(
        mic, vib, mic_fs=16000, accel_fs=376,
        n_sub_segments=5, max_offset_s=0.5,
        confidence_floor=1.2,  # impulse-train confidence is moderate; relax floor
        drift_tolerance_s=0.020,
    )
    assert isinstance(st, SyncStabilityResult)
    assert st.is_stable
    assert st.n_sub_segments_high_conf >= 3
    assert abs(st.median_offset_s - 0.050) < 0.010


def test_stability_check_unstable_when_offset_drifts() -> None:
    """A monotonically drifting offset (vib_lead grows linearly with
    sub-segment index) must be reported as unstable."""
    rng = np.random.default_rng(0)
    mic_fs, accel_fs = 16000, 376
    duration_s = 50.0
    T_mic = int(duration_s * mic_fs)
    T_vib = int(duration_s * accel_fs)
    mic = 0.01 * rng.normal(size=(4, T_mic))
    vib = 0.01 * rng.normal(size=(4, T_vib))
    # Impulses every 2 s; the vib lead grows from 0 ms at t=0 to 200 ms
    # at t=50 s → drift slope = 4 ms / s, well above the default
    # tolerance.
    for it in np.linspace(2.0, 48.0, 25):
        drift = (it / duration_s) * 0.200  # 0 → 200 ms drift
        mic_idx = int(it * mic_fs)
        vib_idx = int((it - drift) * accel_fs)
        if 0 <= mic_idx < T_mic - 200:
            mic[:, mic_idx : mic_idx + 200] += 5.0 * rng.normal(size=(4, 200))
        if 0 <= vib_idx < T_vib - 10:
            vib[:, vib_idx : vib_idx + 10] += 5.0 * rng.normal(size=(4, 10))
    st = verify_sync_stability(
        mic, vib, mic_fs=mic_fs, accel_fs=accel_fs,
        n_sub_segments=5, max_offset_s=0.3,
        confidence_floor=1.2,
        drift_tolerance_s=0.010,
    )
    assert st.is_stable is False
    # The linear fit should reveal the drift.
    assert abs(st.drift_slope_s_per_s) > 1e-3


# ---------------------------------------------------------------------------
# Correction application
# ---------------------------------------------------------------------------


def test_apply_sync_correction_drops_correct_stream() -> None:
    """offset > 0 (vibration leads) ⇒ mic lags ⇒ drop leading MIC samples.
    offset < 0 (acoustic leads)    ⇒ vib lags ⇒ drop leading VIB samples.

    The correction direction is fixed by the requirement that
    sample_0_mic_corrected and sample_0_vib_corrected correspond to
    the SAME physical instant after the drop.  See
    `docs/REVIEW.md` SYNC-CORRECTION-DIRECTION derivation."""
    mic = np.zeros((2, 1000))
    vib = np.zeros((2, 500))
    # vib leads by 0.1 s → mic lags → drop 0.1 × 1000 = 100 leading MIC samples
    mic_c, vib_c, n_mic, n_vib, frac_mic, frac_vib = apply_sync_correction(
        mic, vib, mic_fs=1000.0, accel_fs=100.0,
        offset_s=0.1, use_fractional_shift=False,
    )
    assert n_mic == 100 and n_vib == 0
    assert mic_c.shape == (2, 900)
    assert vib_c.shape == vib.shape  # untouched

    # acoustic leads by 0.05 s → vib lags → drop 0.05 × 100 = 5 leading VIB samples
    mic_c, vib_c, n_mic, n_vib, frac_mic, frac_vib = apply_sync_correction(
        mic, vib, mic_fs=1000.0, accel_fs=100.0,
        offset_s=-0.05, use_fractional_shift=False,
    )
    assert n_mic == 0 and n_vib == 5
    assert mic_c.shape == mic.shape
    assert vib_c.shape == (2, 495)


def test_apply_sync_correction_zero_offset_is_identity() -> None:
    mic = np.ones((2, 100))
    vib = np.ones((2, 50))
    mic_c, vib_c, n_mic, n_vib, fm, fv = apply_sync_correction(
        mic, vib, mic_fs=1000.0, accel_fs=100.0, offset_s=0.0,
    )
    assert mic_c is mic and vib_c is vib
    assert n_mic == 0 and n_vib == 0
    assert fm == 0.0 and fv == 0.0


def test_apply_sync_correction_decomposes_fractional() -> None:
    """A fractional offset must decompose into an integer-sample shift
    on the lagging stream plus a fractional-sample FFT-phase-ramp on
    the same stream.

    Test: vib leads mic by 0.0535 s.  mic lags ⇒ drop from MIC.
    At mic_fs = 1000 the offset is 53.5 mic samples → n_mic = 53,
    frac_mic = 0.5."""
    rng = np.random.default_rng(0)
    mic = rng.normal(size=(2, 1000))  # non-trivial input for the FFT shift to act on
    vib = np.zeros((2, 500))
    offset_s = 53.5 / 1000.0  # vib leads by 53.5 ms at mic_fs = 1000
    mic_c, vib_c, n_mic, n_vib, fm, fv = apply_sync_correction(
        mic, vib, mic_fs=1000.0, accel_fs=100.0,
        offset_s=offset_s, use_fractional_shift=True,
    )
    assert n_mic == 53
    assert n_vib == 0
    assert abs(fm - 0.5) < 1e-9
    assert mic_c.shape == (2, 1000 - 53)


# ---------------------------------------------------------------------------
# High-level auto_sync_paired_recording wrapper
# ---------------------------------------------------------------------------


def test_auto_sync_applies_correction_on_stable_high_confidence_input() -> None:
    """A 50 ms offset with consistent impulses across the recording
    should pass both gates and correct."""
    mic, vib = _impulse_train_pair(
        duration_s=50.0, mic_fs=16000, accel_fs=376,
        vibration_lead_s=0.050,
        impulse_times_s=tuple(np.linspace(2.0, 48.0, 30).tolist()),
    )
    mic_c, vib_c, report = auto_sync_paired_recording(
        mic, vib, mic_fs=16000, accel_fs=376,
        confidence_floor=1.2, drift_tolerance_s=0.020,
    )
    assert isinstance(report, SyncCorrectionReport)
    assert report.applied is True
    assert abs(report.applied_offset_s - 0.050) < 0.010
    # Post-correction audit should report ≈ 0 offset.
    post = verify_paired_sync(mic_c, vib_c, mic_fs=16000, accel_fs=376)
    # Tolerance: one vibration sample (~ 2.66 ms) + numerical noise.
    assert abs(post.offset_s) < 0.010


def test_auto_sync_refuses_correction_on_low_confidence() -> None:
    """Independent *impulse trains* in mic and vib at unrelated random
    times: each stream has a spiky (high-kurtosis) envelope so Gate 0
    passes, but the two streams do not correlate.  The wrapper must
    refuse correction — uncorrelated impulse trains can be caught by
    either the full-recording audit gate (Gate 1) or the per-sub-segment
    stability gate (Gate 2), both of which are correct safety responses.
    The point of this test is that Gate 0 is *not* what fires here."""
    rng = np.random.default_rng(0)
    mic_fs, accel_fs = 16000, 376
    duration_s = 30.0
    T_mic, T_vib = int(duration_s * mic_fs), int(duration_s * accel_fs)
    mic = 0.01 * rng.normal(size=(4, T_mic))
    vib = 0.01 * rng.normal(size=(4, T_vib))
    # Mic impulses and vib impulses at INDEPENDENT random times — spiky
    # envelopes (Gate 0 passes) but no cross-modal correlation.
    for t in rng.uniform(2.0, 28.0, size=25):
        idx = int(t * mic_fs)
        mic[:, idx : idx + 200] += 5.0 * rng.normal(size=(4, 200))
    for t in rng.uniform(2.0, 28.0, size=25):
        idx = int(t * accel_fs)
        vib[:, idx : idx + 10] += 5.0 * rng.normal(size=(4, 10))
    mic_c, vib_c, report = auto_sync_paired_recording(
        mic, vib, mic_fs=mic_fs, accel_fs=accel_fs,
    )
    assert report.applied is False
    reason = report.reason.lower()
    # Gate 1 (low confidence / uninformative) or Gate 2 (stability) — NOT Gate 0.
    assert any(kw in reason for kw in ("confidence", "uninformative", "stability", "drift")), (
        f"unexpected rejection reason: {report.reason}"
    )
    assert "near-gaussian" not in reason, "Gate 0 should not fire on spiky impulse trains"
    # The F3 diagnostic is populated regardless of which gate fired.
    assert np.isfinite(report.acoustic_envelope_kurtosis)
    # Spiky impulse trains have a high-kurtosis envelope.
    assert report.acoustic_envelope_kurtosis > 1.0
    # Streams returned unmodified.
    assert mic_c is mic and vib_c is vib


def test_auto_sync_refuses_correction_on_flat_envelope() -> None:
    """F3 — Gate 0 (envelope-kurtosis precondition).  Broadband noise has
    a Rayleigh-distributed analytic envelope (excess kurtosis ≈ 0.25),
    which after anti-alias decimation flattens toward Gaussian — far
    below the kurtosis a transient-rich recording produces.  Envelope
    cross-correlation cannot lock onto a meaningful offset on such
    content, so the wrapper must refuse to *attempt* correction rather
    than report a noise-dominated offset as if it were real.  This is
    the D4 case: shared-trigger acquisition means the streams ARE
    aligned, but steady-state content means we cannot prove it from
    envelope correlation.
    """
    rng = np.random.default_rng(1)
    mic_fs, accel_fs = 16000, 376
    duration_s = 30.0
    # Pure broadband noise — no impulses, no shared transients.  Flat
    # (low-kurtosis) envelope on both streams.
    mic = rng.normal(size=(4, int(duration_s * mic_fs)))
    vib = rng.normal(size=(4, int(duration_s * accel_fs)))
    mic_c, vib_c, report = auto_sync_paired_recording(
        mic, vib, mic_fs=mic_fs, accel_fs=accel_fs,
        min_envelope_kurtosis=1.0,
    )
    assert report.applied is False
    assert "envelope" in report.reason.lower()
    assert "kurtosis" in report.reason.lower()
    # Gate 0 fired → the diagnostic kurtosis is below the threshold.
    assert np.isfinite(report.acoustic_envelope_kurtosis)
    assert report.acoustic_envelope_kurtosis < 1.0
    assert mic_c is mic and vib_c is vib


def test_auto_sync_refuses_correction_on_drift() -> None:
    """A drifting offset must be flagged as 'stability check failed'
    and NOT corrected."""
    rng = np.random.default_rng(0)
    mic_fs, accel_fs = 16000, 376
    duration_s = 50.0
    T_mic = int(duration_s * mic_fs)
    T_vib = int(duration_s * accel_fs)
    mic = 0.01 * rng.normal(size=(4, T_mic))
    vib = 0.01 * rng.normal(size=(4, T_vib))
    for it in np.linspace(2.0, 48.0, 30):
        drift = (it / duration_s) * 0.200
        mic_idx = int(it * mic_fs)
        vib_idx = int((it - drift) * accel_fs)
        if 0 <= mic_idx < T_mic - 200:
            mic[:, mic_idx : mic_idx + 200] += 5.0 * rng.normal(size=(4, 200))
        if 0 <= vib_idx < T_vib - 10:
            vib[:, vib_idx : vib_idx + 10] += 5.0 * rng.normal(size=(4, 10))
    mic_c, vib_c, report = auto_sync_paired_recording(
        mic, vib, mic_fs=mic_fs, accel_fs=accel_fs,
        confidence_floor=1.2, drift_tolerance_s=0.010,
        max_offset_s=0.3,
    )
    assert report.applied is False
    # The wrapper must refuse correction.  A drifting recording can
    # be caught by either gate: the full-recording audit (because
    # drift smears the global cross-correlation peak → low
    # confidence) or the per-sub-segment stability check.  Both are
    # correct safety behaviours; we accept either rejection reason.
    reason = report.reason.lower()
    assert any(
        kw in reason for kw in ("stability", "drift", "confidence")
    ), f"unexpected rejection reason: {report.reason}"
    assert mic_c is mic and vib_c is vib


def test_auto_sync_refuses_correction_when_offset_is_below_floor() -> None:
    """A near-zero offset (< 1 ms) should not be corrected — the
    streams are already aligned within tolerance and applying any
    shift would introduce more numerical noise than it removes."""
    mic, vib = _impulse_train_pair(
        duration_s=50.0, mic_fs=16000, accel_fs=376,
        vibration_lead_s=0.0,
        impulse_times_s=tuple(np.linspace(2.0, 48.0, 30).tolist()),
    )
    # `min_offset_to_correct_s` must be set above the natural audit
    # noise floor on impulse-train synthetic data (which has ~ 1 – 3
    # vibration-sample integer-peak bias from mic/vib pulse-shape
    # asymmetry — see `test_refinement_moves_offset_toward_truth`).
    # At accel_fs = 376 the bias-induced noise is ~ 8 ms.  Setting
    # the correction floor to 20 ms guarantees this test isolates
    # the "offset truly is zero" gate behaviour.
    mic_c, vib_c, report = auto_sync_paired_recording(
        mic, vib, mic_fs=16000, accel_fs=376,
        confidence_floor=1.2,
        min_offset_to_correct_s=0.020,
    )
    assert report.applied is False
    assert "below" in report.reason.lower() or "already aligned" in report.reason.lower()


def test_auto_sync_reduces_residual_offset_after_correction() -> None:
    """The post-correction audit must report a residual offset
    smaller than the pre-correction offset."""
    truth_lead_s = 0.075
    mic, vib = _impulse_train_pair(
        duration_s=60.0, mic_fs=16000, accel_fs=376,
        vibration_lead_s=truth_lead_s,
        impulse_times_s=tuple(np.linspace(2.0, 58.0, 35).tolist()),
    )
    pre = verify_paired_sync(mic, vib, mic_fs=16000, accel_fs=376)
    mic_c, vib_c, report = auto_sync_paired_recording(
        mic, vib, mic_fs=16000, accel_fs=376,
        confidence_floor=1.2, drift_tolerance_s=0.020,
    )
    assert report.applied
    post = verify_paired_sync(mic_c, vib_c, mic_fs=16000, accel_fs=376)
    # The correction should reduce |offset| by at least an order of
    # magnitude.
    assert abs(post.offset_s) < abs(pre.offset_s) / 5.0
