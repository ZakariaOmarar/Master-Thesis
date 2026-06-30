"""Vibration encoder-input features: per-channel time-aligned multichannel series.

Produces a 3-D tensor with shape ``(n_vib, 3, T_vib)`` where the inner three
channels are ``[amplitude, hilbert_envelope, impulsiveness]``.

The third channel is an *impulsiveness* statistic whose operational
definition depends on the **sample rate of the input stream**:

  - Rolling **excess kurtosis** (Joanes-Gill bias-corrected) when the
    physical-time kurtosis window contains at least
    ``min_kurtosis_samples`` samples.  Picked because excess kurtosis
    is the classical impulsive-vibration diagnostic (Dwyer 1983;
    Antoni, MSSP 2006) and it has a non-trivial null distribution
    only above ~30 samples (σ_kurt ≈ √(24/N) under Gaussian noise).
  - Rolling **crest factor** ``max(|x|) / RMS(x)`` otherwise.  Crest
    factor is the ISO-10816-style impulsiveness indicator that
    remains well-defined down to ~4 samples, so it is the
    statistically-honest substitute on the 4 Hz / 16 Hz peak-amplitude
    streams (D1/D2/D3) where a 100 ms kurtosis window would only
    contain 0–2 samples.

Both statistics encode the same inductive bias for the downstream
encoder ("high channel-2 value = impulsive event"), so the per-dataset
swap is transparent to the V1 / V2 / V4 encoders even though they share
weights across datasets.  The selected mode is exposed via
:func:`channel2_statistic_name` so thesis tables can label the right
quantity per dataset.

Sample-rate map for the four current datasets
(:mod:`src.ingestion.test_dataset_loader`):

==========  =============  ===================  =================
Dataset     ``accel_sr``   default kurtosis     fallback used
==========  =============  ===================  =================
D1 (peak)   4 Hz           N=1  (<31)           crest factor (N=5)
D2 (peak)   4 Hz           N=1  (<31)           crest factor (N=5)
D3 (peak)   16 Hz          N=3  (<31)           crest factor (N=17)
D4 (raw)    ~376 Hz        N=37 (>=31)          excess kurtosis
==========  =============  ===================  =================

Distinct from :mod:`src.features.vibration_envelope`, which produces a
fixed-length feature vector per window for classical baselines; here we
keep the time series so the V1 1-D CNN can convolve along it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import hilbert

from ..config.architecture import VIBRATION_FEATURES
from .feature_cache import disk_cached_feature

Channel2Mode = Literal["kurtosis", "crest_factor", "none"]


def channel2_statistic_name(
    sample_rate: float,
    *,
    kurtosis_window_seconds: float = VIBRATION_FEATURES.kurtosis_window_seconds,
    min_kurtosis_samples: int = VIBRATION_FEATURES.min_kurtosis_samples,
    crest_factor_window_seconds: float = VIBRATION_FEATURES.crest_factor_window_seconds,
    min_crest_factor_samples: int = VIBRATION_FEATURES.min_crest_factor_samples,
) -> Channel2Mode:
    """Return which impulsiveness statistic channel 2 will carry for ``sample_rate``.

    Useful for thesis tables / logging: the V1 / V2 / V4 trainers share
    weights across datasets but channel 2 represents either rolling
    kurtosis (D4 raw, ~376 Hz) or rolling crest factor (D1/D2/D3 peak
    streams) depending on whether the kurtosis window meets the
    statistical-sufficiency floor.
    """
    mode, _ = _select_channel2_mode(
        sample_rate,
        kurtosis_window_seconds=kurtosis_window_seconds,
        min_kurtosis_samples=min_kurtosis_samples,
        crest_factor_window_seconds=crest_factor_window_seconds,
        min_crest_factor_samples=min_crest_factor_samples,
    )
    return mode


@disk_cached_feature
def compute_vibration_input_stack(
    accel_data: np.ndarray,
    *,
    sample_rate: float,
    kurtosis_window_seconds: float = VIBRATION_FEATURES.kurtosis_window_seconds,
    min_kurtosis_samples: int = VIBRATION_FEATURES.min_kurtosis_samples,
    crest_factor_window_seconds: float = VIBRATION_FEATURES.crest_factor_window_seconds,
    min_crest_factor_samples: int = VIBRATION_FEATURES.min_crest_factor_samples,
    standardize: bool = VIBRATION_FEATURES.standardize,
    standardization_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Build the V1 vibration encoder input.

    Args:
        accel_data: ``(n_vib, T_vib)`` raw amplitude (already at the per-dataset
            target sample rate produced by `WavVibrationAdapter`).
        sample_rate: Sample rate of ``accel_data`` in Hz.  Required so the
            physical-time kurtosis window translates to a defensible sample
            count on every dataset (D1/D2: 4 Hz, D3: 16 Hz, D4 raw: ~376 Hz).
        kurtosis_window_seconds: Centred window duration for rolling kurtosis.
            Default 100 ms — the order-of-magnitude ring-down time of a knock
            impulse on the 3D-printed prototype casing; long enough to span
            one event with brief context, short enough not to dilute adjacent
            events.  Override per experiment if your prototype's damping
            differs.
        min_kurtosis_samples: Sample-count floor below which kurtosis is
            replaced by crest factor (default 31).  Below this floor the
            sampling standard error of kurtosis on Gaussian noise (≈√(24/N))
            exceeds ~0.9, so weak impulses (excess kurtosis ~3–5) are
            indistinguishable from noise.  31 yields σ_kurt ≈ 0.88.
        crest_factor_window_seconds: Centred window for the crest-factor
            fallback (default 1 s).  Picked so D1/D2 (4 Hz → 5 samples) and
            D3 (16 Hz → 17 samples) both meet ``min_crest_factor_samples``.
        min_crest_factor_samples: Sample-count floor below which channel 2
            is filled with zeros (signal physically too short to characterise
            impulsiveness at any window).  Default 4 — the minimum for a
            stable max/RMS ratio.
        standardize: When True (default), the amplitude and Hilbert-envelope
            channels are z-score-normalised.  See ``standardization_stats``
            for the granularity options.  The impulsiveness channel is
            dimensionless (kurtosis is scale-invariant; crest factor is a
            ratio) and is **not** re-standardised — the F5 audit
            experiment (2026-05-14) z-scored it per-recording and found no
            measurable benefit to V1 vibration cluster quality (NMI 0.073
            with vs 0.108 without).
        standardization_stats: When ``standardize=True`` and this is None,
            per-recording-per-channel z-scoring is used (legacy default).
            Pass a ``(mean, std)`` tuple of shape ``(n_vib,)`` to apply a
            per-dataset z-score instead — preserves intra-recording amplitude
            differences (e.g. Pump > Standstill) while bridging cross-dataset
            scale (D1 peak ~ 1000 ADU vs D4 raw ~ 18 000 ADU).

    Returns:
        ``(n_vib, 3, T_vib)`` float32 array.  Channels:
          0 — amplitude (zero-mean; unit-variance when ``standardize=True``),
          1 — Hilbert envelope (zero-mean unit-variance when ``standardize=True``),
          2 — impulsiveness: rolling excess kurtosis (Joanes-Gill,
              ``bias=False``) when ``sample_rate`` clears the statistical
              floor, else rolling crest factor ``max(|x|)/RMS(x)``; reflect
              padded edges; dimensionless and never re-standardised.

    Raises:
        ValueError: If ``accel_data`` is not 2-D, or if ``sample_rate`` is
            non-positive, or if ``standardization_stats`` shape mismatches.
    """
    if accel_data.ndim != 2:
        raise ValueError("accel_data must be 2-D (n_vib, T_vib)")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive finite Hz; got {sample_rate!r}")

    n_vib, T = int(accel_data.shape[0]), int(accel_data.shape[1])

    ch2_mode, ch2_window = _select_channel2_mode(
        sample_rate,
        kurtosis_window_seconds=kurtosis_window_seconds,
        min_kurtosis_samples=min_kurtosis_samples,
        crest_factor_window_seconds=crest_factor_window_seconds,
        min_crest_factor_samples=min_crest_factor_samples,
    )
    # Reflect padding needs window <= T; if the segment is shorter than the
    # window, shrink to the largest odd window that fits and fall back further
    # if that drops below the floor.
    if ch2_mode != "none" and ch2_window > T:
        ch2_window = T if T % 2 == 1 else max(1, T - 1)
        if ch2_mode == "kurtosis" and ch2_window < min_kurtosis_samples:
            # Demote to crest factor at the shrunk window if it still clears
            # the crest floor; else give up and emit zeros.
            if ch2_window >= min_crest_factor_samples:
                ch2_mode = "crest_factor"
            else:
                ch2_mode, ch2_window = "none", 0
        elif ch2_mode == "crest_factor" and ch2_window < min_crest_factor_samples:
            ch2_mode, ch2_window = "none", 0

    out = np.zeros((n_vib, 3, T), dtype=np.float32)
    eps = 1e-8

    use_dataset_stats = standardize and standardization_stats is not None
    ds_mean = ds_std = None
    if use_dataset_stats:
        ds_mean = np.asarray(standardization_stats[0], dtype=np.float64)
        ds_std = np.asarray(standardization_stats[1], dtype=np.float64)
        if ds_mean.shape != (n_vib,) or ds_std.shape != (n_vib,):
            raise ValueError(
                f"standardization_stats must have shape ({n_vib},); "
                f"got mean={ds_mean.shape}, std={ds_std.shape}"
            )

    for i in range(n_vib):
        x = accel_data[i].astype(np.float64)

        if use_dataset_stats:
            x_centred = x - float(ds_mean[i])
            amp = x_centred / max(float(ds_std[i]), eps)
            envelope_raw = (
                np.abs(hilbert(x_centred)) if T >= 2 else np.abs(x_centred)
            )
            env_mean = float(np.mean(envelope_raw))
            env_std = float(np.std(envelope_raw))
            envelope = envelope_raw - env_mean
            if env_std > eps:
                envelope = envelope / env_std
            out[i, 0] = amp.astype(np.float32)
            out[i, 1] = envelope.astype(np.float32)
            x_for_ch2 = x_centred
        else:
            x_centred = x - float(np.mean(x))
            amp = x_centred.copy()
            if standardize:
                std_x = float(np.std(amp))
                if std_x > eps:
                    amp = amp / std_x
            out[i, 0] = amp.astype(np.float32)

            envelope = np.abs(hilbert(x_centred)) if T >= 2 else np.abs(x_centred)
            if standardize:
                env_mean = float(np.mean(envelope))
                env_std = float(np.std(envelope))
                envelope = envelope - env_mean
                if env_std > eps:
                    envelope = envelope / env_std
            out[i, 1] = envelope.astype(np.float32)
            x_for_ch2 = x_centred

        if ch2_mode == "kurtosis":
            out[i, 2] = _rolling_excess_kurtosis(x_for_ch2, ch2_window).astype(
                np.float32
            )
        elif ch2_mode == "crest_factor":
            out[i, 2] = _rolling_crest_factor(x_for_ch2, ch2_window).astype(np.float32)
        # else: "none" → leave as zeros (channel is uninformative but the
        # shape contract is preserved for the encoder).

    return out


# ---------------------------------------------------------------------------
# Channel-2 statistic selection and vectorised computation
# ---------------------------------------------------------------------------


def _select_channel2_mode(
    sample_rate: float,
    *,
    kurtosis_window_seconds: float,
    min_kurtosis_samples: int,
    crest_factor_window_seconds: float,
    min_crest_factor_samples: int,
) -> tuple[Channel2Mode, int]:
    """Pick (mode, odd_window_in_samples) for the given sample rate."""
    n_k = _odd_samples(kurtosis_window_seconds * sample_rate)
    if n_k >= max(5, min_kurtosis_samples):
        return "kurtosis", n_k
    n_cf = _odd_samples(crest_factor_window_seconds * sample_rate)
    if n_cf >= max(3, min_crest_factor_samples):
        return "crest_factor", n_cf
    return "none", 0


def _odd_samples(n_float: float) -> int:
    """Round to the nearest positive odd integer (centred windows must be odd)."""
    n = int(round(n_float))
    if n <= 0:
        return 0
    return n if n % 2 == 1 else n + 1


def _rolling_excess_kurtosis(x: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling Joanes-Gill bias-corrected excess kurtosis.

    Reflect-padded edges so the output length equals ``len(x)``.  Equivalent
    to ``scipy.stats.kurtosis(window, fisher=True, bias=False)`` per window,
    but vectorised via :func:`numpy.lib.stride_tricks.sliding_window_view`.
    """
    T = x.shape[0]
    if window <= 0 or T == 0:
        return np.zeros(T, dtype=np.float64)
    half = window // 2
    xp = np.pad(x, half, mode="reflect")
    sw = sliding_window_view(xp, window)  # (T, window)
    mean = sw.mean(axis=-1, keepdims=True)
    centred = sw - mean
    m2 = (centred * centred).mean(axis=-1)
    m4 = (centred * centred * centred * centred).mean(axis=-1)
    # g2 = biased excess kurtosis (m4/m2^2 - 3); guard against m2 ≈ 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        biased = np.where(m2 > 1e-24, m4 / (m2 * m2) - 3.0, 0.0)
    n = window
    if n > 3:
        # Joanes-Gill (1998) bias correction (matches scipy `bias=False`):
        # G2 = (n-1)/((n-2)(n-3)) * ((n+1) * g2 + 6)
        g2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * biased + 6.0)
    else:
        g2 = biased
    return g2


def _rolling_crest_factor(x: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling crest factor ``max(|x|) / RMS(x)``.

    Reflect-padded edges so the output length equals ``len(x)``.  Crest
    factor of pure Gaussian noise is ≈ 3 (slowly grows with N); pure sine
    is ≈ √2; impulsive events produce values ≥ 5.
    """
    T = x.shape[0]
    if window <= 0 or T == 0:
        return np.zeros(T, dtype=np.float64)
    half = window // 2
    xp = np.pad(x, half, mode="reflect")
    sw = sliding_window_view(xp, window)
    abs_max = np.abs(sw).max(axis=-1)
    rms = np.sqrt((sw * sw).mean(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        cf = np.where(rms > 1e-12, abs_max / rms, 0.0)
    return cf


__all__ = ["channel2_statistic_name", "compute_vibration_input_stack"]
