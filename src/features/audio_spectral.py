"""Acoustic encoder-input features: log-mel + CWT scalograms stacked per mic.

Produces a 4-D tensor with shape ``(n_mics, 2, F, T_frames)`` where channel 0 is
a power-to-dB log-mel spectrogram and channel 1 is a CWT scalogram
**re-gridded** to the log-mel ``(F, T)`` shape.  The two channels are
intentionally **complementary spectral views** rather than co-registered
images — see :func:`compute_encoder_input_stack` for the asymmetric-band
design rationale.

Distinct from :mod:`src.features.acoustic_representations`, which exposes
the individual primitives (STFT, MFCC+deltas) for the V0 / classical
baselines.
"""

from __future__ import annotations

import numpy as np

from ..config.architecture import ACOUSTIC_CWT, ACOUSTIC_FEATURES
from .acoustic_representations import compute_cwt_scalogram
from .feature_cache import disk_cached_feature


@disk_cached_feature
def compute_log_mel_spectrogram(
    signal: np.ndarray,
    fs: int,
    *,
    n_fft: int = ACOUSTIC_FEATURES.n_fft,
    hop_length: int = ACOUSTIC_FEATURES.hop_length,
    n_mels: int = ACOUSTIC_FEATURES.n_mels,
    fmin: float = ACOUSTIC_FEATURES.fmin_hz,
    fmax: float | None = ACOUSTIC_FEATURES.fmax_hz,
    top_db: float = ACOUSTIC_FEATURES.top_db,
) -> np.ndarray:
    """Power-to-dB log-mel spectrogram for one channel.

    Returns a 2-D array of shape ``(n_mels, n_frames)`` in **decibels** with
    a fixed dynamic-range floor of ``top_db`` dB below the maximum *that the
    function would otherwise compute* for the input (i.e. clip-from-below
    only — there is no gain normalisation).  This is the standard log-mel
    convention used in audio-classification benchmarks (Choi et al., 2017;
    librosa convention) and is gain-equivariant: a +6 dB gain on the input
    shifts every output bin by +6 dB rather than reshaping the distribution.

    Args:
        signal: 1-D mono microphone waveform.
        fs: Sample rate in Hz.
        n_fft: STFT window length (samples).  These are the bare function
            defaults; the pipeline passes the empirically-selected values from
            `src.config.architecture.ACOUSTIC_FEATURES` (n_fft=4096) — see
            chapter 3 §3.4.2 and `scripts/hop_length_study/analyze_hop_length_full_grid.py`.
        hop_length: STFT hop length (samples).  Pipeline value comes from
            `ACOUSTIC_FEATURES.hop_length` (2048); the grid sweep showed the
            downstream tasks are insensitive to hop once n_fft and n_mels are
            fixed.
        n_mels: Number of mel filterbank bands.  Pipeline value is
            `ACOUSTIC_FEATURES.n_mels` (96).
        fmin: Lower mel-band edge in Hz.  Default 20 Hz rejects microphone
            self-noise roll-off and AC line hum.
        fmax: Upper mel-band edge in Hz; None ⇒ Nyquist.
        top_db: Dynamic-range floor below the peak of THIS recording's mel
            power.  Default 80 dB matches librosa's `power_to_db` default.

    Returns:
        ``(n_mels, n_frames)`` float64 array in dB.  Typical range is
        ``[-top_db, 0]`` relative to the per-recording peak; absolute
        scale is preserved up to ``ref=1.0`` (no gain normalisation).
    """
    import librosa

    if fs <= 0:
        raise ValueError("fs must be > 0")
    if n_mels <= 0:
        raise ValueError("n_mels must be > 0")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be > 0")

    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("signal must be 1-D")
    if x.size == 0:
        raise ValueError("signal cannot be empty")

    nyquist = fs / 2.0
    if fmax is None:
        fmax = nyquist
    if fmax > nyquist:
        fmax = nyquist

    mel_power = librosa.feature.melspectrogram(
        y=x,
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    # `ref=1.0` keeps absolute amplitude (no per-recording peak normalisation),
    # so cross-recording amplitude differences (e.g. Pump > Standstill at the
    # casing wall) survive into the encoder; BatchNorm absorbs the residual
    # bias.  `top_db=80` clips the noise floor at -80 dB below the peak,
    # preventing -∞ on silent bins.
    log_mel = librosa.power_to_db(mel_power, ref=1.0, top_db=top_db)
    return np.asarray(log_mel, dtype=np.float64)


@disk_cached_feature
def compute_encoder_input_stack(
    mic_data: np.ndarray,
    fs: int,
    *,
    n_mels: int = ACOUSTIC_FEATURES.n_mels,
    n_fft: int = ACOUSTIC_FEATURES.n_fft,
    hop_length: int = ACOUSTIC_FEATURES.hop_length,
    cwt_n_scales: int = ACOUSTIC_CWT.n_scales,
    cwt_min_freq_hz: float = ACOUSTIC_CWT.min_freq_hz,
    cwt_max_freq_hz: float = ACOUSTIC_CWT.max_freq_hz,
    standardize: bool = False,
) -> np.ndarray:
    """Build the V1 / V2 acoustic encoder input as two complementary spectral views.

    **Asymmetric-band design (intentional).**  Channel 0 carries a log-mel
    spectrogram over the full **20 Hz – 8 kHz** acoustic band; channel 1
    carries a CWT scalogram over a **narrow 20 – 250 Hz mechanical-tone
    band** that brackets the ROW II target frequencies (5.87 Hz shaft,
    43.75 Hz runner-blade-passing, 100 Hz rotor-pole-passing, 117 Hz
    guide-vane-passing — see `src/config/constants.py`).  After
    construction both channels share the same ``(n_mels, T_frames)``
    tensor shape so the downstream ``Conv2d`` can consume them as a 2-channel
    image, but **row index does not correspond to the same physical
    frequency in both channels** — row 0 is ~ 20 Hz in both, row 63 is
    ~ 8 kHz in log-mel and ~ 250 Hz in CWT.  The Conv2d therefore treats
    the two channels as parallel feature streams rather than co-registered
    views, mirroring the design pattern used in dual-resolution Conv2d
    audio classifiers (e.g. Pons et al. 2019).  This is the deliberate
    division of labour: log-mel covers broadband knock-impulse signatures
    spanning 200 Hz – 4 kHz, CWT zooms in on the narrow mechanical-tone
    band where mode discriminability lives and where short-window log-mel
    is comparatively blind (Khamaisi et al.; Vibrational-Hill-Chart
    studies cited in Chapter 2).

    **Time-axis aggregation.**  The CWT is computed at a 1 kHz decimated
    rate (see `compute_cwt_scalogram`) which yields ~ 32× more time samples
    than the log-mel grid at the publication hop.  Reducing this to
    `T_frames` is done by **non-overlapping max-pool** along the time axis
    (not bilinear interpolation), preserving the peak energy of transient
    events — bilinear interp would average a 10 ms knock impulse with its
    surrounding silence and weaken the very signature the CWT is most
    useful for.  Frequency-axis remapping (typically a no-op since
    `n_mels == cwt_n_scales == 64` in the publication config) uses
    linear interpolation since adjacent CWT scales represent physically
    close frequencies.

    Args:
        mic_data: ``(n_mics, n_samples)`` microphone waveforms.
        fs: Microphone sample rate (16 000 Hz for this thesis).
        n_mels / n_fft / hop_length: log-mel parameters; see
            :func:`compute_log_mel_spectrogram`.  Defaults match the
            publication YAML.
        cwt_n_scales / cwt_min_freq_hz / cwt_max_freq_hz: CWT scalogram
            parameters; see :func:`compute_cwt_scalogram`.  The 20-250 Hz
            default band is justified above.
        standardize: When True, each channel is per-recording z-scored
            across (mics, frequency, time) so the two channels enter the
            Conv2d on a unit-variance scale.  **Default False** — the F4
            audit (2026-05-14) found it not load-bearing once BatchNorm
            is the encoder norm.  Kept as a knob because the
            channel-scale mismatch is real if the SSL objective ever
            changes.

    Returns:
        ``(n_mics, 2, n_mels, T_frames)`` float32 array.  Channel 0 is
        log-mel (dB), channel 1 is the time-max-pooled, frequency-
        interpolated CWT scalogram.  With ``standardize=True`` each
        channel has zero mean and unit variance over (mic, frequency,
        time); otherwise raw scales are preserved.
    """
    if mic_data.ndim != 2:
        raise ValueError("mic_data must be 2-D (n_mics, n_samples)")
    n_mics = int(mic_data.shape[0])

    log_mels: list[np.ndarray] = []
    cwts: list[np.ndarray] = []
    for i in range(n_mics):
        log_mels.append(
            compute_log_mel_spectrogram(
                mic_data[i],
                fs,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
        )
        cwts.append(
            compute_cwt_scalogram(
                mic_data[i],
                fs,
                n_scales=cwt_n_scales,
                min_freq_hz=cwt_min_freq_hz,
                max_freq_hz=cwt_max_freq_hz,
            )
        )

    target_F = log_mels[0].shape[0]
    target_T = log_mels[0].shape[1]

    aligned: list[np.ndarray] = []
    for mel, cwt in zip(log_mels, cwts):
        # Time axis: non-overlapping max-pool to preserve transient peaks.
        # Frequency axis: linear interp (adjacent CWT scales are physically
        # close, so interpolation is a valid magnitude reconstruction).
        cwt_time_pooled = _max_pool_time(cwt, target_T)
        if cwt_time_pooled.shape[0] != target_F:
            cwt_aligned = _linear_resize_freq(cwt_time_pooled, target_F)
        else:
            cwt_aligned = cwt_time_pooled
        aligned.append(np.stack([mel, cwt_aligned], axis=0))

    stack = np.stack(aligned, axis=0).astype(np.float32)

    if standardize:
        eps = 1e-8
        for ch in (0, 1):
            mean = float(stack[:, ch, :, :].mean())
            std = float(stack[:, ch, :, :].std())
            stack[:, ch, :, :] = (stack[:, ch, :, :] - mean) / max(std, eps)

    return stack


# ---------------------------------------------------------------------------
# Re-gridding helpers
# ---------------------------------------------------------------------------


def _max_pool_time(arr: np.ndarray, target_T: int) -> np.ndarray:
    """Non-overlapping max-pool along the last (time) axis to length ``target_T``.

    Preserves transient peak energy under aggressive downsampling
    (~32× from the 1 kHz CWT rate to the 31.25 Hz log-mel grid at the
    publication hop).  A bilinear interpolator would *average* across the
    pooling block and erase short impulse onsets — exactly the signal we
    most want to keep.  When ``target_T > T`` (upsampling, rare),
    nearest-neighbour index replication is used because there is no peak
    energy to preserve in that direction.
    """
    F, T = arr.shape
    if T == target_T:
        return arr.astype(np.float64)
    if T < target_T:
        idx = np.linspace(0, T - 1, target_T).round().astype(int)
        return arr[:, idx].astype(np.float64)
    # Non-overlapping block max-pool: split [0, T] into target_T blocks.
    edges = np.linspace(0, T, target_T + 1).astype(int)
    # Guarantee at least one sample per block.
    edges[1:] = np.maximum(edges[1:], edges[:-1] + 1)
    edges[-1] = T
    out = np.empty((F, target_T), dtype=np.float64)
    for t in range(target_T):
        a, b = int(edges[t]), int(edges[t + 1])
        out[:, t] = arr[:, a:b].max(axis=-1)
    return out


def _linear_resize_freq(arr: np.ndarray, target_F: int) -> np.ndarray:
    """1-D linear interpolation along the first (frequency) axis."""
    cur_F, T = arr.shape
    if cur_F == target_F:
        return arr.astype(np.float64)
    f_src = np.linspace(0.0, 1.0, cur_F)
    f_dst = np.linspace(0.0, 1.0, target_F)
    out = np.empty((target_F, T), dtype=np.float64)
    for c in range(T):
        out[:, c] = np.interp(f_dst, f_src, arr[:, c])
    return out


__all__ = ["compute_encoder_input_stack", "compute_log_mel_spectrogram"]
