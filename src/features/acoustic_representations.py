"""Acoustic time-frequency representations for encoder-oriented workflows.

CWT scalograms are the primary input planned for the cross-attention
transformer being developed in the thesis — they preserve both temporal
and frequency structure that plain STFT spectrograms smear around the
low-frequency turbine tones (≤ 125 Hz on the ROW II machine).  MFCCs
provide a compact perceptual alternative used by the V0 LightGBM mode
classifier.  The STFT utility is kept for classical baselines and
direct comparison with published benchmarks.

**Analysis-grid convention.**  Every spectral primitive here defaults
to ``n_fft = 1024`` (64 ms at 16 kHz) and ``hop_length = 512`` (32 ms
stride) so MFCC, STFT, and the log-mel + CWT encoder stack
(`audio_spectral.compute_encoder_input_stack`) share an identical
time-frequency analysis grid.  This is required for thesis comparisons
across V0 baselines and the V1 / V2 encoder — different ``n_fft`` /
``hop_length`` would confound any feature-set ablation.  Power-magnitude
representations use ``power_to_db`` (dB) compression with ``ref=1.0``
(absolute, no per-recording gain normalisation) so cross-recording
amplitude information survives.
"""

from __future__ import annotations

import numpy as np

from ..config.architecture import ACOUSTIC_CWT, ACOUSTIC_FEATURES


def compute_cwt_scalogram(
    signal: np.ndarray,
    fs: int,
    *,
    wavelet: str = ACOUSTIC_CWT.wavelet,
    n_scales: int = ACOUSTIC_CWT.n_scales,
    min_freq_hz: float = ACOUSTIC_CWT.min_freq_hz,
    max_freq_hz: float | None = ACOUSTIC_CWT.max_freq_hz,
    decimate_to_hz: int | None = ACOUSTIC_CWT.decimate_to_hz,
) -> np.ndarray:
    """Compute a log-compressed CWT scalogram for a single acoustic channel.

    Returns a 2D array with shape ``(n_scales, n_decimated_samples)``.

    Two cost-control parameters that materially affect CPU / memory on
    long recordings (e.g. D4 ~ 11-min waveforms at 16 kHz):

    - ``decimate_to_hz``: if not None and ``fs > decimate_to_hz``, the
      signal is anti-alias-filtered and downsampled to approximately
      ``decimate_to_hz`` (via ``scipy.signal.resample_poly``) before
      ``pywt.cwt`` is called.  CWT memory is O(n_scales · n_samples)
      complex coefficients, so decimating 16 kHz → 1 kHz cuts memory by
      ~ 16× and runtime by a similar factor.  All five characteristic
      ROW II frequencies (≤ 125 Hz; see Chapter 3 §3.1) lie comfortably
      below the 500 Hz Nyquist of the decimated stream.

    - ``max_freq_hz``: caps the upper end of the scale geometric progression.
      Default 250 Hz — sufficient to cover the 125 Hz guide-vane-passing
      tone with one octave of margin.  CWT energy above this is not used
      by the V1 / V2 encoders and need not be computed; ``None`` reverts
      to the prior behaviour (geometric grid up to Nyquist).
    """
    import pywt

    if fs <= 0:
        raise ValueError("fs must be > 0")
    if n_scales <= 1:
        raise ValueError("n_scales must be > 1")
    if min_freq_hz <= 0:
        raise ValueError("min_freq_hz must be > 0")

    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    if x.size == 0:
        raise ValueError("signal cannot be empty")

    # Anti-alias decimation before CWT.  scipy.signal.resample_poly applies
    # an FIR low-pass filter at the new Nyquist before downsampling, so
    # energy above `decimate_to_hz / 2` is removed cleanly — consistent
    # with ``min_freq_hz`` and the 250 Hz upper edge of our scale grid.
    if decimate_to_hz is not None and fs > decimate_to_hz:
        # Greatest-common-divisor-based ratio to keep up/down small.
        from math import gcd

        from scipy.signal import resample_poly
        target = int(decimate_to_hz)
        g = gcd(int(fs), target)
        up = target // g
        down = int(fs) // g
        x = resample_poly(x, up=up, down=down).astype(np.float64)
        eff_fs = float(fs) * up / down
    else:
        eff_fs = float(fs)

    nyquist = eff_fs / 2.0
    if min_freq_hz >= nyquist:
        raise ValueError("min_freq_hz must be below the (post-decimation) Nyquist")
    upper = nyquist if max_freq_hz is None else min(float(max_freq_hz), nyquist)
    if upper <= min_freq_hz:
        raise ValueError("max_freq_hz must be > min_freq_hz")

    freqs = np.geomspace(min_freq_hz, upper, n_scales)
    scales = pywt.frequency2scale(wavelet, freqs / eff_fs)
    coefficients, _ = pywt.cwt(
        x,
        scales,
        wavelet,
        sampling_period=1.0 / eff_fs,
    )

    scalogram = np.log1p(np.abs(coefficients))
    return scalogram.astype(np.float64)


def compute_cwt_scalogram_stack(
    mic_data: np.ndarray,
    fs: int,
    *,
    wavelet: str = ACOUSTIC_CWT.wavelet,
    n_scales: int = ACOUSTIC_CWT.n_scales,
    min_freq_hz: float = ACOUSTIC_CWT.min_freq_hz,
    max_freq_hz: float | None = ACOUSTIC_CWT.max_freq_hz,
    decimate_to_hz: int | None = ACOUSTIC_CWT.decimate_to_hz,
) -> np.ndarray:
    """Compute CWT scalograms for all mic channels.

    Returns a 3D array with shape ``(n_mics, n_scales, n_decimated_samples)``.
    See ``compute_cwt_scalogram`` for the cost-control parameters.
    """
    data = _as_2d_channels(mic_data, name="mic_data")
    out = [
        compute_cwt_scalogram(
            data[i],
            fs,
            wavelet=wavelet,
            n_scales=n_scales,
            min_freq_hz=min_freq_hz,
            max_freq_hz=max_freq_hz,
            decimate_to_hz=decimate_to_hz,
        )
        for i in range(data.shape[0])
    ]
    return np.stack(out, axis=0)


def compute_mfcc_with_deltas(
    signal: np.ndarray,
    fs: int,
    *,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute MFCC, delta, and delta-delta for one channel.

    Returns a 2D array with shape ``(3 * n_mfcc, n_frames)``.  Defaults
    match :func:`audio_spectral.compute_log_mel_spectrogram` so the V0
    LightGBM mode classifier and the V1 / V2 SSL encoder train on
    the same analysis grid; without this alignment a baseline-vs-
    SSL comparison would be confounded by the analysis window.
    Delta and delta-delta are computed with librosa's default 9-frame
    Savitzky-Golay-style estimator.
    """
    import librosa

    if fs <= 0:
        raise ValueError("fs must be > 0")
    if n_mfcc <= 0:
        raise ValueError("n_mfcc must be > 0")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be > 0")

    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    if x.size < n_fft:
        raise ValueError("signal length must be >= n_fft")

    mfcc = librosa.feature.mfcc(
        y=x.astype(np.float32),
        sr=fs,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.concatenate([mfcc, delta, delta2], axis=0).astype(np.float64)


def compute_mfcc_stack(
    mic_data: np.ndarray,
    fs: int,
    *,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute stacked MFCC(+delta/+delta-delta) across mic channels.

    Returns a 3D array with shape ``(n_mics, 3 * n_mfcc, n_frames)``.
    """
    data = _as_2d_channels(mic_data, name="mic_data")
    out = [
        compute_mfcc_with_deltas(
            data[i],
            fs,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        for i in range(data.shape[0])
    ]
    return np.stack(out, axis=0)


def compute_log_stft_spectrogram(
    signal: np.ndarray,
    *,
    n_fft: int = ACOUSTIC_FEATURES.n_fft,
    hop_length: int = ACOUSTIC_FEATURES.hop_length,
    top_db: float = ACOUSTIC_FEATURES.top_db,
) -> np.ndarray:
    """Compute power-to-dB STFT spectrogram for one channel.

    Classical baseline representation, in dB rather than ``log1p`` so
    the dynamic range matches :func:`audio_spectral.compute_log_mel_spectrogram`
    and the published audio-classification baselines this thesis compares
    against.  ``ref=1.0`` (no peak normalisation) preserves cross-recording
    amplitude; ``top_db`` clips the noise floor to avoid -∞ on silent bins.
    Defaults match the log-mel / encoder grid (``n_fft = 1024``,
    ``hop_length = 512``).
    """
    import librosa

    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be > 0")

    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    if x.size < n_fft:
        raise ValueError("signal length must be >= n_fft")

    stft = librosa.stft(x.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    power = np.abs(stft) ** 2
    return librosa.power_to_db(power, ref=1.0, top_db=top_db).astype(np.float64)


def compute_stft_stack(
    mic_data: np.ndarray,
    *,
    n_fft: int = ACOUSTIC_FEATURES.n_fft,
    hop_length: int = ACOUSTIC_FEATURES.hop_length,
) -> np.ndarray:
    """Compute stacked log-STFT representations for all mic channels."""
    data = _as_2d_channels(mic_data, name="mic_data")
    out = [
        compute_log_stft_spectrogram(data[i], n_fft=n_fft, hop_length=hop_length)
        for i in range(data.shape[0])
    ]
    return np.stack(out, axis=0)


def build_cwt_mfcc_encoder_input(
    cwt_stack: np.ndarray,
    mfcc_stack: np.ndarray,
    *,
    target_freq_bins: int | None = None,
    target_time_steps: int | None = None,
) -> np.ndarray:
    """Fuse CWT and MFCC stacks into one encoder tensor.

    The output shape is ``(2 * n_mics, freq_bins, time_steps)`` where the first
    ``n_mics`` channels are CWT and the next ``n_mics`` channels are MFCC.
    """
    cwt = _as_3d_repr(cwt_stack, name="cwt_stack")
    mfcc = _as_3d_repr(mfcc_stack, name="mfcc_stack")

    if cwt.shape[0] != mfcc.shape[0]:
        raise ValueError("cwt_stack and mfcc_stack must have the same n_mics")

    time_steps = (
        int(target_time_steps)
        if target_time_steps is not None
        else min(cwt.shape[2], mfcc.shape[2])
    )
    if time_steps <= 0:
        raise ValueError("target_time_steps must be > 0")

    freq_bins = (
        int(target_freq_bins) if target_freq_bins is not None else int(cwt.shape[1])
    )
    if freq_bins <= 0:
        raise ValueError("target_freq_bins must be > 0")

    cwt_aligned = _resample_axis(cwt, axis=2, new_size=time_steps)
    mfcc_aligned = _resample_axis(mfcc, axis=2, new_size=time_steps)
    cwt_aligned = _resample_axis(cwt_aligned, axis=1, new_size=freq_bins)
    mfcc_aligned = _resample_axis(mfcc_aligned, axis=1, new_size=freq_bins)

    return np.concatenate([cwt_aligned, mfcc_aligned], axis=0).astype(np.float64)


def _as_2d_channels(data: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (channels, samples)")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _as_3d_repr(data: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (channels, bins, steps)")
    if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _resample_axis(data: np.ndarray, *, axis: int, new_size: int) -> np.ndarray:
    if new_size <= 0:
        raise ValueError("new_size must be > 0")
    if data.shape[axis] == new_size:
        return data.astype(np.float64)

    moved = np.moveaxis(data, axis, -1)
    old_size = moved.shape[-1]
    flat = moved.reshape(-1, old_size)

    src = np.linspace(0.0, 1.0, num=old_size)
    dst = np.linspace(0.0, 1.0, num=new_size)

    out = np.empty((flat.shape[0], new_size), dtype=np.float64)
    for i in range(flat.shape[0]):
        out[i] = np.interp(dst, src, flat[i])

    out = out.reshape(*moved.shape[:-1], new_size)
    return np.moveaxis(out, -1, axis)
