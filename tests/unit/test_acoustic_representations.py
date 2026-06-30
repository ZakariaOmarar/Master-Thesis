from __future__ import annotations

import numpy as np

from src.features import (
    build_cwt_mfcc_encoder_input,
    compute_cwt_scalogram_stack,
    compute_mfcc_stack,
    compute_stft_stack,
)


def _build_test_mics(
    n_mics: int = 3, fs: int = 16_000, duration_s: float = 1.0
) -> np.ndarray:
    t = np.arange(int(fs * duration_s), dtype=np.float64) / fs
    base = 0.2 * np.sin(2 * np.pi * 240.0 * t) + 0.05 * np.sin(2 * np.pi * 1_200.0 * t)
    return np.stack([np.roll(base, i * 11) for i in range(n_mics)], axis=0)


def test_cwt_scalogram_stack_shape_no_decimation() -> None:
    """Opt-out path: ``decimate_to_hz=None`` keeps the original time axis."""
    fs = 16_000
    mic = _build_test_mics(fs=fs, duration_s=0.5)

    cwt = compute_cwt_scalogram_stack(
        mic, fs, n_scales=32, decimate_to_hz=None, max_freq_hz=None
    )

    assert cwt.shape == (mic.shape[0], 32, mic.shape[1])
    assert np.all(np.isfinite(cwt))
    assert np.all(cwt >= 0.0)


def test_cwt_scalogram_stack_decimates_by_default() -> None:
    """Publication default: ``decimate_to_hz=1000`` shortens the time axis
    by ~16× on a 16 kHz input.  This is the cost-control path that makes
    CWT tractable on D4's ~ 11-min healthy recordings (see Chapter 3
    §3.4.2 and REVIEW.md fix W).  Spectral content below 250 Hz —
    covering all five ROW II characteristic frequencies — is preserved
    by the anti-alias filter."""
    fs = 16_000
    duration_s = 0.5
    mic = _build_test_mics(fs=fs, duration_s=duration_s)

    cwt = compute_cwt_scalogram_stack(mic, fs, n_scales=32)
    # decimate_to_hz default = 1000 → resample_poly with up=1, down=16 →
    # ~ 500 samples for a 0.5 s clip.
    assert cwt.shape[0] == mic.shape[0]
    assert cwt.shape[1] == 32
    expected_len = int(round(duration_s * 1000))
    assert abs(cwt.shape[2] - expected_len) <= 2
    assert np.all(np.isfinite(cwt))
    assert np.all(cwt >= 0.0)


def test_mfcc_stack_with_deltas_shape() -> None:
    fs = 16_000
    mic = _build_test_mics(fs=fs, duration_s=1.0)

    mfcc = compute_mfcc_stack(
        mic,
        fs,
        n_mfcc=20,
        n_fft=1024,
        hop_length=256,
    )

    assert mfcc.shape[0] == mic.shape[0]
    assert mfcc.shape[1] == 60
    assert mfcc.shape[2] > 0
    assert np.all(np.isfinite(mfcc))


def test_build_cwt_mfcc_encoder_input_aligns_and_concatenates() -> None:
    fs = 16_000
    mic = _build_test_mics(fs=fs, duration_s=1.0)

    # Use the no-decimation path here so the time axes of CWT and MFCC are
    # both anchored on the original 16 kHz time grid; the function under
    # test (`build_cwt_mfcc_encoder_input`) itself does the cross-rate
    # alignment via `min(cwt.shape[2], mfcc.shape[2])`.
    cwt = compute_cwt_scalogram_stack(
        mic, fs, n_scales=32, decimate_to_hz=None, max_freq_hz=None
    )
    mfcc = compute_mfcc_stack(
        mic,
        fs,
        n_mfcc=20,
        n_fft=1024,
        hop_length=256,
    )

    fused = build_cwt_mfcc_encoder_input(cwt, mfcc)

    assert fused.shape[0] == mic.shape[0] * 2
    assert fused.shape[1] == cwt.shape[1]
    assert fused.shape[2] == min(cwt.shape[2], mfcc.shape[2])
    assert np.all(np.isfinite(fused))


def test_stft_stack_baseline_shape() -> None:
    mic = _build_test_mics(duration_s=1.0)

    stft = compute_stft_stack(mic, n_fft=1024, hop_length=256)

    assert stft.shape[0] == mic.shape[0]
    assert stft.shape[1] == 513
    assert stft.shape[2] > 0
    assert np.all(np.isfinite(stft))
