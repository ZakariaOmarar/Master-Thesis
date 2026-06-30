"""Smoke tests for the V1 encoder-input feature extractors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.features.audio_spectral import (
    compute_encoder_input_stack,
    compute_log_mel_spectrogram,
)
from src.features.vibration_temporal import (
    channel2_statistic_name,
    compute_vibration_input_stack,
)
from src.ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.requires_data


def _spec(name: str) -> DatasetSpec:
    # `DatasetSpec.from_yaml` resolves all paths to absolute — no reconstruction needed.
    return DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / f"{name}.yaml")


def _short_segment(name: str, max_seconds: float = 4.0):
    """Load one segment and crop to the first ``max_seconds`` to keep feature
    extraction tests fast."""
    loader = TestDatasetLoader(_spec(name))
    seg = loader.list_segments()[0]
    n_mic_keep = int(round(max_seconds * seg.segment.mic_sample_rate))
    n_vib_keep = max(8, int(round(max_seconds * seg.segment.accel_sample_rate)))
    mic = seg.segment.mic_data[:, :n_mic_keep]
    vib = seg.segment.accel_data[:, :n_vib_keep]
    return seg, mic, vib


def test_log_mel_single_channel_shape() -> None:
    rng = np.random.default_rng(0)
    fs = 16_000
    x = rng.standard_normal(fs * 2)  # 2 s of noise
    mel = compute_log_mel_spectrogram(
        x, fs, n_mels=64, n_fft=1024, hop_length=256, top_db=80.0
    )
    assert mel.ndim == 2
    assert mel.shape[0] == 64
    assert mel.shape[1] >= 1
    assert np.all(np.isfinite(mel))
    # Power-to-dB output is bounded above by 0 dB only for ref=max; with
    # ref=1.0 (absolute) the peak may exceed 0 dB.  The hard guarantee is
    # the top_db floor: every bin must lie within top_db of the max.
    peak = float(mel.max())
    assert float(mel.min()) >= peak - 80.0 - 1e-6


@pytest.mark.parametrize("name", ["d1", "d2", "d3"])
def test_encoder_input_stack_shape(name: str) -> None:
    seg, mic, _ = _short_segment(name)
    stack = compute_encoder_input_stack(
        mic,
        fs=seg.segment.mic_sample_rate,
        n_mels=64,
        n_fft=1024,
        hop_length=512,
        cwt_n_scales=64,
    )
    assert stack.ndim == 4
    assert stack.shape[0] == seg.segment.n_mic_channels
    assert stack.shape[1] == 2  # (log-mel, CWT)
    assert stack.shape[2] == 64  # frequency axis
    assert stack.shape[3] >= 1
    assert np.all(np.isfinite(stack))
    # log-mel and CWT live on the same (F, T) grid
    assert stack[0, 0].shape == stack[0, 1].shape


@pytest.mark.parametrize("name", ["d1", "d2", "d3"])
def test_vibration_input_stack_shape(name: str) -> None:
    seg, _, vib = _short_segment(name)
    fs = float(seg.segment.accel_sample_rate)
    # Default (standardize=True): per-channel z-score on amplitude + envelope.
    stack = compute_vibration_input_stack(vib, sample_rate=fs)
    assert stack.ndim == 3
    assert stack.shape[0] == seg.segment.n_accel_channels
    assert stack.shape[1] == 3  # (amplitude, envelope, impulsiveness)
    assert stack.shape[2] == vib.shape[1]
    assert np.all(np.isfinite(stack))
    # Standardised channels (0=amplitude, 1=envelope) are zero-mean per channel.
    if vib.shape[1] >= 5:
        amp_means = stack[:, 0, :].mean(axis=-1)
        env_means = stack[:, 1, :].mean(axis=-1)
        assert np.allclose(amp_means, 0.0, atol=1e-3)
        assert np.allclose(env_means, 0.0, atol=1e-3)


@pytest.mark.parametrize("name", ["d1", "d2", "d3"])
def test_vibration_input_stack_legacy_unstandardized(name: str) -> None:
    """Setting `standardize=False` reproduces the pre-2026-05 envelope
    invariants — useful for comparison against legacy checkpoints."""
    seg, _, vib = _short_segment(name)
    stack = compute_vibration_input_stack(
        vib, sample_rate=float(seg.segment.accel_sample_rate), standardize=False
    )
    assert np.all(stack[:, 1, :] >= 0.0)  # envelope = |hilbert(x)| ≥ 0


@pytest.mark.parametrize("name", ["d1", "d2", "d3"])
def test_vibration_impulsiveness_is_raw_regardless_of_standardize(name: str) -> None:
    """Channel 2 (impulsiveness — kurtosis OR crest factor depending on
    sample rate) is dimensionless and kept RAW.  The ``standardize`` flag
    only affects amplitude + envelope (channels 0, 1).  The F5 audit
    experiment z-scored channel 2 and found no V1 vibration cluster-quality
    benefit, so it was reverted to the known-good raw form."""
    seg, _, vib = _short_segment(name)
    if vib.shape[1] < 20:
        pytest.skip("segment too short to evaluate rolling impulsiveness stats")
    fs = float(seg.segment.accel_sample_rate)

    stack_std = compute_vibration_input_stack(vib, sample_rate=fs, standardize=True)
    stack_raw = compute_vibration_input_stack(vib, sample_rate=fs, standardize=False)

    # Channel 2 must be byte-identical with and without standardize.
    np.testing.assert_array_equal(stack_std[:, 2, :], stack_raw[:, 2, :])


def test_channel2_mode_selection_per_sample_rate() -> None:
    """The per-sample-rate channel-2 statistic must match the dataset map
    documented in `compute_vibration_input_stack`: D4 raw (~376 Hz) gets
    kurtosis, D1/D2/D3 peak streams (4/16 Hz) get crest factor.  This
    contract is what lets the V1/V2/V4 encoders share weights across
    datasets while channel 2 carries the right impulsiveness statistic
    for the underlying signal."""
    assert channel2_statistic_name(4.0) == "crest_factor"
    assert channel2_statistic_name(16.0) == "crest_factor"
    assert channel2_statistic_name(376.0) == "kurtosis"


def test_rolling_kurtosis_detects_synthetic_impulse() -> None:
    """At 376 Hz a 5 ms impulsive spike on Gaussian background must lift
    channel 2 (kurtosis) far above its background level near the spike.
    This is the basic discrimination property the channel is for."""
    rng = np.random.default_rng(0)
    fs = 376.0
    T = int(fs * 4.0)  # 4 seconds
    x = rng.standard_normal((1, T)) * 0.05
    spike_idx = T // 2
    x[0, spike_idx] += 5.0  # ~100× background std
    stack = compute_vibration_input_stack(
        x, sample_rate=fs, standardize=True
    )
    # Channel 2 should peak inside the (half-window) of the spike location.
    half_window = int(round(0.1 * fs)) // 2
    peak_zone = stack[0, 2, spike_idx - half_window : spike_idx + half_window + 1]
    background = np.concatenate(
        [
            stack[0, 2, : spike_idx - 2 * half_window],
            stack[0, 2, spike_idx + 2 * half_window :],
        ]
    )
    assert float(peak_zone.max()) > float(background.max()) + 2.0


def test_compute_vibration_requires_sample_rate() -> None:
    """sample_rate is a required keyword arg — passing 0 / negative raises."""
    vib = np.zeros((2, 64), dtype=np.float64)
    with pytest.raises(ValueError, match="sample_rate"):
        compute_vibration_input_stack(vib, sample_rate=0.0)
    with pytest.raises(ValueError, match="sample_rate"):
        compute_vibration_input_stack(vib, sample_rate=-4.0)


@pytest.mark.parametrize("name", ["d1", "d2", "d3"])
def test_acoustic_stack_is_per_channel_zscored(name: str) -> None:
    """F4 — log-mel (channel 0) and CWT (channel 1) are each z-scored across
    (mics, frequency, time) when ``standardize=True``.  Under the legacy
    ``standardize=False`` path they retain their raw (different-scale) ranges."""
    seg, mic, _ = _short_segment(name)
    stack_z = compute_encoder_input_stack(
        mic,
        fs=seg.segment.mic_sample_rate,
        n_mels=32,
        n_fft=512,
        hop_length=256,
        cwt_n_scales=24,
        standardize=True,
    )
    stack_raw = compute_encoder_input_stack(
        mic,
        fs=seg.segment.mic_sample_rate,
        n_mels=32,
        n_fft=512,
        hop_length=256,
        cwt_n_scales=24,
        standardize=False,
    )
    for ch in (0, 1):
        ch_z = stack_z[:, ch, :, :]
        ch_raw = stack_raw[:, ch, :, :]
        assert abs(float(ch_z.mean())) < 1e-3
        # Unit variance (non-degenerate input — real mic data is never flat).
        assert abs(float(ch_z.std()) - 1.0) < 1e-2
        # Raw path: dB-compressed log-mel (channel 0) can be negative;
        # log1p-compressed CWT (channel 1) is non-negative.  Both must be
        # finite.
        assert np.all(np.isfinite(ch_raw))
    # log-mel (dB) is bounded by top_db = 80 dB below its peak.
    mel_raw = stack_raw[:, 0, :, :]
    assert float(mel_raw.min()) >= float(mel_raw.max()) - 80.0 - 1e-6
    # CWT (log1p of |coefficients|) is non-negative.
    cwt_raw = stack_raw[:, 1, :, :]
    assert float(cwt_raw.min()) >= 0.0
    # The two channels in the z-scored stack should now have comparable
    # dynamic ranges (within an order of magnitude).
    mel_range = float(np.ptp(stack_z[:, 0, :, :]))
    cwt_range = float(np.ptp(stack_z[:, 1, :, :]))
    ratio = max(mel_range, cwt_range) / max(min(mel_range, cwt_range), 1e-6)
    assert ratio < 10.0, f"channel range mismatch after z-score: ratio={ratio:.2f}"
