"""Unit tests for the 2026-05-18 ingestion audit fixes.

Each test exercises one specific fix from the audit and locks in the
new invariant.  Keeping them in a dedicated file makes the audit
traceable — if any of these regresses, the fix has been undone.

Coverage:
  * Issue 1  — per-channel raw-vibration resampling preserves inter-
               channel timing under per-board clock divergence.
  * Issue 2  — DataSegment.start_time is parsed from CSV pc_time or
               file mtime, not load wall clock.
  * Issue 4  — D1 placeholder geometry sits inside the prototype
               metric space, not at 1 m / 2 m.
  * Issue 5  — filter_vibration_csv_paths partitions raw vs peak.
  * Issue 7  — apply_sync_correction errors loudly on shifts > duration.
  * Issue 8  — _read_vibration_csv errors when the timestamp column is
               missing or has empty values.
  * Issue 9  — peak-vibration rate inference is median-across-channels,
               not channel-0-only.
  * Issue 10 — float-WAV scale detection rescales integer-scale-in-
               float containers.
  * Issue 3  — WavVibrationAdapter sync_correct flag is opt-in and
               propagates a report into metadata.

Issue 6 is documentation-only (no behaviour change).
"""

from __future__ import annotations

import csv
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from src.exceptions import IngestionError
from src.ingestion.adapters import (
    WavVibrationAdapter,
    _infer_recording_start_time,
    _read_first_pc_time,
    _read_vibration_csv,
    _resample_channels_per_channel_rate,
    filter_vibration_csv_paths,
)
from src.ingestion.positions import PositionRegistry
from src.ingestion.sync_verification import apply_sync_correction

# ---------------------------------------------------------------------------
# Synthetic-data helpers (mirror the patterns already used in the
# existing tests/unit/test_ingestion_dual_layout.py)
# ---------------------------------------------------------------------------


def _write_wav_int16(path: Path, sr: int = 16_000, duration_s: float = 0.5) -> None:
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    x = (0.2 * np.sin(2 * np.pi * 220.0 * t) * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, x)


def _write_peak_vibration_csv(
    path: Path, n_rows: int = 16, dt_us: int = 250_000, with_pc_time: bool = False
) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = ["esp_time_us", "amplitude", "frequency"]
        if with_pc_time:
            fieldnames = ["pc_time"] + fieldnames
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_rows):
            row = {
                "esp_time_us": 1_000_000 + i * dt_us,
                "amplitude": 100.0 + i,
                "frequency": 125.0,
            }
            if with_pc_time:
                row["pc_time"] = 1_700_000_000.0 + i * (dt_us / 1_000_000.0)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Issue 5 — DRY raw/peak partitioner
# ---------------------------------------------------------------------------


def test_filter_vibration_csv_paths_partitions_raw_vs_peak() -> None:
    paths = [
        Path("vibration_D.csv"),
        Path("vibration_E.csv"),
        Path("vibration_raw_D.csv"),
        Path("vibration_raw_E.csv"),
    ]
    peak = filter_vibration_csv_paths(paths, "peak")
    raw = filter_vibration_csv_paths(paths, "raw")
    assert {p.name for p in peak} == {"vibration_D.csv", "vibration_E.csv"}
    assert {p.name for p in raw} == {"vibration_raw_D.csv", "vibration_raw_E.csv"}


def test_filter_vibration_csv_paths_rejects_unknown_format() -> None:
    with pytest.raises(IngestionError, match="vibration_format"):
        filter_vibration_csv_paths([], "bogus")


# ---------------------------------------------------------------------------
# Issue 1 — per-channel raw vibration resampling preserves inter-channel timing
# ---------------------------------------------------------------------------


def test_resample_per_channel_rate_places_impulse_at_same_output_index() -> None:
    """Two channels captured at deliberately different rates, each with an
    impulse at the same wall-clock instant, must land that impulse at the
    same OUTPUT sample after per-channel resampling — that is the inter-
    channel timing invariant the V4 structure-borne TDOA head depends on.
    """
    impulse_t_s = 1.000  # wall-clock instant of the impulse
    dst_rate = 376.0
    duration_s = 2.0
    src_rates = np.asarray([376.0, 404.0])  # 7 % per-board divergence (real D4 case)

    n_in_ch0 = int(round(duration_s * src_rates[0]))
    n_in_ch1 = int(round(duration_s * src_rates[1]))
    n_in_common = min(n_in_ch0, n_in_ch1)

    ch0 = np.zeros(n_in_common)
    ch1 = np.zeros(n_in_common)
    # Place a 1.0 impulse at the same WALL-CLOCK instant in each channel,
    # converted to that channel's native sample index.
    ch0[int(round(impulse_t_s * src_rates[0]))] = 1.0
    ch1[int(round(impulse_t_s * src_rates[1]))] = 1.0
    data = np.stack([ch0, ch1])

    n_out = int(round(duration_s * dst_rate))
    out = _resample_channels_per_channel_rate(data, src_rates, n_out, dst_rate)

    # After per-channel resampling, the impulse should land at output sample
    # round(impulse_t_s * dst_rate) in BOTH channels.
    expected_out_idx = int(round(impulse_t_s * dst_rate))
    out_idx_ch0 = int(np.argmax(out[0]))
    out_idx_ch1 = int(np.argmax(out[1]))
    # Linear interpolation can split a delta across two adjacent output bins;
    # the argmax can be off by 1 from the exact ideal index.
    assert abs(out_idx_ch0 - expected_out_idx) <= 1
    assert abs(out_idx_ch1 - expected_out_idx) <= 1
    # And — critical TDOA-correctness invariant — the two channels' impulses
    # must land at the same output index (within 1 sample).
    assert abs(out_idx_ch0 - out_idx_ch1) <= 1


def test_resample_per_channel_rate_shape_check() -> None:
    data = np.zeros((3, 100))
    with pytest.raises(IngestionError, match="src_rates"):
        _resample_channels_per_channel_rate(
            data, np.asarray([100.0, 100.0]), 50, 50.0  # wrong length
        )


def test_resample_per_channel_rate_rejects_nonpositive_rate() -> None:
    data = np.zeros((2, 100))
    with pytest.raises(IngestionError, match="non-positive"):
        _resample_channels_per_channel_rate(
            data, np.asarray([100.0, 0.0]), 50, 50.0
        )


# ---------------------------------------------------------------------------
# Issue 2 — start_time from pc_time / mtime, not load wall clock
# ---------------------------------------------------------------------------


def test_start_time_uses_pc_time_when_available(tmp_path: Path) -> None:
    csv_path = tmp_path / "vibration_X.csv"
    _write_peak_vibration_csv(csv_path, n_rows=4, with_pc_time=True)
    t = _read_first_pc_time(csv_path)
    assert t is not None
    assert t.tzinfo is not None
    # The synthetic pc_time starts at 1_700_000_000 ≈ 2023-11-14 UTC.
    assert t.year == 2023


def test_start_time_falls_back_to_mtime_without_pc_time(tmp_path: Path) -> None:
    csv_path = tmp_path / "vibration_X.csv"
    _write_peak_vibration_csv(csv_path, n_rows=4, with_pc_time=False)
    wav_path = tmp_path / "recorded_X.wav"
    _write_wav_int16(wav_path)
    # Pin the WAV mtime to a known value (~ 2024-01-01) so the test is
    # deterministic against whatever the OS picked.
    fixed_mtime = datetime(2024, 1, 1, tzinfo=UTC).timestamp()
    os.utime(wav_path, (fixed_mtime, fixed_mtime))
    os.utime(csv_path, (fixed_mtime + 60, fixed_mtime + 60))
    start, source = _infer_recording_start_time([csv_path], [wav_path])
    assert source == "file_mtime"
    # Earliest mtime wins, and it must be tz-aware UTC.
    assert start.tzinfo is not None
    assert abs(start.timestamp() - fixed_mtime) < 1.0


def test_start_time_falls_back_to_load_time_when_no_files() -> None:
    before = datetime.now(UTC).timestamp()
    start, source = _infer_recording_start_time([], [])
    after = datetime.now(UTC).timestamp()
    assert source == "load_time"
    assert before - 1 <= start.timestamp() <= after + 1


def test_adapter_propagates_start_time_into_segment(tmp_path: Path) -> None:
    rec = tmp_path / "rec1"
    rec.mkdir()
    for sensor in "BCDE":
        _write_wav_int16(rec / f"recorded_{sensor}.wav")
        _write_peak_vibration_csv(rec / f"vibration_{sensor}.csv")
    fixed_mtime = datetime(2024, 6, 15, tzinfo=UTC).timestamp()
    for p in rec.iterdir():
        os.utime(p, (fixed_mtime, fixed_mtime))

    adapter = WavVibrationAdapter()
    segment = adapter.read_recording_directory(rec)
    # start_time should NOT be wall-clock-now (within a second of test start)
    now = datetime.now(UTC)
    assert abs((segment.start_time - now).total_seconds()) > 60 * 60 * 24
    # Should be tz-aware
    assert segment.start_time.tzinfo is not None
    # And the provenance tag must be in metadata
    assert segment.metadata["start_time_source"] in ("pc_time", "file_mtime")


# ---------------------------------------------------------------------------
# Issue 4 — D1 placeholder geometry at prototype scale
# ---------------------------------------------------------------------------


def test_d1_placeholder_geometry_at_prototype_scale() -> None:
    reg = PositionRegistry.from_source("default")
    mic_xyz = np.stack([reg.lookup_mic(s) for s in "BCDE"])
    vib_xyz = np.stack([reg.lookup_vibration(s) for s in "BCDE"])
    # All coordinates must fit inside the ~ 0.20 m prototype envelope.
    # The legacy 1 m / 2 m geometry would have failed this assertion.
    assert float(np.max(np.abs(mic_xyz))) <= 0.30
    assert float(np.max(np.abs(vib_xyz))) <= 0.30
    # Mic ring must have non-trivial radius (not collapsed to origin).
    assert float(np.linalg.norm(mic_xyz[:, :2], axis=-1).mean()) > 0.05


# ---------------------------------------------------------------------------
# Issue 7 — apply_sync_correction errors on oversize shifts
# ---------------------------------------------------------------------------


def test_apply_sync_correction_errors_on_oversize_mic_shift() -> None:
    mic = np.zeros((4, 100))  # 100-sample mic stream
    accel = np.zeros((4, 10))
    # Offset of 10 s at 16 kHz → drop 160 000 samples; way more than 100.
    with pytest.raises(ValueError, match="exceeds recording duration"):
        apply_sync_correction(mic, accel, mic_fs=16000, accel_fs=4, offset_s=10.0)


def test_apply_sync_correction_errors_on_oversize_vib_shift() -> None:
    mic = np.zeros((4, 16_000))
    accel = np.zeros((4, 10))  # 10-sample vibration stream
    # Offset of -10 s at 4 Hz → drop 40 samples; more than 10.
    with pytest.raises(ValueError, match="exceeds recording duration"):
        apply_sync_correction(mic, accel, mic_fs=16000, accel_fs=4, offset_s=-10.0)


# ---------------------------------------------------------------------------
# Issue 8 — missing-timestamp CSVs error explicitly
# ---------------------------------------------------------------------------


def test_read_vibration_csv_errors_without_time_column(tmp_path: Path) -> None:
    path = tmp_path / "vibration_X.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["amplitude", "frequency"])
        writer.writeheader()
        for i in range(4):
            writer.writerow({"amplitude": 100.0 + i, "frequency": 125.0})
    with pytest.raises(IngestionError, match="timestamp column"):
        _read_vibration_csv(path)


def test_read_vibration_csv_errors_on_empty_time_value(tmp_path: Path) -> None:
    path = tmp_path / "vibration_X.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["esp_time_us", "amplitude", "frequency"]
        )
        writer.writeheader()
        writer.writerow(
            {"esp_time_us": 1_000_000, "amplitude": 100.0, "frequency": 125.0}
        )
        writer.writerow(
            {"esp_time_us": "", "amplitude": 101.0, "frequency": 125.0}
        )
    with pytest.raises(IngestionError, match="empty"):
        _read_vibration_csv(path)


# ---------------------------------------------------------------------------
# Issue 9 — peak-rate median across all channels
# ---------------------------------------------------------------------------


def test_peak_rate_uses_median_across_channels(tmp_path: Path) -> None:
    """If one channel's timestamps imply a different rate from the other
    three, the inferred stream rate must come from the median across all
    channels — not from channel 0 alone."""
    rec = tmp_path / "rec_med"
    rec.mkdir()
    # All channels at 16 kHz audio
    for sensor in "BCDE":
        _write_wav_int16(rec / f"recorded_{sensor}.wav", sr=16_000, duration_s=0.5)
    # Three channels at 4 Hz (dt = 250 ms); channel 0 simulates a clock
    # anomaly with dt = 200 ms (5 Hz).  Median rate should still be 4 Hz.
    _write_peak_vibration_csv(rec / "vibration_B.csv", n_rows=4, dt_us=200_000)
    for sensor in "CDE":
        _write_peak_vibration_csv(rec / f"vibration_{sensor}.csv", n_rows=4)

    adapter = WavVibrationAdapter()
    segment = adapter.read_recording_directory(rec)
    # The inferred peak-stream rate (in metadata) should be ~ 4 Hz (median),
    # NOT ~ 5 Hz (what channel 0 alone would say).
    raw_rate = float(segment.metadata["vibration_sample_rate_raw"])
    assert abs(raw_rate - 4.0) < 0.1


# ---------------------------------------------------------------------------
# Issue 10 — float-WAV integer-scale-in-float detection
# ---------------------------------------------------------------------------


def test_adapter_rescales_int16_scale_float_wav(tmp_path: Path) -> None:
    """A float32 WAV whose magnitudes sit in the int16 range (a common
    `ffmpeg -c:a pcm_f32le` artefact when converting from int16) must be
    rescaled by 1 / 32768 on load, with a warning, instead of silently
    boosting features by ~ 32 000×."""
    rec = tmp_path / "rec_floatwav"
    rec.mkdir()
    for sensor in "BCDE":
        # Write a float32 WAV with int16-scale values
        n = 16_000
        t = np.arange(n) / 16_000
        x = (0.2 * np.sin(2 * np.pi * 220.0 * t) * 32767.0).astype(np.float32)
        wavfile.write(str(rec / f"recorded_{sensor}.wav"), 16_000, x)
        _write_peak_vibration_csv(rec / f"vibration_{sensor}.csv")

    adapter = WavVibrationAdapter()
    with pytest.warns(UserWarning, match="outside the conventional"):
        segment = adapter.read_recording_directory(rec)
    # After rescaling, the mic data must be inside [-1, +1].
    assert float(np.max(np.abs(segment.mic_data))) < 1.0


def test_adapter_passes_through_well_scaled_float_wav(tmp_path: Path) -> None:
    """A float32 WAV with values in [-1, +1] is the conventional format and
    must NOT be rescaled or warned about."""
    import warnings

    rec = tmp_path / "rec_floatwav_ok"
    rec.mkdir()
    for sensor in "BCDE":
        n = 16_000
        t = np.arange(n) / 16_000
        x = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        wavfile.write(str(rec / f"recorded_{sensor}.wav"), 16_000, x)
        _write_peak_vibration_csv(rec / f"vibration_{sensor}.csv")

    adapter = WavVibrationAdapter()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here is a failure
        adapter.read_recording_directory(rec)


# ---------------------------------------------------------------------------
# Issue 3 — sync_correct opt-in flag in WavVibrationAdapter
# ---------------------------------------------------------------------------


def test_adapter_sync_correct_off_by_default(tmp_path: Path) -> None:
    rec = tmp_path / "rec_no_sync"
    rec.mkdir()
    for sensor in "BCDE":
        _write_wav_int16(rec / f"recorded_{sensor}.wav")
        _write_peak_vibration_csv(rec / f"vibration_{sensor}.csv")

    adapter = WavVibrationAdapter()
    segment = adapter.read_recording_directory(rec)
    # Without the flag, sync_correction metadata is None.
    assert segment.metadata["sync_correction"] is None


def test_adapter_sync_correct_attaches_report_when_enabled(tmp_path: Path) -> None:
    rec = tmp_path / "rec_with_sync"
    rec.mkdir()
    for sensor in "BCDE":
        _write_wav_int16(rec / f"recorded_{sensor}.wav", duration_s=2.0)
        _write_peak_vibration_csv(rec / f"vibration_{sensor}.csv", n_rows=8)

    adapter = WavVibrationAdapter(sync_correct=True)
    segment = adapter.read_recording_directory(rec)
    # Report is attached; on synthetic steady-tone data the four-gate pipeline
    # will refuse to correct (low envelope kurtosis), so `applied=False` is
    # the expected outcome but the report itself must be present.
    report = segment.metadata["sync_correction"]
    assert report is not None
    assert "applied" in report
    assert "reason" in report
    assert "audit_confidence" in report
    assert "acoustic_envelope_kurtosis" in report
