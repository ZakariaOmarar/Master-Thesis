from __future__ import annotations

import csv

import numpy as np
from scipy.io import wavfile

from src.ingestion import (
    RecordingScanner,
    SegmentLoader,
    WavVibrationAdapter,
)


def _write_wav(
    path, sr: int = 16_000, duration_s: float = 1.0, freq_hz: float = 220.0
) -> None:
    t = np.arange(int(sr * duration_s)) / sr
    x = (0.2 * np.sin(2 * np.pi * freq_hz * t) * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, x)


def _write_vibration_csv(path, rows: int = 8) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["esp_time_us", "amplitude", "frequency"]
        )
        writer.writeheader()
        for i in range(rows):
            writer.writerow(
                {
                    "esp_time_us": 1_000_000 + i * 250_000,
                    "amplitude": 100.0 + i,
                    "frequency": 125.0,
                }
            )


def test_scanner_groups_flat_layout_and_loader_accepts_4_mics(tmp_path) -> None:
    for mode in ["Pump", "Turbine"]:
        for sensor in ["B", "C", "D", "E"]:
            _write_wav(tmp_path / f"recorded_{sensor}_{mode}.wav")
            _write_vibration_csv(tmp_path / f"vibration_{sensor}_{mode}.csv")

    scanner = RecordingScanner(tmp_path)
    groups = scanner.scan_groups()

    assert [g.recording_id for g in groups] == ["Pump", "Turbine"]
    assert all(len(g.mic_files) == 4 for g in groups)
    assert all(len(g.vibration_files) == 4 for g in groups)

    loader = SegmentLoader()
    segment = loader.load_group(groups[0])

    assert segment.n_mic_channels == 4
    assert segment.n_accel_channels == 4
    assert segment.metadata["recording_id"] in {"Pump", "Turbine"}


def test_adapter_accepts_9_mic_bundled_directory(tmp_path) -> None:
    rec = tmp_path / "recording_001"
    rec.mkdir()

    for i in range(9):
        _write_wav(rec / f"recorded_m{i}.wav", freq_hz=200.0 + i)
    for i in range(4):
        _write_vibration_csv(rec / f"vibration_a{i}.csv")

    adapter = WavVibrationAdapter()
    segment = adapter.read_recording_directory(rec)

    assert segment.n_mic_channels == 9
    assert segment.n_accel_channels == 4
