"""Smoke tests for D4 ingestion (raw-waveform vibration + (x, y, z) folders).

Three checks:
  1. The raw vibration reader produces a 1-D waveform at ~376 Hz from a
     `vibration_raw_*.csv` row-as-batch file.
  2. D4's label scheme parses both `speed{N}` (Healthy) and
     `RandomFault_knock_unter_speedN/(x, y, z)/` (RandomFault with spatial).
  3. The loader returns the expected number of recordings with correct
     shapes — at least one healthy and one RandomFault, with a meaningful
     vibration sample rate (>100 Hz, vs the 4-16 Hz peak streams of D1/D2/D3).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.adapters import _read_vibration_raw_csv
from src.ingestion.test_dataset_loader import (
    _D4_POS_RE,
    DatasetSpec,
    TestDatasetLoader,
    _parse_d2_context_to_mode,
    _parse_labels,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
D4_ROOT = REPO_ROOT / "data" / "fourth_test_dataset"


pytestmark = [
    pytest.mark.requires_data,
    pytest.mark.skipif(
        not D4_ROOT.exists(),
        reason="D4 dataset not present in this checkout",
    ),
]


# ---------------------------------------------------------------------------
# Raw vibration reader
# ---------------------------------------------------------------------------


def test_read_vibration_raw_csv_returns_continuous_waveform() -> None:
    csv_path = D4_ROOT / "speed1" / "vibration_raw_D.csv"
    waveform, sr = _read_vibration_raw_csv(csv_path)
    assert waveform.ndim == 1
    assert waveform.size > 100_000  # ~11 min at 376 Hz ≈ 250k samples
    # ~376 Hz effective rate (109 real samples per ~290 ms batch)
    assert 200.0 < sr < 500.0
    # Raw 16-bit ADC values (no mean removal at this stage)
    assert waveform.min() > 0.0
    assert waveform.max() < 70_000.0


# ---------------------------------------------------------------------------
# D4 label scheme
# ---------------------------------------------------------------------------


def test_d4_pos_regex_matches_spaced_tuples() -> None:
    assert _D4_POS_RE.match("(2, 4, 8)") is not None
    assert _D4_POS_RE.match("(-20, 0, 0)") is not None
    assert _D4_POS_RE.match("(5.5, -4, 8)") is not None
    # Should NOT match D2 prefixed format or random strings.
    assert _D4_POS_RE.match("pos_(2, 4, 8)") is None
    assert _D4_POS_RE.match("speed1") is None


def test_d4_label_scheme_parses_healthy_and_random() -> None:
    # Healthy speed bucket — speed{N} is fan-noise level, not a mode, so
    # mode_label is None (the pipeline discovers the mode at inference).
    speed_dir = D4_ROOT / "speed1"
    mode, op, spatial, is_anomaly = _parse_labels(
        speed_dir, "speed1", "d4_speed_with_random"
    )
    assert mode is None
    assert op == "speed1"
    assert spatial is None
    assert is_anomaly is False

    # RandomFault with spatial label.  Mode is also None for D4 RandomFault.
    rf_dir = D4_ROOT / "RandomFault_knock_unter_speed1" / "(2, 4, 8)"
    mode, op, spatial, is_anomaly = _parse_labels(
        rf_dir, "(2, 4, 8)", "d4_speed_with_random"
    )
    assert mode is None
    assert op == "speed1"
    assert spatial == pytest.approx((0.02, 0.04, 0.08))
    assert is_anomaly is True


# ---------------------------------------------------------------------------
# End-to-end loader
# ---------------------------------------------------------------------------


def test_d2_context_parser_drops_multi_mode() -> None:
    """D2 RandomFault `<context>` token: keep single-mode, drop multi-mode."""
    assert _parse_d2_context_to_mode("turbine") == "Turbine"
    assert _parse_d2_context_to_mode("pump") == "Pump"
    assert _parse_d2_context_to_mode("standstill") == "Standstill"
    # Multi-mode tokens conflate two regimes in one recording → return None
    # (the V4 trainer filters these out via mode_label is None).
    assert _parse_d2_context_to_mode("turbine_pump") is None
    assert _parse_d2_context_to_mode("pump_turbine") is None
    # Unknown tokens also return None.
    assert _parse_d2_context_to_mode("blade1") is None


def test_d2_label_scheme_drops_multi_mode_and_keeps_single() -> None:
    """End-to-end: D2 single-mode anomaly carries (mode, is_anomaly=True);
    D2 multi-mode anomaly carries (mode=None, is_anomaly=True)."""
    single = Path("data/second_test_dataset/RandomFault/pos_(15,30,15)_turbine")
    mode, _op, spatial, is_anom = _parse_labels(
        single, "pos_(15,30,15)_turbine", "d2_mode_with_spatial"
    )
    assert mode == "Turbine"
    assert is_anom is True
    assert spatial == pytest.approx((0.15, 0.30, 0.15))

    multi = Path("data/second_test_dataset/RandomFault/pos_(0,17,12)_turbine_pump")
    mode, _op, spatial, is_anom = _parse_labels(
        multi, "pos_(0,17,12)_turbine_pump", "d2_mode_with_spatial"
    )
    assert mode is None  # dropped from supervision
    assert is_anom is True
    assert spatial == pytest.approx((0.00, 0.17, 0.12))


def test_d4_loader_inventory() -> None:
    # `DatasetSpec.from_yaml` resolves all paths to absolute — no reconstruction needed.
    spec = DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / "d4.yaml")
    L = TestDatasetLoader(spec)
    segments = L.list_segments()
    assert len(segments) >= 8  # 3 healthy + ≥5 RandomFault (one folder has 5 vib channels and is skipped)

    healthy = [s for s in segments if not s.is_anomaly]
    random_fault = [s for s in segments if s.is_anomaly]
    assert len(healthy) == 3
    assert len(random_fault) >= 5
    # All D4 recordings have mode_label=None — the pipeline discovers it.
    assert all(s.mode_label is None for s in segments)
    # All RandomFault recordings carry a spatial label.
    assert all(s.spatial_label is not None for s in random_fault)

    # Vibration is at the raw ADC rate, not the legacy 4 Hz peak rate.
    seg = healthy[0]
    assert seg.segment.accel_sample_rate >= 200
    assert seg.segment.accel_data.shape[0] == 4
    assert seg.segment.mic_data.shape[0] == 9
    assert seg.segment.mic_sample_rate == 16_000

    # Recordings are long (~11 min) — far above D1/D2/D3 typical durations.
    duration_s = seg.segment.mic_data.shape[1] / seg.segment.mic_sample_rate
    assert duration_s > 300  # > 5 min
