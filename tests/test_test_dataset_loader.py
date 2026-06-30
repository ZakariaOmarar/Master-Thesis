"""Smoke tests for the unified test_dataset loader.

Run with:  pytest tests/test_test_dataset_loader.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ingestion.positions import PositionRegistry
from src.ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.requires_data


def _spec(name: str) -> DatasetSpec:
    # `DatasetSpec.from_yaml` resolves all paths to absolute — no reconstruction needed.
    return DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / f"{name}.yaml")


def test_d1_loads_with_synthetic_geometry() -> None:
    spec = _spec("d1")
    loader = TestDatasetLoader(spec)
    segments = loader.list_segments()
    assert len(segments) >= 4, f"expected at least 4 D1 recordings, got {len(segments)}"

    s = segments[0]
    assert s.segment.n_mic_channels == 4
    assert s.segment.n_accel_channels == 4
    assert s.mic_positions.shape == (4, 3)
    assert s.vib_positions.shape == (4, 3)
    assert s.mode_label in ("Pump", "Standstill", "Turbine", "RandomFault")
    # Synthetic positions are deterministic and small (<= 2 m).
    assert np.all(np.abs(s.mic_positions) <= 3.0)


def test_d2_loads_with_native_positions_and_spatial_labels() -> None:
    spec = _spec("d2")
    loader = TestDatasetLoader(spec)
    segments = loader.list_segments()
    assert len(segments) >= 5, f"expected at least 5 D2 recordings, got {len(segments)}"

    s = segments[0]
    assert s.segment.n_mic_channels == 5
    assert s.segment.n_accel_channels == 5
    assert s.mic_positions.shape == (5, 3)
    assert s.vib_positions.shape == (5, 3)

    # At least one RandomFault recording should expose a parsed spatial label.
    spatial = [seg for seg in segments if seg.spatial_label is not None]
    assert len(spatial) >= 1, "no D2 recording produced a parsed spatial label"
    x, y, z = spatial[0].spatial_label
    # Stored in meters; D2 file gives cm; one of the listed positions is (0,17,12)cm = (0, 0.17, 0.12)m.
    assert -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0 and -1.0 <= z <= 1.0


def test_d3_loads_with_speed_labels_and_hit() -> None:
    spec = _spec("d3")
    loader = TestDatasetLoader(spec)
    segments = loader.list_segments()
    assert len(segments) >= 4, f"expected at least 4 D3 recordings, got {len(segments)}"

    s = segments[0]
    assert s.segment.n_mic_channels == 9
    assert s.segment.n_accel_channels == 4
    assert s.mic_positions.shape == (9, 3)
    assert s.vib_positions.shape == (4, 3)

    op_conditions = {seg.op_condition for seg in segments}
    # Should include at least speed1 and the hit recording's speed.
    assert any(op and op.startswith("speed") for op in op_conditions)


def test_position_registry_is_consistent_across_lookups() -> None:
    reg = PositionRegistry.from_source("default")
    p1 = reg.lookup_mic("B")
    p2 = reg.lookup_mic("B")
    assert np.allclose(p1, p2)
    # Aliases work
    p3 = reg.lookup_mic("b")
    assert np.allclose(p1, p3)


def test_position_registry_d2_mixed_colons() -> None:
    """D2's node_position.txt uses a mix of ASCII and full-width colons."""
    reg = PositionRegistry.from_source(
        "d2_node_position_txt",
        position_path=REPO_ROOT / "data" / "second_test_dataset" / "node_position.txt",
    )
    # Vibration A and microphone D should both parse.
    pos_a = reg.lookup_vibration("A")
    pos_d = reg.lookup_mic("D")
    assert pos_a.shape == (3,) and pos_d.shape == (3,)
    # cm → m: vibration_A is at (10, 0, 23) cm = (0.10, 0.00, 0.23) m
    assert np.allclose(pos_a, [0.10, 0.00, 0.23])


def test_position_registry_d3_alias_handling() -> None:
    reg = PositionRegistry.from_source(
        "d3_position_json",
        position_path=REPO_ROOT / "data" / "third_test_dataset" / "position.json",
    )
    # File has "Dl"; loader extracts "D_l" from filename — both must resolve.
    p1 = reg.lookup_mic("Dl")
    p2 = reg.lookup_mic("D_l")
    assert np.allclose(p1, p2)
    # Vibration in file has "(V)D"; CSV filename gives "D".
    v1 = reg.lookup_vibration("D")
    v2 = reg.lookup_vibration("(V)D")
    assert np.allclose(v1, v2)


@pytest.mark.parametrize("name", ["d1", "d2", "d3"])
def test_segments_have_finite_data(name: str) -> None:
    spec = _spec(name)
    loader = TestDatasetLoader(spec)
    segments = loader.list_segments()
    for s in segments[:2]:
        assert np.all(np.isfinite(s.segment.mic_data))
        assert np.all(np.isfinite(s.segment.accel_data))
        assert s.segment.duration_s > 0
