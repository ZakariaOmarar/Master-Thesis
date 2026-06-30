"""Tests for the `d5_healthy_or_knock` label_scheme parser."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.test_dataset_loader import _parse_labels


def _call(folder: str, parent: str = "knock") -> tuple:
    """Helper — calls _parse_labels with a synthetic source_dir."""
    source_dir = Path("/x") / parent / folder
    return _parse_labels(source_dir, recording_id=folder, scheme="d5_healthy_or_knock")


def test_healthy_folder_returns_healthy_with_no_spatial() -> None:
    source_dir = Path("/x/healthy")
    mode, op, spatial, is_anom = _parse_labels(
        source_dir, recording_id="healthy", scheme="d5_healthy_or_knock"
    )
    assert (mode, op, spatial, is_anom) == (None, None, None, False)


def test_all_six_d5_knock_positions_parse() -> None:
    """The six D5 knock subfolder names — including negatives, missing
    spaces, and z>0 — must all parse to centimetre→metre coordinates with
    `is_anomaly=True`."""
    cases = {
        "(-11, 0, 0)":    (-0.11, 0.0, 0.0),
        "(22, 0, 0)":     (0.22, 0.0, 0.0),
        "(3, -3, 8)":     (0.03, -0.03, 0.08),
        "(6, -25, 0)":    (0.06, -0.25, 0.0),
        "(6,-15, 0)":     (0.06, -0.15, 0.0),  # missing-space variant
        "(9, -3, 8)":     (0.09, -0.03, 0.08),
    }
    for folder, expected_xyz in cases.items():
        mode, op, spatial, is_anom = _call(folder)
        assert mode is None
        assert op is None
        assert spatial == expected_xyz, f"{folder}: got {spatial}, expected {expected_xyz}"
        assert is_anom is True


def test_unknown_folder_returns_default_healthy() -> None:
    """Folders that match neither `healthy` nor the `(x, y, z)` pattern
    fall through to a "no-op label" (no mode, no spatial, not anomaly).
    This is intentional — the recursive scanner walks intermediate
    directories that aren't expected to carry semantic labels."""
    mode, op, spatial, is_anom = _call("some_random_subdir", parent="healthy")
    assert (mode, op, spatial, is_anom) == (None, None, None, False)


def test_parent_healthy_propagates() -> None:
    """A subdirectory under `healthy/` (e.g. a future cohort split)
    inherits the healthy classification."""
    source_dir = Path("/x/healthy/cohort_2")
    mode, op, spatial, is_anom = _parse_labels(
        source_dir, recording_id="cohort_2", scheme="d5_healthy_or_knock"
    )
    assert is_anom is False
