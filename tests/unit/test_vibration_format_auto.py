"""Tests for `vibration_format="auto"` resolution policy.

Policy: when any `vibration_raw_*.csv` is present in the candidate paths,
`auto` resolves to `raw`; otherwise to `peak`.  Lives in
`src.ingestion.adapters.resolve_vibration_format` so the loader and
adapter never disagree on the resolution for the same recording.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.adapters import (
    filter_vibration_csv_paths,
    resolve_vibration_format,
)


def _paths(*names: str, base: Path | None = None) -> list[Path]:
    base = base or Path("/x")
    return [base / n for n in names]


def test_auto_prefers_raw_when_present() -> None:
    paths = _paths("vibration_D.csv", "vibration_raw_D.csv", "vibration_raw_E.csv")
    assert resolve_vibration_format(paths, "auto") == "raw"


def test_auto_falls_back_to_peak_when_no_raw() -> None:
    paths = _paths("vibration_D.csv", "vibration_E.csv", "vibration_F.csv")
    assert resolve_vibration_format(paths, "auto") == "peak"


def test_auto_with_only_raw_resolves_to_raw() -> None:
    paths = _paths("vibration_raw_D.csv", "vibration_raw_E.csv")
    assert resolve_vibration_format(paths, "auto") == "raw"


def test_explicit_peak_overrides_auto_inference() -> None:
    """An explicit `vibration_format="peak"` keeps peak even when raw CSVs
    are present — this is the ablation entry point for "what would V3 see
    on D5 if we forced the peak path?" """
    paths = _paths("vibration_D.csv", "vibration_raw_D.csv")
    assert resolve_vibration_format(paths, "peak") == "peak"
    assert resolve_vibration_format(paths, "raw") == "raw"


def test_invalid_format_raises() -> None:
    with pytest.raises(Exception, match="vibration_format"):
        resolve_vibration_format([], "neither")


def test_filter_with_auto_returns_correct_partition() -> None:
    paths = _paths("vibration_D.csv", "vibration_raw_D.csv", "vibration_raw_E.csv")
    # auto -> raw, keep only the raw files
    filtered = filter_vibration_csv_paths(paths, "auto")
    assert tuple(p.name for p in filtered) == ("vibration_raw_D.csv", "vibration_raw_E.csv")

    # peak-only path -> auto -> peak
    peak_only = _paths("vibration_D.csv", "vibration_E.csv")
    filtered = filter_vibration_csv_paths(peak_only, "auto")
    assert tuple(p.name for p in filtered) == ("vibration_D.csv", "vibration_E.csv")
