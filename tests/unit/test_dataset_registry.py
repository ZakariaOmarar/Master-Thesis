"""Tests for `src.config.dataset_registry`."""

from __future__ import annotations

import pytest

from src.config.dataset_registry import (
    _VALID_POSITION_SOURCES,
    REGISTRY,
    DatasetRegistry,
)


def test_registry_loads_six_canonical_ids() -> None:
    """D1 through D5 plus the illwerke_raw stub."""
    assert REGISTRY.all_ids() == ["d1", "d2", "d3", "d4", "d5", "illwerke_raw"]
    assert len(REGISTRY) == 6


def test_indices_are_alphabetical_and_stable() -> None:
    """Adding a new dataset whose canonical id sorts before existing ids
    would silently renumber embedding indices; the registry's
    alphabetical-sort contract is what makes the index assignment
    deterministic.  Pin it."""
    for i, m in enumerate(REGISTRY):
        assert m.index == i, f"{m.id}: expected index {i}, got {m.index}"


def test_alias_resolution() -> None:
    """The illwerke_raw stub exposes the legacy `illwerke` alias for
    backwards compatibility with older checkpoints."""
    assert REGISTRY.has("illwerke")
    assert REGISTRY.index_of("illwerke") == REGISTRY.index_of("illwerke_raw")
    assert REGISTRY.get("illwerke").id == "illwerke_raw"


def test_unknown_id_raises() -> None:
    with pytest.raises(KeyError, match="unknown dataset_id"):
        REGISTRY.get("d99")


def test_position_source_enum_values_are_known() -> None:
    """Every registered dataset's position_source must be one of the parsers
    `PositionRegistry.from_source` knows about — otherwise the loader
    raises at startup."""
    for meta in REGISTRY:
        assert meta.position_source in _VALID_POSITION_SOURCES, (
            f"{meta.id}: position_source={meta.position_source!r} not in "
            f"{sorted(_VALID_POSITION_SOURCES)}"
        )


def test_path_based_sources_have_position_path() -> None:
    """`d2_node_position_txt` and `d3_position_json` need a `position_path`."""
    for meta in REGISTRY:
        if meta.position_source in ("default", "rowii"):
            continue
        assert meta.position_path is not None, (
            f"{meta.id}: position_source={meta.position_source} but "
            f"position_path is None"
        )


def test_d5_per_sensor_overrides_present() -> None:
    """D5 sensor E runs ~5% faster than D/F/J; the override must be
    recorded in the registry so the adapter can use it."""
    meta = REGISTRY.get("d5")
    assert meta.accel_sr_overrides == {"E": 471}


def test_window_scales_non_empty() -> None:
    """Every dataset's multi-scale tuple must have at least one scale."""
    for meta in REGISTRY:
        assert meta.window_scales_seconds, (
            f"{meta.id}: window_scales_seconds is empty"
        )


def test_missing_accel_target_sr_raises(tmp_path) -> None:
    """The registry must refuse to load a dataset whose `accel_target_sr` is
    missing or 0 — no silent fallback."""
    yaml_path = tmp_path / "broken.yaml"
    yaml_path.write_text(
        "id: broken\n"
        "root: data/broken\n"
        "n_mics: 1\n"
        "n_vibrations: 1\n"
        "accel_target_sr: 0\n"
        "position_source: default\n"
        "label_scheme: d1_mode\n"
        "window_scales_seconds: [1.0]\n"
        "v3_window_seconds: 1.0\n"
        "v4_window_seconds: 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="accel_target_sr"):
        DatasetRegistry(configs_dir=tmp_path)
