"""Shared pytest configuration and data-availability gating.

Part of the suite exercises the real turbine-rig recordings under ``data/``.
Those captures are not redistributed with the repository, so the tests that
need them are marked ``@pytest.mark.requires_data`` and are skipped
automatically on a checkout that has no ``data/`` directory. The
synthetic-only tests (encoders, pooling, the data contract, statistics,
multilateration, …) carry no marker and run anywhere::

    pytest -m "not requires_data"   # synthetic-only — needs no data
    pytest                          # full suite — data tests skip if absent

See the README "Datasets" section for how to obtain / lay out the recordings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"


def _data_available() -> bool:
    """True if at least one dataset directory is present under ``data/``."""
    return DATA_ROOT.is_dir() and any(DATA_ROOT.glob("*_test_dataset"))


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``requires_data`` tests when no recordings are checked out."""
    if _data_available():
        return
    skip_no_data = pytest.mark.skip(
        reason="no recordings under data/ (see README → Datasets)"
    )
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip_no_data)
