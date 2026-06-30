"""Thesis hydropower starter package.

The top-level ``DataSegment`` export is resolved lazily via ``__getattr__`` so
submodule entry-points (e.g. ``python -m src.modeling.orchestration.full_run``)
do not pull the optional feature-stack dependencies unless the symbol is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # let type checkers see the lazily-exported symbol
    from .data import DataSegment

__all__ = ["DataSegment"]


def __getattr__(name: str) -> Any:
    if name == "DataSegment":
        from .data import DataSegment

        return DataSegment
    raise AttributeError(f"module 'src' has no attribute {name!r}")
