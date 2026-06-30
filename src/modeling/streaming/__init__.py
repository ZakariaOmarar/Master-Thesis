"""Gated streaming-inference pipeline.

`GatedPipeline` wires V2 → V3 → (gated) V4.  Exports:
  - `StreamingDecision`  — per-window output schema
  - `GatedPipeline`      — the runtime
  - `CostQualityReport`  — Chapter 6 deployment-shape evidence
  - `cost_quality_study` — gated vs. continuous comparison helper
"""

from .inference import (
    CostQualityReport,
    GatedPipeline,
    StreamingDecision,
    cost_quality_study,
)

__all__ = [
    "CostQualityReport",
    "GatedPipeline",
    "StreamingDecision",
    "cost_quality_study",
]
