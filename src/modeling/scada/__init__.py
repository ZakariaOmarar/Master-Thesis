"""V5 SCADA injection — D3 speed conditioning + Illwerke MI ranking.

Two pieces, kept deliberately separate:
  - `d3_speed`        : V5.1 — D3 speed bucket → one-hot SCADA tensor for the
                        V4 head's `s_t` slot.
  - `channel_mining`  : V5.2 — mutual-information ranking of Allg_M1 channels
                        against the legacy 5-layer pipeline anomaly events.
                        Pure offline analysis, no injection on D2/D3.
"""

from .channel_mining import (
    MIRanking,
    anomaly_indicator,
    load_anomaly_events,
    physical_family,
    rank_channels_by_mi,
)
from .d3_speed import (
    D3_SCADA_DIM,
    D3_SPEED_BUCKETS,
    d3_speed_lookup,
    d3_speed_one_hot,
)

__all__ = [
    "D3_SCADA_DIM",
    "D3_SPEED_BUCKETS",
    "MIRanking",
    "anomaly_indicator",
    "d3_speed_lookup",
    "d3_speed_one_hot",
    "load_anomaly_events",
    "physical_family",
    "rank_channels_by_mi",
]
