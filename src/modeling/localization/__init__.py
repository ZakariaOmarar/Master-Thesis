# pyright: reportMissingImports=false
"""Localization sub-package.

`classical` ships the classical signal-processing primitives (GCC-PHAT,
SRP-PHAT).  `v4_features`, `v4_loc_head`, and `v4_trainer` add the V4
anomaly-gated learning-based localization head (the live pipeline).
`localization_head` holds the earlier neural S2/S3/dual-SRP localizers, kept as
Chapter 6 baselines and built on the `classical` primitives; it is not imported
by the live pipeline.
"""

from .array_geometry import (
    FootprintVerdict,
    array_sensor_xyz,
    classify_position,
    classify_positions,
)
from .v4_features import (
    C_AIR_MS,
    C_PLASTIC_3DP_MS,
    V4_CANDIDATE_GRID,
    GridSpec,
    compute_accel_tdoa_tokens,
    compute_burst_aware_srp_phat_volume,
    compute_srp_phat_volume,
    find_burst_window,
    srp_peak_sharpness,
)
from .v4_knock_events import (
    KnockEventConfig,
    assert_no_position_leak,
    precompute_v4_knock_event_samples,
)
from .v4_loc_head import (
    FiLMResidualHead,
    HeatmapCross3D,
    TDOASetEncoder,
    V4LocalizationHead,
    soft_argmax_3d,
)
from .v4_metrics import event_aggregated_mae
from .v4_synthetic import (
    SyntheticArraySpec,
    generate_synthetic_knock_samples,
)
from .v4_temporal import (
    AlertBurst,
    detect_alert_bursts,
    evaluate_burst_localization,
    smooth_predictions_over_bursts,
)
from .v4_trainer import (
    V4Config,
    V4Result,
    V4Sample,
    precompute_v4_samples,
    split_samples_by_dataset,
    split_samples_by_position,
    train_v4_localization,
)

__all__ = [
    "C_AIR_MS",
    "C_PLASTIC_3DP_MS",
    "V4_CANDIDATE_GRID",
    "AlertBurst",
    "FiLMResidualHead",
    "FootprintVerdict",
    "GridSpec",
    "HeatmapCross3D",
    "KnockEventConfig",
    "SyntheticArraySpec",
    "TDOASetEncoder",
    "V4Config",
    "V4LocalizationHead",
    "V4Result",
    "V4Sample",
    "array_sensor_xyz",
    "assert_no_position_leak",
    "classify_position",
    "classify_positions",
    "compute_accel_tdoa_tokens",
    "compute_burst_aware_srp_phat_volume",
    "compute_srp_phat_volume",
    "detect_alert_bursts",
    "evaluate_burst_localization",
    "event_aggregated_mae",
    "find_burst_window",
    "generate_synthetic_knock_samples",
    "precompute_v4_knock_event_samples",
    "precompute_v4_samples",
    "smooth_predictions_over_bursts",
    "soft_argmax_3d",
    "split_samples_by_dataset",
    "split_samples_by_position",
    "srp_peak_sharpness",
    "train_v4_localization",
]
