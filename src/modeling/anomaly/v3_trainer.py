"""V3 trainer — frozen V2 encoder → CNF on healthy `(x, c)` pairs → per-cluster thresholds.

`x` = mean-pool of the fused-token sequence (a fixed pool, distinct from PMA).
`c` = `c_t` = PMA pool (the V2 context vector).  The flow learns the conditional
density `p(x | c)` and emits ``-log p(x | c)`` as the anomaly score.

Two ablation knobs:
  - ``cfg.unconditional=True`` → A2 ablation.  Zeros are passed as `c` to the
    flow at both training and inference, so the FiLM modulation degenerates
    to identity and the flow becomes unconditional.
  - The synthetic transition stress-test (`make_transition_segment` +
    `score_segment`) splices two healthy segments with a linear crossfade and
    measures the false-alert rate over the transition windows.
"""

from .v3_trainer_config import V3Config, V3Result, XtPoolKind
from .v3_trainer_data import (
    _cache_fused,
    _extract_x_for_segment,
    _extract_xc,
    _make_override_v2_cfg,
    _pool_cached_x,
    _resolve_v3_override,
    _stack_c_from_cache,
    _stack_labels_from_cache,
    precompute_paired,
)
from .v3_trainer_model import _augment_anchor, _fit_anchor_norm, _XtPool
from .v3_trainer_train import (
    encoder_level_transition_fpr,
    gate_samples_by_alert,
    make_transition_segment,
    score_segments,
    train_v3_cnf,
    transition_fpr,
)

__all__ = [
    "V3Config",
    "V3Result",
    "XtPoolKind",
    "_XtPool",
    "_augment_anchor",
    "_cache_fused",
    "_extract_x_for_segment",
    "_extract_xc",
    "_fit_anchor_norm",
    "_make_override_v2_cfg",
    "_pool_cached_x",
    "_resolve_v3_override",
    "_stack_c_from_cache",
    "_stack_labels_from_cache",
    "encoder_level_transition_fpr",
    "gate_samples_by_alert",
    "make_transition_segment",
    "precompute_paired",
    "score_segments",
    "train_v3_cnf",
    "transition_fpr",
]
