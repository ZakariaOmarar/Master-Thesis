"""V3 conditional anomaly head.

Three components, all label-free at training time:
  - `cnf_head`     : RealNVP CNF with FiLM(c) coupling (FiLMMLP / FiLMCoupling /
                     ConditionalRealNVP).
  - `threshold`    : K-means(K) + per-cluster percentile thresholds — the alert
                     mechanism replacing CANDE-CP's per-mode buckets with
                     per-cluster buckets to preserve the label-leakage invariant.
  - `v3_trainer`   : extracts (mean-pool x, PMA c) pairs from a frozen V2
                     encoder, trains the CNF, fits per-cluster thresholds,
                     and provides the synthetic transition stress-test.

A2 ablation: pass `unconditional=True` to zero `c` at train + infer.
"""

from .cnf_head import ConditionalRealNVP, FiLMCoupling, FiLMMLP
from .threshold import PerClusterThresholds
from .v3_trainer import (
    V3Config,
    V3Result,
    gate_samples_by_alert,
    make_transition_segment,
    score_segments,
    train_v3_cnf,
    transition_fpr,
)

__all__ = [
    "ConditionalRealNVP",
    "FiLMCoupling",
    "FiLMMLP",
    "PerClusterThresholds",
    "V3Config",
    "V3Result",
    "gate_samples_by_alert",
    "make_transition_segment",
    "score_segments",
    "train_v3_cnf",
    "transition_fpr",
]
