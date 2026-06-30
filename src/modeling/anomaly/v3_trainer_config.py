"""V3 anomaly trainer configuration and result dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch.nn as nn

from ...config.architecture import V3_ANOMALY
from .cnf_head import ConditionalRealNVP
from .threshold import PerClusterThresholds

XtPoolKind = Literal["mean", "pma2"]


@dataclass(frozen=True)
class V3Config:
    """V3 conditional anomaly head config."""

    # Per-stage window override (2026-05-19).  When set, the V3 trainer
    # builds a shallow copy of the V2 config with
    # `window_scales_seconds=(override,)` (single-scale) before
    # constructing `_PairedWindowedDataset`.  The frozen V2 encoder is
    # scale-invariant by virtue of ASP pooling (Okabe 2018), so the
    # smaller window is consumed without re-pretraining V2.  May be:
    #
    #   * ``None`` — inherit V2's scale set (legacy behaviour),
    #   * a ``float`` — single override applied to every dataset, or
    #   * a ``dict[dataset_id, float]`` — per-dataset override.  Publication
    #     defaults are 1.0 s on D3/D4 (transient-tight; the smallest sub-2 s
    #     scale where ASP-σ still has ≥ 30 post-MaxPool frames at hop=43)
    #     and 3.0 s on D1/D2 (which cannot go below 1.5 s).
    # Default ``None`` keeps V3 backwards-compatible (inherits v2_cfg
    # window).  The orchestrator (full_run.v3_config) sets the publication
    # per-dataset dict from `WINDOWING.v3_window_seconds_override`.
    window_seconds_override: float | dict[str, float] | None = None

    # V3 channel-token pool kind for `x_t`.  Mean-pool dilutes transient
    # signatures on the channel-token axis; PMA-2 (the publication
    # default) learns 2 attention seeds that capture both stationary-mode
    # and transient-event patterns.  See `_XtPool` docstring above for the
    # full justification.
    xt_pool: XtPoolKind = V3_ANOMALY.xt_pool
    xt_pool_num_heads: int = V3_ANOMALY.xt_pool_num_heads

    # CNF dims — sourced from `V3_ANOMALY` in architecture.py.
    n_layers: int = V3_ANOMALY.n_layers
    hidden_dim: int = V3_ANOMALY.hidden_dim
    n_hidden_per_net: int = 2   # CNF coupling MLP depth; not centralised
    scale_max: float = V3_ANOMALY.scale_max
    conditional_base: bool = V3_ANOMALY.conditional_base
    inject_impulse_anchor: bool = V3_ANOMALY.inject_impulse_anchor

    # Training schedule — per-experiment, not centralised.
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.3
    use_cosine_lr: bool = True

    # A2 ablation — zero c at train+infer for unconditional flow.
    unconditional: bool = False

    # Threshold fit — fully unsupervised on healthy data.
    n_threshold_clusters: int = V3_ANOMALY.n_threshold_clusters
    threshold_percentile: int = V3_ANOMALY.threshold_percentile
    threshold_shrinkage: float = V3_ANOMALY.threshold_shrinkage

    # Nested held-out split inside `val_segs`: a `threshold_fit_val_ratio`
    # fraction of val recordings goes to *fitting* the K-means centroids and
    # per-cluster percentile bar; the remaining fraction is the *reportable*
    # held-out cohort whose NLL and alert rate are quoted in the thesis.
    # Without this split the same windows define the clusters, set the bar,
    # and are scored against the bar — per-cluster p95/p99 are optimistically
    # biased.  Split is by recording (uses `_split_segments_by_recording`),
    # so windows from a single recording cannot span both halves.
    threshold_fit_val_ratio: float = 0.5
    # The Youden's-J calibration helper exists in `threshold.py` for
    # post-hoc analysis but is **not** wired into the orchestrator.  It
    # would require per-window anomaly labels (or an assumption that all
    # D2 RF / D3 hit windows are anomalous), which the field-collection
    # protocol does not provide.  Threshold quality is instead validated
    # post-hoc via per-cohort alert rates (the orchestrator's
    # `v3_alert_rate_per_cohort` metric).
    calibrate_with_anomalies: bool = False

    # Early stopping on val mean NLL.  Patience=5 (vs V1/V2's 3) because the
    # CNF's val NLL is noisier (4-recording fit cohort + outlier batches —
    # see val_nll_max audit signal).  Restore-best restores the FLOW state;
    # the V2 encoder is frozen so nothing to restore there.  Tensor-clone
    # snapshot (not copy.deepcopy) — see V1 trainer for rationale.
    patience: int = 5
    restore_best: bool = True
    early_stop_min_delta: float = 1e-3  # NLL units; flow loss is O(10-100)

    # Cohort the early-stopping / restore-best selection minimises.
    #   * "val_eval" (default, legacy): selects the epoch on the SAME held-out
    #     cohort whose NLL is reported.  This couples model selection to the
    #     reported metric, so `val_nll` is a (mildly optimistic) selection
    #     metric, not an unbiased generalisation estimate.  Kept as the default
    #     so existing thesis runs reproduce bit-for-bit.
    #   * "val_fit": selects on the threshold-fit cohort instead, leaving
    #     `val_eval` a never-selected hold-out.  This is the unbiased protocol;
    #     enable it for runs that want an honest held-out NLL.  It changes which
    #     epoch is chosen and therefore the reported numbers, so it is opt-in.
    # The headline RQ2 metrics (per-cluster threshold FPR / recall) are already
    # protected: thresholds are fit on `val_fit` and scored on the disjoint
    # `val_eval`, so this flag only affects the auxiliary `val_nll` figure.
    select_on_fit_cohort: bool = False

    # CNF coupling MLP dropout — defends against the +56 % train/val NLL gap
    # the audit identified.  Default 0.0 keeps the dataclass byte-equivalent
    # to pre-fix behaviour; the orchestrator `v3_config` builder sets 0.1.
    dropout_p: float = 0.0

    # System
    seed: int = 42
    device: str = "auto"


# ---------------------------------------------------------------------------
# Encoder feature extraction
# ---------------------------------------------------------------------------
@dataclass
class V3Result:
    flow: ConditionalRealNVP
    thresholds: PerClusterThresholds
    train_nll: list[float]
    val_nll: list[float]
    # F6 — per-epoch outlier-batch tracking (min / max of per-batch
    # train NLL and per-window val NLL).  Spread between max and the
    # mean diagnoses single-batch loss spikes hiding in the average.
    train_nll_min: list[float]
    train_nll_max: list[float]
    val_nll_min: list[float]
    val_nll_max: list[float]
    train_recording_ids: list[str]
    # Reportable held-out cohort — disjoint from train AND from the cohort
    # used to fit per-cluster thresholds.  All quoted val numbers
    # (`val_scores`, `val_contexts`, `val_labels`, per-epoch `val_nll`)
    # come from this cohort.
    val_recording_ids: list[str]
    # Cohort used to fit K-means centroids + per-cluster p95/p99 — disjoint
    # from the reportable val cohort.  Exposed for transparency / auditing.
    threshold_fit_recording_ids: list[str]
    val_scores: np.ndarray
    val_contexts: np.ndarray
    val_labels: list[str]
    unconditional: bool
    # Per-window recording id for the reportable val cohort, aligned to
    # `val_scores` / `val_contexts`.  Enables the held-out NLL paired test
    # (V3 vs A2) to resample at the recording level rather than the window
    # level — see `eval.statistics.paired_bootstrap_test(groups=)`.  None on
    # the legacy mean-pool path (which does not cache per-window recording ids).
    val_recording_ids_per_window: list[str] | None = None
    # `_XtPool` module trained jointly with the flow.  None when the legacy
    # mean-pool path was used (`xt_pool="mean"`).  Carries learned PMA-2
    # weights that scoring / streaming inference must reuse to obtain
    # comparable scores on new windows.
    xt_pool: nn.Module | None = None
    # Impulse+spectral anchor standardization (healthy mean/std) when
    # `inject_impulse_anchor` is on; None otherwise.  Persisted so the RQ2 eval
    # and the V4 gate can recompute + standardize the anchor identically.
    anchor_mean: np.ndarray | None = None
    anchor_std: np.ndarray | None = None
    early_stopped_epoch: int | None = None
    best_val_nll: float = float("nan")
    # Cached x / c arrays for the deep-vs-simple comparison (KDE-on-c_t in
    # `kde_baseline.py`).  These are the same arrays the flow trained on /
    # was evaluated against, so any KDE built from them is apples-to-apples
    # with the CNF's NLL.  Stored on CPU as numpy to keep V3Result picklable.
    train_x: np.ndarray | None = None
    train_contexts: np.ndarray | None = None
    val_x: np.ndarray | None = None
