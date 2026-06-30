"""V1-V4 stage hyperparameter builders — the single source of truth.

Each ``vN_config(quick)`` returns the trainer config for one pipeline stage. The
orchestrator (``full_run``) and every experiment driver under ``scripts/``
import these, so a hyperparameter is defined in exactly one place; change a
value here, never at a call site.

The builders are re-exported from ``full_run`` (``from .stage_configs import
v1_config, ...``) so that ``full_run.main()`` resolves them from its own module
namespace. That keeps the multi-seed and hop-length drivers' monkeypatching of
``full_run.vN_config`` working: reassigning the name on the orchestrator module
still overrides what ``main()`` calls.

The leading underscore marks these as research configuration rather than a
stable public API — read them freely, but change the literals here, with care.

``quick=True`` shrinks epoch counts for smoke tests; everything else is the
publication profile. Inline comments carry the per-hyperparameter rationale and
cross-reference the thesis chapters.
"""

from __future__ import annotations

from ...config.architecture import WINDOWING
from ...config.dataset_registry import REGISTRY
from ..anomaly import V3Config
from ..context.v1_ssl import V1SSLConfig
from ..context.v2_ssl import V2SSLConfig
from ..localization import V4Config


def v1_config(quick: bool) -> V1SSLConfig:
    return V1SSLConfig(
        window_seconds=2.0,
        window_stride_seconds=1.0,
        feature_dim=64,
        embed_dim=64,
        n_heads=4,
        proj_dim=32,
        # Bumped 6→12 epochs for the full profile so c_t reaches a stable
        # mode-discriminative state — a stronger c_t directly improves
        # V3's per-cluster CNF density estimates and consequently the
        # quality of the unsupervised p95 threshold.  CPU wall-clock cost
        # ~ 25 min on V1 acoustic, ~ 20 min on V1 vibration.
        epochs=3 if quick else 12,
        batch_size=16,
        lr=1e-3,
        # 1e-4 matches the standard SimCLR / AdamW convention. The dataclass
        # default of 1e-5 is one order too low and was the dominant contributor
        # to the overfitting audit's train/val gaps (it predates early stopping).
        # Set on every V1-V4 builder here, not at the dataclass, so older
        # import-sites stay byte-reproducible.
        weight_decay=1e-4,
        temperature=0.1,
        val_ratio=0.3,
        # n_mels / n_fft / hop_length intentionally not set here — inherited
        # from `ACOUSTIC_FEATURES` in src/config/architecture.py:
        #   n_fft=4096, hop_length=2048, n_mels=96
        # These are the Pareto-optimal values from the empirical grid search
        # documented in chapter 3 §3.4.2 and reproduced by
        # `scripts/hop_length_study/analyze_hop_length_full_grid.py`.
        cwt_n_scales=32,
        # CWT scalogram re-enabled for the publication run.  CWT is the
        # primary representation for non-stationary mode-transition energy;
        # it was disabled in the prior run only for a CPU runtime budget.
        # Re-enabling it is the simplest single unimodal lift available given
        # V1-acoustic already leads at 0.727 purity on log-mel-only.
        # Wall-clock impact: V1 / V2 epoch ~ 1.7×.
        use_cwt=True,
        # Multi-scale window cadence — sourced from the dataset registry
        # (configs/datasets/d*.yaml). Change there, not here.
        window_scales_seconds_per_dataset={
            m.id: m.window_scales_seconds for m in REGISTRY
        },
        window_scale_strategy=WINDOWING.window_scale_strategy,
        window_stride_ratio=WINDOWING.window_stride_ratio,
        # Vibration channel-2: physical-time defaults (100 ms kurtosis target,
        # 31-sample statistical floor, 1 s crest-factor fallback).  Auto-
        # selects kurtosis on D4 raw (~376 Hz → 37 samples) and crest
        # factor on D1/D2/D3 peak streams (4-16 Hz).  Override here only if
        # the impulse-duration assumption changes.
        # R1b — push acoustic SSL augmentation harder.  The 2026-05-16 B5
        # run reached V1-acoustic NMI 0.729 with the prior 6/6/8 settings;
        # V0 LightGBM hits macro-F1 = 1.0 on D1, so the features carry
        # enough mode signal for NMI ≳ 0.95.  Stronger augmentation forces
        # the contrastive task to rely on deeper mode-structural cues
        # rather than surface invariances.  Vibration shares the same
        # config but receives only the time-domain masks (envelope channel
        # is robust to gain_jitter and freq_mask doesn't apply).
        gain_jitter_db=9.0,
        channel_dropout_p=0.2,
        spec_augment_freq_mask=12,
        spec_augment_time_mask=16,
        # R1a REVERTED (2026-05-16, after the first wider-CNN retrain): the
        # 2× width regressed V1-acoustic NMI from 0.729 → 0.689 — the wider
        # CNN over-fit the tiny per-mode cohort (12 epochs × ~10 recordings
        # per mode) despite the stronger augmentation (R1b).  Reverted to
        # the published 32/64/128 backbone.  Keep R1b (augmentation push)
        # and R1c (CMA on by default) since those are independent.
        acoustic_cnn_width_mult=1,
        seed=42,
    )


def v2_config(quick: bool) -> V2SSLConfig:
    return V2SSLConfig(
        window_seconds=2.0,
        window_stride_seconds=1.0,
        feature_dim=64,
        embed_dim=64,
        n_heads=4,
        proj_dim=32,
        # Bumped 6→12 epochs (matches V1).  With asymmetric modality dropout
        # (vibration_dropout_p=0.5), the fusion block sees only ~ 50 % of
        # the vibration stream per batch and needs the extra epochs to
        # converge.
        epochs=3 if quick else 12,
        batch_size=16,
        lr=1e-3,
        # 1e-4 matches the standard SimCLR / AdamW convention. The dataclass
        # default of 1e-5 is one order too low and was the dominant contributor
        # to the overfitting audit's train/val gaps (it predates early stopping).
        # Set on every V1-V4 builder here, not at the dataclass, so older
        # import-sites stay byte-reproducible.
        weight_decay=1e-4,
        temperature=0.1,
        val_ratio=0.3,
        # n_mels / n_fft / hop_length inherited from `ACOUSTIC_FEATURES` — see v1_config note above.
        cwt_n_scales=32,
        # CWT enabled — see v1_config note.
        use_cwt=True,
        # Multi-scale window cadence — sourced from the dataset registry
        # (configs/datasets/d*.yaml). V1 and V2 must use the same dict
        # (V1→V2 weight transfer enforces shape parity per scale).
        window_scales_seconds_per_dataset={
            m.id: m.window_scales_seconds for m in REGISTRY
        },
        window_scale_strategy=WINDOWING.window_scale_strategy,
        window_stride_ratio=WINDOWING.window_stride_ratio,
        # Vibration channel-2: physical-time defaults inherited from
        # `compute_vibration_input_stack` (kurtosis on D4 raw; crest factor
        # on D1/D2/D3 peak streams).  See v1_config above for justification.
        # R1b — augmentation parity with V1.  See V1 v1_config note above.
        gain_jitter_db=9.0,
        channel_dropout_p=0.2,
        spec_augment_freq_mask=12,
        spec_augment_time_mask=16,
        lmm_mask_p=0.3,
        lmm_weight=1.0,
        # Asymmetric modality dropout: acoustic is the strong mode-
        # discriminator (V1-acoustic purity 0.727 vs V1-vib 0.572 in the
        # prior run).  Dropping vibration twice as often as acoustic stops
        # the fusion block from diluting the acoustic signal — the
        # mechanism that produced V2 purity 0.612 < V1-acoustic 0.727.
        modality_dropout_p=0.0,  # legacy fallback off
        acoustic_dropout_p=0.0,
        vibration_dropout_p=0.5,
        # R1c — CMA on by default.  The 2026-05-16 B5 ablation showed
        # cma_weight=0.5 lifts V1-acoustic NMI from 0.649 (baseline) to
        # 0.729 — the largest absolute V1 improvement of the Phase-B
        # sweep.  Keeping it on costs only a small per-step compute
        # increment.  Doesn't fix the joint-PMA fusion gap (still negative
        # at the modality-probe level) but the unimodal V1 trunks are
        # strictly stronger with it on, which matters for R2 (V3-acoustic /
        # V3-vibration adapters inherit V1 weights).
        cma_weight=0.5,
        cma_temperature=0.1,
        # Two PMA seeds in the context pool — one summary is bottlenecked
        # for a 9–14-token fused sequence.
        num_context_seeds=2,
        # R1a REVERTED — see `v1_config` note.  Match V1's narrow CNN.
        acoustic_cnn_width_mult=1,
        seed=42,
    )


def v3_config(quick: bool) -> V3Config:
    return V3Config(
        n_layers=6,
        hidden_dim=64,
        n_hidden_per_net=2,
        epochs=8 if quick else 15,
        batch_size=32,
        lr=1e-3,
        # 1e-4 matches the standard SimCLR / AdamW convention. The dataclass
        # default of 1e-5 is one order too low and was the dominant contributor
        # to the overfitting audit's train/val gaps (it predates early stopping).
        # Set on every V1-V4 builder here, not at the dataclass, so older
        # import-sites stay byte-reproducible.
        weight_decay=1e-4,
        val_ratio=0.3,
        unconditional=False,
        # K = 3 matches the 3-mode hypothesis (Pump / Standstill / Turbine).
        n_threshold_clusters=3,
        # p95 = lower-FPR-tolerant operating point that's still defensible
        # vs healthy variance (chapter 3 §3.4.6).
        threshold_percentile=95,
        # Per-stage V3 window override — sourced from the dataset registry
        # (configs/datasets/d*.yaml::v3_window_seconds).  Chapter 3 §3.4.4 +
        # chapter 4 §A.4 for the per-dataset justifications.
        window_seconds_override={m.id: m.v3_window_seconds for m in REGISTRY},
        # PMA-2 xt pool — see `_XtPool` docstring in `v3_trainer.py` for the
        # full thesis defense (closes the second mean-pool dilution stage
        # on the channel-token axis).
        xt_pool="pma2",
        xt_pool_num_heads=4,
        # CNF coupling MLP dropout — set as part of the 2026-05-22 baseline_v2
        # shift to defend against the +56 % V3 train/val NLL gap the audit
        # identified.  Threaded into FiLMMLP between GELU activations via
        # cnf_head.py.
        dropout_p=0.1,
        seed=42,
    )


def v4_config(quick: bool, scada_dim: int = 0, unconditional: bool = False) -> V4Config:
    return V4Config(
        cnn_feature_dim=64,
        tdoa_feature_dim=32,
        hidden_dim=64,
        n_heads_tdoa=2,
        scada_dim=scada_dim,
        unconditional=unconditional,
        # Per-stage V4 window override — sourced from the dataset registry
        # (configs/datasets/d*.yaml::v4_window_seconds).  Chapter 4 §A.4 for
        # the 4× SRP-PHAT SNR justification on D3/D4.
        window_seconds_override={m.id: m.v4_window_seconds for m in REGISTRY},
        # Soft-argmax + FiLM-residual head: most of the work is in the
        # 3-D CNN trunk, which is unchanged from the original Cross3D.
        # 30 epochs is enough for the residual to stabilise.
        epochs=15 if quick else 30,
        batch_size=8,
        lr=1e-3,
        # 1e-4 matches the standard SimCLR / AdamW convention. The dataclass
        # default of 1e-5 is one order too low and was the dominant contributor
        # to the overfitting audit's train/val gaps (it predates early stopping).
        # Set on every V1-V4 builder here, not at the dataclass, so older
        # import-sites stay byte-reproducible.
        weight_decay=1e-4,
        val_ratio=0.3,
        seed=42,
        # Soft-argmax / residual / loss / augmentation: see V4Config defaults.
        # Residual half-range = 20 cm. Soft-argmax has a centre-bias of 10–15 cm
        # on corner-of-grid positions (e.g. D4 `(-20, 0, 0)`); a tighter cap
        # cannot correct it (0.192 m MAE vs a dense regressor's 0.160 m).
        residual_scale_m=0.20,
        soft_argmax_temperature=1.0,
        train_in_centimetres=True,
        smooth_l1_beta=1.0,  # = 1 cm in the post-scale unit
        target_pos_noise_m=0.002,
        srp_volume_noise_std=0.02,
        tdoa_jitter_m=0.001,
        augment=True,
        # FiLM-residual head dropout — set as part of the 2026-05-22 baseline_v2
        # shift to defend against the +1236 % V4 train/val gap the audit
        # identified (10 labeled recordings, 50 fixed epochs, no early stop).
        # Threaded into FiLMResidualHead between GELU activations via
        # v4_loc_head.py.
        head_dropout_p=0.1,
    )
