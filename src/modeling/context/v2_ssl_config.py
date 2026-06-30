"""V2 SSL configuration and result dataclasses."""

from dataclasses import dataclass, field
from typing import Literal

from ...config.architecture import (
    ACOUSTIC_CWT,
    ACOUSTIC_FEATURES,
    ENCODER,
    VIBRATION_FEATURES,
    WINDOWING,
)
from ...config.dataset_registry import REGISTRY
from .v2_fusion import V2FusionEncoder
from .v2_ssl_model import _ProjectionHead


def _registry_window_scales() -> dict[str, tuple[float, ...]]:
    """Per-dataset multi-scale window cadence sourced from the registry."""
    return {m.id: m.window_scales_seconds for m in REGISTRY}


# Note: `hop_for_dataset` removed 2026-05-21 after the empirical grid sweep
# (scripts/hop_length_study/analyze_hop_length_full_grid.py + chapter 3 §3.4.2) found that
# hop_length contributes < 0.005 ROC AUC across the [32, 4096] range once
# (n_fft, n_mels) are set correctly.  The previous per-dataset hop plumbing
# was based on a "cross-modal alignment" intuition that the empirical test
# refuted.  A single global hop_length=2048 (from ACOUSTIC_FEATURES) is now
# the architectural choice on all datasets.


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class V2SSLConfig:
    # Defaults sourced from `src.config.architecture` — change there, not here.

    # Window cadence (wall-clock seconds).  The legacy single-scale knobs
    # remain authoritative when `window_scales_seconds` is the empty tuple
    # (byte-equivalent to pre-2026-05-19).
    window_seconds: float = WINDOWING.window_seconds
    window_stride_seconds: float = WINDOWING.window_stride_seconds

    # Multi-scale window cadence (see `V1SSLConfig` for the full
    # justification).  When the per-dataset dict is populated, each batch
    # is single-dataset × single-scale via the grouped batch sampler's
    # `(channel_count, n_frames_ac, n_frames_vib)` bucket key.  Both
    # `n_ac` and `n_vib` are computed from the same per-batch scale so
    # the wall-clock pairing in `_PairedWindowedDataset` is preserved.
    window_scales_seconds: tuple[float, ...] = ()
    window_scales_seconds_per_dataset: dict[str, tuple[float, ...]] = field(
        default_factory=_registry_window_scales
    )
    window_scale_strategy: Literal["fixed", "uniform"] = WINDOWING.window_scale_strategy
    window_stride_ratio: float = WINDOWING.window_stride_ratio

    # Encoder dims (must match V1 checkpoints when loaded)
    feature_dim: int = ENCODER.feature_dim
    embed_dim: int = ENCODER.embed_dim
    n_heads: int = ENCODER.n_heads
    proj_dim: int = ENCODER.proj_dim

    # Training schedule — per-experiment, not centralised.
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    temperature: float = 0.1
    val_ratio: float = 0.3

    # Acoustic features
    n_mels: int = ACOUSTIC_FEATURES.n_mels
    n_fft: int = ACOUSTIC_FEATURES.n_fft
    hop_length: int = ACOUSTIC_FEATURES.hop_length
    cwt_n_scales: int = ACOUSTIC_CWT.n_scales
    use_cwt: bool = True
    # F4 toggle — see `V1SSLConfig.standardize_acoustic`.  Must match the
    # V1 setting used to pretrain the inherited encoder weights, otherwise
    # the V2 fusion block sees an acoustic input distribution the V1 CNN
    # was never trained on.  Default False (F4 found not load-bearing).
    standardize_acoustic: bool = False

    # Vibration features — physical-time window with per-dataset
    # channel-2 statistic (kurtosis vs crest factor); see
    # `compute_vibration_input_stack` for the justification.
    vib_kurtosis_window_seconds: float = VIBRATION_FEATURES.kurtosis_window_seconds
    vib_min_kurtosis_samples: int = VIBRATION_FEATURES.min_kurtosis_samples
    vib_crest_factor_window_seconds: float = VIBRATION_FEATURES.crest_factor_window_seconds
    vib_min_crest_factor_samples: int = VIBRATION_FEATURES.min_crest_factor_samples
    # Vibration amplitude + envelope z-score — see
    # `V1SSLConfig.standardize_vibration`.
    standardize_vibration: bool = True

    # R1a — Acoustic2DCNN channel-width multiplier; must match the V1 value
    # used to pretrain the inherited V1-acoustic weights or
    # `load_v1_weights(strict=True)` will fail.  See
    # `V1SSLConfig.acoustic_cnn_width_mult`.
    acoustic_cnn_width_mult: int = ENCODER.acoustic_cnn_width_mult

    # Augmentations (feature space) — per-experiment knobs.
    gain_jitter_db: float = 6.0
    channel_dropout_p: float = 0.2
    spec_augment_freq_mask: int = 6
    spec_augment_time_mask: int = 8

    # Latent Masked Modeling — per-experiment knobs.
    lmm_mask_p: float = 0.3
    lmm_weight: float = 1.0

    # Modality dropout — independent per-modality dropout probabilities,
    # applied per batch before the cross-attention block.  Asymmetric by
    # default: the V1 / V2 evaluations show acoustic is the strong
    # mode-discriminator (purity 0.727) while vibration is weak (0.572)
    # and hurts fusion (V2 purity 0.612 < V1-acoustic 0.727).  Dropping
    # vibration more often than acoustic teaches the fusion block to
    # treat acoustic as the trunk and vibration as auxiliary, recovering
    # the V1-acoustic ceiling rather than the unweighted-mean of the two.
    # The legacy `modality_dropout_p` field still exists for back-compat:
    # when both new fields are 0.0 it falls back to the symmetric 50/50
    # coin-flip with that probability.
    # Modality dropout — asymmetric: vibration dropped 50 % to stop the
    # fusion block from diluting the dominant acoustic mode-discriminator.
    modality_dropout_p: float = 0.0  # legacy symmetric fallback
    acoustic_dropout_p: float = 0.0
    vibration_dropout_p: float = 0.5

    # Cross-modal alignment (CMA) — NT-Xent between per-modality PMA
    # summaries.  R1c publication run uses cma_weight = 0.5.
    cma_weight: float = 0.0
    cma_temperature: float = 0.1

    # Context-aggregation mode for c_t — see V2FusionEncoder.ContextMode.
    context_mode: str = "joint_pma"

    # Number of learned PMA seeds in the V2 joint-context pool — the
    # ARCHITECTURAL choice (Set Transformer §3.2, Lee et al. ICML 2019).
    num_context_seeds: int = ENCODER.num_context_seeds

    # Ablation A1 — drop vibration branch (zero its features at the input)
    drop_vibration: bool = False

    # Labels — healthy operating modes only.  RandomFault must not enter
    # SSL training (label-leakage invariant).
    healthy_modes: tuple[str, ...] = (
        "Pump",
        "Standstill",
        "Turbine",
        "Healthy",
    )

    # Early stopping on val total-loss (simclr + lmm*w + cma*w).  See V1SSLConfig
    # for the snapshot-strategy rationale (tensor-clone, not deepcopy).
    patience: int = 3
    restore_best: bool = True
    early_stop_min_delta: float = 1e-4

    # Cross-recording mixup — see V1SSLConfig docstring for the design.  V2
    # blends both modalities with the same λ from the same partner recording,
    # preserving the cross-modal pairing inside each mixed window.  Default
    # 0.0 disables; recommended ablation values: {0.0, 0.2, 0.4}.
    mixup_alpha: float = 0.0

    # System
    seed: int = 42
    device: str = "auto"

    extra: dict = field(default_factory=dict)
def _dataset_idx(dataset_id: str) -> int:
    """Registry-driven dataset index for embedding lookups.  See v1_ssl._dataset_idx."""
    return REGISTRY.index_of(dataset_id)


# ---------------------------------------------------------------------------
# Paired feature precomputation
# ---------------------------------------------------------------------------
@dataclass
class V2Result:
    encoder: V2FusionEncoder
    projection: _ProjectionHead
    train_loss_history: list[float]
    val_loss_history: list[float]
    train_simclr_history: list[float]
    train_lmm_history: list[float]
    train_recording_ids: list[str]
    val_recording_ids: list[str]
    rq1: dict
    drop_vibration: bool
    early_stopped_epoch: int | None = None
    best_val_loss: float = float("nan")
