"""V1 SSL configuration and result dataclasses."""
from __future__ import annotations

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
from ..encoders import PerModalityEncoder
from .v1_ssl_model import _ProjectionHead


def _registry_window_scales() -> dict[str, tuple[float, ...]]:
    """Per-dataset multi-scale window cadence sourced from the registry."""
    return {m.id: m.window_scales_seconds for m in REGISTRY}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class V1SSLConfig:
    """Hyperparameters for V1 per-modality SSL warmup.

    Every field default below is sourced from
    :mod:`src.config.architecture` — that file is the single source of
    truth for thesis-relevant numerical choices.  Do not change defaults
    here; edit `architecture.py` and let the change propagate.
    """

    # Window cadence (legacy single-scale knobs; preserved for back-compat).
    # When `window_scales_seconds == ()` the dataset falls back to the
    # legacy single scale `(window_seconds,)` with the explicit
    # `window_stride_seconds` — byte-equivalent to pre-2026-05-19 behaviour.
    window_seconds: float = WINDOWING.window_seconds
    window_stride_seconds: float = WINDOWING.window_stride_seconds

    # Multi-scale window cadence (2026-05-19, full justification in
    # `docs/chapters/chapter_3_field_data_and_preprocessing.md` §3.4.4):
    #
    #   * `window_scales_seconds`: a tuple of per-segment scales applied to
    #     every dataset.  Empty tuple means "use the legacy single scale".
    #   * `window_scales_seconds_per_dataset`: a dict
    #     `{dataset_id: (scale1, scale2, ...)}` that OVERRIDES the global
    #     tuple per dataset.  This is the publication path because the
    #     slowest-vibration constraint differs across datasets:
    #         - D1, D2 at 4 Hz vibration cannot resolve windows below
    #           5 / 4 = 1.25 s (one Vibration1DCNN kernel of length 5);
    #         - D3 at 16 Hz vibration can go down to 0.31 s;
    #         - D4 at 376 Hz raw vibration is effectively unconstrained.
    #
    # When `window_scale_strategy="uniform"`, each training batch is
    # single-dataset × single-scale (enforced by `_GroupedBatchSampler`'s
    # `(channel_count, n_frames)` bucket key); the scale within a segment's
    # valid set is sampled uniformly across all its windows by virtue of
    # the dataset emitting one (start, n_frames) tuple per (segment, scale).
    # Stride per scale = `scale * window_stride_ratio` (50 % overlap by
    # default — every frame appears in exactly two windows, the
    # audio-SSL convention).
    window_scales_seconds: tuple[float, ...] = ()
    window_scales_seconds_per_dataset: dict[str, tuple[float, ...]] = field(
        default_factory=_registry_window_scales
    )
    window_scale_strategy: Literal["fixed", "uniform"] = WINDOWING.window_scale_strategy
    window_stride_ratio: float = WINDOWING.window_stride_ratio

    # Encoder dims — sourced from `ENCODER` in architecture.py.
    feature_dim: int = ENCODER.feature_dim
    embed_dim: int = ENCODER.embed_dim
    n_heads: int = ENCODER.n_heads
    proj_dim: int = ENCODER.proj_dim

    # Training schedule — tuned per experiment, not centralised in
    # architecture.py.
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    temperature: float = 0.1
    val_ratio: float = 0.3

    # Acoustic feature parameters — sourced from `ACOUSTIC_FEATURES`.
    n_mels: int = ACOUSTIC_FEATURES.n_mels
    n_fft: int = ACOUSTIC_FEATURES.n_fft
    hop_length: int = ACOUSTIC_FEATURES.hop_length
    cwt_n_scales: int = ACOUSTIC_CWT.n_scales
    use_cwt: bool = True  # smoke-test override: skip CWT to speed up
    # F4 toggle — per-channel z-score of the log-mel + CWT stack before the
    # CNN.  Default False: the 2026-05-14 audit found F4 is not load-bearing
    # — the V1 acoustic encoder trains fine without it once BatchNorm is the
    # encoder norm (the real collapse cause was F7/GroupNorm, not F4).  Kept
    # as a knob because the log-mel-vs-CWT channel-scale mismatch is a real
    # concern worth revisiting if the SSL objective changes.
    standardize_acoustic: bool = False

    # Vibration feature parameters — see `compute_vibration_input_stack` for
    # the full justification.  Channel-2 statistic (kurtosis vs crest factor)
    # is selected per-dataset from the segment's accel_sample_rate so a
    # single physical-time knob works across D1/D2 (4 Hz), D3 (16 Hz), and
    # D4 raw (~376 Hz).
    vib_kurtosis_window_seconds: float = VIBRATION_FEATURES.kurtosis_window_seconds
    vib_min_kurtosis_samples: int = VIBRATION_FEATURES.min_kurtosis_samples
    vib_crest_factor_window_seconds: float = VIBRATION_FEATURES.crest_factor_window_seconds
    vib_min_crest_factor_samples: int = VIBRATION_FEATURES.min_crest_factor_samples
    # Vibration amplitude + envelope z-score toggle (pre-existing behaviour;
    # default True).  Channel 2 (impulsiveness) is dimensionless and never
    # re-standardised — the F5 experiment that z-scored it showed no benefit
    # and was reverted.
    standardize_vibration: bool = True

    # R1a — Acoustic2DCNN channel-width multiplier.  Default 1 reproduces
    # the published 32/64/128 backbone; set to 2 for the wider 64/128/256
    # variant.  V1 and V2 must use the same value so that V2 can load
    # V1 acoustic weights without shape mismatch.  Vibration backbone is
    # not scaled — the R1 experiment changes acoustic only.
    acoustic_cnn_width_mult: int = ENCODER.acoustic_cnn_width_mult

    # Augmentations (applied in feature space) — per-experiment knobs;
    # the orchestrator's `v1_config(quick)` overrides to publication values.
    gain_jitter_db: float = 6.0
    channel_dropout_p: float = 0.2
    spec_augment_freq_mask: int = 6
    spec_augment_time_mask: int = 8

    # Labels — healthy operating modes only.  RandomFault is anomaly data and
    # must not enter SSL training or the cluster-purity eval (which uses K=3
    # against {Pump, Standstill, Turbine}).  D3's `Healthy` token covers its
    # speed-bucket recordings.
    healthy_modes: tuple[str, ...] = (
        "Pump",
        "Standstill",
        "Turbine",
        "Healthy",
    )

    # Early stopping on val NT-Xent loss.  Patience counts consecutive epochs
    # without > `early_stop_min_delta` absolute improvement; when reached, the
    # loop breaks and (if `restore_best`) the encoder weights are reset to the
    # epoch with the best (lowest) val loss.  Best-state snapshot uses a
    # tensor-clone dict comprehension rather than `copy.deepcopy` to avoid
    # RAM fragmentation and PyTorch-autograd edge cases on long runs.
    patience: int = 3
    restore_best: bool = True
    early_stop_min_delta: float = 1e-4

    # Cross-recording mixup.  When > 0, each training window is linearly
    # blended with a partner window from a *different recording, same mode,
    # same dataset, same shape*: `feat = λ * feat_anchor + (1-λ) * feat_partner`
    # with `λ ~ Beta(α, α)`.  Inflates the effective recording cohort from
    # O(N_rec) to O(N_rec²) — the highest-ROI data-side regularizer for the
    # 6-10-recording cohort.  Default 0.0 disables (byte-equivalent to pre-fix
    # behaviour); recommended ablation values: {0.0, 0.2, 0.4}.  Only applied
    # to TRAIN windows — val windows are passed through untouched.
    mixup_alpha: float = 0.0

    # System
    seed: int = 42
    device: str = "auto"

    extra: dict = field(default_factory=dict)
def _dataset_idx(dataset_id: str) -> int:
    """Registry-driven dataset index for embedding lookups.

    The integer index comes from ``DatasetRegistry`` (alphabetical-sorted by
    canonical id) and is stable across runs.  Aliases (e.g. ``illwerke`` ->
    ``illwerke_raw``) resolve to the same canonical index.
    """
    return REGISTRY.index_of(dataset_id)


# ---------------------------------------------------------------------------
# Feature precomputation
# ---------------------------------------------------------------------------
@dataclass
class V1Result:
    encoder: PerModalityEncoder
    projection: _ProjectionHead
    train_loss_history: list[float]
    val_loss_history: list[float]
    train_recording_ids: list[str]
    val_recording_ids: list[str]
    sanity_gate: dict
    modality: str
    # None means the loop ran to `cfg.epochs`; an int N means the loop broke
    # after epoch N because val loss did not improve for `cfg.patience` epochs.
    # `best_val_loss` is the min val loss observed (which `restore_best`
    # restored the encoder to if enabled).
    early_stopped_epoch: int | None = None
    best_val_loss: float = float("nan")
