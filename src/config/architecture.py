"""Single source of truth for the **architecture-defining** numerical choices.

Scoped deliberately to the constants worth centralising — the ones that:

  * Encode a thesis-level design decision (e.g. ASP pool, multi-scale
    window dict, per-stage overrides, CWT-band).
  * Are cross-referenced by multiple modules (feature extractor +
    encoder + dataset).
  * Must be changed atomically across the pipeline to avoid
    inconsistency (e.g. hop_length affects acoustic_fs in
    `_PairedSegment`, log-mel time grid, CWT alignment, and V2 cross-
    attention temporal cadence — change it in one place).

not scoped here, intentionally:

  * Training schedules (LR, batch size, epochs, weight decay,
    temperature) — these belong on each stage's dataclass because they
    are tuned per experiment and reported per row of the results
    chapter.
  * Augmentation strengths (gain jitter dB, channel dropout, SpecAug
    masks) — these are R1 / R1b ablation knobs reported alongside
    the corresponding result, not architectural commitments.
  * Numerical-conditioning knobs (V4's centimetre-rescaling,
    smooth-L1 beta, augmentation noise stds) — these are
    implementation details that aren't part of the architectural
    contract.
  * Per-experiment seeds, devices, validation ratios — operational.
  * Healthy-mode tuples, dataset indices — fixed by the recording
    protocol / labeling convention, not "config".

Every numerical default below carries a one-line comment naming
either the data property that pinned it (e.g. ROW II tone spacing,
D4 vibration rate, knock-impulse duration) or the paper that argued
for it.  Long-form justifications live in:
  * `docs/chapters/chapter_3_field_data_and_preprocessing.md` §3.4
  * `docs/chapters/chapter_4_encoder_architecture_addendum.md`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Re-export the ROW II / sensor-array physical constants from
# `constants.py` for a single import.
from .constants import (
    ACCEL_COUNT,
    ACCEL_SAMPLE_RATE_TARGET,
    CASING_RADIUS_M,
    GENERATOR_LEVEL_Z_M,
    MIC_COUNT,
    MIC_SAMPLE_RATE,
    SENSOR_LAYOUT,
    TURBINE_LEVEL_Z_M,
    mic_cartesian_positions,
)

# ============================================================================
# 1 · Acquisition — per-dataset sensor rates
# ============================================================================


@dataclass(frozen=True)
class _Acquisition:
    """Acquisition facts that are GLOBAL across datasets.

    Per-dataset acquisition facts (vibration target rate, vibration format,
    per-sensor SR overrides) used to live here as `accel_target_hz_per_dataset`
    / `vibration_format_per_dataset` dicts.  Those have moved to the
    canonical source — ``configs/datasets/d*.yaml`` — and are exposed via
    :data:`src.config.dataset_registry.REGISTRY`.  Use
    ``REGISTRY.get(dataset_id).accel_target_sr`` (etc.) instead of any
    hardcoded dict.
    """

    # 16 kHz mono WAV across all datasets.
    mic_sample_rate_hz: int = 16_000


ACQUISITION = _Acquisition()


# ============================================================================
# 2 · Acoustic feature extraction
# ============================================================================


@dataclass(frozen=True)
class _AcousticFeatures:
    """STFT + log-mel grid — empirically validated by full grid search on all
    5 datasets.  See chapter 3 §3.4.2 and `scripts/hop_length_study/analyze_hop_length_full_grid.py`.

    The selected configuration is the Pareto-optimal point in a 152-configuration
    sweep over (n_fft, hop_length, n_mels), measured by ROC AUC of per-frame
    knock-band log-mel energy as a binary anomaly-vs-healthy classifier on
    held-out recordings from D1–D5.
    """

    # n_fft = 4096 → 3.9 Hz frequency-bin width at 16 kHz mic.  Empirically
    # required to resolve the ROW II 100 / 117 Hz target-tone pair (synthetic
    # two-sine test in `analyze_hop_length_full_grid`).  Smaller n_fft loses
    # 4–14 AUC points on D2 / D3 (the two datasets whose anomaly signatures
    # depend on the tone-pair separation).
    n_fft: int = 4096

    # hop_length = 2048 → 50 % STFT overlap, 7.8 Hz acoustic frame rate.
    # The Pareto-optimal hop in the full grid: mean AUC = 0.9278 at 0.047 s
    # per recording, within 0.0007 AUC of the absolute peak (hop=32, cost=
    # 2.57 s — 55× more expensive).  Once n_fft and n_mels are right, hop
    # is empirically irrelevant: AUC varies by only 0.0045 across the full
    # hop ∈ [32, 4096] sweep at fixed n_fft=4096, n_mels=96.  The legacy
    # cross-modal grid alignment argument (matching acoustic frame rate to
    # raw vibration rate) is not load-bearing — V2's cross-attention learns
    # alignment from data and is insensitive to absolute T-axis dimensionality.
    hop_length: int = 2048

    # n_mels = 96 → highest mel-filterbank resolution in the empirical sweep.
    # Gives uniformly higher AUC than n_mels ∈ {32, 48, 64} at the same
    # (n_fft, hop) on every dataset; cost penalty is sublinear because the
    # mel projection is a small matrix multiply.
    n_mels: int = 96

    fmin_hz: float = 20.0   # mic roll-off + 50 Hz AC-line rejection floor
    fmax_hz: float | None = None  # None → Nyquist (8 kHz at 16 kHz mic)

    # power_to_db: ref=1.0 preserves cross-recording amplitude; top_db
    # clips noise floor 80 dB below peak.
    top_db: float = 80.0


ACOUSTIC_FEATURES = _AcousticFeatures()


# ============================================================================
# 3 · CWT scalogram (channel 1 of the acoustic encoder input)
# ============================================================================


@dataclass(frozen=True)
class _AcousticCWT:
    """Complex-Morlet CWT scalogram.  Chapter 3 §3.4.2."""

    # B=1.5 / C=1.0 — balanced time/frequency localisation.
    wavelet: str = "cmor1.5-1.0"

    # 64 scales over 20–250 Hz = 17.6 bins / octave → ~4 Hz spacing at
    # 100 Hz, cleanly separating the 100 / 117 Hz pair the mel filterbank
    # cannot resolve.  This is the resolution argument for the asymmetric-
    # band design.
    n_scales: int = 64
    min_freq_hz: float = 20.0
    max_freq_hz: float = 250.0   # one octave above 117 Hz vane-pass

    # Decimate to 1 kHz before CWT: 16× memory savings, no info loss
    # below the 500 Hz post-decimation Nyquist (covers all ROW II tones).
    decimate_to_hz: int = 1000


ACOUSTIC_CWT = _AcousticCWT()


# ============================================================================
# 4 · Vibration feature extraction (channel-2 statistic thresholds)
# ============================================================================


@dataclass(frozen=True)
class _VibrationFeatures:
    """Vibration channel-2 statistic.  Chapter 3 §3.4.3."""

    # Knock-impulse ring-down ~ 10–100 ms ⇒ 100 ms physical window.
    kurtosis_window_seconds: float = 0.10

    # σ_kurtosis ≈ √(24/N) ≤ 0.9 under Gaussian H_0 ⇒ N ≥ 30; we round
    # up to the next odd integer.
    min_kurtosis_samples: int = 31

    # Below the kurtosis floor we fall back to crest factor over a 1 s
    # window — defined down to N = 4 (peak / RMS estimator), ISO-10816
    # impulsiveness indicator.
    crest_factor_window_seconds: float = 1.0
    min_crest_factor_samples: int = 4

    # Z-score amplitude + envelope channels.  The channel-2 impulsiveness
    # statistic is dimensionless and never re-standardised (F5 audit,
    # 2026-05-14).
    standardize: bool = True


VIBRATION_FEATURES = _VibrationFeatures()


# ============================================================================
# 5 · Encoder architecture (CNN backbone + pool + set-transformer dims)
# ============================================================================


@dataclass(frozen=True)
class _Encoder:
    """Per-modality CNN + pool + set-transformer dims.  Chapter 4 §A.1."""

    # Per-modality summary dim and per-token embed dim.  Must match
    # across V1 and V2 (V1→V2 weight transfer enforces this).
    feature_dim: int = 128
    embed_dim: int = 128
    n_heads: int = 4
    # SimCLR projection-head output dim (V1/V2 contrastive only —
    # discarded at downstream evaluation).
    proj_dim: int = 64

    # CNN backbone channel widths (the published 32/64/128 backbone).
    # `width_mult` scales all three together; R1a (2×) regressed V1
    # acoustic NMI 0.729 → 0.689, so default stays at 1.
    cnn_c1: int = 32
    cnn_c2: int = 64
    cnn_c3: int = 128
    acoustic_cnn_width_mult: int = 1

    # BatchNorm — load-bearing for SimCLR-style contrastive (the F7 audit
    # showed GroupNorm collapses V1 acoustic).
    norm: Literal["batch", "group"] = "batch"

    # ASP — Okabe et al. Interspeech 2018; r=8 inherited from ECAPA
    # (Desplanques et al. 2020).  See chapter 4 §A.1.4 for the
    # data-grounded r=8 justification (5-mode discrimination + 2× headroom).
    pool_type: Literal["asp", "avg"] = "asp"
    pool_reduction: int = 8

    # Multi-seed PMA for V2's joint context pool (Lee et al. ICML 2019).
    # 1 → 2 lifts c_t capacity at no downstream c_dim cost.
    num_context_seeds: int = 2


ENCODER = _Encoder()


# ============================================================================
# 6 · Windowing — multi-scale + per-stage overrides
# ============================================================================


@dataclass(frozen=True)
class _Windowing:
    """Window cadence — GLOBAL knobs only.

    Per-dataset cadence (multi-scale tuple, V3 inference window, V4 inference
    window) used to live here as `window_scales_seconds_per_dataset` /
    `v3_window_seconds_override` / `v4_window_seconds_override` dicts.  Those
    have moved to ``configs/datasets/d*.yaml`` and are exposed via
    :data:`src.config.dataset_registry.REGISTRY`.  Use
    ``REGISTRY.get(dataset_id).window_scales_seconds`` (etc.) instead of any
    hardcoded dict.
    """

    # Legacy single-scale fallback (used when a dataset's
    # ``window_scales_seconds`` is empty).  Equivalent to pre-2026-05-19
    # single-scale behaviour; kept so existing tests / ablation runs that
    # bypass the multi-scale path stay reproducible.
    window_seconds: float = 2.0
    window_stride_seconds: float = 1.0

    window_scale_strategy: Literal["uniform", "fixed"] = "uniform"

    # 50 % overlap — audio-SSL convention (every frame appears in two
    # windows; Choi et al. 2017).
    window_stride_ratio: float = 0.5


WINDOWING = _Windowing()


# ============================================================================
# 7 · V3 anomaly head — architectural choices (not training schedule)
# ============================================================================


@dataclass(frozen=True)
class _V3Anomaly:
    """V3 conditional flow architecture + xt_pool + threshold scheme."""

    # CNF depth + width.
    n_layers: int = 6
    hidden_dim: int = 64

    # Tanh log-scale bound per coupling (Dinh et al. 2017 RealNVP);
    # 2.0 = each layer's Jacobian in [e⁻², e²].
    scale_max: float = 2.0

    # K = 3 matches the operating-mode hypothesis (Pump/Standstill/Turbine).
    # p95 = lower-FPR-tolerant operating point.
    n_threshold_clusters: int = 3
    threshold_percentile: int = 95

    # Empirical-Bayes shrinkage of each cluster's percentile toward the global
    # one (weight n_k/(n_k+shrinkage)).  Stops a single small/mis-fit cluster
    # from blowing the held-out healthy alert rate to 0.7+ (observed on the
    # conditional V3 acoustic arm).  0 = pure per-cluster (legacy).  At the
    # typical fit-cluster size (~200-400 windows) shrinkage=300 pulls a cluster
    # ~40-60 % toward the stable global boundary — enough to kill the blowup
    # while keeping meaningful per-regime adaptation.  Tune up if a cluster
    # still over-fires, down toward 0 for the pure context-conditional ablation.
    threshold_shrinkage: float = 300.0

    # xt_pool kind: PMA-2 = publication (chapter 4 §A.3); mean = legacy.
    xt_pool: Literal["pma2", "mean"] = "pma2"
    xt_pool_num_heads: int = 4

    # Context-conditional base distribution N(μ(c), σ(c)²) for the CNF.  Lets
    # the "centre of normal" move with the operating regime so -log p(x|c) is
    # regime-normalised by construction (fixes the cross-regime confound that
    # masked single-regime faults).  Zero-init ⇒ identical to N(0,I) at init.
    conditional_base: bool = True

    # Append impulse+spectral condition-monitoring features to the conditional
    # flow's input x (RQ2): the SSL embedding discards the impulsiveness a knock
    # produces, so the conditional detector cannot see it.  Injecting the
    # hand-crafted anchor restores the anomaly signal WITHOUT abandoning
    # context-conditioning (the flow still conditions on c_t).  Ablatable:
    # False reproduces the embedding-only conditional flow.
    inject_impulse_anchor: bool = True


V3_ANOMALY = _V3Anomaly()


# ============================================================================
# 8 · V4 localisation — architectural choices (not training schedule / aug)
# ============================================================================


@dataclass(frozen=True)
class _V4Localization:
    """V4 head architecture.  Numerical-conditioning + augmentation knobs
    stay on `V4Config` itself because they are per-run tunables."""

    # Head dims — narrower than V1/V2 because V4 supervision is tiny
    # (~6 labelled recordings); the residual head over-fits at full
    # encoder width.
    cnn_feature_dim: int = 64
    tdoa_feature_dim: int = 32
    hidden_dim: int = 64
    n_heads_tdoa: int = 2

    # Residual half-range = 20 cm.  Sized to cover the gap between the
    # SRP-PHAT soft-argmax prior and the labelled source position on the
    # current prototypes (D3/D4 circular rig: ~ 10 cm sensor envelope;
    # D2 rectangular rig: ~ 41 cm y-axis extent).  The relevant comparison
    # is "how far from the grid centre can a corner-of-grid soft-argmax
    # be from its true voxel" — the inward-bias of soft-argmax on edge
    # positions is ~ 10-15 cm on the D2 cohort, and 20 cm leaves a safety margin without
    # letting the residual swing wildly under noisy gradients.  An
    # earlier comment here claimed "any voxel in the 10 cm prototype
    # bounding box" — incorrect for D2; see positions.py for the
    # actual per-dataset envelopes.
    residual_scale_m: float = 0.20
    soft_argmax_temperature: float = 1.0


V4_LOCALIZATION = _V4Localization()


# ============================================================================
# 9 · Cross-modal sync correction — gating thresholds
# ============================================================================


@dataclass(frozen=True)
class _Sync:
    """Four-gate auto-sync thresholds.  See ingestion/sync_verification.py."""

    # Search window half-width for envelope cross-correlation.
    max_offset_s: float = 0.5

    # Sub-segments for the time-stability check (Bregler & Konig 1998).
    n_sub_segments: int = 5

    # Confidence ratio floor — peak-to-second-largest cross-corr value.
    confidence_floor: float = 1.5

    # Stability check tolerances.
    drift_tolerance_s: float = 0.010       # max drift across sub-segments
    min_offset_to_correct_s: float = 0.001  # below this correction has no effect

    # Gate 0 — envelope kurtosis floor.  Below 1.0 the acoustic envelope
    # is near-Gaussian; xcorr has no peak to lock onto.
    min_envelope_kurtosis: float = 1.0

    use_fractional_shift: bool = True


SYNC = _Sync()


# ============================================================================
# Public API
# ============================================================================

__all__ = [  # noqa: RUF022 — grouped by origin (own settings vs constants.py re-exports)
    "ACQUISITION",
    "ACOUSTIC_FEATURES",
    "ACOUSTIC_CWT",
    "VIBRATION_FEATURES",
    "ENCODER",
    "WINDOWING",
    "V3_ANOMALY",
    "V4_LOCALIZATION",
    "SYNC",
    # Re-exports from constants.py
    "ACCEL_COUNT",
    "ACCEL_SAMPLE_RATE_TARGET",
    "CASING_RADIUS_M",
    "GENERATOR_LEVEL_Z_M",
    "MIC_COUNT",
    "MIC_SAMPLE_RATE",
    "SENSOR_LAYOUT",
    "TURBINE_LEVEL_Z_M",
    "mic_cartesian_positions",
]
