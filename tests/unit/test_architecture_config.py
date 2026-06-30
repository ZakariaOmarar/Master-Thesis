"""Self-consistency tests for `src.config.architecture`.

This file is the single source of truth for the pipeline's
architecture-defining numerical choices.  Each test below pins one
load-bearing property of the contents — if any of them regresses, the
publication architecture has drifted from its documented design.
"""

from __future__ import annotations

from src.config.architecture import (
    ACOUSTIC_CWT,
    ACOUSTIC_FEATURES,
    ACQUISITION,
    ENCODER,
    SYNC,
    V3_ANOMALY,
    V4_LOCALIZATION,
    VIBRATION_FEATURES,
    WINDOWING,
)
from src.config.dataset_registry import REGISTRY

# ---------------------------------------------------------------------------
# 1 · Acquisition: per-dataset rates and formats (registry-sourced)
# ---------------------------------------------------------------------------


def test_registry_covers_all_published_datasets() -> None:
    """Every dataset the orchestrator iterates over must be in the registry,
    with `accel_target_sr` populated.  A missing entry would crash the
    loader rather than silently routing through a 4 Hz default — but the
    test pins the published cohort so a stale d*.yaml deletion is caught."""
    expected = {"d1", "d2", "d3", "d4", "d5", "illwerke_raw"}
    assert set(REGISTRY.all_ids()) == expected
    for meta in REGISTRY:
        assert meta.accel_target_sr > 0, (
            f"{meta.id}: accel_target_sr={meta.accel_target_sr} not populated"
        )


def test_d4_and_d5_ship_raw_vibration() -> None:
    """D4 and D5 ship `vibration_raw_*.csv` waveforms; D1/D2/D3 ship only
    the peak-amplitude stream.  The `vibration_format: auto` policy resolves
    to raw on D4/D5 and to peak on D1/D2/D3 via path inspection — but the
    registry's declared format stays "auto" for all.  This test pins the
    *shipping fact*: the data directories of D4/D5 must contain raw CSVs."""
    for ds in ("d4", "d5"):
        meta = REGISTRY.get(ds)
        raw_files = list(meta.root.rglob("vibration_raw_*.csv"))
        assert raw_files, f"{ds}: expected vibration_raw_*.csv files under {meta.root}"
    for ds in ("d1", "d2", "d3"):
        meta = REGISTRY.get(ds)
        if not meta.root.exists():
            continue
        raw_files = list(meta.root.rglob("vibration_raw_*.csv"))
        assert not raw_files, f"{ds}: unexpected raw CSVs under {meta.root}"


# ---------------------------------------------------------------------------
# 2 · Acoustic features: ROW II target-tone resolution + hop / vib match
# ---------------------------------------------------------------------------


def test_stft_bin_width_resolves_rotor_pole_vs_vane_pass() -> None:
    """STFT bin width must be ≤ 17 Hz (100 Hz rotor-pole − 117 Hz vane-pass).

    The empirical sweep (chapter 3 §3.4.2 + scripts/hop_length_study/analyze_hop_length_full_grid.py)
    showed n_fft=1024 (bin width 15.6 Hz) does NOT cleanly separate the two
    tones; only n_fft >= 2048 (bin width ≤ 7.8 Hz) does.  The publication
    pick is n_fft=4096 (bin width 3.9 Hz) for maximum AUC on the D2/D3
    cohorts whose anomaly signatures depend on tone-pair separation."""
    bin_width = ACQUISITION.mic_sample_rate_hz / ACOUSTIC_FEATURES.n_fft
    # Empirical floor: 8 Hz is the largest bin width that resolved the
    # 100/117 Hz pair in the synthetic two-sine test.
    assert bin_width <= 8.0, (
        f"STFT bin width {bin_width:.2f} Hz fails the empirical "
        f"100/117 Hz tone-resolution floor (8 Hz) — chapter 3 §3.4.2"
    )


def test_n_fft_resolves_target_tone_pair_empirically() -> None:
    """Direct empirical test: at the current n_fft, can we actually
    distinguish 100 Hz and 117 Hz peaks in a synthetic two-sine mixture?"""
    import numpy as np
    fs = ACQUISITION.mic_sample_rate_hz
    n_fft = ACOUSTIC_FEATURES.n_fft
    t = np.arange(n_fft) / fs
    sig = (np.sin(2 * np.pi * 100.0 * t) + np.sin(2 * np.pi * 117.0 * t)) * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mask = (freqs >= 80.0) & (freqs <= 140.0)
    local_spec = spec[mask]
    local_freqs = freqs[mask]
    peaks = [
        i for i in range(1, len(local_spec) - 1)
        if local_spec[i] > local_spec[i - 1] and local_spec[i] > local_spec[i + 1]
    ]
    peak_freqs = [float(local_freqs[i]) for i in peaks]
    bin_width = fs / n_fft
    # Both 100 Hz and 117 Hz must have a corresponding peak within 1.5 bins.
    for target in (100.0, 117.0):
        assert any(abs(pf - target) < 1.5 * bin_width for pf in peak_freqs), (
            f"n_fft={n_fft} failed to expose a peak near {target} Hz "
            f"(peaks found: {peak_freqs})"
        )


# Removed `test_hop_length_has_nyquist_margin_over_highest_target_tone`:
# The "acoustic frame rate must Nyquist-sample the 117 Hz vane-pass tone"
# argument conflated frequency resolution (set by n_fft) with temporal
# resolution (set by hop).  The 117 Hz tone is resolved as a discrete
# spectral peak in EACH STFT frame regardless of frame rate; the hop only
# affects how quickly we'd track that tone's *amplitude modulations* — a
# concern only if the modulation rate exceeds 1/2 × frame_rate, which is
# not the case for any ROW II tone (modulations are sub-Hz).  See
# chapter 3 §3.4.2 empirical sweep: AUC was hop-invariant across the
# [32, 4096] range.


# ---------------------------------------------------------------------------
# 3 · CWT: ROW II target-tone resolution
# ---------------------------------------------------------------------------


def test_cwt_band_covers_all_row_ii_target_tones() -> None:
    """The 20–250 Hz CWT band must cover every ROW II target tone
    (5.87, 6.25, 43.75, 100, 117 Hz) with at least one octave of margin
    above the highest tone."""
    target_tones = [5.87, 6.25, 43.75, 100.0, 117.0]
    for tone in target_tones:
        # Below CWT min_freq_hz one would not detect that tone; this is OK
        # for the 5.87 / 6.25 Hz fundamentals because they are tracked by
        # the mel filterbank, which goes down to 20 Hz with its own log
        # spacing.  We only require that the >= 20 Hz tones are inside
        # the CWT band.
        if tone >= ACOUSTIC_CWT.min_freq_hz:
            assert tone <= ACOUSTIC_CWT.max_freq_hz, (
                f"target tone {tone} Hz outside CWT max {ACOUSTIC_CWT.max_freq_hz} Hz"
            )


def test_cwt_decimate_to_hz_below_nyquist_of_target_band() -> None:
    """Post-decimation Nyquist must exceed the CWT max_freq with a 2× margin."""
    post_decim_nyquist = ACOUSTIC_CWT.decimate_to_hz / 2
    assert post_decim_nyquist >= 2 * ACOUSTIC_CWT.max_freq_hz, (
        f"post-decimation Nyquist {post_decim_nyquist} Hz < 2× CWT max "
        f"({2 * ACOUSTIC_CWT.max_freq_hz} Hz); CWT band would alias"
    )


# ---------------------------------------------------------------------------
# 4 · Vibration features
# ---------------------------------------------------------------------------


def test_kurtosis_statistical_floor_is_robust_to_gaussian_null() -> None:
    """σ_kurtosis ≈ √(24/N) must be ≤ 0.9 under the Gaussian H_0 at the
    floor — the threshold chosen so weak transient impulses (excess
    kurtosis ~3–5) are not drowned by estimator noise."""
    n = VIBRATION_FEATURES.min_kurtosis_samples
    sigma_kurt = (24.0 / n) ** 0.5
    assert sigma_kurt <= 0.9, (
        f"σ_kurt = {sigma_kurt:.2f} > 0.9 at N={n}; chapter 3 §3.4.3 invariant broken"
    )


def test_crest_factor_floor_is_finite_at_minimum() -> None:
    """Crest factor (peak / RMS) needs ≥ 4 samples for a stable RMS."""
    assert VIBRATION_FEATURES.min_crest_factor_samples >= 4


# ---------------------------------------------------------------------------
# 5 · Encoder
# ---------------------------------------------------------------------------


def test_encoder_pool_default_is_asp() -> None:
    """ASP is the publication default after the 2026-05-19 architectural audit.

    Switching back to `avg` recreates the V3-at-hop=43 dilution regression."""
    assert ENCODER.pool_type == "asp"


def test_encoder_pool_reduction_has_headroom_over_discriminative_floor() -> None:
    """Pool MLP bottleneck = C / r must be ≥ 2× the minimum discriminative
    dimensionality (5 attention modes = 4 ROW II tones + 1 transient).
    Chapter 4 §A.1.4."""
    bottleneck = ENCODER.cnn_c3 // ENCODER.pool_reduction
    min_discriminative_modes = 5
    assert bottleneck >= 2 * min_discriminative_modes, (
        f"pool bottleneck {bottleneck} below 2× discriminative floor "
        f"({2 * min_discriminative_modes}); r={ENCODER.pool_reduction} too aggressive"
    )


def test_v2_context_pool_seeds_strictly_greater_than_one() -> None:
    """V2's joint-context PMA uses ≥ 2 seeds (Set Transformer §3.2)."""
    assert ENCODER.num_context_seeds >= 2


# ---------------------------------------------------------------------------
# 6 · Windowing
# ---------------------------------------------------------------------------


def test_per_dataset_scales_respect_vibration_kernel_constraint() -> None:
    """Every per-dataset scale must give ≥ 5 vibration samples per window
    (one full Vibration1DCNN kernel of length 5).  Now sourced from the
    dataset registry (configs/datasets/d*.yaml)."""
    for meta in REGISTRY:
        if meta.id == "illwerke_raw":
            continue  # stub config; data dir not present
        for scale_s in meta.window_scales_seconds:
            n_vib = scale_s * meta.accel_target_sr
            assert n_vib >= 5, (
                f"dataset {meta.id} scale {scale_s} s gives {n_vib} vib samples "
                f"< Vibration1DCNN kernel = 5"
            )


def test_v3_window_override_smaller_than_v2_default() -> None:
    """V3 transient-tightness invariant: per-dataset override ≤ V2 default
    (2.0 s) on the high-rate datasets (D3, D4, D5)."""
    for ds in ("d3", "d4", "d5"):
        meta = REGISTRY.get(ds)
        assert meta.v3_window_seconds <= WINDOWING.window_seconds


def test_v4_window_override_smaller_than_or_equal_v3() -> None:
    """V4's SRP-PHAT integration window is ≤ V3's anomaly window
    (V4 needs tighter spatial signature on transient events)."""
    for ds in ("d3", "d4", "d5"):
        meta = REGISTRY.get(ds)
        assert meta.v4_window_seconds <= meta.v3_window_seconds, (
            f"dataset {ds}: V4 window {meta.v4_window_seconds} > V3 window "
            f"{meta.v3_window_seconds} — V4 should be ≤ V3 for tighter "
            f"spatial signature"
        )


# ---------------------------------------------------------------------------
# 7 · V3 / V4 architectural choices
# ---------------------------------------------------------------------------


def test_v3_xt_pool_is_pma2() -> None:
    """PMA-2 is the publication default for V3's xt pool (chapter 4 §A.3).
    Reverting to ``mean`` recreates the channel-token dilution stage."""
    assert V3_ANOMALY.xt_pool == "pma2"


def test_v3_threshold_clusters_match_mode_count() -> None:
    """K = 3 matches the operating-mode hypothesis (Pump/Standstill/Turbine).
    Larger K splits modes into noise sub-clusters; smaller K conflates them."""
    assert V3_ANOMALY.n_threshold_clusters == 3


def test_v4_residual_scale_covers_prototype_bounding_box() -> None:
    """The residual head's tanh range must reach any voxel in the 10 cm
    prototype bounding box (≥ 0.10 m)."""
    assert V4_LOCALIZATION.residual_scale_m >= 0.10


# ---------------------------------------------------------------------------
# 8 · Sync gating
# ---------------------------------------------------------------------------


def test_sync_max_offset_below_one_second() -> None:
    """Sub-second cross-correlation search window — wider than this would
    catch second-scale spurious peaks."""
    assert SYNC.max_offset_s <= 1.0


def test_sync_confidence_floor_is_strictly_above_one() -> None:
    """confidence_floor = 1 means peak == second-largest; the gate must be
    strictly above that to be informative."""
    assert SYNC.confidence_floor > 1.0
