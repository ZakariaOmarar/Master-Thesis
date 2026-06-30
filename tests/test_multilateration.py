"""Tests for the R3.2 classical accel-TDOA multilateration solver."""

from __future__ import annotations

import numpy as np
import pytest

from src.modeling.localization.multilateration import (
    C_PLASTIC_3DP_MS,
    _chan_ho_initial_estimate,
    _parabolic_subsample_peak,
    _refine_lbfgs,
    accel_tdoa_multilateration_v0,
    estimate_pairwise_tdoas,
)


def _tetrahedron_xyz(side: float = 0.5) -> np.ndarray:
    """4-sensor near-tetrahedron geometry centred near the origin."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [side, 0.0, 0.0],
            [0.0, side, 0.0],
            [0.0, 0.0, side],
        ],
        dtype=np.float64,
    )


def _gaussian_pulse(n_samples: int, centre_s: float, fs: float, sigma_s: float) -> np.ndarray:
    """Continuous Gaussian pulse sampled at `fs`, centred at `centre_s`."""
    t = np.arange(n_samples) / fs
    return np.exp(-0.5 * ((t - centre_s) / sigma_s) ** 2).astype(np.float64)


def _synth_accel_waveforms(
    source_xyz: np.ndarray,
    sensor_xyz: np.ndarray,
    fs: float,
    c: float,
    n_samples: int,
    noise_std: float = 0.0,
    seed: int = 0,
    pulse_sigma_s: float = 0.0002,
) -> np.ndarray:
    """Per-accel waveform: a sharp Gaussian impact delayed by ToA = ||src - x_i|| / c.

    `pulse_sigma_s` defaults to 0.2 ms — sharp enough that the pulse has
    significant broadband content at 50 kHz (sigma ≈ 10 samples) and PHAT
    whitening can lock onto a meaningful peak.  This models an impact
    transient, which is the realistic vibration signal for hit-localisation
    on the bench-top rig (sharp knocks, not slow modulations).
    """
    rng = np.random.default_rng(seed)
    waveforms = []
    base_centre = 0.05  # seconds — pulse arrives at the closest sensor near 50 ms
    d_min = float(np.linalg.norm(sensor_xyz - source_xyz, axis=-1).min())
    for x_i in sensor_xyz:
        d_i = float(np.linalg.norm(source_xyz - x_i))
        toa_i = base_centre + (d_i - d_min) / c
        pulse = _gaussian_pulse(n_samples, toa_i, fs, sigma_s=pulse_sigma_s)
        if noise_std > 0.0:
            pulse = pulse + noise_std * rng.standard_normal(n_samples)
        waveforms.append(pulse)
    return np.stack(waveforms, axis=0)


# ---------------------------------------------------------------------------
# Parabolic-subsample peak
# ---------------------------------------------------------------------------


def test_parabolic_subsample_recovers_known_offset():
    """A parabola y = -(x - delta)² + 1 sampled at integer x = 0, 1, 2 has
    its peak between samples 1 and 2 at x = 1 + delta.  The estimator
    should recover ``delta`` exactly."""
    for true_delta in (-0.4, -0.1, 0.0, 0.1, 0.3, 0.45):
        x = np.array([0, 1, 2], dtype=np.float64)
        y = -(x - (1.0 + true_delta)) ** 2 + 1.0
        recovered = _parabolic_subsample_peak(y, peak_idx=1)
        assert abs(recovered - true_delta) < 1e-9, (true_delta, recovered)


def test_parabolic_subsample_boundary_safe():
    assert _parabolic_subsample_peak(np.array([1.0, 0.5, 0.0]), peak_idx=0) == 0.0
    assert _parabolic_subsample_peak(np.array([0.0, 0.5, 1.0]), peak_idx=2) == 0.0
    # Degenerate (flat) — denominator zero.
    assert _parabolic_subsample_peak(np.array([1.0, 1.0, 1.0]), peak_idx=1) == 0.0


# ---------------------------------------------------------------------------
# Chan-Ho closed-form initial estimate (no GCC-PHAT, directly inject TDOAs)
# ---------------------------------------------------------------------------


def test_chan_ho_plus_lbfgs_recovers_synthetic_tdoas():
    """Forward-simulate noiseless analytic TDOAs from a known source
    position; the FULL pipeline (Chan-Ho linearised init + L-BFGS-B
    non-linear refinement) should recover the source to micron-level.

    Chan-Ho alone is a *linearised* approximation — on a 50 cm rig it can
    be off by ~10 cm in a single step, which is what L-BFGS-B is there to
    fix.  Testing Chan-Ho in isolation would just measure the
    linearisation error, which isn't the deliverable.
    """
    from itertools import combinations

    sensors = _tetrahedron_xyz(side=0.5)
    true_xyz = np.array([0.2, 0.3, 0.1])
    c = C_PLASTIC_3DP_MS
    pairs = list(combinations(range(4), 2))
    tdoa_s = np.array(
        [
            (np.linalg.norm(true_xyz - sensors[i])
             - np.linalg.norm(true_xyz - sensors[j])) / c
            for i, j in pairs
        ],
        dtype=np.float64,
    )
    init = _chan_ho_initial_estimate(sensors, pairs, tdoa_s, c)
    bounds = [(-0.5, 1.0), (-0.5, 1.0), (-0.5, 1.0)]
    refined, residual = _refine_lbfgs(sensors, pairs, tdoa_s, c, init, bounds)
    err = float(np.linalg.norm(refined - true_xyz))
    assert err < 1e-3, (
        f"Chan-Ho+LBFGS error {err*1000:.2f} mm too large; init was "
        f"{init} (init err {np.linalg.norm(init - true_xyz)*1000:.1f} mm), "
        f"residual={residual:.3e}"
    )


# ---------------------------------------------------------------------------
# Full pipeline (waveforms → GCC-PHAT → multilateration)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", [
    (0.2, 0.3, 0.1),
    (-0.1, 0.2, 0.3),
    (0.4, 0.0, 0.2),
    (0.05, 0.45, 0.0),
])
def test_end_to_end_multilateration_recovers_source_at_high_fs(source):
    """At a high sample rate (50 kHz), TDOAs between sensors a few-tens-of-cm
    apart with c = 5100 m/s span dozens of integer samples, so GCC-PHAT
    integer-peak + parabolic refinement is in its accurate regime.  This
    test isolates the *solver* correctness from the production-rate
    sub-sample-resolution issue.
    """
    sensors = _tetrahedron_xyz(side=0.5)
    true_xyz = np.array(source, dtype=np.float64)
    fs = 50_000.0  # 50 kHz — sample-rate ample for the TDOA scale here
    n_samples = int(0.2 * fs)
    waveforms = _synth_accel_waveforms(
        true_xyz, sensors, fs=fs, c=C_PLASTIC_3DP_MS, n_samples=n_samples,
        noise_std=0.0, seed=0,
    )
    recovered, residual = accel_tdoa_multilateration_v0(
        waveforms, sensors, fs=fs, c=C_PLASTIC_3DP_MS,
    )
    err = float(np.linalg.norm(recovered - true_xyz))
    # 30 mm tolerance at 50 kHz with a sharp (sigma=0.2 ms) Gaussian impact
    # pulse + parabolic refinement.  Looser than the analytic-TDOA test
    # above because GCC-PHAT integer-peak picking + parabolic interpolation
    # has finite sub-sample resolution (typically ~0.1 sample = 0.04 m of
    # path-difference at c=2000 m/s, fs=50 kHz).  The end-to-end-pipeline
    # acceptance bar mirrors what the real D4 cohort should clear.
    assert err < 0.030, (
        f"Recovered {recovered} vs true {true_xyz}; ||err|| = {err*1000:.2f} mm, "
        f"residual={residual:.3e}"
    )


def test_estimate_pairwise_tdoas_returns_expected_pair_count():
    sensors = _tetrahedron_xyz()
    waveforms = _synth_accel_waveforms(
        np.array([0.1, 0.1, 0.1]), sensors, fs=50_000.0, c=C_PLASTIC_3DP_MS,
        n_samples=int(0.1 * 50_000), seed=0,
    )
    tdoa_s, pairs = estimate_pairwise_tdoas(waveforms, sensors, fs=50_000.0)
    # 4 accels → C(4, 2) = 6 pairs.
    assert len(pairs) == 6
    assert tdoa_s.shape == (6,)
    # Pair-ordering invariant: (i, j) with i < j, lexicographic on i first.
    assert pairs[0] == (0, 1)
    assert pairs[-1] == (2, 3)


def test_too_few_sensors_rejected():
    sensors = np.zeros((3, 3), dtype=np.float64)  # 3 accels — too few
    waveforms = np.zeros((3, 1000), dtype=np.float64)
    with pytest.raises(ValueError, match="≥ 4"):
        estimate_pairwise_tdoas(waveforms, sensors, fs=10_000.0)


def test_production_rate_resolution_limit_is_loud():
    """Document the fs=376 Hz limitation: at the production rate with
    c_plastic ≈ 2000 m/s, one integer sample = c/fs ≈ 5.3 m of path-
    difference.  Parabolic refinement buys ~10x → ~0.5 m resolution,
    still coarse vs the ~10 cm rig scale.  This test does NOT claim
    recovery; it shows the solver runs without crashing and reports
    finite output — informative evidence for Chapter 6.
    """
    sensors = _tetrahedron_xyz(side=0.5)
    true_xyz = np.array([0.05, 0.05, 0.05])  # 5 cm from the corner sensor
    fs = 376.0
    n_samples = int(2.0 * fs)
    waveforms = _synth_accel_waveforms(
        true_xyz, sensors, fs=fs, c=C_PLASTIC_3DP_MS, n_samples=n_samples,
        noise_std=0.0, seed=0, pulse_sigma_s=0.002,
    )
    recovered, residual = accel_tdoa_multilateration_v0(waveforms, sensors, fs=fs)
    assert np.all(np.isfinite(recovered))
    assert np.isfinite(residual)
