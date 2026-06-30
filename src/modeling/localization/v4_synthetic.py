"""Synthetic-knock generation for pretraining the V4 localization geometry.

Operationalises the forward acoustic model of Haller & Länzlinger's
``pysoundlocalization`` (the predecessor TDOA library the related-work chapter
cites): for a source at ``p`` and a sensor at ``m``, the signal arrives delayed
by ``||p - m|| / c`` and is otherwise the same waveform (their eqs 26–31).  We
place one coherent impulsive "knock" per source position at each sensor's
geometric delay, add per-sensor noise, then push the result through the **same**
SRP-PHAT / accel-TDOA front-end the real pipeline uses.

Why this helps RQ3
------------------
The supervised cohort has only ~16 real positions, so the 3-D CNN never sees
what an SRP volume looks like for a source *between* them.  pysoundlocalization
showed classical GCC-PHAT localizes a single clean source to ~0.009 m — i.e. a
synthetic SRP volume is an almost-perfect geometric label.  Generating knocks on
a dense grid of positions (using the *real* array geometry) gives the head a
strong spatial prior to pretrain on; the real knocks then only have to fine-tune
it.  The synthetic samples carry a zero context vector, so they train the
acoustic/vibration geometry under the A3-invariant identity FiLM and leave the
context conditioning to the real fine-tuning stage.

Everything here is offline data synthesis — no learned components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .array_geometry import array_sensor_xyz, classify_position
from .v4_features import (
    C_AIR_MS,
    C_PLASTIC_3DP_MS,
    GridSpec,
    compute_accel_tdoa_tokens,
    compute_srp_phat_volume,
    srp_peak_sharpness,
)
from .v4_trainer import V4Sample


@dataclass(frozen=True)
class SyntheticArraySpec:
    """One real array's geometry + sample rates, reused for synthesis."""

    dataset_id: str
    mic_xyz: np.ndarray  # (n_mics, 3) metres
    vib_xyz: np.ndarray  # (n_vib, 3) metres
    mic_fs: int
    accel_fs: int


def _place_coherent_impulse(
    sensor_xyz: np.ndarray,
    source_xyz: np.ndarray,
    fs: int,
    c: float,
    *,
    n_samples: int,
    source_sig: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    n_reflections: int = 0,
    reflection_gain: float = 0.5,
    reflection_max_delay_s: float = 0.004,
) -> np.ndarray:
    """Build ``(n_sensors, n_samples)`` with `source_sig` delayed per sensor.

    Sensor ``m`` receives ``source_sig`` starting at ``base + round(||p-m||/c *
    fs)`` (a 1/distance amplitude falloff), plus independent Gaussian noise at
    the requested SNR.  This is the discrete form of pysoundlocalization's
    per-microphone construction (delay = distance / speed of sound).

    ``n_reflections > 0`` adds that many attenuated, randomly-delayed echoes of
    the direct arrival per sensor — a lightweight image-source proxy for the
    rig's reverberation, which smears the GCC-PHAT peak the way real recordings
    do.  Free-field synthetic SRP volumes (no echoes) are unrealistically clean;
    adding multipath narrows the synthetic-to-real domain gap for pretraining.
    """
    n_sensors = sensor_xyz.shape[0]
    out = np.zeros((n_sensors, n_samples), dtype=np.float64)
    burst = source_sig.shape[0]
    dists = np.linalg.norm(sensor_xyz - source_xyz[None, :], axis=1)
    delays = np.round(dists / c * fs).astype(int)
    base = n_samples // 2 - burst // 2 - int(delays.max())
    base = max(0, base)
    max_refl_d = max(1, int(round(reflection_max_delay_s * fs)))
    sig_rms = float(np.sqrt(np.mean(source_sig**2)) + 1e-12)
    noise_std = sig_rms / (10.0 ** (snr_db / 20.0))
    for m in range(n_sensors):
        amp = 1.0 / max(dists[m], 1e-3)
        # Direct path + per-sensor reflections (independent extra delays).
        arrivals = [(int(delays[m]), amp)]
        for k in range(n_reflections):
            extra = int(rng.integers(1, max_refl_d + 1))
            g = amp * (reflection_gain ** (k + 1)) * float(rng.choice([-1.0, 1.0]))
            arrivals.append((int(delays[m]) + extra, g))
        for d, g in arrivals:
            start = base + d
            end = start + burst
            if 0 <= start and end <= n_samples:
                out[m, start:end] += g * source_sig
    out += noise_std * rng.standard_normal(out.shape)
    return out


def _sample_positions_in_hull(
    sensor_xyz: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    margin_m: float,
    max_tries_mult: int = 20,
) -> list[np.ndarray]:
    """Rejection-sample `n` positions inside the array footprint (+margin)."""
    lo = sensor_xyz.min(axis=0) - margin_m
    hi = sensor_xyz.max(axis=0) + margin_m
    out: list[np.ndarray] = []
    tries = 0
    while len(out) < n and tries < n * max_tries_mult:
        tries += 1
        p = rng.uniform(lo, hi)
        if classify_position(p, sensor_xyz, margin_m=margin_m).inside:
            out.append(p.astype(np.float64))
    return out


def generate_synthetic_knock_samples(
    arrays: list[SyntheticArraySpec],
    grid: GridSpec,
    *,
    c_dim: int,
    n_positions_per_array: int = 80,
    n_repeats: int = 1,
    crop_seconds: float = 0.12,
    burst_seconds: float = 0.004,
    snr_db: float = 10.0,
    c_air: float = C_AIR_MS,
    c_plastic: float = C_PLASTIC_3DP_MS,
    gcc_oversample: int = 1,
    n_reflections: int = 0,
    reflection_gain: float = 0.5,
    hull_margin_m: float = 0.03,
    seed: int = 0,
) -> list[V4Sample]:
    """Synthesize a pretraining cohort of single-source knocks.

    For each array geometry, rejection-samples `n_positions_per_array` source
    positions inside its footprint, synthesizes the mic + accel signals, and
    runs the real SRP/TDOA front-end to emit `V4Sample`s with a zero context
    vector (`c_dim`).  `target_xyz` is the true source position.

    The samples are tagged `dataset_id="synthetic"` and unique recording ids so
    a position-grouped split treats each synthetic position as its own group.
    """
    rng = np.random.default_rng(seed)
    samples: list[V4Sample] = []
    for spec in arrays:
        sensors = array_sensor_xyz(spec.mic_xyz, spec.vib_xyz)
        positions = _sample_positions_in_hull(
            sensors, n_positions_per_array, rng, margin_m=hull_margin_m
        )
        n_mic = max(8, int(round(crop_seconds * spec.mic_fs)))
        n_acc = max(2, int(round(crop_seconds * spec.accel_fs)))
        burst_mic = max(2, int(round(burst_seconds * spec.mic_fs)))
        for pi, pos in enumerate(positions):
            for r in range(n_repeats):
                src = rng.standard_normal(burst_mic)
                mic = _place_coherent_impulse(
                    spec.mic_xyz, pos, spec.mic_fs, c_air,
                    n_samples=n_mic, source_sig=src, snr_db=snr_db, rng=rng,
                    n_reflections=n_reflections, reflection_gain=reflection_gain,
                )
                # Accel burst spans a few low-rate samples; reuse a short kernel.
                burst_acc = max(2, int(round(burst_seconds * spec.accel_fs)) + 1)
                src_a = rng.standard_normal(burst_acc)
                acc = _place_coherent_impulse(
                    spec.vib_xyz, pos, spec.accel_fs, c_plastic,
                    n_samples=n_acc, source_sig=src_a, snr_db=snr_db, rng=rng,
                    n_reflections=n_reflections, reflection_gain=reflection_gain,
                )
                vol = compute_srp_phat_volume(
                    mic, spec.mic_xyz, fs=spec.mic_fs, grid=grid,
                    gcc_oversample=gcc_oversample,
                )
                tdoa = compute_accel_tdoa_tokens(acc, spec.vib_xyz, fs=spec.accel_fs)
                samples.append(
                    V4Sample(
                        srp_volume=vol,
                        tdoa_tokens=tdoa,
                        context=np.zeros(c_dim, dtype=np.float32),
                        x_for_v3=np.zeros(c_dim, dtype=np.float32),
                        target_xyz=pos.astype(np.float32),
                        scada=None,
                        mode_label=None,
                        recording_id=f"syn_{spec.dataset_id}_{pi:04d}_{r}",
                        source_dir="synthetic",
                        dataset_id="synthetic",
                        multilat_xyz=None,
                        window_start_s=0.0,
                        srp_psr=float(srp_peak_sharpness(vol)),
                    )
                )
    return samples


__all__ = [
    "SyntheticArraySpec",
    "generate_synthetic_knock_samples",
]
