"""V4 localization front-end — channel-agnostic feature extraction.

Two outputs per anomaly window, both channel-agnostic so the same trained head
runs on D1 (4 mic + 4 vib), D2 (5+5), D3 (9+4), and any future Illwerke array:

  - **`compute_srp_phat_volume`** — SRP-PHAT 3-D power volume on a fixed
    candidate grid in metres.  Volume shape `(Nx, Ny, Nz)` is determined by
    the grid, not by mic count, so the downstream Cross3D CNN sees a fixed
    input regardless of array size.

  - **`compute_accel_tdoa_tokens`** — per-pair structure-borne TDOA features
    (TDOA-in-seconds, two endpoint positions, pair distance) for accelerometer
    pairs.  Shape `(n_pairs, 8)` with `n_pairs = N(N-1)/2`.  Variable
    `n_pairs` is consumed by a Set-Transformer pool in the V4 head.

Both functions operate on raw waveforms (NumPy), not on V1/V2 features.
Speed of sound: 343 m/s (acoustic), ~ 2000 m/s (3D-printed plastic
casing — the rig is plastic, not steel; corrected 2026-05-16).  A
prior `C_STEEL_MS = 5100` constant was removed (2026-05-20) — it was
physically wrong for this rig and a back-compat hazard.  Chapter 6
reports a wave-speed sensitivity sweep (1500-2500 m/s) because plastic
speed varies with infill, layer adhesion, and surface- vs bulk-mode
coupling — it is a measurable, not a fixed constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...config.constants import C_PLASTIC_3DP_MS
from .classical import gcc_phat, srp_phat_3d

C_AIR_MS = 343.0
# Structure-borne wave speed for the 3D-printed plastic rig casing.  Single
# source of truth is `src/config/constants.C_PLASTIC_3DP_MS`; re-exported here
# (and from `multilateration`) for backward-compatible imports.  See module
# docstring for the correction history.


@dataclass(frozen=True)
class GridSpec:
    """Fixed 3-D candidate grid for SRP-PHAT (metres)."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    n: tuple[int, int, int]  # (Nx, Ny, Nz)

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.linspace(self.lo[0], self.hi[0], self.n[0], dtype=np.float64),
            np.linspace(self.lo[1], self.hi[1], self.n[1], dtype=np.float64),
            np.linspace(self.lo[2], self.hi[2], self.n[2], dtype=np.float64),
        )

    def grid_to_continuous(self, idx: np.ndarray) -> np.ndarray:
        """Convert (Nx, Ny, Nz) integer indices to (x, y, z) metres."""
        ax_x, ax_y, ax_z = self.axes()
        return np.stack([ax_x[idx[..., 0]], ax_y[idx[..., 1]], ax_z[idx[..., 2]]], axis=-1)


# Canonical V4 SRP-PHAT candidate volume: a 32×32×16 grid (~2 cm spacing)
# covering the union of the D2/D3/D4/D5 rig footprints in metres. Used by the
# orchestrator, every cross-validation driver, and the V4 sweep scripts — keep
# it as the single source so a precompute grid never silently drifts from the
# grid a downstream `soft_argmax` expects.
V4_CANDIDATE_GRID = GridSpec(lo=(-0.22, -0.22, -0.02), hi=(0.40, 0.42, 0.30), n=(32, 32, 16))


def _all_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def compute_srp_phat_volume(
    mic_data: np.ndarray,
    mic_xyz: np.ndarray,
    fs: float,
    grid: GridSpec,
    *,
    c: float = C_AIR_MS,
    max_delay_seconds: float | None = None,
    gcc_oversample: int = 1,
    phat_beta: float = 1.0,
    linear_corr: bool = False,
) -> np.ndarray:
    """Compute the SRP-PHAT 3-D power volume on a fixed grid.

    Args:
      mic_data:  (n_mics, T) waveform per mic.
      mic_xyz:   (n_mics, 3) positions in metres.
      fs:        sample rate (Hz).
      grid:      `GridSpec` defining the candidate volume.
      c:         speed of sound in air (m/s).
      max_delay_seconds: half-width of the GCC-PHAT vector.  Defaults to the
         max TDOA implied by the array's spatial extent + a 1.2× safety margin.
      gcc_oversample: integer GCC up-sampling factor (default 1 = unchanged).
         ``O > 1`` resolves the steered-delay to ``1/O`` of an input sample
         (see :func:`classical.gcc_phat`); the steered indexing uses ``fs * O``
         so the peak sharpens below the ~2 cm voxel grid.

    Returns:
      float32 array of shape `(Nx, Ny, Nz)`, normalised to its peak so the
      downstream CNN doesn't have to learn an absolute scale that varies with
      mic count.  The volume is a sum over `N(N-1)/2` pair contributions.
    """
    if mic_data.ndim != 2:
        raise ValueError(f"mic_data must be (n_mics, T); got {mic_data.shape}")
    n_mics = mic_data.shape[0]
    if n_mics < 2:
        raise ValueError("SRP-PHAT requires ≥2 mics")
    if mic_xyz.shape != (n_mics, 3):
        raise ValueError(f"mic_xyz {mic_xyz.shape} must be ({n_mics}, 3)")

    if max_delay_seconds is None:
        # Furthest pair distance / c, with a small margin.
        diffs = mic_xyz[:, None, :] - mic_xyz[None, :, :]
        max_dist = float(np.linalg.norm(diffs, axis=-1).max())
        max_delay_seconds = (max_dist / c) * 1.2

    oversample = max(1, int(gcc_oversample))
    max_delay_samples = max(1, int(round(max_delay_seconds * fs)))
    pairs = _all_pairs(n_mics)
    n_pairs = len(pairs)

    L = 2 * max_delay_samples * oversample + 1
    gcc_stack = np.zeros((n_pairs, L), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        gcc_stack[k] = gcc_phat(
            mic_data[i].astype(np.float64),
            mic_data[j].astype(np.float64),
            max_delay_samples=max_delay_samples,
            oversample=oversample,
            phat_beta=phat_beta,
            linear=linear_corr,
        )

    grid_x, grid_y, grid_z = grid.axes()
    vol = srp_phat_3d(
        gcc_stack,
        mic_xyz.astype(np.float64),
        grid_x,
        grid_y,
        grid_z,
        fs=float(fs) * oversample,  # finer lag grid → scale the steered fs
        c=float(c),
        mic_pairs=pairs,
    )
    peak = float(vol.max())
    if peak > 1e-12:
        vol = vol / peak
    return vol.astype(np.float32)


def bandpass_filter(
    data: np.ndarray, fs: float, lo_hz: float, hi_hz: float, *, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth band-pass over the last axis (per channel).

    Mirrors the pysoundlocalization preprocessing filters (scipy Butterworth),
    but uses ``filtfilt`` for zero phase so the inter-channel TDOA the SRP peak
    depends on is preserved exactly.  Cutoffs are clamped to ``(0, Nyquist)``;
    a degenerate band returns the input unchanged.  Isolating the knock's
    informative band (dropping sub-100 Hz rig rumble and the high-frequency
    hiss that PHAT over-weights) sharpens the GCC-PHAT peak before SRP.
    """
    from scipy.signal import butter, filtfilt

    data = np.asarray(data, dtype=np.float64)
    nyq = 0.5 * float(fs)
    lo = max(1e-3, float(lo_hz)) / nyq
    hi = min(float(hi_hz), nyq * 0.999) / nyq
    if not (0.0 < lo < hi < 1.0):
        return data
    b, a = butter(order, [lo, hi], btype="band")  # type: ignore[misc]  # scipy ba-output stub
    # filtfilt needs length > 3*max(len(a),len(b)); fall back if the crop is short.
    if data.shape[-1] <= 3 * max(len(a), len(b)):
        return data
    return filtfilt(b, a, data, axis=-1)


def srp_peak_sharpness(volume: np.ndarray) -> float:
    """Peak-to-average ratio of a (peak-normalised) SRP volume — a knock-quality
    proxy.

    A sharp, well-localised SRP peak sits far above the surrounding floor (high
    ratio); a diffuse / reverberation-smeared volume has its mass spread out
    (ratio → 1).  Used to down-weight low-confidence knocks when aggregating
    per-knock predictions into one event estimate.  Cheap — reuses the volume
    the front-end already computed.
    """
    vol = np.asarray(volume, dtype=np.float64)
    peak = float(vol.max())
    mean = float(vol.mean())
    if mean <= 1e-9:
        return 1.0
    return peak / mean


def find_burst_window(
    mic_data: np.ndarray,
    fs: float,
    *,
    burst_seconds: float = 0.10,
) -> tuple[int, int]:
    """Locate the highest-energy ``burst_seconds`` sub-window across all mics.

    Detection criterion: the Hilbert-envelope summed over mics, smoothed by
    a moving average of length ``burst_seconds * fs``, has its argmax at
    the centre of the most-energetic sub-window.  Returns
    ``(start_idx, end_idx)`` half-open slice indices into the input array.

    Designed for impact-event localization on long windows that contain a
    short transient (≪ window length) — the canonical D4 RandomFault case
    where a knock occupies ~ 50 ms inside a 2 s analysis window.  Computing
    SRP-PHAT on the burst crop instead of the full window concentrates the
    GCC-PHAT cross-correlation on samples that actually carry source
    information, sharpening the SRP peak.
    """
    if mic_data.ndim != 2 or mic_data.shape[1] < 2:
        raise ValueError(
            f"mic_data must be (n_mics, T) with T ≥ 2; got {mic_data.shape}"
        )
    from scipy.signal import hilbert

    n_burst = max(2, int(round(burst_seconds * fs)))
    T = int(mic_data.shape[1])
    if n_burst >= T:
        return 0, T

    # Cheap envelope: mean |hilbert| across mics, smoothed by uniform filter.
    env = np.abs(hilbert(mic_data.astype(np.float64), axis=-1)).mean(axis=0)
    # Cumulative-sum smoothing for an O(T) moving average.
    csum = np.concatenate([[0.0], np.cumsum(env)])
    win = csum[n_burst:] - csum[:-n_burst]  # length T - n_burst + 1
    peak = int(np.argmax(win))
    return peak, peak + n_burst


def compute_burst_aware_srp_phat_volume(
    mic_data: np.ndarray,
    mic_xyz: np.ndarray,
    fs: float,
    grid: GridSpec,
    *,
    c: float = C_AIR_MS,
    burst_seconds: float = 0.10,
    max_delay_seconds: float | None = None,
) -> np.ndarray:
    """SRP-PHAT computed on the highest-energy sub-window of the input.

    See :func:`find_burst_window` for the sub-window selection rule and
    :func:`compute_srp_phat_volume` for the SRP-PHAT computation itself.
    Falls back to the full window when ``burst_seconds * fs ≥ T``.

    Physical motivation: classical SRP-PHAT averages GCC-PHAT
    cross-correlations over the full analysis window.  When a window's
    source-coherent signal occupies only a fraction of the window (e.g.
    a 50 ms knock in a 2 s window), the remaining ~ 97 % of the window
    contributes incoherent noise to every pair's GCC, smearing the SRP
    peak.  Cropping to the burst removes that noise floor; the SRP peak
    sharpens and the volume becomes informative.  This is the standard
    pre-processing in the impact-source-localization literature.

    For continuously-anomalous recordings (D2 RandomFault, D3 hit) the
    detected burst window is approximately the most-energetic 100 ms of
    the segment, which is typically as informative as the full window
    and never worse.
    """
    s, e = find_burst_window(mic_data, fs, burst_seconds=burst_seconds)
    crop = mic_data[:, s:e]
    return compute_srp_phat_volume(
        crop, mic_xyz, fs, grid,
        c=c, max_delay_seconds=max_delay_seconds,
    )


def _parabolic_delta(y: np.ndarray, p: int) -> float:
    """Sub-sample peak offset by parabolic interpolation around index ``p``.

    Returns ``delta`` in ``[-0.5, 0.5]`` to add to ``p``; 0 at the array edges
    (where the three-point fit is undefined).  Jacovitti & Scarano (1993).
    """
    if p <= 0 or p >= len(y) - 1:
        return 0.0
    a, b, cc = float(y[p - 1]), float(y[p]), float(y[p + 1])
    denom = a - 2.0 * b + cc
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (a - cc) / denom, -0.5, 0.5))


def compute_accel_tdoa_tokens(
    accel_data: np.ndarray,
    accel_xyz: np.ndarray,
    fs: float,
    *,
    c: float = C_PLASTIC_3DP_MS,
    max_delay_seconds: float | None = None,
    gcc_oversample: int = 1,
) -> np.ndarray:
    """Per-pair structure-borne TDOA features.

    For each accelerometer pair (i, j) compute the GCC-PHAT peak lag (in
    seconds), convert it to a **path-difference in metres** via the assumed
    structure-borne wave speed, and emit an 8-D feature token
    `[path_diff_m, pos_i (3), pos_j (3), distance_m]`.  Returns shape
    `(n_pairs, 8)` where `n_pairs = N(N-1)/2`.

    Why path-difference and not seconds: positions are in metres
    (~10⁻¹), pair distances are in metres (~10⁻¹), and a TDOA in seconds
    on a 10 cm steel prototype is ~10⁻⁵.  Mixing magnitudes by 10⁴ in a
    single token forces the per-pair MLP to learn a giant input rescaling
    before any signal can flow.  Multiplying by `c_steel` puts every
    feature in metres on the same order, so the MLP starts learning
    geometry from the first epoch.

    Variable `n_pairs` is the channel-agnostic interface for the V4 head's
    Set-Transformer TDOA pool.

    ``gcc_oversample`` (+ parabolic sub-sample refinement, always on) is the fix
    for the low accel sample rate: at 376 Hz one integer lag is ~2.7 ms, so a
    plain ``argmax`` quantises ``path_diff_m`` to ``±c/fs`` (~5 m at c=2000) —
    useless on a 0.1 m rig.  Oversampling the GCC ``O×`` then parabolically
    refining the peak recovers a continuous sub-sample TDOA, so the token's
    ``path_diff_m`` column finally carries real structure-borne timing instead
    of a 3-valued noise feature.  Pair with a physically-plausible (lower) ``c``.
    """
    if accel_data.ndim != 2:
        raise ValueError(f"accel_data must be (n_vib, T); got {accel_data.shape}")
    oversample = max(1, int(gcc_oversample))
    n = accel_data.shape[0]
    if n < 2:
        # Single-channel accelerometer — emit an empty token sequence; the
        # head's TDOA encoder must handle n_pairs == 0.
        return np.zeros((0, 8), dtype=np.float32)
    if accel_xyz.shape != (n, 3):
        raise ValueError(f"accel_xyz {accel_xyz.shape} must be ({n}, 3)")

    if max_delay_seconds is None:
        diffs = accel_xyz[:, None, :] - accel_xyz[None, :, :]
        max_dist = float(np.linalg.norm(diffs, axis=-1).max())
        # Generous margin: structure-borne speeds are uncertain in practice.
        max_delay_seconds = max(1.0 / fs, (max_dist / c) * 1.5)

    max_delay_samples = max(1, int(round(max_delay_seconds * fs)))
    centre = max_delay_samples * oversample  # zero-lag index in the (oversampled) GCC
    pairs = _all_pairs(n)
    tokens = np.zeros((len(pairs), 8), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        gcc = gcc_phat(
            accel_data[i].astype(np.float64),
            accel_data[j].astype(np.float64),
            max_delay_samples=max_delay_samples,
            oversample=oversample,
        )
        peak_idx = int(np.argmax(gcc))
        # Sub-sample peak: integer lag + parabolic refinement, in oversampled
        # samples, converted back to seconds via the oversampled rate.
        refined = (peak_idx - centre) + _parabolic_delta(gcc, peak_idx)
        tdoa_s = refined / (float(fs) * oversample)
        path_diff_m = float(tdoa_s * c)  # in metres, same units as positions
        pos_i = accel_xyz[i].astype(np.float32)
        pos_j = accel_xyz[j].astype(np.float32)
        dist = float(np.linalg.norm(pos_i - pos_j))
        tokens[k, 0] = path_diff_m
        tokens[k, 1:4] = pos_i
        tokens[k, 4:7] = pos_j
        tokens[k, 7] = dist
    return tokens


__all__ = [
    "C_AIR_MS",
    "C_PLASTIC_3DP_MS",
    "GridSpec",
    "bandpass_filter",
    "compute_accel_tdoa_tokens",
    "compute_burst_aware_srp_phat_volume",
    "compute_srp_phat_volume",
    "find_burst_window",
    "srp_peak_sharpness",
]
