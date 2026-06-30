"""R3.2 — Classical TDOA multilateration for accelerometer-only localisation.

Provides ``accel_tdoa_multilateration_v0(...)`` — a standalone, non-learned
vibration-only (x, y, z) estimator that mirrors `srp_phat_3d` as the
acoustic-only classical baseline.  Used as:

  * The "V0 accel multilat" row of the RQ3 paradigm comparison table.
  * The spatial init for V4's `channel_mode = "vibration_only_learned"`
    pathway (R3.3), replacing the acoustic SRP-PHAT soft-argmax init.

Pipeline:

  1. **Parabolic sub-sample GCC-PHAT TDOAs** per accelerometer pair.  The
     bench-top rig samples vibration at 376 Hz; the structure-borne wave speed
     for the 3D-printed plastic casing is ~2000 m/s (``C_PLASTIC_3DP_MS``), so
     one integer-sample step corresponds to ~5.3 m of path-difference — useless
     on a ~10 cm rig.  Parabolic interpolation
     around the GCC peak buys ~10× finer effective resolution (Jacovitti &
     Scarano 1993, "Discrete time techniques for time delay estimation",
     IEEE TSP).
  2. **Chan & Ho 1994 closed-form initial estimate** (linearised hyperbolic
     least squares, no iteration) — gives a starting point without
     requiring an acoustic SRP-PHAT prior.  Reference:
     Y. T. Chan & K. C. Ho, "A simple and efficient estimator for
     hyperbolic location", IEEE TSP 42(8), 1905-1915, 1994.
  3. **L-BFGS-B non-linear refinement** of the closed-form initial estimate
     against the hyperbolic TDOA residual (predicted minus measured
     path-difference), with accelerometer-pair geometry and the rig's
     structure-borne wave-speed.

Returns the refined `(x, y, z)` plus a confidence proxy (final residual
sum-of-squares).

Output sign convention: positive ``tdoa_s`` for pair ``(i, j)`` means
**signal arrived at i after j** (j-leads-i), matching `gcc_phat` (peak at
positive lag when j leads i).  Path difference is then
``r_i - r_j = c · tdoa_s`` in metres.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ...config.constants import C_PLASTIC_3DP_MS
from .classical import gcc_phat

# Wave-speed default for the 3D-printed PLA/ABS bench-top rig.  Single source of
# truth is `src/config/constants.C_PLASTIC_3DP_MS` (re-exported here so existing
# `from .multilateration import C_PLASTIC_3DP_MS` call sites keep working).  The
# prior `C_STEEL_MS = 5100` was removed (2026-05-20): the rig is plastic, not
# steel, and the wrong value inflated the path-difference scaling ~2.55×.


def _parabolic_subsample_peak(gcc: np.ndarray, peak_idx: int) -> float:
    """Parabolic interpolation around an integer GCC peak.

    Returns the sub-sample peak offset ``delta`` (in samples, in
    ``[-0.5, +0.5]``) added to ``peak_idx`` for the refined peak position.
    If the peak is at a boundary or the denominator is degenerate, returns
    0.0 (i.e. fall back to the integer peak).

    The fit is the standard 3-point parabolic estimator:

        delta = (y_-1 - y_+1) / (2 * (y_-1 - 2*y_0 + y_+1))

    bounded to ``[-0.5, +0.5]`` (a peak with |delta| > 0.5 means the
    integer peak picker was wrong; we ignore the refinement in that case).
    """
    n = int(gcc.size)
    if peak_idx <= 0 or peak_idx >= n - 1:
        return 0.0
    y_m, y_0, y_p = float(gcc[peak_idx - 1]), float(gcc[peak_idx]), float(gcc[peak_idx + 1])
    denom = 2.0 * (y_m - 2.0 * y_0 + y_p)
    if abs(denom) < 1e-12:
        return 0.0
    delta = (y_m - y_p) / denom
    if delta < -0.5 or delta > 0.5:
        return 0.0
    return float(delta)


def estimate_pairwise_tdoas(
    accel_data: np.ndarray,
    accel_xyz: np.ndarray,
    fs: float,
    *,
    c: float = C_PLASTIC_3DP_MS,
    max_delay_seconds: float | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Estimate parabolic-refined GCC-PHAT TDOAs for every accel pair.

    Returns ``(tdoa_seconds, pairs)`` where ``tdoa_seconds[k]`` is the
    refined TDOA for pair ``pairs[k] = (i, j)`` in seconds (positive ⇒ j
    leads i; signal arrives at i after j).  No (x, y, z) solve here — just
    the per-pair time-delay estimates.
    """
    if accel_data.ndim != 2:
        raise ValueError(f"accel_data must be (n_accel, T); got {accel_data.shape}")
    n = int(accel_data.shape[0])
    if n < 4:
        raise ValueError(
            f"accel_tdoa_multilateration_v0 needs ≥ 4 accelerometers for a "
            f"3-D least-squares solve; got {n}"
        )
    if accel_xyz.shape != (n, 3):
        raise ValueError(f"accel_xyz {accel_xyz.shape} must be ({n}, 3)")

    if max_delay_seconds is None:
        diffs = accel_xyz[:, None, :] - accel_xyz[None, :, :]
        max_dist = float(np.linalg.norm(diffs, axis=-1).max())
        # Generous margin — structure-borne speeds vary in practice.
        max_delay_seconds = max(1.0 / fs, (max_dist / c) * 1.5)
    max_delay_samples = max(2, int(round(max_delay_seconds * fs)))

    pairs = list(combinations(range(n), 2))
    tdoa_s = np.zeros(len(pairs), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        gcc = gcc_phat(
            accel_data[i].astype(np.float64),
            accel_data[j].astype(np.float64),
            max_delay_samples=max_delay_samples,
        )
        peak_idx = int(np.argmax(gcc))
        delta = _parabolic_subsample_peak(gcc, peak_idx)
        refined_lag_samples = (peak_idx - max_delay_samples) + delta
        tdoa_s[k] = refined_lag_samples / float(fs)
    return tdoa_s, pairs


def _chan_ho_initial_estimate(
    sensor_xyz: np.ndarray,
    pairs: list[tuple[int, int]],
    tdoa_s: np.ndarray,
    c: float,
) -> np.ndarray:
    """Chan & Ho 1994 closed-form linearised hyperbolic LS initial estimate.

    Builds the linear system A·θ = b where ``θ = [x, y, z, R_0]`` and
    ``R_0 = ||p - x_0||`` is the source-to-reference-sensor distance.
    The reference sensor is index 0 (which always appears in pairs of the
    form (0, j) under the `combinations` ordering).

    The system is solved via `np.linalg.lstsq`; the leading 3 entries of
    the solution are the (x, y, z) estimate.

    With fewer than 4 sensors (3 unknowns + R_0 = 4 equations needed) the
    system is rank-deficient.  The caller has already enforced n ≥ 4.
    """
    ref_pairs = [(k, (i, j)) for k, (i, j) in enumerate(pairs) if i == 0]
    if len(ref_pairs) < 3:
        # Fall back: use all (i, j) with i < j, rewrite as differences from x_0.
        # Each general TDOA t_{ij} = (R_i - R_j)/c, but Chan's closed form
        # needs reference-sensor pairs.  In our `combinations` ordering the
        # first n-1 pairs are (0, 1), (0, 2), ..., (0, n-1) — guaranteed
        # ≥ 3 when n ≥ 4.  This branch is defensive.
        raise RuntimeError(
            "Chan-Ho initial estimate needs ≥ 3 reference-sensor (0, j) pairs"
        )

    x0 = sensor_xyz[0]
    # Build A · [x, y, z, R_0]^T = b.
    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    for k, (_i, j) in ref_pairs:
        # i == 0, j != 0
        xj = sensor_xyz[j]
        r_j0 = float(c * tdoa_s[k])  # path-diff r_j - r_0 in metres
        # Linearised hyperbolic eqn (Chan-Ho derivation):
        #   2 * (x_j - x_0) · p + 2 * r_j0 * R_0  =  |x_j|^2 - |x_0|^2 - r_j0^2
        A_row = np.zeros(4, dtype=np.float64)
        A_row[:3] = 2.0 * (xj - x0)
        A_row[3] = 2.0 * r_j0
        rhs = float(np.dot(xj, xj) - np.dot(x0, x0) - r_j0 ** 2)
        A_rows.append(A_row)
        b_rows.append(rhs)

    A = np.stack(A_rows, axis=0)
    b = np.asarray(b_rows, dtype=np.float64)
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return theta[:3].astype(np.float64)


def _refine_lbfgs(
    sensor_xyz: np.ndarray,
    pairs: list[tuple[int, int]],
    tdoa_s: np.ndarray,
    c: float,
    init_xyz: np.ndarray,
    bounds: list[tuple[float, float]] | None,
) -> tuple[np.ndarray, float]:
    """L-BFGS-B refinement of the Chan-Ho initial estimate.

    Minimises the hyperbolic TDOA residual but takes pairs + TDOAs directly
    (rather than a `gcc_stack` — we already have sub-sample-refined TDOAs and
    don't want to throw that resolution away by re-running an integer argmax
    inside the refinement).
    """
    from scipy.optimize import minimize  # type: ignore[import-untyped]

    sensor_i = np.stack([sensor_xyz[i] for i, _ in pairs])  # (n_pairs, 3)
    sensor_j = np.stack([sensor_xyz[j] for _, j in pairs])

    def _residuals(pos: np.ndarray) -> float:
        di = np.linalg.norm(sensor_i - pos, axis=-1)
        dj = np.linalg.norm(sensor_j - pos, axis=-1)
        return float(np.sum((tdoa_s - (di - dj) / c) ** 2))

    def _jac(pos: np.ndarray) -> np.ndarray:
        di = np.linalg.norm(sensor_i - pos, axis=-1, keepdims=True) + 1e-12
        dj = np.linalg.norm(sensor_j - pos, axis=-1, keepdims=True) + 1e-12
        theory = ((di - dj) / c).squeeze(-1)
        residuals = tdoa_s - theory
        grad_di = (pos - sensor_i) / (di * c)
        grad_dj = (pos - sensor_j) / (dj * c)
        grad_theory = grad_di - grad_dj
        return (-2.0 * residuals[:, None] * grad_theory).sum(axis=0)

    res = minimize(
        _residuals,
        x0=init_xyz,
        jac=_jac,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10},
    )
    return res.x.astype(np.float64), float(res.fun)


def accel_tdoa_multilateration_v0(
    accel_data: np.ndarray,
    accel_xyz: np.ndarray,
    fs: float,
    *,
    c: float = C_PLASTIC_3DP_MS,
    max_delay_seconds: float | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, float]:
    """Vibration-only classical-localisation V0 baseline.

    Args:
      accel_data:  ``(n_accel, T)`` raw waveforms in arbitrary units (PHAT
                   whitening makes the absolute scale irrelevant).
      accel_xyz:   ``(n_accel, 3)`` accelerometer positions in metres.
      fs:          Accelerometer sample rate in Hz.
      c:           Structure-borne wave speed (m/s).  Default 5100
                   (steel longitudinal).
      max_delay_seconds: Half-width of the GCC-PHAT search window.  Default
                   ``max(1/fs, 1.5 · max_pair_distance / c)``.
      bounds:      Per-axis ``(lo, hi)`` bounds (m) for the L-BFGS-B
                   refinement; defaults to a 40 cm margin around the
                   sensor bounding box.

    Returns:
      ``(xyz_m, residual_sum_sq)`` — refined position in metres + final
      residual.  The residual scales with `(n_pairs · c⁻²)`, so absolute
      comparison across runs needs the same accel count / wave speed.
    """
    tdoa_s, pairs = estimate_pairwise_tdoas(
        accel_data, accel_xyz, fs, c=c, max_delay_seconds=max_delay_seconds,
    )
    if bounds is None:
        lo = accel_xyz.min(axis=0) - 0.40
        hi = accel_xyz.max(axis=0) + 0.40
        bounds = [(float(lo[i]), float(hi[i])) for i in range(3)]

    # Multi-start refinement.  Hyperbolic-intersection systems have
    # mirror-image ambiguities (Chan & Ho 1994 §III); the wrong solution
    # often has a low-but-not-machine-zero residual while the right
    # solution has a near-machine-zero one, so picking the minimum-
    # residual point across multiple starting points is reliable.
    # Start set, in order:
    #   (1) Chan-Ho closed-form init (geometric prior, biased toward
    #       sensor centroid),
    #   (2) sensor centroid (always inside the convex hull),
    #   (3) each sensor position (physical prior: impact-localisation
    #       sources are often near a sensor — and starting near the true
    #       source basin guarantees convergence to it).
    # Total: ~ 2 + n_accel L-BFGS-B calls.  Each call is sub-millisecond.
    def _clip(p: np.ndarray) -> np.ndarray:
        return np.array([
            float(np.clip(p[i], bounds[i][0], bounds[i][1])) for i in range(3)
        ])

    starts: list[np.ndarray] = []
    try:
        starts.append(_clip(_chan_ho_initial_estimate(accel_xyz, pairs, tdoa_s, c)))
    except np.linalg.LinAlgError:
        pass
    starts.append(_clip(accel_xyz.mean(axis=0)))
    for x_i in accel_xyz:
        starts.append(_clip(x_i))

    best_pos: np.ndarray | None = None
    best_residual = float("inf")
    for init in starts:
        pos, residual = _refine_lbfgs(accel_xyz, pairs, tdoa_s, c, init, bounds)
        if residual < best_residual:
            best_pos, best_residual = pos, residual
    assert best_pos is not None  # at least one start always succeeds
    return best_pos, best_residual


__all__ = [
    "C_PLASTIC_3DP_MS",
    "accel_tdoa_multilateration_v0",
    "estimate_pairwise_tdoas",
]
