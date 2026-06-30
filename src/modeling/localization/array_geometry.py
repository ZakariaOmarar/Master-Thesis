"""Sensor-array footprint geometry — the RQ3 "is this position localizable?" test.

The RQ3 per-position error map (Chapter 6, `fig25_per_position_error_map`)
shows the worst-localized knock positions all lie **on or outside the convex
hull of the combined sensor array** (microphones + accelerometers).  A source
outside the array aperture is geometrically ill-posed for TDOA / SRP-PHAT:
every pair's hyperboloid grazes the source at a shallow angle, so a millimetre
of timing error maps to centimetres of position error.  No amount of head
capacity fixes that — it is an array-design limit, not a model limit.

This module makes that boundary *computable* so the localization evaluation can:

  * report the headline MAE on the in-footprint positions the array can
    actually resolve, and
  * list the out-of-footprint positions separately as geometric outliers
    (the disposition the thesis already argues for in prose).

`classify_position` returns a signed distance to the hull boundary (negative =
inside, positive = outside) plus an `inside` flag with a configurable margin, so
"just outside by 1 cm" and "20 cm beyond the array" are distinguishable.

Degenerate arrays (all sensors coplanar — e.g. a single-elevation accel ring)
make a 3-D convex hull ill-defined; we fall back to an axis-aligned bounding
box inflated by the same margin, which is the conservative footprint proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FootprintVerdict:
    """One position's relationship to a sensor-array footprint."""

    inside: bool
    signed_distance_m: float  # < 0 inside the hull, > 0 outside (metres)
    method: str  # "convex_hull" | "bounding_box"


def _hull_equations(points: np.ndarray) -> np.ndarray | None:
    """Return ConvexHull facet equations `[n_facets, dim+1]`, or None if degenerate.

    Each row is `[normal (unit), offset]` with the convention
    `normal · x + offset <= 0` for points inside the hull, so
    `max_f (normal_f · x + offset_f)` is the signed distance to the nearest
    facet — negative inside, positive outside.  Returns None when Qhull cannot
    build a full-dimensional hull (collinear / coplanar sensors), signalling the
    caller to use the bounding-box fallback.
    """
    try:
        from scipy.spatial import ConvexHull, QhullError  # type: ignore
    except Exception:  # pragma: no cover - older scipy namespace
        try:
            from scipy.spatial import ConvexHull
            from scipy.spatial.qhull import QhullError  # type: ignore[attr-defined]
        except Exception:
            return None
    try:
        hull = ConvexHull(points)
    except QhullError:  # type: ignore[misc]  # QhullError import is version-dependent
        return None
    except Exception:
        return None
    return np.asarray(hull.equations, dtype=np.float64)


def array_sensor_xyz(mic_xyz: np.ndarray, vib_xyz: np.ndarray) -> np.ndarray:
    """Stack mic + accel positions into one `(n_sensors, 3)` point cloud.

    Both modalities bound the source jointly: the fused localizer can resolve a
    source the microphones bracket OR the accelerometers bracket, so the
    relevant footprint is the union array's hull, not either modality alone.
    """
    mic_xyz = np.asarray(mic_xyz, dtype=np.float64).reshape(-1, 3)
    vib_xyz = np.asarray(vib_xyz, dtype=np.float64).reshape(-1, 3)
    return np.concatenate([mic_xyz, vib_xyz], axis=0)


def classify_position(
    point: np.ndarray,
    sensor_xyz: np.ndarray,
    *,
    margin_m: float = 0.05,
) -> FootprintVerdict:
    """Classify `point` against the convex hull of `sensor_xyz`.

    `margin_m` widens the admissible region: a point is `inside` when its
    signed distance to the hull boundary is `<= margin_m`.  A 5 cm default
    tolerates the ~2 cm voxel quantisation of the SRP grid plus a knock that
    sits a little proud of the outermost sensor without being a true outlier.
    """
    point = np.asarray(point, dtype=np.float64).reshape(3)
    sensor_xyz = np.asarray(sensor_xyz, dtype=np.float64).reshape(-1, 3)

    eqs = _hull_equations(sensor_xyz)
    if eqs is not None:
        signed = float(np.max(eqs[:, :-1] @ point + eqs[:, -1]))
        return FootprintVerdict(
            inside=signed <= margin_m,
            signed_distance_m=signed,
            method="convex_hull",
        )

    # Degenerate hull → axis-aligned bounding box fallback.  Signed distance is
    # the Chebyshev-style max over per-axis excursions beyond the box (negative
    # when the point sits inside every axis interval).
    lo = sensor_xyz.min(axis=0)
    hi = sensor_xyz.max(axis=0)
    over = np.maximum(lo - point, point - hi)  # >0 on an axis where outside
    signed = float(np.max(over))
    return FootprintVerdict(
        inside=signed <= margin_m,
        signed_distance_m=signed,
        method="bounding_box",
    )


def classify_positions(
    records: list[tuple[tuple[float, float, float], np.ndarray, np.ndarray]],
    *,
    margin_m: float = 0.05,
    ndigits: int = 3,
) -> dict[tuple[float, float, float], dict]:
    """Aggregate footprint verdicts per unique position over many recordings.

    `records` is a list of `(position_xyz_m, mic_xyz, vib_xyz)` — one entry per
    labelled recording (a physical position can be recorded by several arrays
    across campaigns).  A position is reported `inside` when it is in-footprint
    for **any** array that recorded it (the most generous reading: a true
    outlier is one that *no* array brackets), and its signed distance is the
    minimum (closest-to-inside) over those arrays.

    Returns `{position_key: {"inside", "min_signed_distance_m", "n_recordings",
    "method"}}` keyed on the rounded cm grid so it joins directly to
    `v4_trainer._position_key`.
    """
    out: dict[tuple[float, float, float], dict] = {}
    for pos, mic_xyz, vib_xyz in records:
        key = tuple(round(float(v), ndigits) for v in np.asarray(pos).ravel()[:3])
        verdict = classify_position(
            np.asarray(pos), array_sensor_xyz(mic_xyz, vib_xyz), margin_m=margin_m
        )
        cur = out.get(key)
        if cur is None:
            out[key] = {
                "inside": verdict.inside,
                "min_signed_distance_m": verdict.signed_distance_m,
                "n_recordings": 1,
                "method": verdict.method,
            }
        else:
            cur["inside"] = cur["inside"] or verdict.inside
            cur["min_signed_distance_m"] = min(
                cur["min_signed_distance_m"], verdict.signed_distance_m
            )
            cur["n_recordings"] += 1
    return out


__all__ = [
    "FootprintVerdict",
    "array_sensor_xyz",
    "classify_position",
    "classify_positions",
]
