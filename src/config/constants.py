"""Physical constants and sensor layout for the ROW II hydropower sensor array.

The Rodundwerk II (ROW II) facility in Vorarlberg, Austria operates a single
reversible Francis pump-turbine at 375 rpm nominally. The sensor array spans
two elevation levels:

  Level 1 (generator level, z = 3.0 m): four microphones at 0°, 90°, 180°, 270°.
  Level 2 (turbine level, z = 0.0 m):  five microphones at 72° spacing and
                                        four accelerometers at 90° spacing.

The machine rotational frequency is 375 / 60 ≈ 6.25 Hz. The nominal fundamental
used in feature extraction is 5.867 Hz (adjusted for guide vane loading effects);
the vane-pass frequency is ≈ 117.3 Hz.

These constants drive the TDOA source-localization geometry and fix channel
indices throughout the pipeline.
"""

from __future__ import annotations

import math

MIC_SAMPLE_RATE = 16_000
"""Audio sample rate in Hz (mono WAV, 16-bit)."""

MIC_COUNT = 9
"""Total microphone channels: 4 (upper ring) + 5 (lower ring)."""

ACCEL_COUNT = 4
"""Total vibration channels (one per accelerometer position)."""

ACCEL_SAMPLE_RATE_TARGET = 4
"""Target resampled rate for vibration amplitude streams in Hz."""

CASING_RADIUS_M = 1.5
GENERATOR_LEVEL_Z_M = 3.0
TURBINE_LEVEL_Z_M = 0.0

# Structure-borne wave speed for the 3D-printed PLA/ABS bench-top rig, used by
# every accelerometer-TDOA path (multilateration init + V4 TDOA tokens +
# synthetic-knock generator).  Single source of truth so the two consumers
# (``localization.multilateration`` and ``localization.v4_features``) can never
# disagree — a thesis-load-bearing constant: an over-estimate compresses the
# TDOA geometry and suppresses the vibration localization channel (Results §setup).
# The earlier steel value (5100 m/s) was wrong for plastic and inflated the
# path-difference scaling ~2.55×; a ±25 % sensitivity sweep is reported in
# Chapter 6 because plastic wave speed varies with infill / layer adhesion.
C_PLASTIC_3DP_MS = 2000.0

SENSOR_LAYOUT = {
    "microphones": {
        "level_1": [
            {"name": "L1_0", "angle_deg": 0, "channel": 0, "z_m": GENERATOR_LEVEL_Z_M},
            {
                "name": "L1_90",
                "angle_deg": 90,
                "channel": 1,
                "z_m": GENERATOR_LEVEL_Z_M,
            },
            {
                "name": "L1_180",
                "angle_deg": 180,
                "channel": 2,
                "z_m": GENERATOR_LEVEL_Z_M,
            },
            {
                "name": "L1_270",
                "angle_deg": 270,
                "channel": 3,
                "z_m": GENERATOR_LEVEL_Z_M,
            },
        ],
        "level_2": [
            {"name": "L2_0", "angle_deg": 0, "channel": 4, "z_m": TURBINE_LEVEL_Z_M},
            {"name": "L2_72", "angle_deg": 72, "channel": 5, "z_m": TURBINE_LEVEL_Z_M},
            {
                "name": "L2_144",
                "angle_deg": 144,
                "channel": 6,
                "z_m": TURBINE_LEVEL_Z_M,
            },
            {
                "name": "L2_216",
                "angle_deg": 216,
                "channel": 7,
                "z_m": TURBINE_LEVEL_Z_M,
            },
            {
                "name": "L2_288",
                "angle_deg": 288,
                "channel": 8,
                "z_m": TURBINE_LEVEL_Z_M,
            },
        ],
    },
    "accelerometers": [
        {"name": "A_0", "angle_deg": 0, "channel": 0, "z_m": TURBINE_LEVEL_Z_M},
        {"name": "A_90", "angle_deg": 90, "channel": 1, "z_m": TURBINE_LEVEL_Z_M},
        {"name": "A_180", "angle_deg": 180, "channel": 2, "z_m": TURBINE_LEVEL_Z_M},
        {"name": "A_270", "angle_deg": 270, "channel": 3, "z_m": TURBINE_LEVEL_Z_M},
    ],
}


def mic_cartesian_positions() -> list[tuple[float, float, float]]:
    """Return (x, y, z) Cartesian coordinates for all 9 microphones, ordered by channel index.

    Used by the TDOA source-localization component to compute inter-microphone
    propagation delays. Positions are derived from casing radius and polar angle;
    both elevation levels sit on the same radius but at different z heights.
    """
    all_mics = (
        SENSOR_LAYOUT["microphones"]["level_1"]
        + SENSOR_LAYOUT["microphones"]["level_2"]
    )
    all_mics = sorted(all_mics, key=lambda x: x["channel"])

    xyz: list[tuple[float, float, float]] = []
    for m in all_mics:
        theta = math.radians(m["angle_deg"])
        x = CASING_RADIUS_M * math.cos(theta)
        y = CASING_RADIUS_M * math.sin(theta)
        z = m["z_m"]
        xyz.append((x, y, z))
    return xyz
