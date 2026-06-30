"""Neural source localizers — bench-top baselines for the Chapter 6 comparison.

Acoustic (Transformer) and structural (3-D CNN) localization heads with
FiLM-conditioned fusion, developed on the second/third datasets. These are not
part of the live V0-V5 pipeline — which localizes via `v4_loc_head` on top of the
`classical` SRP-PHAT primitives — but are kept to regenerate the `loc_s2_*` /
`loc_s3_*` baseline results that Chapter 6 reports against. The shared GCC-PHAT /
SRP-PHAT primitives live in `classical.py` and are re-imported here.

Public API — acoustic stream:
  LocalizationCNNS2              — Transformer localizer for second dataset (5 mics)
  LocalizationCNNS3              — Transformer localizer for third dataset (9 mics)
  supervised_localization_loss_s2/s3 — supervised MSE loss
  geometric_consistency_loss_s2/s3   — physics-based self-supervised loss
  compute_gcc_stack_s2_multiwindow   — multi-window GCC-PHAT (acoustic, S2)
  compute_gcc_stack_s3_multiwindow   — multi-window GCC-PHAT (acoustic, S3)
  gcc_phat, srp_phat_3d              — re-exported from `classical` (see that module)
  srp_phat_3d_hierarchical           — coarse-to-fine SRP-PHAT grid search
  tdoa_triangulate                   — non-linear LS TDOA refinement

Public API — structural stream (new):
  compute_gcc_stack_structural_multiwindow — GCC-PHAT on accelerometer signals
  structural_srp_phat_3d           — structural SRP-PHAT power map
  synthetic_gcc_stack              — ideal GCC stack from geometry (no audio needed)
  bandpass_iir                     — AE-band Butterworth bandpass filter
  C_STRUCT_MS                      — default steel compressional wave speed

Public API — fusion (new):
  LocalizationDualSRPNet           — Cross3D CNN on 2-channel SRP maps + FiLM context
  dual_srp_localization_loss        — heteroscedastic NLL training loss
  information_fusion               — inverse-covariance (information-form) fusion
  srp_covariance, tdoa_covariance, neural_covariance — per-method covariance estimates

Geometry constants (second dataset, bench-top prototype):
  MIC_PAIRS_S2, N_PAIRS_S2, S2_MIC_XYZ, S2_VIB_XYZ, S2_FAULT_POSITIONS_M
  VIB_PAIRS_S2, N_VIB_PAIRS_S2

Geometry constants (third dataset, bench-top prototype v2):
  MIC_PAIRS_S3, N_PAIRS_S3, S3_MIC_XYZ, S3_VIB_XYZ
  VIB_PAIRS_S3, N_VIB_PAIRS_S3

Geometry constants (ROW II turbine, for thesis reference):
  TURBINE, FREQUENCIES_HZ, SENSOR_XYZ, MIC_UPPER_XY, MIC_ALL_XYZ, ACCEL_XYZ
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "localization_head requires PyTorch. Install with: pip install torch"
    ) from exc

from .classical import gcc_phat, srp_phat_3d

# ---------------------------------------------------------------------------
# Turbine geometry — Rodund 2 | Voith | Drawing 2821-028589-ROW Rev F
# ---------------------------------------------------------------------------
# All positions in metres.  Origin: turbine rotation axis at upper-mic height.
# x-axis: positive toward the entrance opening.
# y-axis: positive 90° CCW from entrance.
# z-axis: positive upward.
# Angle convention: 0° = entrance (+x); CCW positive; CW negative.
#
# BARRIER_RADIUS_M must be confirmed on site (estimated 5.5 m from axis to wall).
# ---------------------------------------------------------------------------

# -- Turbine parameters -------------------------------------------------------
TURBINE: dict = {
    "name": "Francis Turbine — Rodund 2",
    "type": "pump-turbine (reversible Francis)",
    "manufacturer": "Voith",
    "drawing_number": "2821-028589-ROW",
    "drawing_revision": "F",
    "drawing_date": "2019-04-04",
    "facility_id": "27011",
    "rpm": 375,
    "runner_blades": 7,
    "guide_vanes": 20,
    "stator_slots": 288,
    "rotor_poles": 16,
}

# -- Characteristic frequencies (Hz) -----------------------------------------
FREQUENCIES_HZ: dict = {
    "shaft_hz": 6.25,
    "runner_blade_passing_hz": 43.75,
    "guide_vane_passing_hz": 125.0,
    "rotor_pole_passing_hz": 100.0,
    "electrical_network_hz": 50.0,
}

# -- Vertical levels ----------------------------------------------------------
BARRIER_RADIUS_M: float = 5.5  # *** CONFIRM ON SITE ***
Z_UPPER_MIC: float = 0.00  # m — reference level
Z_LOWER_MIC: float = -0.80  # m — 80 cm below upper mics
Z_ACCEL_SENSOR: float = -0.90  # m — 10 cm below lower mics
SPEED_OF_SOUND_MS: float = 343.0  # m/s at ~20 °C

# -- Individual sensor XYZ positions (metres) ---------------------------------
# Computed from arc-distance table on drawing 2821-028589-ROW, R = 5.5 m.
_SENSOR_XYZ_RAW: dict[str, list[float]] = {
    # Upper microphones  (z = Z_UPPER_MIC = 0.00 m)
    "MIC_UP_2": [5.345904528398038, 1.2927895316923597, 0.0],
    "MIC_UP_4": [4.732751574776851, 2.8019747556762953, 0.0],
    "MIC_UP_6": [5.5, 0.0, 0.0],
    "MIC_UP_8": [2.269476735317571, -5.009937659078434, 0.0],
    # Lower (bottom) microphones  (z = Z_LOWER_MIC = -0.80 m)
    "MIC_BOT_1": [5.1039378506168624, 2.0493458510072227, -0.8],
    "MIC_BOT_3": [-1.5397624963450893, 5.2800692661033475, -0.8],
    "MIC_BOT_5": [5.07359032360746, -2.123365542763833, -0.8],
    "MIC_BOT_7": [5.394303917240584, -1.0730728066831692, -0.8],
    "MIC_BOT_9": [2.269476735317571, -5.009937659078434, -0.8],
    # Accelerometers  (z = Z_ACCEL_SENSOR = -0.90 m)
    "ACCEL_1": [5.5, 0.0, -0.9],
    "ACCEL_2": [4.732751574776851, 2.8019747556762953, -0.9],
    "ACCEL_3": [3.2960976627194523, 4.402924050879752, -0.9],
    "ACCEL_4": [2.269476735317571, -5.009937659078434, -0.9],
}

SENSOR_XYZ: dict[str, np.ndarray] = {
    k: np.array(v, dtype=np.float64) for k, v in _SENSOR_XYZ_RAW.items()
}

# -- Convenience arrays -------------------------------------------------------
# Upper mics in label order → channel indices 0–3 for the 4-mic early dataset.
#   ch 0 = MIC_UP_2,  ch 1 = MIC_UP_4,  ch 2 = MIC_UP_6,  ch 3 = MIC_UP_8
MIC_UPPER_XY: np.ndarray = np.array(
    [SENSOR_XYZ[k][:2] for k in ("MIC_UP_2", "MIC_UP_4", "MIC_UP_6", "MIC_UP_8")],
    dtype=np.float64,
)  # shape (4, 2)

# All 9 mics in numerical label order (BOT/UP interleaved: 1,2,3,4,5,6,7,8,9).
#   ch 0 = MIC_BOT_1, ch 1 = MIC_UP_2, ch 2 = MIC_BOT_3, ch 3 = MIC_UP_4,
#   ch 4 = MIC_BOT_5, ch 5 = MIC_UP_6, ch 6 = MIC_BOT_7, ch 7 = MIC_UP_8,
#   ch 8 = MIC_BOT_9
MIC_ALL_XYZ: np.ndarray = np.array(
    [
        SENSOR_XYZ[k]
        for k in (
            "MIC_BOT_1",
            "MIC_UP_2",
            "MIC_BOT_3",
            "MIC_UP_4",
            "MIC_BOT_5",
            "MIC_UP_6",
            "MIC_BOT_7",
            "MIC_UP_8",
            "MIC_BOT_9",
        )
    ],
    dtype=np.float64,
)  # shape (9, 3)

# 4 accelerometers in label order → channel indices 0–3.
#   ch 0 = ACCEL_1,  ch 1 = ACCEL_2,  ch 2 = ACCEL_3,  ch 3 = ACCEL_4
ACCEL_XYZ: np.ndarray = np.array(
    [SENSOR_XYZ[k] for k in ("ACCEL_1", "ACCEL_2", "ACCEL_3", "ACCEL_4")],
    dtype=np.float64,
)  # shape (4, 3)


# ==============================================================================
#  SECOND TEST DATASET GEOMETRY — Bench-top prototype setup
# ==============================================================================
# Source: data/second_test_dataset/node_position.txt (unit: cm → converted to m)
#
# 5 microphones : D, E, F, G, I  (channel order matches sorted filename: D=0 … I=4)
# 5 vibration sensors: A, B, C, D, E  (channel order A=0 … E=4)
#
# Coordinate system: arbitrary lab frame, origin at corner of the test box.
#   x-axis : width of the box
#   y-axis : depth (length) of the box
#   z-axis : height above the table surface
#
# Fault injection positions (folder names in data/second_test_dataset/RandomFault/)
# correspond to the vibration sensor positions below (in integer cm, rounded).
# ==============================================================================

# Raw positions from node_position.txt (cm)
_S2_VIB_XYZ_CM: dict[str, list[float]] = {
    "vibration_A": [10.0, 0.0, 23.0],
    "vibration_B": [15.5, 6.0, 15.0],
    "vibration_C": [0.0, 17.0, 12.0],
    "vibration_D": [0.0, 40.0, 15.0],
    "vibration_E": [15.5, 30.0, 16.0],
}
_S2_MIC_XYZ_CM: dict[str, list[float]] = {
    "mic_D": [0.0, 41.0, 15.0],
    "mic_E": [0.0, 31.0, 16.0],
    "mic_F": [10.0, 0.0, 24.0],
    "mic_G": [15.5, 5.0, 15.0],
    "mic_I": [0.0, 10.0, 15.0],
}

# Converted to metres — used throughout the localization pipeline
S2_VIB_XYZ_M: dict[str, np.ndarray] = {
    k: np.array(v, dtype=np.float64) / 100.0 for k, v in _S2_VIB_XYZ_CM.items()
}
S2_MIC_XYZ_M: dict[str, np.ndarray] = {
    k: np.array(v, dtype=np.float64) / 100.0 for k, v in _S2_MIC_XYZ_CM.items()
}

# Ordered arrays (channel index = alphabetical / label order as returned by glob sort)
#   mic channels: D=0, E=1, F=2, G=3, I=4
#   vib channels: A=0, B=1, C=2, D=3, E=4
S2_MIC_XYZ: np.ndarray = np.array(
    [S2_MIC_XYZ_M[k] for k in ("mic_D", "mic_E", "mic_F", "mic_G", "mic_I")],
    dtype=np.float64,
)  # shape (5, 3)

S2_VIB_XYZ: np.ndarray = np.array(
    [
        S2_VIB_XYZ_M[k]
        for k in (
            "vibration_A",
            "vibration_B",
            "vibration_C",
            "vibration_D",
            "vibration_E",
        )
    ],
    dtype=np.float64,
)  # shape (5, 3)

# Ground truth fault positions (cm, integer) → label as written in folder names
S2_FAULT_POSITIONS_CM: dict[str, np.ndarray] = {
    "pos_(10,0,23)": np.array([10.0, 0.0, 23.0]),  # vibration_A
    "pos_(15,6,15)": np.array([15.0, 6.0, 15.0]),  # vibration_B (≈15.5)
    "pos_(0,17,12)": np.array([0.0, 17.0, 12.0]),  # vibration_C
    "pos_(0,40,15)": np.array([0.0, 40.0, 15.0]),  # vibration_D
    "pos_(15,30,15)": np.array([15.0, 30.0, 15.0]),  # vibration_E (≈15.5, 16→15)
}
# Same set in metres for direct comparison with localization output
S2_FAULT_POSITIONS_M: dict[str, np.ndarray] = {
    k: v / 100.0 for k, v in S2_FAULT_POSITIONS_CM.items()
}

# Max inter-mic distance for TDOA range (mic_D ↔ mic_F ≈ diagonal of the box)
_S2_MAX_MIC_DIST_M: float = float(
    max(
        np.linalg.norm(S2_MIC_XYZ[i] - S2_MIC_XYZ[j])
        for i in range(len(S2_MIC_XYZ))
        for j in range(i + 1, len(S2_MIC_XYZ))
    )
)

# Mic-pair indices for second dataset: C(5,2) = 10 pairs
MIC_PAIRS_S2: list[tuple[int, int]] = [
    (i, j) for i in range(5) for j in range(i + 1, 5)
]
N_PAIRS_S2: int = len(MIC_PAIRS_S2)  # 10

# Vibration-pair indices for second dataset: C(5,2) = 10 pairs
VIB_PAIRS_S2: list[tuple[int, int]] = [
    (i, j) for i in range(5) for j in range(i + 1, 5)
]
N_VIB_PAIRS_S2: int = len(VIB_PAIRS_S2)  # 10


# ==============================================================================
# Second test dataset — LocalizationCNNS2
# ==============================================================================
# For the bench-top prototype: 5 mics, 10 pairs, 3-D output (x, y, z).
# No FiLM context required — standalone module, does not depend on DetectionHead.
#
# Training modes:
#   Supervised  : supervised_localization_loss_s2  (ground-truth position labels)
#   Self-supervised: geometric_consistency_loss_s2 (no labels required, physics-only)
#   Combined    : supervised + lambda * geometric_consistency (recommended)
# ==============================================================================

# Derived constants for the second dataset
_S2_MAX_DELAY_SAMPLES: int = int(_S2_MAX_MIC_DIST_M / 343.0 * 16000)
_S2_GCC_LENGTH: int = 2 * _S2_MAX_DELAY_SAMPLES + 1


def compute_gcc_stack_s2_multiwindow(
    mic_data: np.ndarray,
    fs: float,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    c: float = 343.0,
) -> np.ndarray:
    """Multi-window averaged GCC-PHAT stack for the 5-mic second dataset.

    Short-window PHAT normalization followed by time-averaging reduces
    reverberation noise compared to computing GCC on the full signal once.
    Each window gets its own PHAT whitening, so transient artefacts stay local.

    Args:
        mic_data: (5, N_samples) float array — raw mic waveforms.
        fs: Sample rate in Hz.
        window_s: Analysis window length in seconds.
        hop_s: Hop between successive windows in seconds.
        c: Speed of sound in m/s.

    Returns:
        gcc_avg: float32 array of shape (10, L_s2).
    """
    window_samples = int(window_s * fs)
    hop_samples = max(1, int(hop_s * fs))
    max_delay = _S2_MAX_DELAY_SAMPLES
    n_samples = mic_data.shape[1]

    stacks: list[np.ndarray] = []
    start = 0
    while start + window_samples <= n_samples:
        frame = mic_data[:, start : start + window_samples]
        rows = [gcc_phat(frame[i], frame[j], max_delay) for i, j in MIC_PAIRS_S2]
        stacks.append(np.stack(rows, axis=0))  # (10, L)
        start += hop_samples

    if not stacks:
        # Recording shorter than one window — process full signal as-is
        rows = [gcc_phat(mic_data[i], mic_data[j], max_delay) for i, j in MIC_PAIRS_S2]
        return np.stack(rows, axis=0)

    return np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32)  # (10, L)


# ---------------------------------------------------------------------------
# Geometry-aware pair embeddings for 10 pairs in 3-D space
# ---------------------------------------------------------------------------


class _GeometryAwarePairEmbeddingS2(nn.Module):
    """Encodes physical geometry of each of 10 mic pairs into d_model space.

    Features per pair: [mid_x, mid_y, mid_z, length, angle_xy, angle_z]  (6-D).
    A linear projection maps to d_model so the Transformer attention can
    exploit the 3-D physical layout of the second dataset's sensor array.

    Args:
        mic_xyz: (5, 3) float Tensor — mic positions in metres (S2_MIC_XYZ).
        d_model: Embedding dimension.
    """

    def __init__(self, mic_xyz: torch.Tensor, d_model: int) -> None:
        super().__init__()
        geom = self._make_geom_features(mic_xyz)  # (10, 6)
        self.register_buffer("_geom", geom)
        self._proj = nn.Linear(6, d_model)

    @staticmethod
    def _make_geom_features(mic_xyz: torch.Tensor) -> torch.Tensor:
        rows: list[list[float]] = []
        for i, j in MIC_PAIRS_S2:
            pi = mic_xyz[i]  # (3,)
            pj = mic_xyz[j]
            mid_x = float(((pi[0] + pj[0]) / 2).item())
            mid_y = float(((pi[1] + pj[1]) / 2).item())
            mid_z = float(((pi[2] + pj[2]) / 2).item())
            dx = float((pj[0] - pi[0]).item())
            dy = float((pj[1] - pi[1]).item())
            dz = float((pj[2] - pi[2]).item())
            length = float(torch.norm(pj - pi).item())
            angle_xy = math.atan2(dy, dx)
            angle_z = math.atan2(dz, math.sqrt(dx**2 + dy**2))
            rows.append([mid_x, mid_y, mid_z, length, angle_xy, angle_z])
        return torch.tensor(rows, dtype=torch.float32)  # (10, 6)

    def forward(self) -> torch.Tensor:
        """Returns pair embeddings of shape (10, d_model)."""
        return self._proj(self._geom)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LocalizationCNNS2: the combined SRP+neural estimator for the second dataset
# ---------------------------------------------------------------------------


class LocalizationCNNS2(nn.Module):
    """Transformer-based 3-D source localizer for the second bench-top dataset.

    Architecture:
        - Projects each of 10 GCC-PHAT vectors (L_s2,) → d_model
        - Adds geometry-aware pair embeddings (3-D midpoint / length / angles)
        - Concatenates SRP-PHAT peak position as an explicit spatial prior token
        - 2-layer Transformer encoder over 11 tokens (10 pairs + 1 SRP prior)
        - Mean-pool → MLP head → (x, y, z) in metres

    The SRP-PHAT prior token anchors the neural output close to the physics-
    based estimate; the self-attention then refines it by weighting pairs
    selectively based on the geometry and the observed GCC peaks.

    Args:
        d_model: Internal feature dimension (must be divisible by n_heads=4).
        dropout: Dropout rate.
    """

    # Fixed mic geometry for second dataset
    _MIC_XYZ: np.ndarray = S2_MIC_XYZ  # (5, 3)

    def __init__(
        self,
        d_model: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        # L_s2 is determined at runtime from the gcc input; store for reference
        self._max_delay_samples = _S2_MAX_DELAY_SAMPLES

        mic_t = torch.tensor(self._MIC_XYZ, dtype=torch.float32)
        self._pair_pe = _GeometryAwarePairEmbeddingS2(mic_t, d_model)

        # Project each GCC-PHAT vector (L_s2,) → d_model
        self._gcc_proj = nn.Linear(_S2_GCC_LENGTH, d_model)

        # SRP prior token: encode the 3-D SRP-PHAT peak position → d_model
        self._srp_prior_proj = nn.Linear(3, d_model)

        # Transformer over 10 pair tokens + 1 SRP prior token = 11 tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self._drop = nn.Dropout(p=dropout)

        # Regression head: d_model → (x, y, z) in metres
        self._head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model // 2, 3),
        )

    def forward(
        self,
        gcc: torch.Tensor,
        srp_prior_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            gcc:            (batch, 10, L_s2) — multi-window averaged GCC-PHAT stack.
            srp_prior_xyz:  (batch, 3) — SRP-PHAT peak position in metres.

        Returns:
            pos: (batch, 3) — refined (x, y, z) position in metres.
        """
        # GCC tokens: (batch, 10, L_s2) → (batch, 10, d_model) with pair PE
        tokens = self._gcc_proj(gcc)  # (batch, 10, d_model)
        pair_pe = self._pair_pe()  # (10, d_model)
        tokens = tokens + pair_pe.unsqueeze(0)  # broadcast over batch

        # SRP prior token: (batch, 3) → (batch, 1, d_model)
        srp_token = self._srp_prior_proj(srp_prior_xyz).unsqueeze(1)  # (batch, 1, d)

        # Concatenate: 10 pair tokens + 1 SRP prior = 11 tokens
        all_tokens = torch.cat([tokens, srp_token], dim=1)  # (batch, 11, d_model)

        # Self-attention: pairs attend to each other AND to the SRP prior
        out = self._encoder(all_tokens)  # (batch, 11, d_model)

        # Pool only over the 10 pair tokens (ignore the SRP prior token)
        x = out[:, :10, :].mean(dim=1)  # (batch, d_model)
        x = self._drop(x)
        return self._head(x)  # (batch, 3)


# ---------------------------------------------------------------------------
# Losses for LocalizationCNNS2
# ---------------------------------------------------------------------------


def supervised_localization_loss_s2(
    pred_xyz: torch.Tensor,
    true_xyz: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for 3-D position regression (second dataset).

    Args:
        pred_xyz: (batch, 3) — predicted (x, y, z) in metres.
        true_xyz: (batch, 3) — ground-truth (x, y, z) in metres.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(pred_xyz, true_xyz)


def geometric_consistency_loss_s2(
    pred_xyz: torch.Tensor,
    gcc_stack: torch.Tensor,
    mic_xyz: torch.Tensor,
    max_delay_samples: int,
    c_air: float = 343.0,
    fs: float = 16000.0,
) -> torch.Tensor:
    """Self-supervised 3-D consistency: predicted position → implied TDOAs must
    match the GCC-PHAT peaks across all 10 mic pairs.

    No ground-truth position labels required. Physics-derived TDOAs for the
    predicted (x,y,z) are matched to the measured GCC-PHAT soft peaks via
    differentiable soft-argmax, enabling gradient-based training without labels.

    Args:
        pred_xyz:          (batch, 3) — predicted (x, y, z) in metres.
        gcc_stack:         (batch, 10, L) — GCC-PHAT stack (10 pairs).
        mic_xyz:           (5, 3) float Tensor — mic positions in metres.
        max_delay_samples: Half-width of the GCC vector.
        c_air:             Speed of sound (m/s).
        fs:                Sample rate (Hz).

    Returns:
        Scalar mean MSE over all 10 pairs.
    """
    total = torch.zeros(1, device=pred_xyz.device, dtype=pred_xyz.dtype)
    for k, (i, j) in enumerate(MIC_PAIRS_S2):
        pi = mic_xyz[i]  # (3,)
        pj = mic_xyz[j]  # (3,)

        d_i = torch.norm(pred_xyz - pi.unsqueeze(0), dim=-1)  # (batch,)
        d_j = torch.norm(pred_xyz - pj.unsqueeze(0), dim=-1)
        tdoa_expected = (d_i - d_j) / c_air * fs  # (batch,) in samples

        gcc_k = gcc_stack[:, k, :]  # (batch, L)
        weights = torch.softmax(gcc_k * 10.0, dim=-1)  # sharpened soft-argmax
        lags = torch.arange(gcc_k.shape[-1], device=gcc_k.device, dtype=gcc_k.dtype)
        tdoa_measured = (weights * lags).sum(dim=-1) - float(max_delay_samples)

        total = total + F.mse_loss(tdoa_expected, tdoa_measured)

    return total / float(N_PAIRS_S2)


# ==============================================================================
#  THIRD TEST DATASET GEOMETRY — Bench-top prototype v2 setup
# ==============================================================================
# Source: data/third_test_dataset/position.json (unit: cm → converted to m)
#
# 9 microphones: D_l, D_r, E, F_l, F_r, G_l, G_r, J_l, J_r
#   Stereo pairs stored as separate mono WAV files.
#   Channel order = sorted filename order (alphabetical by full filename):
#     ch 0 = recorded_D_l.wav  → Dl  (6,  5,  1) cm
#     ch 1 = recorded_D_r.wav  → Dr  (11, 1,  1) cm
#     ch 2 = recorded_E.wav    → E   (9, -4,  1) cm
#     ch 3 = recorded_F_l.wav  → Fl  (6, -5,  8) cm
#     ch 4 = recorded_F_r.wav  → Fr  (0,  0,  1) cm
#     ch 5 = recorded_G_l.wav  → Gl  (4, -5,  1) cm
#     ch 6 = recorded_G_r.wav  → Gr  (11, 0,  8) cm
#     ch 7 = recorded_J_l.wav  → Jl  (6,  5,  8) cm
#     ch 8 = recorded_J_r.wav  → Jr  (0,  0,  8) cm
#
# 4 vibration sensors: D, E, F, J
#   Channel order = sorted filename order:
#     ch 0 = vibration_D.csv → (V)D  (6, -5, 3) cm
#     ch 1 = vibration_E.csv → (V)E  (11, 0, 3) cm
#     ch 2 = vibration_F.csv → (V)F  (0,  0, 3) cm
#     ch 3 = vibration_J.csv → (V)J  (6,  5, 3) cm
#
# Coordinate system: same lab frame as second dataset (cm).
# ==============================================================================

_S3_MIC_XYZ_CM: dict[str, list[float]] = {
    "mic_Dl": [6.0, 5.0, 1.0],    # ch 0 — recorded_D_l.wav
    "mic_Dr": [11.0, 1.0, 1.0],   # ch 1 — recorded_D_r.wav
    "mic_E":  [9.0, -4.0, 1.0],   # ch 2 — recorded_E.wav
    "mic_Fl": [6.0, -5.0, 8.0],   # ch 3 — recorded_F_l.wav
    "mic_Fr": [0.0, 0.0, 1.0],    # ch 4 — recorded_F_r.wav
    "mic_Gl": [4.0, -5.0, 1.0],   # ch 5 — recorded_G_l.wav
    "mic_Gr": [11.0, 0.0, 8.0],   # ch 6 — recorded_G_r.wav
    "mic_Jl": [6.0, 5.0, 8.0],    # ch 7 — recorded_J_l.wav
    "mic_Jr": [0.0, 0.0, 8.0],    # ch 8 — recorded_J_r.wav
}
_S3_VIB_XYZ_CM: dict[str, list[float]] = {
    "vibration_D": [6.0, -5.0, 3.0],   # ch 0 — (V)D
    "vibration_E": [11.0, 0.0, 3.0],   # ch 1 — (V)E
    "vibration_F": [0.0, 0.0, 3.0],    # ch 2 — (V)F
    "vibration_J": [6.0, 5.0, 3.0],    # ch 3 — (V)J
}

_S3_MIC_KEY_ORDER: tuple[str, ...] = (
    "mic_Dl", "mic_Dr", "mic_E", "mic_Fl", "mic_Fr",
    "mic_Gl", "mic_Gr", "mic_Jl", "mic_Jr",
)
_S3_VIB_KEY_ORDER: tuple[str, ...] = (
    "vibration_D", "vibration_E", "vibration_F", "vibration_J",
)

# Ordered arrays in metres (divide cm by 100)
S3_MIC_XYZ: np.ndarray = np.array(
    [_S3_MIC_XYZ_CM[k] for k in _S3_MIC_KEY_ORDER],
    dtype=np.float64,
) / 100.0  # shape (9, 3)

S3_VIB_XYZ: np.ndarray = np.array(
    [_S3_VIB_XYZ_CM[k] for k in _S3_VIB_KEY_ORDER],
    dtype=np.float64,
) / 100.0  # shape (4, 3)

# Ground truth for hit_between_Fl_Gr_speed1:
#   Both Fl=(6,-5,8) cm and Gr=(11,0,8) cm lie at z=8cm — the same height.
#   The hit is constrained to z=8cm (same height as both sensors); x,y is
#   approximated as their centroid.  z=8cm is the reliable constraint;
#   x,y carry larger uncertainty.
S3_HIT_FL_GR_APPROX_CM: np.ndarray = np.array([8.5, -2.5, 8.0], dtype=np.float64)
S3_HIT_FL_GR_APPROX_M: np.ndarray = S3_HIT_FL_GR_APPROX_CM / 100.0

# All C(9,2) = 36 mic pairs
MIC_PAIRS_S3: list[tuple[int, int]] = [
    (i, j) for i in range(9) for j in range(i + 1, 9)
]
N_PAIRS_S3: int = len(MIC_PAIRS_S3)  # 36

# Vibration-pair indices for third dataset: C(4,2) = 6 pairs
VIB_PAIRS_S3: list[tuple[int, int]] = [
    (i, j) for i in range(4) for j in range(i + 1, 4)
]
N_VIB_PAIRS_S3: int = len(VIB_PAIRS_S3)  # 6

# Five-mic zero-shot subset: Dl(0), Dr(1), Fr(4), Gr(6), Jl(7).
# Chosen for maximal spatial spread — used to run the S2 neural model on S3 data
# without retraining it (S2 model expects 10-pair GCC from 5 mics in S2 layout).
S3_ZERO_SHOT_MIC_INDICES: tuple[int, ...] = (0, 1, 4, 6, 7)

_S3_MAX_MIC_DIST_M: float = float(
    max(
        np.linalg.norm(S3_MIC_XYZ[i] - S3_MIC_XYZ[j])
        for i in range(len(S3_MIC_XYZ))
        for j in range(i + 1, len(S3_MIC_XYZ))
    )
)
_S3_MAX_DELAY_SAMPLES: int = int(_S3_MAX_MIC_DIST_M / 343.0 * 16000)
_S3_GCC_LENGTH: int = 2 * _S3_MAX_DELAY_SAMPLES + 1


# ==============================================================================
#  Generic SRP-PHAT / TDOA helpers — dataset-agnostic, parametrised by mic_pairs
# ==============================================================================


def srp_phat_3d_hierarchical(
    gcc_stack: np.ndarray,
    mic_xyz: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    coarse_res: float = 0.02,
    fine_res: float = 0.005,
    fine_margin: float = 0.05,
    fs: float = 16000.0,
    c: float = 343.0,
    *,
    mic_pairs: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, float]:
    """Two-level hierarchical SRP-PHAT (coarse grid then fine grid around peak).

    Dataset-agnostic version; works for any number of mics/pairs.

    Args:
        gcc_stack:  (n_pairs, L) averaged GCC-PHAT stack.
        mic_xyz:    (n_mics, 3) mic positions in metres.
        lo, hi:     (3,) bounding-box corners in metres.
        coarse_res: Grid spacing for the coarse search (m).
        fine_res:   Grid spacing for the fine search (m).
        fine_margin: Half-extent of the fine search cube around the coarse peak (m).
        fs:         Sample rate (Hz).
        c:          Speed of sound (m/s).
        mic_pairs:  List of (i, j) pair indices. Defaults to MIC_PAIRS_S2.

    Returns:
        (estimated_position_m, srp_peak_power) — both float64 / float.
    """
    if mic_pairs is None:
        mic_pairs = MIC_PAIRS_S2

    grid_x = np.arange(lo[0], hi[0] + coarse_res, coarse_res)
    grid_y = np.arange(lo[1], hi[1] + coarse_res, coarse_res)
    grid_z = np.arange(lo[2], hi[2] + coarse_res, coarse_res)
    srp = srp_phat_3d(
        gcc_stack, mic_xyz, grid_x, grid_y, grid_z, fs=fs, c=c, mic_pairs=mic_pairs
    )
    peak_idx = np.unravel_index(int(np.argmax(srp)), srp.shape)
    coarse_peak = np.array(
        [grid_x[peak_idx[0]], grid_y[peak_idx[1]], grid_z[peak_idx[2]]],
        dtype=np.float64,
    )

    fine_lo = np.maximum(coarse_peak - fine_margin, lo)
    fine_hi = np.minimum(coarse_peak + fine_margin, hi)
    fine_gx = np.arange(fine_lo[0], fine_hi[0] + fine_res, fine_res)
    fine_gy = np.arange(fine_lo[1], fine_hi[1] + fine_res, fine_res)
    fine_gz = np.arange(fine_lo[2], fine_hi[2] + fine_res, fine_res)
    srp_fine = srp_phat_3d(
        gcc_stack, mic_xyz, fine_gx, fine_gy, fine_gz, fs=fs, c=c, mic_pairs=mic_pairs
    )
    fine_idx = np.unravel_index(int(np.argmax(srp_fine)), srp_fine.shape)
    estimated = np.array(
        [fine_gx[fine_idx[0]], fine_gy[fine_idx[1]], fine_gz[fine_idx[2]]],
        dtype=np.float64,
    )
    return estimated, float(srp_fine[fine_idx])


def tdoa_triangulate(
    gcc_stack: np.ndarray,
    mic_xyz: np.ndarray,
    srp_init_m: np.ndarray,
    fs: float,
    c: float = 343.0,
    *,
    mic_pairs: list[tuple[int, int]] | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, float]:
    """Refine a source-position estimate via TDOA non-linear least-squares.

    Dataset-agnostic version; the pair list and optimisation bounds are passed
    explicitly so the same function works for both the 10-pair S2 geometry and
    the 36-pair S3 geometry (and any future layout).

    Args:
        gcc_stack:  (n_pairs, L) averaged GCC-PHAT stack.
        mic_xyz:    (n_mics, 3) mic positions in metres.
        srp_init_m: (3,) initial position estimate (from SRP-PHAT).
        fs:         Sample rate (Hz).
        c:          Speed of sound (m/s).
        mic_pairs:  List of (i, j) pair indices. Defaults to MIC_PAIRS_S2.
        bounds:     Per-dimension box bounds [(xlo,xhi),(ylo,yhi),(zlo,zhi)].
                    Defaults to loose 40 cm margins around S2 geometry.

    Returns:
        (refined_position_m, residual_sum_sq)
    """
    from scipy.optimize import minimize  # type: ignore[import-untyped]

    if mic_pairs is None:
        mic_pairs = MIC_PAIRS_S2
    if bounds is None:
        bounds = [(-0.10, 0.30), (-0.10, 0.60), (-0.10, 0.40)]

    max_delay = gcc_stack.shape[1] // 2
    measured_tdoa_s = np.array(
        [
            (float(np.argmax(gcc_stack[k])) - max_delay) / fs
            for k in range(len(mic_pairs))
        ]
    )
    mic_i_xyz = np.stack([mic_xyz[i] for i, j in mic_pairs])  # (n_pairs, 3)
    mic_j_xyz = np.stack([mic_xyz[j] for i, j in mic_pairs])

    def _residuals(pos: np.ndarray) -> float:
        di = np.linalg.norm(mic_i_xyz - pos, axis=-1)
        dj = np.linalg.norm(mic_j_xyz - pos, axis=-1)
        return float(np.sum((measured_tdoa_s - (di - dj) / c) ** 2))

    def _jac(pos: np.ndarray) -> np.ndarray:
        di = np.linalg.norm(mic_i_xyz - pos, axis=-1, keepdims=True) + 1e-9
        dj = np.linalg.norm(mic_j_xyz - pos, axis=-1, keepdims=True) + 1e-9
        theory_tdoa_s = ((di - dj) / c).squeeze(-1)
        residuals = measured_tdoa_s - theory_tdoa_s
        grad_di = (pos - mic_i_xyz) / (di * c)
        grad_dj = (pos - mic_j_xyz) / (dj * c)
        grad_theory = grad_di - grad_dj
        return float(-2) * (residuals[:, None] * grad_theory).sum(axis=0)

    result = minimize(
        _residuals,
        x0=srp_init_m,
        jac=_jac,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10},
    )
    return result.x.astype(np.float64), float(result.fun)


# ==============================================================================
#  Third test dataset — LocalizationCNNS3
# ==============================================================================
# 9 mics, 36 pairs, 3-D output (x, y, z).
# Mirrors LocalizationCNNS2 but scaled for the larger sensor array.
#
# Training modes:
#   Supervised      : supervised_localization_loss_s3
#   Self-supervised : geometric_consistency_loss_s3
#   Combined        : supervised + lambda * geometric_consistency (recommended)
#
# With only one ground-truth fault position available, geo-consistency loss
# provides the primary training signal (recommended weight: 0.20–0.30).
# ==============================================================================


def compute_gcc_stack_s3_multiwindow(
    mic_data: np.ndarray,
    fs: float,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    c: float = 343.0,
) -> np.ndarray:
    """Multi-window averaged GCC-PHAT stack for the 9-mic third dataset (36 pairs).

    Args:
        mic_data: (9, N_samples) float array — raw mic waveforms.
        fs: Sample rate in Hz.
        window_s: Analysis window length in seconds.
        hop_s: Hop between successive windows in seconds.
        c: Speed of sound in m/s.

    Returns:
        gcc_avg: float32 array of shape (36, L_s3).
    """
    window_samples = int(window_s * fs)
    hop_samples = max(1, int(hop_s * fs))
    max_delay = _S3_MAX_DELAY_SAMPLES
    n_samples = mic_data.shape[1]

    stacks: list[np.ndarray] = []
    start = 0
    while start + window_samples <= n_samples:
        frame = mic_data[:, start : start + window_samples]
        rows = [gcc_phat(frame[i], frame[j], max_delay) for i, j in MIC_PAIRS_S3]
        stacks.append(np.stack(rows, axis=0))  # (36, L)
        start += hop_samples

    if not stacks:
        rows = [gcc_phat(mic_data[i], mic_data[j], max_delay) for i, j in MIC_PAIRS_S3]
        return np.stack(rows, axis=0)

    return np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32)  # (36, L)


class _GeometryAwarePairEmbeddingS3(nn.Module):
    """Geometry-aware pair embeddings for the 9-mic third dataset (36 pairs).

    Identical to _GeometryAwarePairEmbeddingS2 but parameterised for 36 pairs
    and the S3 sensor geometry.

    Args:
        mic_xyz: (9, 3) float Tensor — mic positions in metres (S3_MIC_XYZ).
        d_model: Embedding dimension.
    """

    def __init__(self, mic_xyz: torch.Tensor, d_model: int) -> None:
        super().__init__()
        geom = self._make_geom_features(mic_xyz)  # (36, 6)
        self.register_buffer("_geom", geom)
        self._proj = nn.Linear(6, d_model)

    @staticmethod
    def _make_geom_features(mic_xyz: torch.Tensor) -> torch.Tensor:
        rows: list[list[float]] = []
        for i, j in MIC_PAIRS_S3:
            pi = mic_xyz[i]
            pj = mic_xyz[j]
            mid_x = float(((pi[0] + pj[0]) / 2).item())
            mid_y = float(((pi[1] + pj[1]) / 2).item())
            mid_z = float(((pi[2] + pj[2]) / 2).item())
            dx = float((pj[0] - pi[0]).item())
            dy = float((pj[1] - pi[1]).item())
            dz = float((pj[2] - pi[2]).item())
            length = float(torch.norm(pj - pi).item())
            angle_xy = math.atan2(dy, dx)
            angle_z = math.atan2(dz, math.sqrt(dx**2 + dy**2))
            rows.append([mid_x, mid_y, mid_z, length, angle_xy, angle_z])
        return torch.tensor(rows, dtype=torch.float32)  # (36, 6)

    def forward(self) -> torch.Tensor:
        """Returns pair embeddings of shape (36, d_model)."""
        return self._proj(self._geom)  # type: ignore[arg-type]


class LocalizationCNNS3(nn.Module):
    """Transformer-based 3-D source localizer for the third bench-top dataset.

    Architecture mirrors LocalizationCNNS2 but scaled for 9 mics / 36 pairs:
        - Projects each of 36 GCC-PHAT vectors (L_s3,) → d_model
        - Adds geometry-aware pair embeddings (3-D midpoint / length / angles)
        - Concatenates SRP-PHAT prior token
        - 2-layer Transformer encoder over 37 tokens (36 pairs + 1 SRP prior)
        - Mean-pool over 36 pair tokens → MLP head → (x, y, z) in metres

    With 36 TDOA constraints vs 10 in S2, the model has richer spatial
    information. The harder acoustic environment (fan noise) makes the
    self-supervised geometric consistency loss particularly valuable when
    only one ground-truth fault position is available for supervised training.

    Args:
        d_model: Internal feature dimension (must be divisible by n_heads=4).
        dropout: Dropout rate.
    """

    _MIC_XYZ: np.ndarray = S3_MIC_XYZ  # (9, 3)

    def __init__(
        self,
        d_model: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self._max_delay_samples = _S3_MAX_DELAY_SAMPLES

        mic_t = torch.tensor(self._MIC_XYZ, dtype=torch.float32)
        self._pair_pe = _GeometryAwarePairEmbeddingS3(mic_t, d_model)

        # Project each GCC-PHAT vector (L_s3,) → d_model
        self._gcc_proj = nn.Linear(_S3_GCC_LENGTH, d_model)

        # SRP prior token: 3-D SRP-PHAT peak position → d_model
        self._srp_prior_proj = nn.Linear(3, d_model)

        # Transformer over 36 pair tokens + 1 SRP prior = 37 tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self._drop = nn.Dropout(p=dropout)

        # Regression head: d_model → (x, y, z) in metres
        self._head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model // 2, 3),
        )

    def forward(
        self,
        gcc: torch.Tensor,
        srp_prior_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            gcc:           (batch, 36, L_s3) — multi-window averaged GCC-PHAT stack.
            srp_prior_xyz: (batch, 3) — SRP-PHAT peak position in metres.

        Returns:
            pos: (batch, 3) — refined (x, y, z) position in metres.
        """
        tokens = self._gcc_proj(gcc)                         # (batch, 36, d_model)
        pair_pe = self._pair_pe()                            # (36, d_model)
        tokens = tokens + pair_pe.unsqueeze(0)               # broadcast over batch

        srp_token = self._srp_prior_proj(srp_prior_xyz).unsqueeze(1)  # (batch, 1, d)
        all_tokens = torch.cat([tokens, srp_token], dim=1)   # (batch, 37, d_model)

        out = self._encoder(all_tokens)                      # (batch, 37, d_model)
        x = out[:, :36, :].mean(dim=1)                      # pool over pair tokens
        x = self._drop(x)
        return self._head(x)                                 # (batch, 3)


# ---------------------------------------------------------------------------
# Losses for LocalizationCNNS3
# ---------------------------------------------------------------------------


def supervised_localization_loss_s3(
    pred_xyz: torch.Tensor,
    true_xyz: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for 3-D position regression (third dataset).

    Args:
        pred_xyz: (batch, 3) — predicted (x, y, z) in metres.
        true_xyz: (batch, 3) — ground-truth (x, y, z) in metres.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(pred_xyz, true_xyz)


def geometric_consistency_loss_s3(
    pred_xyz: torch.Tensor,
    gcc_stack: torch.Tensor,
    mic_xyz: torch.Tensor,
    max_delay_samples: int,
    c_air: float = 343.0,
    fs: float = 16000.0,
) -> torch.Tensor:
    """Self-supervised 3-D TDOA consistency loss for the third dataset (36 pairs).

    Mirrors geometric_consistency_loss_s2 but uses MIC_PAIRS_S3 / N_PAIRS_S3.
    Particularly important for S3 training where only one ground-truth fault
    position is available — the physics-based loss provides gradient signal
    from every window without requiring additional labelled examples.

    Args:
        pred_xyz:          (batch, 3) — predicted (x, y, z) in metres.
        gcc_stack:         (batch, 36, L) — GCC-PHAT stack (36 pairs).
        mic_xyz:           (9, 3) float Tensor — mic positions in metres.
        max_delay_samples: Half-width of the GCC vector.
        c_air:             Speed of sound (m/s).
        fs:                Sample rate (Hz).

    Returns:
        Scalar mean MSE over all 36 pairs.
    """
    total = torch.zeros(1, device=pred_xyz.device, dtype=pred_xyz.dtype)
    for k, (i, j) in enumerate(MIC_PAIRS_S3):
        pi = mic_xyz[i]
        pj = mic_xyz[j]

        d_i = torch.norm(pred_xyz - pi.unsqueeze(0), dim=-1)  # (batch,)
        d_j = torch.norm(pred_xyz - pj.unsqueeze(0), dim=-1)
        tdoa_expected = (d_i - d_j) / c_air * fs              # (batch,) in samples

        gcc_k = gcc_stack[:, k, :]                            # (batch, L)
        weights = torch.softmax(gcc_k * 10.0, dim=-1)
        lags = torch.arange(gcc_k.shape[-1], device=gcc_k.device, dtype=gcc_k.dtype)
        tdoa_measured = (weights * lags).sum(dim=-1) - float(max_delay_samples)

        total = total + F.mse_loss(tdoa_expected, tdoa_measured)

    return total / float(N_PAIRS_S3)


# ==============================================================================
#  STRUCTURAL WAVE GCC-PHAT — Accelerometer-based TDOA stream
# ==============================================================================
# Treats vibration accelerometers as a second TDOA stream using structural
# (Lamb / bending) wave propagation at c_struct instead of airborne c_air.
#
# Physical differences from acoustic GCC-PHAT:
#   - Wave speed: ~5000 m/s (steel compressional) vs 343 m/s (air)
#   - Max TDOA is much smaller → sub-sample parabolic interpolation in the GCC
#     peak is essential when even a 40 cm baseline gives < 1 sample of delay
#   - AE-band bandpass (100–3000 Hz) applied before correlating to suppress
#     low-frequency hydrodynamic noise that dominates accelerometer spectra
# ==============================================================================

C_STRUCT_MS: float = 5000.0       # m/s — steel compressional (co-estimate in practice)
_VIB_BANDPASS_LO_HZ: float = 100.0
_VIB_BANDPASS_HI_HZ: float = 3000.0


def bandpass_iir(
    signal: np.ndarray,
    fs: float,
    lo_hz: float = _VIB_BANDPASS_LO_HZ,
    hi_hz: float = _VIB_BANDPASS_HI_HZ,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter for AE-band vibration signals.

    Args:
        signal: 1-D float array of samples.
        fs:     Sample rate in Hz.
        lo_hz:  Lower passband edge (Hz).
        hi_hz:  Upper passband edge (Hz).
        order:  One-sided filter order (effective order = 2× via sosfiltfilt).

    Returns:
        Filtered signal as float32 array of same length.
    """
    from scipy.signal import butter, sosfiltfilt  # type: ignore[import-untyped]

    nyq = fs / 2.0
    lo = min(lo_hz / nyq, 0.99)
    hi = min(hi_hz / nyq, 0.99)
    if lo >= hi or len(signal) < order * 6:
        return signal.astype(np.float32)
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def compute_gcc_stack_structural_multiwindow(
    vib_data: np.ndarray,
    fs: float,
    vib_xyz: np.ndarray,
    vib_pairs: list[tuple[int, int]],
    c_struct: float = C_STRUCT_MS,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    bandpass: bool = True,
) -> np.ndarray:
    """Multi-window GCC-PHAT stack for structural (accelerometer) signals.

    Mirrors :func:`compute_gcc_stack_s2_multiwindow` but uses *c_struct* to
    compute the max-lag window and optionally applies AE-band bandpass
    filtering before each window's GCC-PHAT computation.

    Sub-sample delay resolution is preserved in the GCC vector — even when the
    max structural delay is < 1 sample the sinc-interpolated GCC peak retains
    fractional-sample information.

    Args:
        vib_data:  (n_vib, N_samples) vibration amplitude waveforms at native rate.
        fs:        Native vibration sample rate in Hz.
        vib_xyz:   (n_vib, 3) sensor positions in metres (used to compute max lag).
        vib_pairs: List of (i, j) accelerometer pair indices.
        c_struct:  Structural wave speed in m/s.
        window_s:  Analysis window length in seconds.
        hop_s:     Hop between successive windows in seconds.
        bandpass:  Apply AE-band Butterworth bandpass when fs > 2×hi_hz.

    Returns:
        gcc_avg: float32 array of shape (n_pairs, L_struct).
            L_struct = 2 * max_delay_samples + 1 (minimum 3).
    """
    max_dist = max(
        float(np.linalg.norm(vib_xyz[i] - vib_xyz[j])) for i, j in vib_pairs
    ) if vib_pairs else 0.20
    max_delay_samples = max(1, int(math.ceil(max_dist / c_struct * fs)))

    window_samples = int(window_s * fs)
    hop_samples = max(1, int(hop_s * fs))
    n_samples = vib_data.shape[1]

    def _gcc_row(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        xi_f = xi.astype(np.float64)
        xj_f = xj.astype(np.float64)
        if bandpass and fs > 2.0 * _VIB_BANDPASS_HI_HZ:
            xi_f = bandpass_iir(xi_f, fs).astype(np.float64)
            xj_f = bandpass_iir(xj_f, fs).astype(np.float64)
        return gcc_phat(xi_f, xj_f, max_delay_samples)

    stacks: list[np.ndarray] = []
    start = 0
    while start + window_samples <= n_samples:
        frame = vib_data[:, start : start + window_samples]
        rows = [_gcc_row(frame[i], frame[j]) for i, j in vib_pairs]
        stacks.append(np.stack(rows, axis=0))
        start += hop_samples

    if not stacks:
        rows = [_gcc_row(vib_data[i], vib_data[j]) for i, j in vib_pairs]
        return np.stack(rows, axis=0)

    return np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32)


def structural_srp_phat_3d(
    gcc_struct_stack: np.ndarray,
    vib_xyz: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    fs: float,
    c_struct: float = C_STRUCT_MS,
    vib_pairs: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Vectorised SRP-PHAT over a 3-D grid using the structural GCC stack.

    Evaluates structural TDOA consistency at every grid point using *c_struct*.
    The resulting map can be stacked with the acoustic SRP map to form the
    2-channel input of :class:`LocalizationDualSRPNet`.

    Args:
        gcc_struct_stack: (n_vib_pairs, L_struct) structural GCC-PHAT stack.
        vib_xyz:          (n_vib, 3) vibration sensor positions in metres.
        grid_x/y/z:       1-D coordinate arrays for the search grid (metres).
        fs:               Vibration signal sample rate (Hz).
        c_struct:         Structural wave speed (m/s).
        vib_pairs:        Accelerometer pair indices. Defaults to all C(n_vib,2).

    Returns:
        float32 array of shape (Nx, Ny, Nz) — structural SRP power map.
    """
    n_vib = vib_xyz.shape[0]
    if vib_pairs is None:
        vib_pairs = [(i, j) for i in range(n_vib) for j in range(i + 1, n_vib)]

    max_delay = gcc_struct_stack.shape[1] // 2
    L = gcc_struct_stack.shape[1]

    gx, gy, gz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid_pts = np.stack([gx, gy, gz], axis=-1)  # (Nx, Ny, Nz, 3)
    srp = np.zeros((len(grid_x), len(grid_y), len(grid_z)), dtype=np.float32)

    for k, (i, j) in enumerate(vib_pairs):
        pi = vib_xyz[i]
        pj = vib_xyz[j]
        di = np.linalg.norm(grid_pts - pi, axis=-1)
        dj = np.linalg.norm(grid_pts - pj, axis=-1)
        tdoa_idx = np.round((di - dj) / c_struct * fs).astype(np.int32) + max_delay
        valid = (tdoa_idx >= 0) & (tdoa_idx < L)
        idx_safe = np.clip(tdoa_idx, 0, L - 1)
        srp += (gcc_struct_stack[k, idx_safe] * valid).astype(np.float32)

    return srp


def synthetic_gcc_stack(
    source_xyz: np.ndarray,
    mic_xyz: np.ndarray,
    mic_pairs: list[tuple[int, int]],
    fs: float,
    c: float = 343.0,
    max_delay_samples: int | None = None,
    *,
    sigma_samples: float = 2.0,
    noise_floor: float = 0.10,
) -> np.ndarray:
    """Ideal GCC-PHAT stack for a point source at source_xyz (no real audio needed).

    For each mic pair (i, j) places a Gaussian-smoothed Dirac at the expected
    TDOA sample index, then adds a flat noise floor to simulate reverberation.
    Output shape (n_pairs, 2*max_delay_samples+1) is compatible with srp_phat_3d().

    Args:
        source_xyz:        (3,) source position in metres.
        mic_xyz:           (N, 3) microphone positions in metres.
        mic_pairs:         list of (i, j) index pairs — same convention as MIC_PAIRS_S2/S3.
        fs:                sample rate in Hz (e.g. 16000.0).
        c:                 wave speed in m/s (343.0 for air).
        max_delay_samples: GCC half-width; if None, derived from mic geometry.
        sigma_samples:     Gaussian std in samples — simulates window-induced peak spreading.
        noise_floor:       Flat additive background before row-normalisation.

    Returns:
        (n_pairs, L) float32 array, each row normalised to [0, 1].
    """
    if max_delay_samples is None:
        max_dist = max(
            float(np.linalg.norm(mic_xyz[i] - mic_xyz[j])) for i, j in mic_pairs
        )
        max_delay_samples = int(math.ceil(max_dist / c * fs))

    L = 2 * max_delay_samples + 1
    lags = np.arange(L, dtype=np.float64) - max_delay_samples
    gcc = np.empty((len(mic_pairs), L), dtype=np.float32)

    for k, (i, j) in enumerate(mic_pairs):
        di = float(np.linalg.norm(source_xyz - mic_xyz[i]))
        dj = float(np.linalg.norm(source_xyz - mic_xyz[j]))
        tdoa_samp = (di - dj) / c * fs

        if abs(tdoa_samp) > max_delay_samples:
            gcc[k] = 0.0
            continue

        row = np.exp(-0.5 * ((lags - tdoa_samp) / sigma_samples) ** 2) + noise_floor
        row_max = float(row.max())
        gcc[k] = (row / (row_max + 1e-6)).astype(np.float32)

    return gcc


# ==============================================================================
#  Covariance estimation helpers for information-form fusion
# ==============================================================================


def srp_covariance(srp_peak_power: float, scale_m: float = 0.15) -> np.ndarray:
    """Isotropic position covariance from SRP-PHAT peak power.

    High peak power → narrow SRP lobe → smaller uncertainty.
    σ = max(scale_m / sqrt(peak + ε), 0.15), Σ = σ² I₃.

    scale_m=0.15 and floor=0.15 m calibrated to observed SRP-PHAT errors of
    ~20 cm on bench-top datasets (S2 mean 23 cm, S3 21.6 cm).

    Args:
        srp_peak_power: SRP-PHAT peak value (higher = sharper, more certain).
        scale_m:        Baseline std-dev in metres at unit peak power.

    Returns:
        (3, 3) float64 diagonal covariance matrix.
    """
    sigma = scale_m / math.sqrt(max(float(srp_peak_power), 1e-6))
    sigma = max(sigma, 0.15)  # floor at 15 cm: bench-top SRP resolution limit
    return np.eye(3, dtype=np.float64) * (sigma ** 2)


def tdoa_covariance(
    residual_sum_sq: float,
    n_pairs: int,
    fs: float,
    c: float,
    scale: float = 4.0,
) -> np.ndarray:
    """Isotropic position covariance from TDOA non-linear LS residual.

    Converts per-pair time residuals (s²) to position uncertainty (m²):
    σ²_pos ≈ (c / fs)² × (residual_sum_sq / n_pairs) × scale.

    Args:
        residual_sum_sq: Summed squared TDOA residuals from the L-BFGS-B run.
        n_pairs:         Number of sensor pairs used in the TDOA fit.
        fs:              Sample rate (Hz) of the signal used for TDOA.
        c:               Wave speed used (m/s).
        scale:           Coverage factor (~4 for 95 % confidence, isotropic).

    Returns:
        (3, 3) float64 diagonal covariance matrix.
    """
    res_per_pair = residual_sum_sq / max(n_pairs, 1)
    sigma_sq = (c / fs) ** 2 * res_per_pair * scale
    # Floor at 10 cm: one acoustic wavelength at ~3.4 kHz.  GCC-PHAT peaks in
    # reverberant conditions carry systematic multipath bias of this order; a tiny
    # optimiser residual does not imply sub-centimetre position accuracy.
    sigma_sq = max(sigma_sq, (0.10) ** 2)
    return np.eye(3, dtype=np.float64) * sigma_sq


def neural_covariance(log_std: np.ndarray) -> np.ndarray:
    """Diagonal covariance from a neural network log-std output.

    Args:
        log_std: (3,) per-axis log standard deviation (from DualSRPNet).

    Returns:
        (3, 3) float64 diagonal covariance.
    """
    std = np.exp(np.clip(log_std.astype(np.float64), -6.0, 3.0))
    return np.diag(std ** 2)


def information_fusion(
    estimates: list[np.ndarray],
    covariances: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal inverse-covariance (information-form) fusion of position estimates.

    Combines independent estimates using the optimal linear unbiased estimator:

        p_fused = (Σᵢ Σᵢ⁻¹)⁻¹ Σᵢ Σᵢ⁻¹ pᵢ

    Replaces all hand-tuned weights; the contribution of each estimate is
    determined entirely by its covariance.  If a covariance is large (high
    uncertainty) the branch is automatically down-weighted.

    Args:
        estimates:   List of (3,) position vectors in metres.
        covariances: List of (3, 3) PSD covariance matrices.

    Returns:
        (fused_pos, fused_cov) — (3,) fused position, (3,3) fused covariance.
    """
    eps = np.eye(3, dtype=np.float64) * 1e-12
    info_sum = np.zeros((3, 3), dtype=np.float64)
    info_weighted = np.zeros(3, dtype=np.float64)

    for pos, cov in zip(estimates, covariances):
        cov_reg = cov.astype(np.float64) + eps
        info = np.linalg.solve(cov_reg, np.eye(3))
        info_sum += info
        info_weighted += info @ pos.astype(np.float64)

    fused_cov = np.linalg.solve(info_sum + eps, np.eye(3))
    fused_pos = fused_cov @ info_weighted
    return fused_pos.astype(np.float64), fused_cov.astype(np.float64)


# ==============================================================================
#  LocalizationDualSRPNet — Cross3D CNN with FiLM mode conditioning
# ==============================================================================
# The thesis contribution: a 3-D convolutional network that takes the stacked
# acoustic + structural SRP-PHAT power maps and is conditioned on the operating-
# mode context vector via FiLM (Feature-wise Linear Modulation).
#
# Motivation for FiLM conditioning:
#   Reverberation pattern, interference spectrum, and dominant structural wave
#   mode all shift between operating modes (Pump / Turbine / speed1 / speed2 …).
#   Without conditioning, the network must learn a single mapping valid across
#   all modes — impossible if the SRP bias is mode-dependent.  FiLM injects the
#   mode label as per-channel scale γ and shift β between convolutional blocks,
#   allowing the network to adapt its feature weighting to the current mode.
# ==============================================================================


class _FiLMGenerator(nn.Module):
    """Generates per-channel FiLM (γ, β) from an operating-mode vector.

    Args:
        n_modes:     Dimensionality of the one-hot mode input.
        in_channels: Number of feature-map channels to modulate.
        d_film:      Hidden size of the generator MLP.
    """

    def __init__(self, n_modes: int, in_channels: int, d_film: int = 32) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._net = nn.Sequential(
            nn.Linear(n_modes, d_film),
            nn.ReLU(inplace=True),
            nn.Linear(d_film, 2 * in_channels),
        )
        # Initialise output layer to zero → identity transform at start of training
        nn.init.zeros_(self._net[-1].weight)
        nn.init.zeros_(self._net[-1].bias)

    def forward(
        self, mode_vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mode_vec: (batch, n_modes).

        Returns:
            gamma: (batch, C, 1, 1, 1) — scale (initialised ≈ 1).
            beta:  (batch, C, 1, 1, 1) — shift (initialised = 0).
        """
        out = self._net(mode_vec)                         # (B, 2C)
        gamma = (1.0 + out[:, : self._in_channels]).view(
            -1, self._in_channels, 1, 1, 1
        )
        beta = out[:, self._in_channels :].view(-1, self._in_channels, 1, 1, 1)
        return gamma, beta


class LocalizationDualSRPNet(nn.Module):
    """Cross3D-style 3-D CNN localizer on stacked acoustic + structural SRP maps.

    Takes two SRP-PHAT power maps — one computed with c_air using microphone
    pairs, one computed with c_struct using accelerometer pairs — stacks them
    as a 2-channel 3-D tensor, normalises spatial resolution via adaptive
    pooling, then processes through FiLM-conditioned convolutional blocks.

    The network outputs both a position estimate AND a per-axis log standard
    deviation, enabling inverse-covariance fusion with other localization
    branches (SRP-PHAT, TDOA) without hand-tuned weights.

    Architecture:
        (2, Nx, Ny, Nz) → AdaptiveAvgPool3d(16,16,8)
        → Conv3D(2→16,3³)+BN+ReLU → FiLM₁(mode_vec)
        → Conv3D(16→32,3³,s=2)+BN+ReLU → FiLM₂(mode_vec)
        → Conv3D(32→32,3³)+BN+ReLU
        → GlobalAvgPool3D → Dropout
        → Linear→GELU→Dropout→Linear(→3)   [position head]
        → Linear(→3)                         [log-std head]

    Args:
        n_modes: Number of mode classes (e.g. 2 for Pump/Turbine, 3 for speeds).
        d_film:  Hidden dim of each FiLM MLP.
        dropout: Dropout probability in regression heads.
    """

    SPATIAL_TARGET: tuple[int, int, int] = (16, 16, 8)

    def __init__(
        self,
        n_modes: int,
        d_film: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_modes = n_modes

        self._adaptive_pool = nn.AdaptiveAvgPool3d(self.SPATIAL_TARGET)

        self._conv1 = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self._film1 = _FiLMGenerator(n_modes, in_channels=16, d_film=d_film)

        self._conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self._film2 = _FiLMGenerator(n_modes, in_channels=32, d_film=d_film)

        self._conv3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self._global_pool = nn.AdaptiveAvgPool3d(1)
        self._drop = nn.Dropout(p=dropout)

        self._pos_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 3),
        )
        self._unc_head = nn.Linear(32, 3)

    def forward(
        self,
        srp_ac: torch.Tensor,
        srp_str: torch.Tensor,
        mode_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            srp_ac:   (batch, Nx, Ny, Nz) — acoustic SRP-PHAT power map.
            srp_str:  (batch, Nx, Ny, Nz) — structural SRP-PHAT power map.
            mode_vec: (batch, n_modes) — one-hot operating-mode indicator.

        Returns:
            pos:     (batch, 3) — estimated position in metres.
            log_std: (batch, 3) — per-axis log standard deviation.
        """
        x = torch.stack([srp_ac, srp_str], dim=1)  # (B, 2, Nx, Ny, Nz)
        x = self._adaptive_pool(x)                   # (B, 2, 16, 16, 8)

        x = self._conv1(x)                           # (B, 16, 16, 16, 8)
        g1, b1 = self._film1(mode_vec)
        x = g1 * x + b1

        x = self._conv2(x)                           # (B, 32, 8, 8, 4)
        g2, b2 = self._film2(mode_vec)
        x = g2 * x + b2

        x = self._conv3(x)                           # (B, 32, 8, 8, 4)

        feat = self._global_pool(x).flatten(1)        # (B, 32)
        feat = self._drop(feat)

        pos = self._pos_head(feat)      # (B, 3)
        log_std = self._unc_head(feat)  # (B, 3)
        return pos, log_std


def dual_srp_localization_loss(
    pred_pos: torch.Tensor,
    pred_log_std: torch.Tensor,
    true_pos: torch.Tensor,
) -> torch.Tensor:
    """Heteroscedastic negative log-likelihood for the DualSRPNet.

    Trains both the position head and the uncertainty head simultaneously.
    Forces the network to be well-calibrated — inflating uncertainty to reduce
    loss is penalised by the 2·log_std term:

        L = 0.5 Σ_d [(p̂_d − p_d)² exp(−2 σ_d) + 2 σ_d]

    Args:
        pred_pos:     (batch, 3) predicted position in metres.
        pred_log_std: (batch, 3) predicted per-axis log std.
        true_pos:     (batch, 3) ground-truth position in metres.

    Returns:
        Scalar mean NLL.
    """
    log_std = pred_log_std.clamp(-6.0, 3.0)
    sq_err = (pred_pos - true_pos) ** 2                        # (B, 3)
    nll = 0.5 * (sq_err * torch.exp(-2.0 * log_std) + 2.0 * log_std)
    return nll.mean()
