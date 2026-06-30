"""Gated streaming-inference pipeline — V2 → V3 → (gated) V4.

Realises the streaming-flow contract — the runtime emits per window:

    (t, mode_label, anomaly_score, alert_flag, (x, y, z) | None)

with `mode_label` and `anomaly_score` always populated, `alert_flag` set when
the per-cluster threshold is exceeded, and the localization tuple computed
**only** on alerted windows.  This is the gated path; the cost/quality study
(`run_continuous`) is reported as Chapter 6 deployment-shape evidence.

`mode_label` is the V2 cluster index after Hungarian-mapping to a
human-readable folder label.  When no mapping has been provided (e.g., on
unlabeled future Illwerke streams), the pipeline emits the integer cluster
id and the consumer can apply its own mapping later.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ...config import resolve_device
from ...features.audio_spectral import compute_encoder_input_stack, compute_log_mel_spectrogram
from ...features.vibration_temporal import compute_vibration_input_stack
from ...ingestion.test_dataset_loader import TestDatasetSegment
from ..anomaly.cnf_head import ConditionalRealNVP
from ..anomaly.impulse_anchor import append_anchor
from ..anomaly.threshold import PerClusterThresholds
from ..context.v2_fusion import V2FusionEncoder
from ..context.v2_ssl import V2SSLConfig, _dataset_idx
from ..localization.v4_features import GridSpec, compute_accel_tdoa_tokens, compute_srp_phat_volume
from ..localization.v4_loc_head import V4LocalizationHead

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class StreamingDecision:
    """One window's emission from the gated pipeline."""

    t_start_s: float
    t_end_s: float
    cluster_id: int  # V2's predicted K-means cluster
    mode_label: str | None  # Hungarian-mapped folder label (None if no mapping)
    anomaly_score: float  # -log p(x | c)
    alert_flag: bool  # score > per-cluster threshold
    xyz: tuple[float, float, float] | None  # only on alert windows under gated mode
    runtime_ms: float = 0.0  # per-window wall-clock time

    def to_dict(self) -> dict:
        return {
            "t_start_s": self.t_start_s,
            "t_end_s": self.t_end_s,
            "cluster_id": int(self.cluster_id),
            "mode_label": self.mode_label,
            "anomaly_score": float(self.anomaly_score),
            "alert_flag": bool(self.alert_flag),
            "xyz": list(self.xyz) if self.xyz is not None else None,
            "runtime_ms": float(self.runtime_ms),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class GatedPipeline:
    """V2 + V3 + V4, glued together with the per-cluster threshold gate.

    Attributes
    ----------
    v2_encoder, flow, thresholds, v4_head:
        Trained models from the V2/V3/V4 trainers.  V4 is optional — the
        pipeline still emits `(mode, anomaly, alert)` if `v4_head is None`,
        with `xyz=None` even on alert windows.
    cluster_to_label:
        Hungarian mapping `{cluster_id: folder_label}`.  When `None`, the
        pipeline emits raw cluster IDs; consumer code may apply its own
        mapping later.
    grid:
        SRP-PHAT candidate grid for V4 inputs.
    v2_cfg:
        Window cadence + feature-extraction parameters (must match training).
    threshold_percentile:
        Which per-cluster threshold tier to use (95 or 99).
    """

    v2_encoder: V2FusionEncoder
    flow: ConditionalRealNVP
    thresholds: PerClusterThresholds
    v4_head: V4LocalizationHead | None
    grid: GridSpec
    v2_cfg: V2SSLConfig
    cluster_to_label: dict[int, str] | None = None
    threshold_percentile: int = 99
    unconditional_anomaly: bool = False  # A2 ablation runtime knob
    unconditional_localization: bool = False  # A3 ablation runtime knob
    device: torch.device = field(default_factory=lambda: resolve_device("auto"))
    # Trained `_XtPool` (PMA-2) from V3 — mirrors `V3Result.xt_pool`.
    # When None, `x_t` falls back to the legacy `fused.mean(dim=1)` (only
    # valid if the flow was trained with `xt_pool="mean"`).
    xt_pool: nn.Module | None = None
    # Impulse+spectral anchor standardization (healthy mean/std) from V3 when
    # `inject_impulse_anchor` was on — mirrors `V3Result.anchor_mean/anchor_std`.
    # When set, the standardized anchor is appended to `x_t` before scoring so the
    # flow input matches its trained dimension; None reproduces a no-anchor flow.
    anchor_norm: tuple[np.ndarray, np.ndarray] | None = None
    # Per-stage window override mirroring `V3Config.window_seconds_override`.
    # When set the streaming window cadence is overridden per-dataset; the
    # `v2_cfg` is not mutated.  Use a float for "single override for all
    # datasets" or `dict[dataset_id, float]` for per-dataset values.
    window_seconds_override: float | dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.v2_encoder = self.v2_encoder.to(self.device).eval()
        self.flow = self.flow.to(self.device).eval()
        if self.v4_head is not None:
            self.v4_head = self.v4_head.to(self.device).eval()
        if self.xt_pool is not None:
            self.xt_pool = self.xt_pool.to(self.device).eval()
        if self.threshold_percentile not in (95, 99):
            raise ValueError("threshold_percentile must be 95 or 99")

    def _window_seconds_for(self, dataset_id: str) -> float:
        """Return the effective window length for one dataset (override → cfg)."""
        ov = self.window_seconds_override
        if ov is None:
            return float(self.v2_cfg.window_seconds)
        if isinstance(ov, dict):
            return float(ov.get(dataset_id, self.v2_cfg.window_seconds))
        return float(ov)

    # ------------------------------------------------------------------ run
    @torch.no_grad()
    def run_segment(
        self,
        segment: TestDatasetSegment,
        *,
        gated: bool = True,
        scada_lookup: dict[str, np.ndarray] | None = None,
    ) -> list[StreamingDecision]:
        """Stream one segment window-by-window; emit a list of decisions.

        Parameters
        ----------
        gated : bool
            Default `True` — V4 runs only on alerted windows (deployment shape).
            `False` runs V4 on every window for the cost/quality study.
        scada_lookup : dict[recording_id, vec] | None
            Optional V5 SCADA tensor; passed to V4's `s` slot.  V4 must have
            been built with matching `scada_dim`.
        """
        cfg = self.v2_cfg

        # Pre-compute paired V2 features once per segment.
        ac_feats, vib_feats = self._compute_v2_features(segment)
        ac_fs = float(segment.segment.mic_sample_rate) / float(cfg.hop_length)
        vib_fs = float(segment.segment.accel_sample_rate)
        mic_fs_raw = int(segment.segment.mic_sample_rate)
        accel_fs_raw = int(segment.segment.accel_sample_rate)

        # Resolve the per-stage window override: V3 trains at a tighter
        # transient-scoped window (1.0 s on D3/D4, 3.0 s on D1/D2 in the
        # publication config); streaming must match for scores to be
        # comparable with the trained thresholds.  Stride keeps the legacy
        # ratio ``window_stride_seconds / window_seconds``.
        win_s = self._window_seconds_for(segment.dataset_id)
        stride_s = (
            win_s * (float(cfg.window_stride_seconds) / float(cfg.window_seconds))
            if cfg.window_seconds > 0
            else win_s * 0.5
        )
        n_ac = max(2, int(round(win_s * ac_fs)))
        stride_ac = max(1, int(round(stride_s * ac_fs)))
        n_vib = max(2, int(round(win_s * vib_fs)))
        n_mic_raw = max(8, int(round(win_s * mic_fs_raw)))
        n_acc_raw = max(2, int(round(win_s * accel_fs_raw)))

        T_ac = ac_feats.shape[-1]
        T_vib = vib_feats.shape[-1]
        T_mic = segment.segment.mic_data.shape[1]
        T_acc = segment.segment.accel_data.shape[1]

        ds_idx = torch.tensor(
            [_dataset_idx(segment.dataset_id)], dtype=torch.long, device=self.device
        )
        scada = None
        if scada_lookup is not None and segment.recording_id in scada_lookup:
            scada = torch.from_numpy(
                scada_lookup[segment.recording_id].astype(np.float32)
            ).unsqueeze(0).to(self.device)

        decisions: list[StreamingDecision] = []
        for start_ac in range(0, T_ac - n_ac + 1, stride_ac):
            t_start = start_ac / max(ac_fs, 1e-9)
            start_vib = int(round(t_start * vib_fs))
            start_mic = int(round(t_start * mic_fs_raw))
            start_acc = int(round(t_start * accel_fs_raw))
            if (
                start_vib + n_vib > T_vib
                or start_mic + n_mic_raw > T_mic
                or start_acc + n_acc_raw > T_acc
            ):
                continue

            t_window_start = time.perf_counter()

            # ── V2: c_t and the mean-pool x ──────────────────────────
            ac_win = (
                torch.from_numpy(np.ascontiguousarray(ac_feats[..., start_ac : start_ac + n_ac]))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            vib_win = (
                torch.from_numpy(np.ascontiguousarray(vib_feats[..., start_vib : start_vib + n_vib]))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            ac_xyz = torch.from_numpy(segment.mic_positions.astype(np.float32)).unsqueeze(0).to(self.device)
            vib_xyz = torch.from_numpy(segment.vib_positions.astype(np.float32)).unsqueeze(0).to(self.device)
            v2_out = self.v2_encoder(ac_win, ac_xyz, vib_win, vib_xyz, ds_idx, mask_p=0.0)
            fused = torch.cat([v2_out["a_fused"], v2_out["v_fused"]], dim=1)
            if self.xt_pool is not None:
                # PMA-2 path — mirrors `V3Result.xt_pool`; reproduces the
                # exact summary the flow was trained on.
                x_t = self.xt_pool(fused)
            else:
                # Legacy mean-pool fallback (`xt_pool="mean"` at train time).
                x_t = fused.mean(dim=1)
            c_t = v2_out["context"]

            # ── V3: anomaly score + per-cluster gate ─────────────────
            # Append the impulse+spectral anchor (RQ2) so the flow input matches
            # its trained dimension; no-op when anchor_norm is None.  Uses the
            # same windowed log-mel+CWT features the encoder consumed.
            c_for_flow = torch.zeros_like(c_t) if self.unconditional_anomaly else c_t
            x_for_flow = append_anchor(x_t, ac_win, vib_win, self.anchor_norm)
            score = float(self.flow.anomaly_score(x_for_flow, c_for_flow).item())

            c_np = c_t.squeeze(0).cpu().numpy()
            cluster_id = int(self.thresholds.assign(c_np[None, :])[0])
            thresh = (
                self.thresholds.p99[cluster_id]
                if self.threshold_percentile == 99
                else self.thresholds.p95[cluster_id]
            )
            alert_flag = bool(score > thresh)
            mode_label = (
                self.cluster_to_label.get(cluster_id) if self.cluster_to_label else None
            )

            xyz = None
            if self.v4_head is not None and (alert_flag or not gated):
                # Only compute the SRP-PHAT volume + accel TDOA when needed.
                mic_seg = segment.segment.mic_data[:, start_mic : start_mic + n_mic_raw]
                acc_seg = segment.segment.accel_data[:, start_acc : start_acc + n_acc_raw]
                volume = compute_srp_phat_volume(
                    mic_seg, segment.mic_positions, fs=mic_fs_raw, grid=self.grid
                )
                tdoa = compute_accel_tdoa_tokens(
                    acc_seg, segment.vib_positions, fs=accel_fs_raw
                )
                vol_t = torch.from_numpy(volume).unsqueeze(0).float().to(self.device)
                tdoa_t = (
                    torch.from_numpy(tdoa).unsqueeze(0).float().to(self.device)
                    if tdoa.shape[0] > 0
                    else torch.zeros(1, 1, 8, device=self.device)
                )
                pred = self.v4_head(
                    vol_t,
                    tdoa_t,
                    c_t,
                    scada,
                    unconditional=self.unconditional_localization,
                )
                xyz_arr = pred.squeeze(0).cpu().numpy()
                xyz = (float(xyz_arr[0]), float(xyz_arr[1]), float(xyz_arr[2]))

            t_window_end = time.perf_counter()

            decisions.append(
                StreamingDecision(
                    t_start_s=float(t_start),
                    # Use the effective per-dataset window length actually sliced
                    # (`win_s`), not the base `cfg.window_seconds`: under a
                    # per-stage window override (the publication config tightens
                    # V3/V4 to 1.0 s on D3/D4) the two differ, and using cfg here
                    # reported a window end inconsistent with the scored span.
                    t_end_s=float(t_start + win_s),
                    cluster_id=cluster_id,
                    mode_label=mode_label,
                    anomaly_score=score,
                    alert_flag=alert_flag,
                    xyz=xyz,
                    runtime_ms=float((t_window_end - t_window_start) * 1000.0),
                )
            )
        return decisions

    @torch.no_grad()
    def run_segments(
        self,
        segments: Iterable[TestDatasetSegment],
        *,
        gated: bool = True,
        scada_lookup: dict[str, np.ndarray] | None = None,
    ) -> list[StreamingDecision]:
        """Run multiple segments back-to-back.  Time axis resets per segment;
        consumers concatenating across segments must shift `t_start_s` themselves.
        """
        out: list[StreamingDecision] = []
        for s in segments:
            out.extend(self.run_segment(s, gated=gated, scada_lookup=scada_lookup))
        return out

    # ------------------------------------------------------ feature helper
    def _compute_v2_features(
        self, segment: TestDatasetSegment
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.v2_cfg
        if cfg.use_cwt:
            ac = compute_encoder_input_stack(
                segment.segment.mic_data,
                fs=int(segment.segment.mic_sample_rate),
                n_mels=cfg.n_mels,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                cwt_n_scales=cfg.cwt_n_scales,
            )
        else:
            mels = []
            for ch in range(segment.segment.n_mic_channels):
                m = compute_log_mel_spectrogram(
                    segment.segment.mic_data[ch],
                    fs=int(segment.segment.mic_sample_rate),
                    n_fft=cfg.n_fft,
                    hop_length=cfg.hop_length,
                    n_mels=cfg.n_mels,
                )
                mels.append(np.stack([m, m], axis=0).astype(np.float32))
            ac = np.stack(mels, axis=0)
        vib = compute_vibration_input_stack(
            segment.segment.accel_data,
            sample_rate=float(segment.segment.accel_sample_rate),
            kurtosis_window_seconds=cfg.vib_kurtosis_window_seconds,
            min_kurtosis_samples=cfg.vib_min_kurtosis_samples,
            crest_factor_window_seconds=cfg.vib_crest_factor_window_seconds,
            min_crest_factor_samples=cfg.vib_min_crest_factor_samples,
        )
        return ac.astype(np.float32), vib.astype(np.float32)


# ---------------------------------------------------------------------------
# Cost / quality study
# ---------------------------------------------------------------------------


@dataclass
class CostQualityReport:
    """One row of Chapter 6's gated-vs-continuous deployment-shape table."""

    n_windows: int
    n_alerts: int
    gated_total_ms: float
    continuous_total_ms: float
    gated_per_window_ms: float
    continuous_per_window_ms: float
    speedup_x: float

    def to_dict(self) -> dict:
        return {
            "n_windows": self.n_windows,
            "n_alerts": self.n_alerts,
            "gated_total_ms": self.gated_total_ms,
            "continuous_total_ms": self.continuous_total_ms,
            "gated_per_window_ms": self.gated_per_window_ms,
            "continuous_per_window_ms": self.continuous_per_window_ms,
            "speedup_x": self.speedup_x,
        }


def cost_quality_study(
    pipeline: GatedPipeline,
    segments: Iterable[TestDatasetSegment],
) -> CostQualityReport:
    """Run the same segments under gated and continuous modes; report speed-up."""
    segs = list(segments)
    gated = pipeline.run_segments(segs, gated=True)
    cont = pipeline.run_segments(segs, gated=False)
    n = max(len(gated), 1)
    gated_total = sum(d.runtime_ms for d in gated)
    cont_total = sum(d.runtime_ms for d in cont)
    return CostQualityReport(
        n_windows=len(gated),
        n_alerts=int(sum(d.alert_flag for d in gated)),
        gated_total_ms=float(gated_total),
        continuous_total_ms=float(cont_total),
        gated_per_window_ms=float(gated_total / n),
        continuous_per_window_ms=float(cont_total / n),
        speedup_x=float(cont_total / max(gated_total, 1e-9)),
    )


__all__ = [
    "CostQualityReport",
    "GatedPipeline",
    "StreamingDecision",
    "cost_quality_study",
]
