"""V4 trainer — supervised (x, y, z) regression on labeled anomaly windows.

Pipeline:
  1. **Precompute** V4 samples from labeled `TestDatasetSegment`s:
       - SRP-PHAT volume on a fixed grid (raw mic_data → `compute_srp_phat_volume`)
       - Structure-borne TDOA tokens (raw accel_data → `compute_accel_tdoa_tokens`)
       - V2 context vector `c_t` (V2 encoder forward on log-mel/CWT features)
       - Ground-truth `(x, y, z)` from the loader's `spatial_label`
  2. **Train** the V4 head with Smooth-L1 loss on `(x, y, z)`.  V2 is frozen
     by default; `cfg.unfreeze_v2_encoder=True` enables a fine-tune path.
  3. **Report** held-out 3-D Euclidean error (mean + 95th percentile).

Three knobs that matter for Chapter 6:
  - `cfg.unconditional=True` → A3 ablation (zero c at train+infer).
  - `cfg.scada_dim>0`        → SCADA injection slot for V5.1 (D3 speed one-hot)
                                and V5.2 (top-K Allg_M1).
  - `gated_inference`        → infer only on windows flagged anomalous by V3
                                (cost/quality study; a Chapter 6 deployment row).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ...config import describe_device, resolve_device
from ...config.architecture import V4_LOCALIZATION
from ...features.audio_spectral import compute_encoder_input_stack, compute_log_mel_spectrogram
from ...features.vibration_temporal import compute_vibration_input_stack
from ...ingestion.test_dataset_loader import TestDatasetSegment
from ..context.v2_fusion import V2FusionEncoder
from ..context.v2_ssl import V2SSLConfig, _dataset_idx
from ..early_stopping import EarlyStopping, cpu_state_dict
from ..eval.statistics import percentile_bootstrap_ci
from .v4_features import (
    GridSpec,
    compute_accel_tdoa_tokens,
    compute_burst_aware_srp_phat_volume,
    compute_srp_phat_volume,
)
from .v4_loc_head import V4LocalizationHead
from .v4_metrics import event_aggregated_mae

# ---------------------------------------------------------------------------
# Sample + config
# ---------------------------------------------------------------------------


@dataclass
class V4Sample:
    """One labeled anomaly window."""

    srp_volume: np.ndarray  # (Nx, Ny, Nz)
    tdoa_tokens: np.ndarray  # (n_pairs, 8); n_pairs may be 0
    context: np.ndarray  # (c_dim,) — V2 c_t (PMA pool of fused tokens)
    x_for_v3: np.ndarray  # (c_dim,) — mean-pool of fused tokens (V3 flow input)
    target_xyz: np.ndarray  # (3,) metres
    scada: np.ndarray | None  # (s_dim,) or None
    mode_label: str | None
    recording_id: str
    source_dir: str
    dataset_id: str  # so V3-gated cohort assembly can dispatch by source campaign
    # R3.3 / 2026-05-16 — classical accel-TDOA multilateration estimate for
    # this window, used by ``channel_mode="vibration_only_learned"`` as the
    # spatial init for the learned residual head (mirror of the acoustic
    # soft-argmax init).  ``None`` when accel multilateration was not
    # computed (e.g. < 4 accels, or the precompute path skipped it).
    multilat_xyz: np.ndarray | None = None
    # B3 (2026-05-23) — window-centre start time (s) within the recording, so
    # the V3-gated evaluation can match this sample to V3's detected event
    # intervals (V4 only "fires" on windows V3 flags anomalous in deployment).
    window_start_s: float = 0.0
    # Per-knock SRP peak sharpness (peak-to-average of the SRP volume), used as
    # a confidence weight when aggregating a position's knocks into one event
    # estimate.  Defaults to 1.0 (uniform) so legacy samples / the window
    # builder behave as before.
    srp_psr: float = 1.0


@dataclass(frozen=True)
class V4Config:
    """V4 training config."""

    # Head dims — sourced from `V4_LOCALIZATION` in architecture.py.
    cnn_feature_dim: int = V4_LOCALIZATION.cnn_feature_dim
    tdoa_feature_dim: int = V4_LOCALIZATION.tdoa_feature_dim
    hidden_dim: int = V4_LOCALIZATION.hidden_dim
    n_heads_tdoa: int = V4_LOCALIZATION.n_heads_tdoa

    # Heatmap soft-argmax + FiLM-residual head
    soft_argmax_temperature: float = V4_LOCALIZATION.soft_argmax_temperature
    residual_scale_m: float = V4_LOCALIZATION.residual_scale_m

    # Conditioning
    scada_dim: int = 0  # 0 → no SCADA slot; V5.1/V5.2 set this.
    unconditional: bool = False  # A3 ablation
    # Channel-ablation modes (the A5 localization ablation):
    #   - "both": full V4 architecture (acoustic SRP + structure-borne TDOA).
    #   - "srp_only": zero the TDOA tokens at inference and training, so the
    #     head must regress from the SRP volume + FiLM(c) alone.
    #   - "tdoa_only": zero the SRP volume at inference and training, so the
    #     head's soft-argmax starts at the grid centroid for every window
    #     and the localization comes entirely from the FiLM(c)-conditioned
    #     residual on the TDOA tokens.
    #   - "vibration_only_learned" (R3.3, 2026-05-16): zero the SRP volume
    #     AND replace the soft-argmax init with the per-sample classical
    #     accel-TDOA multilateration estimate (V4Sample.multilat_xyz).
    #     This is the structural "vibration-only learned" RQ3 baseline —
    #     the head sees a meaningful spatial prior (multilat output)
    #     instead of the grid-centroid collapse that tdoa_only suffers.
    # This ablation isolates the marginal contribution of the structure-
    # borne accelerometer TDOA pathway from the acoustic SRP pathway —
    # i.e. it answers "which modality is doing the localization work?"
    # at the V4-head input level, complementary to the A1 modality-severing
    # ablation that lives in V2.
    channel_mode: Literal[
        "both", "srp_only", "tdoa_only", "vibration_only_learned"
    ] = "both"

    # Per-stage window override (2026-05-19) — mirrors
    # `V3Config.window_seconds_override`.  When set, the V4 trainer
    # overrides `v2_cfg.window_seconds` for both feature-extraction and
    # raw-waveform window cadence inside `precompute_v4_samples`.
    # Publication default: 0.5 s on D3/D4 (4× SNR improvement on SRP-PHAT
    # cross-correlation peak detection — the knock occupies 50 ms / 500 ms
    # = 10 % of the integration window vs 2.5 % at 2 s), 1.5 s on D1/D2
    # (which cannot go below the 5 / 4 Hz Vibration1DCNN kernel constraint).
    # May be:
    #   * ``None`` — inherit `v2_cfg.window_seconds` (legacy),
    #   * a ``float`` — single override applied to every dataset,
    #   * a ``dict[dataset_id, float]`` — per-dataset override.
    # Default ``None`` keeps V4 backwards-compatible (inherits v2_cfg
    # window).  The orchestrator (full_run.v4_config) sets the publication
    # per-dataset dict from `WINDOWING.v4_window_seconds_override`.
    window_seconds_override: float | dict[str, float] | None = None

    # Training schedule — per-experiment, not centralised.
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.3
    seed: int = 42
    device: str = "auto"

    # Loss / training-time augmentation — implementation details.
    train_in_centimetres: bool = True
    smooth_l1_beta: float = 1.0
    target_pos_noise_m: float = 0.002      # ± 2 mm Gaussian on the GT
    srp_volume_noise_std: float = 0.02     # additive Gaussian on the SRP volume
    srp_volume_dropout_p: float = 0.0      # reserved for mic-dropout augmentation
    tdoa_jitter_m: float = 0.001           # ± 1 mm Gaussian on path_diff_m
    augment: bool = True

    # Head dropout — defends against the +1236 % train/val gap the audit
    # identified.  Threads into the residual MLP via V4LocalizationHead.
    # Default 0.0 keeps the dataclass byte-equivalent to pre-fix behaviour;
    # the orchestrator `v4_config` builder sets 0.1.
    head_dropout_p: float = 0.0

    # Heatmap auxiliary loss (integral-regression supervision, Sun et al. 2018).
    # When > 0, an auxiliary soft-cross-entropy pulls the SRP logit volume
    # toward a Gaussian centred on the GT voxel, giving the 3-D CNN a dense
    # per-voxel gradient instead of only the final-(x,y,z) Smooth-L1 signal —
    # sharpens the soft-argmax.  Default 0.0 keeps training byte-identical.
    # Only bites where the SRP volume is live (channel_mode "both"/"srp_only").
    heatmap_aux_weight: float = 0.0
    heatmap_sigma_m: float = 0.03  # Gaussian target width (~1.5 voxels)

    # Early stopping on val total loss.  Patience=5 (vs V1/V2's 3) because V4
    # has only ~10 labeled recordings and val loss is dramatically noisier.
    # Tensor-clone snapshot (not copy.deepcopy) — see V1 trainer for rationale.
    patience: int = 5
    restore_best: bool = True
    early_stop_min_delta: float = 1e-3


# ---------------------------------------------------------------------------
# Sample precomputation
# ---------------------------------------------------------------------------


def _window_v2_features(
    segment: TestDatasetSegment,
    cfg: V2SSLConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the V2 paired feature stacks for the segment."""
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


def precompute_v4_samples(
    v2_encoder: V2FusionEncoder,
    segments: list[TestDatasetSegment],
    *,
    v2_cfg: V2SSLConfig,
    grid: GridSpec,
    spatial_label_overrides: dict[str, tuple[float, float, float]] | None = None,
    scada_lookup: dict[str, np.ndarray] | None = None,
    window_seconds: float | dict[str, float] | None = None,
    window_stride_seconds: float | None = None,
    burst_aware_srp: bool = False,
    burst_seconds: float = 0.10,
    restrict_to_knock_intervals: bool = True,
    knock_intervals_override: dict[str, list[tuple[float, float]]] | None = None,
    v3_xt_pool: torch.nn.Module | None = None,
    v3_anchor_norm: tuple[np.ndarray, np.ndarray] | None = None,
    device: torch.device | str = "auto",
) -> list[V4Sample]:
    """Walk segments, slice labeled anomaly windows, build `V4Sample`s.

    ``v3_anchor_norm`` (healthy mean/std from V3 training) appends the
    standardized impulse+spectral anchor to ``x_for_v3`` so the V4 gate scores
    exactly the input the conditional V3 flow was trained on (RQ2 anchor
    injection).  Must match ``V3Result.anchor_mean/anchor_std``.

    - Spatial labels come from `segment.spatial_label` if present, with an
      optional `spatial_label_overrides` dict keyed by `recording_id` for
      smoke-test synthesis or for D3 `hit_between_*` recordings whose
      coordinates the loader reports as `None`.
    - SCADA values come from `scada_lookup[recording_id]` when provided
      (V5.1 D3 speed one-hot, V5.2 top-K Allg_M1).  Missing → no SCADA slot
      (the head is built without one).
    - Window cadence defaults to `v2_cfg.window_seconds` / `window_stride_seconds`
      so V2 features and SRP-PHAT/TDOA front-ends operate on the same window
      boundaries.  ``window_seconds`` may be a ``float`` (applied to every
      segment), a ``dict[dataset_id, float]`` (per-dataset, mirroring
      :attr:`V4Config.window_seconds_override`), or None (legacy default).
      The stride keeps a constant ratio to the window when an override is
      applied; this is the audio-SSL convention of equal-per-window
      training-data density across scales.
    """
    device = resolve_device(device)
    v2_encoder = v2_encoder.to(device).eval()
    for p in v2_encoder.parameters():
        p.requires_grad_(False)
    # `x_for_v3` is the input the V3 flow scores at gating time, so it must be
    # pooled the way V3 was trained.  When V3 used the learned PMA-2 pool
    # (`xt_pool="pma2"`), mean-pooling here yields an off-distribution `x` and
    # the flow's NLL saturates (~1e6) — the cause of the historical
    # n_holdout_gated=0/"flag everything" behaviour.  Pool with the V3 xt_pool
    # when supplied; otherwise fall back to the legacy mean (V3 `xt_pool="mean"`).
    if v3_xt_pool is not None:
        v3_xt_pool = v3_xt_pool.to(device).eval()

    def _per_segment_window_s(dataset_id: str) -> float:
        if window_seconds is None:
            return float(v2_cfg.window_seconds)
        if isinstance(window_seconds, dict):
            return float(window_seconds.get(dataset_id, v2_cfg.window_seconds))
        return float(window_seconds)

    base_stride_ratio = (
        float(v2_cfg.window_stride_seconds) / float(v2_cfg.window_seconds)
        if v2_cfg.window_seconds > 0
        else 0.5
    )
    if window_stride_seconds is not None:
        # Caller pinned an absolute stride; honour it directly (only sensible
        # when `window_seconds` is also a scalar).
        legacy_stride = float(window_stride_seconds)
    else:
        legacy_stride = None

    overrides = spatial_label_overrides or {}

    samples: list[V4Sample] = []
    for s in segments:
        win_s = _per_segment_window_s(s.dataset_id)
        if legacy_stride is not None:
            stride_s = legacy_stride
        else:
            stride_s = win_s * base_stride_ratio
        spatial = overrides.get(s.recording_id, s.spatial_label)
        if spatial is None:
            continue  # no ground-truth → skip (V4 needs labels)
        target = np.asarray(spatial, dtype=np.float32)

        scada = None
        if scada_lookup is not None and s.recording_id in scada_lookup:
            scada = np.asarray(scada_lookup[s.recording_id], dtype=np.float32)

        # Pre-compute V2 paired features once per segment.
        ac_feats, vib_feats = _window_v2_features(s, v2_cfg)

        ac_fs = float(s.segment.mic_sample_rate) / float(v2_cfg.hop_length)
        vib_fs = float(s.segment.accel_sample_rate)

        # Window counts (V2 frame-aligned).
        n_ac = max(2, int(round(win_s * ac_fs)))
        stride_ac = max(1, int(round(stride_s * ac_fs)))
        n_vib = max(2, int(round(win_s * vib_fs)))

        # Raw waveform window (used for SRP-PHAT + TDOA).
        mic_fs_raw = int(s.segment.mic_sample_rate)
        accel_fs_raw = int(s.segment.accel_sample_rate)
        n_mic_raw = max(8, int(round(win_s * mic_fs_raw)))
        n_acc_raw = max(2, int(round(win_s * accel_fs_raw)))

        T_ac = ac_feats.shape[-1]
        T_vib = vib_feats.shape[-1]
        T_mic = s.segment.mic_data.shape[1]
        T_acc = s.segment.accel_data.shape[1]

        if T_ac < n_ac or T_vib < n_vib or T_mic < n_mic_raw or T_acc < n_acc_raw:
            continue

        # B2 (2026-05-23) — sparse-anomaly window restriction.  RandomFault
        # recordings carry the knock position on every window even though the
        # knock occupies only a sparse sub-span; training V4 to localise the
        # knock on healthy windows is label noise.  Derive weak knock
        # intervals from the impulse envelope and keep only windows that
        # overlap one.  Empty derivation (no impulse found) → keep all
        # windows (conservative fallback = prior behaviour).  Continuous-
        # anomaly recordings (D2 RandomFault, D3 hit, D5 knock) naturally
        # retain most windows because their bursts cover the span.
        knock_intervals: list[tuple[float, float]] = []
        if restrict_to_knock_intervals:
            if knock_intervals_override is not None:
                # V3-gated (or any externally-supplied) training-window
                # selection: use the provided per-recording intervals instead
                # of the impulse-envelope weak GT.  Empty list for a recording
                # means V3 flagged NOTHING there → drop the whole recording
                # (not "keep all" — that is only the impulse-path fallback).
                knock_intervals = knock_intervals_override.get(s.recording_id, [])
                if not knock_intervals:
                    continue
            else:
                try:
                    from ..anomaly.weak_labels import derive_knock_events
                    knock_intervals = derive_knock_events(s, burst_seconds=burst_seconds)
                except (ValueError, RuntimeError, FloatingPointError):
                    # Expected weak-label failures (no impulse, degenerate
                    # envelope) → conservative fallback "keep all windows".
                    # Narrowed from `except Exception` so a genuine bug
                    # (shape/typing error) surfaces instead of silently
                    # disabling the sparse-anomaly window restriction.
                    knock_intervals = []

        ds_idx = torch.tensor(
            [_dataset_idx(s.dataset_id)], dtype=torch.long, device=device
        )

        for start_ac in range(0, T_ac - n_ac + 1, stride_ac):
            t_start = start_ac / max(ac_fs, 1e-9)
            # Skip windows that don't overlap a selected interval.  Only active
            # when intervals were found; impulse-path empty list = keep all
            # (fallback), V3-gated empty already `continue`d the recording above.
            if knock_intervals:
                from ..anomaly.weak_labels import window_overlaps_any
                if not window_overlaps_any(t_start, t_start + win_s, knock_intervals):
                    continue
            start_vib = int(round(t_start * vib_fs))
            if start_vib + n_vib > T_vib:
                continue
            start_mic = int(round(t_start * mic_fs_raw))
            start_acc = int(round(t_start * accel_fs_raw))
            if start_mic + n_mic_raw > T_mic or start_acc + n_acc_raw > T_acc:
                continue

            # V2 forward → c_t
            ac_win = torch.from_numpy(
                np.ascontiguousarray(ac_feats[..., start_ac : start_ac + n_ac])
            ).unsqueeze(0).float().to(device)
            vib_win = torch.from_numpy(
                np.ascontiguousarray(vib_feats[..., start_vib : start_vib + n_vib])
            ).unsqueeze(0).float().to(device)
            ac_xyz = torch.from_numpy(s.mic_positions.astype(np.float32)).unsqueeze(0).to(device)
            vib_xyz = torch.from_numpy(s.vib_positions.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = v2_encoder(ac_win, ac_xyz, vib_win, vib_xyz, ds_idx, mask_p=0.0)
            c_t = out["context"].squeeze(0).cpu().numpy().astype(np.float32)
            # V3's flow input, pooled exactly as V3 consumes it during training
            # (PMA-2 when supplied, else mean — see `_extract_xc`).  Cached on
            # the V4Sample so V3-gated cohort assembly doesn't re-run the encoder.
            fused = torch.cat([out["a_fused"], out["v_fused"]], dim=1)
            if v3_xt_pool is not None:
                with torch.no_grad():
                    x_t = v3_xt_pool(fused)
            else:
                x_t = fused.mean(dim=1)
            x_for_v3 = x_t.squeeze(0).cpu().numpy().astype(np.float32)
            # RQ2 anchor: append the standardized impulse+spectral anchor so the
            # V4 gate scores the same input the conditional V3 flow trained on.
            if v3_anchor_norm is not None:
                from ..anomaly.impulse_anchor import append_anchor
                x_for_v3 = append_anchor(
                    x_for_v3[None, :], ac_win, vib_win, v3_anchor_norm
                )[0].astype(np.float32)

            # Front-end features on the raw waveform window.  Burst-aware
            # SRP crops to the highest-energy ~100 ms sub-window before
            # computing GCC-PHAT — sharper SRP peak on sparsely-anomalous
            # recordings (D4 RandomFault knocks), no worse on continuously-
            # anomalous recordings (D2 RandomFault, D3 hit).
            mic_seg = s.segment.mic_data[:, start_mic : start_mic + n_mic_raw]
            acc_seg = s.segment.accel_data[:, start_acc : start_acc + n_acc_raw]
            if burst_aware_srp:
                volume = compute_burst_aware_srp_phat_volume(
                    mic_seg, s.mic_positions, fs=mic_fs_raw, grid=grid,
                    burst_seconds=burst_seconds,
                )
            else:
                volume = compute_srp_phat_volume(
                    mic_seg, s.mic_positions, fs=mic_fs_raw, grid=grid,
                )
            tdoa = compute_accel_tdoa_tokens(
                acc_seg, s.vib_positions, fs=accel_fs_raw
            )
            # R3.3 — also run the classical accel-TDOA multilateration so
            # the `channel_mode="vibration_only_learned"` head has a
            # spatial init.  Skips when < 4 accels (solver requirement).
            # Cheap (a few L-BFGS-B calls per window) and unused by other
            # channel_modes, so the overhead is acceptable across the
            # entire V4 cohort precompute.
            multilat_xyz: np.ndarray | None = None
            if acc_seg.shape[0] >= 4:
                try:
                    from .multilateration import accel_tdoa_multilateration_v0
                    pos, _residual = accel_tdoa_multilateration_v0(
                        acc_seg, s.vib_positions, fs=accel_fs_raw,
                    )
                    multilat_xyz = pos.astype(np.float32)
                except (np.linalg.LinAlgError, ValueError, RuntimeError):
                    # Expected solver failures (singular geometry, non-convergent
                    # L-BFGS-B, too few usable pairs) → no spatial init for the
                    # vibration-only-learned head on this window.  Narrowed from
                    # `except Exception` so an unexpected fault is not masked as
                    # "best-effort skipped".
                    multilat_xyz = None

            samples.append(
                V4Sample(
                    srp_volume=volume,
                    tdoa_tokens=tdoa,
                    context=c_t,
                    x_for_v3=x_for_v3,
                    target_xyz=target,
                    scada=scada,
                    mode_label=s.mode_label,
                    recording_id=s.recording_id,
                    source_dir=str(s.source_dir),
                    dataset_id=s.dataset_id,
                    multilat_xyz=multilat_xyz,
                    window_start_s=float(t_start),
                )
            )
    return samples


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------


def _split_samples_by_recording(
    samples: list[V4Sample], val_ratio: float, seed: int
) -> tuple[list[V4Sample], list[V4Sample]]:
    rng = np.random.default_rng(seed)
    keys = sorted({(s.source_dir, s.recording_id) for s in samples})
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_ratio)))
    val_keys = set(keys[:n_val])
    train_keys = set(keys[n_val:])
    if not train_keys:
        train_keys = {val_keys.pop()}
    train = [s for s in samples if (s.source_dir, s.recording_id) in train_keys]
    val = [s for s in samples if (s.source_dir, s.recording_id) in val_keys]
    return train, val


def _position_key(xyz, ndigits: int = 3) -> tuple[float, float, float]:
    """Round a target position to a hashable cm-grid key for spatial splits."""
    a = np.asarray(xyz, dtype=np.float64).ravel()
    return (round(float(a[0]), ndigits), round(float(a[1]), ndigits), round(float(a[2]), ndigits))


def split_samples_by_dataset(
    samples: list[V4Sample],
    holdout_dataset_ids: set[str] | list[str] | tuple[str, ...],
) -> tuple[list[V4Sample], list[V4Sample]]:
    """Split V4 samples into (train, holdout) by ``dataset_id``.

    The held-out dataset IDs (e.g. ``{"d5"}``) never appear in training, so
    holdout MAE measures cross-session transfer — does V4 trained on the
    older sessions (D1-D4) still localize knocks on a newly collected
    session (D5)?  See Phase 5 of the deep campaign.
    """
    hold = {str(d) for d in holdout_dataset_ids}
    train = [s for s in samples if s.dataset_id not in hold]
    holdout = [s for s in samples if s.dataset_id in hold]
    return train, holdout


def split_samples_by_position(
    samples: list[V4Sample],
    holdout_positions: list[tuple[float, float, float]],
    *,
    ndigits: int = 3,
) -> tuple[list[V4Sample], list[V4Sample]]:
    """Split V4 samples into (train, holdout) by TARGET POSITION.

    The held-out positions never appear in training, so the holdout MAE
    measures true "localise an unseen position" generalisation — the correct
    metric for localization, unlike the random-window split which leaks a
    position's other windows into training.

    ``holdout_positions`` are matched on the rounded cm grid (``ndigits``),
    so callers can pass positions in metres exactly as parsed from the folder
    names (e.g. ``(0.22, 0.0, 0.0)`` for the ``(22, 0, 0)`` cm folder).
    """
    hold_keys = {_position_key(p, ndigits) for p in holdout_positions}
    train = [s for s in samples if _position_key(s.target_xyz, ndigits) not in hold_keys]
    holdout = [s for s in samples if _position_key(s.target_xyz, ndigits) in hold_keys]
    return train, holdout


def _stack_tdoa(samples: list[V4Sample]) -> torch.Tensor:
    """Pad TDOA token sequences to the batch's max `n_pairs` with zeros."""
    n_max = max((s.tdoa_tokens.shape[0] for s in samples), default=0)
    out = np.zeros((len(samples), max(n_max, 1), 8), dtype=np.float32)
    if n_max > 0:
        for i, s in enumerate(samples):
            n = s.tdoa_tokens.shape[0]
            if n > 0:
                out[i, :n] = s.tdoa_tokens
    return torch.from_numpy(out)


def _make_batch(
    samples: list[V4Sample],
    *,
    channel_mode: Literal[
        "both", "srp_only", "tdoa_only", "vibration_only_learned"
    ] = "both",
) -> dict:
    volumes = torch.from_numpy(np.stack([s.srp_volume for s in samples], axis=0)).float()
    tdoa = _stack_tdoa(samples).float()
    external_init_xyz: torch.Tensor | None = None
    if channel_mode == "srp_only":
        # Severing the structure-borne TDOA pathway: zero the token
        # features (path_diff_m + endpoint coords + distance) so the
        # TDOA-set encoder reads a no-information input.  The encoder's
        # PMA output is still computed (so the head's input dim is
        # preserved), but it carries no spatial information.
        tdoa = torch.zeros_like(tdoa)
    elif channel_mode == "tdoa_only":
        # Severing the acoustic-SRP pathway: zero the SRP volume so the
        # 3-D CNN sees a flat-zero input.  The soft-argmax over a flat
        # logit volume collapses to the uniform centroid of the grid →
        # init_xyz is the grid centre on every window, and any
        # localization signal must come through the FiLM-conditioned
        # residual on (global_feat, tdoa_feat, init_xyz).
        volumes = torch.zeros_like(volumes)
    elif channel_mode == "vibration_only_learned":
        # R3.3 — same SRP-severing as tdoa_only, plus replace the
        # (meaningless) grid-centroid soft-argmax init with the per-
        # sample classical multilateration estimate.  Requires every
        # V4Sample to have multilat_xyz populated by precompute.
        volumes = torch.zeros_like(volumes)
        if any(s.multilat_xyz is None for s in samples):
            missing = [s.recording_id for s in samples if s.multilat_xyz is None]
            raise ValueError(
                f"channel_mode='vibration_only_learned' requires every V4Sample "
                f"to have multilat_xyz; {len(missing)} samples missing it "
                f"(first: {missing[0]!r})"
            )
        external_init_xyz = torch.from_numpy(
            np.stack([s.multilat_xyz for s in samples], axis=0)
        ).float()
    contexts = torch.from_numpy(np.stack([s.context for s in samples], axis=0)).float()
    targets = torch.from_numpy(np.stack([s.target_xyz for s in samples], axis=0)).float()
    scada = None
    if all(s.scada is not None for s in samples) and samples:
        scada = torch.from_numpy(np.stack([s.scada for s in samples], axis=0)).float()
    return {
        "volumes": volumes,
        "tdoa": tdoa,
        "contexts": contexts,
        "targets": targets,
        "scada": scada,
        "external_init_xyz": external_init_xyz,
    }


@dataclass
class V4Result:
    head: V4LocalizationHead
    train_loss_history: list[float]
    val_loss_history: list[float]
    # Headline MAE is EVENT-AGGREGATED (deployment-faithful): one estimate per
    # recording = mean of its per-knock predictions, error against the GT, then
    # meaned over recordings (computed via `event_aggregated_mae`).  Strictly NaN
    # when there are no val groups — never falls back to a per-window value.
    # `val_predictions`/`val_targets` remain per-knock so component analysis and
    # the window-level bootstrap CI still see every knock.
    val_mae_3d: float
    val_p95_3d: float
    val_predictions: np.ndarray  # (n_val, 3)
    val_targets: np.ndarray  # (n_val, 3)
    train_recording_ids: list[str]
    val_recording_ids: list[str]
    unconditional: bool
    # Diagnostics for the Chapter 6 analysis.
    val_init_xyz: np.ndarray  # (n_val, 3) — pure soft-argmax output
    val_residuals: np.ndarray  # (n_val, 3) — FiLM-residual contribution
    val_recording_breakdown: dict  # recording_id -> {n, mae, target, pred_mean}
    # Bootstrap 95 % CI on val MAE — percentile method, 1000 resamples at the
    # RECORDING level (block / cluster bootstrap).  Per-knock windows from one
    # recording are correlated, so resampling whole recordings rather than
    # individual windows avoids pseudoreplication and gives an honest interval
    # (Davison & Hinkley 1997 §3.8).  With a single val recording the CI is
    # degenerate by design (no between-recording information to resample).
    val_mae_ci_low: float = float("nan")
    val_mae_ci_high: float = float("nan")
    val_mae_ci_method: str = "recording_block_percentile_bootstrap_1000"
    # Number of distinct val recordings the block bootstrap resampled (the CI's
    # true independent-unit count).  None when no val windows were scored.
    val_mae_ci_n_groups: int | None = None
    # Per-window recording id (``source_dir_name/recording_id``), aligned to
    # ``val_predictions`` / ``val_targets``.  Lets downstream paired tests
    # (e.g. V4 vs A3 in full_run) resample at the recording level instead of
    # the window level — see `eval.statistics.paired_bootstrap_test(groups=)`.
    val_groups: list[str] = field(default_factory=list)
    early_stopped_epoch: int | None = None
    best_val_loss: float = float("nan")
    # Train MAE/P95 in metres (3-D Euclidean) — computed via a final forward
    # pass on train_samples after restoring the best head.  Enables a proper
    # metres-scale generalization gap `val_mae_3d - train_mae_3d`.  The
    # `train_loss_history[-1]` values are smooth-L1 in the loss_scale-cm
    # space and are not in metres, despite a legacy field of that name in
    # the deep-sweep metrics.json.
    train_mae_3d: float = float("nan")
    train_p95_3d: float = float("nan")
    # Per-position MAE breakdown (keyed by `_position_key(target_xyz)`).
    # Parallel to `val_recording_breakdown` but groups windows by their
    # spatial target instead of their recording, exposing position-level
    # error for failure-mode analysis (Table 12 in analyze_ablation).
    val_position_breakdown: dict = field(default_factory=dict)
    # Per-recording aggregated errors (||mean(recording's per-knock preds) − GT||),
    # the samples behind the headline; drivers bootstrap a CI over these.
    val_agg_errors: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    val_agg_method: str = "mean_per_recording"


def _grid_coords_from_spec(grid: GridSpec) -> torch.Tensor:
    """Build the `(Nx, Ny, Nz, 3)` voxel-centre tensor from a `GridSpec`."""
    ax_x, ax_y, ax_z = grid.axes()
    xx, yy, zz = np.meshgrid(ax_x, ax_y, ax_z, indexing="ij")
    coords = np.stack([xx, yy, zz], axis=-1).astype(np.float32)  # (Nx, Ny, Nz, 3)
    return torch.from_numpy(coords)


def _heatmap_aux_loss(
    logits: torch.Tensor,
    grid_coords_flat: torch.Tensor,
    targets_xyz: torch.Tensor,
    sigma_m: float,
) -> torch.Tensor:
    """Soft cross-entropy between the SRP logit volume and a Gaussian GT target.

    The target is a softmax-normalised Gaussian over voxels centred on each
    sample's GT position (integral-regression supervision, Sun et al. 2018).
    Returns a scalar; the caller weights it by `cfg.heatmap_aux_weight`.
    """
    B = logits.shape[0]
    flat_logits = logits.reshape(B, -1)  # (B, V)
    coords = grid_coords_flat.to(logits.device).to(logits.dtype)  # (V, 3)
    # (B, V) squared distances from each voxel to each sample's GT.
    d2 = ((coords[None, :, :] - targets_xyz[:, None, :]) ** 2).sum(dim=-1)
    target = F.softmax(-d2 / (2.0 * float(sigma_m) ** 2), dim=-1)  # (B, V)
    log_pred = F.log_softmax(flat_logits, dim=-1)
    return -(target * log_pred).sum(dim=-1).mean()


def train_v4_localization(
    samples: list[V4Sample],
    cfg: V4Config | None = None,
    *,
    grid: GridSpec,
    explicit_split: tuple[list[V4Sample], list[V4Sample]] | None = None,
    init_state: dict | None = None,
) -> V4Result:
    """Train the V4 head supervised on labeled anomaly windows.

    `grid` must match the `GridSpec` used at sample-precompute time —
    the head's soft-argmax operates on its voxel centres, so a mismatch
    silently corrupts the regression targets.

    `explicit_split` — when provided as ``(train_samples, val_samples)``,
    bypasses the internal random recording-split.  Used by the spatial-
    holdout evaluation (train on most positions, validate on the held-out
    positions) so the reported MAE measures localise-an-unseen-position
    generalisation rather than within-position interpolation.

    `init_state` — optional ``state_dict`` to warm-start the head from (loaded
    with ``strict=False``).  Used by the synthetic-knock pretraining ablation:
    pretrain the geometry on simulated knocks, then fine-tune on the real
    cohort.  ``None`` keeps the standard from-scratch init.
    """
    cfg = cfg or V4Config()
    if not samples and explicit_split is None:
        raise RuntimeError("V4: no labeled anomaly samples")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"V4: device={describe_device(device)}")

    if explicit_split is not None:
        train_samples, val_samples = explicit_split
        if not train_samples or not val_samples:
            raise RuntimeError(
                f"V4 explicit_split needs non-empty train+val; got "
                f"{len(train_samples)} train / {len(val_samples)} val"
            )
    else:
        train_samples, val_samples = _split_samples_by_recording(samples, cfg.val_ratio, cfg.seed)

    c_dim = int(train_samples[0].context.shape[0])
    s_dim = cfg.scada_dim
    grid_coords = _grid_coords_from_spec(grid)
    head = V4LocalizationHead(
        grid_coords=grid_coords,
        cnn_feature_dim=cfg.cnn_feature_dim,
        tdoa_feature_dim=cfg.tdoa_feature_dim,
        c_dim=c_dim,
        s_dim=s_dim,
        hidden_dim=cfg.hidden_dim,
        n_heads_tdoa=cfg.n_heads_tdoa,
        residual_scale_m=cfg.residual_scale_m,
        soft_argmax_temperature=cfg.soft_argmax_temperature,
        head_dropout_p=cfg.head_dropout_p,
    ).to(device)
    if init_state is not None:
        # Warm-start from a pretrained head (e.g. synthetic-knock pretraining).
        # strict=False tolerates a c_dim/grid mismatch by skipping those keys.
        head.load_state_dict(init_state, strict=False)
    # Flattened voxel coords for the optional heatmap auxiliary loss.
    heatmap_coords_flat = (
        grid_coords.reshape(-1, 3).to(device) if cfg.heatmap_aux_weight > 0 else None
    )
    optim = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Cosine LR schedule — small head + small labeled pool benefits from a
    # smooth lr decay rather than fixed-rate AdamW.  T_max = cfg.epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, int(cfg.epochs)), eta_min=cfg.lr * 0.01
    )

    n_train = len(train_samples)
    train_history: list[float] = []
    val_history: list[float] = []

    def _to_device(batch: dict) -> dict:
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    loss_scale = 100.0 if cfg.train_in_centimetres else 1.0  # 1 m → 100 cm

    def _augment_batch(batch: dict, gen: torch.Generator) -> dict:
        """Apply zero-mean noise to volumes / TDOA tokens / targets in place.

        Volume noise mimics SRP estimator variance from waveform jitter
        without re-running SRP.  TDOA-token noise (on the path-difference
        column only) mimics structure-borne speed uncertainty.  Target
        noise (≤ 2 mm) regularises the regression head against the small
        spatial-label inventory (~ 6 recordings) and is well below the
        physical position-measurement uncertainty.
        """
        if not cfg.augment:
            return batch
        vol = batch["volumes"]
        if cfg.srp_volume_noise_std > 0:
            noise = torch.randn(vol.shape, generator=gen, device=vol.device, dtype=vol.dtype)
            vol = (vol + cfg.srp_volume_noise_std * noise).clamp_min(0.0)
            batch["volumes"] = vol
        if cfg.tdoa_jitter_m > 0 and batch["tdoa"].numel() > 0:
            tdoa = batch["tdoa"].clone()
            jitter = torch.randn(
                tdoa.shape[0], tdoa.shape[1],
                generator=gen, device=tdoa.device, dtype=tdoa.dtype,
            ) * cfg.tdoa_jitter_m
            # Only the first feature is path_diff_m — leave positions / dist alone.
            tdoa[..., 0] = tdoa[..., 0] + jitter
            batch["tdoa"] = tdoa
        if cfg.target_pos_noise_m > 0:
            t = batch["targets"]
            tnoise = torch.randn(t.shape, generator=gen, device=t.device, dtype=t.dtype)
            batch["targets"] = t + cfg.target_pos_noise_m * tnoise
        return batch

    aug_gen = torch.Generator(device="cpu")
    aug_gen.manual_seed(cfg.seed)

    # Early-stop bookkeeping (see ``modeling.early_stopping``).  V4 has only
    # ~10 labeled recordings so val loss is dramatically noisier than V1/V2's;
    # patience=5 + min_delta=1e-3 reflect that.
    stopper = EarlyStopping(cfg.patience, cfg.early_stop_min_delta,
                            initial=cpu_state_dict(head))
    early_stopped_epoch: int | None = None

    suffix = "unconditional" if cfg.unconditional else f"scada_dim={cfg.scada_dim}"
    epoch_iter = tqdm(
        range(cfg.epochs),
        desc=f"V4 head ({suffix})",
        unit="epoch",
        leave=False,
    )
    for _epoch in epoch_iter:
        head.train()
        perm = list(range(n_train))
        np.random.shuffle(perm)
        loss_sum = 0.0
        n = 0
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            batch = _to_device(
                _augment_batch(_make_batch([train_samples[j] for j in idx], channel_mode=cfg.channel_mode), aug_gen)
            )
            if cfg.heatmap_aux_weight > 0:
                out = head(
                    batch["volumes"],
                    batch["tdoa"],
                    batch["contexts"],
                    batch["scada"],
                    unconditional=cfg.unconditional,
                    external_init_xyz=batch.get("external_init_xyz"),
                    return_components=True,
                )
                pred = out["pred"]
            else:
                pred = head(
                    batch["volumes"],
                    batch["tdoa"],
                    batch["contexts"],
                    batch["scada"],
                    unconditional=cfg.unconditional,
                    external_init_xyz=batch.get("external_init_xyz"),
                )
            loss = F.smooth_l1_loss(
                pred * loss_scale, batch["targets"] * loss_scale, beta=cfg.smooth_l1_beta
            )
            if cfg.heatmap_aux_weight > 0 and heatmap_coords_flat is not None:
                loss = loss + cfg.heatmap_aux_weight * _heatmap_aux_loss(
                    out["logits"], heatmap_coords_flat, batch["targets"], cfg.heatmap_sigma_m
                )
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += float(loss.item()) * pred.shape[0]
            n += pred.shape[0]
        scheduler.step()
        train_history.append(loss_sum / max(1, n))

        head.eval()
        v_loss = 0.0
        v_n = 0
        with torch.no_grad():
            for i in range(0, len(val_samples), cfg.batch_size):
                batch = _to_device(_make_batch(val_samples[i : i + cfg.batch_size], channel_mode=cfg.channel_mode))
                pred = head(
                    batch["volumes"],
                    batch["tdoa"],
                    batch["contexts"],
                    batch["scada"],
                    unconditional=cfg.unconditional,
                    external_init_xyz=batch.get("external_init_xyz"),
                )
                loss = F.smooth_l1_loss(
                    pred * loss_scale, batch["targets"] * loss_scale, beta=cfg.smooth_l1_beta
                )
                v_loss += float(loss.item()) * pred.shape[0]
                v_n += pred.shape[0]
        val_history.append(v_loss / max(1, v_n))
        epoch_iter.set_postfix(
            train=f"{train_history[-1]:.4f}", val=f"{val_history[-1]:.4f}"
        )

        # Early stop on val Smooth-L1.  Snapshot the head every improvement.
        cur_val = val_history[-1]
        if stopper.update(cur_val, lambda: cpu_state_dict(head)):
            early_stopped_epoch = _epoch + 1
            break

    best_val_loss = stopper.best
    if cfg.restore_best:
        head.load_state_dict(stopper.best_snapshot)
    del stopper

    # Final val errors + per-recording diagnostic.  `return_components=True`
    # exposes (init_xyz, delta, pred) so Chapter 6 can attribute MAE to the
    # SRP-soft-argmax prior vs the FiLM-conditioned residual.
    head.eval()
    val_preds: list[np.ndarray] = []
    val_inits: list[np.ndarray] = []
    val_deltas: list[np.ndarray] = []
    val_tgts: list[np.ndarray] = []
    val_keys: list[str] = []
    with torch.no_grad():
        for i in range(0, len(val_samples), cfg.batch_size):
            batch_samples = val_samples[i : i + cfg.batch_size]
            batch = _to_device(_make_batch(batch_samples, channel_mode=cfg.channel_mode))
            out = head(
                batch["volumes"],
                batch["tdoa"],
                batch["contexts"],
                batch["scada"],
                unconditional=cfg.unconditional,
                return_components=True,
                external_init_xyz=batch.get("external_init_xyz"),
            )
            val_preds.append(out["pred"].cpu().numpy())
            val_inits.append(out["init_xyz"].cpu().numpy())
            val_deltas.append(out["delta"].cpu().numpy())
            val_tgts.append(batch["targets"].cpu().numpy())
            for s in batch_samples:
                val_keys.append(f"{Path(s.source_dir).name}/{s.recording_id}")

    if val_preds:
        val_predictions = np.concatenate(val_preds, axis=0)
        val_init = np.concatenate(val_inits, axis=0)
        val_delta = np.concatenate(val_deltas, axis=0)
        val_targets = np.concatenate(val_tgts, axis=0)
        errs = np.linalg.norm(val_predictions - val_targets, axis=-1)
        # 95 % percentile-bootstrap CI on MAE, resampled at the RECORDING level
        # (block / cluster bootstrap).  Per-knock windows from one recording are
        # strongly correlated, so resampling individual windows is statistical
        # pseudoreplication — it treats correlated knocks as independent draws
        # and yields an anti-conservative (falsely tight) interval (Davison &
        # Hinkley 1997 §3.8).  `val_keys` is the per-window recording id, so the
        # block bootstrap resamples whole recordings.  With a single val
        # recording the CI is honestly degenerate (no between-recording
        # information to resample), rather than falsely narrow.
        ci = percentile_bootstrap_ci(
            errs, n_boot=1000, seed=cfg.seed, groups=np.asarray(val_keys),
        )
        ci_low, ci_high = ci.ci_low, ci.ci_high
        ci_n_groups = ci.n_groups
        # Per-recording breakdown: mean(pred) vs target, recording-level MAE.
        breakdown: dict = {}
        for k in set(val_keys):
            mask = np.array([1 if vk == k else 0 for vk in val_keys], dtype=bool)
            if not mask.any():
                continue
            preds_k = val_predictions[mask]
            tgts_k = val_targets[mask]
            errs_k = np.linalg.norm(preds_k - tgts_k, axis=-1)
            breakdown[k] = {
                "n": int(mask.sum()),
                "mae_3d": float(errs_k.mean()),
                "p95_3d": float(np.percentile(errs_k, 95)),
                "target_xyz": tgts_k[0].astype(float).tolist(),
                "pred_xyz_mean": preds_k.mean(axis=0).astype(float).tolist(),
                "init_xyz_mean": val_init[mask].mean(axis=0).astype(float).tolist(),
                "delta_xyz_mean": val_delta[mask].mean(axis=0).astype(float).tolist(),
            }

        # Event-aggregated MAE (headline): one estimate per recording = mean of
        # its per-knock predictions, error against its GT.  Shared helper so the
        # trainer, the V3-gated CV metric, and Stage 5b aggregate identically.
        val_mae_agg, agg_errs, _ = event_aggregated_mae(
            val_predictions, val_targets, val_samples)
        val_p95_agg = float(np.percentile(agg_errs, 95)) if agg_errs.size else float("nan")

        # Per-POSITION breakdown (failure-mode analysis).  Groups val
        # windows by `_position_key(target_xyz)` so multiple windows at
        # the same spatial position aggregate into one entry — exposes
        # which positions the head misses worst.
        pos_breakdown: dict = {}
        pos_keys_per_window = [_position_key(t) for t in val_targets]
        for pk in {tuple(p) for p in pos_keys_per_window}:
            mask = np.array(
                [1 if tuple(p) == pk else 0 for p in pos_keys_per_window],
                dtype=bool,
            )
            if not mask.any():
                continue
            preds_p = val_predictions[mask]
            tgts_p = val_targets[mask]
            errs_p = np.linalg.norm(preds_p - tgts_p, axis=-1)
            pos_breakdown[f"({pk[0]:.3f},{pk[1]:.3f},{pk[2]:.3f})"] = {
                "n": int(mask.sum()),
                "mae_3d": float(errs_p.mean()),
                "p95_3d": float(np.percentile(errs_p, 95)),
                "target_xyz": tgts_p[0].astype(float).tolist(),
                "pred_xyz_mean": preds_p.mean(axis=0).astype(float).tolist(),
                "n_outliers_gt_0_5m": int((errs_p > 0.5).sum()),
            }
    else:
        val_predictions = np.zeros((0, 3), dtype=np.float32)
        val_init = np.zeros((0, 3), dtype=np.float32)
        val_delta = np.zeros((0, 3), dtype=np.float32)
        val_targets = np.zeros((0, 3), dtype=np.float32)
        val_mae_agg = float("nan")
        val_p95_agg = float("nan")
        agg_errs = np.zeros((0,), dtype=np.float64)
        ci_low = float("nan")
        ci_high = float("nan")
        ci_n_groups = None
        breakdown = {}
        pos_breakdown = {}

    # Final TRAIN MAE pass — same forward loop as val, in metres.  Lets the
    # caller report `train_val_mae_gap_m = val_mae - train_mae` (the proper
    # metres-scale gap; the smooth-L1 loss-history values are in loss_scale-cm
    # space and are not metres).
    train_mae_3d_v = float("nan")
    train_p95_3d_v = float("nan")
    if train_samples:
        head.eval()
        tr_preds: list[np.ndarray] = []
        tr_tgts: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(train_samples), cfg.batch_size):
                tb_samples = train_samples[i : i + cfg.batch_size]
                batch = _to_device(_make_batch(tb_samples, channel_mode=cfg.channel_mode))
                tr_pred = head(
                    batch["volumes"],
                    batch["tdoa"],
                    batch["contexts"],
                    batch["scada"],
                    unconditional=cfg.unconditional,
                    external_init_xyz=batch.get("external_init_xyz"),
                )
                tr_preds.append(tr_pred.cpu().numpy())
                tr_tgts.append(batch["targets"].cpu().numpy())
        if tr_preds:
            tr_p = np.concatenate(tr_preds, axis=0)
            tr_t = np.concatenate(tr_tgts, axis=0)
            tr_errs = np.linalg.norm(tr_p - tr_t, axis=-1)
            if tr_errs.size:
                train_mae_3d_v = float(tr_errs.mean())
                train_p95_3d_v = float(np.percentile(tr_errs, 95))

    def _qualify(s: V4Sample) -> str:
        return f"{Path(s.source_dir).name}/{s.recording_id}"

    return V4Result(
        head=head,
        train_loss_history=train_history,
        val_loss_history=val_history,
        # Headline = event-aggregated (per-recording).  Strictly NaN when there
        # are no val groups — never falls back to a per-window value.
        val_mae_3d=val_mae_agg,
        val_p95_3d=val_p95_agg,
        val_predictions=val_predictions,
        val_targets=val_targets,
        train_recording_ids=sorted({_qualify(s) for s in train_samples}),
        val_recording_ids=sorted({_qualify(s) for s in val_samples}),
        unconditional=cfg.unconditional,
        val_init_xyz=val_init,
        val_residuals=val_delta,
        val_recording_breakdown=breakdown,
        val_mae_ci_low=ci_low,
        val_mae_ci_high=ci_high,
        val_mae_ci_n_groups=ci_n_groups,
        val_groups=list(val_keys),
        early_stopped_epoch=early_stopped_epoch,
        best_val_loss=best_val_loss,
        train_mae_3d=train_mae_3d_v,
        train_p95_3d=train_p95_3d_v,
        val_position_breakdown=pos_breakdown,
        val_agg_errors=np.asarray(agg_errs, dtype=np.float32),
    )


__all__ = [
    "V4Config",
    "V4Result",
    "V4Sample",
    "precompute_v4_samples",
    "split_samples_by_dataset",
    "split_samples_by_position",
    "train_v4_localization",
]
