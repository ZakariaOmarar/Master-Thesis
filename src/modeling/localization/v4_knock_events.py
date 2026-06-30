"""Per-knock event expansion for V4 localization training.

Motivation (the RQ3 data-scarcity problem)
-------------------------------------------
The supervised V4 cohort is tiny — ~16 labelled positions across D2/D3/D4/D5.
The live builder (`precompute_v4_samples`) slices a recording into fixed-stride
windows and keeps the ones overlapping a derived knock interval, but it
computes SRP-PHAT / TDOA on the **whole window** (or, with `burst_aware_srp`,
on the single highest-energy 100 ms sub-window) and pastes the *recording-level*
position onto every kept window.  A recording that contains five physical
knocks therefore contributes a handful of coarse, partially-overlapping windows
rather than five clean, transient-centred localization examples.

This module turns each detected knock into its own training sample:

  * `derive_knock_events` (the weak-label detector) already finds **every**
    impulsive burst in a recording via iterative peak-picking, unioned across
    the mic and accel streams.  We localize **each** burst separately.
  * The SRP-PHAT volume and accel-TDOA tokens are computed on a **tight crop**
    around the burst, so the GCC-PHAT cross-correlation integrates over samples
    that actually carry source information instead of ~95 % background — the
    same SNR argument as `compute_burst_aware_srp_phat_volume`, now applied per
    knock instead of once per window.
  * Long continuously-anomalous spans (D2 RandomFault, D5 knock) are tiled into
    several crops so they, too, yield multiple estimates.

The net effect is many more, individually sharper, supervised samples — the
"localize every knock, multiple times" expansion.

Leakage discipline (why this is safe)
-------------------------------------
Every knock in one recording shares that recording's single position label, and
the same physical position can recur across recordings/campaigns.  Splitting the
**expanded** sample set naively (e.g. random per-sample) would leak: near-
duplicate crops of one knock — or other knocks at the same position — would land
on both sides of the train/val boundary and inflate the score.

The expansion is therefore designed to be consumed **only** by group-aware
splitters that group by position (`v4_lopo_cv` / `split_samples_by_position`) or
by recording (`_split_samples_by_recording`).  Each emitted `V4Sample` keeps its
`recording_id`, `source_dir`, and `target_xyz` intact, so those splitters keep
all of a position's knocks together.  `assert_no_position_leak` makes the
invariant checkable at the call site.  Do **not** feed the expanded set to a
per-window random split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ...config import resolve_device
from ...ingestion.test_dataset_loader import TestDatasetSegment
from ..anomaly.weak_labels import derive_knock_events
from ..context.v2_fusion import V2FusionEncoder
from ..context.v2_ssl import V2SSLConfig, _dataset_idx
from .v4_features import (
    GridSpec,
    bandpass_filter,
    compute_accel_tdoa_tokens,
    compute_srp_phat_volume,
    find_burst_window,
    srp_peak_sharpness,
)
from .v4_trainer import V4Sample, _window_v2_features


@dataclass(frozen=True)
class KnockEventConfig:
    """Knobs for per-knock expansion (all in seconds unless noted)."""

    # Tight crop fed to SRP-PHAT / accel-TDOA around each knock centre.  A 120 ms
    # crop comfortably contains a ~50 ms knock plus ring-down while excluding the
    # background that smears the GCC-PHAT peak.
    crop_seconds: float = 0.12
    # A derived interval longer than this is treated as a continuous-anomaly span
    # and tiled into multiple events rather than localized as one.
    max_single_event_seconds: float = 0.20
    # Hop between tiled events inside a long span (controls how many estimates a
    # continuous span yields).
    event_hop_seconds: float = 0.12
    # Knock-detector sensitivity (forwarded to `derive_knock_events`).
    burst_seconds: float = 0.10
    max_events: int = 24
    noise_floor_mult: float = 3.0
    # When no impulse clears the noise floor, fall back to the single highest-
    # energy burst so a labelled recording is never silently dropped.
    fallback_to_loudest_burst: bool = True
    # Acoustic GCC up-sampling factor for the per-knock SRP volume (1 = off).
    # See `compute_srp_phat_volume`; sharpens the SRP peak below the voxel grid.
    gcc_oversample: int = 1
    # Accel-TDOA token sub-sample resolution.  The low accel rate (376 Hz)
    # quantises the integer-lag path_diff to ~±5 m at c=2000 — `tdoa_gcc_oversample
    # > 1` (+ parabolic) recovers a real sub-sample structure-borne TDOA.
    # `accel_c_ms` overrides the assumed structure-borne wave speed (None =
    # default 2000); flexural plate waves in thin 3D-printed plastic are slower
    # (~300-800 m/s), which both rescales the feature and makes TDOA resolvable.
    tdoa_gcc_oversample: int = 1
    accel_c_ms: float | None = None
    # PHAT whitening exponent (1.0 = full PHAT) and linear-vs-circular GCC.
    # Both forwarded to `compute_srp_phat_volume`; defaults preserve behaviour.
    phat_beta: float = 1.0
    linear_corr: bool = False
    # Band-pass the crop before SRP (Hz); None disables.  Isolates the knock's
    # informative band, dropping rig rumble + HF hiss that PHAT over-weights.
    bandpass_hz: tuple[float, float] | None = None
    # Test-time-augmentation multi-crop: emit `crops_per_knock` crops per knock,
    # their centres spread symmetrically over ±`crop_jitter_seconds`.  Each is an
    # independent localization estimate of the same knock, so aggregating them at
    # inference multiplies the √n sharpening the per-position aggregation already
    # exploits.  `crops_per_knock=1` reproduces the single-crop behaviour.
    crops_per_knock: int = 1
    crop_jitter_seconds: float = 0.02
    # Multi-scale crops: one well-CENTRED crop per width (e.g. (0.08, 0.12, 0.20)).
    # Unlike TTA jitter this keeps the impulse centred, so every estimate stays
    # high-quality while still giving the per-position aggregation more
    # independent views.  Takes precedence over `crops_per_knock` when set.
    crop_scales_seconds: tuple[float, ...] | None = None


def _event_centres(
    intervals: list[tuple[float, float]],
    duration_s: float,
    cfg: KnockEventConfig,
) -> list[float]:
    """Expand merged knock intervals into per-event centre times (seconds).

    Short intervals contribute their midpoint; long continuous-anomaly spans are
    tiled at `event_hop_seconds`.  Centres are clamped into `[0, duration_s]`.
    """
    centres: list[float] = []
    for t0, t1 in intervals:
        t0 = max(0.0, float(t0))
        t1 = min(float(duration_s), float(t1))
        if t1 <= t0:
            continue
        span = t1 - t0
        if span <= cfg.max_single_event_seconds:
            centres.append(0.5 * (t0 + t1))
            continue
        k = max(1, int(math.ceil(span / max(cfg.event_hop_seconds, 1e-6))))
        for i in range(k):
            c = t0 + cfg.event_hop_seconds * (0.5 + i)
            if c < t1:
                centres.append(c)
    return centres


def precompute_v4_knock_event_samples(
    v2_encoder: V2FusionEncoder,
    segments: list[TestDatasetSegment],
    *,
    v2_cfg: V2SSLConfig,
    grid: GridSpec,
    spatial_label_overrides: dict[str, tuple[float, float, float]] | None = None,
    window_seconds: float | dict[str, float] | None = None,
    cfg: KnockEventConfig | None = None,
    v3_xt_pool: torch.nn.Module | None = None,
    v3_anchor_norm: tuple[np.ndarray, np.ndarray] | None = None,
    device: torch.device | str = "auto",
) -> list[V4Sample]:
    """Build one `V4Sample` per detected knock (transient-centred crops).

    Drop-in replacement for `precompute_v4_samples`.  Same `V4Sample` schema,
    same grid contract, same V2-context conditioning — only the sampling unit
    changes from "fixed window overlapping a knock interval" to "each knock,
    cropped tight".

    `window_seconds` sets the V2 **context** window length (per-dataset dict
    mirrors `V4Config.window_seconds_override`); the SRP/TDOA crop length is
    independent and set by `cfg.crop_seconds`.

    `v3_xt_pool` / `v3_anchor_norm` mirror `precompute_v4_samples`: when given,
    `x_for_v3` is pooled with V3's learned pool (else mean) and the standardized
    impulse+spectral anchor is appended, so the V4 gate scores exactly the input
    the conditional V3 flow trained on.

    Returns the expanded sample list.  Feed it only to position- or recording-
    grouped splitters (see module docstring); `assert_no_position_leak` checks
    that invariant downstream.
    """
    cfg = cfg or KnockEventConfig()
    device = resolve_device(device)
    v2_encoder = v2_encoder.to(device).eval()
    for p in v2_encoder.parameters():
        p.requires_grad_(False)
    if v3_xt_pool is not None:
        v3_xt_pool = v3_xt_pool.to(device).eval()
    overrides = spatial_label_overrides or {}

    def _per_segment_window_s(dataset_id: str) -> float:
        if window_seconds is None:
            return float(v2_cfg.window_seconds)
        if isinstance(window_seconds, dict):
            return float(window_seconds.get(dataset_id, v2_cfg.window_seconds))
        return float(window_seconds)

    samples: list[V4Sample] = []
    for s in segments:
        spatial = overrides.get(s.recording_id, s.spatial_label)
        if spatial is None:
            continue  # V4 needs a ground-truth position
        target = np.asarray(spatial, dtype=np.float32)

        win_s = _per_segment_window_s(s.dataset_id)

        mic = s.segment.mic_data
        accel = s.segment.accel_data
        mic_fs = int(s.segment.mic_sample_rate)
        accel_fs = int(s.segment.accel_sample_rate)
        T_mic = mic.shape[1]
        T_acc = accel.shape[1]
        duration_s = T_mic / float(mic_fs)

        # --- detect every knock, then expand to per-event centres -------------
        try:
            intervals = derive_knock_events(
                s,
                burst_seconds=cfg.burst_seconds,
                max_events=cfg.max_events,
                noise_floor_mult=cfg.noise_floor_mult,
            )
        except Exception:
            intervals = []
        if not intervals and cfg.fallback_to_loudest_burst:
            try:
                bs, be = find_burst_window(mic, float(mic_fs), burst_seconds=cfg.burst_seconds)
                intervals = [(bs / mic_fs, be / mic_fs)]
            except Exception:
                intervals = []
        centres = _event_centres(intervals, duration_s, cfg)
        if not centres:
            continue

        # --- V2 feature stacks (computed once per recording) ------------------
        ac_feats, vib_feats = _window_v2_features(s, v2_cfg)
        ac_fs = float(mic_fs) / float(v2_cfg.hop_length)
        vib_fs = float(accel_fs)
        T_ac = ac_feats.shape[-1]
        T_vib = vib_feats.shape[-1]
        n_ac = max(2, int(round(win_s * ac_fs)))
        n_vib = max(2, int(round(win_s * vib_fs)))
        if T_ac < n_ac or T_vib < n_vib:
            continue

        ds_idx = torch.tensor([_dataset_idx(s.dataset_id)], dtype=torch.long, device=device)
        ac_xyz = torch.from_numpy(s.mic_positions.astype(np.float32)).unsqueeze(0).to(device)
        vib_xyz = torch.from_numpy(s.vib_positions.astype(np.float32)).unsqueeze(0).to(device)

        # Per-knock crop "views" = (crop_seconds, time_offset).  Three mutually
        # exclusive modes:
        #   - multi-scale (`crop_scales_seconds`): one well-CENTRED crop per
        #     width — complementary integration windows, no decentering.
        #   - TTA jitter (`crops_per_knock>1`): same width, centres spread over
        #     ±jitter (cheap but decenters the impulse → hurts the SRP peak).
        #   - single crop (default).
        if cfg.crop_scales_seconds:
            crop_views = [(float(sc), 0.0) for sc in cfg.crop_scales_seconds]
        elif cfg.crops_per_knock > 1:
            crop_views = [
                (cfg.crop_seconds, float(off))
                for off in np.linspace(
                    -cfg.crop_jitter_seconds, cfg.crop_jitter_seconds, cfg.crops_per_knock
                )
            ]
        else:
            crop_views = [(cfg.crop_seconds, 0.0)]

        for centre_s in centres:
            # V2 context window, centred on the knock (clamped to bounds).
            # Computed once per knock and reused across its TTA crops.
            ac_w0 = int(round(centre_s * ac_fs)) - n_ac // 2
            ac_w0 = int(np.clip(ac_w0, 0, max(0, T_ac - n_ac)))
            vib_w0 = int(round(centre_s * vib_fs)) - n_vib // 2
            vib_w0 = int(np.clip(vib_w0, 0, max(0, T_vib - n_vib)))
            ac_win = (
                torch.from_numpy(np.ascontiguousarray(ac_feats[..., ac_w0 : ac_w0 + n_ac]))
                .unsqueeze(0).float().to(device)
            )
            vib_win = (
                torch.from_numpy(np.ascontiguousarray(vib_feats[..., vib_w0 : vib_w0 + n_vib]))
                .unsqueeze(0).float().to(device)
            )
            with torch.no_grad():
                out = v2_encoder(ac_win, ac_xyz, vib_win, vib_xyz, ds_idx, mask_p=0.0)
            c_t = out["context"].squeeze(0).cpu().numpy().astype(np.float32)
            fused = torch.cat([out["a_fused"], out["v_fused"]], dim=1)
            # Pool x_for_v3 the way V3 consumes it (PMA-2 when supplied, else mean),
            # then optionally append V3's impulse+spectral anchor — so a V3-gated
            # deployment scores the exact input the conditional flow trained on.
            if v3_xt_pool is not None:
                with torch.no_grad():
                    x_t = v3_xt_pool(fused)
            else:
                x_t = fused.mean(dim=1)
            x_for_v3 = x_t.squeeze(0).cpu().numpy().astype(np.float32)
            if v3_anchor_norm is not None:
                from ..anomaly.impulse_anchor import append_anchor
                x_for_v3 = append_anchor(
                    x_for_v3[None, :], ac_win, vib_win, v3_anchor_norm
                )[0].astype(np.float32)

            for view_crop_s, off in crop_views:
                cc = centre_s + off
                n_crop_mic = max(8, int(round(view_crop_s * mic_fs)))
                n_crop_acc = max(2, int(round(view_crop_s * accel_fs)))
                # Tight raw crops for SRP / TDOA, centred on the (jittered) knock.
                mic_start = int(round(cc * mic_fs)) - n_crop_mic // 2
                mic_start = int(np.clip(mic_start, 0, max(0, T_mic - n_crop_mic)))
                acc_start = int(round(cc * accel_fs)) - n_crop_acc // 2
                acc_start = int(np.clip(acc_start, 0, max(0, T_acc - n_crop_acc)))
                mic_crop = mic[:, mic_start : mic_start + n_crop_mic]
                acc_crop = accel[:, acc_start : acc_start + n_crop_acc]
                if mic_crop.shape[1] < 8 or acc_crop.shape[1] < 2:
                    continue
                if cfg.bandpass_hz is not None:
                    mic_crop = bandpass_filter(
                        mic_crop, float(mic_fs), cfg.bandpass_hz[0], cfg.bandpass_hz[1]
                    )

                volume = compute_srp_phat_volume(
                    mic_crop, s.mic_positions, fs=mic_fs, grid=grid,
                    gcc_oversample=cfg.gcc_oversample,
                    phat_beta=cfg.phat_beta, linear_corr=cfg.linear_corr,
                )
                psr = srp_peak_sharpness(volume)
                tdoa_kw = {} if cfg.accel_c_ms is None else {"c": cfg.accel_c_ms}
                tdoa = compute_accel_tdoa_tokens(
                    acc_crop, s.vib_positions, fs=accel_fs,
                    gcc_oversample=cfg.tdoa_gcc_oversample, **tdoa_kw,
                )

                multilat_xyz: np.ndarray | None = None
                if acc_crop.shape[0] >= 4:
                    try:
                        from .multilateration import accel_tdoa_multilateration_v0

                        pos, _residual = accel_tdoa_multilateration_v0(
                            acc_crop, s.vib_positions, fs=accel_fs
                        )
                        multilat_xyz = pos.astype(np.float32)
                    except Exception:
                        multilat_xyz = None

                samples.append(
                    V4Sample(
                        srp_volume=volume,
                        tdoa_tokens=tdoa,
                        context=c_t,
                        x_for_v3=x_for_v3,
                        target_xyz=target,
                        scada=None,
                        mode_label=s.mode_label,
                        recording_id=s.recording_id,
                        source_dir=str(s.source_dir),
                        dataset_id=s.dataset_id,
                        multilat_xyz=multilat_xyz,
                        window_start_s=float(cc),
                        srp_psr=float(psr),
                    )
                )
    return samples


def assert_no_position_leak(
    train: list[V4Sample], val: list[V4Sample], *, ndigits: int = 3
) -> None:
    """Raise if any target position appears in both splits (the leak guard).

    Per-knock expansion is only honest under a position-disjoint split; this is
    the cheap runtime check that a caller actually used one.
    """

    def _keys(xs: list[V4Sample]) -> set[tuple[float, float, float]]:
        return {
            tuple(round(float(v), ndigits) for v in np.asarray(s.target_xyz).ravel()[:3])
            for s in xs
        }

    overlap = _keys(train) & _keys(val)
    if overlap:
        raise AssertionError(
            f"position leak: {len(overlap)} target position(s) appear in both "
            f"train and val (e.g. {sorted(overlap)[:3]}). Per-knock samples must "
            f"be split by position or recording, never per-sample."
        )


__all__ = [
    "KnockEventConfig",
    "assert_no_position_leak",
    "precompute_v4_knock_event_samples",
]
