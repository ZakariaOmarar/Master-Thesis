"""Helper functions for the full-run orchestrator.

Dataset loaders, segment/event-interval utilities, run metadata (logger +
git introspection), the opt-in sync audit, and the per-instance V3/V4 stage
trainers. Kept out of the orchestrator so its ``main`` reads as the stage
sequence alone.
"""
from __future__ import annotations

import datetime as _dt
import subprocess
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ...config.architecture import SYNC
from ...ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader, TestDatasetSegment
from ..anomaly import (
    V3Config,
    train_v3_cnf,
)
from ..anomaly.event_detection import (
    detect_events_from_score_timeline,
    sliding_window_v3_inference,
)
from ..anomaly.v3_trainer import (
    precompute_paired,
)
from ..context.v2_ssl import V2SSLConfig
from ..localization import (
    V4Config,
    train_v4_localization,
)

# V1-V4 hyperparameter builders live in `stage_configs`; they are imported (and
# re-exported) here so `main()` resolves them from this module's namespace,
# which keeps the multi-seed / hop-length drivers' monkeypatching of
# `full_run.vN_config` working.

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def resolved_loader(yaml_name: str) -> TestDatasetLoader:
    """Build a sync-corrected loader for one dataset spec.

    Two things matter here that the legacy implementation missed:

      1. ``vibration_format`` must be propagated when reconstructing
         the spec — D4's spec sets ``vibration_format="raw"`` and the
         default-"peak" fallback would silently pick the wrong CSV
         family (and either error on missing files or, worse, load
         the peak-decimated stream instead of the 376 Hz raw waveform).

      2. ``sync_correct=True`` is set at the loader level so the
         WavVibrationAdapter applies the four-gate cross-modal sync
         correction at load time, before the frozen ``DataSegment`` is
         built.  The legacy orchestrator pattern — load, then mutate
         ``s.segment.mic_data = mic_corr`` after a separate auto-sync
         call — was a silent no-op: the assignment raised
         ``FrozenInstanceError`` on every recording and the bare
         ``except Exception`` in the audit loop swallowed it as
         ``n_skipped += 1``.  Configuring the loader's flag is the
         only working entry point and guarantees every downstream
         stage (V0 through V5) consumes sync-aligned segments.

    The four sync gate thresholds match the orchestrator's historical
    values so the audit-table semantics in chapter 6 carry over.
    """
    # `DatasetSpec.from_yaml` now resolves all paths (root, position_path) to
    # absolute REPO_ROOT-prefixed values, so the legacy reconstruction is
    # unnecessary.
    spec = DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / yaml_name)
    return TestDatasetLoader(
        spec,
        sync_correct=True,
        # Sync gating thresholds sourced from `SYNC` in
        # `src/config/architecture.py` — change there, not here.
        sync_correct_kwargs=dict(
            max_offset_s=SYNC.max_offset_s,
            n_sub_segments=SYNC.n_sub_segments,
            confidence_floor=SYNC.confidence_floor,
            drift_tolerance_s=SYNC.drift_tolerance_s,
            min_offset_to_correct_s=SYNC.min_offset_to_correct_s,
            use_fractional_shift=SYNC.use_fractional_shift,
        ),
    )


def _all_segments(loaders: list[TestDatasetLoader]) -> list[TestDatasetSegment]:
    out: list[TestDatasetSegment] = []
    for L in loaders:
        out.extend(L.list_segments())
    return out


# ---------------------------------------------------------------------------
# V4 spatial holdout — positions reserved as the localization generalisation
# test (folder coords in cm → metres).  Held out of V4 training so the
# reported MAE measures localise-an-unseen-position, not within-position
# interpolation.
# ---------------------------------------------------------------------------

V4_HOLDOUT_POSITIONS_M: list[tuple[float, float, float]] = [
    (0.22, 0.0, 0.0),     # D5 knock (22, 0, 0)
    (0.03, -0.03, 0.08),  # D5 knock (3, -3, 8)
    (0.06, -0.15, 0.0),   # D5 knock (6, -15, 0)
    (0.02, 0.04, 0.08),   # D4 RandomFault_knock (2, 4, 8)
    (0.0, -0.20, 0.0),    # D4 RandomFault_knock (0, -20, 0)
]


def _v3_event_intervals_for_recordings(
    holdout_samples,
    loaders_by_id: dict,
    v2_encoder,
    v3,
    v2_cfg,
    v3_cfg,
) -> dict:
    """Return {recording_id: [(t_start_s, t_end_s), ...]} of V3-detected events.

    Runs V3 (the trained fusion flow + thresholds) at a fine stride on each
    held-out recording and extracts alert events.  Used to GATE the V4
    holdout: V4 only "fires" on windows V3 flags anomalous in deployment, so
    the gated MAE is the deployment-faithful localization number.
    """

    bar = v3.thresholds.p99 if int(v3_cfg.threshold_percentile) >= 99 else v3.thresholds.p95
    # Map each holdout sample's (dataset_id, recording_id) back to a segment.
    wanted: dict[str, str] = {s.recording_id: s.dataset_id for s in holdout_samples}
    out: dict[str, list[tuple[float, float]]] = {}
    for dsid in sorted(set(wanted.values())):
        loader = loaders_by_id.get(dsid)
        if loader is None:
            continue
        for s in loader.list_segments():
            if s.recording_id not in wanted or wanted[s.recording_id] != dsid:
                continue
            paired = precompute_paired(s, v2_cfg)
            if paired is None:
                continue
            try:
                times, scores, contexts = sliding_window_v3_inference(
                    v2_encoder, v3.flow, paired,
                    v2_cfg=v2_cfg, inference_stride_s=0.25,
                    xt_pool=getattr(v3, "xt_pool", None), device=v3_cfg.device,
                    anchor_norm=((v3.anchor_mean, v3.anchor_std)
                                 if getattr(v3, "anchor_mean", None) is not None else None),
                )
            except Exception:
                # Skip a recording whose V3 inference fails; it simply gets no
                # event-interval entry (callers read `out` with .get()).
                continue
            if scores.size == 0:
                continue
            clusters = v3.thresholds.assign(contexts)
            high = float(np.median([float(bar[int(k)]) for k in clusters]))
            # low <= high required; V3 scores are negative NLLs so 0.95*high
            # would invert (see event_detection.v3_real_anomaly_detection).
            low = high - abs(high) * 0.05
            if low > high:
                low = high
            evs = detect_events_from_score_timeline(
                scores, times, high_threshold=high, low_threshold=low,
                min_duration_s=0.10, max_gap_windows=0,
                recording_id=s.recording_id, dataset_id=dsid,
                window_seconds=v2_cfg.window_seconds,
            )
            out[s.recording_id] = [(e.t_start_s, e.t_end_s) for e in evs]
    return out


# ---------------------------------------------------------------------------
# Spatial-label derivation for D3 hits
# ---------------------------------------------------------------------------


# Ground-truth spatial label for D3's `hit_between_Fl_Gr_speed1` family.
# Sensors Fl=(6, -5, 8) cm and Gr=(11, 0, 8) cm both sit at z=8 cm, so the
# knock is constrained to that height and approximated by their centroid
# (cm converted to metres). z is the reliable constraint; x, y carry larger
# uncertainty. This is the only D3 hit family with a usable spatial label.
_D3_HIT_FL_GR_XYZ_M: tuple[float, float, float] = (0.085, -0.025, 0.080)


def _d3_spatial_overrides(d3_segments: list[TestDatasetSegment]) -> dict[str, tuple[float, float, float]]:
    """Derive spatial labels for D3 `hit_between_*_speed*` recordings.

    Only the `hit_between_Fl_Gr_*` family has a usable ground-truth position
    (`_D3_HIT_FL_GR_XYZ_M`); it is matched on either the recording id or the
    source folder name. Other hit pairs lack a reliable label and are skipped.
    """
    out: dict[str, tuple[float, float, float]] = {}
    for s in d3_segments:
        if "hit_between" not in s.recording_id and "hit_between" not in str(s.source_dir).lower():
            continue
        rec_lower = s.recording_id.lower()
        src_lower = str(s.source_dir).lower()
        is_fl_gr = ("fl" in rec_lower and "gr" in rec_lower) or (
            "fl" in src_lower and "gr" in src_lower
        )
        if is_fl_gr:
            out[s.recording_id] = _D3_HIT_FL_GR_XYZ_M
    return out


# ---------------------------------------------------------------------------
# Logging + git provenance
# ---------------------------------------------------------------------------


def _make_logger(out_dir: Path) -> Callable[[str], None]:
    """Return a `log(msg)` that prints and appends to ``run_log.txt``."""
    log_path = out_dir / "run_log.txt"

    def log(msg: str) -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(line + "\n")

    return log


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True, capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Stage 0 — sync verification + correction audit
# ---------------------------------------------------------------------------


def _audit_sync(loaders: list, log: Callable[[str], None]) -> dict:
    out: dict = {}
    for L in loaders:
        ds_name = L.spec.id
        offsets_s: list[float] = []
        confidences: list[float] = []
        env_kurtoses: list[float] = []
        n_applied = 0
        n_rejected_low_conf = 0
        n_rejected_drift = 0
        n_rejected_below_floor = 0
        n_rejected_flat_envelope = 0
        n_skipped = 0
        for s in L.list_segments():
            report = s.segment.metadata.get("sync_correction")
            if report is None:
                n_skipped += 1
                continue
            audit_offset = float(report.get("audit_offset_s", float("nan")))
            audit_conf = float(report.get("audit_confidence", float("nan")))
            env_kurt = float(report.get("acoustic_envelope_kurtosis", float("nan")))
            if not np.isnan(audit_offset):
                offsets_s.append(audit_offset)
                confidences.append(audit_conf)
            if not np.isnan(env_kurt):
                env_kurtoses.append(env_kurt)
            reason = str(report.get("reason") or "").lower()
            if report.get("applied"):
                n_applied += 1
            elif "stability" in reason or "drift" in reason:
                n_rejected_drift += 1
            elif "near-gaussian" in reason:
                n_rejected_flat_envelope += 1
            elif "confidence" in reason or "uninformative" in reason:
                n_rejected_low_conf += 1
            elif "below" in reason or "already aligned" in reason:
                n_rejected_below_floor += 1
        if offsets_s:
            arr = np.asarray(offsets_s)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            mean_conf = float(np.mean(confidences))
            med_kurt = float(np.median(env_kurtoses)) if env_kurtoses else float("nan")
            n_total = (
                n_applied + n_rejected_low_conf + n_rejected_drift
                + n_rejected_below_floor + n_rejected_flat_envelope + n_skipped
            )
            log(
                f"  {ds_name}: n={n_total} median_offset={med*1e3:+.1f} ms "
                f"(MAD ±{mad*1e3:.1f} ms) conf={mean_conf:.2f} "
                f"applied={n_applied} drift={n_rejected_drift} "
                f"low_conf={n_rejected_low_conf} flat={n_rejected_flat_envelope}"
            )
            out[ds_name] = {
                "n_recordings_total": n_total,
                "n_corrections_applied": n_applied,
                "n_rejected_drift": n_rejected_drift,
                "n_rejected_low_confidence": n_rejected_low_conf,
                "n_rejected_flat_envelope": n_rejected_flat_envelope,
                "n_rejected_below_floor": n_rejected_below_floor,
                "n_skipped": n_skipped,
                "median_offset_s": med,
                "median_absolute_deviation_s": mad,
                "min_offset_s": float(np.min(arr)),
                "max_offset_s": float(np.max(arr)),
                "mean_confidence": mean_conf,
                "median_acoustic_envelope_kurtosis": med_kurt,
            }
        else:
            log(f"  {ds_name}: no auditable recordings (n_skipped={n_skipped})")
            out[ds_name] = {"n_recordings_total": 0, "n_skipped": n_skipped}
    return out


# ---------------------------------------------------------------------------
# Stage 3 helper — train one V3 instance and persist artefacts the
# rq2_three_paradigm_eval CLI consumes (flow.pt + thresholds.npz + val_eval.npz).
# ---------------------------------------------------------------------------


def _train_one_v3(
    name: str,
    encoder: torch.nn.Module,
    loaders: list,
    v2_cfg: V2SSLConfig,
    v3_cfg: V3Config,
    out_dir: Path,
    log: Callable[[str], None],
):
    log(f"V3-{name} — training conditional CNF ...")
    t0 = time.time()
    # `encoder` may be the V2FusionEncoder (fusion paradigm) or one of the
    # V3{Acoustic,Vibration}OnlyAdapter wrappers (unimodal paradigms).  All
    # three implement the same `forward(...) -> (paired, c_t, x_t_per_w)`
    # contract train_v3_cnf consumes; the static type of the parameter is
    # narrower than the runtime contract.
    res = train_v3_cnf(encoder, loaders, v2_cfg=v2_cfg, v3_cfg=v3_cfg)  # type: ignore[arg-type]
    log(f"  V3-{name} {time.time()-t0:.0f}s — val NLL={res.val_nll[-1]:.3f}")
    pipe_dir = out_dir / f"v3_{name}"
    pipe_dir.mkdir(parents=True, exist_ok=True)
    torch.save(res.flow.state_dict(), pipe_dir / "flow.pt")
    # Persist the learnable channel-token pool (pma2) alongside the flow.  Without
    # it, downstream re-scoring (rq2_three_paradigm_eval) cannot reproduce the
    # x_t the flow was trained on and silently falls back to mean-pooling, which
    # makes the NLL blow up and the healthy FPR degenerate to 1.0.
    if getattr(res, "xt_pool", None) is not None:
        torch.save(res.xt_pool.state_dict(), pipe_dir / "xt_pool.pt")
    threshold_arrays = dict(
        centroids=res.thresholds.centroids,
        p95=res.thresholds.p95,
        p99=res.thresholds.p99,
        n_per_cluster=res.thresholds.n_per_cluster,
    )
    # Persist the impulse+spectral anchor standardization (RQ2 anchor injection)
    # so the RQ2 eval and V4 gate recompute + standardize the anchor identically.
    if getattr(res, "anchor_mean", None) is not None:
        threshold_arrays["anchor_mean"] = res.anchor_mean
        threshold_arrays["anchor_std"] = res.anchor_std
    np.savez(pipe_dir / "thresholds.npz", **threshold_arrays)
    np.savez(
        pipe_dir / "val_eval.npz",
        scores=res.val_scores,
        contexts=res.val_contexts,
        labels=np.asarray(res.val_labels, dtype="U64"),
    )
    return res


# ---------------------------------------------------------------------------
# Stage 5 helper — train one V4 instance and persist artefacts the
# rq3_three_paradigm_eval CLI consumes (head.pt + val_predictions.npz).
# ---------------------------------------------------------------------------


def _train_one_v4(
    name: str,
    channel_mode: str,
    samples: list,
    grid,
    base_cfg: V4Config,
    out_dir: Path,
    log: Callable[[str], None],
):
    log(f"V4-{name} (channel_mode={channel_mode}) — training localisation head ...")
    cfg = replace(base_cfg, channel_mode=channel_mode)
    t0 = time.time()
    res = train_v4_localization(samples, cfg=cfg, grid=grid)
    dt = time.time() - t0
    log(f"  V4-{name} {dt:.0f}s — val MAE={res.val_mae_3d:.4f} m, p95={res.val_p95_3d:.4f} m")
    pipe_dir = out_dir / f"v4_{name}"
    pipe_dir.mkdir(parents=True, exist_ok=True)
    torch.save(res.head.state_dict(), pipe_dir / "head.pt")
    val_set = set(res.val_recording_ids)
    val_keys: list[str] = []
    for s in samples:
        key = f"{Path(s.source_dir).name}/{s.recording_id}"
        if key in val_set:
            val_keys.append(key)
    np.savez(
        pipe_dir / "val_predictions.npz",
        predictions=res.val_predictions,
        targets=res.val_targets,
        init_xyz=res.val_init_xyz,
        residuals=res.val_residuals,
        recording_keys=np.asarray(val_keys, dtype="U64"),
    )
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


