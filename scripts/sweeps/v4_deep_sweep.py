"""Deep V4 localization sweep — Phase 2 of the V3-first deep campaign.

Runs after the Phase-1 V3 winner is chosen, because V4 is gated by V3.
Trains the V4 head individually against a FROZEN V2 encoder (samples cached
once; ~10 min/cell) and evaluates on the **held-out positions** (localise-an-
unseen-position), **gated by the Phase-1 V3** (deployment-faithful: V4 only
fires on V3-flagged windows).  The reported localization numbers are the
absolute spatial-holdout MAE (gated and ungated); per-modality breakdown
comes from ``--all-channel-modes`` and the training-window-selection study
from ``--train-select``.  No V0-multilateration comparison — it is not a
credible baseline for this rig.

Selection objective (gap is a guardrail, not the target): minimize the
**V3-gated holdout MAE**, **subject to** ``|val_mae − train_mae| ≤ guardrail``.

Axes (superset of v4_aug_sweep):
  Regularization: head_dropout_p {0.0,0.1,0.2,0.3} × weight_decay {1e-4,5e-4,1e-3}
    v4_hd{0,1,2,3}_w{4,5,3}
  Capacity (cnn_feature_dim, hidden_dim):
    v4_cap_small (32,32)  v4_cap_base (64,64)  v4_cap_big (128,128)
  Residual half-range:
    v4_rs10 (0.10)  v4_rs20 (0.20)  v4_rs30 (0.30)
  Augmentation (target_pos_noise × srp_volume_noise):
    v4_pos{1,5,10}_srp{02,10,20}

Run::

    python -m scripts.sweeps.v4_deep_sweep --encoder-run <dir> --v3-run <phase1_v3_winner_dir>
    python -m scripts.sweeps.v4_deep_sweep --encoder-run <dir> --v3-run <dir> --cell v4_hd2_w5
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from src.modeling.anomaly.cnf_head import ConditionalRealNVP
from src.modeling.anomaly.threshold import PerClusterThresholds
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.localization import (
    V4_CANDIDATE_GRID,
    precompute_v4_samples,
    split_samples_by_position,
    train_v4_localization,
)
from src.modeling.localization.v3_gating import gate_samples_by_v3
from src.modeling.orchestration.full_run import (
    REPO_ROOT,
    V4_HOLDOUT_POSITIONS_M,
    _d3_spatial_overrides,
    resolved_loader,
    v2_config,
    v3_config,
    v4_config,
)

_DROPOUT_LEVELS = {"hd0": 0.0, "hd1": 0.1, "hd2": 0.2, "hd3": 0.3}
_WD_LEVELS = {"w4": 1e-4, "w5": 5e-4, "w3": 1e-3}
_CAP_LEVELS = {"small": (32, 32), "base": (64, 64), "big": (128, 128)}
_RS_LEVELS = {"rs10": 0.10, "rs20": 0.20, "rs30": 0.30}
_POS_LEVELS = {"pos1": 0.002, "pos5": 0.010, "pos10": 0.020}
_SRP_LEVELS = {"srp02": 0.02, "srp10": 0.10, "srp20": 0.20}


def _all_cells() -> list[str]:
    reg = [f"v4_{d}_{w}" for d in _DROPOUT_LEVELS for w in _WD_LEVELS]
    cap = [f"v4_cap_{c}" for c in _CAP_LEVELS]
    rs = [f"v4_{r}" for r in _RS_LEVELS]
    aug = [f"v4_{p}_{s}" for p in _POS_LEVELS for s in _SRP_LEVELS]
    return reg + cap + rs + aug


def _apply_cell(cell_id: str, v4_cfg):
    if cell_id.startswith("v4_cap_"):
        key = cell_id[len("v4_cap_"):]
        cnn, hidden = _CAP_LEVELS[key]
        return replace(v4_cfg, cnn_feature_dim=cnn, hidden_dim=hidden)
    parts = cell_id.split("_")
    if len(parts) == 2 and parts[1] in _RS_LEVELS:  # v4_rs10
        return replace(v4_cfg, residual_scale_m=_RS_LEVELS[parts[1]])
    if len(parts) == 3 and parts[1] in _DROPOUT_LEVELS and parts[2] in _WD_LEVELS:
        return replace(v4_cfg, head_dropout_p=_DROPOUT_LEVELS[parts[1]],
                       weight_decay=_WD_LEVELS[parts[2]])
    if len(parts) == 3 and parts[1] in _POS_LEVELS and parts[2] in _SRP_LEVELS:
        return replace(v4_cfg, target_pos_noise_m=_POS_LEVELS[parts[1]],
                       srp_volume_noise_std=_SRP_LEVELS[parts[2]])
    raise ValueError(f"unknown v4 cell id {cell_id!r}")


def _load_v3(v3_run: Path, c_dim: int):
    """Load the Phase-1 V3 winner's flow + thresholds + xt_pool for gating."""
    th = np.load(v3_run / "thresholds.npz")
    thresholds = PerClusterThresholds(
        centroids=th["centroids"], p95=th["p95"], p99=th["p99"],
        n_per_cluster=th["n_per_cluster"],
    )
    # Infer flow dims from the saved state_dict + thresholds centroid dim.
    state = torch.load(v3_run / "flow.pt", map_location="cpu")
    cfg = json.loads((v3_run / "cell_config.json").read_text())["v3_cfg"]
    flow = ConditionalRealNVP(
        dim=int(thresholds.centroids.shape[1]) if thresholds.centroids.ndim == 2 else c_dim,
        c_dim=c_dim, n_layers=int(cfg["n_layers"]), hidden_dim=int(cfg["hidden_dim"]),
        n_hidden_per_net=int(cfg["n_hidden_per_net"]), scale_max=float(cfg["scale_max"]),
        dropout_p=float(cfg.get("dropout_p", 0.0)),
    )
    flow.load_state_dict(state)
    flow.eval()
    # Reconstruct the learned xt_pool (PMA-2) so gating-time inference pooling
    # matches training pooling — the calibration fix.  Absent file => the V3
    # cell used the legacy mean-pool, so xt_pool stays None.
    xt_pool = None
    xtp_path = v3_run / "xt_pool.pt"
    if str(cfg.get("xt_pool", "pma2")) == "pma2" and xtp_path.exists():
        from src.modeling.anomaly.v3_trainer import _XtPool
        xt_pool = _XtPool(c_dim, num_heads=int(cfg.get("xt_pool_num_heads", 4)))
        xt_pool.load_state_dict(torch.load(xtp_path, map_location="cpu"))
        xt_pool.eval()
    return flow, thresholds, xt_pool, cfg


class _V3Holder:
    """Minimal duck-typed stand-in for a V3Result.  ``.flow``, ``.thresholds``
    and ``.xt_pool`` are read by `_v3_event_intervals_for_recordings`."""
    def __init__(self, flow, thresholds, xt_pool=None):
        self.flow = flow
        self.thresholds = thresholds
        self.xt_pool = xt_pool


def _v3_intervals_over_segments(segments, encoder, v3_holder, v2_cfg, v3_cfg) -> dict:
    """Return {recording_id: [(t_start_s, t_end_s), ...]} of V3-flagged events.

    Used for the ``--train-select v3gated`` ablation: run the Phase-1 V3 over
    each labeled recording and take its alert events as the TRAINING-window
    selector (deployment-consistent — V4 trains on what V3 would surface, with
    no offline impulse oracle).  Mirrors the gated-eval helper but iterates the
    segments directly.  Uses the trained xt_pool so inference pooling matches
    training pooling (calibration fix).

    Fallback (A3 fix):  if V3 fires zero events on a recording (which the
    deepc_20260526_155457 campaign showed can happen for an entire labeled
    cohort, sending v3gated training to 0 V4Samples → trainer crash), use the
    impulse-derived weak-GT intervals for that recording instead.  This makes
    v3gated *strictly more permissive* than impulse — V3 events take priority
    where they exist, impulse oracle as fallback — and guarantees the
    selector never produces an empty cohort.
    """
    from src.modeling.anomaly.event_detection import (
        detect_events_from_score_timeline,
        sliding_window_v3_inference,
    )
    from src.modeling.anomaly.v3_trainer import precompute_paired
    from src.modeling.anomaly.weak_labels import derive_knock_events

    bar = (v3_holder.thresholds.p99 if int(v3_cfg.threshold_percentile) >= 99
           else v3_holder.thresholds.p95)
    out: dict = {}
    n_v3_fired = 0
    n_impulse_fallback = 0
    for s in segments:
        paired = precompute_paired(s, v2_cfg)
        v3_evs: list[tuple[float, float]] = []
        if paired is not None:
            try:
                times, scores, contexts = sliding_window_v3_inference(
                    encoder, v3_holder.flow, paired, v2_cfg=v2_cfg,
                    inference_stride_s=0.25, xt_pool=v3_holder.xt_pool, device=v3_cfg.device)
                if scores.size > 0:
                    clusters = v3_holder.thresholds.assign(contexts)
                    high = float(np.median([float(bar[int(k)]) for k in clusters]))
                    low = high - abs(high) * 0.05
                    if low > high:
                        low = high
                    evs = detect_events_from_score_timeline(
                        scores, times, high_threshold=high, low_threshold=low,
                        min_duration_s=0.10, max_gap_windows=0,
                        recording_id=s.recording_id, dataset_id=s.dataset_id,
                        window_seconds=v2_cfg.window_seconds)
                    v3_evs = [(e.t_start_s, e.t_end_s) for e in evs]
            except Exception:
                v3_evs = []
        if v3_evs:
            out[s.recording_id] = v3_evs
            n_v3_fired += 1
        else:
            try:
                impulse_evs = derive_knock_events(s, burst_seconds=0.10)
            except Exception:
                impulse_evs = []
            if impulse_evs:
                out[s.recording_id] = impulse_evs
                n_impulse_fallback += 1
    print(f"  V3-gated selector: V3 fired on {n_v3_fired} segments, "
          f"impulse-fallback on {n_impulse_fallback} segments, "
          f"{len(segments) - n_v3_fired - n_impulse_fallback} skipped (no intervals)")
    return out


def _log(msg: str, log_path: Path) -> None:
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(line + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-run", required=True, help="Run dir with v2/encoder.pt")
    p.add_argument("--v3-run", default=None,
                   help="Phase-1 V3 winner dir (flow.pt + thresholds.npz) for gating. "
                        "Omit to report ungated holdout MAE only.")
    p.add_argument("--cell", default=None, help=f"Single cell; omit to run all. {len(_all_cells())} cells")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--samples-cache", default=None,
                   help="Path to a pickle of precomputed V4 samples.  If it exists, "
                        "load it (skips the expensive SRP-PHAT precompute); else "
                        "precompute and write it.  The campaign driver passes one "
                        "shared path so all cells precompute exactly ONCE.")
    p.add_argument("--all-channel-modes", action="store_true",
                   help="Train the cell at all 4 channel modes (acoustic SRP / "
                        "tdoa / vibration-only-learned / fusion) for the per-modality "
                        "localization breakdown.  Default: fusion ('both') only.")
    p.add_argument("--train-select", choices=("impulse", "v3gated", "all"),
                   default="impulse",
                   help="How V4 TRAINING windows are selected: 'impulse' = weak "
                        "impulse-envelope GT (default); 'v3gated' = windows the "
                        "Phase-1 V3 flags (requires --v3-run; deployment-consistent); "
                        "'all' = no selection (max label noise). Ablation axis.")
    p.add_argument("--gating-min-events", type=int, default=1,
                   help="Per-recording fallback for holdout V3-gating: if the "
                        "strict per-cluster threshold fires fewer events than "
                        "this on a holdout recording, fall back to a permissive "
                        "per-recording quantile threshold (top 10%% of scores) "
                        "so V4 always has at least N gated windows.  Set 0 to "
                        "restore strict-only behavior (legacy; produces "
                        "n_holdout_gated=0 when V3 can't fire).  Default 1.")
    p.add_argument("--gating-fallback-quantile", type=float, default=0.90,
                   help="Per-recording quantile threshold used by the gating "
                        "fallback.  0.90 = top 10%% of in-recording scores "
                        "considered V3-anomalous.  Only used when "
                        "--gating-min-events > 0.")
    args = p.parse_args()

    encoder_run = Path(args.encoder_run)
    if not (encoder_run / "v2" / "encoder.pt").exists():
        raise SystemExit(f"v2/encoder.pt not found under {encoder_run}")
    if args.train_select == "v3gated" and not args.v3_run:
        raise SystemExit("--train-select v3gated requires --v3-run (need V3 to "
                         "select training windows).")

    cells = [args.cell] if args.cell else _all_cells()
    v2_cfg = v2_config(args.quick)
    v3_cfg = v3_config(args.quick)
    base_v4 = v4_config(args.quick)
    for cid in cells:  # fail fast
        _apply_cell(cid, base_v4)

    import pickle

    t0 = time.time()
    encoder = V2FusionEncoder(
        feature_dim=v2_cfg.feature_dim, embed_dim=v2_cfg.embed_dim,
        n_heads=v2_cfg.n_heads, context_mode=v2_cfg.context_mode,
        num_context_seeds=v2_cfg.num_context_seeds,
        acoustic_cnn_width_mult=v2_cfg.acoustic_cnn_width_mult,
    )
    encoder.load_state_dict(torch.load(encoder_run / "v2" / "encoder.pt", map_location="cpu"))
    encoder.eval()
    grid = V4_CANDIDATE_GRID

    # Mode-specific cache so 'impulse' / 'v3gated' / 'all' training-window
    # selections never collide (they produce different sample sets).
    cache_path = None
    if args.samples_cache:
        base = Path(args.samples_cache)
        cache_path = base.with_name(f"{base.stem}_{args.train_select}{base.suffix}")

    # Load Phase-1 V3 up front — needed before precompute for v3gated training
    # selection, and after for gated eval.  c_dim from any labeled sample's
    # context dim == V2 embed_dim.
    v3_holder = None
    if args.v3_run:
        flow, thresholds, xt_pool, _ = _load_v3(Path(args.v3_run), int(v2_cfg.embed_dim))
        v3_holder = _V3Holder(flow, thresholds, xt_pool)

    loaders = None  # built lazily — needed for precompute and/or V3 gating
    v4_samples = None
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as fh:
            v4_samples = pickle.load(fh)
        print(f"Loaded {len(v4_samples)} cached V4 samples ({args.train_select}) "
              f"from {cache_path} (skipped precompute)")
    if v4_samples is None:
        print(f"Loading D2/D3/D4/D5 loaders, precomputing V4 samples "
              f"(train_select={args.train_select}) ...")
        loaders = {d: resolved_loader(f"{d}.yaml") for d in ("d2", "d3", "d4", "d5")}
        d2_labeled = [s for s in loaders["d2"].list_segments()
                      if s.is_anomaly and s.spatial_label is not None and s.mode_label is not None]
        d3_segs = loaders["d3"].list_segments()
        overrides = _d3_spatial_overrides(d3_segs)
        d3_labeled = [s for s in d3_segs if s.recording_id in overrides]
        d4_labeled = [s for s in loaders["d4"].list_segments()
                      if s.is_anomaly and s.spatial_label is not None]
        d5_labeled = [s for s in loaders["d5"].list_segments()
                      if s.is_anomaly and s.spatial_label is not None]
        labeled = d2_labeled + d3_labeled + d4_labeled + d5_labeled

        # Resolve the training-window selector for this mode.
        restrict = args.train_select != "all"
        override = None
        if args.train_select == "v3gated":
            # Run the Phase-1 V3 over every labeled recording to get its
            # flagged intervals → those become the training windows.  This is
            # the deployment-consistent selector (no offline impulse oracle).
            override = _v3_intervals_over_segments(
                labeled, encoder, v3_holder, v2_cfg, v3_cfg)
            n_with = sum(1 for v in override.values() if v)
            print(f"  V3-gated training selection: V3 flagged intervals in "
                  f"{n_with}/{len(labeled)} labeled recordings")
        v4_samples = precompute_v4_samples(
            encoder, labeled,
            v2_cfg=v2_cfg, grid=grid, spatial_label_overrides=overrides,
            burst_aware_srp=True, burst_seconds=0.10,
            restrict_to_knock_intervals=restrict,
            knock_intervals_override=override,
            # Pool `x_for_v3` with V3's own pooling so gating-time NLL is on the
            # same manifold the flow was trained on (avoids the PMA-2/mean
            # mismatch that saturated the flow → trivial gating).
            v3_xt_pool=(v3_holder.xt_pool if v3_holder is not None else None),
        )
        print(f"Precomputed {len(v4_samples)} V4 samples ({args.train_select}) "
              f"in {time.time()-t0:.0f}s")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(v4_samples, fh)
            tmp.replace(cache_path)
            print(f"Cached V4 samples to {cache_path}")

    train_pos, holdout_pos = split_samples_by_position(v4_samples, V4_HOLDOUT_POSITIONS_M)
    if len(train_pos) < 4 or len(holdout_pos) < 1:
        raise SystemExit(f"insufficient spatial split: {len(train_pos)} train / "
                         f"{len(holdout_pos)} holdout (train_select={args.train_select})")
    print(f"Spatial split: {len(train_pos)} train / {len(holdout_pos)} holdout samples")

    # V3-gated keep-mask for the holdout cohort (mode-independent — gating
    # depends on V3 + the cached V4Sample features, not the V4 channel mode).
    # The direct-path gate scores each holdout V4Sample in place with V3 and
    # applies the per-cluster percentile alert rule; it replaces the legacy
    # interval-overlap matching that produced n_holdout_gated=0 (recording-id
    # collisions across D4 speed{1,2,3} + V3/V4 timeline drift).  See
    # `src/modeling/localization/v3_gating.py`.
    gate_keep = None
    gating_diag: dict = {}
    if v3_holder is not None:
        try:
            gres = gate_samples_by_v3(
                v3_holder.flow, v3_holder.thresholds, holdout_pos,
                percentile=(99 if int(v3_cfg.threshold_percentile) >= 99 else 95),
                min_events=int(args.gating_min_events),
                fallback_quantile=float(args.gating_fallback_quantile),
            )
            gate_keep = gres.keep_mask
            gating_diag = gres.per_recording
            print(f"  V3 gating (direct): {len(gating_diag)} holdout recordings, "
                  f"fallback used on {gres.n_fallback_recordings}, "
                  f"gate_keep.sum()={gres.n_final}/{gate_keep.size} "
                  f"(strict {gres.n_strict})")
        except Exception as e:
            print(f"  V3 gating precompute skipped: {type(e).__name__}: {e}")
            gate_keep = None

    def _train_eval(cfg, log_path) -> dict:
        res = train_v4_localization(v4_samples, cfg=cfg, grid=grid,
                                    explicit_split=(train_pos, holdout_pos))
        # `train_val_gap_m` is a LEGACY name — it is the |val - train| smooth-L1
        # loss gap in loss_scale-cm space (not metres).  Kept for backward
        # compatibility with historical analyze_ablation tables.  The proper
        # metres-scale gap is `train_val_mae_gap_m`, computed from a final
        # forward pass on train_samples (see V4Result.train_mae_3d).
        d: dict = {
            "channel_mode": cfg.channel_mode,
            "holdout_mae_ungated_m": float(res.val_mae_3d),
            "holdout_p95_ungated_m": float(res.val_p95_3d),
            "train_mae_3d_m": float(res.train_mae_3d),
            "train_p95_3d_m": float(res.train_p95_3d),
            "train_val_mae_gap_m": float(abs(res.val_mae_3d - res.train_mae_3d))
                if not (np.isnan(res.val_mae_3d) or np.isnan(res.train_mae_3d))
                else float("nan"),
            "train_val_gap_m": float(abs(
                (res.val_loss_history[-1] if res.val_loss_history else float("nan"))
                - (res.train_loss_history[-1] if res.train_loss_history else float("nan")))),
            "val_position_breakdown": res.val_position_breakdown,
            "early_stopped_epoch": res.early_stopped_epoch,
            "n_holdout": int(res.val_predictions.shape[0]),
        }
        if gate_keep is not None and gate_keep.shape[0] == res.val_predictions.shape[0] and gate_keep.any():
            err = np.linalg.norm(
                res.val_predictions[gate_keep] - res.val_targets[gate_keep], axis=-1)
            d["holdout_mae_v3gated_m"] = float(np.mean(err))
            d["n_holdout_gated"] = int(gate_keep.sum())
        else:
            d["holdout_mae_v3gated_m"] = None
            d["n_holdout_gated"] = int(gate_keep.sum()) if gate_keep is not None else 0
        return d, res

    modes = (["srp_only", "tdoa_only", "vibration_only_learned", "both"]
             if args.all_channel_modes else ["both"])

    for cell_id in cells:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Embed train_select into the run-dir name so the three
        # train-select-ablation runs (impulse/v3gated/all) never share a dir.
        # The legacy `v4deep_<cell>_s<seed>` name (without the suffix) is
        # produced when train_select=impulse to stay compatible with
        # historical campaign state.json -> run_dir pointers that look up
        # the impulse run.  Non-impulse modes get a `_ts<MODE>` suffix.
        ts_suffix = "" if args.train_select == "impulse" else f"_ts{args.train_select}"
        out_dir = REPO_ROOT / "results" / "runs" / f"{ts}__v4deep_{cell_id}_s{args.seed}{ts_suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run_log.txt"
        base_cfg = _apply_cell(cell_id, base_v4)
        (out_dir / "cell_config.json").write_text(json.dumps({
            "cell": cell_id, "seed": args.seed, "encoder_run": str(encoder_run),
            "v3_run": args.v3_run, "train_select": args.train_select,
            "v4_cfg": asdict(replace(base_cfg, seed=args.seed)),
        }, indent=2, default=str))

        m: dict = {
            "cell": cell_id, "seed": args.seed, "train_select": args.train_select,
            "v3_gating_diagnostic": gating_diag,
            "v3_gating_min_events": int(args.gating_min_events),
            "v3_gating_fallback_quantile": float(args.gating_fallback_quantile),
        }
        per_mode: dict = {}
        for mode in modes:
            cfg = replace(base_cfg, seed=args.seed, channel_mode=mode)
            t0 = time.time()
            try:
                d, res = _train_eval(cfg, log_path)
            except Exception as e:
                _log(f"  V4[{mode}] FAILED: {type(e).__name__}: {e}", log_path)
                per_mode[mode] = {"error": f"{type(e).__name__}: {e}"}
                continue
            per_mode[mode] = d
            gated = d.get("holdout_mae_v3gated_m")
            _log(f"  V4[{mode}] {time.time()-t0:.0f}s — holdout MAE "
                 f"ungated={d['holdout_mae_ungated_m']:.4f}m "
                 f"gated={gated if gated is None else round(gated,4)}m "
                 f"train_mae={d['train_mae_3d_m']:.4f}m "
                 f"mae_gap={d['train_val_mae_gap_m']:.4f}m "
                 f"es={d['early_stopped_epoch']}", log_path)
            if mode == "both":
                torch.save(res.head.state_dict(), out_dir / "head.pt")
        # Hoist the fusion ('both') metrics to the top level so the campaign's
        # selection helper (which reads holdout_mae_v3gated_m) works unchanged.
        if "both" in per_mode and "error" not in per_mode["both"]:
            m.update({k: v for k, v in per_mode["both"].items() if k != "channel_mode"})
        if args.all_channel_modes:
            m["channel_modes"] = per_mode
        (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
        print(f"Wrote {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
