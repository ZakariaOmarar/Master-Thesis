"""End-to-end thesis orchestrator (b5_cma + three/four paradigms).

Single-process pipeline that produces every artefact and metric the
Chapter 6 paradigm-comparison tables need under one timestamped output
directory ``results/runs/<ts>__full_pipeline_b5_cma/``:

  Stage 0  cross-modal sync verification + correction audit
  Stage 1  V0 baselines: RQ2 anomaly reference (Khamaisi trio + KDE, pooled,
             acoustic + vibration) plus per-dataset LightGBM mode + SRP-PHAT
  Stage 2  V1 + V2 trained with the ``b5_cma`` intervention
             (cma_weight=0.5, cma_temperature=0.1)
             + V2 A1 ablation (drop_vibration) + modality-balance probe
  Stage 3  V3 three paradigms (V3-acoustic / V3-vibration / V3-fusion)
  Stage 4  V3 fusion depth — A2 unconditional + paired bootstrap V3 vs A2,
             synthetic anomaly ROC-AUC, transition FPR, per-cluster
             threshold breakdown, sliding-window event extraction
  Stage 5  V4 four paradigms (acoustic / vibration / tdoa_legacy / fusion)
             + V0 SRP-PHAT and accel-multilateration per dataset
  Stage 6  V4 fusion depth — A3 unconditional + paired bootstrap V4 vs A3
  Stage 7  V5.1 fan-noise robustness conditioning (speed one-hot SCADA)
  Stage 8  Inline late-fusion eval (LF AND / OR / score-weighted / MAX
             rows via ``rq2_three_paradigm_eval`` on this run dir)
  Stage 9  Inline RQ3 localisation paradigm eval (LF confidence-gated +
             LORO cross-validation via ``rq3_three_paradigm_eval``)

The module-level config builders (`resolved_loader`, `v1_config`, `v2_config`,
`v3_config`, `v4_config`, `_d3_spatial_overrides`) are the canonical source of
truth shared with the sibling orchestrators in this package and the
ablation scripts under ``scripts/`` — change configs here, not at the
caller.

Run::

    python -m src.modeling.orchestration.full_run           # full (~2 h CPU)
    python -m src.modeling.orchestration.full_run --quick   # smoke (~25 min)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...config import resolve_device
from ...config.dataset_registry import REGISTRY
from ...ingestion.test_dataset_loader import TestDatasetLoader
from ..anomaly import (
    train_v3_cnf,
)
from ..anomaly.event_detection import (
    detect_events_from_score_timeline,
    sliding_window_v3_inference,
    summarise_events,
)
from ..anomaly.synthetic_eval import evaluate_synthetic_anomaly_auc
from ..anomaly.threshold import per_cluster_alert_breakdown
from ..anomaly.v3_per_modality import V3AcousticOnlyAdapter, V3VibrationOnlyAdapter
from ..anomaly.v3_trainer import (
    encoder_level_transition_fpr,
    precompute_paired,
    transition_fpr,
)
from ..context.modality_probe import run_modality_balance_probe
from ..context.v1_ssl import train_v1_per_modality
from ..context.v2_ssl import train_v2_fusion
from ..eval import paired_bootstrap_test
from ..localization import (
    V4_CANDIDATE_GRID,
    V4Sample,
    event_aggregated_mae,
    precompute_v4_knock_event_samples,
    train_v4_localization,
)
from ..scada import d3_speed_lookup
from ._run_helpers import (
    REPO_ROOT,
    V4_HOLDOUT_POSITIONS_M,
    _all_segments,
    _audit_sync,
    _d3_spatial_overrides,
    _git_commit,
    _git_dirty,
    _make_logger,
    _train_one_v3,
    _train_one_v4,
    _v3_event_intervals_for_recordings,
    resolved_loader,
)

# V1-V4 hyperparameter builders live in `stage_configs`; they are imported (and
# re-exported) here so `main()` resolves them from this module's namespace,
# which keeps the multi-seed / hop-length drivers' monkeypatching of
# `full_run.vN_config` working.
from .stage_configs import v1_config, v2_config, v3_config, v4_config
from .v0_stage import run_v0, run_v0_multilateration

# Re-exported for callers that import these from this module (multi_seed,
# the V4 CV drivers, rq2 eval, baseline scripts).
__all__ = [
    "REPO_ROOT",
    "_all_segments",
    "_d3_spatial_overrides",
    "_v3_event_intervals_for_recordings",
    "main",
    "resolved_loader",
    "v1_config",
    "v2_config",
    "v3_config",
    "v4_config",
]


class PipelineContext:
    """Mutable state threaded through the V1-V5 pipeline stages."""

    def __init__(self, quick: bool) -> None:
        self.quick = quick
        self.stage_t0 = [time.time()]
        # Determinism — Python / NumPy / PyTorch RNGs pinned, deterministic
        # algorithms enabled where available (warn_only so non-deterministic
        # kernels fall through rather than crash).  BLAS thread scheduling
        # variance is bounded, not eliminated — `multi_seed.py` remains the
        # canonical mean ± std reporter for publication numbers.
        os.environ.setdefault("PYTHONHASHSEED", "0")
        torch.manual_seed(42)
        np.random.seed(42)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, AttributeError):
            # Older torch lacks the API; a few ops have no deterministic kernel
            # even under warn_only. Determinism here is best-effort.
            pass

        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = "full_pipeline_b5_cma" + ("_quick" if quick else "")
        out_dir = REPO_ROOT / "results" / "runs" / f"{timestamp}__{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("v1", "v2", "v3", "v4", "v5_1"):
            (out_dir / sub).mkdir(exist_ok=True)

        log = _make_logger(out_dir)
        metrics: dict = {
            "quick": quick,
            "variant": "b5_cma",
            "timestamp": timestamp,
            "stages": {},
            "timings_s": {},
        }
        log(f"REPO_ROOT = {REPO_ROOT}")
        log(f"out_dir = {out_dir}")
        log(f"quick = {quick}, variant = b5_cma")

        # Result-neutral feature cache (CWT/mel + vibration stacks).  Shared across
        # V1 / V2 / the A1 ablation / V4 — keyed on exact input bytes + params, so
        # it only ever speeds things up, never changes a number.  Enabled by default
        # at a stable repo-level dir; set HYDRO_FEATURE_CACHE_DIR (empty to disable).
        if os.environ.get("HYDRO_FEATURE_CACHE_DIR") is None:
            os.environ["HYDRO_FEATURE_CACHE_DIR"] = str(REPO_ROOT / ".feature_cache")
        _fc = os.environ.get("HYDRO_FEATURE_CACHE_DIR")
        log(f"feature cache = {_fc or '(disabled)'}")

        # ----------------------------------------------------------------- data
        # Loaders are built dynamically from the registry — adding a future
        # dataset is a YAML edit (configs/datasets/dN.yaml).  Subsets per
        # downstream stage are chosen below, not by hardcoding loaders here.
        LOADERS_BY_ID: dict[str, TestDatasetLoader] = {}
        for meta in REGISTRY:
            if not meta.root.exists():
                log(f"  skipping {meta.id} (root does not exist: {meta.root})")
                continue
            log(f"  loading {meta.id} from {meta.root.relative_to(REPO_ROOT)} ...")
            LOADERS_BY_ID[meta.id] = resolved_loader(f"{meta.id}.yaml")

        # SSL stages (V1, V2) — user direction: D5 has no operating-mode label
        # and is reserved for V3/V4 only, so the SSL cohort stays D1..D4.
        SSL_IDS = [i for i in ("d1", "d2", "d3", "d4") if i in LOADERS_BY_ID]
        SSL_LOADERS = [LOADERS_BY_ID[i] for i in SSL_IDS]
        # Anomaly stage (V3) — D5 contributes both healthy density-fit data and
        # held-out knock anomalies (label_scheme=d5_healthy_or_knock).
        ANOM_IDS = [i for i in ("d1", "d2", "d3", "d4", "d5") if i in LOADERS_BY_ID]
        ANOM_LOADERS = [LOADERS_BY_ID[i] for i in ANOM_IDS]
        log(f"SSL cohort: {SSL_IDS} | Anomaly cohort: {ANOM_IDS}")

        # Backward-compat per-loader names used by stage-specific helpers below
        # (transition FPR, labeled-segment gathering, RQ3 evaluation, ...).
        # Stage-specific code paths still index loaders by their fixed role
        # (D2 = rectangular bench rig, D3/D4 = circular rig, ...), so the
        # aliases stay but are now dict-driven and trivially extend to D5.
        D1 = LOADERS_BY_ID.get("d1")
        D2 = LOADERS_BY_ID.get("d2")
        D3 = LOADERS_BY_ID.get("d3")
        D4 = LOADERS_BY_ID.get("d4")
        D5 = LOADERS_BY_ID.get("d5")
        self.timestamp = timestamp
        self.label = label
        self.out_dir = out_dir
        self.log = log
        self.metrics = metrics
        self.LOADERS_BY_ID = LOADERS_BY_ID
        self.SSL_LOADERS = SSL_LOADERS
        self.ANOM_LOADERS = ANOM_LOADERS
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.D4 = D4
        self.D5 = D5
        # Stage outputs, populated as the pipeline runs (each stage sets the
        # ones it produces before a later stage reads them).
        self.v1_cfg: Any = None
        self.v2_cfg: Any = None
        self.v2_cfg_base: Any = None
        self.v3_cfg: Any = None
        self.v4_cfg: Any = None
        self.v1_a: Any = None
        self.v1_v: Any = None
        self.v2: Any = None
        self.v2_a1: Any = None
        self.v3_results: dict = {}
        self.v3: Any = None
        self.v4_samples: list = []
        self.overrides: Any = None
        self.grid: Any = None
        self.d3_segments: Any = None
        self.v4_results: dict = {}
        self.metrics_path: Any = None

    def stage_done(self, name: str) -> None:
        dt = time.time() - self.stage_t0[0]
        self.metrics['timings_s'][name] = dt
        self.log(f"=== stage '{name}' complete in {dt:.0f}s ===\n")
        self.stage_t0[0] = time.time()


def _run_optional_stages(ctx, run_sync_audit, run_v0_baselines):
    log, metrics = ctx.log, ctx.metrics
    SSL_LOADERS, ANOM_LOADERS = ctx.SSL_LOADERS, ctx.ANOM_LOADERS
    # ===================================================== S0 / S1 (opt-in)
    # The cross-modal sync audit and the V0 reference baselines are expensive
    # and off by default; enable them with run_sync_audit / run_v0_baselines
    # (or the matching CLI flags) for an ad-hoc or full-provenance run.
    if run_sync_audit:
        log("\n=== Stage 0 — cross-modal sync verification + correction audit ===")
        try:
            metrics["stages"]["sync_correction"] = _audit_sync(SSL_LOADERS, log)
        except Exception as e:
            log(f"sync audit failed: {type(e).__name__}: {e}")
            metrics["stages"]["sync_correction"] = {"skipped_reason": f"{type(e).__name__}: {e}"}
        ctx.stage_done("stage_0_sync")

    if run_v0_baselines:
        log("=== Stage 1 — V0 baselines (RQ2 anomaly trio+KDE / LightGBM / SRP-PHAT) ===")
        metrics["stages"]["v0"] = run_v0(SSL_LOADERS, log, anom_loaders=ANOM_LOADERS)
        ctx.stage_done("stage_1_v0")


def _stage_v1_v2(ctx: PipelineContext) -> None:
    quick = ctx.quick
    out_dir = ctx.out_dir
    log = ctx.log
    metrics = ctx.metrics
    SSL_LOADERS = ctx.SSL_LOADERS
    # ================================================================= S2
    log("=== Stage 2 — V1 + V2 with b5_cma intervention ===")
    v1_cfg = v1_config(quick)
    v2_cfg_base = v2_config(quick)
    # b5_cma: CMA loss on with cma_weight=0.5 and tightened temperature.
    # Source of truth: `scripts/campaigns/run_v1_v2_only.py::_apply_variant("b5_cma")`.
    v2_cfg = replace(v2_cfg_base, cma_weight=0.5, cma_temperature=0.1)
    log(f"V1 config: epochs={v1_cfg.epochs}, n_mels={v1_cfg.n_mels}, use_cwt={v1_cfg.use_cwt}")
    log(f"V2 config: epochs={v2_cfg.epochs}, cma_weight={v2_cfg.cma_weight}, "
        f"cma_temperature={v2_cfg.cma_temperature}, "
        f"context_mode={v2_cfg.context_mode}, "
        f"acoustic_dropout_p={v2_cfg.acoustic_dropout_p}, "
        f"vibration_dropout_p={v2_cfg.vibration_dropout_p}")

    log("V1 acoustic — training on D1+D2+D3+D4 healthy ...")
    t0 = time.time()
    v1_a = train_v1_per_modality(SSL_LOADERS, modality="acoustic", cfg=v1_cfg)
    log(f"  V1 acoustic {time.time()-t0:.0f}s — sanity NMI={v1_a.sanity_gate.get('nmi',0):.3f} "
        f"ARI={v1_a.sanity_gate.get('ari',0):.3f} purity={v1_a.sanity_gate.get('purity',0):.3f}")
    torch.save(v1_a.encoder.state_dict(), out_dir / "v1" / "acoustic.pt")
    metrics["stages"]["v1_acoustic"] = {
        "epochs": v1_cfg.epochs,
        "train_loss_final": v1_a.train_loss_history[-1],
        "val_loss_final": v1_a.val_loss_history[-1],
        "sanity_nmi": v1_a.sanity_gate.get("nmi", 0.0),
        "sanity_ari": v1_a.sanity_gate.get("ari", 0.0),
        "sanity_purity": v1_a.sanity_gate.get("purity", 0.0),
        "sanity_n_windows": v1_a.sanity_gate.get("n_windows", 0),
        "sanity_label_set": list(v1_a.sanity_gate.get("label_set", ())),
        "n_train_recordings": len(v1_a.train_recording_ids),
        "n_val_recordings": len(v1_a.val_recording_ids),
    }

    log("V1 vibration — training on D1+D2+D3+D4 healthy ...")
    t0 = time.time()
    v1_v = train_v1_per_modality(SSL_LOADERS, modality="vibration", cfg=v1_cfg)
    log(f"  V1 vibration {time.time()-t0:.0f}s — sanity NMI={v1_v.sanity_gate.get('nmi',0):.3f} "
        f"ARI={v1_v.sanity_gate.get('ari',0):.3f} purity={v1_v.sanity_gate.get('purity',0):.3f}")
    torch.save(v1_v.encoder.state_dict(), out_dir / "v1" / "vibration.pt")
    metrics["stages"]["v1_vibration"] = {
        "epochs": v1_cfg.epochs,
        "train_loss_final": v1_v.train_loss_history[-1],
        "val_loss_final": v1_v.val_loss_history[-1],
        "sanity_nmi": v1_v.sanity_gate.get("nmi", 0.0),
        "sanity_ari": v1_v.sanity_gate.get("ari", 0.0),
        "sanity_purity": v1_v.sanity_gate.get("purity", 0.0),
        "sanity_label_set": list(v1_v.sanity_gate.get("label_set", ())),
    }

    log("V2 — training fusion (inherits V1 weights) with b5_cma ...")
    t0 = time.time()
    v2 = train_v2_fusion(
        SSL_LOADERS, cfg=v2_cfg,
        v1_acoustic_state_dict=v1_a.encoder.state_dict(),
        v1_vibration_state_dict=v1_v.encoder.state_dict(),
    )
    log(f"  V2 {time.time()-t0:.0f}s — RQ1 NMI={v2.rq1.get('nmi',0):.3f} "
        f"ARI={v2.rq1.get('ari',0):.3f} purity={v2.rq1.get('purity',0):.3f}")
    torch.save(v2.encoder.state_dict(), out_dir / "v2" / "encoder.pt")
    torch.save(v2.projection.state_dict(), out_dir / "v2" / "projection.pt")
    metrics["stages"]["v2"] = {
        "epochs": v2_cfg.epochs,
        "train_loss_final": v2.train_loss_history[-1],
        "val_loss_final": v2.val_loss_history[-1],
        "train_simclr_final": v2.train_simclr_history[-1],
        "train_lmm_final": v2.train_lmm_history[-1],
        "rq1_nmi": v2.rq1.get("nmi", 0.0),
        "rq1_ari": v2.rq1.get("ari", 0.0),
        "rq1_purity": v2.rq1.get("purity", 0.0),
        "rq1_n_windows": v2.rq1.get("n_windows", 0),
        "rq1_label_set": list(v2.rq1.get("label_set", ())),
    }

    # V2 A1 ablation: drop vibration.
    #
    # IMPORTANT — the cross-modal alignment (CMA) loss aligns the acoustic
    # summary to the vibration summary.  With vibration zeroed, the vibration
    # summary is a per-dataset CONSTANT (it depends only on vib_xyz / dataset
    # idx, not on the zeroed signal), so the CMA contrastive collapses the
    # shared acoustic/context representation toward that single point — K-means
    # then finds one populated cluster and NMI is exactly 0.000.  The severed
    # retrain is acoustic-only by construction, so CMA (and vibration dropout)
    # are meaningless here and must be switched off; otherwise the ablation
    # measures representation collapse, not "can acoustic alone recover mode".
    log("V2 A1 ablation (drop_vibration=True) ...")
    t0 = time.time()
    a1_cfg = replace(v2_cfg, drop_vibration=True, cma_weight=0.0, vibration_dropout_p=0.0)
    v2_a1 = train_v2_fusion(
        SSL_LOADERS, cfg=a1_cfg,
        v1_acoustic_state_dict=v1_a.encoder.state_dict(),
        v1_vibration_state_dict=v1_v.encoder.state_dict(),
    )
    log(f"  V2 A1 {time.time()-t0:.0f}s — NMI={v2_a1.rq1.get('nmi',0):.3f}")
    if v2_a1.rq1.get("collapsed", False):
        log("  WARNING: V2 A1 context COLLAPSED (n_effective_clusters="
            f"{v2_a1.rq1.get('n_effective_clusters', '?')}, embedding_std="
            f"{v2_a1.rq1.get('embedding_std', float('nan')):.2e}); NMI is "
            "degenerate, not a valid ablation result.")
    metrics["stages"]["v2_a1_drop_vibration"] = {
        "rq1_nmi": v2_a1.rq1.get("nmi", 0.0),
        "rq1_ari": v2_a1.rq1.get("ari", 0.0),
        "rq1_purity": v2_a1.rq1.get("purity", 0.0),
        "collapsed": bool(v2_a1.rq1.get("collapsed", False)),
        "n_effective_clusters": v2_a1.rq1.get("n_effective_clusters"),
    }

    # V2 modality-balance probe (the headline Phase-B metric).
    try:
        from ..context.v2_ssl import _gather_labeled_segments
        labeled_segs = _gather_labeled_segments(SSL_LOADERS, v2_cfg)
        probe = run_modality_balance_probe(
            v2.encoder, labeled_segs, v2_cfg=v2_cfg, n_clusters=3, seed=v2_cfg.seed,
        )
        log(f"V2 modality probe: both NMI={probe.both.get('nmi',0):.3f}, "
            f"acoustic_only={probe.acoustic_only.get('nmi',0):.3f}, "
            f"vibration_only={probe.vibration_only.get('nmi',0):.3f}")
        metrics["stages"]["v2_modality_probe"] = {
            "both": {k: v for k, v in probe.both.items() if k not in ("confusion", "cluster_idx")},
            "acoustic_only": {k: v for k, v in probe.acoustic_only.items() if k not in ("confusion", "cluster_idx")},
            "vibration_only": {k: v for k, v in probe.vibration_only.items() if k not in ("confusion", "cluster_idx")},
            "n_segments": len(probe.healthy_segments_used),
            "delta_nmi_both_minus_acoustic": float(
                probe.both.get("nmi", 0.0) - probe.acoustic_only.get("nmi", 0.0)
            ),
        }
    except Exception as e:
        log(f"V2 modality probe skipped: {type(e).__name__}: {e}")
        metrics["stages"]["v2_modality_probe"] = {"skipped": f"{type(e).__name__}: {e}"}

    ctx.stage_done("stage_2_v1_v2_b5_cma")
    ctx.v1_cfg = v1_cfg
    ctx.v2_cfg = v2_cfg
    ctx.v2_cfg_base = v2_cfg_base
    ctx.v1_a = v1_a
    ctx.v1_v = v1_v
    ctx.v2 = v2
    ctx.v2_a1 = v2_a1


def _stage_v3_paradigms(ctx: PipelineContext) -> None:
    quick = ctx.quick
    out_dir = ctx.out_dir
    log = ctx.log
    metrics = ctx.metrics
    ANOM_LOADERS = ctx.ANOM_LOADERS
    v2_cfg = ctx.v2_cfg
    v1_a = ctx.v1_a
    v1_v = ctx.v1_v
    v2 = ctx.v2
    # ================================================================= S3
    log("=== Stage 3 — V3 three paradigms (acoustic / vibration / fusion) ===")
    v3_cfg = v3_config(quick)
    log(f"V3 config: epochs={v3_cfg.epochs}, K={v3_cfg.n_threshold_clusters}, "
        f"percentile={v3_cfg.threshold_percentile}")

    v3_acoustic_adapter = V3AcousticOnlyAdapter(v1_a.encoder)
    v3_vibration_adapter = V3VibrationOnlyAdapter(v1_v.encoder)

    v3_results: dict = {}
    pipelines = [
        ("acoustic", v3_acoustic_adapter),
        ("vibration", v3_vibration_adapter),
        ("fusion", v2.encoder),
    ]
    metrics["stages"]["v3_three_paradigms"] = {}
    for name, enc in pipelines:
        try:
            res = _train_one_v3(name, enc, ANOM_LOADERS, v2_cfg, v3_cfg, out_dir, log)
            v3_results[name] = res
            metrics["stages"]["v3_three_paradigms"][name] = {
                "val_nll_final": float(res.val_nll[-1]),
                "val_nll_min_final": float(res.val_nll_min[-1]) if res.val_nll_min else float("nan"),
                "val_nll_max_final": float(res.val_nll_max[-1]) if res.val_nll_max else float("nan"),
                "p95_per_cluster": res.thresholds.p95.tolist(),
                "p99_per_cluster": res.thresholds.p99.tolist(),
                "n_val_windows": int(res.val_scores.shape[0]),
                "n_threshold_fit_recordings": len(res.threshold_fit_recording_ids),
                "n_val_eval_recordings": len(res.val_recording_ids),
                "n_clusters_fit": int(res.thresholds.centroids.shape[0]),
            }
        except Exception as e:
            log(f"V3-{name} FAILED: {type(e).__name__}: {e}")
            metrics["stages"]["v3_three_paradigms"][name] = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()

    # Mirror v3-fusion artefacts into out_dir/v3/ so the legacy archive
    # layout used by `archive.py` and other downstream tools still finds
    # them at the expected path.
    if "fusion" in v3_results:
        import shutil
        for fname in ("flow.pt", "xt_pool.pt", "thresholds.npz", "val_eval.npz"):
            src = out_dir / "v3_fusion" / fname
            if src.exists():
                shutil.copy2(src, out_dir / "v3" / fname)

    ctx.stage_done("stage_3_v3_three_paradigms")
    ctx.v3_cfg = v3_cfg
    ctx.v3_results = v3_results


def _stage_v3_depth(ctx: PipelineContext) -> None:
    log = ctx.log
    metrics = ctx.metrics
    ANOM_LOADERS = ctx.ANOM_LOADERS
    D1 = ctx.D1
    D2 = ctx.D2
    D3 = ctx.D3
    D4 = ctx.D4
    D5 = ctx.D5
    v2_cfg = ctx.v2_cfg
    v3_cfg = ctx.v3_cfg
    v2 = ctx.v2
    v3_results = ctx.v3_results
    v3 = ctx.v3
    # ================================================================= S4
    log("=== Stage 4 — V3 fusion deeper diagnostics ===")
    if "fusion" not in v3_results:
        log("Skipped — V3 fusion failed in Stage 3")
        metrics["stages"]["v3_fusion_depth"] = {"skipped": "v3_fusion failed"}
    else:
        v3 = v3_results["fusion"]
        v3_depth: dict = {}
        v3_a2 = None

        # A2 unconditional flow ablation.
        try:
            log("V3 A2 ablation (unconditional flow) ...")
            t0 = time.time()
            a2_cfg = replace(v3_cfg, unconditional=True)
            v3_a2 = train_v3_cnf(v2.encoder, ANOM_LOADERS, v2_cfg=v2_cfg, v3_cfg=a2_cfg)
            log(f"  V3 A2 {time.time()-t0:.0f}s — val NLL={v3_a2.val_nll[-1]:.3f}")
            v3_depth["a2_unconditional"] = {
                "val_nll_final": float(v3_a2.val_nll[-1]),
                "p99_per_cluster": v3_a2.thresholds.p99.tolist(),
            }

            # Paired bootstrap V3 vs A2 on per-window NLL.
            if (
                v3.val_scores.shape[0] == v3_a2.val_scores.shape[0]
                and v3.val_scores.shape[0] >= 4
            ):
                # Resample held-out windows at the RECORDING level: per-window
                # NLLs within a recording are autocorrelated, so a window-level
                # paired test is pseudoreplicated (it overstates n and shrinks
                # the p-value).  V3 and A2 share the same recording split, so
                # v3's per-window recording ids align with both score arrays.
                # Falls back to window-level when ids are unavailable (legacy
                # mean-pool path).
                v3_groups = None
                rec_ids = getattr(v3, "val_recording_ids_per_window", None)
                if rec_ids is not None and len(rec_ids) == v3.val_scores.shape[0]:
                    v3_groups = np.asarray(rec_ids)
                pt = paired_bootstrap_test(
                    v3.val_scores, v3_a2.val_scores,
                    lower_is_better=True, n_boot=1000, seed=v3_cfg.seed,
                    groups=v3_groups,
                )
                log(f"  V3 vs A2 paired test: Δ={pt.delta_point:.3f} "
                    f"[{pt.delta_ci_low:.3f}, {pt.delta_ci_high:.3f}] "
                    f"p={pt.p_value_two_sided:.4f} "
                    f"({pt.method}, n_groups={pt.n_groups})")
                v3_depth["v3_vs_a2_paired_test"] = {
                    "delta_point": pt.delta_point,
                    "delta_ci95_low": pt.delta_ci_low,
                    "delta_ci95_high": pt.delta_ci_high,
                    "p_value_two_sided": pt.p_value_two_sided,
                    "direction": pt.direction,
                    # `n_paired` kept (windows) for backward compat with
                    # scripts/baselines/assemble_comparison.py; `n_recordings`
                    # is the block-bootstrap's true independent-unit count.
                    "n_paired": int(v3.val_scores.shape[0]),
                    "n_recordings": pt.n_groups,
                    "method": pt.method,
                }
        except Exception as e:
            log(f"V3 A2 / paired test skipped: {type(e).__name__}: {e}")
            v3_depth["a2_unconditional"] = {"skipped": f"{type(e).__name__}: {e}"}

        # Synthetic anomaly ROC-AUC across SNR ladder.
        try:
            if v3.val_contexts.shape[0] >= 4:
                import torch.utils.data as _tud

                from ..anomaly.v3_trainer import _extract_xc
                from ..context.v2_ssl import (
                    _collate,
                    _gather_paired_segments,
                    _PairedGroupedBatchSampler,
                    _PairedWindowedDataset,
                    _split_segments_by_recording,
                )
                segs_all = _gather_paired_segments(ANOM_LOADERS, v2_cfg)
                _, _val_segs_full = _split_segments_by_recording(
                    segs_all, v3_cfg.val_ratio, v3_cfg.seed,
                )
                _, val_segs_for_auc = _split_segments_by_recording(
                    _val_segs_full, v3_cfg.threshold_fit_val_ratio, v3_cfg.seed + 1,
                )
                val_ds = _PairedWindowedDataset(val_segs_for_auc, v2_cfg)
                if len(val_ds) > 0:
                    val_loader = _tud.DataLoader(
                        val_ds,
                        batch_sampler=_PairedGroupedBatchSampler(
                            val_ds, v3_cfg.batch_size, shuffle=False, seed=v3_cfg.seed,
                        ),
                        collate_fn=_collate,
                    )
                    x_val, c_val, _ = _extract_xc(
                        v2.encoder, val_loader, resolve_device(v3_cfg.device),
                        xt_pool=getattr(v3, "xt_pool", None),
                        anchor_norm=((v3.anchor_mean, v3.anchor_std)
                                     if getattr(v3, "anchor_mean", None) is not None else None),
                    )
                    auc = evaluate_synthetic_anomaly_auc(
                        v3.flow, x_val.numpy(), c_val.numpy(),
                        snr_db_list=(-10.0, -5.0, 0.0, 5.0, 10.0),
                        n_boot=500, seed=v3_cfg.seed,
                    )
                    log("V3 synthetic-anomaly ROC-AUC:")
                    for snr in sorted(auc.snr_db_to_auc):
                        log(f"  SNR={snr:+.1f} dB: AUC={auc.snr_db_to_auc[snr]:.3f}")
                    v3_depth["synthetic_anomaly_auc"] = {
                        "auc_conditional": auc.snr_db_to_auc,
                        "auc_conditional_ci_low": auc.snr_db_to_auc_ci_low,
                        "auc_conditional_ci_high": auc.snr_db_to_auc_ci_high,
                        "n_clean": auc.snr_db_to_n_clean,
                    }
                    if v3_a2 is not None:
                        auc_a2 = evaluate_synthetic_anomaly_auc(
                            v3_a2.flow, x_val.numpy(), c_val.numpy(),
                            snr_db_list=(-10.0, -5.0, 0.0, 5.0, 10.0),
                            n_boot=500, seed=v3_cfg.seed,
                        )
                        v3_depth["synthetic_anomaly_auc"]["auc_unconditional"] = auc_a2.snr_db_to_auc
        except Exception as e:
            log(f"V3 synthetic AUC skipped: {type(e).__name__}: {e}")

        # Transition FPR (within-D1 + cross-dataset same-mode).
        try:
            _val_eval_set = set(v3.val_recording_ids)
            paired_by: dict[tuple[str, str], list] = {}
            paired_fb: dict[tuple[str, str], list] = {}
            for L in (D1, D2):
                for s in L.list_segments():
                    if s.mode_label is None or s.is_anomaly:
                        continue
                    p = precompute_paired(s, v2_cfg)
                    if p is None:
                        continue
                    key = (s.dataset_id, s.mode_label)
                    qual = f"{Path(s.source_dir).name}/{s.recording_id}"
                    (paired_by if qual in _val_eval_set else paired_fb).setdefault(key, []).append(p)
            for key, segs in paired_fb.items():
                if key not in paired_by:
                    log(f"  transition fallback for {key} (training-pool recording)")
                    paired_by[key] = segs

            pairs = [
                ("d1_pump_to_turbine", ("d1", "Pump"), ("d1", "Turbine"), "raw"),
                ("d1_turbine_to_pump", ("d1", "Turbine"), ("d1", "Pump"), "raw"),
                ("d1_to_d2_pump", ("d1", "Pump"), ("d2", "Pump"), "encoder"),
                ("d1_to_d2_turbine", ("d1", "Turbine"), ("d2", "Turbine"), "encoder"),
            ]
            transition_results: dict[str, float] = {}
            for plabel, ka, kb, level in pairs:
                if ka not in paired_by or kb not in paired_by:
                    log(f"  transition {plabel}: skipped (missing source)")
                    continue
                seg_a, seg_b = paired_by[ka][0], paired_by[kb][0]
                _anchor_norm = ((v3.anchor_mean, v3.anchor_std)
                                if getattr(v3, "anchor_mean", None) is not None else None)
                if level == "raw":
                    out_pair = transition_fpr(
                        v2.encoder, v3.flow, v3.thresholds, seg_a, seg_b,
                        v2_cfg=v2_cfg, crossfade_seconds=1.0,
                        percentile=v3_cfg.threshold_percentile,
                        xt_pool=getattr(v3, "xt_pool", None), anchor_norm=_anchor_norm,
                    )
                else:
                    out_pair = encoder_level_transition_fpr(
                        v2.encoder, v3.flow, v3.thresholds, seg_a, seg_b,
                        v2_cfg=v2_cfg, n_crossfade_windows=8,
                        percentile=v3_cfg.threshold_percentile,
                        xt_pool=getattr(v3, "xt_pool", None), anchor_norm=_anchor_norm,
                    )
                transition_results[plabel] = out_pair["fpr"]
                log(f"  transition {plabel} ({level}): fpr={out_pair['fpr']:.3f} "
                    f"({out_pair['n_alerts']}/{out_pair['n_windows']})")
            v3_depth["transition_fpr"] = transition_results
        except Exception as e:
            log(f"V3 transition FPR skipped: {type(e).__name__}: {e}")

        # Per-cluster threshold breakdown on healthy holdout.
        try:
            if v3.val_scores.size > 0:
                breakdown = per_cluster_alert_breakdown(
                    v3.thresholds, v3.val_contexts, v3.val_scores,
                    percentile=v3_cfg.threshold_percentile,
                )
                v3_depth["per_cluster_breakdown_healthy"] = breakdown
                log("V3 per-cluster healthy alert rates: "
                    + " | ".join(
                        f"k{k}={r['n_alerts']}/{r['n']}({r['alert_rate']:.2f})"
                        for k, r in breakdown["per_cluster"].items() if r["n"] > 0
                    ))
        except Exception as e:
            log(f"V3 per-cluster breakdown skipped: {type(e).__name__}: {e}")

        # Sliding-window event extraction per anomaly cohort.
        try:
            log("V3 sliding-window event extraction (stride=0.25s) ...")
            n_clusters = int(v3.thresholds.centroids.shape[0])
            per_cluster_p95 = v3.thresholds.p95.tolist()
            per_cluster_p90: list[float] = []
            if v3.val_scores.size > 0 and v3.val_contexts.shape[0] > 0:
                assign = v3.thresholds.assign(v3.val_contexts)
                for k in range(n_clusters):
                    mask = assign == k
                    if int(mask.sum()) > 4:
                        per_cluster_p90.append(float(np.percentile(v3.val_scores[mask], 90)))
                    else:
                        per_cluster_p90.append(float(np.percentile(v3.val_scores, 90)))
            else:
                per_cluster_p90 = list(v3.thresholds.p95)

            cohort_event_summary: dict = {}
            for cohort_label, loader, dsid in (
                ("d2_random_fault", D2, "d2"),
                ("d3_hit", D3, "d3"),
                ("d4_random_fault", D4, "d4"),
            ):
                events: list = []
                n_rec = 0
                for s in loader.list_segments():
                    if not s.is_anomaly:
                        continue
                    seg = precompute_paired(s, v2_cfg)
                    if seg is None:
                        continue
                    try:
                        times_s, scores, contexts = sliding_window_v3_inference(
                            v2.encoder, v3.flow, seg,
                            v2_cfg=v2_cfg, inference_stride_s=0.25,
                            xt_pool=v3.xt_pool,
                            device=resolve_device(v3_cfg.device),
                            anchor_norm=((v3.anchor_mean, v3.anchor_std)
                                         if getattr(v3, "anchor_mean", None) is not None else None),
                        )
                    except Exception as inner:
                        log(f"    {cohort_label}/{s.recording_id} skipped: {inner}")
                        continue
                    if scores.size == 0:
                        continue
                    w_clusters = v3.thresholds.assign(contexts)
                    rec_high = float(np.median([per_cluster_p95[int(k)] for k in w_clusters]))
                    rec_low = float(np.median([per_cluster_p90[int(k)] for k in w_clusters]))
                    if rec_low > rec_high:
                        rec_low = rec_high
                    evs = detect_events_from_score_timeline(
                        scores, times_s, high_threshold=rec_high, low_threshold=rec_low,
                        min_duration_s=0.10, max_gap_windows=0,
                        recording_id=s.recording_id, dataset_id=dsid,
                        window_seconds=v2_cfg.window_seconds,
                    )
                    events.extend(evs)
                    n_rec += 1
                summary = summarise_events(events)
                summary["n_recordings_audited"] = n_rec
                cohort_event_summary[cohort_label] = summary
                log(f"  {cohort_label}: n_rec={n_rec} n_events={summary['n_events']}")
            v3_depth["sliding_window_events"] = cohort_event_summary
        except Exception as e:
            log(f"V3 sliding-window events skipped: {type(e).__name__}: {e}")

        # Real-anomaly detection vs weak knock GT.  Scores V3's detected
        # events against impulse-derived knock intervals on the sparse-anomaly
        # cohorts (precision / recall / F1 / onset-timing).  This is a
        # prerequisite metric: V4 cannot be trusted until V3 detects the real
        # anomalies well.
        try:
            from ..anomaly.event_detection import v3_real_anomaly_detection
            rf_segments = []
            for loader, _dsid in ((D4, "d4"), (D2, "d2"), (D5, "d5")):
                if loader is None:
                    continue
                rf_segments += [s for s in loader.list_segments() if s.is_anomaly]
            real_det = v3_real_anomaly_detection(
                v2.encoder, v3.flow, v3.thresholds, rf_segments,
                v2_cfg=v2_cfg, percentile=v3_cfg.threshold_percentile,
                inference_stride_s=0.25, xt_pool=v3.xt_pool, device=v3_cfg.device,
                anchor_norm=((v3.anchor_mean, v3.anchor_std)
                             if getattr(v3, "anchor_mean", None) is not None else None),
            )
            v3_depth["real_anomaly_detection"] = real_det
            metrics["stages"]["v3_real_anomaly"] = real_det
            log(f"V3 real-anomaly: P={real_det['precision']:.3f} "
                f"R={real_det['recall']:.3f} F1={real_det['f1']:.3f} "
                f"onset_err={real_det['median_onset_error_s']:.3f}s "
                f"(scored {real_det['n_recordings_scored']} recs, "
                f"{real_det['n_recordings_no_weak_gt']} had no weak GT, "
                f"{real_det['n_recordings_inference_failed']} inference-failed)")
        except Exception as e:
            log(f"V3 real-anomaly detection skipped: {type(e).__name__}: {e}")
            metrics["stages"]["v3_real_anomaly"] = {"skipped": f"{type(e).__name__}: {e}"}

        metrics["stages"]["v3_fusion_depth"] = v3_depth
    ctx.stage_done("stage_4_v3_depth")


def _stage_v4_paradigms(ctx: PipelineContext) -> None:
    quick = ctx.quick
    out_dir = ctx.out_dir
    log = ctx.log
    metrics = ctx.metrics
    D2 = ctx.D2
    D3 = ctx.D3
    D4 = ctx.D4
    D5 = ctx.D5
    v2_cfg = ctx.v2_cfg
    v2 = ctx.v2
    v3 = ctx.v3
    # ================================================================= S5
    log("=== Stage 5 — V4 four paradigms + V0 classical localisation ===")
    d2_labeled = [
        s for s in D2.list_segments()
        if s.is_anomaly and s.spatial_label is not None and s.mode_label is not None
    ]
    d3_segments = D3.list_segments()
    overrides = _d3_spatial_overrides(d3_segments)
    d3_labeled = [s for s in d3_segments if s.recording_id in overrides]
    d4_labeled = [
        s for s in D4.list_segments() if s.is_anomaly and s.spatial_label is not None
    ]
    # B1 (2026-05-23) — D5 knock recordings carry parsed positions
    # (`d5_healthy_or_knock` scheme → spatial_label set, is_anomaly=True) and
    # `d5.yaml` explicitly lists them as V4 localisation labels, but the
    # cohort builder previously concatenated only D2/D3/D4 and silently
    # dropped D5.  Including D5 roughly doubles the position inventory.
    d5_labeled = [
        s for s in D5.list_segments() if s.is_anomaly and s.spatial_label is not None
    ] if D5 is not None else []
    n_positions = len({
        tuple(np.round(s.spatial_label, 3)) for s in
        (d2_labeled + d3_labeled + d4_labeled + d5_labeled)
        if s.spatial_label is not None
    })
    log(f"Labelled segments: D2={len(d2_labeled)} D3={len(d3_labeled)} "
        f"D4={len(d4_labeled)} D5={len(d5_labeled)} | distinct positions={n_positions}")

    grid = V4_CANDIDATE_GRID

    log("Precomputing per-knock V4 samples (SRP-PHAT + accel TDOA + V2 c_t) ...")
    t0 = time.time()
    v4_samples = precompute_v4_knock_event_samples(
        v2.encoder, d2_labeled + d3_labeled + d4_labeled + d5_labeled,
        v2_cfg=v2_cfg, grid=grid,
        spatial_label_overrides=overrides,
        # Pool `x_for_v3` with V3's pooling so gating-time NLL matches the
        # manifold the flow was trained on (avoids the PMA-2/mean saturation).
        v3_xt_pool=getattr(v3, "xt_pool", None),
        # Append V3's impulse+spectral anchor so the gate scores the exact input
        # the conditional flow trained on (RQ2 anchor injection).
        v3_anchor_norm=((v3.anchor_mean, v3.anchor_std)
                        if getattr(v3, "anchor_mean", None) is not None else None),
    )
    log(f"  {len(v4_samples)} per-knock V4 samples in {time.time()-t0:.0f}s")
    n_with_multilat = sum(1 for s in v4_samples if s.multilat_xyz is not None)
    log(f"  multilat init available on {n_with_multilat}/{len(v4_samples)} samples")
    metrics["stages"]["v4_samples"] = {
        "n_total": len(v4_samples),
        "n_with_multilat": n_with_multilat,
    }

    v4_cfg = v4_config(quick)
    v4_paradigms = [
        ("acoustic", "srp_only"),
        ("vibration", "vibration_only_learned"),
        ("vibration_tdoa_only_legacy", "tdoa_only"),
        ("fusion", "both"),
    ]
    v4_results: dict = {}
    metrics["stages"]["v4_four_paradigms"] = {}
    if len(v4_samples) < 4:
        log(f"V4 SKIPPED — only {len(v4_samples)} labelled samples (need ≥4)")
        metrics["stages"]["v4_four_paradigms"] = {"skipped": True}
    else:
        for name, mode in v4_paradigms:
            try:
                if mode == "vibration_only_learned" and n_with_multilat < len(v4_samples):
                    log(f"  filtering to {n_with_multilat} samples with multilat for {name}")
                    samples_in = [s for s in v4_samples if s.multilat_xyz is not None]
                else:
                    samples_in = v4_samples
                res = _train_one_v4(name, mode, samples_in, grid, v4_cfg, out_dir, log)
                v4_results[name] = res
                metrics["stages"]["v4_four_paradigms"][name] = {
                    "channel_mode": mode,
                    "val_mae_3d": float(res.val_mae_3d),
                    "val_mae_ci95_low": float(res.val_mae_ci_low),
                    "val_mae_ci95_high": float(res.val_mae_ci_high),
                    "val_p95_3d": float(res.val_p95_3d),
                    "n_val": int(res.val_predictions.shape[0]),
                    "n_train_recordings": len(res.train_recording_ids),
                    "n_val_recordings": len(res.val_recording_ids),
                    "train_loss_final": float(res.train_loss_history[-1]) if res.train_loss_history else float("nan"),
                    "val_loss_final": float(res.val_loss_history[-1]) if res.val_loss_history else float("nan"),
                }
            except Exception as e:
                log(f"V4-{name} FAILED: {type(e).__name__}: {e}")
                metrics["stages"]["v4_four_paradigms"][name] = {"error": f"{type(e).__name__}: {e}"}
                traceback.print_exc()

        if "fusion" in v4_results:
            import shutil
            for fname in ("head.pt", "val_predictions.npz"):
                src = out_dir / "v4_fusion" / fname
                if src.exists():
                    shutil.copy2(src, out_dir / "v4" / fname)

    log("V0 accel multilateration per dataset ...")
    metrics["stages"]["v0_multilateration"] = run_v0_multilateration(
        [D2, D3, D4], overrides, log
    )
    ctx.v4_cfg = v4_cfg
    ctx.v4_samples = v4_samples
    ctx.overrides = overrides
    ctx.grid = grid
    ctx.d3_segments = d3_segments
    ctx.v4_results = v4_results


def _stage_v4_holdout(ctx: PipelineContext) -> None:
    log = ctx.log
    metrics = ctx.metrics
    v3_cfg = ctx.v3_cfg
    v4_cfg = ctx.v4_cfg
    v3_results = ctx.v3_results
    v4_samples = ctx.v4_samples
    grid = ctx.grid
    # ============================================== Stage 5b — spatial holdout
    # Train fusion V4 on all positions EXCEPT the reserved held-out set,
    # then report holdout MAE (localise-an-unseen-position), the V3-GATED
    # holdout MAE (deployment-faithful: V4 only fires on V3-flagged windows),
    # and V0 multilateration on the same held-out samples.  Gated by
    # `gated_v4_eval` (CLI --ungated disables the gating column).
    log("=== Stage 5b — V4 spatial-holdout + V3-gated eval ===")
    try:
        from ..localization import split_samples_by_position
        train_pos, holdout_pos = split_samples_by_position(
            v4_samples, V4_HOLDOUT_POSITIONS_M,
        )
        n_hold_pos = len({tuple(np.round(s.target_xyz, 3)) for s in holdout_pos})
        log(f"  spatial split: {len(train_pos)} train / {len(holdout_pos)} holdout "
            f"samples across {n_hold_pos} held-out positions")
        if len(train_pos) >= 4 and len(holdout_pos) >= 1:
            sh: dict = {"n_train_samples": len(train_pos),
                        "n_holdout_samples": len(holdout_pos),
                        "n_holdout_positions": n_hold_pos,
                        "holdout_positions_m": [list(p) for p in V4_HOLDOUT_POSITIONS_M]}
            res_sh = train_v4_localization(
                v4_samples, cfg=replace(v4_cfg, channel_mode="both"), grid=grid,
                explicit_split=(train_pos, holdout_pos),
            )
            sh["holdout_mae_ungated_m"] = float(res_sh.val_mae_3d)
            sh["holdout_p95_ungated_m"] = float(res_sh.val_p95_3d)
            sh["holdout_train_val_gap_m"] = float(abs(
                (res_sh.val_loss_history[-1] if res_sh.val_loss_history else float("nan"))
                - (res_sh.train_loss_history[-1] if res_sh.train_loss_history else float("nan"))
            ))
            log(f"  holdout MAE (ungated) = {res_sh.val_mae_3d:.4f} m")

            # V3-gated holdout: keep holdout windows V3 flags as anomalous,
            # scored directly on each sample's cached x_for_v3 + context (the
            # direct-path gate).  Replaces the legacy interval-overlap matching
            # (`_v3_event_intervals_for_recordings` → `window_overlaps_any`),
            # whose recording-id collisions + timeline drift produced
            # n_holdout_gated=0.  See `..localization.v3_gating`.
            gated_v4_eval = True
            if gated_v4_eval and "fusion" in v3_results:
                try:
                    from ..localization.v3_gating import gate_samples_by_v3
                    v3f = v3_results["fusion"]
                    gres = gate_samples_by_v3(
                        v3f.flow, v3f.thresholds, holdout_pos,
                        percentile=(99 if int(v3_cfg.threshold_percentile) >= 99 else 95),
                    )
                    keep = gres.keep_mask
                    sh["v3_gating_diagnostic"] = gres.per_recording
                    if keep.shape[0] == res_sh.val_predictions.shape[0] and keep.any():
                        # keep is a numpy bool array but holdout_pos is a list ->
                        # comprehension, never holdout_pos[keep].  Event-aggregate
                        # the kept knocks (same aggregation as the ungated headline).
                        kept = [s for s, k in zip(holdout_pos, keep) if k]
                        g_mae, _g_agg, g_recs = event_aggregated_mae(
                            res_sh.val_predictions[keep], res_sh.val_targets[keep], kept)
                        sh["holdout_mae_v3gated_m"] = float(g_mae) if np.isfinite(g_mae) else None
                        sh["n_holdout_gated"] = int(keep.sum())
                        sh["n_holdout_gated_recordings"] = int(g_recs)
                        log(f"  holdout MAE (V3-gated, event-agg) = {g_mae:.4f} m on "
                            f"{int(keep.sum())}/{keep.shape[0]} V3-flagged knocks "
                            f"across {int(g_recs)} recordings")
                    else:
                        sh["holdout_mae_v3gated_m"] = None
                        sh["n_holdout_gated"] = int(keep.sum()) if keep.size else 0
                        sh["n_holdout_gated_recordings"] = 0
                        log("  V3-gated holdout: no knocks flagged (or shape mismatch)")
                except Exception as e:
                    log(f"  V3-gated holdout skipped: {type(e).__name__}: {e}")
                    sh["holdout_mae_v3gated_m"] = None

            # V0 multilateration on the same held-out samples (recording-level).
            try:
                hold_recs = {s.recording_id for s in holdout_pos}
                v0_errs: list[float] = []
                for payload in metrics["stages"]["v0_multilateration"].values():
                    for rec in payload.get("per_recording", []):
                        if rec.get("recording_id") in hold_recs and "error_m" in rec:
                            v0_errs.append(float(rec["error_m"]))
                sh["holdout_v0_multilat_mae_m"] = float(np.mean(v0_errs)) if v0_errs else None
                sh["n_holdout_v0"] = len(v0_errs)
                if v0_errs and "holdout_mae_v3gated_m" in sh and sh["holdout_mae_v3gated_m"] is not None:
                    sh["delta_v4gated_minus_v0_m"] = sh["holdout_mae_v3gated_m"] - float(np.mean(v0_errs))
                    log(f"  V0 multilat on holdout = {np.mean(v0_errs):.4f} m | "
                        f"Δ(V4gated − V0) = {sh['delta_v4gated_minus_v0_m']:+.4f} m")
            except Exception as e:
                log(f"  V0-on-holdout skipped: {type(e).__name__}: {e}")
            metrics["stages"]["v4_spatial_holdout"] = sh
        else:
            log("  spatial-holdout SKIPPED — insufficient train/holdout samples")
            metrics["stages"]["v4_spatial_holdout"] = {
                "skipped": f"train={len(train_pos)} holdout={len(holdout_pos)}"}
    except Exception as e:
        log(f"Stage 5b spatial-holdout skipped: {type(e).__name__}: {e}")
        metrics["stages"]["v4_spatial_holdout"] = {"skipped_reason": f"{type(e).__name__}: {e}"}

    ctx.stage_done("stage_5_v4_four_paradigms")


def _stage_v4_depth(ctx: PipelineContext) -> None:
    log = ctx.log
    metrics = ctx.metrics
    v4_cfg = ctx.v4_cfg
    v4_samples = ctx.v4_samples
    grid = ctx.grid
    v4_results = ctx.v4_results
    # ================================================================= S6
    log("=== Stage 6 — V4 fusion deeper diagnostics ===")
    if "fusion" not in v4_results:
        log("Skipped — V4 fusion failed or skipped in Stage 5")
        metrics["stages"]["v4_fusion_depth"] = {"skipped": "v4_fusion unavailable"}
    else:
        v4 = v4_results["fusion"]
        v4_depth: dict = {}
        try:
            log("V4 A3 ablation (unconditional=True) ...")
            t0 = time.time()
            a3_cfg = replace(v4_cfg, unconditional=True)
            v4_a3 = train_v4_localization(v4_samples, cfg=a3_cfg, grid=grid)
            log(f"  V4 A3 {time.time()-t0:.0f}s — val MAE={v4_a3.val_mae_3d:.4f} m")
            v4_depth["a3_unconditional"] = {
                "val_mae_3d": float(v4_a3.val_mae_3d),
                "val_mae_ci95_low": float(v4_a3.val_mae_ci_low),
                "val_mae_ci95_high": float(v4_a3.val_mae_ci_high),
                "val_p95_3d": float(v4_a3.val_p95_3d),
            }
            if (
                v4.val_predictions.shape[0] == v4_a3.val_predictions.shape[0]
                and v4.val_predictions.shape[0] >= 4
            ):
                err_v4 = np.linalg.norm(v4.val_predictions - v4.val_targets, axis=-1).astype(np.float64)
                err_a3 = np.linalg.norm(v4_a3.val_predictions - v4_a3.val_targets, axis=-1).astype(np.float64)
                # Resample at the RECORDING level (block bootstrap): V4 and A3
                # share the same recording-level val split (same seed), so the
                # per-window recording ids align across both error arrays.
                # Window-level resampling here would be pseudoreplication — the
                # per-knock errors within a recording are correlated.  Falls
                # back to window-level only if the group ids are unavailable.
                v4_groups = (
                    np.asarray(v4.val_groups)
                    if len(v4.val_groups) == err_v4.shape[0] else None
                )
                pt = paired_bootstrap_test(
                    err_v4, err_a3, lower_is_better=True, n_boot=1000,
                    seed=v4_cfg.seed, groups=v4_groups,
                )
                log(f"  V4 vs A3 paired test: Δ_MAE={pt.delta_point*1000:.1f} mm "
                    f"[{pt.delta_ci_low*1000:.1f}, {pt.delta_ci_high*1000:.1f}] mm "
                    f"p={pt.p_value_two_sided:.4f} "
                    f"({pt.method}, n_groups={pt.n_groups})")
                v4_depth["v4_vs_a3_paired_test"] = {
                    "delta_mae_m": pt.delta_point,
                    "delta_mae_ci95_low_m": pt.delta_ci_low,
                    "delta_mae_ci95_high_m": pt.delta_ci_high,
                    "p_value_two_sided": pt.p_value_two_sided,
                    "direction": pt.direction,
                    "n_paired_windows": int(err_v4.shape[0]),
                    "n_recordings": pt.n_groups,
                    "method": pt.method,
                }
        except Exception as e:
            log(f"V4 A3 skipped: {type(e).__name__}: {e}")
            v4_depth["a3_unconditional"] = {"skipped": f"{type(e).__name__}: {e}"}

        metrics["stages"]["v4_fusion_depth"] = v4_depth
    ctx.stage_done("stage_6_v4_depth")


def _stage_v5(ctx: PipelineContext) -> None:
    quick = ctx.quick
    out_dir = ctx.out_dir
    log = ctx.log
    metrics = ctx.metrics
    D4 = ctx.D4
    v4_samples = ctx.v4_samples
    grid = ctx.grid
    d3_segments = ctx.d3_segments
    # ================================================================= S7
    log("=== Stage 7 — V5.1 fan-noise robustness conditioning ===")
    try:
        speed_lookup = {**d3_speed_lookup(d3_segments), **d3_speed_lookup(D4.list_segments())}
        if speed_lookup and len(v4_samples) >= 4:
            log(f"  speed lookup: {len(speed_lookup)} recordings")
            scada_dim = next(iter(speed_lookup.values())).shape[0]
            v5_1_samples = []
            for s in v4_samples:
                scada = speed_lookup.get(s.recording_id)
                if scada is None:
                    scada = np.zeros(scada_dim, dtype=np.float32)
                v5_1_samples.append(V4Sample(
                    srp_volume=s.srp_volume, tdoa_tokens=s.tdoa_tokens,
                    context=s.context, x_for_v3=s.x_for_v3,
                    target_xyz=s.target_xyz, scada=scada,
                    mode_label=s.mode_label, recording_id=s.recording_id,
                    source_dir=s.source_dir, dataset_id=s.dataset_id,
                    multilat_xyz=s.multilat_xyz,
                ))
            v5_1_cfg = v4_config(quick, scada_dim=scada_dim)
            t0 = time.time()
            v5_1 = train_v4_localization(v5_1_samples, cfg=v5_1_cfg, grid=grid)
            log(f"  V5.1 {time.time()-t0:.0f}s — val MAE={v5_1.val_mae_3d:.3f} m "
                f"[{v5_1.val_mae_ci_low:.3f}, {v5_1.val_mae_ci_high:.3f}]")
            torch.save(v5_1.head.state_dict(), out_dir / "v5_1" / "head_speed.pt")
            metrics["stages"]["v5_1"] = {
                "scada_dim": scada_dim,
                "val_mae_3d": float(v5_1.val_mae_3d),
                "val_mae_ci95_low": float(v5_1.val_mae_ci_low),
                "val_mae_ci95_high": float(v5_1.val_mae_ci_high),
                "val_p95_3d": float(v5_1.val_p95_3d),
            }
        else:
            log("V5.1 SKIPPED — no speed segments or insufficient V4 samples")
            metrics["stages"]["v5_1"] = {"skipped": True}
    except Exception as e:
        log(f"V5.1 skipped: {type(e).__name__}: {e}")
        metrics["stages"]["v5_1"] = {"skipped_reason": f"{type(e).__name__}: {e}"}
    ctx.stage_done("stage_7_v5_1")


def _stage_deep_vs_simple(ctx: PipelineContext) -> None:
    log = ctx.log
    metrics = ctx.metrics
    v3_cfg = ctx.v3_cfg
    v3_results = ctx.v3_results
    # =============================================== deep-vs-simple summary
    # One-stop comparison block for the thesis chapter: each deep stage's
    # headline metric vs the closed-form / classical baseline already
    # computed elsewhere in this pipeline.  A near-zero or negative Δ on
    # any row means the deep model has not earned its complexity for that
    # stage on the current cohort.
    log("\n=== Deep-vs-simple summary ===")
    deep_vs_simple: dict = {}

    # V3 fusion vs KDE-on-c_t (per-cluster gaussian_kde on V2 c_t buckets).
    try:
        from ..anomaly.kde_baseline import fit_and_score_kde_on_ct

        v3_fus = v3_results.get("fusion")
        if v3_fus is not None and v3_fus.train_x is not None and v3_fus.val_x is not None:
            kde_res = fit_and_score_kde_on_ct(
                x_train=v3_fus.train_x,
                c_train=v3_fus.train_contexts,
                x_val=v3_fus.val_x,
                c_val=v3_fus.val_contexts,
                n_clusters=v3_cfg.n_threshold_clusters,
                seed=v3_cfg.seed,
            )
            v3_nll_val = float(v3_fus.val_nll[-1]) if v3_fus.val_nll else float("nan")
            delta_nll = v3_nll_val - kde_res.val_nll_mean  # CNF wins if < 0
            deep_vs_simple["anomaly"] = {
                "deep_model": "V3 CNF (fusion)",
                "simple_baseline": "KDE-on-c_t per K-means cluster",
                "deep_val_nll_mean": v3_nll_val,
                "simple_val_nll_mean": kde_res.val_nll_mean,
                "delta_deep_minus_simple": delta_nll,
                "deep_wins": delta_nll < 0.0,
                "n_clusters_used": kde_res.n_clusters_used,
                "kde_n_per_cluster_train": kde_res.n_per_cluster_train.tolist(),
                "kde_n_per_cluster_val": kde_res.n_per_cluster_val.tolist(),
            }
            log(f"  V3 vs KDE: V3 NLL={v3_nll_val:.3f} | KDE NLL={kde_res.val_nll_mean:.3f} | "
                f"Δ={delta_nll:+.3f} ({'V3 wins' if delta_nll < 0 else 'KDE wins'})")
        else:
            deep_vs_simple["anomaly"] = {"skipped": "v3_fusion train_x/val_x unavailable"}
    except Exception as e:
        log(f"  V3 vs KDE skipped: {type(e).__name__}: {e}")
        deep_vs_simple["anomaly"] = {"skipped_reason": f"{type(e).__name__}: {e}"}

    # V4 fusion vs V0 accel-TDOA multilateration (closed-form, no trainable
    # parameters).  Both already computed above; this block surfaces the Δ.
    try:
        v4_fus_metrics = metrics["stages"].get("v4_four_paradigms", {}).get("fusion", {})
        v0_multi = metrics["stages"].get("v0_multilateration", {})
        v4_mae = v4_fus_metrics.get("val_mae_3d")
        # Pool V0 MAE across D2/D3/D4 (mean of per-dataset means, weighted
        # by n_successful).  Single value comparable to V4's pooled val MAE.
        v0_errs: list[float] = []
        for payload in v0_multi.values():
            n = int(payload.get("n_successful", 0))
            mean = payload.get("mean_error_m")
            if n > 0 and isinstance(mean, (int, float)) and not (isinstance(mean, float) and (mean != mean)):
                v0_errs.extend([float(mean)] * n)
        v0_mae = float(np.mean(v0_errs)) if v0_errs else float("nan")
        if isinstance(v4_mae, (int, float)) and v0_errs:
            delta_mae = float(v4_mae) - v0_mae  # V4 wins if < 0
            deep_vs_simple["localisation"] = {
                "deep_model": "V4 fusion head",
                "simple_baseline": "V0 accel-TDOA multilateration (closed-form)",
                "deep_val_mae_m": float(v4_mae),
                "simple_val_mae_m": v0_mae,
                "delta_deep_minus_simple_m": delta_mae,
                "deep_wins": delta_mae < 0.0,
                "n_recordings_v0": len(v0_errs),
            }
            log(f"  V4 vs V0 multilat: V4 MAE={v4_mae:.3f} m | V0 MAE={v0_mae:.3f} m | "
                f"Δ={delta_mae:+.3f} m ({'V4 wins' if delta_mae < 0 else 'V0 wins'})")
        else:
            deep_vs_simple["localisation"] = {"skipped": "v4_fusion or v0_multilateration unavailable"}
    except Exception as e:
        log(f"  V4 vs V0 multilat skipped: {type(e).__name__}: {e}")
        deep_vs_simple["localisation"] = {"skipped_reason": f"{type(e).__name__}: {e}"}

    # V2 fusion clustering vs LightGBM mode classifier (D1).  Metrics are not
    # directly comparable (NMI vs macro-F1) so we report both side-by-side
    # without a Δ; the thesis chapter can frame the comparison qualitatively.
    try:
        v2_metrics = metrics["stages"].get("v2", {})
        v0_lgbm = metrics["stages"].get("v0", {}).get("v0_lgbm_d1", {})
        if v2_metrics and v0_lgbm and "val_macro_f1" in v0_lgbm:
            deep_vs_simple["mode_clustering"] = {
                "deep_model": "V2 fusion (K=3 K-means on c_t)",
                "simple_baseline": "V0 LightGBM mode classifier (D1)",
                "deep_rq1_nmi": v2_metrics.get("rq1_nmi"),
                "deep_rq1_purity": v2_metrics.get("rq1_purity"),
                "simple_val_macro_f1": v0_lgbm.get("val_macro_f1"),
                "note": "metrics not directly comparable; reported side-by-side",
            }
            log(f"  V2 vs V0 LGBM: V2 NMI={v2_metrics.get('rq1_nmi', 0):.3f} | "
                f"V0 F1={v0_lgbm.get('val_macro_f1', 0):.3f} (different units)")
    except Exception as e:
        log(f"  V2 vs V0 LGBM skipped: {type(e).__name__}: {e}")
        deep_vs_simple["mode_clustering"] = {"skipped_reason": f"{type(e).__name__}: {e}"}

    metrics["deep_vs_simple"] = deep_vs_simple
    ctx.stage_done("stage_7b_deep_vs_simple")


def _persist(ctx: PipelineContext) -> None:
    quick = ctx.quick
    timestamp = ctx.timestamp
    label = ctx.label
    out_dir = ctx.out_dir
    log = ctx.log
    metrics = ctx.metrics
    v1_cfg = ctx.v1_cfg
    v2_cfg = ctx.v2_cfg
    v3_cfg = ctx.v3_cfg
    v4_cfg = ctx.v4_cfg
    # ============================================================ persist
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log(f"\nWrote interim metrics to {metrics_path}")

    manifest = {
        "timestamp": timestamp,
        "label": label,
        "variant": "b5_cma",
        "quick": quick,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "host": socket.gethostname(),
        "configs": {
            "v1_cfg": asdict(v1_cfg),
            "v2_cfg": asdict(v2_cfg),
            "v3_cfg": asdict(v3_cfg),
            "v4_cfg": asdict(v4_cfg),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"Wrote manifest to {out_dir / 'manifest.json'}")
    ctx.metrics_path = metrics_path


def _stage_rq2_lf(ctx: PipelineContext) -> None:
    out_dir = ctx.out_dir
    log = ctx.log
    metrics = ctx.metrics
    # ================================================================= S8
    log("\n=== Stage 8 — late-fusion (AND / OR / score-weighted / MAX) eval ===")
    try:
        rel_run = str(out_dir.relative_to(REPO_ROOT))
        rc = subprocess.call(
            [sys.executable, "-m", "src.modeling.eval.rq2_three_paradigm_eval",
             "--v3-three-run", rel_run, "--source-run", rel_run],
            cwd=str(REPO_ROOT),
        )
        if rc == 0:
            comparison = out_dir / "rq2_paradigm_comparison.json"
            if comparison.exists():
                metrics["stages"]["rq2_paradigm_comparison"] = json.loads(comparison.read_text())
                log("  rq2 eval done — see rq2_paradigm_comparison.{json,md}")
        else:
            log(f"  rq2 eval exited with rc={rc}")
            metrics["stages"]["rq2_paradigm_comparison"] = {"skipped_reason": f"rc={rc}"}
    except Exception as e:
        log(f"rq2 eval skipped: {type(e).__name__}: {e}")
        metrics["stages"]["rq2_paradigm_comparison"] = {"skipped_reason": f"{type(e).__name__}: {e}"}
    ctx.stage_done("stage_8_rq2_lf_eval")


def _stage_rq3_lf(ctx: PipelineContext) -> None:
    out_dir = ctx.out_dir
    log = ctx.log
    metrics = ctx.metrics
    metrics_path = ctx.metrics_path
    # ================================================================= S9
    log("=== Stage 9 — RQ3 LF confidence-gated localisation eval ===")
    try:
        rel_run = str(out_dir.relative_to(REPO_ROOT))
        rc = subprocess.call(
            [sys.executable, "-m", "src.modeling.eval.rq3_three_paradigm_eval",
             "--v4-three-run", rel_run],
            cwd=str(REPO_ROOT),
        )
        if rc == 0:
            comp = out_dir / "rq3_paradigm_comparison.json"
            if comp.exists():
                metrics["stages"]["rq3_paradigm_comparison"] = json.loads(comp.read_text())
                log("  rq3 eval done — see rq3_paradigm_comparison.{json,md}")
        else:
            log(f"  rq3 eval exited with rc={rc}")
            metrics["stages"]["rq3_paradigm_comparison"] = {"skipped_reason": f"rc={rc}"}
    except Exception as e:
        log(f"rq3 eval skipped: {type(e).__name__}: {e}")
        metrics["stages"]["rq3_paradigm_comparison"] = {"skipped_reason": f"{type(e).__name__}: {e}"}
    ctx.stage_done("stage_9_rq3_lf_eval")

    metrics_path.write_text(json.dumps(metrics, indent=2))
    log(f"Final metrics written to {metrics_path}")

    total = sum(metrics["timings_s"].values())
    log(f"\nTotal wall-clock: {total:.0f}s ({total/60:.1f} min)")


def main(
    quick: bool = False,
    *,
    run_sync_audit: bool = False,
    run_v0_baselines: bool = False,
) -> dict:
    """Run the end-to-end V1-V5 pipeline and return the metrics dict.

    Args:
        quick: halve epoch counts at every stage for a smoke run.
        run_sync_audit: also run the opt-in cross-modal sync audit (Stage 0).
        run_v0_baselines: also run the opt-in V0 reference baselines (Stage 1).
    """
    ctx = PipelineContext(quick)
    _run_optional_stages(ctx, run_sync_audit, run_v0_baselines)
    _stage_v1_v2(ctx)
    _stage_v3_paradigms(ctx)
    ctx.v3 = ctx.v3_results.get('fusion')
    _stage_v3_depth(ctx)
    _stage_v4_paradigms(ctx)
    _stage_v4_holdout(ctx)
    _stage_v4_depth(ctx)
    _stage_v5(ctx)
    _stage_deep_vs_simple(ctx)
    _persist(ctx)
    _stage_rq2_lf(ctx)
    _stage_rq3_lf(ctx)
    ctx.metrics_path.write_text(json.dumps(ctx.metrics, indent=2))
    ctx.log(f'Final metrics written to {ctx.metrics_path}')
    total = sum(ctx.metrics['timings_s'].values())
    ctx.log(f'\nTotal wall-clock: {total:.0f}s ({total / 60:.1f} min)')
    return ctx.metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--quick", action="store_true",
        help="Halve epoch counts at every stage for a smoke run (~25 min CPU).",
    )
    p.add_argument(
        "--sync-audit", action="store_true",
        help="Also run the opt-in cross-modal sync audit (Stage 0).",
    )
    p.add_argument(
        "--v0-baselines", action="store_true",
        help="Also run the opt-in V0 reference baselines (Stage 1).",
    )
    args = p.parse_args()
    main(quick=args.quick, run_sync_audit=args.sync_audit, run_v0_baselines=args.v0_baselines)
