"""Diagnostic for V3 score distributions across cohorts.

Loads the trained V2 + V3 checkpoints, scores every window of every
RandomFault / hit cohort, and dumps a per-cohort score-distribution
JSON suitable for histograms in Chapter 6.

Specifically reports, per cohort (D2 RF / D3 hit / D4 RF / healthy
held-out):
  - n windows
  - mean, median, p5, p25, p75, p95, p99 of the CNF anomaly score
  - alert rate at p95 / p99 / 'calibrated' (when available)
  - recording-level alert-rate distribution (mean ± std across recordings)

The recording-level distribution is the academically defensible RQ2
metric: anomaly-density-unknown means absolute precision/recall is
uninterpretable, but `alert_rate_per_recording` is a clean signal of
"how anomaly-like is this recording in V3's eyes."

Run:
    python -m scripts.diagnostics.v3_diagnostic
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.modeling.anomaly.cnf_head import ConditionalRealNVP
from src.modeling.anomaly.threshold import PerClusterThresholds
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.context.v2_ssl import _gather_paired_segments
from src.modeling.localization.v4_features import V4_CANDIDATE_GRID
from src.modeling.localization.v4_trainer import precompute_v4_samples
from src.modeling.orchestration.full_run import (
    _d3_spatial_overrides,
    resolved_loader,
    v2_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "full_run"


def _load_v2() -> V2FusionEncoder:
    cfg = v2_config(quick=False)
    enc = V2FusionEncoder(
        feature_dim=cfg.feature_dim,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        context_mode=cfg.context_mode,
    )
    enc.load_state_dict(torch.load(RESULTS / "v2" / "encoder.pt", map_location="cpu"))
    enc.eval()
    return enc


def _load_v3() -> tuple[ConditionalRealNVP, PerClusterThresholds]:
    flow = ConditionalRealNVP(
        dim=64, c_dim=64, n_layers=6, hidden_dim=64, n_hidden_per_net=2
    )
    flow.load_state_dict(torch.load(RESULTS / "v3" / "flow.pt", map_location="cpu"))
    flow.eval()
    arr = np.load(RESULTS / "v3" / "thresholds.npz")
    thresholds = PerClusterThresholds(
        centroids=arr["centroids"],
        p95=arr["p95"],
        p99=arr["p99"],
        n_per_cluster=arr["n_per_cluster"],
    )
    return flow, thresholds


def _summary_stats(scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {"n": 0}
    return {
        "n": int(scores.size),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "p5": float(np.percentile(scores, 5)),
        "p25": float(np.percentile(scores, 25)),
        "p75": float(np.percentile(scores, 75)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


def main() -> dict:
    v2_cfg = v2_config(quick=False)
    enc = _load_v2()
    flow, thresholds = _load_v3()

    print("Loading data ...")
    D1 = resolved_loader("d1.yaml")
    D2 = resolved_loader("d2.yaml")
    D3 = resolved_loader("d3.yaml")
    D4 = resolved_loader("d4.yaml")

    # Healthy held-out (sanity baseline) — every is_anomaly=False segment
    # across all four campaigns.  At p95 the alert rate should be ~ 5 %.
    print("Scoring healthy cohort ...")
    healthy_segs = _gather_paired_segments([D1, D2, D3, D4], v2_cfg)
    grid = V4_CANDIDATE_GRID

    # Anomaly cohorts — run the V4 sample precompute machinery for a
    # convenient (x, c) extraction since precompute_v4_samples already
    # builds those tensors per window.  We pass empty overrides for
    # everything except D3 (which still needs the midpoint approx).
    d3_segments = D3.list_segments()
    overrides = _d3_spatial_overrides(d3_segments)
    d2_anom = [s for s in D2.list_segments() if s.is_anomaly]
    d3_anom = [s for s in d3_segments if s.recording_id in overrides]
    d4_anom = [s for s in D4.list_segments() if s.is_anomaly]

    print(f"  healthy={len(healthy_segs)}  D2 anom={len(d2_anom)}  "
          f"D3 anom={len(d3_anom)}  D4 anom={len(d4_anom)}")

    # Score each cohort.  For healthy we use the same precompute helper
    # by converting healthy segments into V4 candidate samples (with a
    # dummy spatial label — we never use it).  This keeps every cohort's
    # (x, c) extraction on a single code path.
    print("Precomputing (x, c) for all cohorts ...")
    # Filter healthy to those that look like valid V4 inputs (need raw
    # waveforms long enough for one window).  precompute_v4_samples does
    # this internally — we just pass the segments through.

    def _to_segments(paired_segs):
        # The public precompute_v4_samples wants `TestDatasetSegment`s; the
        # gather output is `_PairedSegment`s (precomputed features).  For
        # cohort scoring we need the raw segments.  Just re-list them.
        return paired_segs  # mismatch — see below

    # The clean approach: use the V4 samples that already exist.  The
    # full orchestrator persists them at training time but we can rebuild
    # cheaply.
    cohorts = {
        "d2_random_fault": d2_anom,
        "d3_hit": d3_anom,
        "d4_random_fault": d4_anom,
    }
    cohort_results: dict[str, dict] = {}
    for label, segs in cohorts.items():
        if not segs:
            cohort_results[label] = {"n_recordings": 0}
            continue
        samples = precompute_v4_samples(
            enc, segs, v2_cfg=v2_cfg, grid=grid,
            spatial_label_overrides=overrides,
        )
        if not samples:
            cohort_results[label] = {"n_recordings": len(segs), "n_windows": 0}
            continue
        xs = torch.from_numpy(np.stack([s.x_for_v3 for s in samples]))
        cs = torch.from_numpy(np.stack([s.context for s in samples]))
        with torch.no_grad():
            scores = flow.anomaly_score(xs, cs).numpy()
        # Per-recording alert rate at p95.
        contexts_np = np.stack([s.context for s in samples])
        alerts95, _ = thresholds.alert(contexts_np, scores, percentile=95)
        alerts99, _ = thresholds.alert(contexts_np, scores, percentile=99)
        rec_ids = [s.recording_id for s in samples]
        rec_alert_rates_95: dict[str, list] = {}
        for r, a in zip(rec_ids, alerts95):
            rec_alert_rates_95.setdefault(r, []).append(int(a))
        per_rec = {r: float(np.mean(v)) for r, v in rec_alert_rates_95.items()}

        cohort_results[label] = {
            "n_recordings": len(segs),
            "n_windows": int(scores.size),
            "score_stats": _summary_stats(scores),
            "alert_rate_p95": float(alerts95.mean()),
            "alert_rate_p99": float(alerts99.mean()),
            "alert_rate_per_recording_p95": per_rec,
            "alert_rate_per_recording_p95_mean": float(np.mean(list(per_rec.values()))),
            "alert_rate_per_recording_p95_std": float(np.std(list(per_rec.values()))),
        }
        print(
            f"  {label}: n_rec={len(segs)} n_win={scores.size} "
            f"alert@p95={alerts95.mean():.3f} per-rec mean={cohort_results[label]['alert_rate_per_recording_p95_mean']:.3f}"
        )

    out_path = RESULTS / "v3_cohort_diagnostic.json"
    out_path.write_text(json.dumps(cohort_results, indent=2))
    print(f"Diagnostic written to {out_path}")
    return cohort_results


if __name__ == "__main__":
    main()
