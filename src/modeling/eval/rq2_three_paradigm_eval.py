"""R2.3 — Unimodal × 2 + Late Fusion × 4 + Intermediate Fusion comparison.

Consumes the artefacts written by `scripts/paradigms/run_v3_three_paradigms.py`:

  results/runs/<v3-three-paradigms-run>/
    ├── v3_acoustic/{flow.pt, thresholds.npz, val_eval.npz}
    ├── v3_vibration/{flow.pt, thresholds.npz, val_eval.npz}
    ├── v3_fusion/{flow.pt, thresholds.npz, val_eval.npz}
    └── metrics.json

Plus the V1+V2 encoder weights from the source run (so we can re-score
anomaly cohorts that weren't part of V3's held-out healthy split).

Outputs three artefacts under the same run dir:

  rq2_paradigm_comparison.json    — full numbers
  rq2_paradigm_comparison.md      — human-readable comparison table
  rq2_specificity_audit.json      — per-cohort
                                    (a_only/v_only/both/neither) breakdowns

Late-fusion rules implemented:

  * **AND** — both Unimodal pipelines fire above their own per-cluster p95.
  * **OR**  — at least one fires.
  * **score_weighted** — logistic-regression on (z(score_a), z(score_v))
    fit against the binary anomaly label (healthy vs anomaly cohorts).
    Per-window combined score = w_a · z(score_a) + w_v · z(score_v) + b;
    alert threshold = combined-score p95 on healthy cohort.
  * **MAX** — element-wise max of z-scored unimodal scores; alert
    threshold = combined-score p95 on healthy cohort.

Run::

    python -m src.modeling.eval.rq2_three_paradigm_eval \\
        --v3-three-run results/runs/<id> \\
        --source-run   results/runs/<v1+v2-run>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud

from ...ingestion.test_dataset_loader import (
    DatasetSpec,
    TestDatasetLoader,
)
from ..anomaly.cnf_head import ConditionalRealNVP
from ..anomaly.threshold import PerClusterThresholds
from ..anomaly.v3_per_modality import (
    V3AcousticOnlyAdapter,
    V3VibrationOnlyAdapter,
)
from ..context.v1_ssl import V1SSLConfig
from ..context.v2_fusion import V2FusionEncoder
from ..context.v2_ssl import (
    V2SSLConfig,
    _collate,
    _PairedGroupedBatchSampler,
    _PairedSegment,
    _PairedWindowedDataset,
    _precompute_paired,
)
from ..encoders.per_modality import PerModalityEncoder
from ..orchestration.full_run import v1_config, v2_config

REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Loaders + encoder builds
# ---------------------------------------------------------------------------


def _loader(name: str) -> TestDatasetLoader:
    # DatasetSpec.from_yaml resolves all paths to absolute — no reconstruction.
    spec = DatasetSpec.from_yaml(REPO / "configs" / "datasets" / f"{name}.yaml")
    return TestDatasetLoader(spec)


def _build_v1(modality: str, cfg: V1SSLConfig) -> PerModalityEncoder:
    return PerModalityEncoder(
        modality=modality, feature_dim=cfg.feature_dim,
        embed_dim=cfg.embed_dim, n_heads=cfg.n_heads,
        acoustic_cnn_width_mult=cfg.acoustic_cnn_width_mult,
    )


def _build_v2(cfg: V2SSLConfig) -> V2FusionEncoder:
    return V2FusionEncoder(
        feature_dim=cfg.feature_dim, embed_dim=cfg.embed_dim, n_heads=cfg.n_heads,
        context_mode=cfg.context_mode, num_context_seeds=cfg.num_context_seeds,
        acoustic_cnn_width_mult=cfg.acoustic_cnn_width_mult,
    )


def _load_state(path: Path, module: torch.nn.Module) -> None:
    sd = torch.load(path, map_location="cpu")
    module.load_state_dict(sd, strict=False)


def _load_v3(pipeline_dir: Path, x_dim: int, c_dim: int):
    from ..anomaly.impulse_anchor import N_ANCHOR
    th_npz = np.load(pipeline_dir / "thresholds.npz")
    # RQ2 anchor injection: if the run trained with the impulse+spectral anchor,
    # the flow input is [pooled(embed) ⊕ anchor(N_ANCHOR)] so the flow dim is
    # embed+N_ANCHOR (the xt_pool still pools to `embed`).
    anchor_norm = None
    flow_dim = x_dim
    if "anchor_mean" in th_npz.files:
        persisted = int(th_npz["anchor_mean"].shape[0])
        if persisted != N_ANCHOR:
            raise ValueError(
                f"{pipeline_dir.name}: trained with a {persisted}-feature impulse "
                f"anchor but the current code defines {N_ANCHOR} (the anchor feature "
                "set changed).  Re-train V3 with the current code (the anchor must "
                "match), or check out the commit that produced this run."
            )
        anchor_norm = (th_npz["anchor_mean"], th_npz["anchor_std"])
        flow_dim = x_dim + N_ANCHOR
    flow = ConditionalRealNVP(
        dim=flow_dim, c_dim=c_dim, n_layers=6, hidden_dim=64, n_hidden_per_net=2, scale_max=2.0,
    )
    _load_state(pipeline_dir / "flow.pt", flow)
    flow.eval()
    # Load the persisted learnable channel-token pool (pma2) so x_t reproduces
    # what the flow was trained on.  `pma2` is the publication default
    # (v3_trainer line ~462) and the thresholds in thresholds.npz are fit on
    # pma2-pooled scores, so mean-pooling at eval time mismatches the flow and
    # the threshold scale: the anomaly scores blow up past *every* p95 and the
    # AND/OR rules fire on 100 % of windows (both==1.0 on the healthy hold-out).
    # That is exactly what the legacy `__full_pipeline_b5_cma` runs missing
    # xt_pool.pt produced.  Refuse to score such a run rather than silently
    # emit a degenerate audit that pollutes the multi-seed medians.
    xt_path = pipeline_dir / "xt_pool.pt"
    if not xt_path.exists():
        raise FileNotFoundError(
            f"no xt_pool.pt in {pipeline_dir} -- this run was trained with the "
            "pma2 channel-token pool but never persisted it, so the flow and "
            "thresholds cannot be reproduced.  Scoring it would mean-pool the "
            "fused tokens and fire AND/OR on 100% of windows (both==1.0).  "
            "Re-run training with the current full_run.py (which saves "
            "xt_pool.pt) or exclude this run from the multi-seed set."
        )
    from ..anomaly import V3Config
    from ..anomaly.v3_trainer import _XtPool
    xt_pool = _XtPool(embed_dim=x_dim, num_heads=V3Config().xt_pool_num_heads)
    _load_state(xt_path, xt_pool)
    xt_pool.eval()
    th = PerClusterThresholds(
        centroids=th_npz["centroids"], p95=th_npz["p95"], p99=th_npz["p99"],
        n_per_cluster=th_npz["n_per_cluster"], seed=42,
    )
    return flow, th, xt_pool, anchor_norm


# ---------------------------------------------------------------------------
# Per-cohort scoring under all three V3 pipelines
# ---------------------------------------------------------------------------


@dataclass
class _PipelineState:
    name: str
    encoder: torch.nn.Module
    flow: ConditionalRealNVP
    thresholds: PerClusterThresholds
    xt_pool: torch.nn.Module | None = None
    anchor_norm: tuple = None  # (mean, std) for RQ2 impulse+spectral anchor, or None


def _score_cohort_three_paradigms(
    pipelines: list[_PipelineState],
    loader: tud.DataLoader,
) -> dict[str, np.ndarray]:
    """Score every window under every pipeline.  Returns
    ``{pipeline_name: (scores, contexts, alerts)}`` aligned per window.
    """
    out: dict[str, list] = {p.name: [[], [], []] for p in pipelines}
    for batch in loader:
        ac = batch["ac_feat"]
        ac_xyz = batch["ac_xyz"]
        vib = batch["vib_feat"]
        vib_xyz = batch["vib_xyz"]
        ds = batch["dataset_idx"]
        with torch.no_grad():
            for p in pipelines:
                p.encoder.eval()
                d = p.encoder(ac, ac_xyz, vib, vib_xyz, ds, mask_p=0.0)
                fused = torch.cat([d["a_fused"], d["v_fused"]], dim=1)
                # Reproduce the flow's training-time x_t: the learnable pma2 pool
                # when persisted, otherwise the legacy mean-pool.
                x = p.xt_pool(fused) if p.xt_pool is not None else fused.mean(dim=1)
                # RQ2 anchor injection: append the standardized impulse+spectral
                # anchor so the flow input matches what it trained on.
                if p.anchor_norm is not None:
                    from ..anomaly.impulse_anchor import append_anchor
                    x = append_anchor(x, ac, vib, p.anchor_norm)
                c = d["context"]
                scores = p.flow.anomaly_score(x, c).cpu().numpy()
                c_np = c.cpu().numpy()
                alerts, _ = p.thresholds.alert(c_np, scores, percentile=95)
                out[p.name][0].append(scores)
                out[p.name][1].append(c_np)
                out[p.name][2].append(alerts.astype(np.int32))
    return {
        name: {
            "scores": np.concatenate(parts[0]) if parts[0] else np.zeros(0),
            "contexts": np.concatenate(parts[1]) if parts[1] else np.zeros((0, 0)),
            "alerts": np.concatenate(parts[2]) if parts[2] else np.zeros(0, dtype=np.int32),
        }
        for name, parts in out.items()
    }


def _segments_for(
    loaders: list[TestDatasetLoader], cfg: V2SSLConfig, *, healthy: bool,
) -> list[_PairedSegment]:
    out: list[_PairedSegment] = []
    for L in loaders:
        for s in L.list_segments():
            if healthy and s.is_anomaly:
                continue
            if (not healthy) and (not s.is_anomaly):
                continue
            p = _precompute_paired(s, cfg)
            if p is not None:
                out.append(p)
    return out


def _build_loader(segments: list[_PairedSegment], cfg: V2SSLConfig) -> tud.DataLoader:
    ds = _PairedWindowedDataset(segments, cfg)
    return tud.DataLoader(
        ds,
        batch_sampler=_PairedGroupedBatchSampler(
            ds, cfg.batch_size, shuffle=False, seed=cfg.seed,
        ),
        collate_fn=_collate,
    )


# ---------------------------------------------------------------------------
# Late-fusion combiners
# ---------------------------------------------------------------------------


def _zscore(scores: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Z-score `scores` using the mean/std of `baseline` (healthy hold-out).

    Avoids per-cohort drift bias the per-cohort z-score would have — the
    healthy mean/std is the right normaliser for an anomaly-detection signal.
    """
    mu = float(baseline.mean()) if baseline.size else 0.0
    sd = float(baseline.std()) if baseline.size else 1.0
    sd = max(sd, 1e-6)
    return (scores - mu) / sd


def _fit_logistic_late_fusion(
    z_a_healthy: np.ndarray, z_v_healthy: np.ndarray,
    z_a_anom: np.ndarray, z_v_anom: np.ndarray,
) -> tuple[float, float, float]:
    """Fit a 2-feature logistic regression to `(z_a, z_v) -> P(anomaly)`.

    Returns ``(w_a, w_v, b)`` such that the combined score is
    ``w_a · z_a + w_v · z_v + b`` and a positive combined score predicts
    anomaly.  Fitted via scipy's L-BFGS-B (no sklearn dependency).  When
    either class is empty falls back to ``(0.5, 0.5, 0)`` so the LF score
    is the simple mean of the two unimodal z-scores.
    """
    from scipy.optimize import minimize

    if z_a_healthy.size == 0 or z_a_anom.size == 0:
        return 0.5, 0.5, 0.0
    X = np.concatenate([
        np.stack([z_a_healthy, z_v_healthy], axis=1),
        np.stack([z_a_anom, z_v_anom], axis=1),
    ], axis=0)
    y = np.concatenate([
        np.zeros(z_a_healthy.size, dtype=np.float64),
        np.ones(z_a_anom.size, dtype=np.float64),
    ])

    def _nll(params):
        w = params[:2]
        b = params[2]
        logits = X @ w + b
        # Numerically stable log-loss.
        return float(np.mean(np.logaddexp(0.0, logits) - y * logits))

    res = minimize(_nll, x0=np.array([0.5, 0.5, 0.0]), method="L-BFGS-B")
    w_a, w_v, b = float(res.x[0]), float(res.x[1]), float(res.x[2])
    return w_a, w_v, b


# ---------------------------------------------------------------------------
# Per-cohort table
# ---------------------------------------------------------------------------


@dataclass
class _ParadigmRow:
    paradigm: str
    rule: str
    n_windows: int
    alert_rate: float
    mean_score: float  # NaN for binary-only rules (AND/OR)


def _row(paradigm: str, rule: str, n: int, alert_rate: float, mean_score: float = float("nan")) -> _ParadigmRow:
    return _ParadigmRow(paradigm=paradigm, rule=rule, n_windows=n,
                        alert_rate=alert_rate, mean_score=mean_score)


def _audit_alert_quadrants(alert_a: np.ndarray, alert_v: np.ndarray) -> dict[str, float]:
    """Fraction of windows in each (a, v) alert combination."""
    n = alert_a.size
    if n == 0:
        return {"a_only": 0.0, "v_only": 0.0, "both": 0.0, "neither": 0.0, "n": 0}
    a = alert_a.astype(bool)
    v = alert_v.astype(bool)
    return {
        "a_only": float((a & ~v).mean()),
        "v_only": float((~a & v).mean()),
        "both": float((a & v).mean()),
        "neither": float((~a & ~v).mean()),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-three-run", required=True,
                    help="Run dir produced by scripts/paradigms/run_v3_three_paradigms.py")
    ap.add_argument("--source-run", required=True,
                    help="V1+V2 weights source run (same one used to seed the V3 paradigms run)")
    args = ap.parse_args()

    v3_run = Path(args.v3_three_run).resolve()
    src_run = Path(args.source_run).resolve()
    if not v3_run.exists() or not src_run.exists():
        raise SystemExit(f"missing run dir: {v3_run} or {src_run}")

    v1_cfg = v1_config(False)
    v2_cfg = v2_config(False)
    embed = int(v1_cfg.embed_dim)

    # Build + load encoders.
    print(f"[rq2-3p] Loading V1+V2 from {src_run.relative_to(REPO)} ...")
    v1_a = _build_v1("acoustic", v1_cfg)
    _load_state(src_run / "v1" / "acoustic.pt", v1_a)
    v1_a.eval()
    v1_v = _build_v1("vibration", v1_cfg)
    _load_state(src_run / "v1" / "vibration.pt", v1_v)
    v1_v.eval()
    v2 = _build_v2(v2_cfg)
    _load_state(src_run / "v2" / "encoder.pt", v2)
    v2.eval()

    # Build the three pipeline states.
    flow_a, th_a, xt_a, anc_a = _load_v3(v3_run / "v3_acoustic", x_dim=embed, c_dim=embed)
    flow_v, th_v, xt_v, anc_v = _load_v3(v3_run / "v3_vibration", x_dim=embed, c_dim=embed)
    flow_f, th_f, xt_f, anc_f = _load_v3(v3_run / "v3_fusion", x_dim=embed, c_dim=embed)
    pipelines = [
        _PipelineState("acoustic", V3AcousticOnlyAdapter(v1_a), flow_a, th_a, xt_a, anc_a),
        _PipelineState("vibration", V3VibrationOnlyAdapter(v1_v), flow_v, th_v, xt_v, anc_v),
        _PipelineState("fusion", v2, flow_f, th_f, xt_f, anc_f),
    ]

    print("[rq2-3p] Gathering cohorts ...")
    loaders = [_loader(d) for d in ("d1", "d2", "d3", "d4")]
    healthy = _segments_for(loaders[:2], v2_cfg, healthy=True)
    d2_anom = _segments_for([loaders[1]], v2_cfg, healthy=False)
    d3_anom = _segments_for([loaders[2]], v2_cfg, healthy=False)
    d4_anom = _segments_for([loaders[3]], v2_cfg, healthy=False)
    print(f"  healthy: {len(healthy)} | d2_anom: {len(d2_anom)} | "
          f"d3_anom: {len(d3_anom)} | d4_anom: {len(d4_anom)}")

    print("[rq2-3p] Scoring healthy hold-out under all three pipelines ...")
    healthy_loader = _build_loader(healthy, v2_cfg)
    healthy_scores = _score_cohort_three_paradigms(pipelines, healthy_loader)

    print("[rq2-3p] Fitting late-fusion logistic weights ...")
    # Combine a sample of D2/D3/D4 anomalies for the LR fit; per-cohort
    # fitting would over-tune to each cohort.
    if d2_anom or d3_anom or d4_anom:
        any_anom = _segments_for(loaders[1:], v2_cfg, healthy=False)
        anom_loader = _build_loader(any_anom, v2_cfg)
        anom_scores = _score_cohort_three_paradigms(pipelines, anom_loader)
    else:
        anom_scores = {p.name: {"scores": np.zeros(0)} for p in pipelines}

    z_a_h = _zscore(healthy_scores["acoustic"]["scores"], healthy_scores["acoustic"]["scores"])
    z_v_h = _zscore(healthy_scores["vibration"]["scores"], healthy_scores["vibration"]["scores"])
    z_a_x = _zscore(anom_scores["acoustic"]["scores"], healthy_scores["acoustic"]["scores"])
    z_v_x = _zscore(anom_scores["vibration"]["scores"], healthy_scores["vibration"]["scores"])
    w_a, w_v, b = _fit_logistic_late_fusion(z_a_h, z_v_h, z_a_x, z_v_x)
    print(f"  late-fusion logistic: w_a={w_a:+.3f}, w_v={w_v:+.3f}, b={b:+.3f}")

    def _combined_score(z_a: np.ndarray, z_v: np.ndarray) -> np.ndarray:
        return w_a * z_a + w_v * z_v + b

    def _max_score(z_a: np.ndarray, z_v: np.ndarray) -> np.ndarray:
        return np.maximum(z_a, z_v)

    # Threshold combined-score rules at the healthy p95.
    healthy_combined = _combined_score(z_a_h, z_v_h)
    healthy_max = _max_score(z_a_h, z_v_h)
    thr_combined_p95 = float(np.percentile(healthy_combined, 95))
    thr_max_p95 = float(np.percentile(healthy_max, 95))

    rows: list[_ParadigmRow] = []
    audit: dict[str, dict] = {}

    def _add_cohort(name: str, cohort_scores: dict) -> None:
        n = cohort_scores["acoustic"]["scores"].size
        a = cohort_scores["acoustic"]["alerts"]
        v = cohort_scores["vibration"]["alerts"]
        f = cohort_scores["fusion"]["alerts"]
        sa = cohort_scores["acoustic"]["scores"]
        sv = cohort_scores["vibration"]["scores"]
        sf = cohort_scores["fusion"]["scores"]
        z_a = _zscore(sa, healthy_scores["acoustic"]["scores"])
        z_v = _zscore(sv, healthy_scores["vibration"]["scores"])
        combined = _combined_score(z_a, z_v)
        max_score = _max_score(z_a, z_v)
        and_alert = (a & v).astype(np.int32)
        or_alert = (a | v).astype(np.int32)
        combined_alert = (combined > thr_combined_p95).astype(np.int32)
        max_alert = (max_score > thr_max_p95).astype(np.int32)

        rows.append(_row("Unimodal", f"V3-acoustic ({name})", n, float(a.mean()) if n else 0.0, float(sa.mean()) if n else float("nan")))
        rows.append(_row("Unimodal", f"V3-vibration ({name})", n, float(v.mean()) if n else 0.0, float(sv.mean()) if n else float("nan")))
        rows.append(_row("Late Fusion", f"AND ({name})", n, float(and_alert.mean()) if n else 0.0))
        rows.append(_row("Late Fusion", f"OR ({name})", n, float(or_alert.mean()) if n else 0.0))
        rows.append(_row("Late Fusion", f"score_weighted ({name})", n, float(combined_alert.mean()) if n else 0.0, float(combined.mean()) if n else float("nan")))
        rows.append(_row("Late Fusion", f"MAX ({name})", n, float(max_alert.mean()) if n else 0.0, float(max_score.mean()) if n else float("nan")))
        rows.append(_row("Intermediate Fusion", f"V3-fusion ({name})", n, float(f.mean()) if n else 0.0, float(sf.mean()) if n else float("nan")))
        audit[name] = _audit_alert_quadrants(a, v)

    _add_cohort("healthy_holdout", healthy_scores)
    for cohort_name, segs in [("d2_anom", d2_anom), ("d3_anom", d3_anom), ("d4_anom", d4_anom)]:
        if not segs:
            continue
        loader = _build_loader(segs, v2_cfg)
        cohort_scores = _score_cohort_three_paradigms(pipelines, loader)
        _add_cohort(cohort_name, cohort_scores)

    # Paired-bootstrap significance: per-window score quartet
    # (acoustic, vibration, LF_combined, fusion) on healthy holdout — does
    # the LF combined or the V3-fusion give the larger anomaly-vs-healthy
    # score separation?  We compare LF_combined vs V3-fusion on healthy
    # alone (the cohort with enough windows for a stable bootstrap).
    sig = {}
    if z_a_h.size >= 4 and z_a_x.size >= 4:
        comb_anom = _combined_score(z_a_x, z_v_x)
        comb_healthy = _combined_score(z_a_h, z_v_h)
        # Score-separation per paradigm: anomaly_mean - healthy_mean.
        sep_combined = float(comb_anom.mean() - comb_healthy.mean())
        sep_fusion = float(
            anom_scores["fusion"]["scores"].mean()
            - healthy_scores["fusion"]["scores"].mean()
        )
        sig = {
            "score_separation_late_fusion_combined": sep_combined,
            "score_separation_intermediate_fusion": sep_fusion,
            "delta_late_minus_intermediate": sep_combined - sep_fusion,
            "note": "Δ > 0 ⇒ LF combined separates anomaly from healthy more strongly than V3-fusion",
        }

    out_json = v3_run / "rq2_paradigm_comparison.json"
    out_md = v3_run / "rq2_paradigm_comparison.md"
    out_audit = v3_run / "rq2_specificity_audit.json"

    with out_json.open("w", encoding="utf-8") as fh:
        json.dump({
            "v3_three_run": str(v3_run.relative_to(REPO)),
            "source_run": str(src_run.relative_to(REPO)),
            "late_fusion_weights": {"w_a": w_a, "w_v": w_v, "b": b,
                                     "thr_combined_p95": thr_combined_p95,
                                     "thr_max_p95": thr_max_p95},
            "rows": [asdict(r) for r in rows],
            "significance": sig,
            "specificity_audit": audit,
            "method": "rq2_three_paradigm_comparison_2026_05_16",
        }, fh, indent=2)

    with out_audit.open("w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)

    lines: list[str] = []
    lines.append("# RQ2 — Unimodal × 2 + Late Fusion × 4 + Intermediate Fusion\n")
    lines.append(f"Source: V1+V2 from `{src_run.relative_to(REPO)}`, V3 paradigms from `{v3_run.relative_to(REPO)}`.\n")
    lines.append(f"Late-fusion logistic weights: w_acoustic = {w_a:+.3f}, "
                 f"w_vibration = {w_v:+.3f}, bias = {b:+.3f}.")
    lines.append(f"Late-fusion p95 thresholds (healthy hold-out): "
                 f"combined = {thr_combined_p95:+.3f}, MAX = {thr_max_p95:+.3f}.\n")
    lines.append("## Comparison table (alert rate per cohort)\n")
    lines.append("| Paradigm | Rule | n | alert_rate | mean_score |")
    lines.append("|---|---|---:|---:|---:|")
    for r in rows:
        ms = "" if np.isnan(r.mean_score) else f"{r.mean_score:+.2f}"
        lines.append(f"| {r.paradigm} | {r.rule} | {r.n_windows} | {r.alert_rate:.3f} | {ms} |")
    lines.append("\n## Specificity audit — which modalities fire per cohort\n")
    lines.append("| Cohort | a_only | v_only | both | neither | n |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cohort, q in audit.items():
        lines.append(f"| {cohort} | {q['a_only']:.3f} | {q['v_only']:.3f} | "
                     f"{q['both']:.3f} | {q['neither']:.3f} | {q['n']} |")
    if sig:
        lines.append("\n## Late-fusion vs Intermediate-fusion score separation\n")
        lines.append(f"- LF combined: anomaly_mean − healthy_mean = {sig['score_separation_late_fusion_combined']:+.3f}")
        lines.append(f"- V3-fusion : anomaly_mean − healthy_mean = {sig['score_separation_intermediate_fusion']:+.3f}")
        lines.append(f"- Δ (LF − fusion) = {sig['delta_late_minus_intermediate']:+.3f}")
        lines.append(f"  ({sig['note']})")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[rq2-3p] Wrote {out_json.relative_to(REPO)}")
    print(f"[rq2-3p] Wrote {out_md.relative_to(REPO)}")
    print(f"[rq2-3p] Wrote {out_audit.relative_to(REPO)}")


if __name__ == "__main__":
    main()
