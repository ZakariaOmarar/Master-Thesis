"""Measure vibration's actual contribution to V3 anomaly detection.

A central thesis question is whether vibration meaningfully helps anomaly
detection (RQ2) and localisation (RQ3).  RQ3 is already settled adversely by
the V4 A5 ablation CI overlap (srp_only 0.172 [0.163, 0.181] vs full 0.168
[0.157, 0.179] — adding accelerometer-TDOA to acoustic SRP-PHAT provides no
statistically significant lift).

This module answers the same question for RQ2: **how much of V3's anomaly
discrimination is actually driven by vibration?**  Three saved-weights
measurements on the 2026-05-15 V2 encoder + V3 flow:

  1. **Per-cohort alert-rate ablation** — score the held-out healthy cohort
     and the D2/D3/D4 anomaly cohorts twice: once with full V2 forward,
     once with vibration zeroed at the V2 input.  If recall on anomaly
     cohorts is unchanged when vibration is zeroed, V3's anomaly signal is
     acoustic-only.

  2. **Per-window NLL gradient sensitivity** — for each held-out healthy
     window, compute ``||∂NLL/∂vib_input|| / ||∂NLL/∂ac_input||``.  Direct
     counterfactual on how much V3 actually uses vibration.

  3. **Per-window NLL Δ under vibration zeroing** — for each window,
     compare ``-log p(x, c)`` with full V2 forward vs ``-log p(x', c')``
     with vibration zeroed.  Paired bootstrap on the per-window Δ gives a
     significance test on "does V3's anomaly signal change when vibration
     is removed?"

Outputs:
  * ``results/v3_vibration_contribution.json``
  * ``results/v3_vibration_contribution.md``

Run::

    python -m src.modeling.eval.v3_vibration_contribution
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud

from ...ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader
from ..anomaly.cnf_head import ConditionalRealNVP
from ..anomaly.threshold import PerClusterThresholds
from ..context.v2_fusion import V2FusionEncoder
from ..context.v2_ssl import (
    V2SSLConfig,
    _collate,
    _PairedGroupedBatchSampler,
    _PairedSegment,
    _PairedWindowedDataset,
    _precompute_paired,
)
from ..eval.statistics import paired_bootstrap_test

REPO = Path(__file__).resolve().parents[3]
RUN_DIR = REPO / "results" / "runs" / "20260515_064625__full_seed42"
V2_WEIGHTS = RUN_DIR / "v2" / "encoder.pt"
V3_FLOW = RUN_DIR / "v3" / "flow.pt"
V3_THRESHOLDS = RUN_DIR / "v3" / "thresholds.npz"

OUT_JSON = REPO / "results" / "v3_vibration_contribution.json"
OUT_MD = REPO / "results" / "v3_vibration_contribution.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _loader(name: str) -> TestDatasetLoader:
    # DatasetSpec.from_yaml resolves all paths to absolute — no reconstruction.
    spec = DatasetSpec.from_yaml(REPO / "configs" / "datasets" / f"{name}.yaml")
    return TestDatasetLoader(spec)


def v2_config() -> V2SSLConfig:
    return V2SSLConfig(
        window_seconds=2.0, window_stride_seconds=1.0, feature_dim=64, embed_dim=64,
        n_heads=4, proj_dim=32, epochs=12, batch_size=16, lr=1e-3, weight_decay=1e-5,
        temperature=0.1, val_ratio=0.3,
        # n_mels / n_fft / hop_length inherited from ACOUSTIC_FEATURES
        # (n_fft=4096, hop=2048, n_mels=96 per chapter 3 §3.4.2 grid sweep).
        cwt_n_scales=32, use_cwt=True, gain_jitter_db=6.0,
        channel_dropout_p=0.2, spec_augment_freq_mask=6, spec_augment_time_mask=8,
        lmm_mask_p=0.3, lmm_weight=1.0,
        modality_dropout_p=0.0, acoustic_dropout_p=0.0, vibration_dropout_p=0.5,
        cma_weight=0.0, cma_temperature=0.1,
        context_mode="joint_pma", num_context_seeds=2, seed=42,
    )


def _load_v2() -> V2FusionEncoder:
    cfg = v2_config()
    enc = V2FusionEncoder(
        feature_dim=cfg.feature_dim, embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads, context_mode=cfg.context_mode,
        num_context_seeds=cfg.num_context_seeds,
    )
    enc.load_state_dict(torch.load(V2_WEIGHTS, map_location="cpu"), strict=False)
    enc.eval()
    return enc


def _load_v3(c_dim: int) -> tuple[ConditionalRealNVP, PerClusterThresholds]:
    # x_dim is the mean-pool feature dim — same as embed_dim (64) for this run.
    flow = ConditionalRealNVP(
        dim=64, c_dim=c_dim, n_layers=6, hidden_dim=64, n_hidden_per_net=2, scale_max=2.0,
    )
    flow.load_state_dict(torch.load(V3_FLOW, map_location="cpu"), strict=False)
    flow.eval()
    th_npz = np.load(V3_THRESHOLDS)
    th = PerClusterThresholds(
        centroids=th_npz["centroids"], p95=th_npz["p95"], p99=th_npz["p99"],
        n_per_cluster=th_npz["n_per_cluster"], seed=42,
    )
    return flow, th


def _segments_for(
    loaders: list[TestDatasetLoader],
    cfg: V2SSLConfig,
    *,
    healthy: bool,
    require_label: tuple[str, ...] = (),
) -> list[_PairedSegment]:
    """Gather and precompute paired segments meeting the filter.

    ``healthy=True`` means ``is_anomaly is False``; ``require_label`` if
    non-empty restricts to segments whose ``mode_label`` is in the tuple.
    """
    out: list[_PairedSegment] = []
    for L in loaders:
        for s in L.list_segments():
            if healthy and s.is_anomaly:
                continue
            if (not healthy) and (not s.is_anomaly):
                continue
            if require_label and (s.mode_label not in require_label):
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
# Measurements
# ---------------------------------------------------------------------------


@dataclass
class CohortAlertStats:
    cohort: str
    n_windows: int
    alert_rate_full: float
    alert_rate_vib_zeroed: float
    alert_rate_delta: float  # full - vib_zeroed; positive = vibration helps
    mean_score_full: float
    mean_score_vib_zeroed: float
    mean_score_delta: float


@dataclass
class GradientSensitivityStats:
    n_windows: int
    grad_norm_acoustic_mean: float
    grad_norm_vibration_mean: float
    grad_norm_ratio_acoustic_to_vibration: float


@dataclass
class PerWindowNLLDelta:
    n_windows: int
    mean_nll_full: float
    mean_nll_vib_zeroed: float
    delta_mean: float  # full - vib_zeroed
    delta_ci95_low: float
    delta_ci95_high: float
    p_value_two_sided: float


def _score_cohort_dual(
    encoder: V2FusionEncoder,
    flow: ConditionalRealNVP,
    thresholds: PerClusterThresholds,
    loader: tud.DataLoader,
    percentile: int = 95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score every window twice: full V2 forward, vibration zeroed at input.

    Returns ``(scores_full, scores_vib_zeroed, alerts_full, alerts_vib_zeroed)``
    as 1-D numpy arrays of length n_windows.
    """
    scores_full_list: list[np.ndarray] = []
    scores_vibz_list: list[np.ndarray] = []
    alerts_full_list: list[np.ndarray] = []
    alerts_vibz_list: list[np.ndarray] = []
    encoder.eval()
    flow.eval()
    with torch.no_grad():
        for batch in loader:
            ac = batch["ac_feat"]
            vib = batch["vib_feat"]
            ac_xyz = batch["ac_xyz"]
            vib_xyz = batch["vib_xyz"]
            ds_idx = batch["dataset_idx"]

            # Full forward.
            out = encoder(ac, ac_xyz, vib, vib_xyz, ds_idx, mask_p=0.0)
            fused = torch.cat([out["a_fused"], out["v_fused"]], dim=1)
            x = fused.mean(dim=1)
            c = out["context"]
            s_full = flow.anomaly_score(x, c).cpu().numpy()
            a_full, _ = thresholds.alert(c.cpu().numpy(), s_full, percentile=percentile)

            # Vibration zeroed at the input.
            out_z = encoder(ac, ac_xyz, torch.zeros_like(vib), vib_xyz, ds_idx, mask_p=0.0)
            fused_z = torch.cat([out_z["a_fused"], out_z["v_fused"]], dim=1)
            x_z = fused_z.mean(dim=1)
            c_z = out_z["context"]
            s_vibz = flow.anomaly_score(x_z, c_z).cpu().numpy()
            a_vibz, _ = thresholds.alert(c_z.cpu().numpy(), s_vibz, percentile=percentile)

            scores_full_list.append(s_full)
            scores_vibz_list.append(s_vibz)
            alerts_full_list.append(a_full.astype(np.int32))
            alerts_vibz_list.append(a_vibz.astype(np.int32))
    if not scores_full_list:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty.astype(np.int32), empty.astype(np.int32)
    return (
        np.concatenate(scores_full_list),
        np.concatenate(scores_vibz_list),
        np.concatenate(alerts_full_list),
        np.concatenate(alerts_vibz_list),
    )


def _gradient_sensitivity(
    encoder: V2FusionEncoder,
    flow: ConditionalRealNVP,
    loader: tud.DataLoader,
    max_batches: int = 5,
) -> GradientSensitivityStats:
    """Per-window ``||∂NLL/∂x||`` for each input modality.

    A modality the V3 score doesn't use will have gradient ≈ 0 on every
    window.  Limited to ``max_batches`` because each call needs a fresh
    backward pass.
    """
    grad_a_norms: list[float] = []
    grad_v_norms: list[float] = []
    n_seen = 0
    for batch in loader:
        if n_seen >= max_batches:
            break
        ac = batch["ac_feat"].detach().requires_grad_(True)
        vib = batch["vib_feat"].detach().requires_grad_(True)
        out = encoder(ac, batch["ac_xyz"], vib, batch["vib_xyz"], batch["dataset_idx"], mask_p=0.0)
        fused = torch.cat([out["a_fused"], out["v_fused"]], dim=1)
        x = fused.mean(dim=1)
        c = out["context"]
        nll = flow.anomaly_score(x, c).sum()  # scalar
        grad_ac, grad_vib = torch.autograd.grad(
            nll, [ac, vib], retain_graph=False, create_graph=False,
        )
        g_a = grad_ac.detach().flatten(1).pow(2).sum(dim=1).sqrt()
        g_v = grad_vib.detach().flatten(1).pow(2).sum(dim=1).sqrt()
        grad_a_norms.extend(g_a.tolist())
        grad_v_norms.extend(g_v.tolist())
        n_seen += 1
    ga = np.asarray(grad_a_norms)
    gv = np.asarray(grad_v_norms)
    return GradientSensitivityStats(
        n_windows=int(ga.size),
        grad_norm_acoustic_mean=float(ga.mean()) if ga.size else float("nan"),
        grad_norm_vibration_mean=float(gv.mean()) if gv.size else float("nan"),
        grad_norm_ratio_acoustic_to_vibration=(
            float(ga.mean() / max(gv.mean(), 1e-12)) if ga.size and gv.size else float("nan")
        ),
    )


def _nll_delta_significance(
    scores_full: np.ndarray, scores_vibz: np.ndarray
) -> PerWindowNLLDelta:
    """Paired bootstrap on (full − vib_zeroed) per-window scores.

    Direction A < B in `paired_bootstrap_test` means "A is smaller on average
    than B"; here we want the sign of the mean Δ.
    """
    if scores_full.size < 4 or scores_full.size != scores_vibz.size:
        return PerWindowNLLDelta(
            n_windows=int(scores_full.size),
            mean_nll_full=float("nan"), mean_nll_vib_zeroed=float("nan"),
            delta_mean=float("nan"),
            delta_ci95_low=float("nan"), delta_ci95_high=float("nan"),
            p_value_two_sided=float("nan"),
        )
    res = paired_bootstrap_test(
        scores_full, scores_vibz, lower_is_better=False, n_boot=1000, seed=42,
    )
    return PerWindowNLLDelta(
        n_windows=int(scores_full.size),
        mean_nll_full=float(scores_full.mean()),
        mean_nll_vib_zeroed=float(scores_vibz.mean()),
        delta_mean=float(res.delta_point),
        delta_ci95_low=float(res.delta_ci_low),
        delta_ci95_high=float(res.delta_ci_high),
        p_value_two_sided=float(res.p_value_two_sided),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _verdict(
    healthy: CohortAlertStats,
    anomaly_cohorts: list[CohortAlertStats],
    grads: GradientSensitivityStats,
    nll_delta: PerWindowNLLDelta,
) -> str:
    """Pick a one-sentence verdict on RQ2's vibration-contribution question.

    Rules of thumb:
      * grad ratio > 50× → vibration is essentially ignored by V3.
      * per-cohort alert-rate delta < 0.02 across all anomaly cohorts →
        vibration provides no measurable anomaly-detection benefit.
      * NLL paired bootstrap CI straddles zero → no significant change
        when vibration is zeroed.
    """
    vib_ignored = grads.grad_norm_ratio_acoustic_to_vibration > 50.0
    no_recall_lift = all(
        abs(c.alert_rate_delta) < 0.02 for c in anomaly_cohorts
    )
    no_significant_nll_change = (
        nll_delta.delta_ci95_low <= 0.0 <= nll_delta.delta_ci95_high
    )
    if vib_ignored and no_recall_lift and no_significant_nll_change:
        return (
            "Vibration is essentially NOT load-bearing for V3 anomaly detection: "
            f"grad ratio {grads.grad_norm_ratio_acoustic_to_vibration:.0f}x, "
            "no cohort alert-rate shift when vibration is zeroed, "
            f"NLL Δ CI95 [{nll_delta.delta_ci95_low:+.3f}, {nll_delta.delta_ci95_high:+.3f}] "
            "straddles zero. RQ2 vibration-helps claim is unsupported under the current "
            "pipeline; structural change required (e.g., concatenate vibration mean-pool "
            "features to V3 input x instead of routing through c_t)."
        )
    if vib_ignored:
        return (
            "Vibration is mostly ignored at V3 inference (grad ratio "
            f"{grads.grad_norm_ratio_acoustic_to_vibration:.0f}x toward acoustic) but "
            "the alert-rate / NLL signal is not entirely zero — vibration has marginal "
            "but non-zero contribution.  Statistical significance: NLL Δ p="
            f"{nll_delta.p_value_two_sided:.3f}."
        )
    return (
        "Vibration does contribute to V3 anomaly detection: grad ratio "
        f"{grads.grad_norm_ratio_acoustic_to_vibration:.1f}x (not extreme), "
        f"NLL Δ p={nll_delta.p_value_two_sided:.3f}. "
        "Magnitude of contribution depends on the per-cohort recall deltas below."
    )


def _write_report(
    healthy: CohortAlertStats,
    anomaly_cohorts: list[CohortAlertStats],
    grads: GradientSensitivityStats,
    nll_delta: PerWindowNLLDelta,
    verdict: str,
) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": verdict,
        "healthy_holdout": asdict(healthy),
        "anomaly_cohorts": [asdict(c) for c in anomaly_cohorts],
        "gradient_sensitivity": asdict(grads),
        "per_window_nll_delta": asdict(nll_delta),
        "method": "v3_vibration_contribution_saved_weights_2026_05_15",
        "v2_weights": str(V2_WEIGHTS.relative_to(REPO)),
        "v3_flow": str(V3_FLOW.relative_to(REPO)),
    }
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    lines: list[str] = []
    lines.append("# V3 vibration-contribution forensics — 2026-05-15 archived weights\n")
    lines.append(f"**Verdict.** {verdict}\n")
    lines.append("## Measurements\n")
    lines.append("### 1. Per-cohort alert-rate ablation (full V2 vs vibration zeroed at V2 input)\n")
    lines.append(
        "| cohort | n | alert_full | alert_vib_zeroed | Δ | score_full | score_vib_zeroed |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in [healthy, *anomaly_cohorts]:
        lines.append(
            f"| {c.cohort} | {c.n_windows} | {c.alert_rate_full:.3f} | "
            f"{c.alert_rate_vib_zeroed:.3f} | {c.alert_rate_delta:+.3f} | "
            f"{c.mean_score_full:+.2f} | {c.mean_score_vib_zeroed:+.2f} |"
        )
    lines.append("")
    lines.append("### 2. Per-window NLL gradient sensitivity (held-out healthy)\n")
    lines.append(f"- ||grad NLL / acoustic input|| mean: {grads.grad_norm_acoustic_mean:.4f}")
    lines.append(f"- ||grad NLL / vibration input|| mean: {grads.grad_norm_vibration_mean:.4f}")
    lines.append(
        f"- ratio acoustic / vibration: **{grads.grad_norm_ratio_acoustic_to_vibration:.2f}x** "
        "(values > 50x indicate vibration is effectively ignored)"
    )
    lines.append("")
    lines.append("### 3. Per-window NLL Δ paired bootstrap (full − vibration-zeroed)\n")
    lines.append(f"- n_windows: {nll_delta.n_windows}")
    lines.append(f"- mean NLL full: {nll_delta.mean_nll_full:+.3f}")
    lines.append(f"- mean NLL vibration zeroed: {nll_delta.mean_nll_vib_zeroed:+.3f}")
    lines.append(
        f"- mean Δ: **{nll_delta.delta_mean:+.4f}** "
        f"[95% CI {nll_delta.delta_ci95_low:+.4f}, {nll_delta.delta_ci95_high:+.4f}], "
        f"p={nll_delta.p_value_two_sided:.4f}"
    )
    lines.append("")
    lines.append(
        "(CI straddling 0 ⇒ no significant change in V3 anomaly score when vibration "
        "is removed at the V2 input ⇒ vibration not load-bearing for RQ2.)"
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = v2_config()
    print(f"[v3-vib] Loading V2 from {V2_WEIGHTS} and V3 from {V3_FLOW} ...")
    encoder = _load_v2()
    # Probe c dim from one forward pass.
    loaders = [_loader("d1"), _loader("d2"), _loader("d3"), _loader("d4")]
    probe_segs = _segments_for(loaders[:1], cfg, healthy=True)[:1]
    if not probe_segs:
        raise RuntimeError("no healthy segments to probe c_dim")
    probe_loader = _build_loader(probe_segs, cfg)
    with torch.no_grad():
        batch = next(iter(probe_loader))
        out = encoder(
            batch["ac_feat"], batch["ac_xyz"],
            batch["vib_feat"], batch["vib_xyz"], batch["dataset_idx"],
        )
        c_dim = int(out["context"].shape[-1])
    print(f"[v3-vib] c_dim = {c_dim}")
    flow, thresholds = _load_v3(c_dim)
    print(f"[v3-vib] thresholds: p95 per cluster = {thresholds.p95.tolist()}")

    # --- 1. Per-cohort alert rates ---------------------------------------
    print("\n[v3-vib] Gathering cohorts ...")

    healthy_segs = _segments_for(loaders[:2], cfg, healthy=True)
    print(f"  healthy_holdout (D1+D2):  {len(healthy_segs)} paired segments")

    # Anomaly cohorts: D2 RandomFault, D3 hit, D4 RandomFault.
    d2_anom = _segments_for([loaders[1]], cfg, healthy=False)
    d3_anom = _segments_for([loaders[2]], cfg, healthy=False)
    d4_anom = _segments_for([loaders[3]], cfg, healthy=False)
    print(f"  d2_anom:  {len(d2_anom)} paired segments")
    print(f"  d3_anom:  {len(d3_anom)} paired segments")
    print(f"  d4_anom:  {len(d4_anom)} paired segments")

    cohort_results: list[CohortAlertStats] = []
    healthy_loader = _build_loader(healthy_segs, cfg)
    s_full_h, s_vibz_h, a_full_h, a_vibz_h = _score_cohort_dual(
        encoder, flow, thresholds, healthy_loader,
    )
    healthy_stats = CohortAlertStats(
        cohort="healthy_holdout(D1+D2)",
        n_windows=int(s_full_h.size),
        alert_rate_full=float(a_full_h.mean()) if a_full_h.size else 0.0,
        alert_rate_vib_zeroed=float(a_vibz_h.mean()) if a_vibz_h.size else 0.0,
        alert_rate_delta=float(a_full_h.mean() - a_vibz_h.mean()) if a_full_h.size else 0.0,
        mean_score_full=float(s_full_h.mean()) if s_full_h.size else 0.0,
        mean_score_vib_zeroed=float(s_vibz_h.mean()) if s_vibz_h.size else 0.0,
        mean_score_delta=float(s_full_h.mean() - s_vibz_h.mean()) if s_full_h.size else 0.0,
    )

    for name, segs in (("d2_anom", d2_anom), ("d3_anom", d3_anom), ("d4_anom", d4_anom)):
        if not segs:
            cohort_results.append(CohortAlertStats(
                cohort=name, n_windows=0,
                alert_rate_full=0.0, alert_rate_vib_zeroed=0.0, alert_rate_delta=0.0,
                mean_score_full=0.0, mean_score_vib_zeroed=0.0, mean_score_delta=0.0,
            ))
            continue
        loader = _build_loader(segs, cfg)
        s_f, s_v, a_f, a_v = _score_cohort_dual(encoder, flow, thresholds, loader)
        cohort_results.append(CohortAlertStats(
            cohort=name, n_windows=int(s_f.size),
            alert_rate_full=float(a_f.mean()) if a_f.size else 0.0,
            alert_rate_vib_zeroed=float(a_v.mean()) if a_v.size else 0.0,
            alert_rate_delta=float(a_f.mean() - a_v.mean()) if a_f.size else 0.0,
            mean_score_full=float(s_f.mean()) if s_f.size else 0.0,
            mean_score_vib_zeroed=float(s_v.mean()) if s_v.size else 0.0,
            mean_score_delta=float(s_f.mean() - s_v.mean()) if s_f.size else 0.0,
        ))

    # --- 2. Gradient sensitivity (healthy only) --------------------------
    print("\n[v3-vib] Computing gradient sensitivity (healthy holdout) ...")
    grads = _gradient_sensitivity(encoder, flow, healthy_loader, max_batches=5)

    # --- 3. NLL paired bootstrap (healthy holdout) -----------------------
    print("[v3-vib] NLL paired bootstrap (healthy holdout) ...")
    nll_delta = _nll_delta_significance(s_full_h, s_vibz_h)

    verdict = _verdict(healthy_stats, cohort_results, grads, nll_delta)
    _write_report(healthy_stats, cohort_results, grads, nll_delta, verdict)
    try:
        print(f"\n[v3-vib] VERDICT: {verdict}")
    except UnicodeEncodeError:
        print(f"\n[v3-vib] VERDICT: {verdict.encode('ascii', errors='replace').decode('ascii')}")
    print(f"\n[v3-vib] Wrote {OUT_JSON.relative_to(REPO)}")
    print(f"[v3-vib] Wrote {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
