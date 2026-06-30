"""Probe: does a clean healthy/anomaly separating threshold exist per modality?

Loads ONE valid (xt_pool-persisted) canonical run, scores the healthy hold-out
and the D2/D3/D4 anomaly cohorts under each modality, and reports:

  * AUC (anomaly-vs-healthy) per modality+cohort  -- the threshold-free ceiling.
  * Healthy FPR + per-cohort anomaly alert rate (TPR proxy) under three
    threshold strategies, all fit on the EVALUATION healthy hold-out so the
    operating point is honest and seed-stable:
      - saved      : the thresholds.npz p95 (current behaviour)
      - fresh_pc   : per-cluster p95 re-fit on the eval healthy hold-out
      - fresh_glob : a single global p95 on the eval healthy hold-out

Run:  python -m scripts.diagnostics.probe_v3_separation [run_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.anomaly.v3_per_modality import (
    V3AcousticOnlyAdapter,
    V3VibrationOnlyAdapter,
)
from src.modeling.eval.rq2_three_paradigm_eval import (
    _build_loader,
    _build_v1,
    _build_v2,
    _load_state,
    _load_v3,
    _loader,
    _PipelineState,
    _score_cohort_three_paradigms,
    _segments_for,
)
from src.modeling.orchestration.full_run import v1_config, v2_config

DEFAULT_RUN = REPO / "results" / "runs" / "20260616_022513__full_pipeline_b5_cma"


def _auc(healthy: np.ndarray, anom: np.ndarray) -> float:
    """Rank AUC: P(anom score > healthy score).  0.5 = no separation."""
    if healthy.size == 0 or anom.size == 0:
        return float("nan")
    allv = np.concatenate([healthy, anom])
    ranks = allv.argsort().argsort().astype(np.float64) + 1.0
    r_anom = ranks[healthy.size:].sum()
    n_h, n_a = healthy.size, anom.size
    return float((r_anom - n_a * (n_a + 1) / 2.0) / (n_h * n_a))


def main() -> int:
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RUN
    src = run  # full_pipeline runs hold v1/v2 in the same dir
    print(f"run = {run.name}")

    v1_cfg = v1_config(False)
    v2_cfg = v2_config(False)
    embed = int(v1_cfg.embed_dim)

    v1_a = _build_v1("acoustic", v1_cfg); _load_state(src / "v1" / "acoustic.pt", v1_a); v1_a.eval()
    v1_v = _build_v1("vibration", v1_cfg); _load_state(src / "v1" / "vibration.pt", v1_v); v1_v.eval()
    v2 = _build_v2(v2_cfg); _load_state(src / "v2" / "encoder.pt", v2); v2.eval()

    flow_a, th_a, xt_a, anc_a = _load_v3(run / "v3_acoustic", x_dim=embed, c_dim=embed)
    flow_v, th_v, xt_v, anc_v = _load_v3(run / "v3_vibration", x_dim=embed, c_dim=embed)
    flow_f, th_f, xt_f, anc_f = _load_v3(run / "v3_fusion", x_dim=embed, c_dim=embed)
    pipelines = [
        _PipelineState("acoustic", V3AcousticOnlyAdapter(v1_a), flow_a, th_a, xt_a, anc_a),
        _PipelineState("vibration", V3VibrationOnlyAdapter(v1_v), flow_v, th_v, xt_v, anc_v),
        _PipelineState("fusion", v2, flow_f, th_f, xt_f, anc_f),
    ]

    # Score each dataset's healthy and ANOMALY partitions separately, so we can
    # compare an anomaly cohort against (a) the GLOBAL healthy baseline and
    # (b) its OWN-dataset healthy baseline.  If a dataset runs in a different
    # operating regime, the global baseline confounds "anomalous" with "different
    # regime"; the own-dataset baseline is the regime-matched, label-free
    # reference the conditional flow is meant to use.
    loaders = [_loader(d) for d in ("d1", "d2", "d3", "d4")]
    print("gathering cohorts ...")
    ds_names = ["d1", "d2", "d3", "d4"]
    scored_h: dict[str, dict] = {}
    scored_a: dict[str, dict] = {}
    for i, dn in enumerate(ds_names):
        h = _segments_for([loaders[i]], v2_cfg, healthy=True)
        a = _segments_for([loaders[i]], v2_cfg, healthy=False)
        if h:
            print(f"scoring {dn} HEALTHY ({len(h)} segs) ...", flush=True)
            scored_h[dn] = _score_cohort_three_paradigms(pipelines, _build_loader(h, v2_cfg))
        if a:
            print(f"scoring {dn} ANOMALY ({len(a)} segs) ...", flush=True)
            scored_a[dn] = _score_cohort_three_paradigms(pipelines, _build_loader(a, v2_cfg))

    # AUC is threshold-FREE separability; det@p99 uses a LABEL-FREE boundary
    # (99th percentile of the relevant healthy scores).  "_glob" = global
    # healthy baseline (all datasets' healthy); "_own" = same-dataset healthy.
    for mod in ("acoustic", "vibration", "fusion"):
        glob_h = np.concatenate([scored_h[dn][mod]["scores"] for dn in scored_h])
        t_glob99 = float(np.percentile(glob_h, 99))
        print(f"\n========== {mod} ==========")
        print(f"global-healthy NLL: p50={np.percentile(glob_h,50):.2f} "
              f"p99={t_glob99:.2f} (n={glob_h.size})")
        # Per-dataset healthy FPR at the GLOBAL p99 boundary (should be ~1% if
        # the global baseline is regime-fair; >>1% on a dataset means that
        # dataset's healthy already looks anomalous globally = regime shift).
        print("  per-dataset healthy FPR @ global-p99: "
              + ", ".join(f"{dn}={(scored_h[dn][mod]['scores']>t_glob99).mean():.3f}"
                          for dn in scored_h))
        print(f"{'anom ds':<8} {'AUCvsGlob':>9} {'AUCvsOwn':>9} | "
              f"{'det@glob99':>10} {'det@own99':>9} {'ownFPR99':>9}")
        for dn in scored_a:
            a_s = scored_a[dn][mod]["scores"]
            auc_g = _auc(glob_h, a_s)
            own_h = scored_h[dn][mod]["scores"] if dn in scored_h else None
            if own_h is not None and own_h.size:
                t_own99 = float(np.percentile(own_h, 99))
                auc_o = _auc(own_h, a_s)
                det_o = float((a_s > t_own99).mean())
                own_fpr = float((own_h > t_own99).mean())
            else:
                auc_o = det_o = own_fpr = float("nan")
            det_g = float((a_s > t_glob99).mean())
            print(f"{dn:<8} {auc_g:>9.3f} {auc_o:>9.3f} | "
                  f"{det_g:>10.3f} {det_o:>9.3f} {own_fpr:>9.3f}")

    # === Option 1: context-local score normalization (post-hoc, label-free) ===
    # residual(x) = score(x) - mean_H( scores of the k nearest POOLED-healthy
    # windows to x in context space ).  One global model + one global threshold,
    # no campaign label: the regime is read off the (already-computed) context
    # vector, not the dataset id.  This is a proxy for "what perfect context
    # conditioning would buy" -- if D2's residual-AUC against POOLED healthy
    # jumps toward the own-healthy AUC, then strengthening the flow's
    # conditioning (thesis Option 3) should separate it with a single global
    # baseline.  rawAUC repeats AUCvsGlob above for side-by-side comparison.
    from sklearn.neighbors import NearestNeighbors
    K = 128
    print(f"\n########## context-local normalized residual "
          f"(k={K}, own-pipeline context) ##########")
    for mod in ("acoustic", "vibration", "fusion"):
        base_c = np.concatenate([scored_h[dn][mod]["contexts"] for dn in scored_h])
        base_s = np.concatenate([scored_h[dn][mod]["scores"] for dn in scored_h])
        nn = NearestNeighbors(n_neighbors=K + 1).fit(base_c)

        def _resid(ctx: np.ndarray, scr: np.ndarray, drop_self: bool,
                   nn=nn, base_s=base_s) -> np.ndarray:
            _, idx = nn.kneighbors(ctx)
            idx = idx[:, 1:] if drop_self else idx[:, :K]
            return scr - base_s[idx].mean(axis=1)

        h_res = {dn: _resid(scored_h[dn][mod]["contexts"],
                            scored_h[dn][mod]["scores"], True) for dn in scored_h}
        all_h_res = np.concatenate(list(h_res.values()))
        thr = float(np.percentile(all_h_res, 99))
        print(f"\n[{mod}] pooled-healthy residual: p50={np.median(all_h_res):.3f} "
              f"p99={thr:.3f}  FPR@p99={(all_h_res > thr).mean():.3f}")
        print(f"{'anom ds':<8} {'rawAUC':>7} {'residAUC':>9} {'det@residp99':>13}")
        for dn in scored_a:
            a_res = _resid(scored_a[dn][mod]["contexts"],
                           scored_a[dn][mod]["scores"], False)
            raw_auc = _auc(base_s, scored_a[dn][mod]["scores"])
            res_auc = _auc(all_h_res, a_res)
            det = float((a_res > thr).mean())
            print(f"{dn:<8} {raw_auc:>7.3f} {res_auc:>9.3f} {det:>13.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
