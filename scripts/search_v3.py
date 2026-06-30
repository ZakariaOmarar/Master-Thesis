"""Hyperparameter search for the conditional V3 anomaly detector (RQ2).

Tunes both RQ2 arms fairly so the thesis ablation is best-vs-best, not
tuned-vs-untuned:
  * "anchor"    — conditioning + impulse/spectral anchor (the new headline V3)
  * "cond_only" — conditioning only (no anchor; the ablation baseline)

It trains the real pipeline V3 (``train_v3_cnf`` on the frozen V2 of an existing
run, so it does not re-pay the expensive V1/V2 stages) and scores the actual
anomaly cohorts.  Objective per trial:

    obj = mean_AUC(healthy vs {D2,D3,D4} anomaly)  -  penalty * |healthy_FPR - 0.05|

i.e. separability (threshold-free, what conditioning + the anchor improve) minus
a calibration penalty (what `threshold_shrinkage` controls).  Reports a
leaderboard, per-hyperparameter marginal effects, and the best config per arm,
which you then set in `V3_ANOMALY` for the final multi-seed `full_run`.

Run (reuse a run that already trained v1/v2):
    python -m scripts.search_v3 --from-run results/runs/<id> --arms anchor cond_only --trials 10
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.anomaly.v3_trainer import V3Config, score_segments, train_v3_cnf
from src.modeling.eval.rq2_three_paradigm_eval import (
    _build_v2,
    _load_state,
    _loader,
    _segments_for,
)
from src.modeling.orchestration.stage_configs import v2_config

DS_ANOM = ("d2", "d3", "d4")

# Search space (K=3 fixed = the operating-mode hypothesis; inject_impulse_anchor
# is set per arm).  Modest by design so a full sweep finishes overnight on a GPU.
SPACE = {
    "conditional_base": [True, False],
    "n_layers": [6, 8],
    "hidden_dim": [64, 128],
    "lr": [3e-4, 1e-3],
    "epochs": [12, 18],
    "threshold_shrinkage": [100.0, 300.0, 500.0],
}


def _auc(healthy: np.ndarray, anom: np.ndarray) -> float:
    if healthy.size == 0 or anom.size == 0:
        return float("nan")
    allv = np.concatenate([healthy, anom])
    r = allv.argsort().argsort().astype(np.float64) + 1.0
    n_h, n_a = healthy.size, anom.size
    return float((r[n_h:].sum() - n_a * (n_a + 1) / 2.0) / (n_h * n_a))


def _evaluate(cfg, v2, loaders, v2_cfg, anom_segs, dev, percentile) -> dict:
    res = train_v3_cnf(v2, loaders, v2_cfg=v2_cfg, v3_cfg=cfg)
    anchor_norm = ((res.anchor_mean, res.anchor_std)
                   if getattr(res, "anchor_mean", None) is not None else None)
    # held-out healthy FPR at the (shrunk) per-cluster threshold
    h_alerts, _ = res.thresholds.alert(res.val_contexts, res.val_scores, percentile=percentile)
    healthy_fpr = float(h_alerts.mean()) if h_alerts.size else float("nan")
    per_ds, aucs, recalls = {}, [], {}
    for dn, segs in anom_segs.items():
        if not segs:
            continue
        sc, ctx, _ = score_segments(
            v2, res.flow, segs, v2_cfg=v2_cfg, device=dev,
            xt_pool=res.xt_pool, anchor_norm=anchor_norm,
        )
        if sc.size == 0:
            continue
        a, _ = res.thresholds.alert(ctx, sc, percentile=percentile)
        auc = _auc(res.val_scores, sc)
        aucs.append(auc); recalls[dn] = float(a.mean())
        per_ds[dn] = {"auc": round(auc, 3), "recall_at_thr": round(float(a.mean()), 3)}
    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    return {"mean_auc": mean_auc, "healthy_fpr": healthy_fpr, "per_ds": per_ds,
            "val_nll": float(res.val_nll[-1]) if res.val_nll else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", required=True, help="run dir with v1/ + v2/encoder.pt")
    ap.add_argument("--arms", nargs="*", default=["anchor", "cond_only"],
                    choices=["anchor", "cond_only"])
    ap.add_argument("--trials", type=int, default=10, help="trials PER arm")
    ap.add_argument("--penalty", type=float, default=1.0, help="weight on |FPR-0.05|")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/v3_search.json")
    args = ap.parse_args()
    run = Path(args.from_run).resolve()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {dev} | reusing V1/V2 from {run.name}")

    v2_cfg = v2_config(False)
    v2 = _build_v2(v2_cfg); _load_state(run / "v2" / "encoder.pt", v2); v2.eval()
    loaders = [_loader(d) for d in ("d1", "d2", "d3", "d4")]
    anom_segs = {dn: _segments_for([loaders[i]], v2_cfg, healthy=False)
                 for i, dn in zip((1, 2, 3), DS_ANOM)}
    percentile = int(V3Config().threshold_percentile)
    rng = np.random.default_rng(args.seed)

    all_results: dict = {"arms": {}}
    for arm in args.arms:
        inject = (arm == "anchor")
        print(f"\n############## ARM: {arm} (inject_impulse_anchor={inject}) ##############")
        trials = []
        for t in range(args.trials):
            choice = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
            cfg = replace(V3Config(), inject_impulse_anchor=inject, n_threshold_clusters=3,
                          seed=args.seed, device=str(dev), **choice)
            print(f"\n=== {arm} trial {t+1}/{args.trials}: {choice} ===", flush=True)
            try:
                m = _evaluate(cfg, v2, loaders, v2_cfg, anom_segs, dev, percentile)
                obj = m["mean_auc"] - args.penalty * abs(m["healthy_fpr"] - 0.05)
                print(f"    obj={obj:.4f} | mean_AUC={m['mean_auc']:.3f} "
                      f"healthy_FPR={m['healthy_fpr']:.3f} per_ds={m['per_ds']}", flush=True)
                trials.append({"params": choice, "objective": obj, **m})
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
                trials.append({"params": choice, "objective": float("nan"), "error": str(e)})

        ok = sorted([t for t in trials if np.isfinite(t["objective"])],
                    key=lambda t: t["objective"], reverse=True)
        print(f"\n===== {arm} leaderboard (top 5) =====")
        for t in ok[:5]:
            print(f"  obj={t['objective']:.4f}  AUC={t['mean_auc']:.3f} "
                  f"FPR={t['healthy_fpr']:.3f}  {t['params']}")
        print(f"\n----- {arm} per-hyperparameter marginal mean objective -----")
        for k, vals in SPACE.items():
            cells = [f"{v}={np.mean([t['objective'] for t in ok if t['params'][k]==v]):.3f}"
                     f"(n{sum(t['params'][k]==v for t in ok)})" for v in vals]
            print(f"  {k:<20} " + "  ".join(cells))
        best = ok[0] if ok else None
        if best:
            print(f"\n  BEST {arm}: obj={best['objective']:.4f} per_ds={best['per_ds']}")
            print(f"  -> set V3_ANOMALY: inject_impulse_anchor={inject}, "
                  f"conditional_base={best['params']['conditional_base']}, "
                  f"n_layers={best['params']['n_layers']}, hidden_dim={best['params']['hidden_dim']}, "
                  f"threshold_shrinkage={best['params']['threshold_shrinkage']}; "
                  f"V3Config: lr={best['params']['lr']}, epochs={best['params']['epochs']}")
        all_results["arms"][arm] = {"trials": trials, "best": best,
                                    "inject_impulse_anchor": inject}

    out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
