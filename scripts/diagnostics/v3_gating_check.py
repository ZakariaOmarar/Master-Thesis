"""Offline check of the V3 gate that produces the deployment-faithful RQ3 metric.

Loads a cached ``v4_samples`` pickle and a trained V3 run (``flow.pt`` +
``thresholds.npz`` + ``cell_config.json``), runs the *direct-path* gate
(:func:`src.modeling.localization.v3_gating.gate_samples_by_v3`) on the
leave-position-out holdout, and reports how many holdout windows V3 flags — so
the ``n_holdout_gated = 0`` symptom can be diagnosed (and the metric made
computable) without retraining anything.

Run::

    python -m scripts.diagnostics.v3_gating_check \
        --samples results/runs/deepc_20260526_155457/v4_samples_all.pkl \
        --v3-run  results/runs/20260526_170001__v3deep_v3_d0_w5_s42 \
        --percentile 95 --min-events 1
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.anomaly.cnf_head import ConditionalRealNVP
from src.modeling.anomaly.threshold import PerClusterThresholds
from src.modeling.localization.v3_gating import gate_samples_by_v3
from src.modeling.localization.v4_trainer import split_samples_by_position
from src.modeling.orchestration.full_run import V4_HOLDOUT_POSITIONS_M


def _load_v3(v3_run: Path, c_dim: int):
    """Rebuild the flow + thresholds (mirrors v4_deep_sweep._load_v3)."""
    th = np.load(v3_run / "thresholds.npz")
    thresholds = PerClusterThresholds(
        centroids=th["centroids"], p95=th["p95"], p99=th["p99"],
        n_per_cluster=th["n_per_cluster"],
    )
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
    return flow, thresholds, cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", required=True)
    p.add_argument("--v3-run", required=True)
    p.add_argument("--percentile", type=int, default=95, choices=(95, 99))
    p.add_argument("--min-events", type=int, default=0)
    p.add_argument("--fallback-quantile", type=float, default=0.90)
    args = p.parse_args(argv)

    with open(args.samples, "rb") as fh:
        samples = pickle.load(fh)
    _train, holdout = split_samples_by_position(samples, V4_HOLDOUT_POSITIONS_M)
    if not holdout:
        print("no holdout samples — check V4_HOLDOUT_POSITIONS_M vs the cache")
        return 1
    c_dim = int(np.asarray(holdout[0].context).shape[0])
    flow, thresholds, cfg = _load_v3(Path(args.v3_run), c_dim)

    res = gate_samples_by_v3(
        flow, thresholds, holdout,
        percentile=args.percentile, min_events=args.min_events,
        fallback_quantile=args.fallback_quantile,
    )

    p_thr = thresholds.p95 if args.percentile == 95 else thresholds.p99
    print(f"holdout windows : {len(holdout)} across {len(res.per_recording)} recordings")
    print(f"V3 scores       : min={res.scores.min():.2f}  "
          f"p50={np.median(res.scores):.2f}  p95={np.quantile(res.scores, 0.95):.2f}  "
          f"max={res.scores.max():.2f}")
    print(f"per-cluster p{args.percentile} thresholds : "
          f"{np.array2string(np.asarray(p_thr), precision=2)}")
    print(f"strict n_gated  : {res.n_strict}  ({res.n_strict / len(holdout):.1%} of holdout)")
    print(f"final  n_gated  : {res.n_final}  "
          f"(fallback rescued {res.n_fallback_recordings} recordings)")
    print()
    print(f"{'dataset/recording':<46} {'n':>4} {'p50':>7} {'p95':>7} {'max':>7} {'strict':>6} {'fb':>3}")
    for rid, dg in sorted(res.per_recording.items()):
        print(f"{(dg['dataset_id'] + '/' + rid)[:46]:<46} {dg['n_windows']:>4} "
              f"{(dg['score_p50'] or float('nan')):>7.1f} {(dg['score_p95'] or float('nan')):>7.1f} "
              f"{(dg['score_max'] or float('nan')):>7.1f} {dg['strict_n_alerts']:>6} "
              f"{'Y' if dg['used_fallback'] else '-':>3}")
    print()
    verdict = ("COMPUTABLE: V3 flags holdout windows under the strict rule"
               if res.n_strict > 0 else
               "STRICT GATE EMPTY: holdout scores sit below the healthy threshold; "
               "use --min-events 1 (per-recording fallback) for a computable metric")
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
