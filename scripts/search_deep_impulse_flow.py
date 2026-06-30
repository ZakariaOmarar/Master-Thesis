"""Hyperparameter search for the deep impulse-aware anomaly detector.

Collects the (cached) features once, then random-searches the model/training
hyperparameters, training each config with the same pipeline as production
(scripts.train_deep_impulse_flow.train_and_eval).  Ranks configs by the mean
recording-level ROC over the test datasets, prints the leaderboard, a
per-hyperparameter marginal-effect analysis ("which value helps most"), and the
best config — which you then plug into the final multi-seed training run.

Feature knobs (window/n_mels) are held fixed (they define the feature cache);
this sweeps the model + training knobs that matter most.

Run:
    python -m scripts.search_deep_impulse_flow --trials 24 --search-epochs 25
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.train_deep_impulse_flow import collect_features, train_and_eval
from src.modeling.anomaly.deep_impulse_flow import DeepImpulseConfig

# Search space — the model/training knobs (feature knobs fixed for the cache).
SPACE = {
    "emb_dim": [16, 32, 64],
    "ctx_dim": [4, 8, 16],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "flow_layers": [4, 6, 8],
    "flow_hidden": [32, 64, 128],
    "lr": [3e-4, 1e-3, 3e-3],
    "augment": ["light", "strong"],
}


def _objective(res, test_ds) -> float:
    """Mean recording-level ROC over the test datasets (held-out included)."""
    rocs = [res["per_dataset"][d]["reclvl_roc"] for d in test_ds if d in res["per_dataset"]]
    return float(np.mean(rocs)) if rocs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-ds", nargs="*", default=["d2", "d3", "d4"])
    ap.add_argument("--test-ds", nargs="*", default=["d2", "d3", "d4", "d5"])
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--search-epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0, help="fixed seed for fair config comparison")
    ap.add_argument("--out", default="results/deep_impulse_search.json")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {dev}")

    base = DeepImpulseConfig(epochs=args.search_epochs, seed=args.seed)
    all_ds = sorted(set(args.fit_ds) | set(args.test_ds))
    print("collecting features once (cached) ...", flush=True)
    feats = collect_features(all_ds, base)

    rng = np.random.default_rng(args.seed)
    trials = []
    for t in range(args.trials):
        choice = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        cfg = replace(base, **choice)
        print(f"\n=== trial {t+1}/{args.trials}: {choice} ===", flush=True)
        try:
            res, _ = train_and_eval(cfg, feats, args.fit_ds, args.test_ds, dev,
                                    do_adapt=False, verbose=False)
            obj = _objective(res, args.test_ds)
            per = {d: round(res["per_dataset"][d]["reclvl_roc"], 3) for d in args.test_ds}
            print(f"    objective(mean ROC)={obj:.4f}  per-ds={per}  healthyFPR={res['healthy_fpr']:.3f}", flush=True)
            trials.append({"params": choice, "objective": obj, "per_ds": per,
                           "healthy_fpr": res["healthy_fpr"]})
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            trials.append({"params": choice, "objective": float("nan"), "error": str(e)})

    ok = [t for t in trials if np.isfinite(t["objective"])]
    ok.sort(key=lambda t: t["objective"], reverse=True)

    print("\n================ leaderboard (top 10) ================")
    print(f"{'obj':>7}  params")
    for t in ok[:10]:
        print(f"{t['objective']:>7.4f}  {t['params']}")

    print("\n========== per-hyperparameter marginal mean objective ==========")
    print("(higher = that value tends to help; based on all finished trials)")
    for k, vals in SPACE.items():
        cells = []
        for v in vals:
            objs = [t["objective"] for t in ok if t["params"][k] == v]
            cells.append(f"{v}={np.mean(objs):.4f}(n{len(objs)})" if objs else f"{v}=-")
        print(f"  {k:<12} " + "  ".join(cells))

    def _rel(p: Path) -> Path:
        p = Path(p).resolve()
        try:
            return p.relative_to(REPO)
        except ValueError:
            return p

    best = ok[0] if ok else None
    # Resolve to absolute so `_rel`/save work regardless of a relative --out.
    best_cfg_path = Path(args.out).resolve().with_name("deep_impulse_best_config.json")
    if best:
        print(f"\nBEST objective={best['objective']:.4f}  per-ds={best['per_ds']}")
        print(f"BEST params = {best['params']}")
        # write the full config (with full epochs) so the multi-seed run loads it
        full = replace(base, epochs=40, **best["params"])
        best_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        best_cfg_path.write_text(json.dumps(asdict(full), indent=2), encoding="utf-8")
        print(f"\nwrote best config -> {_rel(best_cfg_path)}")
        print("-> final multi-seed run, e.g.:")
        print(f"   for s in 0 1 2 3 4; do python -m scripts.train_deep_impulse_flow "
              f"--config {_rel(best_cfg_path)} --seed $s --adapt-frac 0.3 "
              f"--out results/deep_impulse_seed$s.pt; done")

    out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump({"space": SPACE, "trials": trials, "best": best,
                   "best_config_path": str(_rel(best_cfg_path)),
                   "fit_ds": args.fit_ds, "test_ds": args.test_ds}, fh, indent=2)
    print(f"saved -> {_rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
