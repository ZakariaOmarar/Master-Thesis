"""Multi-seed V0 domain-shift FPR: is the OC-SVM collapse robust, or one lucky split?

The single-draw domain-shift false-positive rate (for example, the acoustic OC-SVM
at 0.95) depends on which held-out conditions land on each side of the transfer
split. This re-runs the V0 domain-shift evaluation over several seeds. Each seed
re-draws the train/held-out recording split, the per-cluster threshold fit, and the
shift_fit/shift_eval condition split, so the reported statistic is the distribution
of the held-out healthy alert rate under threshold transfer rather than a single
realization. The claim that no unconditional baseline is domain-robust then rests on
how often, and how badly, each scorer's calibration breaks across seeds.

Run::

    python -m scripts.baselines.v0_domain_shift_multiseed
    python -m scripts.baselines.v0_domain_shift_multiseed --seeds 42 1337 2024 7 99 --include-lstm
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader
from src.modeling.anomaly_baselines.lstm_ae import V0Config
from src.modeling.anomaly_baselines.v0_evaluation import evaluate_v0_anomaly

DATASETS = ("d1", "d2", "d3", "d4", "d5")
COLLAPSE_THRESHOLD = 0.20  # held-out shift FPR above this means calibration broke (4x target)


def _loader(ds: str) -> TestDatasetLoader:
    return TestDatasetLoader(DatasetSpec.from_yaml(REPO / "configs" / "datasets" / f"{ds}.yaml"))


def _log(msg: str) -> None:
    line = f"[{_dt.datetime.now():%H:%M:%S}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024, 7, 99])
    ap.add_argument("--models", nargs="+", default=["kmeans", "ocsvm", "kde"])
    ap.add_argument("--modalities", nargs="+", default=["acoustic", "vibration"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--include-lstm", action="store_true", help="also run the (slow) acoustic LSTM-AE")
    args = ap.parse_args(argv)

    loaders = [_loader(d) for d in args.datasets]
    models = list(args.models) + (["lstm_ae"] if args.include_lstm else [])
    cells: dict[tuple[str, str], list[dict]] = {}

    for seed in args.seeds:
        cfg = V0Config(seed=seed)
        for modality in args.modalities:
            for model in models:
                if model == "lstm_ae" and modality == "vibration":
                    continue  # rate-incompatible pool, skipped by design
                t0 = time.time()
                try:
                    res = evaluate_v0_anomaly(loaders, model, modality, cfg, n_boot=0)
                except Exception as e:
                    _log(f"seed {seed} {modality}/{model} FAILED: {type(e).__name__}: {e}")
                    continue
                shift_ds = sorted((res.details.get("domain_shift_fpr_by_dataset") or {}).keys())
                cells.setdefault((modality, model), []).append({
                    "seed": seed, "roc_auc": res.roc_auc,
                    "fpr_in": res.fpr_in_distribution, "fpr_shift": res.fpr_domain_shift,
                    "shift_eval_datasets": shift_ds,
                })
                _log(f"seed {seed} {modality:9} {model:8} shift={res.fpr_domain_shift:.3f} "
                     f"(in={res.fpr_in_distribution:.3f}, eval={shift_ds}) {time.time()-t0:.0f}s")

    def _arr(recs, key):
        return np.array([r[key] for r in recs if r[key] == r[key]], dtype=float)

    agg: dict[str, dict] = {}
    for (modality, model), recs in cells.items():
        shifts, ins, rocs = _arr(recs, "fpr_shift"), _arr(recs, "fpr_in"), _arr(recs, "roc_auc")
        agg[f"{modality}/{model}"] = {
            "roc_auc_mean": round(float(rocs.mean()), 3) if rocs.size else None,
            "fpr_in_mean": round(float(ins.mean()), 3) if ins.size else None,
            "fpr_shift_per_seed": [round(float(x), 3) for x in shifts],
            "fpr_shift_mean": round(float(shifts.mean()), 3) if shifts.size else None,
            "fpr_shift_std": round(float(shifts.std()), 3) if shifts.size else None,
            "fpr_shift_min": round(float(shifts.min()), 3) if shifts.size else None,
            "fpr_shift_max": round(float(shifts.max()), 3) if shifts.size else None,
            "n_seeds": int(shifts.size),
            "n_collapse": int((shifts > COLLAPSE_THRESHOLD).sum()),
        }

    out = {
        "generated": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "seeds": args.seeds, "collapse_threshold": COLLAPSE_THRESHOLD,
        "note": ("Each seed re-draws the train/held-out split, the per-cluster threshold fit, "
                 "and the shift_fit/shift_eval condition split. fpr_shift is the held-out healthy "
                 "alert rate under a threshold transferred to disjoint conditions."),
        "aggregate": agg,
        "per_seed": {f"{m}/{md}": recs for (m, md), recs in cells.items()},
    }
    out_dir = REPO / "results" / "v0_anomaly"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"v0_domain_shift_multiseed_{out['generated']}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    _log("")
    _log("=== domain-shift FPR across seeds (per scorer) ===")
    _log(f"{'scorer':18} {'ROC':>5} {'FPRin':>6} {'shiftMean':>9} {'[min,max]':>13} {'collapse':>9}")
    for cell, a in agg.items():
        rng = f"[{a['fpr_shift_min']},{a['fpr_shift_max']}]"
        _log(f"{cell:18} {a['roc_auc_mean']!s:>5} {a['fpr_in_mean']!s:>6} {a['fpr_shift_mean']!s:>9} "
             f"{rng:>13} {str(a['n_collapse'])+'/'+str(a['n_seeds']):>9}")
    _log(f"wrote {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
