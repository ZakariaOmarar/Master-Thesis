"""Aggregate deep-impulse-flow results across seeds -> median [min, max].

Reads results/deep_impulse_seed*.json (one per seed, written by
train_deep_impulse_flow.py --out results/deep_impulse_seed<N>.pt) and prints the
thesis-format table for zero-shot and few-shot-adapted detection.

Run:  python -m scripts.aggregate_deep_impulse
      python -m scripts.aggregate_deep_impulse --glob "results/deep_impulse_seed*.json"
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _agg(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    return f"{np.median(v):.3f}[{v.min():.2f},{v.max():.2f}]" if v.size else "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/deep_impulse_seed*.json")
    args = ap.parse_args()
    files = sorted(glob.glob(str(REPO / args.glob)))
    if not files:
        print(f"no files match {args.glob}")
        return 1
    runs = [json.loads(Path(f).read_text()) for f in files]
    n = len(runs)
    test_ds = list(runs[0]["per_dataset"].keys())
    print(f"{n} seeds: {[Path(f).stem for f in files]}\n")

    print("=== zero-shot (median [min,max]) ===")
    print(f"{'ds':<5} {'reclvl_ROC':>18} {'reclvl_PR':>18} {'flag@thr':>18} {'heldout':>8}")
    for d in test_ds:
        roc = [r["per_dataset"][d]["reclvl_roc"] for r in runs]
        pr = [r["per_dataset"][d]["reclvl_pr"] for r in runs]
        fl = [r["per_dataset"][d]["flag_at_thr"] for r in runs]
        ho = runs[0]["per_dataset"][d].get("held_out", False)
        print(f"{d:<5} {_agg(roc):>18} {_agg(pr):>18} {_agg(fl):>18} {'YES' if ho else '':>8}")

    if any("adapted" in r["per_dataset"][d] for r in runs for d in test_ds):
        print("\n=== few-shot adapted (held-out campaigns, median [min,max]) ===")
        print(f"{'ds':<5} {'reclvl_ROC':>18} {'reclvl_PR':>18} {'flag@thr':>18} {'eval_FPR':>18}")
        for d in test_ds:
            if not any("adapted" in r["per_dataset"][d] for r in runs):
                continue
            A = [r["per_dataset"][d]["adapted"] for r in runs if "adapted" in r["per_dataset"][d]]
            print(f"{d:<5} {_agg([a['reclvl_roc'] for a in A]):>18} "
                  f"{_agg([a['reclvl_pr'] for a in A]):>18} "
                  f"{_agg([a['flag_at_thr'] for a in A]):>18} "
                  f"{_agg([a['eval_fpr'] for a in A]):>18}")

    hf = [r["healthy_fpr"] for r in runs]
    print(f"\nhealthy FPR: {_agg(hf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
