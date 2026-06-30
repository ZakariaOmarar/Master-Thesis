"""Aggregate physical-detector results across the canonical seeds.

Reads ``physical_detector_eval.json`` from each seed's run dir (listed in the
multiseed report, or pass --runs) and prints median [min, max] per cohort for
ac/vib/sum AUC and detection@thr — the thesis-table format.

Run:  python -m scripts.aggregate_physical_detector
      python -m scripts.aggregate_physical_detector --runs A B C
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"
COHORTS = ("d1", "d2", "d3", "d4")
METRICS = ("ac_auc", "vib_auc", "sum_auc", "detect_at_thr")


def _canonical_runs() -> list[str]:
    reps = sorted(glob.glob(str(REPO / "results" / "reports" / "multiseed_complete_*.json")))
    if reps:
        runs = json.loads(Path(reps[-1]).read_text()).get("runs", {})
        if runs:
            return list(runs.values())
    return [Path(p).name for p in glob.glob(str(RUNS / "*__full_pipeline_b5_cma"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="explicit run dir names")
    args = ap.parse_args()
    run_names = args.runs or _canonical_runs()

    data: dict[str, dict[str, list]] = {c: {m: [] for m in METRICS} for c in COHORTS}
    used = 0
    for rn in run_names:
        p = RUNS / rn / "physical_detector_eval.json"
        if not p.exists():
            print(f"[skip] no eval json in {rn}")
            continue
        used += 1
        pc = json.loads(p.read_text())["per_cohort"]
        for c in COHORTS:
            if c in pc:
                for m in METRICS:
                    data[c][m].append(pc[c][m])

    print(f"\nphysical detector — {used} seeds, median [min, max]\n")
    print(f"{'cohort':<8} {'ac_AUC':>16} {'vib_AUC':>16} {'SUM_AUC':>16} {'detect@thr':>16}")
    for c in COHORTS:
        cells = []
        for m in METRICS:
            v = np.array(data[c][m], dtype=float)
            cells.append(f"{np.median(v):.3f}[{v.min():.2f},{v.max():.2f}]" if v.size else "—")
        print(f"{c:<8} " + " ".join(f"{x:>16}" for x in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
