"""Classical localization baselines with bootstrap CIs (audit #12).

``full_run`` computes SRP-PHAT per recording but only persists mean/p95/median,
so Table 6.1's classical rows carry no interval. This recomputes SRP-PHAT per
recording for D2/D3/D4 and bootstraps a 95% CI on the mean error. (accel-TDOA
CIs come from ``scripts.finalize_results`` via the saved ``per_recording``
errors of the ``v0_multilateration`` stage.)

Classical and CPU-only — fast, no GPU.

Run::

    python -m scripts.baselines.localization_baselines_ci
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader
from src.modeling.anomaly_baselines.srp_phat_baseline import (
    SRPConfig,
    evaluate_srp_phat,
)

DATASETS = ("d2", "d3", "d4")


def _loader(ds: str) -> TestDatasetLoader:
    spec = DatasetSpec.from_yaml(REPO / "configs" / "datasets" / f"{ds}.yaml")
    return TestDatasetLoader(spec)


def _boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="accepted for driver parity; ignored")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args(argv)

    rows: dict[str, dict] = {}
    for ds in args.datasets:
        try:
            recs = evaluate_srp_phat(_loader(ds), SRPConfig())
        except Exception as e:
            rows[ds] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{ds}: SRP-PHAT FAILED ({type(e).__name__}: {e})", flush=True)
            continue
        errs = np.array([r["error_m"] for r in recs], dtype=float)
        if errs.size == 0:
            rows[ds] = {"n": 0}
            print(f"{ds}: no ground-truth recordings", flush=True)
            continue
        lo, hi = _boot_ci(errs)
        rows[ds] = {
            "n": int(errs.size),
            "mean_error_m": round(float(errs.mean()), 4),
            "p95_error_m": round(float(np.percentile(errs, 95)), 4),
            "ci95_low_m": round(lo, 4),
            "ci95_high_m": round(hi, 4),
        }
        print(f"{ds}: SRP-PHAT MAE={errs.mean():.3f} m  95% CI [{lo:.3f}, {hi:.3f}]  (n={errs.size})",
              flush=True)

    out = {
        "generated": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "method": "SRP-PHAT per-recording error, 2000x bootstrap CI on the mean",
        "srp_phat": rows,
    }
    out_dir = REPO / "results" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"localization_baselines_ci_{out['generated']}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
