"""V2 architectural sweep: CMA weight grid × context-aggregation variants.

Cells (each cell = one V2 training run from V1 init, 6 epochs on D1+D2):

  CMA-weight sweep at the original `joint_pma` aggregation:
    cma=0.1, cma=0.25  (0.5 and 1.0 already in results/full_run/v2_cma/)

  Architectural variants at the best CMA weight from the previous study (0.5):
    skip      + cma=0.5
    dual_pma  + cma=0.5

  Reference rows already on disk:
    joint_pma + cma=0.0   (results/full_run/metrics.json — vanilla V2 baseline)
    joint_pma + cma=0.5   (results/full_run/v2_cma/cma_w0_5.pt — 0.651 best so far)
    joint_pma + cma=1.0   (results/full_run/v2_cma/cma_w1_0.pt — 0.599)

Run:
    python -m scripts.diagnostics.v2_sweep
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.modeling.context.v2_ssl import V2SSLConfig, train_v2_fusion
from src.modeling.orchestration.full_run import resolved_loader, v2_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> dict:
    out_dir = REPO_ROOT / "results" / "full_run" / "v2_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = v2_config(quick=False)

    cells = [
        ("cma_w0_1_joint", {"cma_weight": 0.1, "context_mode": "joint_pma"}),
        ("cma_w0_25_joint", {"cma_weight": 0.25, "context_mode": "joint_pma"}),
        ("cma_w0_5_skip", {"cma_weight": 0.5, "context_mode": "skip"}),
        ("cma_w0_5_dual", {"cma_weight": 0.5, "context_mode": "dual_pma"}),
    ]
    cfgs = {name: V2SSLConfig(**{**asdict(base), **overrides}) for name, overrides in cells}

    v1_root = REPO_ROOT / "results" / "full_run" / "v1"
    v1_acoustic_sd = torch.load(v1_root / "acoustic.pt", map_location="cpu")
    v1_vibration_sd = torch.load(v1_root / "vibration.pt", map_location="cpu")

    print("Loading D1, D2 ...", flush=True)
    D1 = resolved_loader("d1.yaml")
    D2 = resolved_loader("d2.yaml")

    results: dict = {}
    for name, cfg in cfgs.items():
        print(
            f"\n=== {name}  cma={cfg.cma_weight}  context={cfg.context_mode} ===",
            flush=True,
        )
        t0 = time.time()
        out = train_v2_fusion(
            [D1, D2],
            cfg=cfg,
            v1_acoustic_state_dict=v1_acoustic_sd,
            v1_vibration_state_dict=v1_vibration_sd,
        )
        dt = time.time() - t0
        purity = out.rq1.get("purity", 0.0)
        nmi = out.rq1.get("nmi", 0.0)
        print(f"  done in {dt:.1f}s -- RQ1 purity {purity:.3f}  NMI {nmi:.3f}", flush=True)
        torch.save(out.encoder.state_dict(), out_dir / f"{name}.pt")
        results[name] = {
            "cma_weight": cfg.cma_weight,
            "context_mode": cfg.context_mode,
            "epochs": cfg.epochs,
            "rq1_purity": purity,
            "rq1_nmi": nmi,
            "train_loss_final": out.train_loss_history[-1],
            "val_loss_final": out.val_loss_history[-1],
            "wall_clock_s": dt,
        }

    # Reference rows for the comparison table.
    full = json.loads((REPO_ROOT / "results" / "full_run" / "metrics.json").read_text())
    cma_existing = json.loads(
        (REPO_ROOT / "results" / "full_run" / "v2_cma" / "results.json").read_text()
    )
    results["__reference__cma_w0_0_joint"] = {
        "cma_weight": 0.0,
        "context_mode": "joint_pma",
        "rq1_purity": full["stages"]["v2"]["rq1_purity"],
        "rq1_nmi": full["stages"]["v2"]["rq1_nmi"],
        "note": "vanilla V2 from full_run/metrics.json",
    }
    results["__reference__cma_w0_5_joint"] = {
        "cma_weight": 0.5,
        "context_mode": "joint_pma",
        "rq1_purity": cma_existing["cma_w0_5"]["rq1_purity"],
        "rq1_nmi": cma_existing["cma_w0_5"]["rq1_nmi"],
        "note": "best from CMA-only study",
    }
    results["__reference__cma_w1_0_joint"] = {
        "cma_weight": 1.0,
        "context_mode": "joint_pma",
        "rq1_purity": cma_existing["cma_w1_0"]["rq1_purity"],
        "rq1_nmi": cma_existing["cma_w1_0"]["rq1_nmi"],
    }

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}", flush=True)
    return results


if __name__ == "__main__":
    main()
