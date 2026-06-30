"""Train V2 with cross-modal alignment (CMA) loss enabled, on top of the
existing V1 checkpoints, and compare RQ1 cluster purity against the vanilla
V2 trained without CMA.

CMA term: NT-Xent between V2's per-modality PMA summaries (`a_summary`,
`v_summary`) of the same window before cross-attention.  Pulls acoustic and
vibration of the same window into a shared embedding space; pushes apart
mismatched windows.  Combined with SimCLR-on-c_t and LMM, this gives the
cross-attention block aligned modality embeddings to fuse.

Run:
    python -m scripts.diagnostics.train_v2_cma
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
    out_dir = REPO_ROOT / "results" / "full_run" / "v2_cma"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = v2_config(quick=False)
    # Two CMA weights: a small one (0.5×) and a parity one (1.0×) relative to SimCLR.
    cfgs = {
        "cma_w0_5": V2SSLConfig(**{**asdict(base_cfg), "cma_weight": 0.5}),
        "cma_w1_0": V2SSLConfig(**{**asdict(base_cfg), "cma_weight": 1.0}),
    }

    v1_root = REPO_ROOT / "results" / "full_run" / "v1"
    v1_acoustic_sd = torch.load(v1_root / "acoustic.pt", map_location="cpu")
    v1_vibration_sd = torch.load(v1_root / "vibration.pt", map_location="cpu")

    print("Loading D1, D2 ...")
    D1 = resolved_loader("d1.yaml")
    D2 = resolved_loader("d2.yaml")

    results: dict = {}
    for name, cfg in cfgs.items():
        print(f"\n=== Training V2 + CMA  ({name}, weight={cfg.cma_weight}) ===")
        t0 = time.time()
        out = train_v2_fusion(
            [D1, D2],
            cfg=cfg,
            v1_acoustic_state_dict=v1_acoustic_sd,
            v1_vibration_state_dict=v1_vibration_sd,
        )
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s — RQ1 purity: {out.rq1.get('purity', 0.0):.3f}")
        torch.save(out.encoder.state_dict(), out_dir / f"{name}.pt")
        results[name] = {
            "cma_weight": cfg.cma_weight,
            "epochs": cfg.epochs,
            "rq1_purity": out.rq1.get("purity", 0.0),
            "rq1_nmi": out.rq1.get("nmi", 0.0),
            "train_loss_final": out.train_loss_history[-1],
            "val_loss_final": out.val_loss_history[-1],
            "train_simclr_final": out.train_simclr_history[-1],
            "train_lmm_final": out.train_lmm_history[-1],
            "wall_clock_s": dt,
        }

    # Reference: vanilla V2 from the full_run (cma_weight=0)
    metrics_path = REPO_ROOT / "results" / "full_run" / "metrics.json"
    if metrics_path.exists():
        full = json.loads(metrics_path.read_text())
        v2_full = full.get("stages", {}).get("v2", {})
        results["cma_w0_0_baseline"] = {
            "cma_weight": 0.0,
            "rq1_purity": v2_full.get("rq1_purity", 0.0),
            "rq1_nmi": v2_full.get("rq1_nmi", 0.0),
            "epochs": v2_full.get("epochs", 0),
            "note": "vanilla V2 from full_run/metrics.json",
        }

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    return results


if __name__ == "__main__":
    main()
