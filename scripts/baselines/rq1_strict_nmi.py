"""Strict RQ1 mode-discovery NMI on a finished full_pipeline run.

The in-pipeline sanity gate reports the model-*selection* cohort (a held-out
subset), which is optimistic.  This script computes the **strict** number the
Chapter-6 headline should use: cluster-purity / NMI of the trained encoder's
representation over *all* labelled Pump+Standstill+Turbine healthy windows of
D1+D2 (no held-out split), exactly the reeval_k3 cohort, at K=3.

It loads the run's own V1/V2 encoders (current architecture), so it must be run
against a run produced by the current code.  Writes
``<run-dir>/rq1_strict_nmi.json``.

Run::

    python -m scripts.baselines.rq1_strict_nmi --run-dir results/runs/<id>
    python -m scripts.baselines.rq1_strict_nmi            # newest full_pipeline run
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.diagnostics.reeval_k3 as R
from src.modeling.context.cluster_metric import cluster_purity_and_nmi
from src.modeling.context.v2_ssl import (
    _collate,
    _PairedGroupedBatchSampler,
    _PairedWindowedDataset,
)
from src.modeling.encoders import PerModalityEncoder
from src.modeling.orchestration.full_run import v2_config


def _newest_run() -> Path | None:
    cands = sorted(
        glob.glob(str(REPO / "results" / "runs" / "*__full_pipeline_b5_cma")),
        key=os.path.getmtime, reverse=True,
    )
    for c in cands:
        if (Path(c) / "v2" / "encoder.pt").exists():
            return Path(c)
    return None


@torch.no_grad()
def _v1_strict(run: Path, name: str, segs, cfg) -> dict:
    enc = PerModalityEncoder(
        modality=name, feature_dim=cfg.feature_dim,
        embed_dim=cfg.embed_dim, n_heads=cfg.n_heads,
    )
    enc.load_state_dict(torch.load(run / "v1" / f"{name}.pt", map_location="cpu"))
    enc.eval()
    ds = _PairedWindowedDataset(segs, cfg)
    sampler = _PairedGroupedBatchSampler(ds, cfg.batch_size, shuffle=False, seed=cfg.seed)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=_collate)
    embs, labels = [], []
    for b in loader:
        feat = b["ac_feat"] if name == "acoustic" else b["vib_feat"]
        xyz = b["ac_xyz"] if name == "acoustic" else b["vib_xyz"]
        _, summary = enc(feat, xyz, b["dataset_idx"])
        embs.append(summary.cpu().numpy())
        labels.extend(b["mode_label"])
    m = cluster_purity_and_nmi(np.concatenate(embs), labels, n_clusters=3, seed=0)
    return {"nmi": float(m["nmi"]), "purity": float(m["purity"]), "n_windows": int(len(labels))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="full_pipeline run dir with v1/ + v2/. Default: newest.")
    args = ap.parse_args(argv)

    run = args.run_dir.resolve() if args.run_dir else _newest_run()
    if run is None or not (run / "v2" / "encoder.pt").exists():
        print("No usable run dir (need v1/ + v2/encoder.pt).")
        return 1
    print(f"[rq1-strict] run = {run}")

    cfg = v2_config(False)
    cfg_clean, segs = R._gather_healthy_only(cfg)
    from collections import Counter
    rec_per_mode = dict(Counter(s.mode_label for s in segs))
    print(f"[rq1-strict] strict cohort recordings/mode = {rec_per_mode}")

    out: dict = {
        "generated": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "run_dir": str(run.relative_to(REPO)) if str(run).startswith(str(REPO)) else str(run),
        "cohort": "all labelled D1+D2 healthy Pump/Standstill/Turbine windows (K=3, no holdout)",
        "recordings_per_mode": rec_per_mode,
        "strict": {},
    }
    for name in ("acoustic", "vibration"):
        try:
            out["strict"][f"v1_{name}"] = _v1_strict(run, name, segs, cfg_clean)
        except Exception as e:
            out["strict"][f"v1_{name}"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        enc = R._load_encoder(run / "v2" / "encoder.pt", cfg_clean, context_mode="joint_pma")
        if enc is None:
            out["strict"]["v2_fusion"] = {"error": "incompatible checkpoint"}
        else:
            m = R._purity_k3(enc, segs, cfg_clean)
            out["strict"]["v2_fusion"] = {
                "nmi": float(m["nmi"]), "purity": float(m["purity"]),
                "n_windows": int(m["n_windows"]),
            }
    except Exception as e:
        out["strict"]["v2_fusion"] = {"error": f"{type(e).__name__}: {e}"}

    out_path = run / "rq1_strict_nmi.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[rq1-strict] STRICT NMI (the Chapter-6 headline):")
    for k, v in out["strict"].items():
        if "nmi" in v:
            print(f"    {k:<14} NMI={v['nmi']:.4f} purity={v['purity']:.4f} n={v['n_windows']}")
        else:
            print(f"    {k:<14} {v.get('error')}")
    print(f"[rq1-strict] wrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
