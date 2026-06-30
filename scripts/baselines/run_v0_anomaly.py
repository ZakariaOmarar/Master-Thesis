"""Run the full V0 anomaly baseline (Khamaisi reference) and report RQ2.

Sweeps every ``dataset × modality × model`` cell, fits each scorer on healthy
data, and scores it through the proposed head's own RQ2 protocol (per-cluster
percentile thresholding, healthy-alert calibration, synthetic-anomaly ROC-AUC
ladder, anomaly-cohort alert ranking).  Writes one JSON artefact and prints a
comparison table.

Models (Khamaisi et al. 2025 trio + the Experiments-chapter density baseline):
  lstm_ae  — LSTM autoencoder, per-window reconstruction MSE
  kmeans   — distance to nearest healthy K-means centroid
  ocsvm    — One-Class SVM signed distance
  kde      — Gaussian KDE negative log-density (PCA-whitened)

Run::

    python -m scripts.baselines.run_v0_anomaly                      # full sweep
    python -m scripts.baselines.run_v0_anomaly --datasets d2 d3     # subset
    python -m scripts.baselines.run_v0_anomaly --models kmeans ocsvm
    python -m scripts.baselines.run_v0_anomaly --quick              # fast smoke
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader
from src.modeling.anomaly_baselines.lstm_ae import V0Config
from src.modeling.anomaly_baselines.v0_evaluation import (
    ALL_MODELS,
    MODALITIES,
    evaluate_v0_anomaly,
)

DEFAULT_DATASETS = ("d1", "d2", "d3", "d4", "d5")
RESULTS_DIR = REPO / "results" / "v0_anomaly"


def _log(msg: str) -> None:
    line = f"[{_dt.datetime.now():%H:%M:%S}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def _loader(dataset_id: str) -> TestDatasetLoader:
    spec = DatasetSpec.from_yaml(REPO / "configs" / "datasets" / f"{dataset_id}.yaml")
    return TestDatasetLoader(spec)


def _cfg(args: argparse.Namespace) -> V0Config:
    # val_ratio default 0.3 matches the head's V3 split; raise it for tiny
    # single-rate corpora (e.g. one campaign's 3 healthy recordings) so the
    # held-out pool still has >= 2 recordings for the domain-shift split.
    if args.quick:
        return V0Config(
            n_mels=32, n_fft=512, hop_length=256, window_seconds=1.0,
            hidden_dim=32, latent_dim=8, n_layers=1, epochs=3,
            val_ratio=args.val_ratio, seed=args.seed,
        )
    return V0Config(epochs=args.epochs, val_ratio=args.val_ratio, seed=args.seed)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES)
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    p.add_argument("--percentile", type=int, default=95, choices=(95, 99))
    p.add_argument("--n-clusters", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30, help="LSTM-AE training epochs")
    p.add_argument("--n-boot", type=int, default=500, help="synthetic-AUC bootstrap resamples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.3,
                   help="held-out fraction; raise for tiny single-rate corpora")
    p.add_argument("--quick", action="store_true", help="tiny config for a fast smoke run")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    cfg = _cfg(args)
    snr_ladder = (-10.0, -5.0, 0.0, 5.0, 10.0)

    # Pool the requested datasets into one corpus (mirrors the head's training
    # on ANOM_LOADERS); only this gives enough healthy recordings for a fair
    # held-out calibration split.
    loaders = []
    for ds in args.datasets:
        try:
            loaders.append(_loader(ds))
        except Exception as e:
            _log(f"{ds}: loader failed — {type(e).__name__}: {e}")
    if not loaders:
        _log("no datasets loaded; nothing to do")
        return 1
    corpus = "+".join(args.datasets)

    results: list[dict] = []
    for modality in args.modalities:
        for model in args.models:
            tag = f"{corpus}/{modality}/{model}"
            t0 = time.time()
            try:
                res = evaluate_v0_anomaly(
                    loaders, model, modality, cfg,
                    percentile=args.percentile, n_clusters=args.n_clusters,
                    snr_db_list=snr_ladder, n_boot=args.n_boot,
                )
                results.append(res.to_dict())
                by_ds = res.details.get("roc_auc_by_dataset", {})
                rank = " ".join(f"{k}={v:.2f}" for k, v in sorted(by_ds.items()))
                _log(
                    f"{tag} {time.time()-t0:.0f}s — "
                    f"ROC-AUC={res.roc_auc:.3f} "
                    f"FPR(in-dist={res.fpr_in_distribution:.3f} "
                    f"shift={res.fpr_domain_shift:.3f}) | per-ds AUC: {rank}"
                )
            except Exception as e:
                _log(f"{tag} skipped: {type(e).__name__}: {e}")
                results.append({
                    "dataset_ids": args.datasets, "modality": modality,
                    "model": model, "skipped": f"{type(e).__name__}: {e}",
                })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or (RESULTS_DIR / f"v0_anomaly_{ts}.json")
    payload = {
        "generated": ts,
        "config": {
            "datasets": args.datasets, "modalities": args.modalities,
            "models": args.models, "percentile": args.percentile,
            "n_clusters": args.n_clusters, "epochs": cfg.epochs,
            "seed": args.seed, "quick": args.quick, "snr_ladder": list(snr_ladder),
        },
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _log(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    _print_table(results, args.percentile)
    return 0


def _print_table(results: list[dict], percentile: int) -> None:
    rows = [r for r in results if "skipped" not in r]
    if not rows:
        _log("no successful cells to tabulate")
        return

    # Table 1 — detection + the calibration contrast (with 95% CIs).
    print()
    print(f"  RQ2 V0 anomaly baselines  (FPR target = {(100 - percentile) / 100:.2f})")
    h1 = (f"  {'modality':<10} {'model':<8} {'ROC-AUC':>16} "
          f"{'FPR in-dist':>16} {'FPR shift':>16}")
    print(h1)
    print("  " + "-" * (len(h1) - 2))

    def _ci(v, ci):
        lo, hi = ci if isinstance(ci, (list, tuple)) and len(ci) == 2 else (float("nan"),) * 2
        return f"{v:.2f}[{lo:.2f},{hi:.2f}]"

    for r in sorted(rows, key=lambda x: (x["modality"], x["model"])):
        print(
            f"  {r['modality']:<10} {r['model']:<8} "
            f"{_ci(r['roc_auc'], r.get('roc_auc_ci')):>16} "
            f"{_ci(r['fpr_in_distribution'], r.get('fpr_in_ci')):>16} "
            f"{_ci(r['fpr_domain_shift'], r.get('fpr_shift_ci')):>16}"
        )

    # Table 2 — per-campaign healthy-vs-anomaly ROC-AUC.
    datasets = sorted({ds for r in rows for ds in r.get("roc_auc_by_dataset", {})})
    if datasets:
        print()
        print("  Healthy-vs-anomaly ROC-AUC per campaign (1.0 = perfectly separable)")
        h2 = f"  {'modality':<10} {'model':<8} " + " ".join(f"{d:>7}" for d in datasets)
        print(h2)
        print("  " + "-" * (len(h2) - 2))
        for r in sorted(rows, key=lambda x: (x["modality"], x["model"])):
            by_ds = r.get("roc_auc_by_dataset", {})
            cells = " ".join(f"{by_ds.get(d, float('nan')):>7.2f}" for d in datasets)
            print(f"  {r['modality']:<10} {r['model']:<8} {cells}")

    print()
    print("  Read: ROC-AUC is within-campaign detection (Khamaisi's headline) -- V0 detects")
    print("  anomalies competently. The story is the FPR contrast: 'in-dist' calibrates the")
    print("  threshold to the condition under test (should sit near the target); 'shift'")
    print("  calibrates to OTHER conditions. The gap between them is the domain-shift failure")
    print("  the conditional head removes. CIs are bootstrap (AUC) / Wilson (FPR).")


if __name__ == "__main__":
    raise SystemExit(main())
