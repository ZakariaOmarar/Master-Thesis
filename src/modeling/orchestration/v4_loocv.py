"""V4 leave-one-recording-out cross-validation driver.

With 10 spatially-labeled recordings (3 D2 RandomFault single-mode +
1 D3 hit_between_Fl_Gr + 6 D4 RandomFault positions), a single 70/30
random split is too noisy to publish: per-recording MAE variance is
the dominant uncertainty in the headline V4 number.  LOOCV gives 10
independent MAE estimates and decouples "V4 head variance" from
"V2 encoder variance" (the V2 encoder is held fixed across folds).

This is the small-sample-CV discipline of Kohavi (1995), "A study of
cross-validation and bootstrap for accuracy estimation," IJCAI:
LOOCV is the unbiased estimator of generalisation error when n is
small enough that more-aggressive k-fold splits would leave folds
with too few examples to estimate the error.

Usage:

    python -m src.modeling.orchestration.v4_loocv \\
        [--quick] [--burst-aware/--no-burst-aware]

The driver expects a V2 encoder + V3 thresholds artefact from a
previous `full_run.py` invocation (in `results/full_run/v2/` and
`results/full_run/v3/`).  It does not retrain V1 / V2 / V3.  For
each of the K labeled V4 recordings, it:

  1. Re-precomputes the V4 candidate samples on **all** labeled
     recordings (so V3-gating still uses the full healthy
     distribution).
  2. Holds out that recording's windows for validation; trains V4
     on the rest.
  3. Records val MAE + bootstrap CI for the held-out recording.

Output: ``results/full_run/v4_loocv.json`` with per-fold MAE,
aggregate mean ± std, and per-recording breakdown.  All quantities
are also persisted as ``metrics.json`` entries for the run-archival
sidecar.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..context.v2_fusion import V2FusionEncoder
from ..eval import percentile_bootstrap_ci
from ..localization import (
    V4_CANDIDATE_GRID,
    V4Config,
    precompute_v4_knock_event_samples,
    train_v4_localization,
)
from .full_run import (
    REPO_ROOT,
    _d3_spatial_overrides,
    resolved_loader,
    v2_config,
    v4_config,
)
from .v4_cv_common import gated_fold_mae, load_v3_for_gating


def _qualify(s) -> str:
    return f"{Path(s.source_dir).name}/{s.recording_id}"


def _split_loocv_holdout(samples: list, hold_key: str) -> tuple[list, list]:
    """Split V4Sample list into (train_for_this_fold, val_for_this_fold)
    where the val partition contains exactly the windows of the held-out
    recording identified by `hold_key`."""
    train = [s for s in samples if _qualify(s) != hold_key]
    val = [s for s in samples if _qualify(s) == hold_key]
    return train, val


def run_loocv(
    *,
    quick: bool = False,
    burst_aware_srp: bool = True,
    epochs_override: int | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Run V4 LOOCV using an existing V2 encoder + V3 thresholds.

    Args:
      quick: pass through to `v4_config(quick=...)` for fast smoke runs.
      burst_aware_srp: use the burst-aware SRP precompute (default True
        — matches the headline configuration).
      epochs_override: override V4 epochs (default uses `v4_config`'s
        value, which is 30 for the full profile).
      out_dir: directory for the JSON summary.  Defaults to
        `results/full_run/`.

    Returns:
      dict matching the JSON schema written to disk.
    """
    out_dir = out_dir or (REPO_ROOT / "results" / "full_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    v2_path = out_dir / "v2" / "encoder.pt"
    if not v2_path.exists():
        raise FileNotFoundError(
            f"V2 encoder not found at {v2_path}.  Run `full_run.py` first; "
            f"V4 LOOCV is a post-hoc analysis on a trained V2."
        )

    print(f"V4 LOOCV: loading V2 encoder from {v2_path}")
    v2_cfg = v2_config(quick)
    v4_cfg = v4_config(quick)
    if epochs_override is not None:
        v4_cfg = V4Config(**{**asdict(v4_cfg), "epochs": int(epochs_override)})

    encoder = V2FusionEncoder.from_checkpoint(v2_path, v2_cfg)

    # Trained fusion V3 (saved in the same run dir) for the deployment-faithful
    # V3-gated per-fold MAE; degrades to ungated-only if the artefacts are absent.
    v3 = load_v3_for_gating(out_dir, embed_dim=int(v2_cfg.embed_dim), log_prefix="V4 LOOCV")

    print("V4 LOOCV: gathering labeled segments ...")
    D2 = resolved_loader("d2.yaml")
    D3 = resolved_loader("d3.yaml")
    D4 = resolved_loader("d4.yaml")
    d2_labeled = [
        s for s in D2.list_segments()
        if s.is_anomaly and s.spatial_label is not None and s.mode_label is not None
    ]
    d3_segments = D3.list_segments()
    overrides = _d3_spatial_overrides(d3_segments)
    d3_labeled = [s for s in d3_segments if s.recording_id in overrides]
    d4_labeled = [
        s for s in D4.list_segments() if s.is_anomaly and s.spatial_label is not None
    ]
    all_labeled = d2_labeled + d3_labeled + d4_labeled
    print(
        f"  D2={len(d2_labeled)}, D3={len(d3_labeled)}, D4={len(d4_labeled)}, "
        f"total={len(all_labeled)} labeled recordings"
    )

    grid = V4_CANDIDATE_GRID
    print("V4 LOOCV: precomputing per-knock samples once (used in every fold) ...")
    t0 = time.time()
    samples = precompute_v4_knock_event_samples(
        encoder, all_labeled,
        v2_cfg=v2_cfg, grid=grid,
        spatial_label_overrides=overrides,
        # Build x_for_v3 the way V3 scores it (pool + anchor) so the per-fold
        # V3-gate is on-distribution; only affects the gating input, not the
        # SRP/TDOA/c_t features the head trains on (ungated MAE unchanged).
        v3_xt_pool=(v3.xt_pool if v3 is not None else None),
        v3_anchor_norm=(v3.anchor_norm if v3 is not None else None),
    )
    print(f"  {len(samples)} per-knock candidate samples in {time.time() - t0:.1f}s")

    fold_keys = sorted({_qualify(s) for s in samples})
    print(f"V4 LOOCV: running {len(fold_keys)} folds ...")
    fold_results: list[dict] = []
    per_fold_maes: list[float] = []
    per_fold_gated_maes: list[float] = []

    for i, hold in enumerate(fold_keys, start=1):
        tr, va = _split_loocv_holdout(samples, hold)
        if len(va) < 2 or len(tr) < 4:
            print(f"  fold {i:02d} ({hold}): SKIPPED (val n={len(va)}, train n={len(tr)})")
            fold_results.append({
                "fold": i,
                "hold_out": hold,
                "skipped": True,
                "n_train": len(tr),
                "n_val": len(va),
            })
            continue
        t0 = time.time()
        # Train on the explicit (train, val) split so the held-out recording is
        # honoured exactly.  Uses the canonical trainer (same as LOPO /
        # cross-dataset), so the event-aggregated headline + channel-mode
        # handling all match the rest of RQ3.
        result = train_v4_localization(
            samples, cfg=v4_cfg, grid=grid, explicit_split=(tr, va)
        )
        elapsed = time.time() - t0
        ci = percentile_bootstrap_ci(
            np.linalg.norm(result.val_predictions - result.val_targets, axis=-1),
            n_boot=1000, seed=v4_cfg.seed,
        ) if result.val_predictions.size else None
        # result.val_mae_3d is the event-aggregated headline (one estimate per
        # held-out recording = mean of its per-knock predictions).
        mae_headline = float(result.val_mae_3d)
        # Deployment-faithful V3-gated MAE on this fold's holdout knocks (same
        # event-aggregation as the headline); reported beside, not over, ungated.
        gated = gated_fold_mae(v3, va, result.val_predictions, result.val_targets)
        fold_results.append({
            "fold": i,
            "hold_out": hold,
            "skipped": False,
            "n_train_recordings": len({_qualify(s) for s in tr}),
            "n_val_recordings": 1,
            "n_train_windows": len(tr),
            "n_val_windows": len(va),
            "val_mae_3d": mae_headline,
            "val_p95_3d": float(result.val_p95_3d),
            "val_mae_ci95_low": ci.ci_low if ci else float("nan"),
            "val_mae_ci95_high": ci.ci_high if ci else float("nan"),
            "elapsed_seconds": elapsed,
            **gated,
        })
        per_fold_maes.append(mae_headline)
        if gated["val_mae_3d_v3gated_m"] is not None:
            per_fold_gated_maes.append(gated["val_mae_3d_v3gated_m"])
        g_mae = gated["val_mae_3d_v3gated_m"]
        g_str = (f"{g_mae:.3f} m on {gated['n_val_gated']}/{gated['n_val_total']}"
                 if g_mae is not None else f"n/a ({gated['n_val_gated']}/{gated['n_val_total']} gated)")
        print(
            f"  fold {i:02d} ({hold}): MAE(ungated)={mae_headline:.3f} m "
            f"| MAE(V3-gated)={g_str} in {elapsed:.1f}s"
        )

    summary = {
        "n_folds": len(fold_keys),
        "n_folds_completed": sum(1 for r in fold_results if not r.get("skipped")),
        "fold_results": fold_results,
        "aggregate_mae_m": (
            {
                "mean": float(np.mean(per_fold_maes)) if per_fold_maes else float("nan"),
                "std": float(np.std(per_fold_maes)) if per_fold_maes else float("nan"),
                "min": float(np.min(per_fold_maes)) if per_fold_maes else float("nan"),
                "max": float(np.max(per_fold_maes)) if per_fold_maes else float("nan"),
                "n": len(per_fold_maes),
            }
        ),
        "aggregate_mae_v3gated_m": (
            {
                "mean": float(np.mean(per_fold_gated_maes)) if per_fold_gated_maes else float("nan"),
                "std": float(np.std(per_fold_gated_maes)) if per_fold_gated_maes else float("nan"),
                "min": float(np.min(per_fold_gated_maes)) if per_fold_gated_maes else float("nan"),
                "max": float(np.max(per_fold_gated_maes)) if per_fold_gated_maes else float("nan"),
                "n": len(per_fold_gated_maes),
                "n_val_gated": int(sum(r.get("n_val_gated", 0)
                                       for r in fold_results if not r.get("skipped"))),
                "n_val_total": int(sum(r.get("n_val_total", 0)
                                       for r in fold_results if not r.get("skipped"))),
            }
        ),
        "burst_aware_srp": burst_aware_srp,
        "v4_epochs": v4_cfg.epochs,
        "method": "leave_one_recording_out_cv",
    }
    json_path = out_dir / "v4_loocv.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(
        f"V4 LOOCV: mean MAE (ungated) = {summary['aggregate_mae_m']['mean']:.3f} m "
        f"± {summary['aggregate_mae_m']['std']:.3f} m "
        f"(n={summary['aggregate_mae_m']['n']} folds)"
    )
    g = summary["aggregate_mae_v3gated_m"]
    print(
        f"V4 LOOCV: mean MAE (V3-gated) = {g['mean']:.3f} m ± {g['std']:.3f} m "
        f"(n={g['n']} folds, {g['n_val_gated']}/{g['n_val_total']} knocks gated)"
    )
    print(f"V4 LOOCV: summary written to {json_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V4 leave-one-recording-out CV")
    parser.add_argument("--encoder-run", "--out-dir", dest="encoder_run", default=None, type=Path,
                        help="Run dir containing v2/encoder.pt; the v4_loocv.json summary is "
                             "written here (default: results/full_run).")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-burst-aware", action="store_true",
                        help="disable burst-aware SRP (use full-window SRP).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override V4 epochs per fold")
    args = parser.parse_args()
    run_loocv(
        quick=args.quick,
        burst_aware_srp=not args.no_burst_aware,
        epochs_override=args.epochs,
        out_dir=args.encoder_run,
    )
