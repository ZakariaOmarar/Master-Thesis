"""V4 leave-one-position-out cross-validation driver.

Position-keyed sibling of `v4_loocv.py`.  The headline V4 holdout MAE in the
deep campaign rests on only 5 fixed held-out positions — too noisy to publish
as a generalization claim.  LOPO-by-position folds over all ~23 labelled
positions in the D2/D3/D4/D5 cohort and reports a mean ± std MAE across folds,
which is the standard small-sample generalization estimator (Kohavi 1995).

For each unique target position (rounded to the cm grid via
`_position_key`), the driver:

  1. Pulls the precomputed V4Sample list (one cache shared across folds).
  2. Splits samples into (all-except-this-position, this-position-only).
  3. Trains V4 from scratch on the train split, validates on the held-out
     position's windows.  Records val MAE, p95, bootstrap CI.
  4. Optionally repeats per channel mode (`--all-channel-modes`).

Output:  `<out_dir>/summary.json` (mean ± std + per-modality aggregates)
+ `<out_dir>/folds.jsonl` (per-position detail).

Usage:

    python -m src.modeling.orchestration.v4_lopo_cv \\
        --encoder-run <dir> [--v3-run <dir>] [--samples-cache <path>] \\
        [--all-channel-modes] [--out-dir <dir>] [--seed 42]

Unlike `v4_loocv.py` this driver is wired directly into the deep-campaign
pipeline (Phase 4 in `scripts/campaigns/run_deep_v3v4_campaign.py`).  It does not
retrain V1/V2/V3 and reuses the campaign's V2 encoder + precomputed V4
sample cache.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..context.v2_fusion import V2FusionEncoder
from ..eval import percentile_bootstrap_ci
from ..localization import (
    V4_CANDIDATE_GRID,
    train_v4_localization,
)
from ..localization.v4_trainer import _position_key
from .full_run import REPO_ROOT, v2_config, v4_config
from .v4_cv_common import (
    CHANNEL_MODES,
    gated_fold_mae,
    load_or_precompute_cv_samples,
    load_v3_for_gating,
)


def _qualify_position(s) -> tuple[float, float, float]:
    return _position_key(s.target_xyz)


def _position_str(p: tuple[float, float, float]) -> str:
    return f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"


def _split_lopo(samples: list, hold_key: tuple[float, float, float]) -> tuple[list, list]:
    train = [s for s in samples if _qualify_position(s) != hold_key]
    val = [s for s in samples if _qualify_position(s) == hold_key]
    return train, val


def run_lopo(
    *,
    encoder_run: Path,
    v3_run: Path | None = None,
    samples_cache: Path | None = None,
    all_channel_modes: bool = False,
    out_dir: Path | None = None,
    seed: int = 42,
    quick: bool = False,
    burst_aware_srp: bool = True,
    min_fold_windows: int = 2,
) -> dict:
    """Run V4 leave-one-position-out cross-validation.

    Returns the summary dict (also written to `out_dir/summary.json`).
    """
    out_dir = out_dir or (REPO_ROOT / "results" / "lopo")
    out_dir.mkdir(parents=True, exist_ok=True)

    v2_cfg = v2_config(quick)
    v4_cfg = v4_config(quick)

    print(f"V4 LOPO: loading V2 encoder from {encoder_run}/v2/encoder.pt")
    encoder = V2FusionEncoder.from_checkpoint(encoder_run / "v2" / "encoder.pt", v2_cfg)

    # Load the trained fusion V3 (from --v3-run, else the encoder run dir) so each
    # fold also reports the deployment-faithful V3-gated MAE — the same filtering
    # full_run Stage 5b applies.  Degrades to ungated-only if V3 is absent.
    v3 = load_v3_for_gating(
        v3_run or encoder_run, embed_dim=int(v2_cfg.embed_dim), log_prefix="V4 LOPO")

    samples = load_or_precompute_cv_samples(
        encoder, v2_cfg,
        samples_cache=samples_cache,
        burst_aware_srp=burst_aware_srp,
        log_prefix="V4 LOPO",
        v3=v3,
    )
    grid = V4_CANDIDATE_GRID

    # Unique positions = fold keys.  Skip positions with too few windows to
    # produce a meaningful val MAE.
    pos_counts: dict[tuple[float, float, float], int] = {}
    for s in samples:
        pos_counts[_qualify_position(s)] = pos_counts.get(_qualify_position(s), 0) + 1
    fold_keys = sorted(p for p, n in pos_counts.items() if n >= min_fold_windows)
    skipped = sorted(p for p, n in pos_counts.items() if n < min_fold_windows)
    print(f"V4 LOPO: {len(fold_keys)} folds ({len(skipped)} positions skipped "
          f"with < {min_fold_windows} windows)")

    modes = list(CHANNEL_MODES) if all_channel_modes else ["both"]

    folds_path = out_dir / "folds.jsonl"
    folds_path.write_text("")  # truncate

    per_mode_results: dict[str, list[dict]] = {m: [] for m in modes}
    fold_preds: dict[int, dict[str, tuple]] = {}  # fi -> {mode: (preds, targets)} for #15

    for fi, hold in enumerate(fold_keys, start=1):
        tr, va = _split_lopo(samples, hold)
        if not tr or not va:
            continue
        for mode in modes:
            cfg = replace(v4_cfg, seed=seed, channel_mode=mode)
            t0 = time.time()
            try:
                res = train_v4_localization(
                    samples, cfg=cfg, grid=grid, explicit_split=(tr, va)
                )
            except Exception as e:
                print(f"  fold {fi}/{len(fold_keys)} [{mode}] @ {_position_str(hold)}: "
                      f"FAILED ({type(e).__name__}: {e})")
                per_mode_results[mode].append({
                    "fold": fi, "position_xyz": list(hold),
                    "channel_mode": mode, "error": f"{type(e).__name__}: {e}",
                })
                continue
            errs = np.linalg.norm(res.val_predictions - res.val_targets, axis=-1)
            fold_preds.setdefault(fi, {})[mode] = (
                np.asarray(res.val_predictions), np.asarray(res.val_targets))
            ci_low, ci_high = float("nan"), float("nan")
            if errs.size >= 2:
                ci = percentile_bootstrap_ci(errs, n_boot=1000, seed=seed)
                ci_low, ci_high = ci.ci_low, ci.ci_high
            # res.val_mae_3d is the event-aggregated headline (one estimate per
            # recording = mean of its per-knock predictions).
            mae_headline = float(res.val_mae_3d)
            # Deployment-faithful V3-gated MAE on this fold's holdout knocks
            # (same event-aggregation as the headline); does not touch mae_headline.
            gated = gated_fold_mae(v3, va, res.val_predictions, res.val_targets)
            rec = {
                "fold": fi, "position_xyz": list(hold),
                "channel_mode": mode,
                "n_train_windows": len(tr),
                "n_val_windows": len(va),
                "val_mae_3d_m": mae_headline,
                "val_p95_3d_m": float(res.val_p95_3d),
                "train_mae_3d_m": float(res.train_mae_3d),
                "ci95_low_m": ci_low,
                "ci95_high_m": ci_high,
                "elapsed_seconds": float(time.time() - t0),
                **gated,
            }
            per_mode_results[mode].append(rec)
            with folds_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            g_mae = gated["val_mae_3d_v3gated_m"]
            g_str = (f"{g_mae:.3f}m on {gated['n_val_gated']}/{gated['n_val_total']}"
                     if g_mae is not None else f"n/a ({gated['n_val_gated']}/{gated['n_val_total']} gated)")
            print(f"  fold {fi}/{len(fold_keys)} [{mode}] @ {_position_str(hold)}: "
                  f"MAE(ungated)={mae_headline:.3f}m (train {res.train_mae_3d:.3f}m) "
                  f"| MAE(V3-gated)={g_str} n_val={len(va)} in {time.time()-t0:.0f}s")

    # #15 — late-fusion paradigm: uniform average of the two unimodal heads'
    # predictions on each held-out position, then EVENT-aggregated (mean over
    # the fold's knocks) to match the headline metric (guarded; only when both
    # unimodal modes were run and their val windows align).
    if "srp_only" in modes and "tdoa_only" in modes:
        lf_rows: list[dict] = []
        for fi, mp in fold_preds.items():
            sp, td = mp.get("srp_only"), mp.get("tdoa_only")
            if sp is None or td is None or sp[0].shape != td[0].shape or sp[0].size == 0:
                continue
            lf_pred = 0.5 * (sp[0] + td[0])
            # Aggregate the fold's per-knock late-fused predictions into one
            # estimate (the held-out position's GT is shared across its knocks).
            lf_err = float(np.linalg.norm(lf_pred.mean(axis=0) - sp[1].mean(axis=0)))
            lf_rows.append({"fold": fi, "channel_mode": "late_fusion_uniform",
                            "val_mae_3d_m": lf_err,
                            "n_val_windows": int(sp[0].shape[0])})
        if lf_rows:
            per_mode_results["late_fusion_uniform"] = lf_rows
            modes = modes + ["late_fusion_uniform"]

    aggregate: dict[str, dict] = {}
    for mode, rows in per_mode_results.items():
        ok = [r for r in rows if "error" not in r]
        if not ok:
            aggregate[mode] = {"n_folds": 0, "mean_mae_m": float("nan"),
                               "std_mae_m": float("nan")}
            continue
        maes = np.array([r["val_mae_3d_m"] for r in ok])
        # V3-gated aggregate: mean over folds that produced a gated MAE (some
        # folds may have zero V3-flagged holdout knocks -> None), plus the pooled
        # kept/total knock tally.  Reported alongside, never replacing, ungated.
        gated_maes = np.array(
            [r["val_mae_3d_v3gated_m"] for r in ok if r.get("val_mae_3d_v3gated_m") is not None])
        n_gated = int(sum(r.get("n_val_gated", 0) for r in ok))
        n_gated_total = int(sum(r.get("n_val_total", 0) for r in ok))
        aggregate[mode] = {
            "n_folds": int(len(ok)),
            "n_skipped_or_failed": int(len(rows) - len(ok)),
            "mean_mae_m": float(maes.mean()),
            "std_mae_m": float(maes.std()),
            "min_mae_m": float(maes.min()),
            "max_mae_m": float(maes.max()),
            "median_mae_m": float(np.median(maes)),
            "n_folds_v3gated": int(gated_maes.size),
            "mean_mae_v3gated_m": float(gated_maes.mean()) if gated_maes.size else float("nan"),
            "std_mae_v3gated_m": float(gated_maes.std()) if gated_maes.size else float("nan"),
            "n_val_gated": n_gated,
            "n_val_total": n_gated_total,
        }

    summary = {
        "encoder_run": str(encoder_run),
        "v3_run": str(v3_run) if v3_run else None,
        "seed": int(seed),
        "n_unique_positions": len(pos_counts),
        "n_folds": len(fold_keys),
        "skipped_positions": [list(p) for p in skipped],
        "channel_modes": modes,
        "aggregate_per_mode": aggregate,
        "v4_cfg": asdict(replace(v4_cfg, seed=seed)),
        "method": "leave_one_position_out_cv",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("V4 LOPO: per-mode mean MAE (ungated) — "
          + ", ".join(f"{m}={aggregate[m].get('mean_mae_m', float('nan')):.3f}m "
                      f"± {aggregate[m].get('std_mae_m', float('nan')):.3f}m"
                      for m in modes))
    print("V4 LOPO: per-mode mean MAE (V3-gated) — "
          + ", ".join(f"{m}={aggregate[m].get('mean_mae_v3gated_m', float('nan')):.3f}m "
                      f"± {aggregate[m].get('std_mae_v3gated_m', float('nan')):.3f}m "
                      f"[{aggregate[m].get('n_val_gated', 0)}/{aggregate[m].get('n_val_total', 0)} knocks]"
                      for m in modes))
    print(f"V4 LOPO: summary written to {out_dir / 'summary.json'}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-run", required=True, type=Path,
                   help="Run dir containing v2/encoder.pt")
    p.add_argument("--v3-run", default=None, type=Path,
                   help="Dir with the trained fusion V3 (v3/ or v3_fusion/) used "
                        "to compute the per-fold V3-gated MAE. Defaults to "
                        "--encoder-run (full_run saves V3 there).")
    p.add_argument("--samples-cache", default=None, type=Path,
                   help="Shared V4Sample pickle (re-used across folds)")
    p.add_argument("--all-channel-modes", action="store_true",
                   help="Run all 4 channel modes per fold (4× wall time)")
    p.add_argument("--out-dir", default=None, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    run_lopo(
        encoder_run=args.encoder_run, v3_run=args.v3_run,
        samples_cache=args.samples_cache,
        all_channel_modes=args.all_channel_modes,
        out_dir=args.out_dir, seed=args.seed, quick=args.quick,
    )


if __name__ == "__main__":
    main()
