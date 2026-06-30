"""V4 cross-dataset transfer driver.

Train V4 on one set of datasets (e.g. D1+D2+D3+D4), evaluate on a disjoint
set (e.g. D5 only).  Confirms that the V4 head generalizes across recording
sessions / rig states — not just across positions within the same session.

For each direction (`train_ids → test_ids`), the driver:
  1. Pulls the precomputed V4Sample list shared with the campaign.
  2. Splits via `split_samples_by_dataset(samples, holdout=test_ids)`.
  3. Trains V4 from scratch on the train split; evaluates on test split.
  4. Optionally repeats per channel mode (`--all-channel-modes`).

Output:  `<out_dir>/summary.json` (per-direction × per-modality MAE).

Usage:

    python -m src.modeling.orchestration.v4_cross_dataset \\
        --encoder-run <dir> [--samples-cache <path>] [--all-channel-modes] \\
        [--out-dir <dir>] [--seed 42]

Phase 5 of `scripts/campaigns/run_deep_v3v4_campaign.py` invokes this on the V4 winner
only.  Reuses the same V2 encoder + V4 sample cache as Phase 2 / Phase 4
(no re-precompute).
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
    split_samples_by_dataset,
    train_v4_localization,
)
from .full_run import REPO_ROOT, v2_config, v4_config
from .v4_cv_common import (
    CHANNEL_MODES,
    gated_fold_mae,
    load_or_precompute_cv_samples,
    load_v3_for_gating,
)

# Direction = (label, train_dataset_ids, test_dataset_ids).
# Primary: train on the four older sessions, test on the newer D5 session.
# Secondary: reverse for symmetry sanity (much smaller train cohort).
_DEFAULT_DIRECTIONS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("d1to4_to_d5", ("d2", "d3", "d4"), ("d5",)),
    ("d5_to_d4",    ("d5",),             ("d4",)),
)


def run_cross_dataset(
    *,
    encoder_run: Path,
    v3_run: Path | None = None,
    samples_cache: Path | None = None,
    all_channel_modes: bool = False,
    directions: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = _DEFAULT_DIRECTIONS,
    out_dir: Path | None = None,
    seed: int = 42,
    quick: bool = False,
    burst_aware_srp: bool = True,
) -> dict:
    out_dir = out_dir or (REPO_ROOT / "results" / "cross_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    v2_cfg = v2_config(quick)
    v4_cfg = v4_config(quick)

    print(f"V4 cross-dataset: loading V2 encoder from {encoder_run}/v2/encoder.pt")
    encoder = V2FusionEncoder.from_checkpoint(encoder_run / "v2" / "encoder.pt", v2_cfg)

    # Trained fusion V3 (from --v3-run, else the encoder run dir) for the
    # deployment-faithful V3-gated MAE column; degrades to ungated-only if absent.
    v3 = load_v3_for_gating(
        v3_run or encoder_run, embed_dim=int(v2_cfg.embed_dim),
        log_prefix="V4 cross-dataset")

    samples = load_or_precompute_cv_samples(
        encoder, v2_cfg,
        samples_cache=samples_cache,
        burst_aware_srp=burst_aware_srp,
        log_prefix="V4 cross-dataset",
        v3=v3,
    )
    grid = V4_CANDIDATE_GRID
    modes = list(CHANNEL_MODES) if all_channel_modes else ["both"]

    results: dict[str, dict] = {}
    for label, train_ids, test_ids in directions:
        # Drop samples whose dataset_id is in neither set (e.g. d2 when
        # train=d5, test=d4).  The split helper would otherwise put them
        # in the train half; here we want disjoint partitions only.
        keep_ids = set(train_ids) | set(test_ids)
        sub_samples = [s for s in samples if s.dataset_id in keep_ids]
        tr, te = split_samples_by_dataset(sub_samples, set(test_ids))
        if not tr or not te:
            print(f"  [{label}] SKIPPED (train={len(tr)}, test={len(te)})")
            results[label] = {"error": "empty train or test split",
                              "n_train": len(tr), "n_test": len(te)}
            continue
        n_train_positions = len({tuple(s.target_xyz) for s in tr})
        n_test_positions = len({tuple(s.target_xyz) for s in te})
        per_mode: dict[str, dict] = {}
        for mode in modes:
            cfg = replace(v4_cfg, seed=seed, channel_mode=mode)
            t0 = time.time()
            try:
                res = train_v4_localization(
                    sub_samples, cfg=cfg, grid=grid, explicit_split=(tr, te)
                )
            except Exception as e:
                per_mode[mode] = {"error": f"{type(e).__name__}: {e}"}
                continue
            errs = np.linalg.norm(res.val_predictions - res.val_targets, axis=-1)
            ci_low, ci_high = float("nan"), float("nan")
            if errs.size >= 2:
                ci = percentile_bootstrap_ci(errs, n_boot=1000, seed=seed)
                ci_low, ci_high = ci.ci_low, ci.ci_high
            mae_headline = float(res.val_mae_3d)  # event-aggregated headline
            # Deployment-faithful V3-gated MAE on the test split's knocks (same
            # event-aggregation as the headline); reported beside, not over, ungated.
            gated = gated_fold_mae(v3, te, res.val_predictions, res.val_targets)
            per_mode[mode] = {
                "val_mae_3d_m": mae_headline,
                "val_p95_3d_m": float(res.val_p95_3d),
                "train_mae_3d_m": float(res.train_mae_3d),
                "ci95_low_m": ci_low,
                "ci95_high_m": ci_high,
                "elapsed_seconds": float(time.time() - t0),
                **gated,
            }
            g_mae = gated["val_mae_3d_v3gated_m"]
            g_str = (f"{g_mae:.3f}m on {gated['n_val_gated']}/{gated['n_val_total']}"
                     if g_mae is not None else f"n/a ({gated['n_val_gated']}/{gated['n_val_total']} gated)")
            print(f"  [{label}/{mode}] MAE(ungated)={mae_headline:.3f}m "
                  f"(train {res.train_mae_3d:.3f}m) | MAE(V3-gated)={g_str} "
                  f"n_train_pos={n_train_positions} n_test_pos={n_test_positions} "
                  f"in {time.time()-t0:.0f}s")
        results[label] = {
            "train_dataset_ids": list(train_ids),
            "test_dataset_ids": list(test_ids),
            "n_train_windows": len(tr),
            "n_test_windows": len(te),
            "n_train_positions": n_train_positions,
            "n_test_positions": n_test_positions,
            "per_channel_mode": per_mode,
        }

    summary = {
        "encoder_run": str(encoder_run),
        "seed": int(seed),
        "channel_modes": modes,
        "directions": results,
        "v4_cfg": asdict(replace(v4_cfg, seed=seed)),
        "method": "cross_dataset_transfer",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"V4 cross-dataset: summary written to {out_dir / 'summary.json'}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-run", required=True, type=Path,
                   help="Run dir containing v2/encoder.pt")
    p.add_argument("--v3-run", default=None, type=Path,
                   help="Dir with the trained fusion V3 (v3/ or v3_fusion/) used "
                        "for the per-direction V3-gated MAE. Defaults to "
                        "--encoder-run (full_run saves V3 there).")
    p.add_argument("--samples-cache", default=None, type=Path,
                   help="Shared V4Sample pickle (re-used; precomputed once)")
    p.add_argument("--all-channel-modes", action="store_true",
                   help="Run all 4 channel modes per direction (4× wall time)")
    p.add_argument("--out-dir", default=None, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    run_cross_dataset(
        encoder_run=args.encoder_run,
        v3_run=args.v3_run,
        samples_cache=args.samples_cache,
        all_channel_modes=args.all_channel_modes,
        out_dir=args.out_dir, seed=args.seed, quick=args.quick,
    )


if __name__ == "__main__":
    main()
