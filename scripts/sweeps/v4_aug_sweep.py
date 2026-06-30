"""V4-only augmentation sweep — Phase 6 of the campaign.

Reuses a completed baseline_v2 run's V2 encoder + V4 samples; only the
V4 head is retrained per cell.  This isolates the V4-augmentation knob
sweep from V1/V2/V3 cohort noise — the same V4 training data (precomputed
SRP volumes, TDOA tokens, c_t) is consumed by every cell, so any
between-cell delta is attributable to augmentation alone.

Cells (9 total): ``v4_pos{1,5,10}_srp{02,10,20}`` —
``target_pos_noise_m × srp_volume_noise_std``:
  - pos1=0.002 (V4Config default)
  - pos5=0.010
  - pos10=0.020
  - srp02=0.02 (V4Config default)
  - srp10=0.10
  - srp20=0.20

Output: ``results/runs/<ts>__v4aug_<cell>/`` with ``metrics.json`` +
``cell_config.json``.  Sample-precomputation happens once at startup
and is shared across cells in the same invocation; if `--cell` is set
to a single cell, that cost is amortised over fewer cells but the
sample arrays still dominate per-cell wall-clock budget.

Run::

    python -m scripts.sweeps.v4_aug_sweep --baseline-run results/runs/<ts>__baseline_v2
    python -m scripts.sweeps.v4_aug_sweep --baseline-run <path> --cell v4_pos5_srp10
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.localization import (
    V4_CANDIDATE_GRID,
    precompute_v4_samples,
    train_v4_localization,
)
from src.modeling.orchestration.full_run import (
    REPO_ROOT,
    _d3_spatial_overrides,
    resolved_loader,
    v2_config,
    v4_config,
)

# Augmentation level dicts — pos = target_pos_noise_m, srp = srp_volume_noise_std.
_POS_LEVELS: dict[str, float] = {
    "pos1": 0.002,   # V4Config default
    "pos5": 0.010,
    "pos10": 0.020,
}

_SRP_LEVELS: dict[str, float] = {
    "srp02": 0.02,   # V4Config default
    "srp10": 0.10,
    "srp20": 0.20,
}


def _all_cells() -> list[str]:
    return [f"v4_{p}_{s}" for p in _POS_LEVELS for s in _SRP_LEVELS]


def _apply_v4_aug_cell(cell_id: str, v4_cfg):
    parts = cell_id.split("_")
    if len(parts) != 3 or parts[0] != "v4":
        raise ValueError(f"malformed v4 cell id: {cell_id!r}")
    p_key, s_key = parts[1], parts[2]
    if p_key not in _POS_LEVELS or s_key not in _SRP_LEVELS:
        raise ValueError(f"unknown axis level in {cell_id!r}")
    return replace(
        v4_cfg,
        target_pos_noise_m=_POS_LEVELS[p_key],
        srp_volume_noise_std=_SRP_LEVELS[s_key],
    )


def _log(msg: str, log_path: Path) -> None:
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(line + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline-run", required=True,
        help="Path to a completed baseline_v2 run dir (must contain v2/encoder.pt).",
    )
    p.add_argument(
        "--cell", default=None,
        help=f"Single cell id; omit to run all 9.  Choices: {_all_cells()}",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    baseline_dir = Path(args.baseline_run)
    if not baseline_dir.exists():
        raise SystemExit(f"baseline run dir does not exist: {baseline_dir}")
    v2_state = baseline_dir / "v2" / "encoder.pt"
    if not v2_state.exists():
        raise SystemExit(f"v2/encoder.pt not found in baseline run: {v2_state}")

    cells = [args.cell] if args.cell else _all_cells()
    for cid in cells:
        # Trigger ValueError early for typos so we don't pay sample-precompute
        # cost to discover a bad cell id at the very end.
        _apply_v4_aug_cell(cid, v4_config(False))

    # Build V2 encoder + load weights from baseline.  v2_cfg dims must match
    # baseline; we read the orchestrator default which is what baseline_v2 ran.
    v2_cfg = v2_config(False)
    v4_cfg = v4_config(False)
    encoder = V2FusionEncoder.from_checkpoint(v2_state, v2_cfg)

    # Loaders + spatial-label resolution + V4 sample precompute (shared
    # across all cells in this invocation).
    print("Loading D2/D3/D4 loaders and precomputing V4 samples ...")
    t0 = time.time()
    LOADERS = {dsid: resolved_loader(f"{dsid}.yaml") for dsid in ("d2", "d3", "d4")}
    d2_labeled = [
        s for s in LOADERS["d2"].list_segments()
        if s.is_anomaly and s.spatial_label is not None and s.mode_label is not None
    ]
    d3_segments = LOADERS["d3"].list_segments()
    overrides = _d3_spatial_overrides(d3_segments)
    d3_labeled = [s for s in d3_segments if s.recording_id in overrides]
    d4_labeled = [
        s for s in LOADERS["d4"].list_segments() if s.is_anomaly and s.spatial_label is not None
    ]
    grid = V4_CANDIDATE_GRID
    v4_samples = precompute_v4_samples(
        encoder, d2_labeled + d3_labeled + d4_labeled,
        v2_cfg=v2_cfg, grid=grid,
        spatial_label_overrides=overrides,
        burst_aware_srp=True, burst_seconds=0.10,
    )
    print(f"Precomputed {len(v4_samples)} V4 samples in {time.time()-t0:.0f}s")
    if len(v4_samples) < 4:
        raise SystemExit(
            f"V4 sweep aborted — only {len(v4_samples)} labelled samples "
            f"available (need ≥ 4 for train/val split)."
        )

    for cell_id in cells:
        cell_cfg = _apply_v4_aug_cell(cell_id, v4_cfg)
        cell_cfg = replace(cell_cfg, seed=args.seed)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "results" / "runs" / f"{ts}__v4aug_{cell_id}_s{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "v4").mkdir(exist_ok=True)
        log_path = out_dir / "run_log.txt"
        _log(f"cell={cell_id} target_pos_noise_m={cell_cfg.target_pos_noise_m} "
             f"srp_volume_noise_std={cell_cfg.srp_volume_noise_std}", log_path)

        # Persist resolved config alongside metrics for reproducibility.
        (out_dir / "cell_config.json").write_text(json.dumps({
            "cell": cell_id,
            "seed": args.seed,
            "baseline_run": str(baseline_dir),
            "v4_cfg": asdict(cell_cfg),
        }, indent=2, default=str))

        t0 = time.time()
        try:
            res = train_v4_localization(v4_samples, cfg=cell_cfg, grid=grid)
        except Exception as e:
            _log(f"  V4 cell FAILED: {type(e).__name__}: {e}", log_path)
            (out_dir / "metrics.json").write_text(json.dumps({
                "cell": cell_id, "seed": args.seed,
                "error": f"{type(e).__name__}: {e}",
            }, indent=2))
            continue
        dt = time.time() - t0
        _log(f"  V4 trained in {dt:.0f}s — val MAE={res.val_mae_3d:.4f} m "
             f"early_stopped_epoch={res.early_stopped_epoch}", log_path)
        torch.save(res.head.state_dict(), out_dir / "v4" / "head.pt")
        (out_dir / "metrics.json").write_text(json.dumps({
            "cell": cell_id,
            "seed": args.seed,
            "v4": {
                "epochs_planned": cell_cfg.epochs,
                "early_stopped_epoch": res.early_stopped_epoch,
                "best_val_loss": res.best_val_loss,
                "train_loss_final": float(res.train_loss_history[-1]) if res.train_loss_history else float("nan"),
                "val_loss_final": float(res.val_loss_history[-1]) if res.val_loss_history else float("nan"),
                "val_mae_3d": float(res.val_mae_3d),
                "val_p95_3d": float(res.val_p95_3d),
                "val_mae_ci_low": float(res.val_mae_ci_low),
                "val_mae_ci_high": float(res.val_mae_ci_high),
                "n_train_recordings": len(res.train_recording_ids),
                "n_val_recordings": len(res.val_recording_ids),
            },
        }, indent=2, default=str))
        print(f"Wrote {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
