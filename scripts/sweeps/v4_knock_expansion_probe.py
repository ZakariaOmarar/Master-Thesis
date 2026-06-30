"""RQ3 probe: per-knock expansion + outlier-aware LOPO, head-to-head vs baseline.

What it answers
---------------
1. Does localizing **every knock** (transient-centred crops) instead of
   fixed windows lower the leave-one-position-out (LOPO) MAE?  It reruns the
   exact published LOPO protocol on two sample sets — `window` (the current
   `precompute_v4_samples`) and `knock` (`precompute_v4_knock_event_samples`) —
   so the only thing that changes is the sampling unit.
2. How much of the residual error is **geometric outliers** — knock positions
   on/outside the convex hull of the sensor array that no head can resolve?
   Every aggregate is reported twice: over all positions, and over the
   in-footprint positions only (outliers listed separately).

Leakage is controlled by construction: folds are leave-one-POSITION-out, so all
of a position's knocks (across recordings) stay on one side of the split.
`assert_no_position_leak` re-checks every fold.

This script trains only the small V4 head; it reuses a V2 encoder from a
completed run and never touches V1/V2/V3.  Nothing is run automatically — see
the module footer for the commands.

Run (full)::

    python -m scripts.sweeps.v4_knock_expansion_probe \
        --encoder-run results/runs/<ts>__full_pipeline_b5_cma \
        --modes window,knock --channel-modes tdoa_only,srp_only,both \
        --exclude-out-of-hull --seeds 42,7,99

Quick smoke (few epochs, one seed, classify only first)::

    python -m scripts.sweeps.v4_knock_expansion_probe --encoder-run <dir> --classify-only
    python -m scripts.sweeps.v4_knock_expansion_probe --encoder-run <dir> --quick --seeds 42
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.localization import (
    V4_CANDIDATE_GRID,
    KnockEventConfig,
    assert_no_position_leak,
    classify_positions,
    precompute_v4_knock_event_samples,
    precompute_v4_samples,
    train_v4_localization,
)
from src.modeling.localization.v4_trainer import _position_key
from src.modeling.orchestration.full_run import (
    REPO_ROOT,
    _d3_spatial_overrides,
    resolved_loader,
    v2_config,
    v4_config,
)

# Published LOPO baseline (results/.../lopo/summary.json) for context in prints.
_BASELINE_LOPO_MEAN_MAE = {"tdoa_only": 0.132, "both": 0.171, "srp_only": 0.168}


# --------------------------------------------------------------------------- #
# Cohort assembly (keeps segments so we can classify array geometry)
# --------------------------------------------------------------------------- #
def _gather_labeled_segments():
    """Return `(segments, overrides)` for the D2/D3/D4/D5 supervised cohort."""
    D2 = resolved_loader("d2.yaml")
    D3 = resolved_loader("d3.yaml")
    D4 = resolved_loader("d4.yaml")
    D5 = resolved_loader("d5.yaml")
    d2 = [
        s for s in D2.list_segments()
        if s.is_anomaly and s.spatial_label is not None and s.mode_label is not None
    ]
    d3_segs = D3.list_segments()
    overrides = _d3_spatial_overrides(d3_segs)
    d3 = [s for s in d3_segs if s.recording_id in overrides]
    d4 = [s for s in D4.list_segments() if s.is_anomaly and s.spatial_label is not None]
    d5 = [s for s in D5.list_segments() if s.is_anomaly and s.spatial_label is not None]
    segs = d2 + d3 + d4 + d5
    print(f"  labeled recordings: D2={len(d2)} D3={len(d3)} D4={len(d4)} D5={len(d5)} "
          f"total={len(segs)}")
    return segs, overrides


def _position_geometry(segments, overrides, *, margin_m: float) -> dict:
    """Footprint verdict per unique position (in/out of array convex hull)."""
    records = []
    for s in segments:
        pos = overrides.get(s.recording_id, s.spatial_label)
        if pos is None:
            continue
        records.append((pos, s.mic_positions, s.vib_positions))
    return classify_positions(records, margin_m=margin_m)


# --------------------------------------------------------------------------- #
# LOPO evaluation of one sample set
# --------------------------------------------------------------------------- #
def _lopo_one_seed(samples, modes, seed, grid, v4_cfg, fold_keys):
    """Per-position MAE for each mode at one seed.

    Returns `{mode: {position_key: mae}}` (position-keyed, so a skipped or
    failed fold simply omits that key rather than desyncing a positional list).
    Adds a `late_fusion_uniform` mode when both unimodal heads ran.
    """
    per_mode: dict[str, dict[tuple, float]] = {m: {} for m in modes}
    fold_preds: dict[tuple, dict[str, tuple]] = {}
    for hold in fold_keys:
        tr = [s for s in samples if _position_key(s.target_xyz) != hold]
        va = [s for s in samples if _position_key(s.target_xyz) == hold]
        if not tr or not va:
            continue
        assert_no_position_leak(tr, va)  # leak guard, every fold
        for mode in modes:
            cfg = replace(v4_cfg, seed=seed, channel_mode=mode)
            try:
                res = train_v4_localization(samples, cfg=cfg, grid=grid, explicit_split=(tr, va))
            except Exception as e:
                print(f"    fold {hold} [{mode}] FAILED: {type(e).__name__}: {e}")
                continue
            per_mode[mode][hold] = float(res.val_mae_3d)
            fold_preds.setdefault(hold, {})[mode] = (
                np.asarray(res.val_predictions), np.asarray(res.val_targets)
            )
    # Late fusion (uniform average of the two unimodal heads), if both present.
    if "srp_only" in modes and "tdoa_only" in modes:
        lf: dict[tuple, float] = {}
        for hold, mp in fold_preds.items():
            sp, td = mp.get("srp_only"), mp.get("tdoa_only")
            if sp is None or td is None or sp[0].shape != td[0].shape or sp[0].size == 0:
                continue
            err = np.linalg.norm(0.5 * (sp[0] + td[0]) - sp[1], axis=-1)
            lf[hold] = float(err.mean())
        if lf:
            per_mode["late_fusion_uniform"] = lf
    return per_mode


def _evaluate_sampleset(name, samples, modes, seeds, grid, v4_cfg, geom,
                        *, exclude_out_of_hull, min_fold_windows=2):
    """Run multi-seed LOPO; aggregate per mode across folds then seeds."""
    pos_counts: dict[tuple, int] = {}
    for s in samples:
        pos_counts[_position_key(s.target_xyz)] = pos_counts.get(_position_key(s.target_xyz), 0) + 1
    all_keys = sorted(p for p, n in pos_counts.items() if n >= min_fold_windows)

    def _is_outlier(p):
        g = geom.get(p)
        return (g is not None) and (not g["inside"])

    out_keys = [p for p in all_keys if _is_outlier(p)]
    in_keys = [p for p in all_keys if not _is_outlier(p)]
    fold_keys = in_keys if exclude_out_of_hull else all_keys
    print(f"[{name}] {len(samples)} samples, {len(all_keys)} positions "
          f"({len(out_keys)} out-of-hull outliers); folding over {len(fold_keys)}")

    modes_with_lf = list(modes) + (
        ["late_fusion_uniform"] if {"srp_only", "tdoa_only"} <= set(modes) else []
    )
    # seed -> mode -> {position_key: mae}
    seed_results: dict[int, dict[str, dict[tuple, float]]] = {}
    for seed in seeds:
        t0 = time.time()
        per_mode = _lopo_one_seed(samples, modes, seed, grid, v4_cfg, all_keys)
        seed_results[seed] = per_mode
        print(f"  seed {seed}: " + ", ".join(
            f"{m}={np.mean(list(per_mode[m].values())):.3f}m"
            for m in modes_with_lf if per_mode.get(m)
        ) + f"  ({time.time()-t0:.0f}s)")

    # Aggregate: per seed compute mean MAE over the selected fold_keys, then take
    # median/min/max across seeds.  Mirrors the thesis "median [min,max] over
    # seeds" reporting.
    keyset = set(fold_keys)
    aggregate: dict[str, dict] = {}
    for mode in modes_with_lf:
        per_seed_means = []
        for seed in seeds:
            by_pos = seed_results[seed].get(mode, {})
            sel = [m for p, m in by_pos.items() if p in keyset]
            if sel:
                per_seed_means.append(float(np.mean(sel)))
        if per_seed_means:
            arr = np.array(per_seed_means)
            aggregate[mode] = {
                "median_mae_m": float(np.median(arr)),
                "min_mae_m": float(arr.min()),
                "max_mae_m": float(arr.max()),
                "n_seeds": len(per_seed_means),
                "n_folds": len(fold_keys),
            }

    # Per-position median MAE (over seeds) for the two headline modes — exposes
    # which positions drive the error and confirms they are the out-of-hull ones.
    pos_mode_maes: dict[tuple, dict[str, list[float]]] = {p: {} for p in all_keys}
    for seed in seeds:
        per_mode = seed_results[seed]
        for mode in ("tdoa_only", "both"):
            for p, m in per_mode.get(mode, {}).items():
                pos_mode_maes.setdefault(p, {}).setdefault(mode, []).append(m)
    per_position = {}
    for p in all_keys:
        g = geom.get(p, {})
        per_position[f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"] = {
            "inside_hull": bool(g.get("inside", True)),
            "signed_distance_m": float(g.get("min_signed_distance_m", float("nan"))),
            "n_windows": pos_counts[p],
            "tdoa_only_median_mae_m": (
                float(np.median(pos_mode_maes[p]["tdoa_only"]))
                if pos_mode_maes[p].get("tdoa_only") else None
            ),
            "both_median_mae_m": (
                float(np.median(pos_mode_maes[p]["both"]))
                if pos_mode_maes[p].get("both") else None
            ),
        }

    return {
        "n_samples": len(samples),
        "n_positions_total": len(all_keys),
        "out_of_hull_positions": [list(p) for p in out_keys],
        "folded_over": "in_hull_only" if exclude_out_of_hull else "all_positions",
        "aggregate_per_mode": aggregate,
        "per_position": per_position,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RQ3 probe: per-knock expansion + outlier-aware LOPO vs "
        "baseline. See module docstring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--encoder-run", required=True, type=Path,
                    help="Run dir containing v2/encoder.pt")
    ap.add_argument("--modes", default="window,knock",
                    help="Comma list of sample builders to compare: window,knock")
    ap.add_argument("--channel-modes", default="tdoa_only,srp_only,both",
                    help="Comma list of V4 channel modes")
    ap.add_argument("--seeds", default="42",
                    help="Comma list of seeds (thesis uses 5: 42,7,99,123,2024)")
    ap.add_argument("--exclude-out-of-hull", action="store_true",
                    help="Drop out-of-footprint positions from the headline aggregate")
    ap.add_argument("--hull-margin-m", type=float, default=0.05)
    ap.add_argument("--crop-seconds", type=float, default=0.12)
    ap.add_argument("--max-events", type=int, default=24)
    ap.add_argument("--noise-floor-mult", type=float, default=3.0)
    ap.add_argument("--classify-only", action="store_true",
                    help="Only print the in/out-of-hull position table and exit")
    ap.add_argument("--quick", action="store_true", help="Few epochs (smoke)")
    ap.add_argument("--out-dir", default=None, type=Path)
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    channel_modes = [m.strip() for m in args.channel_modes.split(",") if m.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    grid = V4_CANDIDATE_GRID

    v2_cfg = v2_config(args.quick)
    v4_cfg = v4_config(args.quick)
    enc_path = args.encoder_run / "v2" / "encoder.pt"
    if not enc_path.exists():
        raise SystemExit(f"encoder not found: {enc_path}")
    print(f"Loading V2 encoder from {enc_path}")
    encoder = V2FusionEncoder.from_checkpoint(enc_path, v2_cfg)

    print("Gathering labeled cohort + array geometry ...")
    segments, overrides = _gather_labeled_segments()
    geom = _position_geometry(segments, overrides, margin_m=args.hull_margin_m)
    n_out = sum(1 for g in geom.values() if not g["inside"])
    print(f"  {len(geom)} unique positions, {n_out} out-of-hull (margin={args.hull_margin_m} m):")
    for p, g in sorted(geom.items()):
        tag = "IN " if g["inside"] else "OUT"
        print(f"    [{tag}] ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  "
              f"d_hull={g['min_signed_distance_m']:+.3f} m  ({g['method']})")
    if args.classify_only:
        return

    # Build the requested sample sets (shared across all folds/seeds).
    samplesets: dict[str, list] = {}
    if "window" in modes:
        t0 = time.time()
        samplesets["window"] = precompute_v4_samples(
            encoder, segments, v2_cfg=v2_cfg, grid=grid,
            spatial_label_overrides=overrides,
            burst_aware_srp=True, burst_seconds=0.10,
            restrict_to_knock_intervals=True,
        )
        print(f"[window] precomputed {len(samplesets['window'])} samples "
              f"in {time.time()-t0:.0f}s")
    if "knock" in modes:
        t0 = time.time()
        kcfg = KnockEventConfig(
            crop_seconds=args.crop_seconds, max_events=args.max_events,
            noise_floor_mult=args.noise_floor_mult,
        )
        samplesets["knock"] = precompute_v4_knock_event_samples(
            encoder, segments, v2_cfg=v2_cfg, grid=grid,
            spatial_label_overrides=overrides, cfg=kcfg,
        )
        print(f"[knock] precomputed {len(samplesets['knock'])} samples "
              f"in {time.time()-t0:.0f}s")

    out_dir = args.out_dir or (REPO_ROOT / "results" / "knock_expansion"
                               / _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "encoder_run": str(args.encoder_run),
        "channel_modes": channel_modes,
        "seeds": seeds,
        "exclude_out_of_hull": bool(args.exclude_out_of_hull),
        "hull_margin_m": args.hull_margin_m,
        "knock_cfg": {"crop_seconds": args.crop_seconds, "max_events": args.max_events,
                      "noise_floor_mult": args.noise_floor_mult},
        "v4_cfg": asdict(v4_cfg),
        "baseline_lopo_mean_mae": _BASELINE_LOPO_MEAN_MAE,
        "results": {},
    }
    for name, samples in samplesets.items():
        print(f"\n=== Evaluating sample set: {name} ===")
        report["results"][name] = _evaluate_sampleset(
            name, samples, channel_modes, seeds, grid, v4_cfg, geom,
            exclude_out_of_hull=args.exclude_out_of_hull,
        )

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str))

    # Console comparison table.
    print("\n================ SUMMARY (median MAE over seeds, metres) ================")
    print(f"{'sampleset':<8} {'channel':<20} {'MAE':>7}  baseline  delta")
    for name in samplesets:
        agg = report["results"][name]["aggregate_per_mode"]
        for mode, row in agg.items():
            base = _BASELINE_LOPO_MEAN_MAE.get(mode)
            delta = f"{row['median_mae_m']-base:+.3f}" if base else "   -  "
            base_s = f"{base:.3f}" if base else "  -  "
            print(f"{name:<8} {mode:<20} {row['median_mae_m']:>7.3f}  {base_s:>7}  {delta}")
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
