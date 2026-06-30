"""RQ3 localization ablation — one command to test every improvement idea.

Runs leave-one-position-out (LOPO) over the D2/D3/D4/D5 cohort on the per-knock
sample set, toggling each suggested technique independently, and prints which
ones lower the MAE.  Two axes:

TRAINING variants (each retrains the head):
  * ``baseline``           — per-knock samples, standard head (the reference).
  * ``heatmap_aux``        — + Gaussian heatmap auxiliary loss on the SRP logits
                             (#2, integral regression, Sun et al. 2018).
  * ``drop_outliers``      — out-of-hull positions removed from TRAINING (#3).
  * ``gcc_oversample``     — acoustic GCC up-sampled O× so the SRP peak resolves
                             below the voxel grid (#4).
  * ``synthetic_pretrain`` — head pretrained on simulated knocks (pysound-
                             localization forward model) then fine-tuned (#6).

SCORING variants (free — applied to every trained head's val predictions):
  * ``mean_agg``   — event-aggregated: average a position's knock predictions
    into one estimate (the canonical, deployment-faithful metric).
  * ``median_agg`` — robust version of the above.
  * ``psr_agg``    — SRP-sharpness-weighted aggregation (#5).

Leakage is controlled by LOPO + ``assert_no_position_leak`` on every fold.
Nothing runs automatically; see the footer for the one command.

Run::

    python -m scripts.sweeps.v4_localization_ablation \
        --encoder-run results/runs/<ts>__full_pipeline_b5_cma \
        --channel-modes both,tdoa_only,srp_only --seeds 42

Quick smoke (few epochs, subset)::

    python -m scripts.sweeps.v4_localization_ablation --encoder-run <dir> \
        --variants baseline,heatmap_aux --channel-modes both --quick
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
    SyntheticArraySpec,
    assert_no_position_leak,
    classify_positions,
    generate_synthetic_knock_samples,
    precompute_v4_knock_event_samples,
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

# Default set (outliers kept — `drop_outliers` removed per the small impact it
# showed).  New techniques harvested from the pysoundlocalization clone:
#   gcc_oversample  — interp the GCC below the voxel grid (they use interp=16)
#   sharp_gcc       — linear (non-circular) GCC + β-PHAT + oversample
#   bandpass        — Butterworth band-pass the crop before SRP
#   tta_crops       — multi-crop test-time augmentation per knock (amplifies the
#                     per-position aggregation win)
#   synthetic_pretrain — reverberant synthetic-knock pretraining
# Default set after two runs: aggregation is the bedrock; sharp_gcc (linear +
# β-PHAT) is the front-end winner; multi-scale replaces the decentering TTA.
# bandpass / tta_crops / synthetic_pretrain / heatmap_aux stay SELECTABLE but
# left the default since they didn't earn their cost.
ALL_VARIANTS = ["baseline", "gcc_oversample", "sharp_gcc", "multiscale",
                "sharp_multiscale"]
# Variants whose effect is a different per-knock SAMPLE build (vs a training-only
# or scoring-only change).  Each gets its own precomputed sample set.
_SAMPLE_BUILD_VARIANTS = {"gcc_oversample", "sharp_gcc", "bandpass", "tta_crops",
                          "multiscale", "sharp_multiscale", "tdoa_subsample",
                          "tdoa_slow_c", "sharp_all"}
ALL_SCORINGS = ["mean_agg", "median_agg", "psr_agg"]
_BASELINE_LOPO = {"tdoa_only": 0.132, "both": 0.171, "srp_only": 0.168}


def _knock_cfg_for_variant(variant: str, args) -> KnockEventConfig:
    """The per-knock sample-build config for a variant (base for non-build ones)."""
    common = dict(crop_seconds=args.crop_seconds)
    scales = tuple(float(x) for x in args.multiscale_seconds.split(",") if x.strip())
    sharp = dict(gcc_oversample=args.gcc_oversample, linear_corr=True,
                 phat_beta=args.phat_beta)
    if variant == "gcc_oversample":
        return KnockEventConfig(**common, gcc_oversample=args.gcc_oversample)
    if variant == "sharp_gcc":
        return KnockEventConfig(**common, **sharp)
    if variant == "multiscale":
        return KnockEventConfig(**common, crop_scales_seconds=scales)
    if variant == "sharp_multiscale":
        return KnockEventConfig(**common, **sharp, crop_scales_seconds=scales)
    if variant == "bandpass":
        return KnockEventConfig(**common, bandpass_hz=(args.bandpass_lo, args.bandpass_hi))
    if variant == "tta_crops":
        return KnockEventConfig(**common, crops_per_knock=args.tta_crops,
                                crop_jitter_seconds=args.tta_jitter)
    if variant == "tdoa_subsample":
        # Fix the integer-quantised accel TDOA: oversample + parabolic.
        return KnockEventConfig(**common, tdoa_gcc_oversample=args.tdoa_oversample)
    if variant == "tdoa_slow_c":
        # As tdoa_subsample, but with a slow (flexural) structure-borne speed.
        return KnockEventConfig(**common, tdoa_gcc_oversample=args.tdoa_oversample,
                                accel_c_ms=args.accel_c)
    if variant == "sharp_all":
        # Fully-corrected front-end: sharp acoustic GCC + sub-sample accel TDOA.
        return KnockEventConfig(**common, **sharp,
                                tdoa_gcc_oversample=args.tdoa_oversample)
    return KnockEventConfig(**common)  # baseline / heatmap_aux / synthetic_pretrain


# --------------------------------------------------------------------------- #
# Cohort + geometry
# --------------------------------------------------------------------------- #
def _gather_labeled_segments():
    D2, D3, D4, D5 = (resolved_loader(f"d{i}.yaml") for i in (2, 3, 4, 5))
    d2 = [s for s in D2.list_segments()
          if s.is_anomaly and s.spatial_label is not None and s.mode_label is not None]
    d3_segs = D3.list_segments()
    overrides = _d3_spatial_overrides(d3_segs)
    d3 = [s for s in d3_segs if s.recording_id in overrides]
    d4 = [s for s in D4.list_segments() if s.is_anomaly and s.spatial_label is not None]
    d5 = [s for s in D5.list_segments() if s.is_anomaly and s.spatial_label is not None]
    return d2 + d3 + d4 + d5, overrides


def _geometry(segments, overrides, margin_m):
    records = [(overrides.get(s.recording_id, s.spatial_label), s.mic_positions, s.vib_positions)
               for s in segments if overrides.get(s.recording_id, s.spatial_label) is not None]
    return classify_positions(records, margin_m=margin_m)


def _synthetic_arrays(segments) -> list[SyntheticArraySpec]:
    """One SyntheticArraySpec per distinct array geometry in the cohort."""
    seen: dict[tuple, SyntheticArraySpec] = {}
    for s in segments:
        key = (s.dataset_id, s.mic_positions.shape[0], s.vib_positions.shape[0])
        if key in seen:
            continue
        seen[key] = SyntheticArraySpec(
            dataset_id=s.dataset_id,
            mic_xyz=np.asarray(s.mic_positions, dtype=np.float64),
            vib_xyz=np.asarray(s.vib_positions, dtype=np.float64),
            mic_fs=int(s.segment.mic_sample_rate),
            accel_fs=int(s.segment.accel_sample_rate),
        )
    return list(seen.values())


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _score_fold(preds: np.ndarray, targets: np.ndarray, va_samples) -> dict[str, float]:
    """Per-fold error for each scoring variant (val set = one held-out position)."""
    gt = targets.mean(axis=0)  # all val windows share the GT position
    mean_agg = float(np.linalg.norm(preds.mean(axis=0) - gt))
    median_agg = float(np.linalg.norm(np.median(preds, axis=0) - gt))
    w = np.array([max(s.srp_psr, 1e-6) for s in va_samples], dtype=np.float64)
    psr_agg = float(np.linalg.norm((preds * w[:, None]).sum(0) / w.sum() - gt))
    return {"mean_agg": mean_agg, "median_agg": median_agg, "psr_agg": psr_agg}


# --------------------------------------------------------------------------- #
# One (variant, channel_mode, seed) LOPO pass
# --------------------------------------------------------------------------- #
def _run_variant(
    variant, channel_mode, seed, *, samples, fold_keys, grid, v4_cfg,
    heatmap_weight, pretrained_state,
):
    cfg = replace(v4_cfg, seed=seed, channel_mode=channel_mode)
    if variant == "heatmap_aux":
        cfg = replace(cfg, heatmap_aux_weight=heatmap_weight)
    elif variant == "low_temp":
        # Sharper soft-argmax → less centroid bias toward the grid centre.
        cfg = replace(cfg, soft_argmax_temperature=0.5)
    elif variant == "no_residual":
        # Ablate the FiLM residual: does it still help once the GCC is sharp,
        # or is it overfitting?  residual_scale_m=0 → pred = pure soft-argmax.
        cfg = replace(cfg, residual_scale_m=0.0)
    init_state = pretrained_state if variant == "synthetic_pretrain" else None

    per_scoring: dict[str, list[float]] = {sc: [] for sc in ALL_SCORINGS}
    for hold in fold_keys:
        va = [s for s in samples if _position_key(s.target_xyz) == hold]
        tr = [s for s in samples if _position_key(s.target_xyz) != hold]
        if not tr or not va:
            continue
        assert_no_position_leak(tr, va)
        try:
            res = train_v4_localization(
                samples, cfg=cfg, grid=grid, explicit_split=(tr, va), init_state=init_state
            )
        except Exception as e:
            print(f"      fold {hold} FAILED: {type(e).__name__}: {e}")
            continue
        if res.val_predictions.shape[0] == 0:
            continue
        fold = _score_fold(res.val_predictions, res.val_targets, va)
        for sc, v in fold.items():
            per_scoring[sc].append(v)
    return {sc: (float(np.mean(vs)) if vs else float("nan")) for sc, vs in per_scoring.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RQ3 localization ablation: LOPO over the per-knock cohort, "
        "toggling each technique; prints which lowers MAE. See module docstring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--encoder-run", required=True, type=Path)
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--scorings", default=",".join(ALL_SCORINGS))
    ap.add_argument("--channel-modes", default="both,tdoa_only,srp_only")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--exclude-out-of-hull", action="store_true",
                    help="Fold only over in-hull positions (outliers excluded everywhere)")
    ap.add_argument("--hull-margin-m", type=float, default=0.05)
    ap.add_argument("--crop-seconds", type=float, default=0.12)
    ap.add_argument("--gcc-oversample", type=int, default=4,
                    help="GCC interp factor (4 beat 8 on srp_only — 8 over-fits the peak)")
    ap.add_argument("--phat-beta", type=float, default=0.7,
                    help="PHAT exponent for sharp_gcc (beta<1 more robust at low SNR)")
    ap.add_argument("--multiscale-seconds", default="0.08,0.12,0.20",
                    help="crop widths for the multiscale variants (same centre)")
    ap.add_argument("--bandpass-lo", type=float, default=200.0)
    ap.add_argument("--bandpass-hi", type=float, default=6000.0)
    ap.add_argument("--tta-crops", type=int, default=3,
                    help="crops per knock for the tta_crops variant")
    ap.add_argument("--tta-jitter", type=float, default=0.02)
    ap.add_argument("--tdoa-oversample", type=int, default=16,
                    help="accel-GCC interp for tdoa_subsample/tdoa_slow_c variants")
    ap.add_argument("--accel-c", type=float, default=500.0,
                    help="structure-borne wave speed (m/s) for the tdoa_slow_c variant")
    ap.add_argument("--heatmap-weight", type=float, default=0.1)
    ap.add_argument("--synth-positions", type=int, default=60)
    ap.add_argument("--synth-reflections", type=int, default=4,
                    help="reverberation reflections in synthetic pretraining (0=free-field)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default=None, type=Path)
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
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

    print("Gathering labeled cohort + geometry ...")
    segments, overrides = _gather_labeled_segments()
    geom = _geometry(segments, overrides, args.hull_margin_m)
    in_hull_keys = {p for p, g in geom.items() if g["inside"]}
    out_keys = [p for p, g in geom.items() if not g["inside"]]
    print(f"  {len(geom)} positions, {len(out_keys)} out-of-hull")

    # Build one per-knock sample set per distinct KnockEventConfig among the
    # requested variants (variants sharing a config reuse the set).  baseline /
    # heatmap_aux / synthetic_pretrain all use the base config.
    samples_by_variant: dict[str, list] = {}
    cfg_cache: dict[KnockEventConfig, list] = {}
    for variant in variants:
        kcfg = _knock_cfg_for_variant(variant, args)
        if kcfg not in cfg_cache:
            t0 = time.time()
            cfg_cache[kcfg] = precompute_v4_knock_event_samples(
                encoder, segments, v2_cfg=v2_cfg, grid=grid,
                spatial_label_overrides=overrides, cfg=kcfg,
            )
            tag = variant if variant in _SAMPLE_BUILD_VARIANTS else "base"
            print(f"[{tag}] {len(cfg_cache[kcfg])} knock samples in {time.time()-t0:.0f}s")
        samples_by_variant[variant] = cfg_cache[kcfg]
    base_samples = samples_by_variant.get("baseline") or next(iter(cfg_cache.values()))

    # Fold keys (positions with >= 2 windows).  Outliers are KEPT (folded over)
    # unless --exclude-out-of-hull is explicitly passed.
    pos_counts: dict[tuple, int] = {}
    for s in base_samples:
        pos_counts[_position_key(s.target_xyz)] = pos_counts.get(_position_key(s.target_xyz), 0) + 1
    all_keys = sorted(p for p, n in pos_counts.items() if n >= 2)
    fold_keys = [p for p in all_keys if p in in_hull_keys] if args.exclude_out_of_hull else all_keys
    print(f"  folding over {len(fold_keys)} positions")

    # Synthetic pretraining (one head per seed, reused across channel modes/folds).
    pretrained_by_seed: dict[int, dict] = {}
    if "synthetic_pretrain" in variants:
        c_dim = int(base_samples[0].context.shape[0])
        arrays = _synthetic_arrays(segments)
        print(f"[synthetic] generating knocks for {len(arrays)} array geometries ...")
        for seed in seeds:
            syn = generate_synthetic_knock_samples(
                arrays, grid, c_dim=c_dim,
                n_positions_per_array=args.synth_positions,
                crop_seconds=args.crop_seconds, seed=seed,
                n_reflections=args.synth_reflections,
            )
            t0 = time.time()
            res = train_v4_localization(
                syn, cfg=replace(v4_cfg, seed=seed, channel_mode="both"), grid=grid
            )
            pretrained_by_seed[seed] = {k: v.detach().cpu().clone()
                                        for k, v in res.head.state_dict().items()}
            print(f"  seed {seed}: pretrained on {len(syn)} synthetic samples "
                  f"(val MAE {res.val_mae_3d:.3f} m) in {time.time()-t0:.0f}s")

    # Run the ablation matrix.
    # results[variant][channel_mode][scoring] = {median,min,max over seeds}
    results: dict = {}
    for variant in variants:
        results[variant] = {}
        for cm in channel_modes:
            per_seed_by_scoring: dict[str, list[float]] = {sc: [] for sc in ALL_SCORINGS}
            for seed in seeds:
                t0 = time.time()
                fold_means = _run_variant(
                    variant, cm, seed,
                    samples=samples_by_variant[variant],
                    fold_keys=fold_keys, grid=grid,
                    v4_cfg=v4_cfg, heatmap_weight=args.heatmap_weight,
                    pretrained_state=pretrained_by_seed.get(seed),
                )
                for sc, v in fold_means.items():
                    if not np.isnan(v):
                        per_seed_by_scoring[sc].append(v)
                print(f"  [{variant} | {cm} | seed {seed}] "
                      + ", ".join(f"{sc}={fold_means[sc]:.3f}" for sc in ALL_SCORINGS)
                      + f"  ({time.time()-t0:.0f}s)")
            results[variant][cm] = {
                sc: {
                    "median_mae_m": float(np.median(vs)) if vs else float("nan"),
                    "min_mae_m": float(np.min(vs)) if vs else float("nan"),
                    "max_mae_m": float(np.max(vs)) if vs else float("nan"),
                    "n_seeds": len(vs),
                } for sc, vs in per_seed_by_scoring.items()
            }

    out_dir = args.out_dir or (REPO_ROOT / "results" / "loc_ablation"
                               / _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "encoder_run": str(args.encoder_run), "variants": variants,
        "channel_modes": channel_modes, "seeds": seeds,
        "exclude_out_of_hull": bool(args.exclude_out_of_hull),
        "n_fold_positions": len(fold_keys),
        "out_of_hull_positions": [list(p) for p in out_keys],
        "knobs": {
            "gcc_oversample": args.gcc_oversample, "phat_beta": args.phat_beta,
            "bandpass_hz": [args.bandpass_lo, args.bandpass_hi],
            "tta_crops": args.tta_crops, "tta_jitter": args.tta_jitter,
            "heatmap_weight": args.heatmap_weight,
            "synth_positions": args.synth_positions,
            "synth_reflections": args.synth_reflections,
        },
        "baseline_published_lopo": _BASELINE_LOPO,
        "v4_cfg": asdict(v4_cfg), "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str))

    # Console table: per channel mode, variants × scorings (median MAE).
    scorings = [s.strip() for s in args.scorings.split(",") if s.strip()]
    for cm in channel_modes:
        print(f"\n==================== channel_mode = {cm} "
              f"(median MAE m over {len(seeds)} seed[s]) ====================")
        header = f"{'variant':<20}" + "".join(f"{sc:>13}" for sc in scorings)
        print(header)
        base_ma = results["baseline"][cm]["mean_agg"]["median_mae_m"] if "baseline" in results else None
        for variant in variants:
            row = f"{variant:<20}"
            for sc in scorings:
                v = results[variant][cm][sc]["median_mae_m"]
                row += f"{v:>13.3f}"
            print(row)
        if base_ma is not None and not np.isnan(base_ma):
            # Best cell overall vs baseline mean_agg (event-aggregated).
            best = min(
                (results[v][cm][sc]["median_mae_m"], v, sc)
                for v in variants for sc in scorings
                if not np.isnan(results[v][cm][sc]["median_mae_m"])
            )
            print(f"  baseline mean_agg = {base_ma:.3f} m  |  best = {best[0]:.3f} m "
                  f"via [{best[1]} + {best[2]}]  (Δ {best[0]-base_ma:+.3f} m)")
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
