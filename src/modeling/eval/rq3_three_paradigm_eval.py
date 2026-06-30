"""R3.4-eval — RQ3 paradigm comparison built on `run_v4_three_paradigms.py` outputs.

Reads per-paradigm `val_predictions.npz` arrays + classical-baseline JSONs
from a ``results/runs/<run>__v4_three_paradigms/`` directory and produces:

  rq3_paradigm_comparison.json    — full numbers including Late-Fusion rules
  rq3_paradigm_comparison.md      — human-readable table for Chapter 6

Late-Fusion rules implemented (operate on per-window predictions, no
extra training):

  * **LF uniform** — ``pred = 0.5 · pred_a + 0.5 · pred_v``.
  * **LF weighted (per-axis)** — fit ``w_a, w_v`` (per spatial axis,
    ``[w_x, w_y, w_z]`` × 2 modalities) via least squares on
    ``pred_a, pred_v -> target`` over the val cohort.  Closed-form, no
    iteration; one weight set fit on the held-out val cohort and applied
    to itself (so this is best-case-on-val; real-world deployment would
    fit on a separate calibration cohort, which we don't have at this rig
    scale).  Reported with that caveat.
  * **LF confidence-gated** — per-window, use whichever pipeline's
    ``||delta|| = ||pred - init_xyz||`` is smaller (smaller residual norm =
    head trusted the spatial init more).  Falls back to LF uniform when
    one pipeline lacks an `init_xyz` (e.g. a classical row).

Paired-bootstrap significance: V4-fusion vs LF-weighted vs the better
unimodal — does the multimodal architecture matter once we have all the
unimodal predictions?

Run::

    python -m src.modeling.eval.rq3_three_paradigm_eval \\
        --v4-three-run results/runs/<id>__v4_three_paradigms
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .statistics import paired_bootstrap_test

REPO = Path(__file__).resolve().parents[3]


@dataclass
class _ParadigmRow:
    paradigm: str
    row: str
    n: int
    mae_m: float
    p95_m: float
    ci95_low_m: float | None = None
    ci95_high_m: float | None = None


def _bootstrap_mae_ci(errors: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95 % CI for the MAE."""
    if errors.size < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, errors.size, size=errors.size)
        means.append(float(np.mean(errors[idx])))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _per_window_errors(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-window 3-D Euclidean error."""
    return np.linalg.norm(pred - target, axis=-1).astype(np.float64)


def _load_pipeline(pipeline_dir: Path) -> dict | None:
    """Load per-pipeline predictions; return None when the run failed."""
    fp = pipeline_dir / "val_predictions.npz"
    if not fp.exists():
        return None
    npz = np.load(fp)
    return {
        "predictions": npz["predictions"].astype(np.float64),
        "targets": npz["targets"].astype(np.float64),
        "init_xyz": npz["init_xyz"].astype(np.float64),
        "residuals": npz["residuals"].astype(np.float64),
        "recording_keys": npz["recording_keys"],
    }


def _fit_weighted_late_fusion(
    pred_a: np.ndarray, pred_v: np.ndarray, target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis LS fit: target_d = w_a_d · pred_a_d + w_v_d · pred_v_d + b_d.

    Returns ``(w, b)`` with ``w.shape == (3, 2)``, ``b.shape == (3,)``.
    Per-axis (not joint over xyz) because acoustic SRP and vibration
    multilateration carry different per-axis confidences.
    """
    n = pred_a.shape[0]
    w = np.zeros((3, 2), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for d in range(3):
        A = np.stack([pred_a[:, d], pred_v[:, d], np.ones(n)], axis=1)
        sol, *_ = np.linalg.lstsq(A, target[:, d], rcond=None)
        w[d] = sol[:2]
        b[d] = sol[2]
    return w, b


def _apply_weighted_late_fusion(
    pred_a: np.ndarray, pred_v: np.ndarray, w: np.ndarray, b: np.ndarray,
) -> np.ndarray:
    """Apply per-axis fitted LF weights."""
    out = np.zeros_like(pred_a)
    for d in range(3):
        out[:, d] = w[d, 0] * pred_a[:, d] + w[d, 1] * pred_v[:, d] + b[d]
    return out


def _confidence_gated_late_fusion(
    pred_a: np.ndarray, pred_v: np.ndarray,
    residual_a: np.ndarray, residual_v: np.ndarray,
) -> np.ndarray:
    """Pick whichever pipeline's per-window residual norm is smaller."""
    norm_a = np.linalg.norm(residual_a, axis=-1)
    norm_v = np.linalg.norm(residual_v, axis=-1)
    use_a = (norm_a <= norm_v)[:, None]
    return np.where(use_a, pred_a, pred_v)


def _row(paradigm: str, label: str, errors: np.ndarray) -> _ParadigmRow:
    if errors.size == 0:
        return _ParadigmRow(paradigm, label, 0, float("nan"), float("nan"))
    lo, hi = _bootstrap_mae_ci(errors)
    return _ParadigmRow(
        paradigm=paradigm, row=label, n=int(errors.size),
        mae_m=float(errors.mean()), p95_m=float(np.percentile(errors, 95)),
        ci95_low_m=lo, ci95_high_m=hi,
    )


def _recording_level_holdout_split(
    recording_keys: np.ndarray, calibration_ratio: float = 0.5, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the val cohort by RECORDING (not window) into calibration vs test."""
    keys = np.asarray(recording_keys)
    unique = sorted(set(keys.tolist()))
    if len(unique) < 2:
        raise ValueError(
            f"need ≥ 2 val recordings to make a calibration/test split; "
            f"got {unique}"
        )
    rng = np.random.default_rng(seed)
    perm = list(unique)
    rng.shuffle(perm)
    n_calib = max(1, int(round(len(perm) * calibration_ratio)))
    n_calib = min(n_calib, len(perm) - 1)  # ≥ 1 in test
    calib_set = set(perm[:n_calib])
    test_set = set(perm[n_calib:])
    calib_mask = np.array([k in calib_set for k in keys], dtype=bool)
    test_mask = np.array([k in test_set for k in keys], dtype=bool)
    return calib_mask, test_mask


def _loro_folds(recording_keys: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Leave-one-recording-out folds for cross-validating the LF calibration.

    Yields ``(calib_mask, test_mask, held_out_recording_id)`` for every
    distinct recording in ``recording_keys``.  Each fold uses one recording
    as TEST and the rest as calibration.  The right protocol for the tiny
    V4 val cohort (3 recordings ⇒ 3 folds): cross-validation pins down the
    LF-weighted upper / lower bound under the worst- and best-case
    held-out recording, eliminating the cherry-picking risk of a single
    seeded split.
    """
    keys = np.asarray(recording_keys)
    unique = sorted(set(keys.tolist()))
    folds: list[tuple[np.ndarray, np.ndarray, str]] = []
    for held_out in unique:
        test_mask = (keys == held_out)
        calib_mask = ~test_mask
        folds.append((calib_mask, test_mask, held_out))
    return folds


def _in_sample_late_fusion(
    p_a: dict | None, p_v: dict | None
) -> tuple[list[_ParadigmRow], dict]:
    """Late-fusion rows fit *and* scored on the full val cohort.

    This is an upper bound — the weights see the windows they're evaluated on.
    Returns (rows_to_add, summary); both empty when either unimodal run is
    missing or the two were trained on different val splits.
    """
    if p_a is None or p_v is None:
        return [], {}
    if p_a["targets"].shape != p_v["targets"].shape or not np.allclose(
        p_a["targets"], p_v["targets"], atol=1e-6
    ):
        print("[rq3-3p] WARNING: val targets differ between V4-acoustic and "
              "V4-vibration; Late Fusion disabled (samples used different val splits)")
        return [], {}

    target = p_a["targets"]
    rows: list[_ParadigmRow] = []
    # LF uniform.
    lf_uniform = 0.5 * (p_a["predictions"] + p_v["predictions"])
    rows.append(_row("Late Fusion", "uniform_avg",
                     _per_window_errors(lf_uniform, target)))
    # LF weighted (per-axis LS, in-sample).
    w, b = _fit_weighted_late_fusion(p_a["predictions"], p_v["predictions"], target)
    lf_weighted = _apply_weighted_late_fusion(p_a["predictions"], p_v["predictions"], w, b)
    rows.append(_row("Late Fusion", "weighted_avg_in_sample",
                     _per_window_errors(lf_weighted, target)))
    # LF confidence-gated.
    lf_gated = _confidence_gated_late_fusion(
        p_a["predictions"], p_v["predictions"], p_a["residuals"], p_v["residuals"],
    )
    rows.append(_row("Late Fusion", "confidence_gated",
                     _per_window_errors(lf_gated, target)))
    summary = {
        "weighted_axis_weights": w.tolist(),
        "weighted_axis_biases": b.tolist(),
        "warn_in_sample_fit": (
            "weighted_avg_in_sample fits weights on the val cohort and "
            "applies them to the same cohort — best-case for LF.  Treat "
            "as an upper bound on LF performance."
        ),
    }
    return rows, summary


def _held_out_late_fusion(
    p_a: dict | None,
    p_v: dict | None,
    paradigms: dict,
    *,
    calibration_ratio: float,
    seed: int,
) -> tuple[list[_ParadigmRow], dict]:
    """Recording-level held-out calibration/test split for late fusion.

    Fits LF weights on a calibration subset of the val *recordings* and reports
    the defensible test-cohort MAE (the weights never see the test windows).
    Recording-level (not window-level) splitting prevents leakage from shared
    within-recording noise/geometry. Returns (rows_to_add, summary); both empty
    when the unimodal runs or per-window recording keys are unavailable.
    """
    holdout_summary: dict = {}
    holdout_rows: list[_ParadigmRow] = []
    if p_a is not None and p_v is not None and "recording_keys" in p_a:
        try:
            calib_mask, test_mask = _recording_level_holdout_split(
                p_a["recording_keys"],
                calibration_ratio=calibration_ratio,
                seed=seed,
            )
        except ValueError as e:
            print(f"[rq3-3p] held-out LF skipped: {e}")
            calib_mask = test_mask = None
        if calib_mask is not None:
            target = p_a["targets"]
            pa_c = p_a["predictions"][calib_mask]
            pv_c = p_v["predictions"][calib_mask]
            tgt_c = target[calib_mask]
            pa_t = p_a["predictions"][test_mask]
            pv_t = p_v["predictions"][test_mask]
            tgt_t = target[test_mask]

            # Calibration cohort summary so the gap to test is visible.
            holdout_summary["n_calibration_windows"] = int(calib_mask.sum())
            holdout_summary["n_test_windows"] = int(test_mask.sum())
            calib_keys = sorted(set(p_a["recording_keys"][calib_mask].tolist()))
            test_keys = sorted(set(p_a["recording_keys"][test_mask].tolist()))
            holdout_summary["calibration_recordings"] = calib_keys
            holdout_summary["test_recordings"] = test_keys
            print(f"[rq3-3p] held-out split: calibration n={calib_mask.sum()} "
                  f"({len(calib_keys)} recordings), test n={test_mask.sum()} "
                  f"({len(test_keys)} recordings)")
            print(f"  calibration recordings: {calib_keys}")
            print(f"  test recordings: {test_keys}")

            # Fit LF weights on calibration windows only.
            w_h, b_h = _fit_weighted_late_fusion(pa_c, pv_c, tgt_c)
            holdout_summary["weighted_axis_weights"] = w_h.tolist()
            holdout_summary["weighted_axis_biases"] = b_h.tolist()
            # Calibration-cohort MAE for transparency.
            lf_w_calib = _apply_weighted_late_fusion(pa_c, pv_c, w_h, b_h)
            holdout_rows.append(_row(
                "Late Fusion (held-out, calibration)", "weighted_avg",
                _per_window_errors(lf_w_calib, tgt_c),
            ))
            # Test-cohort MAE = the defensible headline number.
            lf_w_test = _apply_weighted_late_fusion(pa_t, pv_t, w_h, b_h)
            holdout_rows.append(_row(
                "Late Fusion (held-out, TEST)", "weighted_avg",
                _per_window_errors(lf_w_test, tgt_t),
            ))
            # Also report uniform + confidence-gated on the TEST subset so
            # all LF rows are apples-to-apples for the headline table.
            lf_u_test = 0.5 * (pa_t + pv_t)
            holdout_rows.append(_row(
                "Late Fusion (held-out, TEST)", "uniform_avg",
                _per_window_errors(lf_u_test, tgt_t),
            ))
            ra_t = p_a["residuals"][test_mask]
            rv_t = p_v["residuals"][test_mask]
            lf_g_test = _confidence_gated_late_fusion(pa_t, pv_t, ra_t, rv_t)
            holdout_rows.append(_row(
                "Late Fusion (held-out, TEST)", "confidence_gated",
                _per_window_errors(lf_g_test, tgt_t),
            ))
            # Also Intermediate Fusion + V4-acoustic on the same test subset
            # so the apples-to-apples Δ vs the LF-weighted held-out number is
            # immediate to read.
            if paradigms.get("fusion") is not None:
                pf_t = paradigms["fusion"]["predictions"][test_mask]
                holdout_rows.append(_row(
                    "Intermediate Fusion (held-out, TEST)", "V4-fusion",
                    _per_window_errors(pf_t, tgt_t),
                ))
            holdout_rows.append(_row(
                "Unimodal (held-out, TEST)", "V4-acoustic",
                _per_window_errors(pa_t, tgt_t),
            ))
            holdout_rows.append(_row(
                "Unimodal (held-out, TEST)", "V4-vibration",
                _per_window_errors(pv_t, tgt_t),
            ))
            # Paired-bootstrap LF-weighted (held-out, test) vs Intermediate
            # Fusion on the same test windows.  Same units, p<0.05 ⇒ Δ is
            # statistically real on the held-out cohort.
            if paradigms.get("fusion") is not None:
                err_lf_t = _per_window_errors(lf_w_test, tgt_t)
                err_fusion_t = _per_window_errors(pf_t, tgt_t)
                res_lf_fusion_t = paired_bootstrap_test(
                    err_fusion_t, err_lf_t, lower_is_better=True,
                    n_boot=1000, seed=42,
                )
                holdout_summary["paired_bootstrap_fusion_vs_lfweighted_holdout"] = {
                    "delta_mae_m": float(res_lf_fusion_t.delta_point),
                    "ci95_m": [float(res_lf_fusion_t.delta_ci_low),
                               float(res_lf_fusion_t.delta_ci_high)],
                    "p_value_two_sided": float(res_lf_fusion_t.p_value_two_sided),
                    "direction": res_lf_fusion_t.direction,
                }
    return holdout_rows, holdout_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-three-run", required=True,
                    help="Run dir produced by scripts/paradigms/run_v4_three_paradigms.py")
    ap.add_argument("--holdout-calibration-ratio", type=float, default=0.5,
                    help="Fraction of val RECORDINGS used to fit LF weights; "
                         "rest is held-out test for the defensible MAE.  "
                         "Default 0.5 (recording-level, seeded).")
    ap.add_argument("--holdout-seed", type=int, default=42,
                    help="Seed for the recording-level calibration/test split.")
    args = ap.parse_args()
    v4_run = Path(args.v4_three_run).resolve()
    if not v4_run.exists():
        raise SystemExit(f"{v4_run} does not exist")

    # Load per-paradigm predictions.
    paradigms = {
        "acoustic":   _load_pipeline(v4_run / "v4_acoustic"),
        "vibration":  _load_pipeline(v4_run / "v4_vibration"),
        "vibration_tdoa_only_legacy": _load_pipeline(v4_run / "v4_vibration_tdoa_only_legacy"),
        "fusion":     _load_pipeline(v4_run / "v4_fusion"),
    }
    missing = [k for k, v in paradigms.items() if v is None]
    if missing:
        print(f"[rq3-3p] WARNING: missing predictions for {missing} (training failed?)")

    # Build rows.
    rows: list[_ParadigmRow] = []
    for name, p in paradigms.items():
        if p is None:
            continue
        errors = _per_window_errors(p["predictions"], p["targets"])
        paradigm = "Unimodal" if name in ("acoustic", "vibration",
                                          "vibration_tdoa_only_legacy") else "Intermediate Fusion"
        rows.append(_row(paradigm, f"V4-{name}", errors))

    # Late Fusion — only when both unimodal pipelines exist + share window
    # geometry.  We assume the same precomputed sample list was used to
    # train every paradigm in the same orchestrator run, so the val splits
    # are identical (same seed, same recording-level split).  Sanity-check
    # by comparing target arrays element-wise.
    p_a = paradigms["acoustic"]
    p_v = paradigms["vibration"]
    lf_rows, late_fusion_summary = _in_sample_late_fusion(p_a, p_v)
    rows.extend(lf_rows)

    # -----------------------------------------------------------------
    # Held-out LF calibration/test split (recording-level)
    # -----------------------------------------------------------------
    # The in-sample LF-weighted row above is an upper bound.  This block
    # splits the V4 val cohort BY RECORDING into a calibration subset
    # (fit LF weights) and a test subset (report MAE), so the headline
    # LF-weighted number is defensible: the weights never see the test
    # windows.  Recording-level (not window-level) split prevents
    # window-level leakage from shared recording noise / geometry.
    holdout_rows, holdout_summary = _held_out_late_fusion(
        p_a, p_v, paradigms,
        calibration_ratio=args.holdout_calibration_ratio,
        seed=args.holdout_seed,
    )

    # -----------------------------------------------------------------
    # Leave-one-recording-out cross-validation (the DEFENSIBLE headline)
    # -----------------------------------------------------------------
    # With only 3 val recordings, a single seeded calibration/test split
    # is sensitive to which recording is held out (the corner position
    # `(-20, 0, 0)` is notoriously the hardest in every prior run).  LORO
    # eliminates the cherry-picking risk: every recording takes a turn
    # as the held-out test, the LF weights are re-fit on the other two
    # recordings per fold, and we report per-fold + aggregate MAE.  The
    # aggregate is the defensible Chapter 6 headline.
    loro_summary: dict = {}
    loro_rows: list[dict] = []
    if p_a is not None and p_v is not None and "recording_keys" in p_a:
        target_all = p_a["targets"]
        folds = _loro_folds(p_a["recording_keys"])
        print(f"\n[rq3-3p] LORO cross-validation: {len(folds)} folds "
              f"(one per val recording)")
        # Per-paradigm aggregate MAE collectors.
        per_paradigm_test_errors: dict[str, list[np.ndarray]] = {
            "V4-acoustic": [],
            "V4-vibration": [],
            "V4-fusion": [],
            "LF_uniform_avg": [],
            "LF_weighted_avg": [],
            "LF_confidence_gated": [],
        }
        for calib_mask, test_mask, held_out in folds:
            pa_c = p_a["predictions"][calib_mask]
            pv_c = p_v["predictions"][calib_mask]
            tgt_c = target_all[calib_mask]
            pa_t = p_a["predictions"][test_mask]
            pv_t = p_v["predictions"][test_mask]
            tgt_t = target_all[test_mask]
            # Fit LF weights on the (n - 1) calibration recordings.
            w_h, b_h = _fit_weighted_late_fusion(pa_c, pv_c, tgt_c)
            lf_w_test = _apply_weighted_late_fusion(pa_t, pv_t, w_h, b_h)
            lf_u_test = 0.5 * (pa_t + pv_t)
            ra_t = p_a["residuals"][test_mask]
            rv_t = p_v["residuals"][test_mask]
            lf_g_test = _confidence_gated_late_fusion(pa_t, pv_t, ra_t, rv_t)
            fold = {
                "held_out_recording": held_out,
                "n_test_windows": int(test_mask.sum()),
                "n_calibration_windows": int(calib_mask.sum()),
                "V4-acoustic_mae_m": float(_per_window_errors(pa_t, tgt_t).mean()),
                "V4-vibration_mae_m": float(_per_window_errors(pv_t, tgt_t).mean()),
                "LF_uniform_avg_mae_m": float(_per_window_errors(lf_u_test, tgt_t).mean()),
                "LF_weighted_avg_mae_m": float(_per_window_errors(lf_w_test, tgt_t).mean()),
                "LF_confidence_gated_mae_m": float(_per_window_errors(lf_g_test, tgt_t).mean()),
            }
            per_paradigm_test_errors["V4-acoustic"].append(_per_window_errors(pa_t, tgt_t))
            per_paradigm_test_errors["V4-vibration"].append(_per_window_errors(pv_t, tgt_t))
            per_paradigm_test_errors["LF_uniform_avg"].append(_per_window_errors(lf_u_test, tgt_t))
            per_paradigm_test_errors["LF_weighted_avg"].append(_per_window_errors(lf_w_test, tgt_t))
            per_paradigm_test_errors["LF_confidence_gated"].append(_per_window_errors(lf_g_test, tgt_t))
            if paradigms.get("fusion") is not None:
                pf_t = paradigms["fusion"]["predictions"][test_mask]
                fold["V4-fusion_mae_m"] = float(_per_window_errors(pf_t, tgt_t).mean())
                per_paradigm_test_errors["V4-fusion"].append(_per_window_errors(pf_t, tgt_t))
            loro_rows.append(fold)
            print(f"  fold held_out={held_out!r}: "
                  f"V4-fusion={fold.get('V4-fusion_mae_m', float('nan')):.3f}, "
                  f"V4-acoustic={fold['V4-acoustic_mae_m']:.3f}, "
                  f"LF_gated={fold['LF_confidence_gated_mae_m']:.3f}, "
                  f"LF_weighted={fold['LF_weighted_avg_mae_m']:.3f}, "
                  f"n_test={fold['n_test_windows']}")

        # Aggregate: macro-mean (per-fold means, then mean across folds —
        # gives every recording equal weight regardless of window count)
        # AND micro-mean (concatenate all per-window errors then mean —
        # reflects window-level average; biased toward recordings with more
        # windows).  Report both.
        agg: dict = {}
        for name, fold_errors in per_paradigm_test_errors.items():
            if not fold_errors:
                continue
            per_fold_means = np.array([float(e.mean()) for e in fold_errors])
            all_errors = np.concatenate(fold_errors)
            lo, hi = _bootstrap_mae_ci(all_errors)
            agg[name] = {
                "macro_mean_mae_m": float(per_fold_means.mean()),
                "macro_std_mae_m": float(per_fold_means.std()),
                "micro_mean_mae_m": float(all_errors.mean()),
                "micro_ci95_m": [lo, hi],
                "n_total_test_windows": int(all_errors.size),
                "per_fold_mae_m": per_fold_means.tolist(),
            }
        loro_summary["aggregate"] = agg
        loro_summary["per_fold"] = loro_rows

    # Paired-bootstrap significance: V4-fusion vs LF-weighted; V4-fusion
    # vs the better unimodal.  All comparisons on per-window 3-D errors.
    sig: dict = {}
    if p_a is not None and paradigms.get("fusion") is not None:
        target = paradigms["fusion"]["targets"]
        if (target.shape == p_a["targets"].shape
                and np.allclose(target, p_a["targets"], atol=1e-6)):
            err_fusion = _per_window_errors(paradigms["fusion"]["predictions"], target)
            err_acoustic = _per_window_errors(p_a["predictions"], target)
            res_fa = paired_bootstrap_test(err_fusion, err_acoustic, lower_is_better=True,
                                           n_boot=1000, seed=42)
            sig["fusion_vs_acoustic_mae_delta_m"] = float(res_fa.delta_point)
            sig["fusion_vs_acoustic_ci95_m"] = [float(res_fa.delta_ci_low), float(res_fa.delta_ci_high)]
            sig["fusion_vs_acoustic_p"] = float(res_fa.p_value_two_sided)
            sig["fusion_vs_acoustic_direction"] = res_fa.direction
            if p_v is not None and target.shape == p_v["targets"].shape and np.allclose(target, p_v["targets"], atol=1e-6):
                err_vib = _per_window_errors(p_v["predictions"], target)
                res_fv = paired_bootstrap_test(err_fusion, err_vib, lower_is_better=True,
                                               n_boot=1000, seed=42)
                sig["fusion_vs_vibration_mae_delta_m"] = float(res_fv.delta_point)
                sig["fusion_vs_vibration_ci95_m"] = [float(res_fv.delta_ci_low), float(res_fv.delta_ci_high)]
                sig["fusion_vs_vibration_p"] = float(res_fv.p_value_two_sided)
                # LF-weighted vs fusion.
                w, b = _fit_weighted_late_fusion(p_a["predictions"], p_v["predictions"], target)
                lf_weighted = _apply_weighted_late_fusion(p_a["predictions"], p_v["predictions"], w, b)
                err_lf = _per_window_errors(lf_weighted, target)
                res_lf = paired_bootstrap_test(err_fusion, err_lf, lower_is_better=True,
                                                n_boot=1000, seed=42)
                sig["fusion_vs_lfweighted_mae_delta_m"] = float(res_lf.delta_point)
                sig["fusion_vs_lfweighted_ci95_m"] = [float(res_lf.delta_ci_low), float(res_lf.delta_ci_high)]
                sig["fusion_vs_lfweighted_p"] = float(res_lf.p_value_two_sided)
                sig["fusion_vs_lfweighted_direction"] = res_lf.direction

    # Classical baselines.
    classical_rows: list[dict] = []
    srp_path = v4_run / "classical" / "v0_srp_phat.json"
    if srp_path.exists():
        srp = json.loads(srp_path.read_text(encoding="utf-8"))
        for ds, s in srp.items():
            if isinstance(s, dict) and "mean_error_m" in s:
                classical_rows.append({"paradigm": "V0 classical", "row": f"SRP-PHAT ({ds})",
                                        "mae_m": float(s["mean_error_m"]),
                                        "n_recordings": int(s.get("n_recordings", 0))})
    multilat_path = v4_run / "classical" / "v0_multilat.json"
    if multilat_path.exists():
        ml = json.loads(multilat_path.read_text(encoding="utf-8"))
        for ds, s in ml.items():
            classical_rows.append({"paradigm": "V0 classical", "row": f"multilat ({ds})",
                                    "mae_m": float(s.get("mean_error_m", float("nan"))),
                                    "n_recordings": int(s.get("n_successful", 0))})

    out_json = v4_run / "rq3_paradigm_comparison.json"
    out_md = v4_run / "rq3_paradigm_comparison.md"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump({
            "v4_run": str(v4_run.relative_to(REPO)),
            "rows": [asdict(r) for r in rows],
            "holdout_rows": [asdict(r) for r in holdout_rows],
            "holdout_summary": holdout_summary,
            "loro_summary": loro_summary,
            "classical_rows": classical_rows,
            "late_fusion": late_fusion_summary,
            "significance": sig,
            "method": "rq3_three_paradigm_comparison_2026_05_17_loro",
        }, fh, indent=2)

    lines: list[str] = []
    lines.append("# RQ3 — Unimodal × 2 + Late Fusion × 3 + Intermediate Fusion + V0 classical\n")
    lines.append("## Per-paradigm 3-D Euclidean error (validation cohort)\n")
    lines.append("| Paradigm | Row | n | MAE (m) | p95 (m) | 95 % CI |")
    lines.append("|---|---|---:|---:|---:|---|")
    for r in rows:
        ci = ""
        if r.ci95_low_m is not None and not np.isnan(r.ci95_low_m):
            ci = f"[{r.ci95_low_m:.3f}, {r.ci95_high_m:.3f}]"
        lines.append(f"| {r.paradigm} | {r.row} | {r.n} | {r.mae_m:.3f} | {r.p95_m:.3f} | {ci} |")
    if classical_rows:
        lines.append("\n### V0 classical references\n")
        lines.append("| Row | MAE (m) | n recordings |")
        lines.append("|---|---:|---:|")
        for c in classical_rows:
            lines.append(f"| {c['row']} | {c['mae_m']:.3f} | {c['n_recordings']} |")
    if sig:
        lines.append("\n## Paired-bootstrap significance (V4-fusion vs the rest)\n")
        for key in ("fusion_vs_acoustic", "fusion_vs_vibration", "fusion_vs_lfweighted"):
            dk = f"{key}_mae_delta_m"
            ck = f"{key}_ci95_m"
            pk = f"{key}_p"
            if dk not in sig:
                continue
            lines.append(
                f"- {key.replace('_', ' ')}: Δ_MAE = {sig[dk]*1000:+.2f} mm "
                f"[{sig[ck][0]*1000:+.2f}, {sig[ck][1]*1000:+.2f}], p = {sig[pk]:.4f}"
            )
    if late_fusion_summary:
        lines.append("\n## Late-Fusion weighted-axis weights (in-sample, upper bound)\n")
        lines.append("(NOTE: in-sample fit; treat as best-case upper bound for the LF-weighted row.)\n")
        if "weighted_axis_weights" in late_fusion_summary:
            w = late_fusion_summary["weighted_axis_weights"]
            b = late_fusion_summary["weighted_axis_biases"]
            lines.append(f"- x: w_a = {w[0][0]:+.3f}, w_v = {w[0][1]:+.3f}, b = {b[0]:+.3f}")
            lines.append(f"- y: w_a = {w[1][0]:+.3f}, w_v = {w[1][1]:+.3f}, b = {b[1]:+.3f}")
            lines.append(f"- z: w_a = {w[2][0]:+.3f}, w_v = {w[2][1]:+.3f}, b = {b[2]:+.3f}")

    if holdout_rows:
        lines.append("\n## Held-out LF calibration / test split (DEFENSIBLE HEADLINE)\n")
        lines.append(
            "LF weights fit on the **calibration** subset of the V4 val cohort "
            "(recording-level split, so no window-level leakage) and applied "
            "to the held-out **TEST** subset.  The TEST rows are the "
            "defensible RQ3 headline for Chapter 6 — the LF combiner never "
            "saw the test recordings during weight fitting.\n"
        )
        lines.append(
            f"- Calibration cohort: n_windows = "
            f"{holdout_summary.get('n_calibration_windows', 0)}, "
            f"recordings = {holdout_summary.get('calibration_recordings', [])}"
        )
        lines.append(
            f"- TEST cohort: n_windows = "
            f"{holdout_summary.get('n_test_windows', 0)}, "
            f"recordings = {holdout_summary.get('test_recordings', [])}\n"
        )
        lines.append("| Paradigm | Row | n | MAE (m) | p95 (m) | 95 % CI |")
        lines.append("|---|---|---:|---:|---:|---|")
        for r in holdout_rows:
            ci = ""
            if r.ci95_low_m is not None and not np.isnan(r.ci95_low_m):
                ci = f"[{r.ci95_low_m:.3f}, {r.ci95_high_m:.3f}]"
            lines.append(f"| {r.paradigm} | {r.row} | {r.n} | {r.mae_m:.3f} | {r.p95_m:.3f} | {ci} |")
        if "weighted_axis_weights" in holdout_summary:
            w_h = holdout_summary["weighted_axis_weights"]
            b_h = holdout_summary["weighted_axis_biases"]
            lines.append("\n### LF-weighted axis weights (fit on calibration only)\n")
            lines.append(f"- x: w_a = {w_h[0][0]:+.3f}, w_v = {w_h[0][1]:+.3f}, b = {b_h[0]:+.3f}")
            lines.append(f"- y: w_a = {w_h[1][0]:+.3f}, w_v = {w_h[1][1]:+.3f}, b = {b_h[1]:+.3f}")
            lines.append(f"- z: w_a = {w_h[2][0]:+.3f}, w_v = {w_h[2][1]:+.3f}, b = {b_h[2]:+.3f}")
        bs = holdout_summary.get("paired_bootstrap_fusion_vs_lfweighted_holdout")
        if bs is not None:
            lines.append("\n### Paired-bootstrap on TEST cohort — V4-fusion vs LF-weighted\n")
            lines.append(
                f"- Δ_MAE (fusion − LF-weighted) = {bs['delta_mae_m']*1000:+.2f} mm "
                f"[{bs['ci95_m'][0]*1000:+.2f}, {bs['ci95_m'][1]*1000:+.2f}], "
                f"p = {bs['p_value_two_sided']:.4f}, direction = {bs['direction']}"
            )

    if loro_summary:
        lines.append("\n## LORO cross-validation — DEFENSIBLE CHAPTER-6 HEADLINE\n")
        lines.append(
            "Leave-one-recording-out cross-validation across the V4 val cohort.  "
            "Each fold uses one recording as TEST and the other N-1 as "
            "calibration; LF-weighted axis weights are re-fit per fold on the "
            "calibration recordings only.  Per-paradigm aggregate MAE is "
            "reported two ways: **macro-mean** (mean of per-fold MAEs — every "
            "recording gets equal weight regardless of window count, the right "
            "metric for generalisation across positions) and **micro-mean** "
            "(window-level mean across all folds with its 95 % bootstrap CI).\n"
        )
        agg = loro_summary.get("aggregate", {})
        if agg:
            lines.append("| Paradigm | macro-mean MAE (m) | macro std | micro-mean MAE (m) | micro 95 % CI | n_total_test |")
            lines.append("|---|---:|---:|---:|---|---:|")
            order = [
                "V4-fusion", "V4-acoustic", "V4-vibration",
                "LF_confidence_gated", "LF_uniform_avg", "LF_weighted_avg",
            ]
            for name in order:
                if name not in agg:
                    continue
                a = agg[name]
                ci = f"[{a['micro_ci95_m'][0]:.3f}, {a['micro_ci95_m'][1]:.3f}]"
                lines.append(
                    f"| {name} | {a['macro_mean_mae_m']:.3f} | "
                    f"{a['macro_std_mae_m']:.3f} | {a['micro_mean_mae_m']:.3f} | "
                    f"{ci} | {a['n_total_test_windows']} |"
                )
        per_fold = loro_summary.get("per_fold", [])
        if per_fold:
            lines.append("\n### Per-fold breakdown (which recording was held out)\n")
            lines.append("| held-out recording | n_test | V4-fusion | V4-acoustic | V4-vibration | LF gated | LF uniform | LF weighted |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for f in per_fold:
                lines.append(
                    f"| {f['held_out_recording']} | {f['n_test_windows']} | "
                    f"{f.get('V4-fusion_mae_m', float('nan')):.3f} | "
                    f"{f['V4-acoustic_mae_m']:.3f} | "
                    f"{f['V4-vibration_mae_m']:.3f} | "
                    f"{f['LF_confidence_gated_mae_m']:.3f} | "
                    f"{f['LF_uniform_avg_mae_m']:.3f} | "
                    f"{f['LF_weighted_avg_mae_m']:.3f} |"
                )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[rq3-3p] Wrote {out_json.relative_to(REPO)}")
    print(f"[rq3-3p] Wrote {out_md.relative_to(REPO)}")


if __name__ == "__main__":
    main()
