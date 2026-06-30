"""V0 reference-baseline stage for the full run.

These are the opt-in ``--v0-baselines`` reference numbers: the supervised
LightGBM mode classifier and unsupervised K-means floor (RQ1), the LSTM-AE and
Khamaisi anomaly trio (RQ2), and classical SRP-PHAT / accel-multilateration
localization (RQ3). They sit beside the learned pipeline as the prior-work
references it is compared against, so they live in their own module rather than
inside the orchestrator's ``main``.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from ..anomaly_baselines import (
    ALL_MODELS,
    MODALITIES,
    SRPConfig,
    V0Config,
    V0ModeConfig,
    cluster_mode_floor,
    evaluate_srp_phat,
    evaluate_v0_anomaly,
    summarise,
    train_v0_lstm_ae,
    train_v0_mode_lgbm,
)
from ..localization.multilateration import accel_tdoa_multilateration_v0


def run_v0(loaders: list, log: Callable[[str], None], anom_loaders: list | None = None) -> dict:
    out: dict = {}
    anom_loaders = anom_loaders if anom_loaders is not None else loaders
    for L in loaders:
        ds_name = L.spec.id
        if ds_name in ("d1", "d2"):
            try:
                t0 = time.time()
                r = train_v0_mode_lgbm(L, V0ModeConfig())
                log(f"  V0 LGBM mode ({ds_name}) {time.time()-t0:.0f}s — "
                    f"val macro-F1={r.val_macro_f1:.3f}")
                out[f"v0_lgbm_{ds_name}"] = {
                    "val_macro_f1": float(r.val_macro_f1),
                    "val_per_class_f1": {str(k): float(v) for k, v in r.val_per_class_f1.items()},
                    "n_train_recordings": len(r.train_recording_ids),
                    "n_val_recordings": len(r.val_recording_ids),
                }
            except Exception as e:
                log(f"  V0 LGBM mode ({ds_name}) skipped: {type(e).__name__}: {e}")
                out[f"v0_lgbm_{ds_name}"] = {"skipped": f"{type(e).__name__}: {e}"}
        try:
            t0 = time.time()
            ae = train_v0_lstm_ae(L, V0Config())
            log(f"  V0 LSTM-AE ({ds_name}) {time.time()-t0:.0f}s — "
                f"val recon MSE={ae.val_loss_history[-1]:.4f}")
            out[f"v0_lstm_ae_{ds_name}"] = {
                "val_loss_final": float(ae.val_loss_history[-1]),
                "n_train_recordings": len(ae.healthy_train_recordings),
                "n_val_recordings": len(ae.healthy_val_recordings),
            }
        except Exception as e:
            log(f"  V0 LSTM-AE ({ds_name}) skipped: {type(e).__name__}: {e}")
            out[f"v0_lstm_ae_{ds_name}"] = {"skipped": f"{type(e).__name__}: {e}"}
        if ds_name in ("d2", "d3", "d4"):
            try:
                recs = evaluate_srp_phat(L, SRPConfig())
                s = summarise(recs)
                log(f"  V0 SRP-PHAT ({ds_name}): {s.get('n_recordings', 0)} recordings, "
                    f"mean MAE={s.get('mean_error_m', float('nan')):.3f} m")
                out[f"v0_srp_phat_{ds_name}"] = s
            except Exception as e:
                log(f"  V0 SRP-PHAT ({ds_name}) skipped: {type(e).__name__}: {e}")
                out[f"v0_srp_phat_{ds_name}"] = {"skipped": f"{type(e).__name__}: {e}"}

    # RQ1 context floor — unsupervised K-means on hand-engineered features,
    # scored against the mode label (NMI / ARI / purity).  The lower bound the
    # label-free encoder must beat; the LightGBM rows above are the supervised
    # upper bound it approaches from below.
    try:
        floor = cluster_mode_floor(loaders, V0ModeConfig())
        log(f"  V0 RQ1 mode-floor (K-means/handcrafted): NMI={floor.nmi:.3f} "
            f"ARI={floor.ari:.3f} purity={floor.purity:.3f} "
            f"({floor.n_windows} win / {floor.n_recordings} rec)")
        out["v0_mode_floor"] = {
            "nmi": floor.nmi, "ari": floor.ari, "purity": floor.purity,
            "n_windows": floor.n_windows, "n_recordings": floor.n_recordings,
            "label_set": list(floor.label_set), "n_clusters": floor.n_clusters,
        }
    except Exception as e:
        log(f"  V0 RQ1 mode-floor skipped: {type(e).__name__}: {e}")
        out["v0_mode_floor"] = {"skipped": f"{type(e).__name__}: {e}"}

    # RQ2 anomaly reference — the full Khamaisi trio + KDE, pooled across all
    # campaigns (like the V3 training cohort) and scored on the same protocol the
    # conditional head reports: within-campaign healthy-vs-anomaly ROC-AUC plus
    # the in-distribution-vs-domain-shift false-positive-rate contrast.  This is
    # the credible prior-work reference V3 must improve on for RQ2.
    out["v0_anomaly_rq2"] = {}
    for modality in MODALITIES:
        for model in ALL_MODELS:
            cell = f"{modality}/{model}"
            try:
                t0 = time.time()
                res = evaluate_v0_anomaly(anom_loaders, model, modality, V0Config())
                out["v0_anomaly_rq2"][cell] = res.to_dict()
                log(f"  V0 RQ2 {cell} {time.time()-t0:.0f}s — "
                    f"ROC-AUC={res.roc_auc:.3f} FPR(in-dist={res.fpr_in_distribution:.3f} "
                    f"shift={res.fpr_domain_shift:.3f})")
            except Exception as e:
                log(f"  V0 RQ2 {cell} skipped: {type(e).__name__}: {e}")
                out["v0_anomaly_rq2"][cell] = {"skipped": f"{type(e).__name__}: {e}"}
    return out


def run_v0_multilateration(
    loaders: list, overrides: dict, log: Callable[[str], None]
) -> dict:
    out: dict = {}
    for L in loaders:
        ds_name = L.spec.id
        if ds_name not in ("d2", "d3", "d4"):
            continue
        per_rec: list[dict] = []
        for s in L.list_segments():
            if not s.is_anomaly or s.spatial_label is None:
                continue
            try:
                if s.segment.accel_data.shape[0] < 4:
                    per_rec.append({"recording_id": s.recording_id, "skipped": "n_accel < 4"})
                    continue
                pos, residual = accel_tdoa_multilateration_v0(
                    s.segment.accel_data, s.vib_positions,
                    fs=float(s.segment.accel_sample_rate),
                )
                target = overrides.get(s.recording_id) if ds_name == "d3" else s.spatial_label
                if target is None:
                    per_rec.append({"recording_id": s.recording_id, "skipped": "no spatial label"})
                    continue
                err = float(np.linalg.norm(pos - np.asarray(target, dtype=np.float64)))
                per_rec.append({
                    "recording_id": s.recording_id,
                    "target": list(map(float, target)),
                    "pred": list(map(float, pos)),
                    "residual": float(residual),
                    "error_m": err,
                })
            except Exception as e:
                per_rec.append({"recording_id": s.recording_id, "error": f"{type(e).__name__}: {e}"})
        errs = [r["error_m"] for r in per_rec if "error_m" in r]
        out[ds_name] = {
            "n_recordings": len(per_rec),
            "n_successful": len(errs),
            "mean_error_m": float(np.mean(errs)) if errs else float("nan"),
            "median_error_m": float(np.median(errs)) if errs else float("nan"),
            "p95_error_m": float(np.percentile(errs, 95)) if errs else float("nan"),
            "per_recording": per_rec,
        }
        log(f"  V0 multilat ({ds_name}): {len(errs)}/{len(per_rec)} resolved, "
            f"mean MAE={out[ds_name]['mean_error_m']:.3f} m")
    return out
