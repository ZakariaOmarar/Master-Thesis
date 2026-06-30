"""Fit + evaluate + persist the raw-waveform impulse+spectral anomaly detector.

Encoder-free, label-free: reads raw mic/accel waveforms, extracts impulse +
spectral window features, fits the healthy Mahalanobis (per modality, sum-fused)
on the FIT datasets, evaluates every TEST dataset (recording-level ROC/PR and
window-level vs healthy-referenced peak labels), and persists the model + a JSON.

Run:
    python -m scripts.fit_raw_impulse_detector
    python -m scripts.fit_raw_impulse_detector --fit-ds d2 d3 d4 --test-ds d2 d3 d4 d5
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.anomaly.raw_impulse_detector import (
    RawImpulseDetector,
    window_features,
)
from src.modeling.anomaly.weak_labels import derive_knock_events, window_overlaps_any
from src.modeling.eval.rq2_three_paradigm_eval import _loader


def _collect(loader, is_anom: bool, win_s: float, stride_s: float):
    Xm, Xa, env, peak = [], [], [], []
    for s in loader.list_segments():
        if s.is_anomaly != is_anom:
            continue
        ds = s.segment
        mic = getattr(ds, "mic_data", None)
        if mic is None or not mic.size:
            continue
        fs_m = float(ds.mic_sample_rate)
        acc = getattr(ds, "accel_data", None)
        fs_a = float(getattr(ds, "accel_sample_rate", 0) or 1.0)
        intervals = derive_knock_events(s, max_events=300, noise_floor_mult=3.0) if is_anom else []
        wm = np.sqrt(np.mean(mic.astype(np.float64) ** 2, axis=0))
        wa = (np.sqrt(np.mean(acc.astype(np.float64) ** 2, axis=0))
              if acc is not None and acc.size else None)
        T = wm.size; step = max(1, int(stride_s * fs_m)); wlen = int(win_s * fs_m)
        for i0 in range(0, max(1, T - wlen + 1), step):
            t0, t1 = i0 / fs_m, (i0 + wlen) / fs_m
            seg = wm[i0:i0 + wlen]
            Xm.append(window_features(seg, fs_m))
            peak.append(float(np.abs(seg).max()) if seg.size else 0.0)
            Xa.append(window_features(wa[int(t0 * fs_a):int(t1 * fs_a)], fs_a)
                      if wa is not None else np.zeros(16))
            env.append(1 if (is_anom and window_overlaps_any(t0, t1, intervals)) else 0)
    return (np.array(Xm), np.array(Xa), np.array(env), np.array(peak))


def main() -> int:
    from sklearn.metrics import average_precision_score, roc_auc_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-ds", nargs="*", default=["d2", "d3", "d4"])
    ap.add_argument("--test-ds", nargs="*", default=["d2", "d3", "d4", "d5"])
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--out", default="results/raw_impulse_detector.pkl")
    args = ap.parse_args()

    all_ds = sorted(set(args.fit_ds) | set(args.test_ds))
    H, A = {}, {}
    print("collecting raw impulse+spectral features ...", flush=True)
    for dn in all_ds:
        L = _loader(dn)
        hm, ha, _, hpk = _collect(L, False, 1.0, 0.5)
        am, aa, env, apk = _collect(L, True, 1.0, 0.5)
        floor = float(np.percentile(hpk, 99.5)) if hpk.size else float("inf")
        H[dn] = {"mic": hm, "accel": ha}
        A[dn] = {"mic": am, "accel": aa, "env": env, "ref": (apk > floor).astype(int)}
        print(f"  {dn}: healthy_win={hm.shape[0]} anom_win={am.shape[0]} "
              f"href+={A[dn]['ref'].mean():.2f}", flush=True)

    det = RawImpulseDetector(target_fpr=args.target_fpr).fit(
        {"mic": np.concatenate([H[d]["mic"] for d in args.fit_ds]),
         "accel": np.concatenate([H[d]["accel"] for d in args.fit_ds])})

    res = {"fit_ds": args.fit_ds, "target_fpr": args.target_fpr, "per_dataset": {}}
    hf = det.fused_score({"mic": np.concatenate([H[d]["mic"] for d in args.fit_ds]),
                          "accel": np.concatenate([H[d]["accel"] for d in args.fit_ds])})
    res["healthy_fpr"] = float((hf > det.threshold).mean())
    print(f"\nfit on {args.fit_ds} healthy | healthy FPR @ threshold = {res['healthy_fpr']:.3f}\n")
    print(f"{'ds':<5} {'reclvl_ROC':>10} {'reclvl_PR':>10} {'href_ROC':>9} "
          f"{'flag@thr':>9} {'heldout':>8}")
    for dn in args.test_ds:
        zh = det.fused_score(H[dn]); za = det.fused_score(A[dn])
        y = np.concatenate([np.zeros(zh.size), np.ones(za.size)])
        s = np.concatenate([zh, za])
        rec_roc, rec_pr = roc_auc_score(y, s), average_precision_score(y, s)
        yr = A[dn]["ref"]
        hr = roc_auc_score(yr, za) if 0 < yr.sum() < yr.size else float("nan")
        flag = float((za > det.threshold).mean())
        res["per_dataset"][dn] = {
            "reclvl_roc": float(rec_roc), "reclvl_pr": float(rec_pr),
            "href_roc": float(hr), "flag_at_thr": flag,
            "healthy_fpr": float((zh > det.threshold).mean()),
            "held_out": dn not in args.fit_ds, "n_anom_win": int(za.size)}
        print(f"{dn:<5} {rec_roc:>10.3f} {rec_pr:>10.3f} "
              f"{hr if not np.isnan(hr) else 0:>9.3f} {flag:>9.3f} "
              f"{'YES' if dn not in args.fit_ds else '':>8}")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(det, fh)
    with out.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nsaved -> {out.relative_to(REPO)}\nsaved -> {out.with_suffix('.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
