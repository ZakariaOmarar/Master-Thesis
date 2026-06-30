"""Probe: raw-waveform IMPULSE + SPECTRAL detector, window- and recording-level.

Findings driving this version:
  * D2/D4 anomalies are IMPULSIVE (amplitude transients) -> impulse features
    (crest/kurtosis/spectral-kurtosis) separate them; window-level peak labels
    (healthy-referenced) are the right ground truth.
  * D3 anomalies are SPECTRAL, not impulsive (0 windows clear the healthy peak
    floor, yet mel-AUC was 0.99) -> need SPECTRAL-shape features; D3 is
    continuous so the recording-level label (all anomaly windows positive) is
    the right ground truth.
So the detector uses both feature families.  Fit on healthy only (unsupervised);
the detector is fit on d2+d3+d4 healthy and D5 is HELD OUT to verify the theory
on an easier campaign.

Metrics per dataset:
  * window-level vs healthy-referenced peak label (impulsive regimes);
  * recording-level ROC: that dataset's healthy windows vs its anomaly windows
    (regime-matched, robust to regime offset; right for continuous regimes);
  * flag@p95 (detector flag-rate at a healthy-calibrated threshold) and the
    healthy FPR.

Run:  python -m scripts.diagnostics.probe_impulse_detector
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.anomaly.weak_labels import derive_knock_events, window_overlaps_any
from src.modeling.eval.rq2_three_paradigm_eval import _loader

WIN_S, STRIDE_S = 1.0, 0.5
FIT_DS = ("d2", "d3", "d4")     # healthy pool the detector is fit on
TEST_DS = ("d2", "d3", "d4", "d5")  # d5 = held-out verification


def _impulse_feats(w: np.ndarray, fs: float) -> list[float]:
    aw = np.abs(w)
    rms = np.sqrt(np.mean(w * w)) + 1e-12
    peak = float(aw.max()); mean_abs = float(aw.mean()) + 1e-12
    mu, sd = w.mean(), w.std() + 1e-12
    sk = 0.0
    try:
        from scipy.signal import stft
        nper = min(256, max(16, w.size // 8))
        _, _, Z = stft(w, fs=fs, nperseg=nper, noverlap=nper // 2)
        mag = np.abs(Z)
        if mag.shape[1] >= 4:
            m = mag.mean(1, keepdims=True); s = mag.std(1, keepdims=True) + 1e-12
            sk = float(np.nanmax((((mag - m) / s) ** 4).mean(1) - 3.0))
    except Exception:
        sk = 0.0
    thr = 3.0 * (np.median(aw) + 1e-12)
    kcount = float(np.sum(np.diff((aw > thr).astype(int)) == 1))
    return [peak / rms, peak / mean_abs,
            peak / (np.mean(np.sqrt(aw)) ** 2 + 1e-12), rms / mean_abs,
            float(np.mean(((w - mu) / sd) ** 4) - 3.0), sk, kcount,
            peak / (np.median(aw) + 1e-12)]


def _spectral_feats(w: np.ndarray, fs: float) -> list[float]:
    n = w.size
    W = np.abs(np.fft.rfft(w * np.hanning(n))) ** 2 + 1e-12
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    Wn = W / W.sum()
    centroid = float((fr * Wn).sum())
    spread = float(np.sqrt(((fr - centroid) ** 2 * Wn).sum()))
    flatness = float(np.exp(np.mean(np.log(W))) / np.mean(W))
    entropy = float(-(Wn * np.log(Wn)).sum())
    cs = np.cumsum(W); rolloff = float(fr[np.searchsorted(cs, 0.85 * cs[-1])])
    th = fr.max() / 3.0 + 1e-9
    low = float(W[fr < th].sum() / W.sum())
    mid = float(W[(fr >= th) & (fr < 2 * th)].sum() / W.sum())
    high = float(W[fr >= 2 * th].sum() / W.sum())
    return [centroid, spread, flatness, entropy, rolloff, low, mid, high]


def _winfeats(w: np.ndarray, fs: float) -> np.ndarray:
    if w.size < 16 or not np.any(w):
        return np.zeros(16)
    w = np.asarray(w, dtype=np.float64)
    return np.array(_impulse_feats(w, fs) + _spectral_feats(w, fs))


def _collect(loader, is_anom: bool):
    """Per-window: feats_mic, feats_acc, env_label, mic_peak."""
    Xm, Xa, yenv, peak = [], [], [], []
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
        T = wm.size; step = max(1, int(STRIDE_S * fs_m)); wlen = int(WIN_S * fs_m)
        for i0 in range(0, max(1, T - wlen + 1), step):
            t0, t1 = i0 / fs_m, (i0 + wlen) / fs_m
            seg_m = wm[i0:i0 + wlen]
            Xm.append(_winfeats(seg_m, fs_m))
            peak.append(float(np.abs(seg_m).max()) if seg_m.size else 0.0)
            if wa is not None:
                Xa.append(_winfeats(wa[int(t0 * fs_a):int(t1 * fs_a)], fs_a))
            else:
                Xa.append(np.zeros(16))
            yenv.append(1 if (is_anom and window_overlaps_any(t0, t1, intervals)) else 0)
    return np.array(Xm), np.array(Xa), np.array(yenv), np.array(peak)


def _maha_fit(Xh):
    mu = Xh.mean(0); cov = np.cov(Xh, rowvar=False) + 1e-6 * np.eye(Xh.shape[1])
    inv = np.linalg.inv(cov); s = np.einsum("ij,jk,ik->i", Xh - mu, inv, Xh - mu)
    return mu, inv, s.mean(), s.std() + 1e-8


def _maha_z(X, p):
    mu, inv, m, sd = p
    return (np.einsum("ij,jk,ik->i", X - mu, inv, X - mu) - m) / sd


def main() -> int:
    from sklearn.metrics import average_precision_score, roc_auc_score
    H, A = {}, {}
    print("collecting raw impulse+spectral features ...", flush=True)
    for dn in TEST_DS:
        L = _loader(dn)
        hm, ha, _, hpk = _collect(L, False)
        am, aa, aenv, apk = _collect(L, True)
        floor = float(np.percentile(hpk, 99.5)) if hpk.size else float("inf")
        H[dn] = (hm, ha)
        A[dn] = {"m": am, "a": aa, "env": aenv,
                 "ref": (apk > floor).astype(int)}
        print(f"  {dn}: healthy_win={hm.shape[0]} anom_win={am.shape[0]} | "
              f"env+={aenv.mean():.2f} healthyref+={A[dn]['ref'].mean():.2f}", flush=True)

    Hm = np.concatenate([H[d][0] for d in FIT_DS])
    Ha = np.concatenate([H[d][1] for d in FIT_DS])
    mm, ms = Hm.mean(0), Hm.std(0) + 1e-8
    aM, aS = Ha.mean(0), Ha.std(0) + 1e-8
    pm = _maha_fit((Hm - mm) / ms); pa = _maha_fit((Ha - aM) / aS)
    thr95 = float(np.percentile(_maha_z((Hm - mm) / ms, pm) + _maha_z((Ha - aM) / aS, pa), 95))

    def zsum(Xm, Xa):
        return _maha_z((Xm - mm) / ms, pm) + _maha_z((Xa - aM) / aS, pa)

    print(f"\nfit on {FIT_DS} healthy | healthy SUM flag@p95 = "
          f"{(zsum(Hm, Ha) > thr95).mean():.3f}\n")
    print(f"{'ds':<5} {'reclvl_ROC':>10} {'reclvl_PR':>10} {'href_ROC':>9} "
          f"{'href_PR':>8} {'flag@p95':>9} {'heldout':>8}")
    for dn in TEST_DS:
        hm, _ = H[dn]
        zs_h = zsum(hm, H[dn][1])               # healthy windows of this ds
        zs_a = zsum(A[dn]["m"], A[dn]["a"])     # anomaly windows of this ds
        # recording-level: this ds healthy (neg) vs anomaly (pos)
        y_rec = np.concatenate([np.zeros(zs_h.size), np.ones(zs_a.size)])
        s_rec = np.concatenate([zs_h, zs_a])
        rec_roc = roc_auc_score(y_rec, s_rec)
        rec_pr = average_precision_score(y_rec, s_rec)
        # window-level healthy-ref (impulsive)
        yr = A[dn]["ref"]
        if 0 < yr.sum() < yr.size:
            hr_roc = f"{roc_auc_score(yr, zs_a):.3f}"; hr_pr = f"{average_precision_score(yr, zs_a):.3f}"
        else:
            hr_roc = hr_pr = "  -  "
        flag = (zs_a > thr95).mean()
        print(f"{dn:<5} {rec_roc:>10.3f} {rec_pr:>10.3f} {hr_roc:>9} {hr_pr:>8} "
              f"{flag:>9.3f} {'YES' if dn not in FIT_DS else '':>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
