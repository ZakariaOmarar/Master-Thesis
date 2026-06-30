"""Raw-waveform anomaly detector: impulse and spectral features, healthy density.

This is the strongest anomaly detector measured on the rig (see
scripts/diagnostics/probe_impulse_detector.py). The SSL/CMA deep pipeline
discards the anomaly-relevant cues, whereas classical condition-monitoring
features on the raw waveform recover them with no encoder at all:

  - Impulse family (amplitude transients, e.g. the D2/D4/D5 knocks): crest,
    impulse, clearance, shape factors, kurtosis, spectral kurtosis (Antoni),
    envelope knock-count, peak/median.
  - Spectral family (spectral anomalies such as D3, which is not impulsive):
    centroid, spread, flatness, entropy, 85%-rolloff, low/mid/high band-energy
    ratios.

Per modality (mic, accel) the features are standardised and scored by
Mahalanobis distance against the healthy feature distribution. The per-modality
z-scores are sum-fused into one global score, thresholded at the (1-target_fpr)
quantile of the healthy fused score. Everything is fit on healthy data only
(unsupervised, label-free).

Verified: D2 ROC 0.98 (PR 0.96), D3 0.91 (spectral), D4 href-ROC 0.94, D5
held-out 0.99 (PR 1.00); healthy FPR pinned to target.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IMPULSE_FEATS = ("crest", "impulse", "clearance", "shape", "kurtosis",
                 "spectral_kurtosis", "knock_count", "peak_over_median")
SPECTRAL_FEATS = ("centroid", "spread", "flatness", "entropy", "rolloff85",
                  "band_low", "band_mid", "band_high")
FEATS = IMPULSE_FEATS + SPECTRAL_FEATS


def _impulse_feats(w: np.ndarray, fs: float) -> list[float]:
    aw = np.abs(w)
    rms = np.sqrt(np.mean(w * w)) + 1e-12
    peak = float(aw.max())
    mean_abs = float(aw.mean()) + 1e-12
    mu, sd = w.mean(), w.std() + 1e-12
    sk = 0.0
    try:
        from scipy.signal import stft
        nper = min(256, max(16, w.size // 8))
        _, _, Z = stft(w, fs=fs, nperseg=nper, noverlap=nper // 2)
        mag = np.abs(Z)
        if mag.shape[1] >= 4:
            m = mag.mean(1, keepdims=True)
            s = mag.std(1, keepdims=True) + 1e-12
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
    cs = np.cumsum(W)
    rolloff = float(fr[np.searchsorted(cs, 0.85 * cs[-1])])
    th = fr.max() / 3.0 + 1e-9
    return [centroid, spread, flatness, entropy, rolloff,
            float(W[fr < th].sum() / W.sum()),
            float(W[(fr >= th) & (fr < 2 * th)].sum() / W.sum()),
            float(W[fr >= 2 * th].sum() / W.sum())]


def window_features(w: np.ndarray, fs: float) -> np.ndarray:
    """16-D impulse+spectral feature vector for one 1-D window."""
    w = np.asarray(w, dtype=np.float64)
    if w.size < 16 or not np.any(w):
        return np.zeros(len(FEATS))
    return np.array(_impulse_feats(w, fs) + _spectral_feats(w, fs))


def recording_windows(signal_1d: np.ndarray, fs: float,
                      win_s: float = 1.0, stride_s: float = 0.5):
    """Window a 1-D signal; yield (t_start, t_end, feature_vector)."""
    T = signal_1d.size
    wlen = int(win_s * fs)
    step = max(1, int(stride_s * fs))
    for i0 in range(0, max(1, T - wlen + 1), step):
        seg = signal_1d[i0:i0 + wlen]
        yield i0 / fs, (i0 + wlen) / fs, window_features(seg, fs)


@dataclass
class _ModalityMaha:
    mean: np.ndarray
    std: np.ndarray
    res_mean: np.ndarray
    inv_cov: np.ndarray
    score_mean: float
    score_std: float

    @classmethod
    def fit(cls, X: np.ndarray) -> _ModalityMaha:
        mean = X.mean(0)
        std = X.std(0) + 1e-8
        Z = (X - mean) / std
        rm = Z.mean(0)
        cov = np.cov(Z - rm, rowvar=False) + 1e-6 * np.eye(Z.shape[1])
        inv = np.linalg.inv(cov)
        s = np.einsum("ij,jk,ik->i", Z - rm, inv, Z - rm)
        return cls(mean, std, rm, inv, float(s.mean()), float(s.std() + 1e-8))

    def z(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.std - self.res_mean
        s = np.einsum("ij,jk,ik->i", Z, self.inv_cov, Z)
        return (s - self.score_mean) / self.score_std


@dataclass
class RawImpulseDetector:
    """Sum-fused impulse+spectral raw-waveform anomaly detector (healthy-fit)."""
    target_fpr: float = 0.05
    win_s: float = 1.0
    stride_s: float = 0.5
    models: dict[str, _ModalityMaha] = field(default_factory=dict)
    threshold: float = 0.0

    def fit(self, healthy: dict[str, np.ndarray]) -> RawImpulseDetector:
        """`healthy` = {modality: (N_windows, n_feats)} from healthy recordings."""
        self.models = {m: _ModalityMaha.fit(X) for m, X in healthy.items()}
        fused = self.fused_score(healthy)
        self.threshold = float(np.percentile(fused, 100.0 * (1.0 - self.target_fpr)))
        return self

    def modality_scores(self, feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {m: self.models[m].z(feats[m]) for m in self.models}

    def fused_score(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        return sum(self.modality_scores(feats).values())

    def alert(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        return self.fused_score(feats) > self.threshold


__all__ = [
    "FEATS",
    "IMPULSE_FEATS",
    "SPECTRAL_FEATS",
    "RawImpulseDetector",
    "recording_windows",
    "window_features",
]
