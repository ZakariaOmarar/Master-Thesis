"""Impulse + spectral anchor features for the conditional V3 flow (RQ2).

The SSL/CMA encoder optimises away the impulsiveness a knock produces (the
flow-input embedding cannot predict crest factor; Ridge R^2 ~ 0).  Rather than
abandon context-conditioning (RQ2's whole point), we augment the
conditional-flow input with the full condition-monitoring feature set — the same
families that let the hand-crafted baseline recover every campaign — so the
conditional density model sees the anomaly while still being conditioned on the
operating context ``c_t``.

Two families per modality (matching the validated baseline, raw_impulse_detector
.py), computed on the same windowed log-mel + CWT features the V2 encoder
consumes (so the anchor is identical at every site that builds the flow input —
V3 training, V3 eval, the V4 ``x_for_v3`` gate, sliding-window inference — no raw
re-windowing, no id/time drift):

  * IMPULSE (amplitude transients — D2/D4/D5 knocks): crest, impulse, clearance,
    shape factors, kurtosis, knock-count, peak/median, spectral-kurtosis.
  * SPECTRAL (spectral anomalies — D3, recovered by exactly these): centroid,
    spread, flatness, entropy, 85%-rolloff, low/mid/high band-energy ratios.

Acoustic features come from the CWT channel (transient) + the log-mel channel
(spectral); vibration from its energy envelope + the envelope FFT.  Stats are
pooled over sensors so the anchor is sensor-count agnostic.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _impulse_set(env: np.ndarray) -> list[float]:
    """7 time-domain impulse condition indicators on a 1-D envelope."""
    env = np.asarray(env, dtype=np.float64)
    if env.size < 4 or not np.any(env):
        return [0.0] * 7
    aw = np.abs(env)
    rms = np.sqrt(np.mean(env * env)) + _EPS
    peak = float(aw.max())
    mean_abs = float(aw.mean()) + _EPS
    mu, sd = env.mean(), env.std() + _EPS
    thr = 3.0 * (np.median(aw) + _EPS)
    kcount = float(np.sum(np.diff((aw > thr).astype(int)) == 1))
    return [peak / rms, peak / mean_abs,
            peak / (np.mean(np.sqrt(aw)) ** 2 + _EPS), rms / mean_abs,
            float(np.mean(((env - mu) / sd) ** 4) - 3.0), kcount,
            peak / (np.median(aw) + _EPS)]


def _spectral_set(spec: np.ndarray) -> list[float]:
    """8 spectral-shape features on a 1-D (non-negative) frequency spectrum."""
    spec = np.asarray(spec, dtype=np.float64)
    F = spec.shape[0]
    if F < 4 or not np.any(spec):
        return [0.0] * 8
    s = np.clip(spec, _EPS, None)
    sn = s / s.sum()
    fr = np.linspace(0.0, 1.0, F)
    centroid = float((fr * sn).sum())
    spread = float(np.sqrt((((fr - centroid) ** 2) * sn).sum()))
    flatness = float(np.exp(np.mean(np.log(s))) / (s.mean() + _EPS))
    entropy = float(-(sn * np.log(sn + _EPS)).sum() / np.log(F))
    cs = np.cumsum(s)
    rolloff = float(np.searchsorted(cs, 0.85 * cs[-1]) / (F - 1))
    th = F / 3.0
    tot = s.sum() + _EPS
    return [centroid, spread, flatness, entropy, rolloff,
            float(s[:int(th)].sum() / tot),
            float(s[int(th):int(2 * th)].sum() / tot),
            float(s[int(2 * th):].sum() / tot)]


def _spectral_kurtosis(M: np.ndarray) -> float:
    """Antoni spectral kurtosis: max over freq of excess kurtosis across time."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[1] < 4:
        return 0.0
    m = M.mean(axis=1, keepdims=True)
    s = M.std(axis=1, keepdims=True) + _EPS
    return float(np.nanmax((((M - m) / s) ** 4).mean(axis=1) - 3.0))


def _acoustic(ac: np.ndarray) -> list[float]:
    """16 features for ONE acoustic window. ac=(N_mic, 2, F, T)."""
    ac = np.asarray(ac, dtype=np.float64)
    if ac.ndim == 4 and ac.shape[1] >= 2:
        mel = ac[:, 0].mean(0)          # (F, T) log-mel, avg mics
        cwt = ac[:, 1].mean(0)          # (F, T) CWT (transient-rich)
    else:
        flat = ac.reshape(ac.shape[0], -1, ac.shape[-1]).mean(0)
        mel = cwt = flat
    env = cwt.mean(axis=0)              # (T,) transient envelope
    impulse = _impulse_set(env) + [_spectral_kurtosis(cwt)]      # 8
    mspec = mel.mean(axis=1)           # (F,) time-averaged spectrum
    mspec = mspec - mspec.min() + _EPS  # log-mel can be negative -> shift positive
    return impulse + _spectral_set(mspec)                        # 8 -> 16


def _vibration(vib: np.ndarray) -> list[float]:
    """16 features for ONE vibration window. vib=(N_vib, C, T)."""
    vib = np.asarray(vib, dtype=np.float64)
    env = np.sqrt((vib ** 2).mean(axis=tuple(range(vib.ndim - 1))))  # (T,)
    sk = 0.0
    if env.size >= 16:
        try:
            from scipy.signal import stft
            nper = min(128, max(16, env.size // 8))
            _, _, Z = stft(env, nperseg=nper, noverlap=nper // 2)
            sk = _spectral_kurtosis(np.abs(Z))
        except Exception:
            sk = 0.0
    impulse = _impulse_set(env) + [sk]                              # 8
    if env.size >= 8:
        spec = np.abs(np.fft.rfft(env * np.hanning(env.size))) ** 2
    else:
        spec = np.zeros(4)
    return impulse + _spectral_set(spec)                           # 8 -> 16


# Fixed order — persisted standardization stats + the flow input dimension depend
# on it; never reorder (only append, and retrain).
ANCHOR_FEATURES = tuple(
    f"ac_{n}" for n in (
        "crest", "impulse", "clearance", "shape", "kurtosis", "knockcount",
        "peakovermed", "speckurt", "centroid", "spread", "flatness", "entropy",
        "rolloff", "bandlow", "bandmid", "bandhigh")
) + tuple(
    f"vib_{n}" for n in (
        "crest", "impulse", "clearance", "shape", "kurtosis", "knockcount",
        "peakovermed", "speckurt", "centroid", "spread", "flatness", "entropy",
        "rolloff", "bandlow", "bandmid", "bandhigh")
)
N_ANCHOR = len(ANCHOR_FEATURES)  # 32


def _to_numpy(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def _anchor_one(ac: np.ndarray, vib: np.ndarray) -> np.ndarray:
    return np.array(_acoustic(ac) + _vibration(vib), dtype=np.float64)


def impulse_spectral_anchor(ac_feat, vib_feat) -> np.ndarray:
    """Batched anchor.  ac_feat=(B,N_mic,2,F,T), vib_feat=(B,N_vib,C,T) -> (B,N_ANCHOR)."""
    ac = np.asarray(_to_numpy(ac_feat), dtype=np.float64)
    vib = np.asarray(_to_numpy(vib_feat), dtype=np.float64)
    out = np.stack([_anchor_one(ac[i], vib[i]) for i in range(ac.shape[0])], axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def append_anchor(x, ac_feat, vib_feat, anchor_norm):
    """Concatenate the standardized impulse+spectral anchor to a pooled flow input.

    THE single source of truth used by every V3-flow scoring site so the anchor
    is computed and standardized identically everywhere.  ``anchor_norm`` =
    ``(mean, std)`` healthy standardization from V3 training, or ``None`` to pass
    ``x`` through unchanged (anchor disabled).
    """
    if anchor_norm is None:
        return x
    mean, std = anchor_norm
    a = (impulse_spectral_anchor(ac_feat, vib_feat) - mean) / std
    if hasattr(x, "detach"):  # torch.Tensor
        import torch
        return torch.cat([x, torch.as_tensor(a, dtype=x.dtype, device=x.device)], dim=1)
    return np.concatenate([np.asarray(x), a.astype(np.asarray(x).dtype)], axis=1)


__all__ = ["ANCHOR_FEATURES", "N_ANCHOR", "append_anchor", "impulse_spectral_anchor"]
