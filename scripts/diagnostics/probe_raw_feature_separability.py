"""Probe: is each campaign's anomaly signal PHYSICALLY present in both modalities?

Model-free.  For every window it computes a handful of cheap, interpretable
statistics directly on the precomputed acoustic / vibration features
(no V1/V2/V3 encoder involved) and reports the anomaly-vs-healthy AUC per
campaign per modality.  A knock is an impulsive event, so it should raise the
energy / crest-factor / temporal-kurtosis of *both* the air-borne (acoustic)
and structure-borne (vibration) features.

Interpretation:
  * If a raw feature on the modality that the LEARNED encoder calls "blind"
    (e.g. vibration on D3, acoustic on D4) still separates the campaign, then
    the signal IS there and the encoder / CMA fusion is discarding it -> "both
    modalities help" is achievable and the fix is architectural.
  * If even the best raw feature is at chance, that sensor is physically blind
    to that fault and single-modality dominance is a hard limit.

AUC is computed against the same-dataset healthy (regime-matched) so the
operating-condition offset does not confound the comparison.

Run:  python -m scripts.diagnostics.probe_raw_feature_separability
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.eval.rq2_three_paradigm_eval import (
    _build_loader,
    _loader,
    _segments_for,
)
from src.modeling.orchestration.full_run import v2_config


def _auc(healthy: np.ndarray, anom: np.ndarray) -> float:
    if healthy.size == 0 or anom.size == 0:
        return float("nan")
    allv = np.concatenate([healthy, anom])
    ranks = allv.argsort().argsort().astype(np.float64) + 1.0
    r_anom = ranks[healthy.size:].sum()
    n_h, n_a = healthy.size, anom.size
    return float((r_anom - n_a * (n_a + 1) / 2.0) / (n_h * n_a))


def _winfeats(feat: np.ndarray) -> dict[str, np.ndarray]:
    """Per-window scalar features from a (B, ..., T) feature batch."""
    b = feat.shape[0]
    x = feat.reshape(b, -1, feat.shape[-1]).astype(np.float64)  # (B, F, T)
    env = x.mean(axis=1)                                        # (B, T) energy envelope
    mu = env.mean(axis=1, keepdims=True)
    sd = env.std(axis=1, keepdims=True) + 1e-8
    kurt = (((env - mu) / sd) ** 4).mean(axis=1) - 3.0          # impulsiveness
    crest = env.max(axis=1) / (np.abs(env.mean(axis=1)) + 1e-8)
    return {
        "mean": x.mean(axis=(1, 2)),
        "std": x.std(axis=(1, 2)),
        "max": x.max(axis=(1, 2)),
        "crest": crest,
        "kurt": kurt,
    }


def _collect(segs, cfg) -> dict[str, dict[str, np.ndarray]]:
    """{modality: {featname: (N,)}} over all windows in `segs`."""
    acc = {"ac": {}, "vib": {}}
    loader = _build_loader(segs, cfg)
    for batch in loader:
        for mod, key in (("ac", "ac_feat"), ("vib", "vib_feat")):
            for fn, val in _winfeats(batch[key].numpy()).items():
                acc[mod].setdefault(fn, []).append(val)
    return {m: {fn: np.concatenate(v) for fn, v in d.items()} for m, d in acc.items()}


def main() -> int:
    cfg = v2_config(False)
    loaders = [_loader(d) for d in ("d1", "d2", "d3", "d4")]
    feats = ["mean", "std", "max", "crest", "kurt"]

    print("collecting per-dataset windows (model-free) ...")
    H, A = {}, {}
    for i, dn in enumerate(("d1", "d2", "d3", "d4")):
        h = _segments_for([loaders[i]], cfg, healthy=True)
        a = _segments_for([loaders[i]], cfg, healthy=False)
        if h:
            print(f"  {dn} healthy: {len(h)} segs", flush=True)
            H[dn] = _collect(h, cfg)
        if a:
            print(f"  {dn} anomaly: {len(a)} segs", flush=True)
            A[dn] = _collect(a, cfg)

    for mod in ("ac", "vib"):
        print(f"\n========== {mod} (raw-feature anomaly-vs-own-healthy AUC) ==========")
        print(f"{'ds':<6} " + " ".join(f"{f:>7}" for f in feats) + f"  {'BEST':>14}")
        for dn in A:
            if dn not in H:
                continue
            row, best, best_f = [], 0.5, ""
            for f in feats:
                au = _auc(H[dn][mod][f], A[dn][mod][f])
                row.append(au)
                if abs(au - 0.5) >= abs(best - 0.5):
                    best, best_f = au, f
            print(f"{dn:<6} " + " ".join(f"{a:>7.3f}" for a in row)
                  + f"  {best_f}={best:.3f}")

    # ----- physical-feature anomaly detector + sum late-fusion -----
    # Per modality: fit a Gaussian on OWN-dataset healthy feature vectors,
    # score by squared Mahalanobis distance, z-score against healthy.  Fuse the
    # two modalities by SUMMING the z-scored scores (conditional-independence
    # late fusion) so both modalities contribute.  Tests whether a simple
    # physical-feature detector + sum-fusion makes both modalities help.
    def _mat(d: dict) -> np.ndarray:
        return np.stack([d[f] for f in feats], axis=1)

    def _maha_z(h: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = h.mean(axis=0)
        cov = np.cov(h, rowvar=False) + 1e-6 * np.eye(h.shape[1])
        inv = np.linalg.inv(cov)
        sc_h = np.einsum("ij,jk,ik->i", h - mu, inv, h - mu)
        sc_q = np.einsum("ij,jk,ik->i", q - mu, inv, q - mu)
        m, s = sc_h.mean(), sc_h.std() + 1e-8
        return (sc_h - m) / s, (sc_q - m) / s

    # OWN-dataset (regime-matched) baseline.
    print("\n########## physical Mahalanobis + SUM fusion — OWN-dataset healthy ##########")
    print(f"{'ds':<6} {'ac_AUC':>7} {'vib_AUC':>8} {'SUM_AUC':>8}")
    for dn in A:
        if dn not in H:
            continue
        zah, zaa = _maha_z(_mat(H[dn]["ac"]), _mat(A[dn]["ac"]))
        zvh, zva = _maha_z(_mat(H[dn]["vib"]), _mat(A[dn]["vib"]))
        print(f"{dn:<6} {_auc(zah, zaa):>7.3f} {_auc(zvh, zva):>8.3f} "
              f"{_auc(zah + zvh, zaa + zva):>8.3f}")

    # GLOBAL (pooled-healthy) baseline — the deployable, single-threshold case.
    # Fit one Mahalanobis per modality on ALL healthy; healthy AUC reference is
    # the pooled healthy. This tests whether the physical detector survives the
    # user's hard constraint (one global baseline, no per-campaign fit).
    print("\n########## physical Mahalanobis + SUM fusion — GLOBAL healthy ##########")
    gh_ac = np.concatenate([_mat(H[dn]["ac"]) for dn in H])
    gh_vib = np.concatenate([_mat(H[dn]["vib"]) for dn in H])
    zgh_a, _ = _maha_z(gh_ac, gh_ac[:1])   # healthy ref scores
    zgh_v, _ = _maha_z(gh_vib, gh_vib[:1])
    print(f"{'ds':<6} {'ac_AUC':>7} {'vib_AUC':>8} {'SUM_AUC':>8}")
    for dn in A:
        _, zaa = _maha_z(gh_ac, _mat(A[dn]["ac"]))
        _, zva = _maha_z(gh_vib, _mat(A[dn]["vib"]))
        print(f"{dn:<6} {_auc(zgh_a, zaa):>7.3f} {_auc(zgh_v, zva):>8.3f} "
              f"{_auc(zgh_a + zgh_v, zaa + zva):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
