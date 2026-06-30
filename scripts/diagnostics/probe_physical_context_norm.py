"""Probe: physical features + context regime-normalization + sum fusion, GLOBAL.

Validates the proposed fix end-to-end, post-hoc and label-free:
  1. physical features per window (energy/impulsiveness) — the signal the
     encoder discards;
  2. context-local regime-normalization — subtract the mean of each physical
     feature over the k nearest POOLED-healthy windows in V2-context space
     (the learnable version of this is the flow's conditional base / FiLM);
  3. a single GLOBAL Mahalanobis detector per modality on the normalized
     features, sum-fused across modalities.

If the GLOBAL per-campaign AUC recovers to ~0.9+ on D3/D4 (which collapsed to
0.69/0.73 under a plain global physical baseline), the design — physical
injection + conditional base + sum fusion, one global threshold — is validated
and worth wiring into training.

Run:  python -m scripts.diagnostics.probe_physical_context_norm [run_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.eval.rq2_three_paradigm_eval import (
    _build_loader,
    _build_v2,
    _load_state,
    _loader,
    _segments_for,
)
from src.modeling.orchestration.full_run import v2_config

DEFAULT_RUN = REPO / "results" / "runs" / "20260616_022513__full_pipeline_b5_cma"
FEATS = ["mean", "std", "max", "crest", "kurt"]


def _winfeats(feat: np.ndarray) -> np.ndarray:
    b = feat.shape[0]
    x = feat.reshape(b, -1, feat.shape[-1]).astype(np.float64)
    env = x.mean(axis=1)
    mu = env.mean(axis=1, keepdims=True)
    sd = env.std(axis=1, keepdims=True) + 1e-8
    return np.stack([
        x.mean(axis=(1, 2)), x.std(axis=(1, 2)), x.max(axis=(1, 2)),
        env.max(axis=1) / (np.abs(env.mean(axis=1)) + 1e-8),
        (((env - mu) / sd) ** 4).mean(axis=1) - 3.0,
    ], axis=1)


def _auc(h: np.ndarray, a: np.ndarray) -> float:
    if h.size == 0 or a.size == 0:
        return float("nan")
    allv = np.concatenate([h, a]); r = allv.argsort().argsort() + 1.0
    return float((r[h.size:].sum() - a.size * (a.size + 1) / 2) / (h.size * a.size))


def main() -> int:
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RUN
    print(f"run = {run.name}")
    v2_cfg = v2_config(False)
    v2 = _build_v2(v2_cfg); _load_state(run / "v2" / "encoder.pt", v2); v2.eval()

    loaders = [_loader(d) for d in ("d1", "d2", "d3", "d4")]
    ctx, ac, vib, dsid, anom = [], [], [], [], []
    for i, dn in enumerate(("d1", "d2", "d3", "d4")):
        for is_anom in (False, True):
            segs = _segments_for([loaders[i]], v2_cfg, healthy=not is_anom)
            if not segs:
                continue
            print(f"scoring {dn} {'anom' if is_anom else 'healthy'} ({len(segs)} segs) ...", flush=True)
            with torch.no_grad():
                for b in _build_loader(segs, v2_cfg):
                    d = v2(b["ac_feat"], b["ac_xyz"], b["vib_feat"], b["vib_xyz"],
                           b["dataset_idx"], mask_p=0.0)
                    n = b["ac_feat"].shape[0]
                    ctx.append(d["context"].cpu().numpy())
                    ac.append(_winfeats(b["ac_feat"].numpy()))
                    vib.append(_winfeats(b["vib_feat"].numpy()))
                    dsid.append(np.full(n, i)); anom.append(np.full(n, is_anom))
    ctx = np.concatenate(ctx); ac = np.concatenate(ac); vib = np.concatenate(vib)
    dsid = np.concatenate(dsid); anom = np.concatenate(anom).astype(bool)

    from sklearn.neighbors import NearestNeighbors
    K = 128
    h = ~anom
    nn = NearestNeighbors(n_neighbors=K + 1).fit(ctx[h])

    def _norm(feat: np.ndarray) -> np.ndarray:
        """Subtract context-local healthy mean (drop self for healthy rows)."""
        out = np.empty_like(feat)
        _, idx_h = nn.kneighbors(ctx[h])
        out[h] = feat[h] - feat[h][idx_h[:, 1:]].mean(axis=1)
        _, idx_a = nn.kneighbors(ctx[~h])
        out[~h] = feat[~h] - feat[h][idx_a[:, :K]].mean(axis=1)
        return out

    def _maha_z(X: np.ndarray, hmask: np.ndarray) -> np.ndarray:
        mu = X[hmask].mean(0); cov = np.cov(X[hmask], rowvar=False) + 1e-6 * np.eye(X.shape[1])
        inv = np.linalg.inv(cov)
        s = np.einsum("ij,jk,ik->i", X - mu, inv, X - mu)
        return (s - s[hmask].mean()) / (s[hmask].std() + 1e-8)

    def _norm_ridge(feat: np.ndarray) -> np.ndarray:
        """Parametric, DEPLOYABLE regime-normalization: residual = feat - mu(c),
        mu = Ridge(context -> feat) fit on pooled healthy only (no memory bank)."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(ctx[h])
        m = Ridge(alpha=10.0).fit(sc.transform(ctx[h]), feat[h])
        return feat - m.predict(sc.transform(ctx))

    variants = [
        ("plain GLOBAL", lambda f: f),
        ("context-norm kNN GLOBAL", _norm),
        ("context-norm Ridge GLOBAL (deployable)", _norm_ridge),
    ]
    for tag, fn in variants:
        za, zv = _maha_z(fn(ac), h), _maha_z(fn(vib), h)
        zsum = za + zv
        print(f"\n########## physical Mahalanobis + SUM — {tag} ##########")
        print(f"{'ds':<6} {'ac_AUC':>7} {'vib_AUC':>8} {'SUM_AUC':>8}")
        for i, dn in enumerate(("d1", "d2", "d3", "d4")):
            m = anom & (dsid == i)
            if not m.any():
                continue
            print(f"{dn:<6} {_auc(za[h], za[m]):>7.3f} {_auc(zv[h], zv[m]):>8.3f} "
                  f"{_auc(zsum[h], zsum[m]):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
