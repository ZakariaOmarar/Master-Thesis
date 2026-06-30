"""Physical-feature anomaly detector with context regime-normalization.

Motivation (see scripts/diagnostics/probe_*; thesis RQ2):
The SSL/CMA encoders optimise away the impulsiveness/energy cues a knock
produces — a Ridge probe shows the flow-input embedding cannot even predict the
raw crest factor (R^2≈0.1; vibration R^2≈0).  A handful of cheap physical
statistics on the raw features separate every campaign (acoustic raw-AUC up to
0.98) and, once regime-normalised against the learned context, a single global
detector reaches AUC 0.93–0.98 with both modalities contributing — which the
deep pipeline never achieved (fusion fell to 0.56 on D4).

Pipeline (all fit on healthy only — label-free):
  1. physical stats per window per modality: mean, std, max, crest, kurt;
  2. regime-normalise: residual = feat - μ(c), μ from the V2 context c
     ("knn" = local healthy mean among k neighbours; "ridge" = parametric,
     deployable, no memory bank);
  3. per-modality squared-Mahalanobis on the residuals, z-scored by healthy;
  4. sum-fuse the per-modality z-scores → one global score + one global
     threshold (the (1-target_fpr) quantile of the healthy fused score).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PHYS_FEATS = ("mean", "std", "max", "crest", "kurt")


def physical_features(feat: np.ndarray) -> np.ndarray:
    """Per-window physical statistics from a ``(B, ..., T)`` feature batch.

    Returns ``(B, len(PHYS_FEATS))`` = [mean, std, max, crest, kurt], where
    crest/kurt are computed on the per-frame energy envelope (impulsiveness).
    """
    b = feat.shape[0]
    x = np.asarray(feat, dtype=np.float64).reshape(b, -1, feat.shape[-1])
    env = x.mean(axis=1)
    mu = env.mean(axis=1, keepdims=True)
    sd = env.std(axis=1, keepdims=True) + 1e-8
    return np.stack([
        x.mean(axis=(1, 2)),
        x.std(axis=(1, 2)),
        x.max(axis=(1, 2)),
        env.max(axis=1) / (np.abs(env.mean(axis=1)) + 1e-8),
        (((env - mu) / sd) ** 4).mean(axis=1) - 3.0,
    ], axis=1)


@dataclass
class _ModalityModel:
    """Per-modality regime-normalizer + Mahalanobis, fit on healthy."""
    normalizer: str
    # ridge params (parametric μ(c))
    ridge_w: np.ndarray | None = None       # (c_dim, n_feat)
    ridge_b: np.ndarray | None = None       # (n_feat,)
    ctx_mean: np.ndarray | None = None       # (c_dim,)
    ctx_std: np.ndarray | None = None        # (c_dim,)
    # knn bank (non-parametric μ(c))
    bank_ctx: np.ndarray | None = None       # (N_h, c_dim)
    bank_feat: np.ndarray | None = None      # (N_h, n_feat)
    k: int = 128
    # Mahalanobis on residuals
    res_mean: np.ndarray | None = None       # (n_feat,)
    res_inv_cov: np.ndarray | None = None    # (n_feat, n_feat)
    score_mean: float = 0.0
    score_std: float = 1.0

    def _mu(self, ctx: np.ndarray) -> np.ndarray:
        if self.normalizer == "ridge":
            cn = (ctx - self.ctx_mean) / self.ctx_std
            return cn @ self.ridge_w + self.ridge_b
        # knn: local healthy mean among k nearest (brute force, fine at scale)
        assert self.bank_ctx is not None and self.bank_feat is not None
        out = np.empty((ctx.shape[0], self.bank_feat.shape[1]), dtype=np.float64)
        bn = (self.bank_ctx ** 2).sum(1)
        for i in range(ctx.shape[0]):
            d = bn - 2.0 * (self.bank_ctx @ ctx[i]) + (ctx[i] ** 2).sum()
            nn = np.argpartition(d, self.k)[: self.k]
            out[i] = self.bank_feat[nn].mean(0)
        return out

    def residual(self, feat: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        return feat - self._mu(ctx)

    def score_z(self, feat: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        r = self.residual(feat, ctx) - self.res_mean
        s = np.einsum("ij,jk,ik->i", r, self.res_inv_cov, r)  # squared Mahalanobis
        return (s - self.score_mean) / self.score_std


@dataclass
class PhysicalFeatureDetector:
    """Sum-fused, context-normalized physical-feature anomaly detector."""
    normalizer: str = "knn"          # "knn" (best AUC) or "ridge" (lightest)
    k: int = 128
    ridge_alpha: float = 10.0
    target_fpr: float = 0.05
    models: dict[str, _ModalityModel] = field(default_factory=dict)
    threshold: float = 0.0

    @staticmethod
    def _fit_modality(feat: np.ndarray, ctx: np.ndarray, normalizer: str,
                      k: int, alpha: float) -> _ModalityModel:
        m = _ModalityModel(normalizer=normalizer, k=k)
        if normalizer == "ridge":
            m.ctx_mean = ctx.mean(0)
            m.ctx_std = ctx.std(0) + 1e-8
            cn = (ctx - m.ctx_mean) / m.ctx_std
            # ridge closed form: w = (XᵀX + αI)⁻¹ Xᵀ y, bias = mean(y) (cn is centred)
            xtx = cn.T @ cn + alpha * np.eye(cn.shape[1])
            m.ridge_w = np.linalg.solve(xtx, cn.T @ (feat - feat.mean(0)))
            m.ridge_b = feat.mean(0)
        elif normalizer == "knn":
            m.bank_ctx = ctx.copy()
            m.bank_feat = feat.copy()
        else:
            raise ValueError(f"unknown normalizer {normalizer!r}")
        # Mahalanobis on healthy residuals (drop-self handled approximately by
        # the +1 neighbour in knn at score time; at fit we use the plain μ).
        res = feat - m._mu(ctx)
        m.res_mean = res.mean(0)
        rc = res - m.res_mean
        cov = np.cov(rc, rowvar=False) + 1e-6 * np.eye(rc.shape[1])
        m.res_inv_cov = np.linalg.inv(cov)
        s = np.einsum("ij,jk,ik->i", rc, m.res_inv_cov, rc)
        m.score_mean, m.score_std = float(s.mean()), float(s.std() + 1e-8)
        return m

    def fit(self, phys: dict[str, np.ndarray], ctx: np.ndarray) -> PhysicalFeatureDetector:
        """Fit on healthy windows.  `phys` = {modality: (N, n_feat)}; `ctx` = (N, c_dim)."""
        self.models = {
            mod: self._fit_modality(phys[mod], ctx, self.normalizer, self.k, self.ridge_alpha)
            for mod in phys
        }
        fused = self.fused_score(phys, ctx)
        self.threshold = float(np.percentile(fused, 100.0 * (1.0 - self.target_fpr)))
        return self

    def modality_scores(self, phys: dict[str, np.ndarray], ctx: np.ndarray) -> dict[str, np.ndarray]:
        return {mod: self.models[mod].score_z(phys[mod], ctx) for mod in self.models}

    def fused_score(self, phys: dict[str, np.ndarray], ctx: np.ndarray) -> np.ndarray:
        return sum(self.modality_scores(phys, ctx).values())

    def alert(self, phys: dict[str, np.ndarray], ctx: np.ndarray) -> np.ndarray:
        return self.fused_score(phys, ctx) > self.threshold


__all__ = ["PHYS_FEATS", "PhysicalFeatureDetector", "physical_features"]
