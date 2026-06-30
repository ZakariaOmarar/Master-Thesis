"""V0-style anomaly baseline — per-cluster diagonal Gaussian on V2 c_t buckets.

A simple, stable density baseline the V3 conditional flow must beat to earn
its complexity.  Without it, V3's val NLL would be reported without reference
to what a non-deep density estimator achieves on the same
`(x = mean_pool(fused), c = PMA-pooled c_t)` pairs.

NOTE (2026-05-23): the original implementation used `scipy.stats.gaussian_kde`,
which is numerically indefensible at this dimensionality (x is 64-128 D with
O(hundreds) healthy points → near-singular bandwidth → logpdf returned ±10³
garbage; the deep-campaign comparison was meaningless).  Replaced with a
per-cluster **diagonal Gaussian** (mean + per-dim variance with a floor),
the standard stable simple-density baseline that is well-defined in any
dimension.  The function name is kept for call-site stability.

Design:
  1. K-means(`n_clusters`) on the V3 healthy training-cohort `c_t` vectors —
     same `n_clusters` (default 3) as `PerClusterThresholds` so the buckets
     line up with V3's threshold structure.
  2. Per-cluster diagonal Gaussian fit on the bucketed `x_for_v3` vectors
     (V3's pre-flow input).  Clusters with < 2 points fall back to the pooled
     Gaussian over all `x_train`.
  3. Score = mean per-window NLL ``0.5 Σ_d[(x-μ)²/σ² + log(2π σ²)]`` — same
     orientation as V3's ``-log p(x|c)`` (lower = more in-distribution).

Reported metric: mean held-out NLL on the same `val_eval` cohort V3 uses.
A V3-vs-baseline delta near zero means V3 has not extracted information
beyond a simple per-cluster Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class KDEResult:
    """Outcome of fitting KDE-on-c_t on a healthy training cohort."""

    centroids: np.ndarray  # (K, c_dim)
    val_nll_mean: float    # mean held-out -log p(x)
    val_nll_per_cluster: dict[int, float]
    n_per_cluster_train: np.ndarray  # (K,)
    n_per_cluster_val: np.ndarray    # (K,)
    n_clusters_used: int             # may be < requested if cohort small


def fit_and_score_kde_on_ct(
    x_train: np.ndarray,
    c_train: np.ndarray,
    x_val: np.ndarray,
    c_val: np.ndarray,
    *,
    n_clusters: int = 3,
    seed: int = 42,
) -> KDEResult:
    """K-means(c_train) → per-cluster KDE on x_train → mean held-out NLL.

    Mirrors V3's evaluation: `x_val` is scored against the per-cluster KDE
    whose centroid is closest to each `c_val[i]`.  When a cluster has < 2
    training points its KDE cannot be fit (singular covariance); those
    windows fall back to the pooled KDE over `x_train`.
    """
    if x_train.ndim != 2 or x_val.ndim != 2 or x_train.shape[1] != x_val.shape[1]:
        raise ValueError(
            f"x_train/x_val shape mismatch: {x_train.shape} vs {x_val.shape}"
        )
    n_clusters_eff = max(1, min(int(n_clusters), c_train.shape[0]))
    km = KMeans(n_clusters=n_clusters_eff, random_state=seed, n_init=10).fit(c_train)
    centroids = km.cluster_centers_.astype(np.float64)
    train_labels = km.labels_
    # Nearest-centroid assignment for val (KMeans.predict refits internally —
    # using argmin over euclidean distances is cheaper and explicit).
    d_val = np.linalg.norm(c_val[:, None, :] - centroids[None, :, :], axis=-1)
    val_labels = d_val.argmin(axis=1)

    # Per-cluster DIAGONAL GAUSSIAN density (not gaussian_kde).
    #
    # The original gaussian_kde baseline was numerically indefensible here:
    # x lives in 64-128 dims with O(hundreds) healthy points, so the KDE
    # bandwidth matrix is near-singular and logpdf returned ±10³ garbage
    # (observed in the 2026-05-23 deep campaign — the comparison was
    # meaningless). A per-cluster diagonal Gaussian — fit mean μ and
    # per-dimension variance σ² with a variance floor — is the standard
    # stable "simple density" baseline and is well-defined in any dimension.
    # NLL(x) = 0.5 Σ_d [ (x_d-μ_d)²/σ_d² + log(2π σ_d²) ].
    var_floor = 1e-6 * float(np.var(x_train) + 1e-12)

    def _fit_gauss(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = rows.mean(axis=0)
        var = rows.var(axis=0) + var_floor
        return mu, var

    def _nll(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        return 0.5 * (((x - mu) ** 2) / var + np.log(2.0 * np.pi * var)).sum(axis=1)

    pooled = _fit_gauss(x_train) if x_train.shape[0] >= 2 else None
    per_cluster: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    n_train_per: list[int] = []
    for k in range(n_clusters_eff):
        mask = train_labels == k
        n_train_per.append(int(mask.sum()))
        if mask.sum() >= 2:
            per_cluster[k] = _fit_gauss(x_train[mask])

    n_val_per: list[int] = []
    nll_per_cluster: dict[int, float] = {}
    nll_accum = np.zeros(x_val.shape[0], dtype=np.float64)
    for k in range(n_clusters_eff):
        mask = val_labels == k
        n_val_per.append(int(mask.sum()))
        if not mask.any():
            continue
        params = per_cluster.get(k, pooled)
        if params is None:
            nll_accum[mask] = float("nan")
            nll_per_cluster[k] = float("nan")
            continue
        mu, var = params
        per_win = _nll(x_val[mask], mu, var)
        nll_accum[mask] = per_win
        nll_per_cluster[k] = float(np.mean(per_win))

    finite = np.isfinite(nll_accum)
    val_nll_mean = float(np.mean(nll_accum[finite])) if finite.any() else float("nan")

    return KDEResult(
        centroids=centroids,
        val_nll_mean=val_nll_mean,
        val_nll_per_cluster=nll_per_cluster,
        n_per_cluster_train=np.asarray(n_train_per, dtype=np.int64),
        n_per_cluster_val=np.asarray(n_val_per, dtype=np.int64),
        n_clusters_used=n_clusters_eff,
    )


__all__ = ["KDEResult", "fit_and_score_kde_on_ct"]
