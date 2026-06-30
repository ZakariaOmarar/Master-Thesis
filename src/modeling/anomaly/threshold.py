"""V3 alert mechanism — per-cluster percentile thresholds.

This design replaces CANDE-CP's reconstruction-AE *score* with the CNF
log-likelihood, but **keeps** CANDE-CP's per-bucket percentile thresholding
logic — re-bound from ground-truth mode label to V2's predicted K-means
cluster of `c_t`.  This preserves the label-leakage invariant: only
cluster IDs (label-free) feed into threshold fitting; mode labels appear only
in Chapter 6's per-mode FPR breakdown, never at fit time.

Fitting:
  1. K-means(`n_clusters`) on healthy `c_t` vectors.
  2. Bucket healthy CNF anomaly scores by cluster ID.
  3. Per-cluster 95th / 99th percentile of the score distribution.

Inference:
  1. Assign each new window to the nearest centroid in `c_t` space.
  2. Compare its anomaly score against the cluster's threshold.
  3. Emit alert if score > threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class PerClusterThresholds:
    """K-means(`K`) clusters fitted on healthy `c_t` + per-cluster percentiles.

    Two threshold tiers are always fitted from healthy scores (`p95`, `p99`).
    A third optional tier — `p_calibrated` — is filled by
    `calibrate_against_anomalies` when known-anomaly scores are available.
    """

    centroids: np.ndarray  # (K, D)
    p95: np.ndarray  # (K,)
    p99: np.ndarray  # (K,)
    n_per_cluster: np.ndarray  # (K,)
    seed: int = 0
    p_calibrated: np.ndarray | None = None  # (K,)
    calibration_youden_j: np.ndarray | None = None  # (K,) max Youden J per cluster
    n_anomaly_per_cluster: np.ndarray | None = None  # (K,)

    @classmethod
    def fit(
        cls,
        contexts: np.ndarray,
        scores: np.ndarray,
        *,
        n_clusters: int = 4,
        seed: int = 0,
        shrinkage: float = 0.0,
    ) -> PerClusterThresholds:
        """Fit per-cluster percentile thresholds, optionally shrunk to global.

        With ``shrinkage > 0`` each cluster's p95/p99 is pulled toward the
        global (pooled) percentile by an empirical-Bayes weight
        ``w = n_k / (n_k + shrinkage)``: small / noisy clusters borrow strength
        from the stable global boundary, large clusters keep their own.  This
        stops a single small cluster's mis-fit percentile from blowing the
        held-out healthy alert rate to 0.7+ (observed on the conditional V3's
        acoustic arm) while preserving the context-conditional design.  An
        empty cluster falls back to the global percentile (not +inf, which
        would silently suppress all alerts in that regime at inference).
        """
        contexts = np.asarray(contexts, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if contexts.ndim != 2:
            raise ValueError(f"contexts must be 2-D; got {contexts.shape}")
        if contexts.shape[0] != scores.shape[0]:
            raise ValueError("contexts and scores must agree on the first dim")
        if contexts.shape[0] < n_clusters:
            raise ValueError(
                f"need at least {n_clusters} healthy samples to fit thresholds; got "
                f"{contexts.shape[0]}"
            )
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(contexts)
        cluster_ids = km.labels_
        global_p95 = float(np.percentile(scores, 95))
        global_p99 = float(np.percentile(scores, 99))
        p95 = np.zeros(n_clusters, dtype=np.float64)
        p99 = np.zeros(n_clusters, dtype=np.float64)
        n = np.zeros(n_clusters, dtype=np.int64)
        for k in range(n_clusters):
            mask = cluster_ids == k
            n[k] = int(mask.sum())
            if n[k] == 0:
                p95[k] = global_p95
                p99[k] = global_p99
                continue
            bucket = scores[mask]
            raw95 = float(np.percentile(bucket, 95))
            raw99 = float(np.percentile(bucket, 99))
            if shrinkage > 0:
                w = n[k] / (n[k] + float(shrinkage))
                p95[k] = w * raw95 + (1.0 - w) * global_p95
                p99[k] = w * raw99 + (1.0 - w) * global_p99
            else:
                p95[k] = raw95
                p99[k] = raw99
        return cls(
            centroids=km.cluster_centers_.astype(np.float64),
            p95=p95,
            p99=p99,
            n_per_cluster=n,
            seed=seed,
        )

    def assign(self, contexts: np.ndarray) -> np.ndarray:
        contexts = np.asarray(contexts, dtype=np.float64)
        # Pairwise Euclidean distance via the (a-b)^2 = a^2 - 2ab + b^2 expansion
        # to avoid materialising a (N, K, D) tensor.
        a2 = (contexts ** 2).sum(axis=1, keepdims=True)  # (N, 1)
        b2 = (self.centroids ** 2).sum(axis=1)[None, :]  # (1, K)
        ab = contexts @ self.centroids.T  # (N, K)
        d2 = a2 - 2.0 * ab + b2
        return d2.argmin(axis=1)

    def threshold_for(
        self, contexts: np.ndarray, percentile: int | str = 99
    ) -> tuple[np.ndarray, np.ndarray]:
        clusters = self.assign(contexts)
        if percentile == 95:
            thresh = self.p95
        elif percentile == 99:
            thresh = self.p99
        elif percentile == "calibrated":
            if self.p_calibrated is None:
                raise ValueError(
                    "percentile='calibrated' requires "
                    "calibrate_against_anomalies(...) to have been run first"
                )
            thresh = self.p_calibrated
        else:
            raise ValueError("percentile must be 95, 99, or 'calibrated'")
        return thresh[clusters], clusters

    def alert(
        self,
        contexts: np.ndarray,
        scores: np.ndarray,
        percentile: int | str = 99,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-window alert mask.  `percentile` is one of {95, 99, 'calibrated'}.

        'calibrated' uses the supervised-threshold tier produced by
        `calibrate_against_anomalies`; it raises if the calibration step
        was not run.
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        thresh, clusters = self.threshold_for(contexts, percentile)
        return scores > thresh, clusters

    # ------------------------------------------------------------------
    # Supervised threshold calibration (D2 RF + D3 hit as positive set)
    # ------------------------------------------------------------------

    def calibrate_against_anomalies(
        self,
        anomaly_contexts: np.ndarray,
        anomaly_scores: np.ndarray,
        healthy_scores_for_calibration: np.ndarray | None = None,
        *,
        target: str = "youden_j",
    ) -> PerClusterThresholds:
        """**Analysis helper — not used for headline thresholds.**

        Fits a per-cluster threshold that maximises Youden's J on a "known
        anomaly" cohort in-place.  The headline thresholds in the chained
        system are the unsupervised `p95` / `p99` ones; the calibration
        path here exists for sensitivity analysis and as a reference for
        the per-cluster threshold the model could pick *if* per-window
        anomaly labels were available.

        Important caveat: the field-collection protocol does **not**
        annotate per-window anomaly status, only recording-level
        containers.  This helper assumes every window in
        `anomaly_contexts` is positive (which is overoptimistic on D2 RF
        and D3 hit, and badly wrong on D4 RF where most windows are
        healthy).  Reading the `p_calibrated` numbers as ground-truth-
        optimal thresholds would therefore be incorrect — they are
        upper-bound references, useful for ablation and to estimate how
        much performance is left on the table by going fully unsupervised.
        """
        if target != "youden_j":
            raise ValueError("only target='youden_j' is implemented for now")
        ac = np.asarray(anomaly_contexts, dtype=np.float64)
        a_scores = np.asarray(anomaly_scores, dtype=np.float64).ravel()
        if ac.shape[0] != a_scores.shape[0]:
            raise ValueError("anomaly_contexts and anomaly_scores length mismatch")

        K = self.centroids.shape[0]
        a_clusters = self.assign(ac)
        # For the healthy reference, regenerate per-cluster scores from the
        # p95/p99 fit if no explicit reference is supplied.  Without an
        # explicit set we fall back to the p95 fit's cluster boundaries —
        # this is documented as the default.
        if healthy_scores_for_calibration is None:
            # Use the p95 percentile as a synthetic "1 healthy point per
            # cluster at the high tail" — Youden's J then collapses to
            # "the lowest anomaly score in each cluster that exceeds p95".
            calibrated = np.zeros(K, dtype=np.float64)
            youden = np.zeros(K, dtype=np.float64)
            n_anom = np.zeros(K, dtype=np.int64)
            for k in range(K):
                mask = a_clusters == k
                n_anom[k] = int(mask.sum())
                if n_anom[k] == 0:
                    calibrated[k] = self.p99[k]
                    continue
                cluster_anomaly = a_scores[mask]
                # Threshold = the score at which TPR_k - FPR_k is maximal,
                # using the p95 as our healthy-tail proxy.  Sweep over
                # candidate thresholds drawn from the union of (anomaly,
                # p95) scores.
                candidates = np.sort(
                    np.concatenate([cluster_anomaly, [self.p95[k], self.p99[k]]])
                )
                best_j = -1.0
                best_t = self.p99[k]
                fpr_at_p95 = 0.05  # by construction of the p95 fit
                for t in candidates:
                    tpr_k = float((cluster_anomaly > t).mean())
                    # Healthy FPR proxy: linear interpolation between p95
                    # (5 % FPR) and p99 (1 % FPR) based on threshold value.
                    if t <= self.p95[k]:
                        fpr_k = fpr_at_p95
                    elif t >= self.p99[k]:
                        fpr_k = 0.01
                    else:
                        # interpolate
                        span = max(self.p99[k] - self.p95[k], 1e-9)
                        frac = (t - self.p95[k]) / span
                        fpr_k = fpr_at_p95 - frac * (fpr_at_p95 - 0.01)
                    j = tpr_k - fpr_k
                    if j > best_j:
                        best_j = j
                        best_t = float(t)
                calibrated[k] = best_t
                youden[k] = best_j
            self.p_calibrated = calibrated
            self.calibration_youden_j = youden
            self.n_anomaly_per_cluster = n_anom
            return self

        # With an explicit healthy reference, do the proper Youden's J sweep
        # over candidate thresholds drawn from both populations.
        h_scores = np.asarray(healthy_scores_for_calibration, dtype=np.float64).ravel()
        if "healthy_clusters" not in dir(self):
            pass  # we don't store healthy clusters — caller can add later
        h_clusters_attr = getattr(self, "_healthy_clusters_calibration", None)
        if h_clusters_attr is None:
            # If we don't have healthy cluster IDs, recompute on the fly via centroid lookup.
            # This requires the original healthy contexts which we don't have — so we fall
            # back to the proxy path above.
            return self.calibrate_against_anomalies(
                anomaly_contexts, anomaly_scores, target="youden_j"
            )

        calibrated = np.zeros(K, dtype=np.float64)
        youden = np.zeros(K, dtype=np.float64)
        n_anom = np.zeros(K, dtype=np.int64)
        for k in range(K):
            am = a_clusters == k
            hm = h_clusters_attr == k
            n_anom[k] = int(am.sum())
            if n_anom[k] == 0 or hm.sum() == 0:
                calibrated[k] = self.p99[k]
                continue
            anom_k = a_scores[am]
            heal_k = h_scores[hm]
            candidates = np.sort(np.concatenate([anom_k, heal_k]))
            best_j = -1.0
            best_t = self.p99[k]
            for t in candidates:
                tpr = float((anom_k > t).mean())
                fpr = float((heal_k > t).mean())
                j = tpr - fpr
                if j > best_j:
                    best_j = j
                    best_t = float(t)
            calibrated[k] = best_t
            youden[k] = best_j
        self.p_calibrated = calibrated
        self.calibration_youden_j = youden
        self.n_anomaly_per_cluster = n_anom
        return self


def per_cluster_alert_breakdown(
    thresholds: PerClusterThresholds,
    contexts: np.ndarray,
    scores: np.ndarray,
    *,
    percentile: int | str = 95,
    label_per_cluster: dict[int, str] | None = None,
) -> dict:
    """Break a cohort's alert rate down by predicted V2 K-means cluster.

    The orchestrator's V3 cohort-validation block reports an aggregate
    alert rate per cohort (healthy hold-out, D2 RF, D3 hit, D4 RF).
    For the **per-mode FPR breakdown** (Chapter 6), each cohort's windows
    are additionally split by which K-means cluster of `c_t` they were
    assigned to.  This answers:

      * "Does the healthy hold-out alert rate hit the 5 % target *in
        every cluster*, or only in the cluster K-means chose as
        biggest?"  (per-cluster healthy alert rate near 5 % is the
        publishable claim that the K = 3 mode hypothesis defends.)

      * "Are the D2 RandomFault alerts concentrated in the cluster
        that maps to the Turbine mode (the regime D2 RF was actually
        recorded in)?"  (this is the cluster-correctness check the
        chained-system narrative depends on.)

    The breakdown is unsupervised in cluster identity (the K-means
    cluster IDs are arbitrary integers).  When ``label_per_cluster`` is
    supplied (typically derived from a Hungarian match in Chapter 6),
    each row is also tagged with the mode name for the result tables.

    Returns a JSON-friendly dict:

    .. code::

        {
          "percentile": 95,
          "n_total": int,
          "n_alerts_total": int,
          "alert_rate_total": float,
          "per_cluster": {
            "0": {"n": ..., "n_alerts": ..., "alert_rate": ..., "label": ...},
            "1": {...},
            ...
          }
        }
    """
    contexts = np.asarray(contexts, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if contexts.shape[0] != scores.shape[0]:
        raise ValueError("contexts and scores must have the same first dim")
    alerts, clusters = thresholds.alert(contexts, scores, percentile=percentile)

    K = int(thresholds.centroids.shape[0])
    per_cluster: dict[str, dict] = {}
    for k in range(K):
        mask = clusters == k
        n_k = int(mask.sum())
        n_alerts_k = int(alerts[mask].sum()) if n_k > 0 else 0
        per_cluster[str(int(k))] = {
            "n": n_k,
            "n_alerts": n_alerts_k,
            "alert_rate": float(n_alerts_k / n_k) if n_k > 0 else float("nan"),
            "label": (label_per_cluster or {}).get(int(k), None),
        }
    n_total = int(scores.shape[0])
    n_alerts_total = int(alerts.sum()) if n_total > 0 else 0
    return {
        "percentile": percentile if isinstance(percentile, str) else int(percentile),
        "n_total": n_total,
        "n_alerts_total": n_alerts_total,
        "alert_rate_total": float(n_alerts_total / n_total) if n_total > 0 else float("nan"),
        "per_cluster": per_cluster,
    }


__all__ = ["PerClusterThresholds", "per_cluster_alert_breakdown"]
