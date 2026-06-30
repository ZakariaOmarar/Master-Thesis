"""Cluster-evaluation metrics for the V1 / V2 sanity gate.

The RQ1 sanity gate (V1) and headline metric (V2) report three
measures, each with different behaviour when ``n_clusters`` differs from
``n_labels`` — the typical case when K-means is forced to K = 3 (the
mode hypothesis) but the held-out validation set contains only some
subset of modes:

  1. **Hungarian-matched cluster purity** [Manning, Raghavan & Schütze,
     *Introduction to Information Retrieval*, 2008 §16.3].  The
     classical purity score after a Hungarian assignment of cluster
     indices to label indices.  Sensitive to the K_cluster ≠ K_label
     asymmetry: a surplus cluster always orphans some samples and lowers
     purity in a way that does not reflect clustering quality.

  2. **Normalised Mutual Information (NMI)** [Strehl & Ghosh,
     *Cluster Ensembles*, JMLR 2002].  Symmetric, normalised to [0, 1],
     and *invariant to cluster cardinality* — the canonical fix for the
     K_cluster ≠ K_labels mismatch.

  3. **Adjusted Rand Index (ARI)** [Hubert & Arabie, "Comparing
     Partitions", *J. Classification* 1985].  Chance-corrected partition
     agreement; also cardinality-invariant.  Complements NMI by
     measuring pair-counting agreement rather than mutual information.

The Chapter 6 headline metric is **NMI** with **ARI** as a secondary
robust check; purity is retained for direct comparison against papers
that report it (e.g. Khamaisi et al.) and as a sanity-gate floor.

This is **evaluation only**.  Mode labels do not enter the V1 / V2
training loops; they only appear here, and the K-means + Hungarian step
makes that explicit (cluster IDs are arbitrary integers up until the
matching step).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_purity(
    cluster_idx: np.ndarray,
    label_idx: np.ndarray,
    n_clusters: int,
    n_labels: int,
) -> tuple[float, dict[int, int], np.ndarray]:
    """Hungarian-matched cluster purity.

    Returns
    -------
    purity : float in [0, 1]
        Sum over predicted clusters of the count of the most-common label in
        that cluster, divided by total samples — after Hungarian assignment of
        cluster indices to label indices.
    mapping : dict[cluster_idx, label_idx]
        The optimal cluster → label assignment.
    confusion : (n_clusters, n_labels) int array
        Pre-mapping confusion matrix, rows = clusters, cols = labels.
    """
    cluster_idx = np.asarray(cluster_idx, dtype=np.int64)
    label_idx = np.asarray(label_idx, dtype=np.int64)
    if cluster_idx.shape != label_idx.shape:
        raise ValueError(
            f"cluster_idx and label_idx must have the same shape; "
            f"got {cluster_idx.shape} and {label_idx.shape}"
        )
    n = int(cluster_idx.shape[0])
    if n == 0:
        return 0.0, {}, np.zeros((n_clusters, n_labels), dtype=np.int64)

    confusion = np.zeros((n_clusters, n_labels), dtype=np.int64)
    for c, y in zip(cluster_idx, label_idx):
        confusion[int(c), int(y)] += 1

    cost = -confusion.astype(np.int64)
    if n_clusters >= n_labels:
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        # linear_sum_assignment requires the rect matrix to have rows ≤ cols
        row_ind, col_ind = linear_sum_assignment(cost.T)
        row_ind, col_ind = col_ind, row_ind

    matched = int(confusion[row_ind, col_ind].sum())
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    return float(matched / n), mapping, confusion


def _normalised_mutual_information(
    cluster_idx: np.ndarray, label_idx: np.ndarray
) -> float:
    """Symmetric normalised mutual information (NMI), matching sklearn's default
    'arithmetic' average."""
    cluster_idx = np.asarray(cluster_idx, dtype=np.int64)
    label_idx = np.asarray(label_idx, dtype=np.int64)
    n = int(cluster_idx.shape[0])
    if n == 0:
        return 0.0

    cu = np.unique(cluster_idx)
    yu = np.unique(label_idx)
    contingency = np.zeros((cu.size, yu.size), dtype=np.float64)
    cu_idx = {int(v): i for i, v in enumerate(cu)}
    yu_idx = {int(v): i for i, v in enumerate(yu)}
    for c, y in zip(cluster_idx, label_idx):
        contingency[cu_idx[int(c)], yu_idx[int(y)]] += 1.0

    p_xy = contingency / n
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log(np.where(p_xy > 0, p_xy / (p_x @ p_y + 1e-12), 1.0))
    mi = float(np.where(p_xy > 0, p_xy * log_term, 0.0).sum())

    h_x = float(-(p_x * np.log(np.where(p_x > 0, p_x, 1.0))).sum())
    h_y = float(-(p_y * np.log(np.where(p_y > 0, p_y, 1.0))).sum())
    denom = 0.5 * (h_x + h_y)
    if denom <= 0:
        return 0.0
    # Clamp to [0, 1] — float-precision drift can push NMI marginally
    # negative when the joint distribution is near-independent.
    return max(0.0, min(1.0, mi / denom))


def _adjusted_rand_index(
    cluster_idx: np.ndarray, label_idx: np.ndarray
) -> float:
    """Hubert & Arabie's chance-corrected Rand Index.

    ARI ∈ [-1, 1]; 0 = chance, 1 = perfect agreement.  Cardinality-
    invariant and label-permutation-invariant — strictly more
    informative than purity when ``n_clusters ≠ n_labels``.

    Implementation matches Hubert & Arabie (1985) Eq. 4 and is
    numerically identical to ``sklearn.metrics.adjusted_rand_score``;
    written out here so the cluster-metric module has no run-time
    dependency on sklearn beyond the K-means clusterer.
    """
    from math import comb

    cluster_idx = np.asarray(cluster_idx, dtype=np.int64)
    label_idx = np.asarray(label_idx, dtype=np.int64)
    n = int(cluster_idx.shape[0])
    if n < 2:
        return 0.0

    cu = np.unique(cluster_idx)
    yu = np.unique(label_idx)
    contingency = np.zeros((cu.size, yu.size), dtype=np.int64)
    cu_idx = {int(v): i for i, v in enumerate(cu)}
    yu_idx = {int(v): i for i, v in enumerate(yu)}
    for c, y in zip(cluster_idx, label_idx):
        contingency[cu_idx[int(c)], yu_idx[int(y)]] += 1

    sum_nij_C2 = sum(int(comb(int(nij), 2)) for nij in contingency.flatten())
    a = contingency.sum(axis=1)
    b = contingency.sum(axis=0)
    sum_a_C2 = sum(int(comb(int(ai), 2)) for ai in a)
    sum_b_C2 = sum(int(comb(int(bj), 2)) for bj in b)
    total_C2 = comb(n, 2)
    if total_C2 == 0:
        return 0.0
    expected_index = sum_a_C2 * sum_b_C2 / total_C2
    max_index = 0.5 * (sum_a_C2 + sum_b_C2)
    if max_index == expected_index:
        return 0.0
    return float((sum_nij_C2 - expected_index) / (max_index - expected_index))


def cluster_purity_and_nmi(
    embeddings: np.ndarray,
    labels: list[str],
    n_clusters: int = 4,
    seed: int = 42,
) -> dict:
    """K-means(k=n_clusters) on embeddings, scored against ``labels``.

    Returns three metrics — `purity`, `nmi`, `ari` — plus the K-means
    cluster assignment, the Hungarian mapping (for reporting), and the
    pre-mapping confusion matrix.  See the module docstring for the
    rationale and citations: NMI is the Chapter 6 headline, ARI is the
    chance-corrected secondary check, purity is retained for direct
    comparison with prior work (Khamaisi et al.).
    """
    from sklearn.cluster import KMeans

    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (N, D); got {embeddings.shape}")
    if embeddings.shape[0] != len(labels):
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows but {len(labels)} labels"
        )

    label_set = sorted(set(labels))
    label_to_idx = {y: i for i, y in enumerate(label_set)}
    label_idx = np.array([label_to_idx[y] for y in labels], dtype=np.int64)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_idx = km.fit_predict(embeddings)

    purity, mapping, confusion = hungarian_purity(
        cluster_idx, label_idx, n_clusters=n_clusters, n_labels=len(label_set)
    )
    nmi = _normalised_mutual_information(cluster_idx, label_idx)
    ari = _adjusted_rand_index(cluster_idx, label_idx)

    # Collapse diagnostics: an NMI of exactly 0 almost always means K-means
    # found a single populated cluster because the embeddings collapsed to
    # (near-)identical vectors — not "poor but real" clustering.  Surface that
    # explicitly so a future collapse is diagnosable rather than mistaken for a
    # weak-but-valid score.  `collapsed` is True when only one cluster is
    # populated OR the embedding spread is numerically negligible.
    n_effective_clusters = int(np.unique(cluster_idx).size)
    embedding_std = float(embeddings.std())
    return {
        "purity": purity,
        "nmi": nmi,
        "ari": ari,
        "n_clusters": n_clusters,
        "n_labels": len(label_set),
        "label_set": tuple(label_set),
        "mapping": mapping,
        "confusion": confusion,
        "cluster_idx": cluster_idx.astype(np.int64),
        "n_effective_clusters": n_effective_clusters,
        "embedding_std": embedding_std,
        "collapsed": bool(n_effective_clusters <= 1 or embedding_std < 1e-6),
    }


def cluster_purity_per_dataset(
    embeddings: np.ndarray,
    labels: list[str],
    dataset_ids: list[str],
    *,
    n_clusters: int = 3,
    seed: int = 42,
) -> dict[str, dict]:
    """Per-dataset stratified cluster-purity / NMI / ARI breakdown.

    The aggregate RQ1 number hides dataset-specific behaviour: a V2
    encoder might win decisively on D2 (5+5 sensor array) while
    losing on D1 (4+4 array).  Chapter 6 reports the per-dataset
    rows so the reviewer sees this directly.

    Important methodological note: the K-means + Hungarian fit is run
    **independently on each dataset's subset of windows**.  This means
    each dataset gets its own cluster boundaries and its own
    Hungarian mapping — the "wins" are not directly comparable in
    absolute purity / NMI value across datasets (e.g. a 0.80 NMI on
    D1's 100 windows is not the same as a 0.80 NMI on D2's 500
    windows), but the *direction* of the V2-vs-A1 ablation gap on
    each dataset is interpretable in isolation.  When the
    ``label_set`` differs across datasets (e.g. D1 has only
    Pump+Turbine while D2 has Pump+Standstill+Turbine), this is
    surfaced in the per-dataset `label_set` field.

    Returns a mapping ``dataset_id → {purity, nmi, ari, ...}``.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if len(labels) != len(dataset_ids):
        raise ValueError(
            f"labels ({len(labels)}) and dataset_ids ({len(dataset_ids)}) "
            "must agree"
        )
    if embeddings.shape[0] != len(labels):
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows but {len(labels)} labels"
        )
    out: dict[str, dict] = {}
    for did in sorted(set(dataset_ids)):
        mask = np.array([did_i == did for did_i in dataset_ids], dtype=bool)
        sub_emb = embeddings[mask]
        sub_lab = [labels[i] for i in range(len(labels)) if mask[i]]
        if len(set(sub_lab)) < 2 or sub_emb.shape[0] < n_clusters:
            # Cannot evaluate a clustering against a single-label set or
            # fewer windows than clusters — report the contents only.
            out[did] = {
                "purity": float("nan"),
                "nmi": float("nan"),
                "ari": float("nan"),
                "n_clusters": n_clusters,
                "n_labels": len(set(sub_lab)),
                "label_set": tuple(sorted(set(sub_lab))),
                "n_windows": int(sub_emb.shape[0]),
                "evaluable": False,
            }
            continue
        row = cluster_purity_and_nmi(
            sub_emb, sub_lab, n_clusters=n_clusters, seed=seed
        )
        row["n_windows"] = int(sub_emb.shape[0])
        row["evaluable"] = True
        out[did] = row
    return out


__all__ = [
    "cluster_purity_and_nmi",
    "cluster_purity_per_dataset",
    "hungarian_purity",
]
