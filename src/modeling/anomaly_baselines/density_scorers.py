"""V0 density / clustering anomaly scorers (Khamaisi reference, completed).

Khamaisi et al. (2025) benchmarked three unsupervised acoustic anomaly models
at ROW~II — an LSTM autoencoder, K-means, and a One-Class SVM — fitted on
healthy data and scored by how far a window falls from the healthy manifold.
The LSTM-AE lives in :mod:`lstm_ae`; this module supplies the remaining
"classical" scorers so the full prior-work baseline can be reproduced
apples-to-apples against the proposed conditional head:

  * :class:`KMeansDistanceScorer` — distance to the nearest healthy centroid
    (Khamaisi's K-means model, used here as an anomaly scorer rather than a
    clusterer).
  * :class:`OneClassSVMScorer` — signed distance to the One-Class SVM boundary
    (Khamaisi's OC-SVM model; the one that hit ROC~AUC ≈ 0.998 steady-state).
  * :class:`KDEScorer` — negative log-density under a Gaussian kernel-density
    estimate on PCA-whitened features (the density baseline named in the
    Experiments chapter).

All three consume a single **flat per-window feature vector** rather than the
``(T, F)`` sequence the LSTM-AE ingests, so this module also owns the shared
feature extractor :func:`extract_window_features`.  The features are derived
from the *same* sliding windows that :mod:`lstm_ae` produces (time-mean and
time-std of each log-mel band, or of each vibration channel), so a window seen
by the AE and by these scorers is the identical slice of signal — only the
scoring mechanism differs.

Score orientation is uniform: **higher = more anomalous**, so a scorer plugs
directly into :class:`src.modeling.anomaly.threshold.PerClusterThresholds`,
whose ``alert`` rule is ``score > threshold``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ...ingestion.test_dataset_loader import TestDatasetSegment
from .lstm_ae import (
    V0Config,
    extract_log_mel_windows,
    extract_vibration_temporal_windows,
)

# ---------------------------------------------------------------------------
# Shared per-window flat feature extractor
# ---------------------------------------------------------------------------

#: Modalities the V0 baselines understand.
Modality = str  # "acoustic" | "vibration"


def _aggregate_windows(windows: np.ndarray) -> np.ndarray:
    """Collapse ``(n_windows, T, F)`` sequence windows to ``(n_windows, 2F)``.

    Each window is summarised by the per-feature time-mean and time-std, the
    standard compact embedding used by classical (non-recurrent) acoustic
    anomaly baselines.  Concatenating both moments keeps a little of the
    within-window dynamics that a bare mean would discard, while staying low
    enough in dimension for a kernel-density estimate to be well-posed.
    """
    if windows.shape[0] == 0:
        f = windows.shape[2] if windows.ndim == 3 else 0
        return np.zeros((0, 2 * f), dtype=np.float32)
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    return np.concatenate([mean, std], axis=1).astype(np.float32)


def extract_window_features(
    segment: TestDatasetSegment,
    cfg: V0Config,
    modality: Modality = "acoustic",
    *,
    pool: str = "mean",
) -> np.ndarray:
    """Flat per-window feature matrix for the density / clustering scorers.

    Returns ``(n_windows, D)`` float32, where ``D = 2 * n_mels`` for the
    acoustic modality and ``D = 2 * 3 = 6`` for the vibration modality.  The
    windows are exactly those :mod:`lstm_ae` slides over the same segment, so
    the comparison against the LSTM-AE isolates the scoring mechanism.
    """
    if modality == "acoustic":
        windows = extract_log_mel_windows(segment, cfg, pool=pool)
    elif modality == "vibration":
        windows = extract_vibration_temporal_windows(segment, cfg, pool=pool)
    else:
        raise ValueError(f"unknown modality {modality!r}")
    return _aggregate_windows(windows)


def gather_features(
    segments: Iterable[TestDatasetSegment],
    cfg: V0Config,
    modality: Modality,
    *,
    healthy_only: bool,
) -> tuple[np.ndarray, list[str]]:
    """Stack per-window features across segments, returning the rec-id per row.

    ``healthy_only`` mirrors :func:`lstm_ae._gather_healthy_windows`: when set,
    segments flagged ``is_anomaly`` are skipped so the model is fitted on
    healthy material only.
    """
    feats: list[np.ndarray] = []
    rec_ids: list[str] = []
    dim = 0
    for s in segments:
        if healthy_only and s.is_anomaly:
            continue
        f = extract_window_features(s, cfg, modality)
        if f.shape[0] == 0:
            continue
        dim = f.shape[1]
        feats.append(f)
        rec_ids.extend([s.recording_id] * f.shape[0])
    if not feats:
        return np.zeros((0, dim), dtype=np.float32), []
    return np.concatenate(feats, axis=0), rec_ids


# ---------------------------------------------------------------------------
# Scorer protocol + implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class AnomalyScorer(Protocol):
    """A fitted-on-healthy scorer with ``higher == more anomalous`` output."""

    name: str

    def fit(self, x: np.ndarray) -> AnomalyScorer: ...

    def score(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class KMeansDistanceScorer:
    """Distance to the nearest healthy K-means centroid (Khamaisi K-means).

    Features are standardised, K-means is fitted on the healthy training set,
    and the anomaly score of a window is the Euclidean distance to its nearest
    centroid: windows that look like healthy operation sit near a centroid,
    anomalies fall far from every healthy prototype.
    """

    n_clusters: int = 8
    seed: int = 42
    name: str = "kmeans"

    def __post_init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._km: KMeans | None = None

    def fit(self, x: np.ndarray) -> KMeansDistanceScorer:
        if x.shape[0] == 0:
            raise ValueError("KMeansDistanceScorer.fit got an empty feature matrix")
        k = max(1, min(self.n_clusters, x.shape[0]))
        self._scaler = StandardScaler().fit(x)
        xs = self._scaler.transform(x)
        self._km = KMeans(n_clusters=k, random_state=self.seed, n_init=10).fit(xs)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self._km is None or self._scaler is None:
            raise RuntimeError("KMeansDistanceScorer.score called before fit")
        xs = self._scaler.transform(x)
        # transform() returns per-centroid distances; the score is the min.
        return self._km.transform(xs).min(axis=1).astype(np.float64)


@dataclass
class OneClassSVMScorer:
    """Signed distance to a One-Class SVM boundary (Khamaisi OC-SVM).

    ``decision_function`` is positive for inliers and negative for outliers, so
    the anomaly score is its negation.  The RBF ``gamma="scale"`` default tracks
    the feature variance, and ``nu`` bounds the healthy training-set outlier
    fraction (left at the conventional 0.05, matching the 5 % calibration
    target the proposed head is held to).
    """

    nu: float = 0.05
    gamma: str | float = "scale"
    kernel: str = "rbf"
    name: str = "ocsvm"

    def __post_init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._svm: OneClassSVM | None = None

    def fit(self, x: np.ndarray) -> OneClassSVMScorer:
        if x.shape[0] == 0:
            raise ValueError("OneClassSVMScorer.fit got an empty feature matrix")
        self._scaler = StandardScaler().fit(x)
        xs = self._scaler.transform(x)
        self._svm = OneClassSVM(nu=self.nu, gamma=self.gamma, kernel=self.kernel).fit(xs)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self._svm is None or self._scaler is None:
            raise RuntimeError("OneClassSVMScorer.score called before fit")
        xs = self._scaler.transform(x)
        # decision_function: >0 inlier, <0 outlier → negate for "higher = worse".
        return (-self._svm.decision_function(xs)).astype(np.float64)


@dataclass
class KDEScorer:
    """Negative log-density under a Gaussian KDE on PCA-whitened features.

    A raw kernel-density estimate is numerically indefensible at the native
    feature dimension (``2 * n_mels`` with only hundreds of healthy windows;
    see the note in ``src/modeling/anomaly/kde_baseline.py``).  We therefore
    whiten with PCA down to ``n_components`` decorrelated, unit-variance axes
    first, which makes the isotropic Gaussian kernel well-posed and the
    Scott's-rule bandwidth meaningful.  The anomaly score is the negative
    log-density: rare windows have low density and thus a high score.
    """

    n_components: int = 16
    bandwidth: float | str = "scott"
    seed: int = 42
    name: str = "kde"

    def __post_init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._kde: KernelDensity | None = None

    def _resolve_bandwidth(self, n: int, d: int) -> float:
        if isinstance(self.bandwidth, (int, float)):
            return float(self.bandwidth)
        if self.bandwidth == "scott":
            # On unit-variance (whitened) data Scott's factor is n^(-1/(d+4)).
            return float(n ** (-1.0 / (d + 4)))
        raise ValueError(f"unknown bandwidth {self.bandwidth!r}")

    def fit(self, x: np.ndarray) -> KDEScorer:
        if x.shape[0] == 0:
            raise ValueError("KDEScorer.fit got an empty feature matrix")
        self._scaler = StandardScaler().fit(x)
        xs = self._scaler.transform(x)
        n_comp = max(1, min(self.n_components, xs.shape[1], xs.shape[0] - 1))
        self._pca = PCA(n_components=n_comp, whiten=True, random_state=self.seed).fit(xs)
        xw = self._pca.transform(xs)
        bw = self._resolve_bandwidth(xw.shape[0], xw.shape[1])
        self._kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(xw)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self._kde is None or self._pca is None or self._scaler is None:
            raise RuntimeError("KDEScorer.score called before fit")
        xw = self._pca.transform(self._scaler.transform(x))
        return (-self._kde.score_samples(xw)).astype(np.float64)


#: Factory for the classical V0 scorers (the LSTM-AE has its own trainer).
def build_scorer(name: str, *, seed: int = 42) -> AnomalyScorer:
    """Instantiate a classical V0 scorer by name (``kmeans``/``ocsvm``/``kde``)."""
    if name == "kmeans":
        return KMeansDistanceScorer(seed=seed)
    if name == "ocsvm":
        return OneClassSVMScorer()
    if name == "kde":
        return KDEScorer(seed=seed)
    raise ValueError(
        f"unknown classical scorer {name!r}; expected one of kmeans/ocsvm/kde"
    )


__all__ = [
    "AnomalyScorer",
    "KDEScorer",
    "KMeansDistanceScorer",
    "OneClassSVMScorer",
    "build_scorer",
    "extract_window_features",
    "gather_features",
]
