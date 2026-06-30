"""V0 LightGBM mode classifier on hand-engineered features.

Produces the **upper-bound reference row** for the RQ1 cluster-purity table:
how well could a classifier do if it *were* allowed to use mode labels?  V1's
per-modality SSL and V2's full multimodal SSL both target this number from
below without using labels at training time.

Per-window features:
  Acoustic (mean-pooled across mics):
    - RMS
    - kurtosis
    - spectral centroid
    - n_mels log-mel band means (default n_mels=64)
  Vibration (mean-pooled across vibration channels):
    - amplitude RMS, kurtosis, mean, std

Training:
  - 4-class: {Pump, Standstill, Turbine, RandomFault} on D1+D2 folder labels.
  - Held-out *recordings* (not held-out windows), matching the sanity-gate
    protocol.  No window-level leakage.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch  # noqa: F401  # keeps deterministic seeding consistent with lstm_ae.py
from scipy.stats import kurtosis as _kurtosis

from ...features.audio_spectral import compute_log_mel_spectrogram
from ...ingestion.test_dataset_loader import (
    TestDatasetLoader,
    TestDatasetSegment,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V0ModeConfig:
    """Hyperparameters for the V0 LightGBM mode classifier.

    target_classes restricted to the three healthy operating modes —
    `RandomFault` is anomaly status, not a mode (separate output of the
    chained system, handled by V3).  Including it here at training time
    would force the LightGBM model to learn the mode-vs-anomaly axis
    simultaneously and conflate the two RQ1 / RQ2 outputs.

    The V0 trainer adapts `num_class` to whichever subset of these three
    modes is actually present in the campaign at hand (D1: Pump + Turbine,
    D2: all three).  Single-mode campaigns are skipped with a clear log.
    """

    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    window_seconds: float = 1.0
    window_overlap: float = 0.5
    target_classes: tuple[str, ...] = (
        "Pump",
        "Standstill",
        "Turbine",
    )
    val_ratio: float = 0.5
    seed: int = 42
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 5  # tiny dataset; loosen the LightGBM defaults
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_mode_features(
    segment: TestDatasetSegment, cfg: V0ModeConfig
) -> tuple[np.ndarray, list[str]]:
    """Compute per-window hand-engineered features for one segment.

    Returns
    -------
    features : (n_windows, n_features) float32
    feature_names : list of human-readable names, length n_features
    """
    fs = int(segment.segment.mic_sample_rate)
    # V0 baselines deliberately use `cfg.hop_length=512` (coarse 31.25 Hz
    # acoustic frame rate) for fast baseline computation.  They do not go
    # through V2's cross-attention, so the registry's per-dataset hop
    # (`hop_for_dataset` in v2_ssl) does not apply.
    hop = cfg.hop_length

    # --- Acoustic side: per-mic log-mel + RMS-style stats, then mean over mics.
    mel_per_mic: list[np.ndarray] = []
    rms_per_mic: list[float] = []
    kurt_per_mic: list[float] = []
    centroid_per_mic: list[float] = []
    raw_mics = segment.segment.mic_data

    for ch in range(segment.segment.n_mic_channels):
        x = raw_mics[ch].astype(np.float64)
        rms_per_mic.append(float(np.sqrt(np.mean(x * x) + 1e-12)))
        kurt_per_mic.append(float(_kurtosis(x, fisher=True, bias=True)))
        # Spectral centroid via FFT magnitude
        mag = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / fs)
        centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))
        centroid_per_mic.append(centroid)
        mel_per_mic.append(
            compute_log_mel_spectrogram(
                x.astype(np.float32),
                fs=fs,
                n_fft=cfg.n_fft,
                hop_length=hop,
                n_mels=cfg.n_mels,
            )
        )

    # mel_pool: (n_mels, n_frames) — mean over mics
    mel_pool = np.stack(mel_per_mic, axis=0).mean(axis=0)
    mic_rms = float(np.mean(rms_per_mic))
    mic_kurt = float(np.mean(kurt_per_mic))
    mic_centroid = float(np.mean(centroid_per_mic))

    frames_per_window = max(1, int(round(cfg.window_seconds * fs / hop)))
    step = max(1, int(round(frames_per_window * (1.0 - cfg.window_overlap))))
    n_frames = mel_pool.shape[1]
    if n_frames < frames_per_window:
        return (
            np.zeros((0, cfg.n_mels + 3 + 4), dtype=np.float32),
            _feature_names(cfg.n_mels),
        )

    # --- Vibration side: per-channel stats, mean over channels.  Computed once
    # per segment (vibration is at ~Hz cadence; per-window slicing of vibration
    # would give too few samples to estimate kurtosis reliably).
    vib = segment.segment.accel_data.astype(np.float64)
    vib_rms = float(np.sqrt(np.mean(vib * vib) + 1e-12))
    vib_kurt = float(np.mean([_kurtosis(v, fisher=True, bias=True) for v in vib]))
    vib_mean = float(np.mean(vib))
    vib_std = float(np.std(vib))

    rows: list[np.ndarray] = []
    for start in range(0, n_frames - frames_per_window + 1, step):
        mel_window = mel_pool[:, start : start + frames_per_window]
        mel_means = mel_window.mean(axis=1)  # (n_mels,)
        feats = np.concatenate(
            [
                mel_means,
                np.array([mic_rms, mic_kurt, mic_centroid], dtype=np.float64),
                np.array([vib_rms, vib_kurt, vib_mean, vib_std], dtype=np.float64),
            ]
        ).astype(np.float32)
        rows.append(feats)

    if not rows:
        return (
            np.zeros((0, cfg.n_mels + 3 + 4), dtype=np.float32),
            _feature_names(cfg.n_mels),
        )
    return np.stack(rows, axis=0), _feature_names(cfg.n_mels)


def _feature_names(n_mels: int) -> list[str]:
    return (
        [f"mel_mean_{i}" for i in range(n_mels)]
        + ["mic_rms", "mic_kurtosis", "mic_spectral_centroid"]
        + ["vib_rms", "vib_kurtosis", "vib_mean", "vib_std"]
    )


# ---------------------------------------------------------------------------
# Train / score
# ---------------------------------------------------------------------------


@dataclass
class ModeTrainResult:
    booster: Any  # lightgbm.Booster
    classes: tuple[str, ...]
    feature_names: list[str]
    standardiser_mean: np.ndarray
    standardiser_std: np.ndarray
    train_recording_ids: list[str]
    val_recording_ids: list[str]
    val_macro_f1: float
    val_per_class_f1: dict[str, float]
    val_confusion: np.ndarray  # (n_classes, n_classes)


def _gather_labelled_windows(
    segments: Iterable[TestDatasetSegment], cfg: V0ModeConfig
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Stack windows from labelled recordings; return X, y, fully-qualified
    recording_id_per_window.

    The recording-key is `<source_dir_basename>/<recording_id>` so that
    D1's same-`recording_id`-in-different-folders case (e.g. `All/Pump` vs
    `Pump/Pump`) is disambiguated for the held-out split.
    """
    feature_names: list[str] | None = None
    Xs: list[np.ndarray] = []
    ys: list[int] = []
    recs: list[str] = []
    label_to_idx = {c: i for i, c in enumerate(cfg.target_classes)}
    for s in segments:
        if (s.mode_label or "") not in label_to_idx:
            continue
        feats, names = extract_mode_features(s, cfg)
        if feats.shape[0] == 0:
            continue
        if feature_names is None:
            feature_names = names
        rec_key = f"{Path(s.source_dir).name}/{s.recording_id}"
        Xs.append(feats)
        ys.extend([label_to_idx[s.mode_label]] * feats.shape[0])
        recs.extend([rec_key] * feats.shape[0])
    if not Xs:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            [],
            feature_names or [],
        )
    return (
        np.concatenate(Xs, axis=0),
        np.asarray(ys, dtype=np.int32),
        recs,
        feature_names or _feature_names(cfg.n_mels),
    )


def _split_by_recording(
    rec_ids: list[str], val_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(rec_ids))
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_ratio)))
    val_ids = unique[:n_val]
    train_ids = unique[n_val:]
    if not train_ids:
        train_ids = [val_ids.pop()]
    return train_ids, val_ids


def _stratified_split_by_recording(
    rec_ids: list[str],
    rec_to_label: dict[str, int],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Stratified-by-class held-out split at the recording level.

    Each class contributes ⌊val_ratio · count⌋ recordings (≥ 1 when
    count ≥ 2) to val.  Single-recording classes stay in train so the
    held-out doesn't degenerate to 1-class purity.
    """
    rng = np.random.default_rng(seed)
    unique_recs = sorted(set(rec_ids))
    by_label: dict[int, list[str]] = {}
    for r in unique_recs:
        by_label.setdefault(rec_to_label[r], []).append(r)
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    for _lbl, recs in by_label.items():
        recs_shuffled = list(recs)
        rng.shuffle(recs_shuffled)
        if len(recs_shuffled) <= 1:
            train_ids.update(recs_shuffled)
            continue
        n_val = max(1, int(round(len(recs_shuffled) * val_ratio)))
        n_val = min(n_val, len(recs_shuffled) - 1)  # keep ≥ 1 in train
        val_ids.update(recs_shuffled[:n_val])
        train_ids.update(recs_shuffled[n_val:])
    return sorted(train_ids), sorted(val_ids)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, list[float], np.ndarray]:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    f1_per: list[float] = []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        f1_per.append(f1)
    return float(np.mean(f1_per)), f1_per, cm


def train_v0_mode_lgbm(
    loader: TestDatasetLoader, cfg: V0ModeConfig | None = None
) -> ModeTrainResult:
    """Train the V0 LightGBM mode classifier on labelled recordings.

    Adapts `num_class` to the modes actually present in the campaign at
    hand (D1: Pump + Turbine = 2-class; D2: all three modes = 3-class).
    Recording-level held-out split is **stratified by class** to ensure
    every held-out fold contains at least one recording of each present
    class — without this, val sets routinely degenerated to a single
    class and macro-F1 collapsed to 0.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "lightgbm is required for the V0 mode classifier; install via 'pip install lightgbm'"
        ) from exc

    cfg = cfg or V0ModeConfig()
    np.random.seed(cfg.seed)

    segments = loader.list_segments()
    X, y, rec_ids, feature_names = _gather_labelled_windows(segments, cfg)
    if X.shape[0] == 0:
        raise RuntimeError("no labelled windows found for V0 mode classifier")

    # Adapt to present classes only.  `y` carries indices into
    # `cfg.target_classes`; we remap to a contiguous index space over the
    # subset that actually appears in this campaign.
    present_label_indices = sorted({int(v) for v in y})
    present_classes = tuple(cfg.target_classes[i] for i in present_label_indices)
    if len(present_classes) < 2:
        raise RuntimeError(
            f"V0 LGBM needs ≥ 2 present classes, got {present_classes}"
        )
    old_to_new = {old: new for new, old in enumerate(present_label_indices)}
    y = np.asarray([old_to_new[int(v)] for v in y], dtype=np.int32)

    rec_to_label = {r: int(yi) for r, yi in zip(rec_ids, y)}
    train_ids, val_ids = _stratified_split_by_recording(
        rec_ids, rec_to_label, cfg.val_ratio, cfg.seed
    )
    train_mask = np.array([r in train_ids for r in rec_ids], dtype=bool)
    val_mask = np.array([r in val_ids for r in rec_ids], dtype=bool)

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    # Binary objective when only 2 classes are present (D1).
    if len(present_classes) == 2:
        objective = "binary"
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": cfg.learning_rate,
            "num_leaves": cfg.num_leaves,
            "min_child_samples": cfg.min_child_samples,
            "verbose": -1,
            "seed": cfg.seed,
            "is_unbalance": True,
        }
    else:
        objective = "multiclass"
        params = {
            "objective": "multiclass",
            "num_class": len(present_classes),
            "metric": "multi_logloss",
            "learning_rate": cfg.learning_rate,
            "num_leaves": cfg.num_leaves,
            "min_child_samples": cfg.min_child_samples,
            "verbose": -1,
            "seed": cfg.seed,
            "class_weight": "balanced",
        }
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=cfg.n_estimators,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)],
    )

    if X_val.shape[0] > 0:
        probs = booster.predict(X_val, num_iteration=booster.best_iteration)
        # Binary classifier returns 1-D probabilities (P(class=1)); cast to
        # the multiclass-style argmax for downstream code uniformity.
        if objective == "binary":
            probs = np.stack([1.0 - probs, probs], axis=1)
        y_pred = np.argmax(probs, axis=1)
        macro_f1, per_class, cm = _macro_f1(y_val, y_pred, len(present_classes))
        per_class_dict = dict(zip(present_classes, per_class))
    else:
        macro_f1 = 0.0
        per_class_dict = dict.fromkeys(present_classes, 0.0)
        cm = np.zeros((len(present_classes), len(present_classes)), dtype=np.int64)

    return ModeTrainResult(
        booster=booster,
        classes=tuple(present_classes),
        feature_names=feature_names,
        standardiser_mean=mean.astype(np.float32),
        standardiser_std=std.astype(np.float32),
        train_recording_ids=sorted(train_ids),
        val_recording_ids=sorted(val_ids),
        val_macro_f1=macro_f1,
        val_per_class_f1=per_class_dict,
        val_confusion=cm,
    )


def predict_modes(
    result: ModeTrainResult,
    segments: Iterable[TestDatasetSegment],
    cfg: V0ModeConfig,
) -> list[dict]:
    """Predict per-window mode probabilities + argmax for each segment.

    Handles both the multiclass case (probs shape ``(N, K)``) and the
    LightGBM binary case (probs shape ``(N,)`` representing P(class=1)),
    casting binary outputs to a 2-column matrix for downstream uniformity.
    """
    out: list[dict] = []
    for s in segments:
        feats, _ = extract_mode_features(s, cfg)
        if feats.shape[0] == 0:
            continue
        norm = (feats - result.standardiser_mean) / result.standardiser_std
        probs = result.booster.predict(norm, num_iteration=result.booster.best_iteration)
        if probs.ndim == 1:  # binary classifier
            probs = np.stack([1.0 - probs, probs], axis=1)
        preds = np.argmax(probs, axis=1)
        out.append(
            {
                "dataset_id": s.dataset_id,
                "recording_id": s.recording_id,
                "mode_label": s.mode_label,
                "n_windows": int(feats.shape[0]),
                "probs": probs.astype(np.float32),
                "predicted_class": np.array(
                    [result.classes[int(c)] for c in preds], dtype=object
                ),
            }
        )
    return out


@dataclass
class ModeFloorResult:
    """Unsupervised clustering quality of hand-engineered features vs mode.

    The RQ1 *lower bound*: how much of the operating mode is recoverable by
    clustering the same hand-engineered features the supervised
    :func:`train_v0_mode_lgbm` ceiling uses, with no representation learning and
    no labels.  The label-free encoder (V1/V2) must beat this floor to justify
    its complexity, and approach the supervised ceiling from below.
    """

    nmi: float
    ari: float
    purity: float
    n_windows: int
    n_recordings: int
    label_set: tuple[str, ...]
    n_clusters: int


def cluster_mode_floor(
    loaders: TestDatasetLoader | Iterable[TestDatasetLoader],
    cfg: V0ModeConfig | None = None,
    *,
    n_clusters: int = 3,
) -> ModeFloorResult:
    """K-means on hand-engineered features, scored against the mode labels.

    Pools the labelled (D1/D2) recordings of one or more loaders, standardises
    the per-window features, clusters them with K-means(``n_clusters``), and
    scores the assignment against the recorded mode with NMI / ARI / Hungarian
    purity — the same metrics Chapter 6 uses for the learned context, so the
    floor, the learned encoder, and the supervised ceiling all live on one axis.
    """
    from sklearn.preprocessing import StandardScaler

    from ..context.cluster_metric import cluster_purity_and_nmi

    cfg = cfg or V0ModeConfig()
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]

    feats: list[np.ndarray] = []
    labels: list[str] = []
    rec_keys: set[str] = set()
    valid = set(cfg.target_classes)
    for loader in loaders:  # type: ignore[union-attr]  # normalised to a list above
        for s in loader.list_segments():
            if (s.mode_label or "") not in valid:
                continue
            f, _ = extract_mode_features(s, cfg)
            if f.shape[0] == 0:
                continue
            feats.append(f)
            labels.extend([s.mode_label] * f.shape[0])
            rec_keys.add(f"{s.dataset_id}::{Path(s.source_dir).name}/{s.recording_id}")
    if not feats:
        raise RuntimeError("no labelled (D1/D2) windows found for the RQ1 floor")

    x = np.concatenate(feats, axis=0)
    x = StandardScaler().fit_transform(x)
    k = max(1, min(n_clusters, len(set(labels))))
    m = cluster_purity_and_nmi(x, labels, n_clusters=k, seed=cfg.seed)
    return ModeFloorResult(
        nmi=float(m["nmi"]),
        ari=float(m["ari"]),
        purity=float(m["purity"]),
        n_windows=int(x.shape[0]),
        n_recordings=len(rec_keys),
        label_set=tuple(m["label_set"]),
        n_clusters=k,
    )


__all__ = [
    "ModeFloorResult",
    "ModeTrainResult",
    "V0ModeConfig",
    "cluster_mode_floor",
    "extract_mode_features",
    "predict_modes",
    "train_v0_mode_lgbm",
]
