"""Standalone V0 baselines for the test_dataset pipeline.

These baselines read raw audio from the `TestDatasetLoader` and compute
features on the fly, so each V0 number compares apples-to-apples against the
subsequent iterations.

Anomaly detection (RQ2) reproduces the full prior-work reference of
Khamaisi et al. (2025) — the three unsupervised acoustic models they
benchmarked at ROW~II plus the density baseline named in the Experiments
chapter — and scores them through the proposed head's own RQ2 protocol:
  - `lstm_ae`        — LSTM autoencoder, per-window reconstruction MSE.
  - `density_scorers`— K-means distance, One-Class SVM, and Gaussian KDE on
                       a shared flat-feature embedding of the same windows.
  - `v0_evaluation`  — per-cluster percentile thresholding, healthy-alert
                       calibration, synthetic-anomaly ROC-AUC ladder, and
                       anomaly-cohort alert ranking, for acoustic *and*
                       vibration modalities.

Two further baselines, one per remaining research question:
  - `mode_lgbm`   — V0 supervised mode classifier on hand-engineered features
                    (RQ1 upper-bound reference; the only place mode labels are
                    legitimately used as a training target).

The V0 SRP-PHAT localization baseline (RQ3 reference) wraps the classical
primitives in `src/modeling/localization/classical.py` and lives there.
"""

from .density_scorers import (
    AnomalyScorer,
    KDEScorer,
    KMeansDistanceScorer,
    OneClassSVMScorer,
    build_scorer,
    extract_window_features,
    gather_features,
)
from .lstm_ae import (
    LSTMAutoencoderV0,
    V0Config,
    extract_log_mel_windows,
    extract_vibration_temporal_windows,
    fit_lstm_ae_on_windows,
    score_recordings,
    train_v0_lstm_ae,
)
from .mode_lgbm import (
    ModeFloorResult,
    ModeTrainResult,
    V0ModeConfig,
    cluster_mode_floor,
    extract_mode_features,
    predict_modes,
    train_v0_mode_lgbm,
)
from .srp_phat_baseline import (
    SRPConfig,
    evaluate_srp_phat,
    predict_srp_phat,
    summarise,
)
from .v0_evaluation import (
    ALL_MODELS,
    CLASSICAL_MODELS,
    MODALITIES,
    V0AnomalyResult,
    WindowBank,
    build_window_bank,
    evaluate_synthetic_anomaly_auc_v0,
    evaluate_v0_anomaly,
)

__all__ = [
    "ALL_MODELS",
    "CLASSICAL_MODELS",
    "MODALITIES",
    "AnomalyScorer",
    "KDEScorer",
    "KMeansDistanceScorer",
    "LSTMAutoencoderV0",
    "ModeFloorResult",
    "ModeTrainResult",
    "OneClassSVMScorer",
    "SRPConfig",
    "V0AnomalyResult",
    "V0Config",
    "V0ModeConfig",
    "WindowBank",
    "build_scorer",
    "build_window_bank",
    "cluster_mode_floor",
    "evaluate_srp_phat",
    "evaluate_synthetic_anomaly_auc_v0",
    "evaluate_v0_anomaly",
    "extract_log_mel_windows",
    "extract_mode_features",
    "extract_vibration_temporal_windows",
    "extract_window_features",
    "fit_lstm_ae_on_windows",
    "gather_features",
    "predict_modes",
    "predict_srp_phat",
    "score_recordings",
    "summarise",
    "train_v0_lstm_ae",
    "train_v0_mode_lgbm",
]
