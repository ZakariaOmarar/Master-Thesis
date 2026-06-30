"""Tests for the completed V0 anomaly baseline (Khamaisi trio + KDE).

Two tiers:
  * pure-synthetic unit tests (no recordings) covering the scorer orientation,
    the flat-feature aggregation, and the model-agnostic synthetic-AUC;
  * a ``requires_data`` smoke test that runs the full ``evaluate_v0_anomaly``
    harness for every model on a clipped copy of D1, acoustic and vibration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ingestion.test_dataset_loader import (
    DatasetSpec,
    TestDatasetLoader,
    TestDatasetSegment,
)
from src.modeling.anomaly_baselines.density_scorers import (
    KDEScorer,
    KMeansDistanceScorer,
    OneClassSVMScorer,
    _aggregate_windows,
    build_scorer,
)
from src.modeling.anomaly_baselines.lstm_ae import V0Config
from src.modeling.anomaly_baselines.v0_evaluation import (
    ALL_MODELS,
    evaluate_synthetic_anomaly_auc_v0,
    evaluate_v0_anomaly,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Pure-synthetic unit tests (no data needed)
# ---------------------------------------------------------------------------


def test_aggregate_windows_concats_mean_and_std() -> None:
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(5, 12, 8)).astype(np.float32)  # (N, T, F)
    feats = _aggregate_windows(windows)
    assert feats.shape == (5, 16)  # 2 * F
    assert np.allclose(feats[:, :8], windows.mean(axis=1), atol=1e-5)
    assert np.allclose(feats[:, 8:], windows.std(axis=1), atol=1e-5)


def test_aggregate_windows_empty() -> None:
    empty = np.zeros((0, 4, 6), dtype=np.float32)
    assert _aggregate_windows(empty).shape == (0, 12)


@pytest.mark.parametrize("name", ["kmeans", "ocsvm", "kde"])
def test_classical_scorer_orientation(name: str) -> None:
    """Far-from-healthy points must score higher (higher == more anomalous)."""
    rng = np.random.default_rng(7)
    healthy = rng.normal(0.0, 1.0, size=(300, 8)).astype(np.float32)
    outliers = rng.normal(10.0, 1.0, size=(60, 8)).astype(np.float32)

    scorer = build_scorer(name, seed=0).fit(healthy)
    s_healthy = scorer.score(healthy)
    s_out = scorer.score(outliers)

    assert s_healthy.shape == (300,)
    assert np.all(np.isfinite(s_healthy)) and np.all(np.isfinite(s_out))
    # Every outlier scores above the 95th percentile of the healthy pool.
    assert float(s_out.mean()) > float(np.percentile(s_healthy, 95))


def test_build_scorer_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        build_scorer("nope")


def test_synthetic_auc_grows_as_perturbation_grows() -> None:
    """A norm scorer should discriminate corrupted windows better at low SNR."""
    rng = np.random.default_rng(3)
    healthy = rng.normal(size=(200, 10))

    def score_fn(x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.asarray(x, dtype=np.float64), axis=1)

    res = evaluate_synthetic_anomaly_auc_v0(
        score_fn, healthy, snr_db_list=(-10.0, 0.0, 10.0), n_boot=0, seed=1
    )
    auc = res.snr_db_to_auc
    assert set(auc) == {-10.0, 0.0, 10.0}
    assert all(0.0 <= v <= 1.0 for v in auc.values())
    # Louder noise (lower SNR) is the easier discrimination.
    assert auc[-10.0] >= auc[10.0]
    assert auc[-10.0] > 0.5


def test_scorer_score_before_fit_raises() -> None:
    for scorer in (KMeansDistanceScorer(), OneClassSVMScorer(), KDEScorer()):
        with pytest.raises(RuntimeError):
            scorer.score(np.zeros((2, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# requires_data end-to-end smoke test on a clipped D1
# ---------------------------------------------------------------------------

pytestmark_data = pytest.mark.requires_data


def _resolved_spec(dataset_id: str) -> DatasetSpec:
    spec = DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / f"{dataset_id}.yaml")
    return spec


def _truncated_loader(dataset_id: str = "d1", max_seconds: float = 6.0):
    """Loader-like stub whose ``list_segments`` yields short clips of one dataset."""
    from src.data import DataSegment

    loader = TestDatasetLoader(_resolved_spec(dataset_id))
    truncated: list[TestDatasetSegment] = []
    for s in loader.list_segments():
        n_mic = int(round(max_seconds * s.segment.mic_sample_rate))
        n_vib = max(8, int(round(max_seconds * s.segment.accel_sample_rate)))
        new_seg = DataSegment.from_arrays(
            mic_data=s.segment.mic_data[:, :n_mic],
            accel_data=s.segment.accel_data[:, :n_vib],
            start_time=s.segment.start_time,
            mic_sr=s.segment.mic_sample_rate,
            accel_sr=s.segment.accel_sample_rate,
            metadata=dict(s.segment.metadata),
        )
        truncated.append(
            TestDatasetSegment(
                segment=new_seg,
                mic_positions=s.mic_positions,
                vib_positions=s.vib_positions,
                mic_ids=s.mic_ids,
                vib_ids=s.vib_ids,
                mode_label=s.mode_label,
                op_condition=s.op_condition,
                spatial_label=s.spatial_label,
                dataset_id=s.dataset_id,
                recording_id=s.recording_id,
                source_dir=s.source_dir,
                is_anomaly=s.is_anomaly,
            )
        )

    class _StubLoader:
        spec = loader.spec
        registry = loader.registry

        def list_segments(self, **_kwargs):
            return list(truncated)

    return _StubLoader()


def _corpus(dataset_ids: list[str], max_seconds: float = 6.0) -> list:
    """A pooled corpus of truncated stub loaders (mirrors the head's ANOM_LOADERS)."""
    return [_truncated_loader(ds, max_seconds) for ds in dataset_ids]


def _quick_cfg() -> V0Config:
    return V0Config(
        n_mels=32, n_fft=512, hop_length=256, window_seconds=0.5,
        hidden_dim=16, latent_dim=8, n_layers=1, epochs=2, batch_size=16,
        val_ratio=0.5, seed=0,
    )


@pytestmark_data
@pytest.mark.parametrize("model", list(ALL_MODELS))
def test_evaluate_v0_anomaly_acoustic(model: str) -> None:
    # Pool three campaigns so the three-way held-out split has >= 3 healthy
    # recordings (the way the head is trained on ANOM_LOADERS).
    corpus = _corpus(["d1", "d2", "d3"], max_seconds=6.0)
    res = evaluate_v0_anomaly(
        corpus, model, "acoustic", _quick_cfg(),
        percentile=95, n_clusters=3, snr_db_list=(-10.0, 0.0, 10.0), n_boot=50,
    )
    assert res.model == model
    assert res.dataset_ids == ["d1", "d2", "d3"]
    # headline within-campaign detection ROC-AUC is finite, a probability, with a CI
    assert np.isfinite(res.roc_auc) and 0.0 <= res.roc_auc <= 1.0
    lo, hi = res.roc_auc_ci
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= res.roc_auc <= hi
    # both calibration regimes are probabilities
    assert 0.0 <= res.fpr_in_distribution <= 1.0
    assert 0.0 <= res.fpr_domain_shift <= 1.0
    d = res.to_dict()
    assert d["n_train"] >= 1 and d["n_heldout"] >= 1
    assert "roc_auc_by_dataset" in d and "synthetic_auc" in d


@pytestmark_data
def test_evaluate_v0_anomaly_vibration() -> None:
    # D1 + D2 share a 4 Hz accelerometer rate, so the vibration sequence windows
    # pool cleanly and even the AE is runnable on this corpus.
    corpus = _corpus(["d1", "d2"], max_seconds=6.0)
    res = evaluate_v0_anomaly(
        corpus, "kmeans", "vibration", _quick_cfg(),
        snr_db_list=(-10.0, 0.0), n_boot=0,
    )
    assert res.modality == "vibration"
    assert 0.0 <= res.fpr_in_distribution <= 1.0


@pytestmark_data
def test_lstm_ae_rejects_mixed_rate_vibration_pool() -> None:
    # D2 (4 Hz) + D3 (16 Hz) vibration cannot be pooled for the sequence AE.
    corpus = _corpus(["d2", "d3"], max_seconds=6.0)
    with pytest.raises(RuntimeError, match="frame counts differ"):
        evaluate_v0_anomaly(corpus, "lstm_ae", "vibration", _quick_cfg(), n_boot=0)
