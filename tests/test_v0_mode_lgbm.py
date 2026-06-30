"""Smoke test for the V0 LightGBM mode classifier — trains a tiny model on a
clipped copy of D1 and verifies the train/predict loop is end-to-end finite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import DataSegment
from src.ingestion.test_dataset_loader import (
    DatasetSpec,
    TestDatasetLoader,
    TestDatasetSegment,
)
from src.modeling.anomaly_baselines.mode_lgbm import (
    V0ModeConfig,
    cluster_mode_floor,
    extract_mode_features,
    predict_modes,
    train_v0_mode_lgbm,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.requires_data


def _resolved_d1_spec() -> DatasetSpec:
    spec = DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / "d1.yaml")
    return DatasetSpec(
        id=spec.id,
        root=REPO_ROOT / spec.root,
        n_mics=spec.n_mics,
        n_vibrations=spec.n_vibrations,
        accel_target_sr=spec.accel_target_sr,
        position_source=spec.position_source,
        label_scheme=spec.label_scheme,
        extra=spec.extra,
    )


def _truncated_loader(max_seconds: float = 6.0) -> tuple:
    loader = TestDatasetLoader(_resolved_d1_spec())
    full_segments = loader.list_segments()
    truncated: list[TestDatasetSegment] = []
    for s in full_segments:
        n_mic_keep = int(round(max_seconds * s.segment.mic_sample_rate))
        n_vib_keep = max(8, int(round(max_seconds * s.segment.accel_sample_rate)))
        new_seg = DataSegment.from_arrays(
            mic_data=s.segment.mic_data[:, :n_mic_keep],
            accel_data=s.segment.accel_data[:, :n_vib_keep],
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
            )
        )

    class _StubLoader:
        spec = loader.spec
        registry = loader.registry

        def list_segments(self, **_kwargs):
            return list(truncated)

    return _StubLoader(), truncated


def test_mode_features_shape() -> None:
    _, segments = _truncated_loader(max_seconds=4.0)
    cfg = V0ModeConfig(
        n_mels=32, n_fft=512, hop_length=256, window_seconds=1.0, window_overlap=0.5
    )
    feats, names = extract_mode_features(segments[0], cfg)
    assert feats.ndim == 2
    # 32 mel means + 3 acoustic stats + 4 vibration stats = 39
    assert feats.shape[1] == 32 + 3 + 4
    assert len(names) == feats.shape[1]
    assert feats.shape[0] >= 1
    assert np.all(np.isfinite(feats))


def test_train_and_predict_end_to_end() -> None:
    pytest.importorskip("lightgbm")
    loader, segments = _truncated_loader(max_seconds=5.0)
    cfg = V0ModeConfig(
        n_mels=32,
        n_fft=512,
        hop_length=256,
        window_seconds=0.5,
        window_overlap=0.5,
        n_estimators=20,
        learning_rate=0.1,
        num_leaves=7,
        min_child_samples=2,
        val_ratio=0.5,
        seed=0,
    )
    result = train_v0_mode_lgbm(loader, cfg)

    # Held-out split must be at the recording level — no leakage.
    train_ids = set(result.train_recording_ids)
    val_ids = set(result.val_recording_ids)
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids) >= 1
    assert len(val_ids) >= 1

    # F1 is well-defined and in [0, 1]; we don't gate on a specific value
    # because tiny synthetic-clip training is not meant to give realistic
    # numbers. We only check the loop ran end-to-end.
    assert 0.0 <= result.val_macro_f1 <= 1.0
    # `result.classes` is the **present** subset of cfg.target_classes —
    # D1 has Pump + Turbine only (no Standstill), so the V0 trainer
    # adapts num_class accordingly.
    assert set(result.val_per_class_f1) == set(result.classes)
    n_present = len(result.classes)
    assert result.val_confusion.shape == (n_present, n_present)

    preds = predict_modes(result, segments, cfg)
    assert len(preds) == len(segments)
    for r in preds:
        assert r["n_windows"] >= 1
        assert r["probs"].shape == (r["n_windows"], n_present)
        assert r["predicted_class"].shape == (r["n_windows"],)
        assert np.allclose(r["probs"].sum(axis=1), 1.0, atol=1e-3)


def test_cluster_mode_floor_end_to_end() -> None:
    loader, _ = _truncated_loader(max_seconds=5.0)
    cfg = V0ModeConfig(
        n_mels=32, n_fft=512, hop_length=256, window_seconds=0.5,
        window_overlap=0.5, seed=0,
    )
    floor = cluster_mode_floor(loader, cfg)
    # K clamps to the number of distinct labelled modes present in D1.
    assert floor.n_clusters == len(floor.label_set)
    assert floor.n_clusters >= 2
    assert floor.n_windows >= 2 and floor.n_recordings >= 2
    for v in (floor.nmi, floor.ari, floor.purity):
        assert np.isfinite(v)
    assert 0.0 <= floor.nmi <= 1.0 and 0.0 <= floor.purity <= 1.0
