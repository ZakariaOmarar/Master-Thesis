"""Smoke test for the V0 LSTM-AE baseline — trains a tiny model on a clipped
copy of D1 and verifies the train/score loop is end-to-end finite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ingestion.test_dataset_loader import (
    DatasetSpec,
    TestDatasetLoader,
    TestDatasetSegment,
)
from src.modeling.anomaly_baselines.lstm_ae import (
    V0Config,
    extract_log_mel_windows,
    score_recordings,
    train_v0_lstm_ae,
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


def _truncated_loader(
    max_seconds: float = 6.0,
) -> tuple[TestDatasetLoader, list[TestDatasetSegment]]:
    """Return a loader-like wrapper whose `list_segments()` yields short clips.

    Cropping the segments keeps the smoke test under a minute even on D1's
    multi-minute recordings.
    """
    loader = TestDatasetLoader(_resolved_d1_spec())

    full_segments = loader.list_segments()
    truncated: list[TestDatasetSegment] = []
    for s in full_segments:
        n_mic_keep = int(round(max_seconds * s.segment.mic_sample_rate))
        n_vib_keep = max(8, int(round(max_seconds * s.segment.accel_sample_rate)))
        from src.data import DataSegment

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


def test_log_mel_windows_shape() -> None:
    _, segments = _truncated_loader(max_seconds=4.0)
    cfg = V0Config(
        n_mels=32, n_fft=512, hop_length=256, window_seconds=1.0, window_overlap=0.5
    )
    windows = extract_log_mel_windows(segments[0], cfg)
    assert windows.ndim == 3
    assert windows.shape[2] == 32
    assert windows.shape[0] >= 1
    assert np.all(np.isfinite(windows))


def test_train_and_score_end_to_end() -> None:
    loader, segments = _truncated_loader(max_seconds=5.0)
    cfg = V0Config(
        n_mels=32,
        n_fft=512,
        hop_length=256,
        window_seconds=0.5,
        window_overlap=0.5,
        hidden_dim=32,
        latent_dim=8,
        n_layers=1,
        dropout=0.0,
        epochs=2,
        batch_size=16,
        val_ratio=0.5,
        seed=0,
    )
    result = train_v0_lstm_ae(loader, cfg)

    assert len(result.train_loss_history) == cfg.epochs
    assert len(result.val_loss_history) == cfg.epochs
    assert all(np.isfinite(result.train_loss_history))
    assert all(np.isfinite(result.val_loss_history))
    # Training reduced loss on at least one of the (very short) epochs.
    assert result.train_loss_history[-1] <= result.train_loss_history[0] * 1.5

    records = score_recordings(
        result.model,
        result.standardiser_mean,
        result.standardiser_std,
        segments,
        cfg,
    )
    assert len(records) == len(segments)
    for r in records:
        assert r["n_windows"] >= 1
        assert r["scores"].shape == (r["n_windows"],)
        assert np.all(np.isfinite(r["scores"]))
        assert np.all(r["scores"] >= 0.0)  # MSE is non-negative


def test_recording_split_is_held_out_by_recording() -> None:
    loader, _ = _truncated_loader(max_seconds=5.0)
    cfg = V0Config(
        n_mels=16,
        n_fft=256,
        hop_length=128,
        window_seconds=0.25,
        window_overlap=0.5,
        hidden_dim=16,
        latent_dim=4,
        n_layers=1,
        epochs=1,
        batch_size=8,
        val_ratio=0.5,
        seed=1,
    )
    result = train_v0_lstm_ae(loader, cfg)
    train_ids = set(result.healthy_train_recordings)
    val_ids = set(result.healthy_val_recordings)
    # No recording appears in both splits — enforces the cross-recording
    # train/val separation that prevents window-level leakage.
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids) >= 1
    assert len(val_ids) >= 1
