"""Smoke tests for `src/modeling/context/modality_probe.py`.

The probe must:
  1. produce three cluster-metric dicts (`both`, `acoustic_only`,
     `vibration_only`),
  2. each carrying NMI / ARI / purity / cluster_idx fields, and
  3. be robust to small smoke corpora (no crashes on degenerate val).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data import DataSegment
from src.ingestion.test_dataset_loader import (
    DatasetSpec,
    TestDatasetLoader,
    TestDatasetSegment,
)
from src.modeling.context.modality_probe import run_modality_balance_probe
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.context.v2_ssl import V2SSLConfig, _precompute_paired

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


def _truncated_d1_paired_segments(max_seconds: float = 4.0):
    loader = TestDatasetLoader(_resolved_d1_spec())
    cfg = V2SSLConfig(
        window_seconds=0.5,
        window_stride_seconds=0.25,
        feature_dim=32,
        embed_dim=32,
        n_heads=2,
        proj_dim=16,
        epochs=1,
        batch_size=8,
        val_ratio=0.5,
        n_mels=32,
        n_fft=256,
        hop_length=128,
        use_cwt=False,
        seed=0,
    )
    out = []
    for s in loader.list_segments():
        if s.is_anomaly:
            continue
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
        s2 = TestDatasetSegment(
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
        pre = _precompute_paired(s2, cfg)
        if pre is not None:
            out.append(pre)
    return out, cfg


def test_modality_probe_returns_three_metric_dicts() -> None:
    segments, cfg = _truncated_d1_paired_segments(max_seconds=4.0)
    torch.manual_seed(0)
    encoder = V2FusionEncoder(feature_dim=cfg.feature_dim, embed_dim=cfg.embed_dim, n_heads=cfg.n_heads)
    result = run_modality_balance_probe(
        encoder, segments, v2_cfg=cfg,
        n_clusters=2,  # D1 has only Pump / Turbine in our smoke corpus
        healthy_mode_labels=("Pump", "Turbine"),
        seed=0,
    )
    for row in (result.both, result.acoustic_only, result.vibration_only):
        for key in ("nmi", "ari", "purity", "n_clusters", "cluster_idx"):
            assert key in row
        # NMI / ARI / purity all in [0, 1] (purity definitionally in [0, 1]
        # after Hungarian-matching; NMI/ARI by construction).
        if not np.isnan(row["nmi"]):
            assert 0.0 <= row["nmi"] <= 1.0
        if not np.isnan(row["purity"]):
            assert 0.0 <= row["purity"] <= 1.0


def test_modality_probe_empty_segments_returns_nan_structure() -> None:
    cfg = V2SSLConfig(
        window_seconds=0.5, window_stride_seconds=0.25,
        feature_dim=32, embed_dim=32, n_heads=2, proj_dim=16,
        epochs=1, batch_size=8, n_mels=32, n_fft=256, hop_length=128,
        use_cwt=False, seed=0,
    )
    torch.manual_seed(0)
    encoder = V2FusionEncoder(feature_dim=cfg.feature_dim, embed_dim=cfg.embed_dim, n_heads=cfg.n_heads)
    result = run_modality_balance_probe(
        encoder, [], v2_cfg=cfg, n_clusters=3, seed=0,
    )
    assert result.both["n_windows"] == 0


# numpy import is conditional on the test that uses it
import numpy as np  # noqa: E402
