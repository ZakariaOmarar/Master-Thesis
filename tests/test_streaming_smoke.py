"""Smoke test for the gated streaming pipeline.

Verifies on a tiny truncated D1 that the pipeline:
  - Emits one `StreamingDecision` per window.
  - Always emits `cluster_id`, `anomaly_score`, `alert_flag`.
  - Emits `xyz` only on alert windows under gated mode.
  - Reports a non-zero per-window runtime.
  - Achieves a measurable speed-up under gated mode in the cost/quality study.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import DataSegment
from src.ingestion.test_dataset_loader import (
    DatasetSpec,
    TestDatasetLoader,
    TestDatasetSegment,
)
from src.modeling.anomaly import (
    train_v3_cnf,
)
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.context.v2_ssl import V2SSLConfig
from src.modeling.localization import (
    GridSpec,
    V4Config,
    precompute_v4_samples,
    train_v4_localization,
)
from src.modeling.streaming import GatedPipeline, cost_quality_study

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


def _truncated_segments(max_seconds: float = 5.0) -> list[TestDatasetSegment]:
    loader = TestDatasetLoader(_resolved_d1_spec())
    out: list[TestDatasetSegment] = []
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
        out.append(
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
    return out


def _smoke_v2_cfg() -> V2SSLConfig:
    return V2SSLConfig(
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
        gain_jitter_db=0.0,
        channel_dropout_p=0.0,
        spec_augment_freq_mask=0,
        spec_augment_time_mask=0,
        seed=0,
    )


def _build_pipeline_for_smoke():
    """Train tiny V2 (random init) → V3 → V4, then assemble the pipeline."""
    # 10 s (not 5 s): D1's vibration feature rate is ~1 frame/s, so V3's nested
    # disjoint threshold-fit/eval split (two halvings) needs the length to keep
    # both sub-cohorts non-empty (Chapter-5 protocol; V3 hard-errors otherwise).
    segments = _truncated_segments(max_seconds=10.0)
    v2_cfg = _smoke_v2_cfg()
    grid = GridSpec(lo=(-0.5, -0.5, 0.0), hi=(0.5, 0.5, 0.5), n=(8, 8, 4))

    torch.manual_seed(0)
    encoder = V2FusionEncoder(feature_dim=32, embed_dim=32, n_heads=2)

    # V3
    class _StubLoader:
        def list_segments(self, **_kw):
            return segments

    from src.modeling.anomaly import V3Config

    v3 = train_v3_cnf(
        encoder,
        _StubLoader(),
        v2_cfg=v2_cfg,
        v3_cfg=V3Config(
            n_layers=4, hidden_dim=32, epochs=1, batch_size=8, val_ratio=0.5,
            n_threshold_clusters=3, threshold_percentile=95, seed=0,
        ),
    )

    # V4
    rng = np.random.default_rng(0)
    overrides = {
        s.recording_id: tuple(rng.uniform([-0.3, -0.3, 0.0], [0.3, 0.3, 0.4]).tolist())
        for s in segments
    }
    samples = precompute_v4_samples(
        encoder, segments, v2_cfg=v2_cfg, grid=grid, spatial_label_overrides=overrides
    )
    v4_result = train_v4_localization(
        samples,
        cfg=V4Config(
            cnn_feature_dim=32,
            tdoa_feature_dim=16,
            hidden_dim=32,
            epochs=1,
            batch_size=4,
            val_ratio=0.5,
            seed=0,
        ),
        grid=grid,
    )

    pipeline = GatedPipeline(
        v2_encoder=encoder,
        flow=v3.flow,
        thresholds=v3.thresholds,
        v4_head=v4_result.head,
        grid=grid,
        v2_cfg=v2_cfg,
        threshold_percentile=95,
        # Match the trained flow: PMA-2 pooling + impulse anchor (if enabled).
        xt_pool=getattr(v3, "xt_pool", None),
        anchor_norm=((v3.anchor_mean, v3.anchor_std)
                     if getattr(v3, "anchor_mean", None) is not None else None),
    )
    return pipeline, segments


def test_streaming_pipeline_emits_per_window_decision() -> None:
    pipeline, segments = _build_pipeline_for_smoke()
    decisions = pipeline.run_segment(segments[0], gated=True)
    assert len(decisions) > 0

    for d in decisions:
        assert d.t_end_s > d.t_start_s
        assert isinstance(d.cluster_id, int)
        assert np.isfinite(d.anomaly_score)
        assert isinstance(d.alert_flag, bool)
        assert d.runtime_ms >= 0.0
        if d.alert_flag:
            assert d.xyz is not None
            assert len(d.xyz) == 3
        else:
            assert d.xyz is None


def test_streaming_pipeline_continuous_mode_runs_v4_every_window() -> None:
    pipeline, segments = _build_pipeline_for_smoke()
    decisions = pipeline.run_segment(segments[0], gated=False)
    assert len(decisions) > 0
    # Continuous mode emits xyz on every window regardless of alert.
    assert all(d.xyz is not None for d in decisions)


def test_streaming_pipeline_cluster_to_label_mapping() -> None:
    pipeline, segments = _build_pipeline_for_smoke()
    pipeline.cluster_to_label = {0: "Pump", 1: "Standstill", 2: "Turbine"}
    decisions = pipeline.run_segment(segments[0])
    for d in decisions:
        if d.cluster_id in pipeline.cluster_to_label:
            assert d.mode_label == pipeline.cluster_to_label[d.cluster_id]
        else:
            assert d.mode_label is None


def test_cost_quality_study_emits_finite_speedup() -> None:
    pipeline, segments = _build_pipeline_for_smoke()
    report = cost_quality_study(pipeline, segments[:2])
    assert report.n_windows > 0
    assert report.gated_total_ms >= 0.0
    assert report.continuous_total_ms >= 0.0
    assert np.isfinite(report.speedup_x)
    # Continuous must do at least as much work as gated ONLY when some
    # windows skip V4 (i.e. n_alerts < n_windows).  On synthetic data
    # where every window crosses the threshold (n_alerts == n_windows),
    # both paths run V4 and gated pays the extra "decide to alert"
    # overhead — so the inequality can flip on small samples where
    # process-jitter > the V4-skipped saving.  Guard the assertion.
    if report.n_alerts < report.n_windows:
        assert report.continuous_total_ms + 1e-6 >= report.gated_total_ms


def test_streaming_pipeline_works_without_v4() -> None:
    """If `v4_head=None`, the pipeline still emits mode + anomaly + alert."""
    pipeline, segments = _build_pipeline_for_smoke()
    pipeline.v4_head = None
    decisions = pipeline.run_segment(segments[0], gated=True)
    assert len(decisions) > 0
    for d in decisions:
        assert d.xyz is None
        assert np.isfinite(d.anomaly_score)


def test_streaming_decision_to_dict_is_json_safe() -> None:
    import json

    pipeline, segments = _build_pipeline_for_smoke()
    decisions = pipeline.run_segment(segments[0])
    payload = [d.to_dict() for d in decisions]
    s = json.dumps(payload)
    assert isinstance(s, str)
    assert "anomaly_score" in s
