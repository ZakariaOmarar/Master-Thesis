"""Smoke test for the V0 classical SRP-PHAT localization baseline.

Verifies the predict + evaluate loop runs end-to-end on D2 RandomFault and
D3 hit recordings, that ground-truth resolution works for both label classes
(folder-encoded for D2, mic-pair-midpoint for D3), and that error magnitudes
are finite + bounded by the search-grid extent.
"""

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
from src.modeling.anomaly_baselines.srp_phat_baseline import (
    SRPConfig,
    evaluate_srp_phat,
    predict_srp_phat,
    summarise,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.requires_data


def _resolved_spec(name: str) -> DatasetSpec:
    # `from_yaml` already resolves root + position_path to absolute paths, so the
    # spec is used as-is.  (The earlier hand-reconstruction dropped position_path,
    # which broke D3, whose position_source="d3_position_json" requires it.)
    return DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / f"{name}.yaml")


def _truncated_loader(name: str, max_seconds: float = 3.0):
    loader = TestDatasetLoader(_resolved_spec(name))
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


def test_predict_srp_phat_returns_in_grid() -> None:
    rng = np.random.default_rng(0)
    fs = 16_000
    n_mics = 5
    # Synthetic mic positions and a synthetic source: SRP-PHAT on white noise
    # only verifies the shape of the loop, not localisation accuracy.
    mic_xyz = rng.uniform(-0.5, 0.5, size=(n_mics, 3))
    mic_data = rng.standard_normal((n_mics, fs))  # 1 s
    cfg = SRPConfig(window_seconds=0.5, grid_step_m=0.1, grid_margin_m=0.1)
    pred = predict_srp_phat(mic_data, fs=fs, mic_xyz=mic_xyz, cfg=cfg)
    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred))
    bbox_lo = mic_xyz.min(axis=0) - (cfg.grid_margin_m + cfg.grid_step_m)
    bbox_hi = mic_xyz.max(axis=0) + (cfg.grid_margin_m + cfg.grid_step_m)
    assert np.all(pred >= bbox_lo)
    assert np.all(pred <= bbox_hi)


def test_evaluate_srp_phat_on_d2() -> None:
    loader, _ = _truncated_loader("d2", max_seconds=3.0)
    cfg = SRPConfig(window_seconds=1.0, grid_step_m=0.1, grid_margin_m=0.1)
    records = evaluate_srp_phat(loader, cfg)
    # D2 has 5 RandomFault subfolders → at least one should resolve a folder
    # spatial label and produce a finite SRP-PHAT estimate.
    assert len(records) >= 1
    for r in records:
        assert r["spatial_label_source"] == "folder"
        assert r["predicted_xyz"].shape == (3,)
        assert r["ground_truth_xyz"].shape == (3,)
        assert np.isfinite(r["error_m"])
        assert r["error_m"] >= 0.0
    s = summarise(records)
    assert s["n_recordings"] == len(records)
    assert s["mean_error_m"] >= 0.0


def test_evaluate_srp_phat_on_d3_resolves_mic_pair_midpoint() -> None:
    loader, _ = _truncated_loader("d3", max_seconds=3.0)
    cfg = SRPConfig(window_seconds=1.0, grid_step_m=0.5, grid_margin_m=0.5)
    records = evaluate_srp_phat(loader, cfg)
    # D3 has at least one hit_between_Fl_Gr_speed1 recording.
    hit_records = [r for r in records if r["spatial_label_source"] == "mic_pair_midpoint"]
    assert len(hit_records) >= 1, "D3 hit recording's mic-pair midpoint did not resolve"
    for r in hit_records:
        assert np.isfinite(r["error_m"])
        assert r["error_m"] >= 0.0
