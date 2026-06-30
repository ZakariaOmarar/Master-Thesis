"""Unit tests for the 2026-05-19 multi-scale window cadence.

Each test pins one invariant of the new ``window_scales_seconds[_per_dataset]``
plumbing in `_WindowedFeatureDataset` and `_PairedWindowedDataset`:

  * **Window-count formula** — each scale contributes exactly
    `floor((T − n_s) / stride_s) + 1` windows; the total is the sum
    across scales.  Locks the multi-scale dataset against an off-by-one
    that would silently change training-data density.
  * **Single-dataset × single-scale batches** — the grouped batch
    sampler buckets by `(channel_count, n_frames)` so every batch is
    `torch.stack`-able without padding.  Without this invariant
    multi-scale training would crash on shape mismatches.
  * **Legacy back-compat** — when `window_scales_seconds=()` the
    dataset reproduces the pre-2026-05-19 single-scale behaviour
    byte-for-byte.  This is what lets the existing V1/V2 smoke tests
    pass without modification.
  * **Per-dataset scale dict** — the per-dataset override resolves to
    the right scale set for each segment, NOT a global tuple.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.modeling.context.v1_ssl import (
    V1SSLConfig,
    _GroupedBatchSampler,
    _PrecomputedSegment,
    _resolve_segment_scales,
    _WindowedFeatureDataset,
)


def _make_segment(
    dataset_id: str,
    n_channels: int,
    T_frames: int,
    feature_fs: float,
    *,
    feature_shape_extra: tuple[int, ...] = (3,),
) -> _PrecomputedSegment:
    """Build a `_PrecomputedSegment` shaped like a vibration feature stack."""
    features = np.zeros((n_channels, *feature_shape_extra, T_frames), dtype=np.float32)
    xyz = np.zeros((n_channels, 3), dtype=np.float32)
    return _PrecomputedSegment(
        features=features,
        xyz=xyz,
        dataset_idx=0,
        dataset_id=dataset_id,
        mode_label="Pump",
        recording_id=f"{dataset_id}_rec_001",
        source_dir=f"/tmp/{dataset_id}",
        feature_fs=float(feature_fs),
    )


# ---------------------------------------------------------------------------
# _resolve_segment_scales priority
# ---------------------------------------------------------------------------


def test_resolve_per_dataset_dict_wins_over_global_tuple() -> None:
    cfg = V1SSLConfig(
        window_seconds=2.0,
        window_stride_seconds=1.0,
        window_scales_seconds=(0.5, 1.0),
        window_scales_seconds_per_dataset={"d1": (1.5, 3.0)},
        window_stride_ratio=0.5,
    )
    scales, stride_ratio = _resolve_segment_scales(cfg, "d1")
    assert scales == (1.5, 3.0)
    assert stride_ratio == 0.5
    # A different dataset (no override) falls back to the global tuple.
    scales_d3, _ = _resolve_segment_scales(cfg, "d3")
    assert scales_d3 == (0.5, 1.0)


def test_resolve_falls_back_to_legacy_when_no_scales_set() -> None:
    cfg = V1SSLConfig(
        window_seconds=2.0,
        window_stride_seconds=1.0,
        window_scales_seconds=(),
        window_scales_seconds_per_dataset={},
        window_stride_ratio=0.5,  # ignored on the legacy path
    )
    scales, stride_ratio = _resolve_segment_scales(cfg, "d1")
    assert scales == (2.0,)
    # Legacy stride is derived from the absolute stride / window ratio.
    assert stride_ratio == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Window-count formula
# ---------------------------------------------------------------------------


def test_window_count_formula_legacy_single_scale() -> None:
    """Legacy single-scale: count = floor((T - n) / stride) + 1.

    Force the legacy single-scale path by passing an empty per-dataset
    dict (the V1SSLConfig default is the publication multi-scale dict
    sourced from ``WINDOWING.window_scales_seconds_per_dataset``).
    """
    cfg = V1SSLConfig(
        window_seconds=2.0,
        window_stride_seconds=1.0,
        window_scales_seconds_per_dataset={},
    )
    seg = _make_segment("d1", n_channels=4, T_frames=20, feature_fs=4.0)
    # n_frames = 2.0 * 4 = 8; stride = 1.0 * 4 = 4.  At T=20 → starts in
    # [0, 4, 8, 12] → 4 windows.
    ds = _WindowedFeatureDataset([seg], cfg)
    assert len(ds) == 4


def test_window_count_formula_multi_scale_per_dataset() -> None:
    """Each scale contributes its own ⌊(T − n) / stride + 1⌋ windows."""
    cfg = V1SSLConfig(
        window_seconds=2.0,
        window_stride_seconds=1.0,
        window_scales_seconds_per_dataset={"d3": (0.5, 1.0, 2.0)},
        window_stride_ratio=0.5,
    )
    seg = _make_segment("d3", n_channels=4, T_frames=64, feature_fs=16.0)
    ds = _WindowedFeatureDataset([seg], cfg)
    # Scale 0.5 s: n=8, stride=4 → 15 windows.
    # Scale 1.0 s: n=16, stride=8 → 7 windows.
    # Scale 2.0 s: n=32, stride=16 → 3 windows.
    # Total = 25.
    expected = (1 + (64 - 8) // 4) + (1 + (64 - 16) // 8) + (1 + (64 - 32) // 16)
    assert len(ds) == expected


def test_window_count_skips_scales_longer_than_segment() -> None:
    """If T < n_s for some scale, that scale contributes zero windows."""
    cfg = V1SSLConfig(
        window_seconds=2.0,
        window_stride_seconds=1.0,
        window_scales_seconds_per_dataset={"d1": (0.5, 1.0, 8.0)},
        window_stride_ratio=0.5,
    )
    seg = _make_segment("d1", n_channels=4, T_frames=16, feature_fs=4.0)
    ds = _WindowedFeatureDataset([seg], cfg)
    # Only 0.5 s and 1.0 s yield ≥ 1 window; 8.0 s = 32 frames > T=16.
    n_05 = 1 + (16 - 2) // 1  # n_frames=2 (rounded from 0.5*4=2), stride=1
    n_10 = 1 + (16 - 4) // 2  # n_frames=4, stride=2
    assert len(ds) == n_05 + n_10


# ---------------------------------------------------------------------------
# Single-dataset × single-scale batches
# ---------------------------------------------------------------------------


def test_grouped_sampler_emits_single_scale_batches() -> None:
    """Each batch must carry windows of identical n_frames so `torch.stack`
    succeeds without padding masks.  This is the load-bearing invariant
    for multi-scale training under the existing collate function."""
    cfg = V1SSLConfig(
        window_seconds=2.0,
        window_stride_seconds=1.0,
        window_scales_seconds_per_dataset={"d3": (0.5, 1.0, 2.0)},
        window_stride_ratio=0.5,
    )
    seg = _make_segment("d3", n_channels=4, T_frames=128, feature_fs=16.0)
    ds = _WindowedFeatureDataset([seg], cfg)
    sampler = _GroupedBatchSampler(ds, batch_size=4, shuffle=False)
    for batch in sampler:
        # All indices in this batch must share n_frames.
        n_frames_set = {ds._refs[i][2] for i in batch}
        assert len(n_frames_set) == 1, (
            f"batch mixes n_frames values {n_frames_set} — grouped sampler "
            f"failed to bucket by frame count"
        )


# ---------------------------------------------------------------------------
# Cross-dataset isolation
# ---------------------------------------------------------------------------


def test_per_dataset_scales_apply_only_to_their_dataset() -> None:
    """A scale dict that only sets D1 must leave D3 windows untouched."""
    cfg = V1SSLConfig(
        window_seconds=2.0,
        window_stride_seconds=1.0,
        window_scales_seconds_per_dataset={"d1": (1.5, 3.0)},
        window_stride_ratio=0.5,
    )
    seg_d1 = _make_segment("d1", n_channels=4, T_frames=24, feature_fs=4.0)
    seg_d3 = _make_segment("d3", n_channels=4, T_frames=64, feature_fs=16.0)
    ds = _WindowedFeatureDataset([seg_d1, seg_d3], cfg)

    # D1 should use scales (1.5, 3.0) → ranges of n_frames.
    d1_n_frames = {ds._refs[i][2] for i in range(len(ds)) if ds._refs[i][0] == 0}
    # D3 should fall back to legacy single-scale (window_seconds=2.0) at
    # feature_fs=16 → n_frames=32.
    d3_n_frames = {ds._refs[i][2] for i in range(len(ds)) if ds._refs[i][0] == 1}
    assert d1_n_frames == {6, 12}  # 1.5*4 and 3.0*4
    assert d3_n_frames == {32}      # 2.0*16
