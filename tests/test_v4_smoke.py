"""Smoke tests for V4 anomaly-gated localization head.

Three groups:
  1. Front-end primitives — `compute_srp_phat_volume` and
     `compute_accel_tdoa_tokens` are channel-agnostic and emit fixed-shape
     volumes / variable-length TDOA tokens.
  2. V4 modules — `HeatmapCross3D`, `TDOASetEncoder`, `V4LocalizationHead`
     forward passes are finite and shape-correct, including the empty-TDOA
     edge case.  The head uses a soft-argmax over the SRP volume + a
     FiLM-conditioned residual MLP — the A3 invariant (zero c → identity
     FiLM) is preserved through the residual MLP's zero-init final layer.
  3. End-to-end V4 training on a tiny truncated D1 with synthetic spatial
     labels — finite loss, held-out recordings disjoint, finite val MAE,
     A3 ablation runs without error.
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
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.context.v2_ssl import V2SSLConfig
from src.modeling.localization import (
    GridSpec,
    HeatmapCross3D,
    TDOASetEncoder,
    V4Config,
    V4LocalizationHead,
    compute_accel_tdoa_tokens,
    compute_burst_aware_srp_phat_volume,
    compute_srp_phat_volume,
    find_burst_window,
    precompute_v4_samples,
    soft_argmax_3d,
    train_v4_localization,
)
from src.modeling.localization.v4_trainer import _grid_coords_from_spec

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


def _smoke_grid() -> GridSpec:
    return GridSpec(lo=(-0.5, -0.5, 0.0), hi=(0.5, 0.5, 0.5), n=(8, 8, 4))


# ---------------------------------------------------------------------------
# 1. Front-end primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_mics", [4, 5, 9])
def test_srp_phat_volume_channel_agnostic(n_mics: int) -> None:
    rng = np.random.default_rng(0)
    fs = 8000.0
    T = 4096
    mic_data = rng.standard_normal((n_mics, T)).astype(np.float32)
    mic_xyz = rng.uniform(-0.3, 0.3, size=(n_mics, 3)).astype(np.float32)
    grid = GridSpec(lo=(-0.5, -0.5, 0.0), hi=(0.5, 0.5, 0.5), n=(8, 8, 4))
    vol = compute_srp_phat_volume(mic_data, mic_xyz, fs=fs, grid=grid)
    assert vol.shape == (8, 8, 4)
    assert np.all(np.isfinite(vol))
    # Normalised to peak ≤ 1.
    assert vol.max() <= 1.0 + 1e-6
    assert vol.max() > 0.0


def test_find_burst_window_picks_the_loudest_subwindow() -> None:
    """A 1-second 16 kHz signal with a 50 ms burst at t=0.4s should
    produce a burst window whose midpoint lies inside the burst."""
    rng = np.random.default_rng(0)
    fs = 16000
    T = fs  # 1 second
    n_mics = 4
    background = 0.01 * rng.standard_normal((n_mics, T))
    burst_start = int(0.40 * fs)
    burst_len = int(0.05 * fs)
    burst = 5.0 * rng.standard_normal((n_mics, burst_len))
    sig = background.copy()
    sig[:, burst_start : burst_start + burst_len] += burst

    s, e = find_burst_window(sig, fs=float(fs), burst_seconds=0.10)
    mid = (s + e) // 2
    assert burst_start - int(0.05 * fs) <= mid <= burst_start + burst_len + int(0.05 * fs), (
        f"burst-window midpoint {mid} not near burst @ {burst_start}..{burst_start + burst_len}"
    )
    assert e - s == int(round(0.10 * fs))


def test_burst_aware_srp_concentrates_on_burst() -> None:
    """The burst-aware SRP volume's peak voxel should be closer to the
    true source than the full-window SRP volume's peak — because the
    full-window SRP averages GCC over ~ 95 % background-only samples."""
    rng = np.random.default_rng(1)
    fs = 16000
    T = fs  # 1 s
    n_mics = 4
    mic_xyz = np.array(
        [[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [0.0, 0.10, 0.0], [0.10, 0.10, 0.0]],
        dtype=np.float64,
    )
    src = np.array([0.05, 0.05, 0.05], dtype=np.float64)

    # Build a synthetic burst: an impulsive pulse delayed at each mic by
    # its true geometric TDOA to the source.
    c_air = 343.0
    burst_t = int(0.40 * fs)
    burst_len = int(0.005 * fs)  # 5 ms pulse
    impulse = rng.standard_normal(burst_len) * 5.0
    sig = 0.01 * rng.standard_normal((n_mics, T)).astype(np.float64)
    for m in range(n_mics):
        d = float(np.linalg.norm(src - mic_xyz[m]))
        delay = int(round(d / c_air * fs))
        s0 = burst_t + delay
        e0 = s0 + burst_len
        if e0 <= T:
            sig[m, s0:e0] += impulse

    grid = GridSpec(lo=(-0.05, -0.05, -0.05), hi=(0.15, 0.15, 0.15), n=(11, 11, 11))
    vol_full = compute_srp_phat_volume(sig, mic_xyz, fs=float(fs), grid=grid)
    vol_burst = compute_burst_aware_srp_phat_volume(
        sig, mic_xyz, fs=float(fs), grid=grid, burst_seconds=0.05,
    )

    ax_x, ax_y, ax_z = grid.axes()
    def _peak_xyz(vol):
        idx = np.unravel_index(int(np.argmax(vol)), vol.shape)
        return np.array([ax_x[idx[0]], ax_y[idx[1]], ax_z[idx[2]]])

    err_full = float(np.linalg.norm(_peak_xyz(vol_full) - src))
    err_burst = float(np.linalg.norm(_peak_xyz(vol_burst) - src))

    # Both arrays are valid SRP volumes
    assert vol_full.shape == (11, 11, 11)
    assert vol_burst.shape == (11, 11, 11)
    assert np.all(np.isfinite(vol_full))
    assert np.all(np.isfinite(vol_burst))
    # Burst-aware SRP must not be *worse* than full-window SRP on this
    # canonical impulse-in-noise scenario.  The lenient assertion
    # accommodates the ~ 2 cm voxel quantisation of the grid.
    assert err_burst <= err_full + 0.02, (
        f"burst-aware SRP regressed: full={err_full:.3f} m, burst={err_burst:.3f} m"
    )


@pytest.mark.parametrize("n_vib", [4, 5])
def test_accel_tdoa_tokens_shape(n_vib: int) -> None:
    rng = np.random.default_rng(0)
    fs = 16.0
    T = 64
    accel = rng.standard_normal((n_vib, T)).astype(np.float32)
    xyz = rng.uniform(-0.3, 0.3, size=(n_vib, 3)).astype(np.float32)
    tokens = compute_accel_tdoa_tokens(accel, xyz, fs=fs)
    expected = n_vib * (n_vib - 1) // 2
    assert tokens.shape == (expected, 8)
    assert np.all(np.isfinite(tokens))


def test_accel_tdoa_tokens_handles_single_channel() -> None:
    accel = np.zeros((1, 32), dtype=np.float32)
    xyz = np.zeros((1, 3), dtype=np.float32)
    tokens = compute_accel_tdoa_tokens(accel, xyz, fs=16.0)
    assert tokens.shape == (0, 8)


# ---------------------------------------------------------------------------
# 2. V4 modules
# ---------------------------------------------------------------------------


def test_heatmap_cross3d_forward() -> None:
    """`HeatmapCross3D` returns a per-voxel logit volume and a global feature.

    The logit volume must match the input grid shape so the soft-argmax
    over voxel coordinates is well-defined.
    """
    cnn = HeatmapCross3D(feature_dim=32)
    cnn.eval()
    vol = torch.randn(2, 8, 8, 4)
    with torch.no_grad():
        logits, global_feat = cnn(vol)
    assert logits.shape == (2, 8, 8, 4)
    assert global_feat.shape == (2, 32)
    assert torch.all(torch.isfinite(logits))
    assert torch.all(torch.isfinite(global_feat))


def test_soft_argmax_3d_recovers_peak() -> None:
    """A peaked logit volume should produce a soft-argmax close to the peak's
    voxel coordinates.  This is the differentiable spatial-decoding
    invariant the V4 head depends on."""
    grid = GridSpec(lo=(-0.5, -0.5, 0.0), hi=(0.5, 0.5, 0.5), n=(8, 8, 4))
    coords = _grid_coords_from_spec(grid)  # (8, 8, 4, 3)
    logits = torch.full((1, 8, 8, 4), -10.0)
    # Place a strong peak at index (5, 3, 2)
    target_idx = (5, 3, 2)
    logits[0, target_idx[0], target_idx[1], target_idx[2]] = 50.0
    pred = soft_argmax_3d(logits, coords, temperature=1.0)
    expected = coords[target_idx].clone().detach().unsqueeze(0)
    assert torch.allclose(pred, expected, atol=5e-3)


@pytest.mark.parametrize("n_pairs", [0, 6, 10])
def test_tdoa_set_encoder_handles_variable_npairs(n_pairs: int) -> None:
    enc = TDOASetEncoder(feature_dim=16, n_heads=2, hidden_dim=32)
    enc.eval()
    tokens = torch.randn(2, n_pairs, 8) if n_pairs > 0 else torch.zeros(2, 0, 8)
    with torch.no_grad():
        feat = enc(tokens)
    assert feat.shape == (2, 16)
    assert torch.all(torch.isfinite(feat))


def test_v4_localization_head_a3_zeroes_conditioner() -> None:
    """`unconditional=True` zeros c inside the head.  Because FiLM γ/β are
    zero-initialised AND the residual MLP's final layer is zero-init, an
    all-zero conditioner makes the head return the same prediction whether
    `unconditional` is True or the input c is already 0 — and at init the
    prediction equals the unconditional soft-argmax."""
    torch.manual_seed(0)
    grid = _smoke_grid()
    coords = _grid_coords_from_spec(grid)
    head = V4LocalizationHead(
        grid_coords=coords,
        cnn_feature_dim=32,
        tdoa_feature_dim=16,
        c_dim=24,
        s_dim=0,
        hidden_dim=32,
    )
    head.eval()
    vol = torch.randn(2, 8, 8, 4)
    tdoa = torch.randn(2, 6, 8)
    c = torch.randn(2, 24)
    with torch.no_grad():
        pred_uncond = head(vol, tdoa, c, unconditional=True)
        pred_with_zero_c = head(vol, tdoa, torch.zeros_like(c))
    assert pred_uncond.shape == (2, 3)
    # FiLM γ/β init zero, so the path under c=0 must be identical to
    # the unconditional path under any c.
    assert torch.allclose(pred_uncond, pred_with_zero_c, atol=1e-6)


def test_v4_localization_head_with_scada() -> None:
    """`s_dim>0` accepts a SCADA tensor and concatenates it before FiLM."""
    torch.manual_seed(0)
    grid = _smoke_grid()
    coords = _grid_coords_from_spec(grid)
    head = V4LocalizationHead(
        grid_coords=coords,
        cnn_feature_dim=32, tdoa_feature_dim=16, c_dim=24, s_dim=4, hidden_dim=32,
    )
    head.eval()
    vol = torch.randn(2, 8, 8, 4)
    tdoa = torch.randn(2, 6, 8)
    c = torch.randn(2, 24)
    s = torch.randn(2, 4)
    with torch.no_grad():
        pred = head(vol, tdoa, c, s)
        pred_no_s = head(vol, tdoa, c, None)  # head fills with zeros
    assert pred.shape == (2, 3)
    assert pred_no_s.shape == (2, 3)
    assert torch.all(torch.isfinite(pred))


# ---------------------------------------------------------------------------
# 3. End-to-end training
# ---------------------------------------------------------------------------


def _v2_encoder_for_smoke() -> V2FusionEncoder:
    torch.manual_seed(0)
    return V2FusionEncoder(feature_dim=32, embed_dim=32, n_heads=2)


def test_train_v4_end_to_end_with_synthetic_labels() -> None:
    segments = _truncated_segments(max_seconds=5.0)
    v2_cfg = _smoke_v2_cfg()
    grid = _smoke_grid()
    encoder = _v2_encoder_for_smoke()

    # Synthetic spatial labels per recording: deterministic but non-trivial.
    rng = np.random.default_rng(0)
    overrides = {}
    for s in segments:
        overrides.setdefault(
            s.recording_id,
            tuple(rng.uniform([-0.3, -0.3, 0.0], [0.3, 0.3, 0.4]).tolist()),
        )

    samples = precompute_v4_samples(
        encoder,
        segments,
        v2_cfg=v2_cfg,
        grid=grid,
        spatial_label_overrides=overrides,
    )
    assert len(samples) >= 4, f"need ≥4 samples for a meaningful split; got {len(samples)}"

    cfg = V4Config(
        cnn_feature_dim=32,
        tdoa_feature_dim=16,
        hidden_dim=32,
        n_heads_tdoa=2,
        epochs=3,
        batch_size=4,
        val_ratio=0.5,
        seed=0,
    )
    result = train_v4_localization(samples, cfg=cfg, grid=grid)

    assert len(result.train_loss_history) == cfg.epochs
    assert len(result.val_loss_history) == cfg.epochs
    assert all(np.isfinite(result.train_loss_history))
    assert all(np.isfinite(result.val_loss_history))

    # Held-out recordings disjoint.
    assert set(result.train_recording_ids).isdisjoint(set(result.val_recording_ids))

    # MAE finite + bounded by the grid extent.
    if result.val_predictions.shape[0] > 0:
        assert np.isfinite(result.val_mae_3d)
        assert np.isfinite(result.val_p95_3d)
        assert result.val_mae_3d < 5.0  # grid is ~1m³ — anything ≥5m means broken
        # Diagnostic fields populated.
        assert result.val_init_xyz.shape == result.val_predictions.shape
        assert result.val_residuals.shape == result.val_predictions.shape
        # Residual is bounded by ±residual_scale_m (default 0.20) per dim.
        assert float(np.abs(result.val_residuals).max()) <= 0.20 + 1e-5
        # Per-recording breakdown is populated for every val recording.
        assert set(result.val_recording_breakdown.keys()) == set(result.val_recording_ids)
        for _k, row in result.val_recording_breakdown.items():
            assert row["n"] >= 1
            assert np.isfinite(row["mae_3d"])
            assert len(row["target_xyz"]) == 3
            assert len(row["pred_xyz_mean"]) == 3


def test_train_v4_unconditional_a3_ablation() -> None:
    segments = _truncated_segments(max_seconds=5.0)
    v2_cfg = _smoke_v2_cfg()
    grid = _smoke_grid()
    encoder = _v2_encoder_for_smoke()
    rng = np.random.default_rng(1)
    overrides = {
        s.recording_id: tuple(rng.uniform([-0.3, -0.3, 0.0], [0.3, 0.3, 0.4]).tolist())
        for s in segments
    }
    samples = precompute_v4_samples(
        encoder, segments, v2_cfg=v2_cfg, grid=grid, spatial_label_overrides=overrides
    )

    cfg = V4Config(
        cnn_feature_dim=32,
        tdoa_feature_dim=16,
        hidden_dim=32,
        epochs=2,
        batch_size=4,
        val_ratio=0.5,
        unconditional=True,
        seed=0,
    )
    result = train_v4_localization(samples, cfg=cfg, grid=grid)
    assert result.unconditional is True
    assert all(np.isfinite(result.train_loss_history))
    if result.val_predictions.shape[0] > 0:
        assert np.isfinite(result.val_mae_3d)
