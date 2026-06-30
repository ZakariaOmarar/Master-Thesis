"""Tests for per-knock V4 expansion + array-footprint outlier classification.

These run without the recorded datasets: the builder test synthesises a tiny
two-knock recording so the leakage / multi-sample behaviour is exercised in CI.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import torch

from src.data import DataSegment
from src.ingestion.test_dataset_loader import TestDatasetSegment
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.context.v2_ssl import V2SSLConfig
from src.modeling.localization import (
    C_PLASTIC_3DP_MS,
    GridSpec,
    KnockEventConfig,
    SyntheticArraySpec,
    array_sensor_xyz,
    assert_no_position_leak,
    classify_position,
    classify_positions,
    compute_srp_phat_volume,
    generate_synthetic_knock_samples,
    precompute_v4_knock_event_samples,
    srp_peak_sharpness,
    train_v4_localization,
)
from src.modeling.localization.classical import gcc_phat
from src.modeling.localization.v4_features import bandpass_filter, compute_accel_tdoa_tokens
from src.modeling.localization.v4_trainer import V4Config


# --------------------------------------------------------------------------- #
# array_geometry
# --------------------------------------------------------------------------- #
def _unit_cube_sensors() -> np.ndarray:
    return np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=np.float64,
    )


def test_classify_position_inside_and_outside() -> None:
    sensors = _unit_cube_sensors()
    inside = classify_position([0.5, 0.5, 0.5], sensors, margin_m=0.0)
    assert inside.inside and inside.signed_distance_m < 0
    assert inside.method == "convex_hull"

    outside = classify_position([2.0, 0.5, 0.5], sensors, margin_m=0.0)
    assert not outside.inside
    assert outside.signed_distance_m > 0.9  # ~1 m beyond the +x face


def test_classify_position_margin_admits_near_boundary() -> None:
    sensors = _unit_cube_sensors()
    p = [1.03, 0.5, 0.5]  # 3 cm outside the +x face
    assert not classify_position(p, sensors, margin_m=0.0).inside
    assert classify_position(p, sensors, margin_m=0.05).inside


def test_classify_position_coplanar_falls_back_to_bbox() -> None:
    # All sensors at z=0 → degenerate 3-D hull → bounding-box fallback.
    coplanar = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    v = classify_position([0.5, 0.5, 0.2], coplanar, margin_m=0.0)
    assert v.method == "bounding_box"
    # z=0.2 is outside the zero-thickness box, so it reads as just-outside.
    assert v.signed_distance_m > 0


def test_array_sensor_xyz_unions_modalities() -> None:
    mic = np.zeros((4, 3))
    vib = np.ones((3, 3))
    assert array_sensor_xyz(mic, vib).shape == (7, 3)


def test_classify_positions_aggregates_inside_if_any() -> None:
    sensors = _unit_cube_sensors()
    far = sensors + np.array([10.0, 0, 0])  # a second, shifted array
    pos = (0.5, 0.5, 0.5)
    records = [(pos, sensors[:4], sensors[4:]), (pos, far[:4], far[4:])]
    out = classify_positions(records, margin_m=0.0)
    key = (0.5, 0.5, 0.5)
    assert out[key]["inside"] is True  # inside for the first array
    assert out[key]["n_recordings"] == 2


# --------------------------------------------------------------------------- #
# leak guard
# --------------------------------------------------------------------------- #
def _fake_sample(pos):
    from src.modeling.localization import V4Sample

    return V4Sample(
        srp_volume=np.zeros((4, 4, 2), np.float32),
        tdoa_tokens=np.zeros((0, 8), np.float32),
        context=np.zeros(8, np.float32),
        x_for_v3=np.zeros(8, np.float32),
        target_xyz=np.asarray(pos, np.float32),
        scada=None, mode_label=None, recording_id="r", source_dir="d", dataset_id="x",
    )


def test_assert_no_position_leak_raises_on_overlap() -> None:
    tr = [_fake_sample((0.1, 0.0, 0.0))]
    va = [_fake_sample((0.1, 0.0, 0.0))]
    try:
        assert_no_position_leak(tr, va)
        raised = False
    except AssertionError:
        raised = True
    assert raised
    # disjoint positions must pass (no exception raised)
    assert_no_position_leak([_fake_sample((0.1, 0, 0))], [_fake_sample((0.2, 0, 0))])


# --------------------------------------------------------------------------- #
# per-knock builder on a synthetic two-knock recording
# --------------------------------------------------------------------------- #
def _two_knock_segment() -> TestDatasetSegment:
    rng = np.random.default_rng(0)
    mic_sr, accel_sr = 16000, 376
    dur = 2.0
    n_mic = int(dur * mic_sr)
    n_acc = int(dur * accel_sr)
    n_mic_ch, n_acc_ch = 4, 4

    mic = (0.01 * rng.standard_normal((n_mic_ch, n_mic))).astype(np.float32)
    acc = (0.01 * rng.standard_normal((n_acc_ch, n_acc))).astype(np.float32)
    for t in (0.5, 1.2):  # two distinct knocks
        m0 = int(t * mic_sr)
        mic[:, m0 : m0 + int(0.03 * mic_sr)] += 8.0 * rng.standard_normal((n_mic_ch, int(0.03 * mic_sr))).astype(np.float32)
        a0 = int(t * accel_sr)
        acc[:, a0 : a0 + 3] += 8.0 * rng.standard_normal((n_acc_ch, 3)).astype(np.float32)

    # Naive datetime is fine: DataSegment.from_arrays attaches UTC itself.
    # (Avoids datetime.UTC, which is Python 3.11+ only — the box runs 3.10.)
    seg = DataSegment.from_arrays(
        mic_data=mic, accel_data=acc,
        start_time=datetime(2026, 1, 1),
        mic_sr=mic_sr, accel_sr=accel_sr, metadata={},
    )
    mic_xyz = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0.2, 0.2, 0]], np.float64)
    vib_xyz = np.array([[0.05, 0.05, 0], [0.15, 0.05, 0], [0.05, 0.15, 0], [0.15, 0.15, 0]], np.float64)
    return TestDatasetSegment(
        segment=seg, mic_positions=mic_xyz, vib_positions=vib_xyz,
        mic_ids=("A", "B", "C", "D"), vib_ids=("a", "b", "c", "d"),
        mode_label=None, op_condition=None, spatial_label=(0.1, 0.1, 0.0),
        dataset_id="d5", recording_id="synthetic_knock", source_dir="syn",
        is_anomaly=True,
    )


def _smoke_v2() -> tuple[V2FusionEncoder, V2SSLConfig]:
    torch.manual_seed(0)
    cfg = V2SSLConfig(
        window_seconds=0.5, window_stride_seconds=0.25,
        feature_dim=32, embed_dim=32, n_heads=2, proj_dim=16,
        epochs=1, batch_size=8, val_ratio=0.5,
        n_mels=32, n_fft=256, hop_length=128, use_cwt=False,
        gain_jitter_db=0.0, channel_dropout_p=0.0,
        spec_augment_freq_mask=0, spec_augment_time_mask=0, seed=0,
    )
    return V2FusionEncoder(feature_dim=32, embed_dim=32, n_heads=2), cfg


def test_knock_builder_emits_multiple_samples_per_recording() -> None:
    encoder, v2_cfg = _smoke_v2()
    grid = GridSpec(lo=(-0.1, -0.1, -0.05), hi=(0.3, 0.3, 0.2), n=(8, 8, 4))
    seg = _two_knock_segment()
    samples = precompute_v4_knock_event_samples(
        encoder, [seg], v2_cfg=v2_cfg, grid=grid,
        cfg=KnockEventConfig(crop_seconds=0.12, noise_floor_mult=3.0),
        device="cpu",
    )
    # Two knocks → at least two transient-centred samples.
    assert len(samples) >= 2
    for s in samples:
        assert s.srp_volume.shape == (8, 8, 4)
        assert np.all(np.isfinite(s.srp_volume))
        assert s.tdoa_tokens.shape[1] == 8
        assert tuple(np.round(s.target_xyz, 3)) == (0.1, 0.1, 0.0)
        assert np.isfinite(s.context).all()
    # Event centres should bracket the two injected knocks (~0.5 s and ~1.2 s).
    centres = sorted(s.window_start_s for s in samples)
    assert any(abs(c - 0.5) < 0.2 for c in centres)
    assert any(abs(c - 1.2) < 0.2 for c in centres)
    # PSR is populated and finite.
    assert all(np.isfinite(s.srp_psr) and s.srp_psr >= 1.0 for s in samples)


# --------------------------------------------------------------------------- #
# front-end: oversampled GCC + PSR
# --------------------------------------------------------------------------- #
def _impulse_in_noise(mic_xyz, src, fs, T, *, c=343.0, seed=1):
    rng = np.random.default_rng(seed)
    burst = int(0.005 * fs)
    impulse = rng.standard_normal(burst) * 5.0
    sig = 0.01 * rng.standard_normal((mic_xyz.shape[0], T))
    for m in range(mic_xyz.shape[0]):
        d = float(np.linalg.norm(src - mic_xyz[m]))
        s0 = int(0.4 * fs) + int(round(d / c * fs))
        if s0 + burst <= T:
            sig[m, s0 : s0 + burst] += impulse
    return sig


def test_srp_peak_sharpness_sharp_vs_diffuse() -> None:
    fs, T = 16000, 16000
    mic_xyz = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0]], float)
    grid = GridSpec(lo=(-0.05, -0.05, -0.05), hi=(0.15, 0.15, 0.15), n=(11, 11, 11))
    sharp = compute_srp_phat_volume(
        _impulse_in_noise(mic_xyz, np.array([0.05, 0.05, 0.05]), fs, T), mic_xyz, fs, grid
    )
    rng = np.random.default_rng(2)
    diffuse = compute_srp_phat_volume(
        rng.standard_normal((4, T)).astype(np.float64), mic_xyz, fs, grid
    )
    assert srp_peak_sharpness(sharp) > srp_peak_sharpness(diffuse)


def test_gcc_oversample_shape_and_accuracy() -> None:
    fs, T = 16000, 16000
    mic_xyz = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0]], float)
    src = np.array([0.05, 0.05, 0.05])
    grid = GridSpec(lo=(-0.05, -0.05, -0.05), hi=(0.15, 0.15, 0.15), n=(11, 11, 11))
    sig = _impulse_in_noise(mic_xyz, src, fs, T)
    v1 = compute_srp_phat_volume(sig, mic_xyz, fs, grid, gcc_oversample=1)
    v4 = compute_srp_phat_volume(sig, mic_xyz, fs, grid, gcc_oversample=4)
    assert v1.shape == v4.shape == (11, 11, 11)
    assert np.all(np.isfinite(v4)) and v4.max() <= 1.0 + 1e-6

    ax = grid.axes()
    def _peak(v):
        idx = np.unravel_index(int(np.argmax(v)), v.shape)
        return np.array([ax[0][idx[0]], ax[1][idx[1]], ax[2][idx[2]]])
    # Oversampling must not regress the peak location on a clean impulse.
    assert np.linalg.norm(_peak(v4) - src) <= np.linalg.norm(_peak(v1) - src) + 0.02


# --------------------------------------------------------------------------- #
# synthetic generator (pysoundlocalization forward model)
# --------------------------------------------------------------------------- #
def test_synthetic_generator_srp_peak_near_source() -> None:
    spec = SyntheticArraySpec(
        dataset_id="d5",
        mic_xyz=np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0.2, 0.2, 0],
                          [0.1, 0.1, 0.15]], float),
        vib_xyz=np.array([[0.05, 0.05, 0], [0.15, 0.05, 0], [0.05, 0.15, 0],
                          [0.15, 0.15, 0]], float),
        mic_fs=16000, accel_fs=376,
    )
    grid = GridSpec(lo=(-0.05, -0.05, -0.05), hi=(0.25, 0.25, 0.2), n=(16, 16, 10))
    samples = generate_synthetic_knock_samples(
        [spec], grid, c_dim=16, n_positions_per_array=6, snr_db=20.0, seed=0,
    )
    assert len(samples) >= 4
    ax = grid.axes()
    errs = []
    for s in samples:
        idx = np.unravel_index(int(np.argmax(s.srp_volume)), s.srp_volume.shape)
        peak = np.array([ax[0][idx[0]], ax[1][idx[1]], ax[2][idx[2]]])
        errs.append(float(np.linalg.norm(peak - s.target_xyz)))
        assert s.context.shape == (16,) and not s.context.any()  # zero context
    # The classical SRP peak should be near the true source for most samples.
    assert np.median(errs) < 0.06


# --------------------------------------------------------------------------- #
# trainer hooks: heatmap aux loss + warm-start
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# new front-end techniques harvested from pysoundlocalization
# --------------------------------------------------------------------------- #
def test_gcc_phat_variants_recover_lag() -> None:
    rng = np.random.default_rng(0)
    N, lag, burst = 512, 7, rng.standard_normal(32)
    x_j = np.zeros(N)
    x_i = np.zeros(N)
    x_j[100:132] = burst
    x_i[100 + lag : 132 + lag] = burst
    for kw in (dict(), dict(phat_beta=0.7), dict(linear=True), dict(oversample=4)):
        os_ = kw.get("oversample", 1)
        g = gcc_phat(x_i, x_j, max_delay_samples=32, **kw)
        assert g.shape[0] == 2 * 32 * os_ + 1
        est = (int(np.argmax(g)) - 32 * os_) / os_
        assert min(abs(est - lag), abs(est + lag)) <= 1.0, f"{kw}: est={est}"


def test_accel_tdoa_subsample_beats_integer_quantization() -> None:
    """At a low accel rate the integer-lag TDOA quantises to ~±c/fs; oversampling
    + parabolic must recover a finer, non-quantised path_diff for a sub-sample
    true delay."""
    fs = 376.0
    n_ch, T = 4, 64
    xyz = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0]], float)
    rng = np.random.default_rng(0)
    # One coherent burst, shifted by a genuinely SUB-sample fractional delay
    # across channels via fractional resampling.
    base = rng.standard_normal(T)
    accel = np.stack([np.interp(np.arange(T) - 0.3 * k, np.arange(T), base)
                      for k in range(n_ch)])
    # Old behaviour (the bug): integer argmax, no oversample, no parabolic.
    md = max(1, int(round(max(1.0 / fs, (0.1414 / C_PLASTIC_3DP_MS) * 1.5) * fs)))
    pairs = [(a, b) for a in range(n_ch) for b in range(a + 1, n_ch)]
    old = np.array([
        (int(np.argmax(gcc_phat(accel[i], accel[j], max_delay_samples=md))) - md)
        / fs * C_PLASTIC_3DP_MS
        for i, j in pairs
    ])
    fine = compute_accel_tdoa_tokens(accel, xyz, fs=fs, gcc_oversample=16)
    assert fine.shape == (len(pairs), 8)
    step = C_PLASTIC_3DP_MS / fs  # one integer-lag path-difference quantum
    frac_old = np.abs((old / step) - np.round(old / step))
    frac_fine = np.abs((fine[:, 0] / step) - np.round(fine[:, 0] / step))
    assert frac_old.max() < 1e-6   # old argmax is grid-locked (the bug)
    assert frac_fine.max() > 1e-3  # fixed version resolves between grid lines
    assert np.all(np.isfinite(fine))


def test_bandpass_filter_attenuates_out_of_band() -> None:
    fs, T = 16000, 16000
    t = np.arange(T) / fs
    in_band = np.sin(2 * np.pi * 1000 * t)        # 1 kHz, inside [200, 6000]
    out_band = np.sin(2 * np.pi * 50 * t)         # 50 Hz, below the low cut
    sig = np.stack([in_band + out_band])          # (1, T)
    filt = bandpass_filter(sig, fs, 200.0, 6000.0)
    assert filt.shape == sig.shape and np.all(np.isfinite(filt))
    # In-band energy retained, out-of-band (50 Hz) strongly attenuated.
    assert np.std(filt[0]) > 0.5 * np.std(in_band)
    only_low = bandpass_filter(np.stack([out_band]), fs, 200.0, 6000.0)
    assert np.std(only_low[0]) < 0.2 * np.std(out_band)


def test_tta_crops_multiplies_samples() -> None:
    encoder, v2_cfg = _smoke_v2()
    grid = GridSpec(lo=(-0.1, -0.1, -0.05), hi=(0.3, 0.3, 0.2), n=(8, 8, 4))
    seg = _two_knock_segment()
    one = precompute_v4_knock_event_samples(
        encoder, [seg], v2_cfg=v2_cfg, grid=grid,
        cfg=KnockEventConfig(crop_seconds=0.12), device="cpu",
    )
    three = precompute_v4_knock_event_samples(
        encoder, [seg], v2_cfg=v2_cfg, grid=grid,
        cfg=KnockEventConfig(crop_seconds=0.12, crops_per_knock=3, crop_jitter_seconds=0.02),
        device="cpu",
    )
    assert len(three) == 3 * len(one)  # 3 crops per detected knock
    # Multi-scale: one centred crop per width (no jitter).
    ms = precompute_v4_knock_event_samples(
        encoder, [seg], v2_cfg=v2_cfg, grid=grid,
        cfg=KnockEventConfig(crop_scales_seconds=(0.08, 0.12, 0.20)), device="cpu",
    )
    assert len(ms) == 3 * len(one)
    assert all(s.window_start_s in {s2.window_start_s for s2 in one} for s in ms), \
        "multi-scale crops must stay centred on the knock (no time offset)"
    # Bandpass option runs and yields finite SRP volumes.
    bp = precompute_v4_knock_event_samples(
        encoder, [seg], v2_cfg=v2_cfg, grid=grid,
        cfg=KnockEventConfig(crop_seconds=0.12, bandpass_hz=(200.0, 6000.0)),
        device="cpu",
    )
    assert bp and all(np.all(np.isfinite(s.srp_volume)) for s in bp)


def test_synthetic_reverb_still_localizes() -> None:
    spec = SyntheticArraySpec(
        dataset_id="d5",
        mic_xyz=np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0.2, 0.2, 0],
                          [0.1, 0.1, 0.15]], float),
        vib_xyz=np.array([[0.05, 0.05, 0], [0.15, 0.05, 0], [0.05, 0.15, 0],
                          [0.15, 0.15, 0]], float),
        mic_fs=16000, accel_fs=376,
    )
    grid = GridSpec(lo=(-0.05, -0.05, -0.05), hi=(0.25, 0.25, 0.2), n=(16, 16, 10))
    samples = generate_synthetic_knock_samples(
        [spec], grid, c_dim=16, n_positions_per_array=6, snr_db=20.0,
        n_reflections=4, reflection_gain=0.5, seed=0,
    )
    assert len(samples) >= 4
    ax = grid.axes()
    errs = []
    for s in samples:
        idx = np.unravel_index(int(np.argmax(s.srp_volume)), s.srp_volume.shape)
        peak = np.array([ax[0][idx[0]], ax[1][idx[1]], ax[2][idx[2]]])
        errs.append(float(np.linalg.norm(peak - s.target_xyz)))
    # Reflections smear the peak but it must still track the source loosely.
    assert np.median(errs) < 0.12


def test_train_v4_heatmap_aux_and_warmstart_run() -> None:
    encoder, v2_cfg = _smoke_v2()
    grid = GridSpec(lo=(-0.1, -0.1, -0.05), hi=(0.3, 0.3, 0.2), n=(8, 8, 4))
    # Two recordings at different positions so a recording split is valid.
    segs = []
    for i, pos in enumerate([(0.1, 0.1, 0.0), (0.2, 0.05, 0.05)]):
        seg = _two_knock_segment()
        segs.append(
            seg.__class__(
                segment=seg.segment, mic_positions=seg.mic_positions,
                vib_positions=seg.vib_positions, mic_ids=seg.mic_ids, vib_ids=seg.vib_ids,
                mode_label=None, op_condition=None, spatial_label=pos,
                dataset_id="d5", recording_id=f"rec{i}", source_dir=f"d{i}", is_anomaly=True,
            )
        )
    samples = precompute_v4_knock_event_samples(
        encoder, segs, v2_cfg=v2_cfg, grid=grid,
        cfg=KnockEventConfig(crop_seconds=0.12), device="cpu",
    )
    assert len(samples) >= 4
    base = V4Config(cnn_feature_dim=32, tdoa_feature_dim=16, hidden_dim=32,
                    epochs=2, batch_size=4, val_ratio=0.5, seed=0, device="cpu")
    # Heatmap auxiliary loss path runs and produces finite losses.
    res_aux = train_v4_localization(
        samples, cfg=replace(base, heatmap_aux_weight=0.1), grid=grid
    )
    assert all(np.isfinite(res_aux.train_loss_history))
    # The headline val_mae_3d is event-aggregated (per-recording).  With val
    # predictions present it must be a finite, non-negative number and there must
    # be at least one per-recording aggregated error behind it (aggregation never
    # silently produces an empty/NaN headline).
    if res_aux.val_predictions.shape[0] > 0:
        assert np.isfinite(res_aux.val_mae_3d)
        assert res_aux.val_mae_3d >= 0.0
        assert res_aux.val_agg_errors.size >= 1
    # Warm-start from the trained head loads without error and trains finite.
    state = {k: v.detach().cpu().clone() for k, v in res_aux.head.state_dict().items()}
    res_ws = train_v4_localization(samples, cfg=base, grid=grid, init_state=state)
    assert all(np.isfinite(res_ws.train_loss_history))
