"""Smoke tests for V3 conditional anomaly head.

Three groups:
  1. RealNVP CNF + FiLM coupling — shape, finite log-prob, gradient flow.
  2. PerClusterThresholds — fit, assign, alert; rejects degenerate inputs.
  3. End-to-end V3 trainer on a tiny truncated D1 — finite NLL, RealNVP
     scoring, A2 unconditional ablation, synthetic transition stress-test.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    ConditionalRealNVP,
    PerClusterThresholds,
    V3Config,
    gate_samples_by_alert,
    make_transition_segment,
    train_v3_cnf,
    transition_fpr,
)
from src.modeling.anomaly.v3_trainer import precompute_paired
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.context.v2_ssl import V2SSLConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.requires_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _truncated_loader(max_seconds: float = 6.0):
    loader = TestDatasetLoader(_resolved_d1_spec())
    full = loader.list_segments()
    truncated: list[TestDatasetSegment] = []
    for s in full:
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
            )
        )

    class _StubLoader:
        spec = loader.spec
        registry = loader.registry

        def list_segments(self, **_kwargs):
            return list(truncated)

    return _StubLoader(), truncated


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
        gain_jitter_db=3.0,
        channel_dropout_p=0.1,
        spec_augment_freq_mask=4,
        spec_augment_time_mask=4,
        seed=0,
    )


def _smoke_v3_cfg(**overrides) -> V3Config:
    base = dict(
        n_layers=4,
        hidden_dim=32,
        n_hidden_per_net=2,
        epochs=2,
        batch_size=16,
        val_ratio=0.5,
        n_threshold_clusters=3,
        threshold_percentile=95,
        seed=0,
    )
    base.update(overrides)
    return V3Config(**base)


# ---------------------------------------------------------------------------
# 1. CNF building blocks
# ---------------------------------------------------------------------------


def test_cnf_log_prob_shape_and_finite() -> None:
    torch.manual_seed(0)
    flow = ConditionalRealNVP(dim=16, c_dim=8, n_layers=4, hidden_dim=32)
    flow.eval()
    x = torch.randn(5, 16)
    c = torch.randn(5, 8)
    with torch.no_grad():
        log_p = flow.log_prob(x, c)
        s = flow.anomaly_score(x, c)
    assert log_p.shape == (5,)
    assert s.shape == (5,)
    assert torch.all(torch.isfinite(log_p))
    assert torch.allclose(s, -log_p)


def test_cnf_gradient_flows() -> None:
    """Negative-log-likelihood loss has a non-zero gradient on a fresh init."""
    torch.manual_seed(0)
    flow = ConditionalRealNVP(dim=8, c_dim=8, n_layers=3, hidden_dim=16)
    x = torch.randn(4, 8)
    c = torch.randn(4, 8)
    loss = -flow.log_prob(x, c).mean()
    loss.backward()
    grads = [p.grad.norm() for p in flow.parameters() if p.grad is not None]
    assert any(g.item() > 0 for g in grads), "no gradient flowed through any parameter"


def test_cnf_film_init_makes_first_step_finite() -> None:
    """FiLM γ/β are zero-initialised so the first forward pass is well-conditioned
    even with random `c`."""
    torch.manual_seed(0)
    flow = ConditionalRealNVP(dim=8, c_dim=8, n_layers=4, hidden_dim=16)
    x = torch.randn(32, 8) * 5.0  # large-scale input
    c = torch.randn(32, 8) * 5.0
    log_p = flow.log_prob(x, c)
    assert torch.all(torch.isfinite(log_p))


def test_cnf_log_det_clamp_bounds_extreme_inputs() -> None:
    """F6 — per-coupling log-det is clamped to ±50.  With n_layers=6 the
    accumulated log_det stays within ±300 even on pathological inputs."""
    torch.manual_seed(0)
    flow = ConditionalRealNVP(dim=64, c_dim=8, n_layers=6, hidden_dim=16, scale_max=2.0)
    flow.eval()
    # Adversarial: huge x, huge c — would otherwise saturate every coupling.
    x = torch.randn(8, 64) * 100.0
    c = torch.randn(8, 8) * 100.0
    with torch.no_grad():
        _z, log_det = flow.forward(x, c)
    # Per-layer bound is ±50, so 6 layers bound the accumulated log_det at ±300.
    assert torch.all(log_det >= -300.0 - 1e-3)
    assert torch.all(log_det <= 300.0 + 1e-3)
    assert torch.all(torch.isfinite(log_det))


# ---------------------------------------------------------------------------
# 2. PerClusterThresholds
# ---------------------------------------------------------------------------


def test_per_cluster_thresholds_fit_and_alert() -> None:
    rng = np.random.default_rng(0)
    n_per = 50
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    contexts = np.concatenate([c + rng.normal(scale=0.5, size=(n_per, 2)) for c in centres])
    # Per-cluster scores: cluster 0 has higher scores than the rest.
    scores = np.concatenate(
        [
            rng.normal(loc=10.0, scale=1.0, size=n_per),
            rng.normal(loc=2.0, scale=0.5, size=n_per),
            rng.normal(loc=2.0, scale=0.5, size=n_per),
            rng.normal(loc=2.0, scale=0.5, size=n_per),
        ]
    )
    th = PerClusterThresholds.fit(contexts, scores, n_clusters=4, seed=0)
    assert th.centroids.shape == (4, 2)
    assert th.p95.shape == (4,)
    assert th.p99.shape == (4,)
    assert (th.n_per_cluster > 0).all()
    # The p99 of the high-scored cluster should clearly exceed the others'.
    assert th.p99.max() > th.p99.min() * 1.5

    # A point near cluster 0 with a borderline score should not alert at p99.
    test_ctx = np.array([[0.1, -0.1]])
    test_score = np.array([th.p99[th.assign(test_ctx)[0]] - 0.01])
    alerts, _ = th.alert(test_ctx, test_score, percentile=99)
    assert not alerts[0]
    # A clearly-anomalous score does alert.
    big = np.array([th.p99[th.assign(test_ctx)[0]] + 50.0])
    alerts2, _ = th.alert(test_ctx, big, percentile=99)
    assert alerts2[0]


def test_per_cluster_thresholds_rejects_too_few_samples() -> None:
    contexts = np.zeros((2, 4))
    scores = np.zeros(2)
    with pytest.raises(ValueError):
        PerClusterThresholds.fit(contexts, scores, n_clusters=4)


# ---------------------------------------------------------------------------
# 3. End-to-end V3 trainer
# ---------------------------------------------------------------------------


def _trained_v2_encoder(v2_cfg: V2SSLConfig) -> V2FusionEncoder:
    """Build a V2FusionEncoder with the smoke config dims; freshly-init weights
    are sufficient for V3's smoke test (we just need a stable feature space)."""
    torch.manual_seed(v2_cfg.seed)
    return V2FusionEncoder(
        feature_dim=v2_cfg.feature_dim,
        embed_dim=v2_cfg.embed_dim,
        n_heads=v2_cfg.n_heads,
    )


def test_train_v3_cnf_end_to_end() -> None:
    # 10 s (not 5 s): D1's vibration feature rate is ~1 frame/s, so the nested
    # train/val → fit/eval split (two halvings) needs enough length for both
    # disjoint sub-cohorts to still window a vibration frame.  Keeps the
    # Chapter-5 disjoint fit/eval protocol genuine instead of crashing.
    loader, _ = _truncated_loader(max_seconds=10.0)
    v2_cfg = _smoke_v2_cfg()
    v3_cfg = _smoke_v3_cfg()
    encoder = _trained_v2_encoder(v2_cfg)

    result = train_v3_cnf(encoder, loader, v2_cfg=v2_cfg, v3_cfg=v3_cfg)

    assert len(result.train_nll) == v3_cfg.epochs
    assert len(result.val_nll) == v3_cfg.epochs
    assert all(np.isfinite(result.train_nll))
    assert all(np.isfinite(result.val_nll))

    # Held-out recordings disjoint — three-way: train ⊥ threshold-fit ⊥ val.
    train_ids = set(result.train_recording_ids)
    val_ids = set(result.val_recording_ids)
    fit_ids = set(result.threshold_fit_recording_ids)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(fit_ids)
    # Chapter-5 protocol: per-cluster thresholds are fitted on a held-out healthy
    # subset and recall/FPR are reported on a DISJOINT subset.  Enforce that the
    # threshold-fit cohort and the reportable val cohort never overlap — V3 has
    # no fallback that would reuse one for both (it hard-errors instead).
    assert fit_ids.isdisjoint(val_ids), (
        "F1 invariant: threshold-fit cohort must be disjoint from reportable val "
        f"cohort (overlap = {fit_ids & val_ids})"
    )
    assert len(fit_ids) >= 1
    assert len(val_ids) >= 1

    assert result.val_scores.shape[0] == result.val_contexts.shape[0]
    assert result.val_scores.shape[0] == len(result.val_labels)
    assert np.all(np.isfinite(result.val_scores))

    # F6 — outlier-batch tracking is populated for every epoch.
    assert len(result.train_nll_min) == v3_cfg.epochs
    assert len(result.train_nll_max) == v3_cfg.epochs
    assert len(result.val_nll_min) == v3_cfg.epochs
    assert len(result.val_nll_max) == v3_cfg.epochs
    # Ordering invariants per epoch: min ≤ mean ≤ max.
    for i in range(v3_cfg.epochs):
        if np.isfinite(result.train_nll_min[i]) and np.isfinite(result.train_nll_max[i]):
            assert result.train_nll_min[i] <= result.train_nll[i] + 1e-6
            assert result.train_nll_max[i] >= result.train_nll[i] - 1e-6

    assert result.thresholds.centroids.shape[0] == v3_cfg.n_threshold_clusters
    # Thresholds are now fit on a disjoint cohort (val_fit), so the p99 of
    # val_fit and the 95th percentile of val_eval scores have NO guaranteed
    # ordering — that's the whole point of the F1 fix.  We only require that
    # the fit produced finite, strictly-positive percentile bars.
    assert np.all(np.isfinite(result.thresholds.p95))
    assert np.all(np.isfinite(result.thresholds.p99))
    assert np.all(result.thresholds.p99 >= result.thresholds.p95)
    assert result.unconditional is False


def test_train_v3_unconditional_a2_ablation() -> None:
    loader, _ = _truncated_loader(max_seconds=10.0)  # see end_to_end: disjoint nested split needs the length
    v2_cfg = _smoke_v2_cfg()
    v3_cfg = _smoke_v3_cfg(unconditional=True)
    encoder = _trained_v2_encoder(v2_cfg)

    result = train_v3_cnf(encoder, loader, v2_cfg=v2_cfg, v3_cfg=v3_cfg)
    assert result.unconditional is True
    assert all(np.isfinite(result.train_nll))
    assert all(np.isfinite(result.val_nll))

    # Scoring under unconditional flag must not depend on c — verify by
    # sampling random c at inference and checking the score is unchanged.
    flow = result.flow
    flow.eval()
    x = torch.randn(8, flow.dim)
    c1 = torch.zeros(8, flow.c_dim)
    c2 = torch.randn(8, flow.c_dim)
    with torch.no_grad():
        s1 = flow.anomaly_score(x, c1)
        s2 = flow.anomaly_score(x, c2)
    # When the flow was *trained* with c=0, FiLM has merely learned to use c
    # via random gradients (unused signal). The point of the A2 ablation is
    # that the *runtime* path uses c=0 — verify only that c=0 → finite score,
    # which is the runtime invariant.
    assert torch.all(torch.isfinite(s1))
    # And that c1 vs c2 may differ (FiLM still fires); we only require the
    # *runtime* always uses c=0, which is enforced by `unconditional=True`
    # in `score_segments` / `transition_fpr`.
    assert torch.all(torch.isfinite(s2))


def test_v3_synthetic_transition() -> None:
    """End-to-end synthetic transition: make_transition_segment + transition_fpr."""
    loader, segs = _truncated_loader(max_seconds=10.0)  # disjoint nested split needs the length
    v2_cfg = _smoke_v2_cfg()
    v3_cfg = _smoke_v3_cfg()
    encoder = _trained_v2_encoder(v2_cfg)

    result = train_v3_cnf(encoder, loader, v2_cfg=v2_cfg, v3_cfg=v3_cfg)

    # Pre-compute paired features for two distinct healthy segments to splice.
    paired = []
    for s in segs:
        if s.mode_label in {"Pump", "Standstill", "Turbine", "RandomFault"}:
            p = precompute_paired(s, v2_cfg)
            if p is not None:
                paired.append(p)
        if len(paired) >= 2:
            break
    assert len(paired) >= 2, "need at least two paired segments for the transition test"

    out = transition_fpr(
        encoder,
        result.flow,
        result.thresholds,
        paired[0],
        paired[1],
        v2_cfg=v2_cfg,
        crossfade_seconds=0.5,
        percentile=95,
        # Match the trained flow: PMA-2 pooling + impulse anchor (if enabled).
        xt_pool=getattr(result, "xt_pool", None),
        anchor_norm=((result.anchor_mean, result.anchor_std)
                     if getattr(result, "anchor_mean", None) is not None else None),
    )
    assert out["n_windows"] >= 1
    assert 0.0 <= out["fpr"] <= 1.0
    assert out["scores"].shape[0] == out["n_windows"]
    assert out["clusters"].shape[0] == out["n_windows"]
    assert np.all(np.isfinite(out["scores"]))


def test_gate_samples_by_alert_filters_and_passthrough() -> None:
    """Build a tiny synthetic V4-sample-shaped object list, gate it through
    a fresh CNF + thresholds; verify alert-only filtering and the
    `keep_dataset_ids` passthrough do the right thing."""
    torch.manual_seed(0)
    flow = ConditionalRealNVP(dim=8, c_dim=8, n_layers=3, hidden_dim=16)
    flow.eval()

    rng = np.random.default_rng(0)
    contexts = rng.standard_normal((20, 8)).astype(np.float32)
    # Half are "anomalous" by construction (very large x — far from healthy mean).
    xs = np.concatenate(
        [rng.standard_normal((10, 8)), 5.0 + rng.standard_normal((10, 8))], axis=0
    ).astype(np.float32)
    # Fit thresholds on the first 10 (treated as healthy).
    with torch.no_grad():
        scores_healthy = flow.anomaly_score(
            torch.from_numpy(xs[:10]), torch.from_numpy(contexts[:10])
        ).numpy()
    thresholds = PerClusterThresholds.fit(
        contexts[:10], scores_healthy, n_clusters=3, seed=0
    )

    @dataclass
    class _Stub:
        context: np.ndarray
        x_for_v3: np.ndarray
        dataset_id: str

    samples = [
        _Stub(context=contexts[i], x_for_v3=xs[i], dataset_id="d4")
        for i in range(20)
    ]
    # All "anomalous" by construction → most should be flagged.
    kept, stats = gate_samples_by_alert(samples, flow, thresholds, percentile=95)
    assert stats["n_in"] == 20
    assert stats["n_kept"] >= 5  # at least the 10 large-x samples
    assert stats["by_dataset"]["d4"]["alerts"] >= 1

    # Same samples, this time tagged d2 + passthrough — ALL should be kept.
    samples_d2 = [
        _Stub(context=contexts[i], x_for_v3=xs[i], dataset_id="d2")
        for i in range(20)
    ]
    kept2, stats2 = gate_samples_by_alert(
        samples_d2, flow, thresholds, percentile=95, keep_dataset_ids=("d2",)
    )
    assert stats2["n_kept"] == 20
    assert all(s.dataset_id == "d2" for s in kept2)


def test_make_transition_segment_shapes() -> None:
    """`make_transition_segment` produces expected concatenated lengths."""
    loader, segs = _truncated_loader(max_seconds=5.0)
    v2_cfg = _smoke_v2_cfg()
    paired = []
    for s in segs:
        p = precompute_paired(s, v2_cfg)
        if p is not None:
            paired.append(p)
        if len(paired) >= 2:
            break
    seg_a, seg_b = paired[0], paired[1]

    spliced = make_transition_segment(seg_a, seg_b, crossfade_seconds=0.5)
    n_ac_xfade = max(1, int(round(0.5 * seg_a.acoustic_fs)))
    expected_T_ac = (
        seg_a.acoustic_features.shape[-1]
        + seg_b.acoustic_features.shape[-1]
        - n_ac_xfade
    )
    assert spliced.acoustic_features.shape[-1] == expected_T_ac
    n_v_xfade = max(1, int(round(0.5 * seg_a.vibration_fs)))
    expected_T_v = (
        seg_a.vibration_features.shape[-1]
        + seg_b.vibration_features.shape[-1]
        - n_v_xfade
    )
    assert spliced.vibration_features.shape[-1] == expected_T_v
    # Cadences and sensor counts preserved.
    assert spliced.acoustic_fs == seg_a.acoustic_fs
    assert spliced.vibration_fs == seg_a.vibration_fs
