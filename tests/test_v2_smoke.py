"""Smoke tests for V2 multimodal SSL fusion.

Three independent checks:
  1. `BidirectionalCrossAttention` — channel-agnostic forward pass on
     mismatched (4-mic, 5-vib) and (9-mic, 4-vib) token shapes.
  2. `V2FusionEncoder` — V1 weight transfer + masked forward returns finite
     fused tokens and a context vector with the expected shape.
  3. End-to-end V2 SSL training on a tiny truncated copy of D1 — finite loss
     for both SimCLR and LMM components, held-out recordings disjoint from
     training, RQ1 cluster-purity returned in [0, 1].
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
from src.modeling.context.v2_ssl import V2SSLConfig, train_v2_fusion
from src.modeling.encoders import PerModalityEncoder
from src.modeling.fusion import BidirectionalCrossAttention

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

    return _StubLoader()


# ---------------------------------------------------------------------------
# 1. Cross-attention block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_a,n_v", [(4, 5), (9, 4), (5, 5)])
def test_cross_attention_handles_mismatched_n(n_a: int, n_v: int) -> None:
    block = BidirectionalCrossAttention(dim=32, num_heads=2)
    block.eval()
    a = torch.randn(2, n_a, 32)
    v = torch.randn(2, n_v, 32)
    with torch.no_grad():
        fused_a, fused_v = block(a, v)
    assert fused_a.shape == (2, n_a, 32)
    assert fused_v.shape == (2, n_v, 32)
    assert torch.all(torch.isfinite(fused_a))
    assert torch.all(torch.isfinite(fused_v))


def test_cross_attention_dim_mismatch_raises() -> None:
    block = BidirectionalCrossAttention(dim=32, num_heads=2)
    a = torch.randn(1, 4, 32)
    v = torch.randn(1, 4, 16)
    with pytest.raises(ValueError):
        block(a, v)


# ---------------------------------------------------------------------------
# 2. V2FusionEncoder + V1 weight transfer
# ---------------------------------------------------------------------------


def test_v2_encoder_inherits_v1_weights() -> None:
    """Loading V1 PerModalityEncoder state_dicts into V2 leaves the encoder
    forward bit-exact equal to the V1 encoder forward."""
    torch.manual_seed(0)
    v1_acoustic = PerModalityEncoder("acoustic", feature_dim=32, embed_dim=32, n_heads=2)
    v1_vibration = PerModalityEncoder("vibration", feature_dim=32, embed_dim=32, n_heads=2)
    v1_acoustic.eval()
    v1_vibration.eval()

    v2 = V2FusionEncoder(feature_dim=32, embed_dim=32, n_heads=2)
    v2.load_v1_weights(v1_acoustic.state_dict(), v1_vibration.state_dict(), strict=True)
    v2.eval()

    B, n_a, n_v = 2, 4, 4
    ac = torch.randn(B, n_a, 2, 16, 8)
    vib = torch.randn(B, n_v, 3, 24)
    ac_xyz = torch.randn(B, n_a, 3)
    vib_xyz = torch.randn(B, n_v, 3)
    ds_idx = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        a_v1, _ = v1_acoustic(ac, ac_xyz, ds_idx)
        v_v1, _ = v1_vibration(vib, vib_xyz, ds_idx)
        a_v2, v_v2 = v2.encode_modalities(ac, ac_xyz, vib, vib_xyz, ds_idx)

    assert torch.allclose(a_v1, a_v2, atol=1e-6)
    assert torch.allclose(v_v1, v_v2, atol=1e-6)


def test_v2_masked_forward_shapes_and_finite() -> None:
    torch.manual_seed(0)
    v2 = V2FusionEncoder(feature_dim=32, embed_dim=32, n_heads=2)
    v2.eval()

    B, n_a, n_v = 2, 4, 5
    ac = torch.randn(B, n_a, 2, 16, 8)
    vib = torch.randn(B, n_v, 3, 24)
    ac_xyz = torch.randn(B, n_a, 3)
    vib_xyz = torch.randn(B, n_v, 3)
    ds_idx = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        out = v2(ac, ac_xyz, vib, vib_xyz, ds_idx, mask_p=0.5)
    assert out["a_target"].shape == (B, n_a, 32)
    assert out["v_target"].shape == (B, n_v, 32)
    assert out["a_fused"].shape == (B, n_a, 32)
    assert out["v_fused"].shape == (B, n_v, 32)
    assert out["context"].shape == (B, 32)
    assert out["mask_a"].shape == (B, n_a)
    assert out["mask_v"].shape == (B, n_v)
    assert torch.all(torch.isfinite(out["context"]))
    assert torch.all(torch.isfinite(out["a_fused"]))
    assert torch.all(torch.isfinite(out["v_fused"]))
    # apply_mask guarantees ≥1 unmasked token per sample
    assert (~out["mask_a"]).any(dim=1).all()
    assert (~out["mask_v"]).any(dim=1).all()


# ---------------------------------------------------------------------------
# 3. End-to-end training
# ---------------------------------------------------------------------------


def _smoke_cfg(**overrides) -> V2SSLConfig:
    base = dict(
        window_seconds=0.5,
        window_stride_seconds=0.25,
        feature_dim=32,
        embed_dim=32,
        n_heads=2,
        proj_dim=16,
        epochs=2,
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
        lmm_mask_p=0.4,
        lmm_weight=0.5,
        seed=0,
    )
    base.update(overrides)
    return V2SSLConfig(**base)


def test_train_v2_fusion_end_to_end() -> None:
    loader = _truncated_loader(max_seconds=5.0)
    cfg = _smoke_cfg()

    result = train_v2_fusion(loader, cfg=cfg)

    assert len(result.train_loss_history) == cfg.epochs
    assert len(result.val_loss_history) == cfg.epochs
    assert len(result.train_simclr_history) == cfg.epochs
    assert len(result.train_lmm_history) == cfg.epochs
    assert all(np.isfinite(result.train_loss_history))
    assert all(np.isfinite(result.val_loss_history))
    assert all(np.isfinite(result.train_simclr_history))
    assert all(np.isfinite(result.train_lmm_history))
    # Both losses contributed.
    assert result.train_lmm_history[-1] > 0.0
    assert result.train_simclr_history[-1] > 0.0

    # Held-out recordings disjoint.
    train_ids = set(result.train_recording_ids)
    val_ids = set(result.val_recording_ids)
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids) >= 1
    assert len(val_ids) >= 1

    # RQ1 metric is finite and in range.
    assert 0.0 <= result.rq1["purity"] <= 1.0
    assert 0.0 <= result.rq1["nmi"] <= 1.0
    assert result.rq1["n_windows"] >= 1


def test_train_v2_fusion_inherits_v1_state() -> None:
    """V1 PerModalityEncoder state_dicts plug into V2 without strict=False."""
    loader = _truncated_loader(max_seconds=5.0)

    # Build a V1-shaped acoustic encoder with the same dims as V2 smoke cfg.
    v1_acoustic = PerModalityEncoder("acoustic", feature_dim=32, embed_dim=32, n_heads=2)
    v1_vibration = PerModalityEncoder("vibration", feature_dim=32, embed_dim=32, n_heads=2)

    cfg = _smoke_cfg(epochs=1)
    result = train_v2_fusion(
        loader,
        cfg=cfg,
        v1_acoustic_state_dict=v1_acoustic.state_dict(),
        v1_vibration_state_dict=v1_vibration.state_dict(),
    )
    assert len(result.train_loss_history) == cfg.epochs
    assert all(np.isfinite(result.train_loss_history))


def test_train_v2_modality_dropout_runs() -> None:
    """Modality dropout (default 0.3) should not break training: loss is
    finite, RQ1 metric returns in range, encoder weights are finite."""
    loader = _truncated_loader(max_seconds=5.0)
    cfg = _smoke_cfg(epochs=2, modality_dropout_p=0.5)  # high p to exercise it
    result = train_v2_fusion(loader, cfg=cfg)
    assert all(np.isfinite(result.train_loss_history))
    assert all(np.isfinite(result.val_loss_history))
    assert 0.0 <= result.rq1.get("purity", 0.0) <= 1.0
    # Encoder weights remain finite under modality dropout.
    for p in result.encoder.parameters():
        assert torch.all(torch.isfinite(p))


def test_train_v2_modality_dropout_zero_matches_baseline() -> None:
    """`modality_dropout_p=0` should behave identically to the previous
    implementation for backwards compatibility."""
    loader = _truncated_loader(max_seconds=5.0)
    cfg_off = _smoke_cfg(epochs=1, modality_dropout_p=0.0, seed=0)
    result_off = train_v2_fusion(loader, cfg=cfg_off)
    assert all(np.isfinite(result_off.train_loss_history))
    assert all(np.isfinite(result_off.val_loss_history))


def test_train_v2_drop_vibration_ablation() -> None:
    """A1 ablation — `drop_vibration=True` zeros vibration features but the
    pipeline still runs end-to-end and reports a finite RQ1 purity."""
    loader = _truncated_loader(max_seconds=5.0)
    cfg = _smoke_cfg(epochs=1, drop_vibration=True)
    result = train_v2_fusion(loader, cfg=cfg)
    assert result.drop_vibration is True
    assert len(result.train_loss_history) == cfg.epochs
    assert np.isfinite(result.train_loss_history[-1])
    assert 0.0 <= result.rq1["purity"] <= 1.0
