"""Smoke tests for V1 per-modality SSL warmup.

Three independent checks:
  1. Encoder building blocks (MAB / PMA / ChannelTokenEnricher / PerModalityEncoder)
     — channel-agnostic forward pass on synthetic 4-channel and 5-channel inputs.
  2. Cluster metric — Hungarian-matched purity is sane on synthetic embeddings.
  3. End-to-end V1 training on a tiny truncated copy of D1 — finite loss,
     held-out recordings disjoint from training, sanity-gate metric returned.
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
from src.modeling.context.cluster_metric import (
    cluster_purity_and_nmi,
    hungarian_purity,
)
from src.modeling.context.v1_ssl import V1SSLConfig, train_v1_per_modality
from src.modeling.encoders.per_modality import PerModalityEncoder

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

    return _StubLoader(), truncated


# ---------------------------------------------------------------------------
# 1. Encoder forward pass — channel-agnostic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_mics", [4, 5, 9])
def test_acoustic_encoder_handles_arbitrary_n_mics(n_mics: int) -> None:
    enc = PerModalityEncoder(modality="acoustic", feature_dim=32, embed_dim=32, n_heads=2)
    enc.eval()
    B, C, F, T = 2, 2, 16, 8
    feat = torch.randn(B, n_mics, C, F, T)
    xyz = torch.randn(B, n_mics, 3)
    ds_idx = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        tokens, summary = enc(feat, xyz, ds_idx)
    assert tokens.shape == (B, n_mics, 32)
    assert summary.shape == (B, 32)
    assert torch.all(torch.isfinite(tokens))
    assert torch.all(torch.isfinite(summary))


@pytest.mark.parametrize("n_vib", [4, 5])
def test_vibration_encoder_handles_arbitrary_n_vib(n_vib: int) -> None:
    enc = PerModalityEncoder(modality="vibration", feature_dim=32, embed_dim=32, n_heads=2)
    enc.eval()
    B, C, T = 2, 3, 24
    feat = torch.randn(B, n_vib, C, T)
    xyz = torch.randn(B, n_vib, 3)
    ds_idx = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        tokens, summary = enc(feat, xyz, ds_idx)
    assert tokens.shape == (B, n_vib, 32)
    assert summary.shape == (B, 32)
    assert torch.all(torch.isfinite(tokens))


def test_encoder_is_permutation_invariant() -> None:
    """Permuting channels and their xyz positions in lockstep must produce the
    same summary (within float tolerance) — this is the constraint-#1 contract."""
    torch.manual_seed(0)
    enc = PerModalityEncoder(modality="acoustic", feature_dim=32, embed_dim=32, n_heads=2)
    enc.eval()
    B, N, C, F, T = 1, 4, 2, 16, 8
    feat = torch.randn(B, N, C, F, T)
    xyz = torch.randn(B, N, 3)
    ds_idx = torch.zeros(B, dtype=torch.long)

    perm = torch.tensor([2, 0, 3, 1])
    feat_p = feat[:, perm]
    xyz_p = xyz[:, perm]

    with torch.no_grad():
        _, s_orig = enc(feat, xyz, ds_idx)
        _, s_perm = enc(feat_p, xyz_p, ds_idx)

    assert torch.allclose(s_orig, s_perm, atol=1e-5), (
        "PerModalityEncoder summary changed under channel permutation; "
        "Set-Transformer pool is not permutation-invariant"
    )


# ---------------------------------------------------------------------------
# 2. Cluster metric
# ---------------------------------------------------------------------------


def test_hungarian_purity_recovers_perfect_clusters() -> None:
    cluster = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    labels = np.array([3, 3, 0, 0, 1, 1, 2, 2])  # cluster→label re-naming
    purity, mapping, _ = hungarian_purity(cluster, labels, n_clusters=4, n_labels=4)
    assert purity == 1.0
    # The mapping should pair every cluster with exactly the right label.
    assert {0: 3, 1: 0, 2: 1, 3: 2} == mapping


def test_cluster_purity_and_nmi_returns_finite_values() -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((40, 8))
    labels = ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10
    out = cluster_purity_and_nmi(embeddings, labels, n_clusters=4, seed=0)
    assert 0.0 <= out["purity"] <= 1.0
    assert 0.0 <= out["nmi"] <= 1.0
    assert out["n_clusters"] == 4
    assert out["confusion"].shape == (4, 4)


# ---------------------------------------------------------------------------
# 3. End-to-end V1 training
# ---------------------------------------------------------------------------


def test_train_v1_acoustic_end_to_end() -> None:
    loader, _ = _truncated_loader(max_seconds=5.0)
    cfg = V1SSLConfig(
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
        use_cwt=False,  # log-mel only — keeps the smoke test fast
        gain_jitter_db=3.0,
        channel_dropout_p=0.1,
        spec_augment_freq_mask=4,
        spec_augment_time_mask=4,
        seed=0,
    )

    result = train_v1_per_modality(loader, modality="acoustic", cfg=cfg)

    assert result.modality == "acoustic"
    assert len(result.train_loss_history) == cfg.epochs
    assert len(result.val_loss_history) == cfg.epochs
    assert all(np.isfinite(result.train_loss_history))
    assert all(np.isfinite(result.val_loss_history))

    # Held-out split is at the recording level — no leakage.
    train_ids = set(result.train_recording_ids)
    val_ids = set(result.val_recording_ids)
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids) >= 1
    assert len(val_ids) >= 1

    # Sanity gate runs and returns a finite purity in [0, 1].
    assert 0.0 <= result.sanity_gate["purity"] <= 1.0
    assert result.sanity_gate["n_windows"] >= 1


def test_train_v1_vibration_end_to_end() -> None:
    loader, _ = _truncated_loader(max_seconds=5.0)
    cfg = V1SSLConfig(
        window_seconds=1.0,
        window_stride_seconds=0.5,
        feature_dim=16,
        embed_dim=16,
        n_heads=2,
        proj_dim=8,
        epochs=2,
        batch_size=8,
        val_ratio=0.5,
        gain_jitter_db=3.0,
        channel_dropout_p=0.1,
        seed=0,
    )

    result = train_v1_per_modality(loader, modality="vibration", cfg=cfg)
    assert result.modality == "vibration"
    assert len(result.train_loss_history) == cfg.epochs
    assert all(np.isfinite(result.train_loss_history))
    assert 0.0 <= result.sanity_gate["purity"] <= 1.0
