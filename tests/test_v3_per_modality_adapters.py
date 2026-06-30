"""Tests for the R2.1 V3 per-modality encoder adapters."""

from __future__ import annotations

import pytest
import torch

from src.modeling.anomaly.v3_per_modality import (
    V3AcousticOnlyAdapter,
    V3VibrationOnlyAdapter,
)
from src.modeling.encoders.per_modality import PerModalityEncoder


@pytest.fixture
def encoders():
    torch.manual_seed(0)
    ac = PerModalityEncoder(
        modality="acoustic", feature_dim=32, embed_dim=32, n_heads=2,
        acoustic_cnn_width_mult=1,
    )
    vib = PerModalityEncoder(
        modality="vibration", feature_dim=32, embed_dim=32, n_heads=2,
    )
    return ac, vib


@pytest.fixture
def batch():
    return dict(
        ac=torch.randn(3, 4, 2, 24, 12),
        vib=torch.randn(3, 4, 3, 8),
        ac_xyz=torch.randn(3, 4, 3),
        vib_xyz=torch.randn(3, 4, 3),
        ds=torch.zeros(3, dtype=torch.long),
    )


def test_acoustic_adapter_returns_v2_api_dict(encoders, batch):
    ac, _ = encoders
    adp = V3AcousticOnlyAdapter(ac)
    out = adp(batch["ac"], batch["ac_xyz"], batch["vib"], batch["vib_xyz"], batch["ds"])
    # V3 reads exactly these three keys.
    for k in ("a_fused", "v_fused", "context"):
        assert k in out, f"adapter must expose {k!r} for V3 trainer"
    assert out["a_fused"].shape == (3, 4, 32)
    assert out["v_fused"].shape == (3, 1, 32)  # length-1 zero token
    assert out["context"].shape == (3, 32)
    # Vibration slot must be exactly zero — that is the whole point of this adapter.
    assert torch.all(out["v_fused"] == 0)


def test_vibration_adapter_returns_v2_api_dict(encoders, batch):
    _, vib = encoders
    adp = V3VibrationOnlyAdapter(vib)
    out = adp(batch["ac"], batch["ac_xyz"], batch["vib"], batch["vib_xyz"], batch["ds"])
    assert out["a_fused"].shape == (3, 1, 32)
    assert out["v_fused"].shape == (3, 4, 32)
    assert out["context"].shape == (3, 32)
    assert torch.all(out["a_fused"] == 0)


def test_modality_mismatch_rejected(encoders):
    ac, vib = encoders
    with pytest.raises(ValueError, match="modality='acoustic'"):
        V3AcousticOnlyAdapter(vib)
    with pytest.raises(ValueError, match="modality='vibration'"):
        V3VibrationOnlyAdapter(ac)


def test_v3_xc_extraction_path(encoders, batch):
    """Replicate `v3_trainer._extract_xc` ‘s x/c extraction and assert shapes."""
    ac, vib = encoders
    adp_a = V3AcousticOnlyAdapter(ac)
    out = adp_a(batch["ac"], batch["ac_xyz"], batch["vib"], batch["vib_xyz"], batch["ds"])
    fused = torch.cat([out["a_fused"], out["v_fused"]], dim=1)
    x = fused.mean(dim=1)
    c = out["context"]
    assert x.shape == (3, 32)
    assert c.shape == (3, 32)
    # x is the mean over the joined sequence (5 tokens: 4 acoustic + 1 zero).
    # So x must equal acoustic_mean * 4/5 (since the zero token contributes 0).
    expected_x = out["a_fused"].mean(dim=1) * (4 / 5)
    assert torch.allclose(x, expected_x, atol=1e-6)


def test_adapter_ignores_unused_modality_inputs(encoders, batch):
    """Changing vibration inputs must not change the acoustic adapter's output."""
    ac, _ = encoders
    adp = V3AcousticOnlyAdapter(ac)
    out1 = adp(batch["ac"], batch["ac_xyz"], batch["vib"], batch["vib_xyz"], batch["ds"])
    out2 = adp(
        batch["ac"], batch["ac_xyz"],
        batch["vib"] * 1000.0, batch["vib_xyz"] + 5.0,
        batch["ds"],
    )
    for k in ("a_fused", "context"):
        assert torch.allclose(out1[k], out2[k], atol=1e-6), (
            f"{k!r} must not depend on vibration inputs"
        )


def test_train_v3_cnf_accepts_per_modality_adapter():
    """Smoke: feed a per-modality adapter into the real V3 trainer on a
    micro batch and confirm the (x, c) extraction + flow init succeed."""
    from src.modeling.anomaly.cnf_head import ConditionalRealNVP
    from src.modeling.anomaly.v3_trainer import _extract_xc

    torch.manual_seed(0)
    ac_enc = PerModalityEncoder(
        modality="acoustic", feature_dim=32, embed_dim=32, n_heads=2,
        acoustic_cnn_width_mult=1,
    )
    adp = V3AcousticOnlyAdapter(ac_enc)
    adp.eval()

    # Mimic one batch from a DataLoader.
    class _SingleBatchLoader:
        def __iter__(self):
            yield {
                "ac_feat": torch.randn(2, 4, 2, 24, 12),
                "ac_xyz": torch.randn(2, 4, 3),
                "vib_feat": torch.randn(2, 4, 3, 8),
                "vib_xyz": torch.randn(2, 4, 3),
                "dataset_idx": torch.zeros(2, dtype=torch.long),
                "mode_label": ["Pump", "Turbine"],
            }

    x, c, labels = _extract_xc(adp, _SingleBatchLoader(), torch.device("cpu"))
    assert x.shape == (2, 32)
    assert c.shape == (2, 32)
    assert labels == ["Pump", "Turbine"]
    flow = ConditionalRealNVP(dim=32, c_dim=32, n_layers=2, hidden_dim=32)
    log_p = flow.log_prob(x, c)
    assert log_p.shape == (2,)
    assert torch.all(torch.isfinite(log_p))
