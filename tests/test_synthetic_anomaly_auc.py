"""Smoke tests for V3 synthetic-anomaly ROC-AUC.

The test verifies two invariants of the synthetic-AUC evaluation:

  * **AUC is monotone-decreasing in SNR**: at very low SNR (loud noise)
    the corrupted windows are easy to separate from clean → AUC ≈ 1;
    at very high SNR (faint noise) the corruption is almost
    imperceptible → AUC ≈ 0.5.
  * **AUC is bounded in [0, 1]** and matches the Mann-Whitney U
    interpretation on hand-constructed score arrays.

These invariants are sufficient to defend the AUC machinery; the
absolute AUC values on real data are reported in Chapter 6.
"""

from __future__ import annotations

import numpy as np
import torch

from src.modeling.anomaly.cnf_head import ConditionalRealNVP
from src.modeling.anomaly.synthetic_eval import (
    _roc_auc,
    evaluate_synthetic_anomaly_auc,
)


def test_roc_auc_perfect_separation() -> None:
    pos = np.array([2.0, 3.0, 4.0])
    neg = np.array([0.0, 1.0])
    assert _roc_auc(pos, neg) == 1.0


def test_roc_auc_zero_separation() -> None:
    pos = np.array([0.0, 1.0])
    neg = np.array([2.0, 3.0, 4.0])
    assert _roc_auc(pos, neg) == 0.0


def test_roc_auc_chance_with_ties() -> None:
    # Identical distributions → AUC = 0.5 exactly with tie handling.
    pos = np.array([1.0, 2.0, 3.0])
    neg = np.array([1.0, 2.0, 3.0])
    assert _roc_auc(pos, neg) == 0.5


def test_synthetic_auc_monotone_in_snr() -> None:
    """AUC must be monotone-decreasing in SNR_dB (loud noise → easy
    discrimination → high AUC; faint noise → hard → AUC → 0.5)."""
    torch.manual_seed(0)
    flow = ConditionalRealNVP(dim=8, c_dim=8, n_layers=3, hidden_dim=16)
    flow.eval()
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(60, 8))
    c = rng.normal(0.0, 1.0, size=(60, 8))
    result = evaluate_synthetic_anomaly_auc(
        flow, x, c,
        snr_db_list=(-10.0, 0.0, 10.0),
        n_boot=100, seed=0,
    )
    auc_low = result.snr_db_to_auc[-10.0]
    auc_mid = result.snr_db_to_auc[0.0]
    auc_high = result.snr_db_to_auc[10.0]
    assert auc_low >= auc_mid >= auc_high  # monotone-decreasing in SNR
    assert auc_low > 0.9  # loud noise is essentially trivial
    # All AUCs in [0, 1] and CIs are bounded.
    for s in (-10.0, 0.0, 10.0):
        assert 0.0 <= result.snr_db_to_auc[s] <= 1.0
        assert result.snr_db_to_auc_ci_low[s] <= result.snr_db_to_auc[s]
        assert result.snr_db_to_auc[s] <= result.snr_db_to_auc_ci_high[s]


def test_synthetic_auc_degenerate_small_n() -> None:
    """Fewer than 4 healthy windows → return NaN structure rather than
    raise."""
    torch.manual_seed(0)
    flow = ConditionalRealNVP(dim=4, c_dim=4, n_layers=2, hidden_dim=8)
    flow.eval()
    x = np.zeros((2, 4))
    c = np.zeros((2, 4))
    result = evaluate_synthetic_anomaly_auc(flow, x, c, snr_db_list=(0.0,), n_boot=10, seed=0)
    assert np.isnan(result.snr_db_to_auc[0.0])
