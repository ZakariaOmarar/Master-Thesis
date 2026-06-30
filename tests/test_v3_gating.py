"""Unit tests for the V3 gate (deployment-faithful RQ3 metric).

Pure-synthetic: a hand-built :class:`PerClusterThresholds` plus fixed scores
exercises the strict per-cluster rule and the per-recording fallback, so the
gating logic is verified without a trained flow or any recordings.
"""

from __future__ import annotations

import numpy as np

from src.modeling.anomaly.threshold import PerClusterThresholds
from src.modeling.localization.v3_gating import gate_scores_to_keep


def _thresholds() -> PerClusterThresholds:
    # One cluster at the origin; p95 alert bar = 5.0, p99 = 9.0.
    return PerClusterThresholds(
        centroids=np.array([[0.0]]), p95=np.array([5.0]), p99=np.array([9.0]),
        n_per_cluster=np.array([100]),
    )


def test_strict_gate_keeps_scores_above_threshold() -> None:
    scores = np.array([1.0, 2.0, 3.0, 10.0, 11.0,  0.0, 1.0, 2.0, 3.0])
    contexts = np.zeros((9, 1))
    recs = ["recA"] * 5 + ["recB"] * 4
    res = gate_scores_to_keep(scores, contexts, _thresholds(), recs, percentile=95)
    # recA: 10 and 11 exceed 5.0 → 2 kept; recB: none exceed → 0.
    assert res.n_strict == 2
    assert res.n_final == 2
    assert res.keep_mask.tolist() == [False, False, False, True, True, False, False, False, False]
    assert res.per_recording["recB"]["strict_n_alerts"] == 0
    assert res.per_recording["recA"]["score_max"] == 11.0


def test_per_recording_fallback_rescues_quiet_recording() -> None:
    scores = np.array([1.0, 2.0, 3.0, 10.0, 11.0,  0.0, 1.0, 2.0, 3.0])
    contexts = np.zeros((9, 1))
    recs = ["recA"] * 5 + ["recB"] * 4
    res = gate_scores_to_keep(
        scores, contexts, _thresholds(), recs,
        percentile=95, min_events=1, fallback_quantile=0.5,
    )
    # recB fires 0 strict alerts < min_events → fallback keeps scores above its
    # median (1.5): the 2.0 and 3.0 windows.  recA already had 2 strict → untouched.
    assert res.n_strict == 2
    assert res.n_fallback_recordings == 1
    assert res.per_recording["recB"]["used_fallback"] is True
    assert res.keep_mask[5:].tolist() == [False, False, True, True]
    assert res.n_final == 4


def test_p99_is_stricter_than_p95() -> None:
    scores = np.array([6.0, 10.0])  # 6 clears p95 (5) but not p99 (9); 10 clears both
    contexts = np.zeros((2, 1))
    recs = ["r", "r"]
    keep95 = gate_scores_to_keep(scores, contexts, _thresholds(), recs, percentile=95).n_strict
    keep99 = gate_scores_to_keep(scores, contexts, _thresholds(), recs, percentile=99).n_strict
    assert keep95 == 2 and keep99 == 1


def test_empty_input() -> None:
    res = gate_scores_to_keep(
        np.zeros(0), np.zeros((0, 1)), _thresholds(), [], percentile=95
    )
    assert res.n_strict == 0 and res.n_final == 0 and res.keep_mask.size == 0
