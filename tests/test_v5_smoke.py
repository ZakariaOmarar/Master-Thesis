"""Smoke tests for V5 SCADA injection.

V5.1 — D3 speed one-hot helpers.
V5.2 — anomaly indicator construction + MI ranking on synthetic data + the
       real `results/illwerke/pipeline/anomaly_events.json` payload (for the
       indicator-builder integration test).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.modeling.scada import (
    D3_SCADA_DIM,
    D3_SPEED_BUCKETS,
    MIRanking,
    anomaly_indicator,
    d3_speed_one_hot,
    load_anomaly_events,
    physical_family,
    rank_channels_by_mi,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# V5.1 — D3 speed one-hot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bucket,idx", [("speed1", 0), ("speed2", 1), ("speed3", 2)])
def test_d3_speed_one_hot_returns_correct_index(bucket: str, idx: int) -> None:
    oh = d3_speed_one_hot(bucket)
    assert oh is not None
    assert oh.shape == (D3_SCADA_DIM,)
    assert oh.sum() == 1.0
    assert int(np.argmax(oh)) == idx


def test_d3_speed_one_hot_handles_unknown() -> None:
    assert d3_speed_one_hot(None) is None
    assert d3_speed_one_hot("speed4") is None
    assert d3_speed_one_hot("foo") is None


def test_d3_speed_buckets_match_dim() -> None:
    assert len(D3_SPEED_BUCKETS) == D3_SCADA_DIM


# ---------------------------------------------------------------------------
# V5.2 — anomaly indicator
# ---------------------------------------------------------------------------


def test_anomaly_indicator_marks_event_windows() -> None:
    fs_ns = 1_000_000_000  # 1 Hz grid
    timestamps = np.arange(0, 100 * fs_ns, fs_ns, dtype=np.int64)
    events = [
        {"t_start_s": 10.0, "t_end_s": 20.0, "severity": "alert"},
        {"t_start_s": 50.0, "t_end_s": 60.0, "severity": "warn"},  # excluded by default
        {"t_start_s": 80.0, "t_end_s": 89.0, "severity": "alert"},
    ]
    ind = anomaly_indicator(timestamps, events)
    # Default severity_set=("alert",) — only events 0 and 2 count.
    assert ind.shape == (100,)
    assert ind[10:21].sum() == 11  # inclusive endpoints
    assert ind[50:61].sum() == 0  # warn-only event excluded
    assert ind[80:90].sum() == 10
    assert ind.sum() == 11 + 10


def test_anomaly_indicator_with_real_events_json() -> None:
    """Sanity check: the real events file parses and yields a non-empty
    indicator on a synthetic 1-Hz timestamp grid covering the campaign span."""
    events_path = REPO_ROOT / "results" / "illwerke" / "pipeline" / "anomaly_events.json"
    if not events_path.exists():
        pytest.skip("legacy anomaly_events.json not present in this checkout")
    events = load_anomaly_events(events_path)
    assert len(events) > 0

    # Build a synthetic 1Hz grid spanning max event end + 100s.
    t_max = max(int(ev["t_end_s"]) for ev in events) + 100
    fs_ns = 1_000_000_000
    timestamps = np.arange(0, (t_max + 1) * fs_ns, fs_ns, dtype=np.int64)
    ind = anomaly_indicator(timestamps, events)
    assert ind.shape == timestamps.shape
    assert ind.sum() > 0


# ---------------------------------------------------------------------------
# V5.2 — physical family classifier
# ---------------------------------------------------------------------------


def test_physical_family_keyword_matching() -> None:
    assert physical_family("Druckwasser_p1") == "pressure"
    assert physical_family("Lager_Temperatur") == "thermal"
    assert physical_family("Durchfluss_Q1") == "hydraulic"
    assert physical_family("Drehzahl_n") == "rotational"
    assert physical_family("Generator_Leistung") == "electrical"
    assert physical_family("Schwingweg") == "vibration"
    assert physical_family("Random_Channel_42") == "other"


# ---------------------------------------------------------------------------
# V5.2 — MI ranking
# ---------------------------------------------------------------------------


def test_rank_channels_by_mi_recovers_informative_channel() -> None:
    rng = np.random.default_rng(0)
    T = 600
    indicator = (rng.random(T) < 0.3).astype(np.int64)

    # Ch0: copy of indicator + small noise → MI should dominate.
    ch0 = indicator + rng.normal(scale=0.05, size=T)
    # Ch1: random noise → MI low.
    ch1 = rng.standard_normal(T)
    # Ch2: zero variance → dropped before ranking.
    ch2 = np.full(T, 1.234)
    # Ch3: random noise → MI low.
    ch3 = rng.standard_normal(T)
    allg = np.stack([ch0, ch1, ch2, ch3], axis=1)
    names = ["pressure_p1", "noise_a", "constant_unused", "Schwing_b"]

    ranking = rank_channels_by_mi(allg, names, indicator, seed=0)
    # Zero-variance channel dropped.
    assert "constant_unused" not in ranking.channel_names
    # Copy-of-indicator must rank #1.
    top = ranking.top_k(1)
    assert top[0][0] == "pressure_p1"
    assert top[0][2] == "pressure"
    # Sanity on full top-K.
    out = ranking.to_dict(k=3)
    assert len(out["ranked"]) == 3
    assert out["n_anomaly_samples"] == int(indicator.sum())
    assert out["n_total_samples"] == T


def test_rank_channels_rejects_zero_indicator() -> None:
    allg = np.random.default_rng(0).standard_normal((100, 3))
    names = ["a", "b", "c"]
    indicator = np.zeros(100, dtype=np.int64)
    with pytest.raises(ValueError):
        rank_channels_by_mi(allg, names, indicator)


def test_mi_ranking_top_k_ordering() -> None:
    """top_k returns names sorted by descending MI."""
    ranking = MIRanking(
        channel_names=["a", "b", "c"],
        mi=np.array([0.1, 0.5, 0.3]),
        families=["other", "other", "other"],
        n_anomaly_samples=10,
        n_total_samples=100,
        seed=0,
    )
    top = ranking.top_k(2)
    assert [name for name, _, _ in top] == ["b", "c"]
