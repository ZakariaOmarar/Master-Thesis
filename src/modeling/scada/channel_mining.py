"""V5.2 — Illwerke SCADA mutual-information ranking.

Take the existing 5-layer pipeline anomaly events
(`results/illwerke/pipeline/anomaly_events.json`) as the binary target;
compute mutual information between each Allg_M1 channel time series and the
target on a shared 1 Hz grid; rank top-K channels and tag each by physical
family.  No injection on D2/D3 — fabricated SCADA stand-ins are deliberately
avoided.

The thesis narrative for RQ4 is then:

  V5.1 proves on D3 that injecting an operational-state variable (speed)
  reduces localization error by X %.
  V5.2 analyzes the real Illwerke SCADA and shows that channels A, B, C, D
  carry the highest MI with turbine anomalies.  The deployment recommendation
  for ROW II is therefore to wire A, B, C, D into s_t whose conditioning
  mechanism V5.1 already validated.

Pure-evaluation utility: V5.2 does not train anything.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif

# Heuristic keyword → physical-family map for Allg_M1 channels.
# The Allg_M1 catalog uses German labels — ``Druck`` (pressure), ``Temp``
# (temperature), ``Durchfluss``/``Q`` (flow), etc.  When a channel doesn't
# match any keyword we tag it as ``other`` so the report stays honest.
# Order matters: the first matching family wins, so the more specific keys
# (e.g. the ``druck`` pressure channels, the ``p_ist``/``lfr`` power-control
# channels) are listed before broad fallbacks.  The Allg_M1 ``1_P_Ist`` /
# ``1_P_Soll`` / ``1_P-Regler`` channels are *active power* (Leistung) in the
# load-frequency-control loop (LFR = Leistungs-Frequenz-Regelung), not pressure;
# the earlier ``_p_`` pressure key mis-tagged them, so it was removed.
_PHYSICAL_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "electrical": (
        "leistung", "power", "spannung", "voltage", "strom", "current",
        "p_ist", "p_soll", "p-regler", "p_regler", "lfr", "erregerstrom", "kv",
    ),
    "pressure": ("druck", "pressure"),
    "thermal": ("temp", "temperatur", "_t1", "_t2"),
    "hydraulic": (
        "durchfluss", "flow", "pegel", "level", "fuell", "leitapparat",
        "q_", "_q1", "_q2",
    ),
    "rotational": ("drehzahl", "rpm", "upm", "speed", "_n_", "frequenz"),
    "vibration": ("vib", "schwing"),
}


def physical_family(channel_name: str) -> str:
    name = channel_name.lower()
    for family, keys in _PHYSICAL_FAMILY_KEYWORDS.items():
        if any(k in name for k in keys):
            return family
    return "other"


# ---------------------------------------------------------------------------
# Anomaly indicator construction
# ---------------------------------------------------------------------------


def load_anomaly_events(events_path: str | Path) -> list[dict]:
    """Load the legacy 5-layer pipeline anomaly events.

    Each event is a dict with at least `t_start_s`, `t_end_s`, `severity`.
    """
    p = Path(events_path)
    if not p.exists():
        raise FileNotFoundError(f"anomaly events file not found: {p}")
    return json.loads(p.read_text())


def anomaly_indicator(
    timestamps_ns: np.ndarray,
    events: Iterable[dict],
    *,
    severity_set: tuple[str, ...] = ("alert",),
    campaign_t0_ns: int | None = None,
) -> np.ndarray:
    """Build a binary anomaly indicator on the channel time grid.

    Args:
      timestamps_ns: `(T,)` int64 absolute nanosecond timestamps from the
        Illwerke campaign (1 Hz grid).
      events: anomaly events from `load_anomaly_events`.  Each event's
        ``t_start_s`` / ``t_end_s`` are seconds *relative to the campaign
        start* (the legacy pipeline's convention).
      severity_set: which severities to count as anomalous.
      campaign_t0_ns: campaign start timestamp.  Defaults to
        ``timestamps_ns[0]``.

    Returns:
      `(T,)` uint8 array — 1 where the timestamp falls inside any selected
      anomaly event, 0 otherwise.
    """
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    if campaign_t0_ns is None:
        if timestamps_ns.size == 0:
            return np.zeros(0, dtype=np.uint8)
        campaign_t0_ns = int(timestamps_ns[0])

    rel_s = (timestamps_ns - campaign_t0_ns) / 1_000_000_000.0
    out = np.zeros(timestamps_ns.shape, dtype=np.uint8)
    for ev in events:
        if ev.get("severity") not in severity_set:
            continue
        t0 = float(ev["t_start_s"])
        t1 = float(ev["t_end_s"])
        mask = (rel_s >= t0) & (rel_s <= t1)
        out[mask] = 1
    return out


# ---------------------------------------------------------------------------
# MI ranking
# ---------------------------------------------------------------------------


@dataclass
class MIRanking:
    channel_names: list[str]  # length N_kept (after dropping zero-variance)
    mi: np.ndarray  # (N_kept,)
    families: list[str]  # length N_kept
    n_anomaly_samples: int
    n_total_samples: int
    seed: int

    def top_k(self, k: int) -> list[tuple[str, float, str]]:
        """Return the top-K channels as `(name, mi, family)` tuples."""
        order = np.argsort(self.mi)[::-1]
        out: list[tuple[str, float, str]] = []
        for i in order[:k]:
            out.append((self.channel_names[i], float(self.mi[i]), self.families[i]))
        return out

    def to_dict(self, k: int | None = None) -> dict:
        order = np.argsort(self.mi)[::-1]
        if k is None:
            k = len(order)
        ranked = [
            {
                "name": self.channel_names[i],
                "mi": float(self.mi[i]),
                "family": self.families[i],
            }
            for i in order[:k]
        ]
        return {
            "ranked": ranked,
            "n_anomaly_samples": int(self.n_anomaly_samples),
            "n_total_samples": int(self.n_total_samples),
            "seed": self.seed,
        }


def rank_channels_by_mi(
    allg_data: np.ndarray,  # (T, N_ch) float
    channel_names: list[str],
    indicator: np.ndarray,  # (T,) int/uint
    *,
    seed: int = 0,
    drop_zero_variance: bool = True,
) -> MIRanking:
    """Compute MI between each channel and the binary anomaly indicator.

    Uses sklearn's `mutual_info_classif` (k-nearest-neighbor estimator).
    Channels with zero variance are dropped (their MI is undefined and
    sklearn would warn).

    Returns a `MIRanking` covering only the kept channels.
    """
    allg_data = np.asarray(allg_data, dtype=np.float64)
    indicator = np.asarray(indicator, dtype=np.int64).ravel()
    if allg_data.ndim != 2:
        raise ValueError(f"allg_data must be 2-D; got {allg_data.shape}")
    if allg_data.shape[0] != indicator.shape[0]:
        raise ValueError("allg_data and indicator must agree on the time axis")
    if len(channel_names) != allg_data.shape[1]:
        raise ValueError(
            f"channel_names ({len(channel_names)}) must match #cols ({allg_data.shape[1]})"
        )
    if indicator.sum() == 0:
        raise ValueError("anomaly indicator is all-zero; MI is degenerate")

    keep_idx = np.arange(allg_data.shape[1])
    if drop_zero_variance:
        # Use a small tolerance: floating-point variance of a numerically-constant
        # column may not be exactly 0 even when the values are identical.
        var = allg_data.var(axis=0)
        keep_idx = np.where(var > 1e-12)[0]
        if keep_idx.size == 0:
            raise ValueError("all channels have zero variance; nothing to rank")

    X = allg_data[:, keep_idx]
    mi = mutual_info_classif(X, indicator, random_state=seed)
    kept_names = [channel_names[i] for i in keep_idx]
    families = [physical_family(n) for n in kept_names]
    return MIRanking(
        channel_names=kept_names,
        mi=np.asarray(mi, dtype=np.float64),
        families=families,
        n_anomaly_samples=int(indicator.sum()),
        n_total_samples=int(indicator.shape[0]),
        seed=seed,
    )


__all__ = [
    "MIRanking",
    "anomaly_indicator",
    "load_anomaly_events",
    "physical_family",
    "rank_channels_by_mi",
]
