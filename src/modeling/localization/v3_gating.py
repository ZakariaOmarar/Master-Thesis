"""V3 gating for the deployment-faithful RQ3 localization metric.

The localization head is meant to fire only on windows the anomaly detector (V3)
flags — "of the anomalies V3 detects, how accurately does V4 place them?".  The
historical campaigns computed this by re-running V3 over each recording with
sliding-window inference and matching V4 windows back by ``(recording_id, time
overlap)``.  That path had two silent failure modes — recording-id collisions
across the D4 ``speed{1,2,3}`` subfolders that share a position-folder name, and
time-coordinate drift between V3's 0.25 s stride and V4's impulse-burst window
placement — and it produced ``n_holdout_gated = 0`` on every cell of the
``deepc_20260526`` campaign even though V3's training-distribution detection F1
was ~0.94.

This module is the **direct path**: every V4 sample already caches its own
``x_for_v3`` and ``context`` (taken from the V2 encoder at exactly the V4
window), so V3 can score each sample in place and gate with the same
:class:`~src.modeling.anomaly.threshold.PerClusterThresholds` alert rule used
everywhere else — no re-inference, no time matching, no id collisions.  An
optional per-recording fallback guarantees the metric is *computable* (at least
``min_events`` gated windows per recording) so a single quiet recording cannot
leave the deployment metric undefined.

The pure-numpy :func:`gate_scores_to_keep` is split out from the model-scoring
:func:`gate_samples_by_v3` so the gating logic is unit-testable without a flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..anomaly.threshold import PerClusterThresholds


@dataclass
class GateResult:
    keep_mask: np.ndarray            # (N,) bool — windows V4 should localize
    n_strict: int                    # alerts under the unsupervised percentile rule
    n_final: int                     # alerts after the per-recording fallback
    n_fallback_recordings: int       # recordings rescued by the fallback
    scores: np.ndarray               # (N,) V3 anomaly scores
    per_recording: dict = field(default_factory=dict)


def gate_scores_to_keep(
    scores: np.ndarray,
    contexts: np.ndarray,
    thresholds: PerClusterThresholds,
    recording_ids: list[str],
    dataset_ids: list[str] | None = None,
    *,
    percentile: int = 95,
    min_events: int = 0,
    fallback_quantile: float = 0.90,
) -> GateResult:
    """Gate windows by V3 score, with an optional per-recording fallback.

    A window is kept (V4 will localize it) when its V3 score exceeds the
    per-cluster ``percentile`` threshold.  When ``min_events > 0`` and a
    recording fires fewer than that many strict alerts, the top
    ``(1 - fallback_quantile)`` of that recording's scores are kept instead, so
    every recording contributes at least ``min_events`` windows and the
    deployment metric is always computable.  The per-recording score
    distribution is returned for diagnostics (it shows *why* a recording did or
    did not fire).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    contexts = np.asarray(contexts, dtype=np.float64)
    n = scores.shape[0]
    if dataset_ids is None:
        dataset_ids = [""] * n
    alerts, _clusters = thresholds.alert(contexts, scores, percentile=percentile)
    keep = alerts.astype(bool)
    n_strict = int(keep.sum())

    rec_to_idx: dict[str, list[int]] = {}
    for i, rid in enumerate(recording_ids):
        rec_to_idx.setdefault(rid, []).append(i)

    diag: dict[str, dict] = {}
    n_fallback = 0
    for rid, idxs in rec_to_idx.items():
        rec_scores = scores[idxs]
        strict_n = int(keep[idxs].sum())
        used_fb = False
        fb_thr = None
        fb_n = None
        if min_events > 0 and strict_n < min_events and rec_scores.size > 0:
            q = float(np.quantile(rec_scores, fallback_quantile))
            fb_mask = rec_scores > q
            if int(fb_mask.sum()) >= min_events:
                for j, k in zip(idxs, fb_mask):
                    keep[j] = bool(k)
                used_fb = True
                fb_thr = q
                fb_n = int(fb_mask.sum())
                n_fallback += 1
        diag[rid] = {
            "dataset_id": dataset_ids[idxs[0]],
            "n_windows": len(idxs),
            "score_p50": float(np.quantile(rec_scores, 0.50)) if rec_scores.size else None,
            "score_p95": float(np.quantile(rec_scores, 0.95)) if rec_scores.size else None,
            "score_max": float(rec_scores.max()) if rec_scores.size else None,
            "strict_n_alerts": strict_n,
            "used_fallback": used_fb,
            "fallback_threshold": fb_thr,
            "fallback_n_alerts": fb_n,
        }
    return GateResult(
        keep_mask=keep,
        n_strict=n_strict,
        n_final=int(keep.sum()),
        n_fallback_recordings=n_fallback,
        scores=scores,
        per_recording=diag,
    )


def gate_samples_by_v3(
    flow,
    thresholds: PerClusterThresholds,
    samples: list,
    *,
    percentile: int = 95,
    min_events: int = 0,
    fallback_quantile: float = 0.90,
) -> GateResult:
    """Score cached V4 samples with V3 in place, then gate them.

    ``samples`` are ``V4Sample``-like objects exposing ``x_for_v3``, ``context``,
    ``recording_id`` and ``dataset_id``.  No sliding-window re-inference and no
    time matching: the cached ``x_for_v3`` / ``context`` are exactly what V3 saw
    at that window.
    """
    import torch

    if not samples:
        return GateResult(np.zeros(0, bool), 0, 0, 0, np.zeros(0), {})
    # Match the flow's device: on a GPU box the flow lives on cuda while these
    # tensors are built on cpu, which raises a device-mismatch in anomaly_score.
    try:
        dev = next(flow.parameters()).device
    except StopIteration:  # pragma: no cover - flow always has parameters
        dev = torch.device("cpu")
    xs = torch.from_numpy(np.stack([np.asarray(s.x_for_v3) for s in samples], axis=0)).float().to(dev)
    cs = torch.from_numpy(np.stack([np.asarray(s.context) for s in samples], axis=0)).float().to(dev)
    with torch.no_grad():
        flow.eval()
        scores = flow.anomaly_score(xs, cs).cpu().numpy()
    contexts = np.stack([np.asarray(s.context) for s in samples], axis=0)
    return gate_scores_to_keep(
        scores, contexts, thresholds,
        recording_ids=[s.recording_id for s in samples],
        dataset_ids=[getattr(s, "dataset_id", "") for s in samples],
        percentile=percentile, min_events=min_events, fallback_quantile=fallback_quantile,
    )


__all__ = ["GateResult", "gate_samples_by_v3", "gate_scores_to_keep"]
