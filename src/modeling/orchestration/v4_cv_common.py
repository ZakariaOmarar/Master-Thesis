"""Shared scaffolding for the post-hoc V4 cross-validation drivers.

`v4_lopo_cv` (leave-one-position-out) and `v4_cross_dataset` (dataset transfer)
both reuse a V2 encoder trained by `full_run` and run on the *same* labelled
cohort — D2/D3/D4/D5, knock-interval-restricted. The pieces that are identical
between them live here so they cannot drift. `v4_loocv` deliberately uses a
narrower cohort (no D5, no knock-interval restriction) and keeps its own gather,
but shares the V3-gating helpers below so its gated metric matches.

V3-gated MAE (deployment-faithful RQ3): alongside the ungated per-fold MAE, the
drivers also report the MAE on only the holdout knocks V3 flags as anomalous —
the exact filtering `full_run` Stage 5b applies (`gate_samples_by_v3`).  The
three helpers here (`load_v3_for_gating`, the `v3=` arg on
`load_or_precompute_cv_samples`, and `gated_fold_mae`) are the single source of
truth for that path so the CV drivers cannot drift from Stage 5b.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..localization import (
    V4_CANDIDATE_GRID,
    event_aggregated_mae,
    precompute_v4_knock_event_samples,
)
from .full_run import _d3_spatial_overrides, resolved_loader

if TYPE_CHECKING:
    from ..context.v2_fusion import V2FusionEncoder
    from ..context.v2_ssl import V2SSLConfig
    from ..localization import V4Sample

ChannelMode = Literal["both", "srp_only", "tdoa_only", "vibration_only_learned"]

# The four V4 channel-ablation paradigms (acoustic SRP, accel TDOA, learned
# vibration-only, and the full fusion) compared in the RQ3 localization tables.
CHANNEL_MODES: tuple[ChannelMode, ...] = (
    "both", "srp_only", "tdoa_only", "vibration_only_learned",
)


# ---------------------------------------------------------------------------
# V3 gating (deployment-faithful RQ3 metric) — mirrors full_run Stage 5b.
# ---------------------------------------------------------------------------


@dataclass
class V3ForGating:
    """The trained fusion V3 needed to gate V4 holdout knocks.

    Bundles exactly the pieces `full_run` Stage 5b reads off ``V3Result`` when it
    computes ``holdout_mae_v3gated_m``: the flow + per-cluster thresholds (the
    alert rule), the learned channel-token pool and impulse+spectral anchor
    standardization (so ``x_for_v3`` is built the way the flow was trained), and
    the deployment percentile.
    """

    flow: object
    thresholds: object
    xt_pool: object | None
    anchor_norm: tuple | None
    percentile: int


def load_v3_for_gating(
    v3_run: Path,
    *,
    embed_dim: int,
    threshold_percentile: int = 95,
    log_prefix: str = "V4 CV",
) -> V3ForGating | None:
    """Load `full_run`'s saved fusion V3 (flow + thresholds + xt_pool + anchor).

    Looks under ``<v3_run>/v3`` then ``<v3_run>/v3_fusion`` (full_run writes both)
    and reuses the canonical `rq2_three_paradigm_eval._load_v3` reconstruction so
    the flow dim / anchor / pool match the saved artefacts exactly.

    Returns ``None`` — and prints why — when the artefacts are absent or not
    reproducible, so callers degrade to ungated-only instead of crashing.
    """
    pipe = None
    for cand in (Path(v3_run) / "v3", Path(v3_run) / "v3_fusion"):
        if (cand / "flow.pt").exists() and (cand / "thresholds.npz").exists():
            pipe = cand
            break
    if pipe is None:
        print(f"{log_prefix}: no V3 artefacts under {v3_run} (looked in v3/ and "
              f"v3_fusion/); V3-gated MAE disabled, reporting ungated only")
        return None
    try:
        from ..eval.rq2_three_paradigm_eval import _load_v3
        flow, thresholds, xt_pool, anchor_norm = _load_v3(
            pipe, x_dim=int(embed_dim), c_dim=int(embed_dim))
    except Exception as e:  # missing xt_pool.pt, dim mismatch, corrupt npz, ...
        print(f"{log_prefix}: V3 at {pipe} not reproducible "
              f"({type(e).__name__}: {e}); V3-gated MAE disabled, reporting ungated only")
        return None
    pct = 99 if int(threshold_percentile) >= 99 else 95
    print(f"{log_prefix}: loaded fusion V3 from {pipe} for gating "
          f"(percentile={pct}, anchor={'yes' if anchor_norm is not None else 'no'}, "
          f"xt_pool={'yes' if xt_pool is not None else 'no'})")
    return V3ForGating(flow=flow, thresholds=thresholds, xt_pool=xt_pool,
                       anchor_norm=anchor_norm, percentile=pct)


def gated_fold_mae(
    v3: V3ForGating | None,
    val_samples: list,
    val_predictions,
    val_targets,
) -> dict:
    """V3-gated, event-aggregated MAE for one CV fold (matches the headline).

    Scores each holdout knock's cached ``x_for_v3`` + ``context`` through V3 in
    place (`gate_samples_by_v3`), keeps the V3-flagged ones, then reports the
    EVENT-AGGREGATED MAE over that kept subset (group kept knocks by recording,
    average each recording's predictions, error vs GT, mean over recordings) via
    the shared `event_aggregated_mae` — the same aggregation as the ungated
    headline and full_run Stage 5b.  ``val_samples`` must be the same explicit
    val split, in the same order, the trainer scored into ``val_predictions``;
    the count guard below degrades to ``None`` if that ever breaks rather than
    mis-pairing.

    Returns the gated MAE (``None`` when V3 is unavailable, fires on nothing, or
    the prediction/sample counts disagree), the knocks-kept / total tallies, and
    the number of distinct recordings contributing a gated estimate.  Never
    raises — gating is a reported add-on, not allowed to abort a fold's ungated
    result.
    """
    out: dict = {
        "val_mae_3d_v3gated_m": None,
        "n_val_gated": 0,
        "n_val_total": int(len(val_samples)),
        "n_val_gated_recordings": 0,
    }
    if v3 is None or len(val_samples) == 0:
        return out
    vp = np.asarray(val_predictions)
    vt = np.asarray(val_targets)
    try:
        from ..localization.v3_gating import gate_samples_by_v3
        gres = gate_samples_by_v3(
            v3.flow, v3.thresholds, val_samples, percentile=v3.percentile)
    except Exception as e:  # device / dim / scoring failure — degrade, don't abort
        out["v3_gating_error"] = f"{type(e).__name__}: {e}"
        return out
    keep = gres.keep_mask
    out["n_val_gated"] = int(keep.sum())
    if keep.shape[0] != vp.shape[0] or not keep.any():
        return out  # 0 knocks kept, or alignment broken -> gated MAE stays None
    # keep is a numpy bool array but val_samples is a Python list -> comprehension,
    # never list[keep].  Numpy arrays may be boolean-indexed directly.
    kept_samples = [s for s, k in zip(val_samples, keep) if k]
    mae, _agg, n_rec = event_aggregated_mae(vp[keep], vt[keep], kept_samples)
    out["n_val_gated_recordings"] = int(n_rec)
    out["val_mae_3d_v3gated_m"] = float(mae) if np.isfinite(mae) else None
    return out


def load_or_precompute_cv_samples(
    encoder: V2FusionEncoder,
    v2_cfg: V2SSLConfig,
    *,
    samples_cache: Path | None,
    burst_aware_srp: bool = True,  # retained for call-site compat; ignored
    log_prefix: str,
    v3: V3ForGating | None = None,
) -> list[V4Sample]:
    """Return the shared V4Sample list for a cross-validation driver.

    Loads from ``samples_cache`` when it exists; otherwise gathers the D2/D3/D4/D5
    labelled recordings and precomputes **per-knock** V4 samples (each detected
    knock localized on its own transient-centred crop — the multi-seed-confirmed
    RQ3 win).  ``log_prefix`` tags the progress prints (e.g. "V4 LOPO").

    When ``v3`` is given, each sample's ``x_for_v3`` is pooled with V3's learned
    pool and gets V3's impulse+spectral anchor appended — i.e. built *exactly* as
    the flow scores it — so the per-fold `gated_fold_mae` is on-distribution and
    dimension-correct.  This only changes ``x_for_v3`` (the gating input); the
    SRP/TDOA/`c_t` features the localization head trains on are untouched, so the
    ungated MAE is bit-for-bit unchanged.  Gating-ready samples are cached under a
    DISTINCT ``*.v3ready`` filename so a stale ungated pickle is never reused.

    NOTE: this builder replaced the older fixed-window
    ``precompute_v4_samples`` path; a ``samples_cache`` written by the old
    builder must be regenerated (delete the pickle) to pick up per-knock samples.
    The ``burst_aware_srp`` argument is retained only for call-site
    compatibility and is no longer used (per-knock cropping is inherent).
    """
    v3_xt_pool = v3.xt_pool if v3 is not None else None
    v3_anchor_norm = v3.anchor_norm if v3 is not None else None
    if samples_cache is not None and v3 is not None:
        samples_cache = Path(samples_cache).with_name(
            f"{Path(samples_cache).stem}.v3ready{Path(samples_cache).suffix}")

    if samples_cache is not None and Path(samples_cache).exists():
        with Path(samples_cache).open("rb") as fh:
            samples = pickle.load(fh)
        print(f"{log_prefix}: loaded {len(samples)} cached V4 samples from {samples_cache}")
        return samples

    print(f"{log_prefix}: gathering labeled segments + precomputing V4 samples ...")
    D2 = resolved_loader("d2.yaml")
    D3 = resolved_loader("d3.yaml")
    D4 = resolved_loader("d4.yaml")
    D5 = resolved_loader("d5.yaml")
    d2_labeled = [
        s for s in D2.list_segments()
        if s.is_anomaly and s.spatial_label is not None and s.mode_label is not None
    ]
    d3_segs = D3.list_segments()
    overrides = _d3_spatial_overrides(d3_segs)
    d3_labeled = [s for s in d3_segs if s.recording_id in overrides]
    d4_labeled = [s for s in D4.list_segments() if s.is_anomaly and s.spatial_label is not None]
    d5_labeled = [s for s in D5.list_segments() if s.is_anomaly and s.spatial_label is not None]
    all_labeled = d2_labeled + d3_labeled + d4_labeled + d5_labeled
    print(
        f"  D2={len(d2_labeled)}, D3={len(d3_labeled)}, D4={len(d4_labeled)}, "
        f"D5={len(d5_labeled)}, total={len(all_labeled)} labeled recordings"
    )
    t0 = time.time()
    samples = precompute_v4_knock_event_samples(
        encoder, all_labeled,
        v2_cfg=v2_cfg, grid=V4_CANDIDATE_GRID,
        spatial_label_overrides=overrides,
        v3_xt_pool=v3_xt_pool, v3_anchor_norm=v3_anchor_norm,
    )
    print(f"  precomputed {len(samples)} per-knock V4 samples in {time.time() - t0:.1f}s")
    if samples_cache is not None:
        with Path(samples_cache).open("wb") as fh:
            pickle.dump(samples, fh)
    return samples
