"""V3 anomaly trainer data pipeline: fused-feature caching, pooling, extraction."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud

from ..context.v2_fusion import V2FusionEncoder
from ..context.v2_ssl import (
    V2SSLConfig,
    _collate,
    _PairedGroupedBatchSampler,
    _PairedSegment,
    _PairedWindowedDataset,
    _precompute_paired,
)
from .impulse_anchor import append_anchor, impulse_spectral_anchor
from .v3_trainer_config import V3Config
from .v3_trainer_model import _augment_anchor


def _extract_xc(
    encoder: V2FusionEncoder,
    loader: tud.DataLoader,
    device: torch.device,
    xt_pool: nn.Module | None = None,
    grad: bool = False,
    anchor_norm: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Run frozen V2 encoder forward; collect x_t, c_t, mode labels.

    When ``xt_pool`` is None (legacy path) or an ``nn.Identity``-like marker,
    ``x_t = fused.mean(dim=1)`` — preserves the original `cfg.xt_pool="mean"`
    behaviour.  When ``xt_pool`` is an :class:`_XtPool`, the V2 encoder is run
    under ``no_grad`` (it is frozen) but the ``xt_pool`` forward is on the
    autograd tape so the V3 trainer can co-optimise it with the flow.

    The ``grad`` flag controls whether ``xt_pool`` is invoked under
    ``torch.no_grad`` (set to True at training time, False for inference /
    feature extraction).
    """
    encoder.eval()
    xs: list[torch.Tensor] = []
    cs: list[torch.Tensor] = []
    labels: list[str] = []
    use_grad = bool(grad and xt_pool is not None)
    for batch in loader:
        ac = batch["ac_feat"].to(device)
        vib = batch["vib_feat"].to(device)
        ac_xyz = batch["ac_xyz"].to(device)
        vib_xyz = batch["vib_xyz"].to(device)
        ds_idx = batch["dataset_idx"].to(device)
        with torch.no_grad():
            out = encoder(ac, ac_xyz, vib, vib_xyz, ds_idx, mask_p=0.0)
            fused = torch.cat([out["a_fused"], out["v_fused"]], dim=1)
            c_t = out["context"]
        if xt_pool is not None:
            if use_grad:
                x_t = xt_pool(fused)
            else:
                with torch.no_grad():
                    x_t = xt_pool(fused)
        else:
            x_t = fused.mean(dim=1)
        x_t = append_anchor(x_t, batch["ac_feat"], batch["vib_feat"], anchor_norm)
        xs.append(x_t.detach().cpu())
        cs.append(c_t.detach().cpu())
        labels.extend(batch["mode_label"])
    return torch.cat(xs, dim=0), torch.cat(cs, dim=0), labels
def _resolve_v3_override(
    v3_cfg: V3Config, dataset_ids: Iterable[str]
) -> dict[str, float] | None:
    """Normalise the V3 ``window_seconds_override`` to a per-dataset dict.

    Returns None when no override is set; otherwise a fully-populated
    dict keyed by the dataset ids the trainer is about to consume.
    """
    override = v3_cfg.window_seconds_override
    if override is None:
        return None
    if isinstance(override, dict):
        return {str(k): float(v) for k, v in override.items()}
    return {str(d): float(override) for d in dataset_ids}
def _make_override_v2_cfg(
    v2_cfg: V2SSLConfig,
    override_per_dataset: dict[str, float] | None,
) -> V2SSLConfig:
    """Return a shallow `V2SSLConfig` copy with the V3 override applied.

    The override is materialised as a per-dataset
    ``window_scales_seconds_per_dataset`` dict so the V3
    ``_PairedWindowedDataset`` reuses the same multi-scale plumbing as
    V1 / V2 SSL.  When ``override_per_dataset`` is None the original
    config is returned unchanged.
    """
    if not override_per_dataset:
        return v2_cfg
    per_ds = {ds: (float(v),) for ds, v in override_per_dataset.items()}
    return replace(v2_cfg, window_scales_seconds_per_dataset=per_ds)
def _cache_fused(
    encoder: V2FusionEncoder,
    loader: tud.DataLoader,
    device: torch.device,
) -> list[dict]:
    """Run the frozen V2 encoder once and cache per-batch (fused, c, labels).

    Used by the ``xt_pool="pma2"`` path so the learnable :class:`_XtPool`
    can be co-optimised with the flow over many epochs without re-running
    the encoder.  Each cache entry preserves the grouped-batch contract
    (uniform `N_a + N_v` within a batch) so `_XtPool(fused)` succeeds
    without padding masks.

    Memory cost is proportional to ``num_windows * (N_a + N_v) * embed_dim``;
    at the typical V3 cohort size (~10 k windows, 13 channels, 128 dims)
    this is ~67 MB on CPU — well below RAM budgets.
    """
    encoder.eval()
    cache: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            ac = batch["ac_feat"].to(device)
            vib = batch["vib_feat"].to(device)
            ac_xyz = batch["ac_xyz"].to(device)
            vib_xyz = batch["vib_xyz"].to(device)
            ds_idx = batch["dataset_idx"].to(device)
            out = encoder(ac, ac_xyz, vib, vib_xyz, ds_idx, mask_p=0.0)
            fused = torch.cat([out["a_fused"], out["v_fused"]], dim=1).detach().cpu()
            c = out["context"].detach().cpu()
            # Per-window impulse+spectral anchor from the same windowed log-mel
            # +CWT features (RQ2: augments the conditional flow input so the
            # detector sees the knock the SSL embedding discards).  Cached so
            # the joint training loop reuses it without recompute.
            anchor = impulse_spectral_anchor(
                batch["ac_feat"].detach().cpu().numpy(),
                batch["vib_feat"].detach().cpu().numpy(),
            )
            cache.append(
                {
                    "fused": fused,
                    "c": c,
                    "anchor": anchor,
                    "labels": list(batch["mode_label"]),
                    # Per-window recording id (kept in cache order) so the
                    # held-out NLL paired test (V3 vs A2) can resample at the
                    # recording level instead of the window level — see
                    # `eval.statistics.paired_bootstrap_test(groups=)`.
                    "recording_ids": list(batch["recording_id"]),
                }
            )
    return cache
def _pool_cached_x(
    cache: list[dict],
    xt_pool: nn.Module | None,
    device: torch.device,
    anchor_norm: tuple[np.ndarray, np.ndarray] | None = None,
) -> torch.Tensor:
    """Apply ``xt_pool`` (or the legacy mean-pool) over a fused cache.

    Always runs under ``no_grad`` — used for the final extract-once paths
    (val NLL evaluation, threshold fitting, post-training scoring).  When
    ``anchor_norm`` is supplied the standardized per-window impulse+spectral
    anchor is concatenated to x (RQ2 conditional-flow input augmentation).
    """
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for entry in cache:
            fused = entry["fused"].to(device)
            x = fused.mean(dim=1) if xt_pool is None else xt_pool(fused)
            x = _augment_anchor(x, entry["anchor"], anchor_norm)
            chunks.append(x.detach().cpu())
    return torch.cat(chunks, dim=0)
def _stack_c_from_cache(cache: list[dict]) -> torch.Tensor:
    return torch.cat([entry["c"] for entry in cache], dim=0)
def _stack_labels_from_cache(cache: list[dict]) -> list[str]:
    out: list[str] = []
    for entry in cache:
        out.extend(entry["labels"])
    return out
def _stack_recording_ids_from_cache(cache: list[dict]) -> list[str]:
    """Per-window recording ids in cache order (mirrors `_stack_labels_from_cache`).

    Used to attach a recording-level group label to each held-out val window so
    the V3-vs-A2 NLL paired test can use a block bootstrap.  Tolerates older
    caches written before `recording_ids` was added by returning ``""`` for
    those windows (which the paired test then treats as a single fallback group).
    """
    out: list[str] = []
    for entry in cache:
        rec = entry.get("recording_ids")
        if rec is None:
            out.extend([""] * len(entry["labels"]))
        else:
            out.extend(rec)
    return out


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def precompute_paired(seg, cfg: V2SSLConfig) -> _PairedSegment | None:
    """Public wrapper for the V2 paired-feature precomputation (used by tests)."""
    return _precompute_paired(seg, cfg)
def _extract_x_for_segment(
    v2_encoder: V2FusionEncoder,
    seg: _PairedSegment,
    cfg: V2SSLConfig,
    device: torch.device,
    xt_pool: nn.Module | None = None,
    anchor_norm: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Run V2 forward over every window in `seg`; return the V3 flow input x.

    Pools with ``xt_pool`` (pma2) when supplied else mean, and appends the
    standardized impulse+spectral anchor when ``anchor_norm`` is given — so x
    matches exactly what the conditional flow was trained on.
    """
    ds = _PairedWindowedDataset([seg], cfg)
    if len(ds) == 0:
        return np.zeros((0, v2_encoder.embed_dim), dtype=np.float32)
    sampler = _PairedGroupedBatchSampler(ds, cfg.batch_size, shuffle=False, seed=0)
    loader = tud.DataLoader(ds, batch_sampler=sampler, collate_fn=_collate)
    xs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            out = v2_encoder(
                batch["ac_feat"].to(device), batch["ac_xyz"].to(device),
                batch["vib_feat"].to(device), batch["vib_xyz"].to(device),
                batch["dataset_idx"].to(device), mask_p=0.0,
            )
            fused = torch.cat([out["a_fused"], out["v_fused"]], dim=1)
            x = xt_pool(fused) if xt_pool is not None else fused.mean(dim=1)
            x = append_anchor(x, batch["ac_feat"], batch["vib_feat"], anchor_norm)
            xs.append(x.cpu().numpy())
    return np.concatenate(xs, axis=0).astype(np.float32)
