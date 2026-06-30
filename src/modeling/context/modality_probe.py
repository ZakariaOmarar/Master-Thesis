"""V2 modality-balance probe — RQ1 depth diagnostic.

The training-time design choice (`acoustic_dropout_p = 0.0`,
`vibration_dropout_p = 0.5`) was motivated by the empirical asymmetry
in V1 mode-discrimination quality (V1 acoustic NMI ≈ 0.69 vs V1
vibration NMI ≈ 0.26 in the headline run).  An exigent reviewer will
ask:

  "You drop vibration twice as often as acoustic during V2 training.
   Did the trained encoder actually *learn* that acoustic is the
   trunk and vibration is auxiliary?  If yes, the fused representation
   should degrade gracefully when vibration is masked at inference
   and catastrophically when acoustic is masked.  Show me."

This module is the answer.  It scores a fixed val set under three
inference-time conditions and reports NMI / ARI / purity for each:

  1. **Both modalities present** (the headline V2 condition).
  2. **Vibration zeroed at inference** (matches the A1 ablation
     pattern but on the trained V2 — no retraining).
  3. **Acoustic zeroed at inference** (the symmetric counterpart).

If the asymmetric-modality-dropout hypothesis is correct, (2) will
score close to (1) and (3) will collapse toward chance.

This is **post-hoc inference-time analysis only** — the V2 encoder
under test is the one already trained.  No new optimisation step,
no new data.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.data as tud

from ...config import resolve_device
from .cluster_metric import cluster_purity_and_nmi
from .v2_fusion import V2FusionEncoder
from .v2_ssl import (
    V2SSLConfig,
    _collate,
    _PairedGroupedBatchSampler,
    _PairedSegment,
    _PairedWindowedDataset,
)


@dataclass
class ModalityBalanceResult:
    """Headline numbers from the modality-balance probe.

    ``both`` / ``acoustic_only`` / ``vibration_only`` each carry the
    full cluster-metric dict (NMI / ARI / purity / mapping /
    confusion).  ``healthy_segments_used`` records which paired
    segments fed the probe so the breakdown is auditable.
    """

    both: dict
    acoustic_only: dict  # vibration zeroed at inference
    vibration_only: dict  # acoustic zeroed at inference
    healthy_segments_used: list[str]
    n_clusters: int


def _encode_with_modality_masking(
    encoder: V2FusionEncoder,
    loader: tud.DataLoader,
    *,
    zero_acoustic: bool,
    zero_vibration: bool,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Run V2 forward with optional per-modality zero-masking at inference.

    Returns ``(c_t_array, mode_label_list)``.

    Why zero-masking rather than dropping the modality from the
    encoder forward signature: matches the A1 / training-time
    modality-dropout convention exactly.  The encoder still consumes
    a (zero-valued) input tensor in the masked modality, so the
    per-modality CNN and Set-Transformer still execute; only the
    information content is removed.  This is the inference-time
    counterpart of the symmetric / asymmetric modality dropout
    studied in Akbari et al. (2021) for VATT and in Han et al.
    (2024) for multi-modal SSL robustness analyses.
    """
    encoder.eval()
    cs: list[torch.Tensor] = []
    labels: list[str] = []
    with torch.no_grad():
        for batch in loader:
            ac = batch["ac_feat"].to(device)
            vib = batch["vib_feat"].to(device)
            if zero_acoustic:
                ac = torch.zeros_like(ac)
            if zero_vibration:
                vib = torch.zeros_like(vib)
            ac_xyz = batch["ac_xyz"].to(device)
            vib_xyz = batch["vib_xyz"].to(device)
            ds_idx = batch["dataset_idx"].to(device)
            out = encoder(ac, ac_xyz, vib, vib_xyz, ds_idx, mask_p=0.0)
            cs.append(out["context"].cpu())
            labels.extend(batch["mode_label"])
    if not cs:
        return np.zeros((0, encoder.embed_dim), dtype=np.float32), labels
    return torch.cat(cs, dim=0).numpy().astype(np.float32), labels


def run_modality_balance_probe(
    encoder: V2FusionEncoder,
    healthy_segments: Iterable[_PairedSegment],
    *,
    v2_cfg: V2SSLConfig,
    n_clusters: int = 3,
    healthy_mode_labels: tuple[str, ...] = ("Pump", "Standstill", "Turbine"),
    seed: int = 42,
    device: torch.device | str = "auto",
) -> ModalityBalanceResult:
    """Evaluate V2 cluster-purity under three inference-time modality regimes.

    Args:
      encoder: trained V2 fusion encoder (V1 weights pre-loaded).
      healthy_segments: paired healthy segments with explicit mode
        labels.  Typically the same `val_segs` the V2 sanity gate uses.
      v2_cfg: V2 config used for window slicing.
      n_clusters: K for the K-means evaluation (matches the K = 3 mode
        hypothesis).
      healthy_mode_labels: only segments whose `mode_label` is in this
        tuple enter the probe — RandomFault and Unknown windows are
        excluded so the K = 3 cluster eval is well-defined.
      seed: K-means RNG seed.
      device: torch device.

    Returns:
      `ModalityBalanceResult` with the three cluster-metric dicts.
    """
    device = resolve_device(device)
    encoder = encoder.to(device)

    healthy = [s for s in healthy_segments if s.mode_label in healthy_mode_labels]
    if not healthy:
        empty = {"purity": float("nan"), "nmi": float("nan"), "ari": float("nan"),
                 "n_clusters": n_clusters, "n_labels": 0, "label_set": (),
                 "mapping": {}, "confusion": np.zeros((0, 0), dtype=np.int64),
                 "cluster_idx": np.zeros(0, dtype=np.int64), "n_windows": 0}
        return ModalityBalanceResult(
            both=empty,
            acoustic_only=empty,
            vibration_only=empty,
            healthy_segments_used=[],
            n_clusters=n_clusters,
        )

    ds = _PairedWindowedDataset(healthy, v2_cfg)
    if len(ds) == 0:
        empty = {"purity": float("nan"), "nmi": float("nan"), "ari": float("nan"),
                 "n_clusters": n_clusters, "n_labels": 0, "label_set": (),
                 "mapping": {}, "confusion": np.zeros((0, 0), dtype=np.int64),
                 "cluster_idx": np.zeros(0, dtype=np.int64), "n_windows": 0}
        return ModalityBalanceResult(
            both=empty,
            acoustic_only=empty,
            vibration_only=empty,
            healthy_segments_used=[],
            n_clusters=n_clusters,
        )
    sampler = _PairedGroupedBatchSampler(ds, v2_cfg.batch_size, shuffle=False, seed=seed)
    loader = tud.DataLoader(ds, batch_sampler=sampler, collate_fn=_collate)

    def _eval(zero_ac: bool, zero_vib: bool) -> dict:
        c, labels = _encode_with_modality_masking(
            encoder, loader,
            zero_acoustic=zero_ac, zero_vibration=zero_vib, device=device,
        )
        if c.shape[0] == 0 or not labels:
            return {"purity": float("nan"), "nmi": float("nan"), "ari": float("nan"),
                    "n_clusters": n_clusters, "n_labels": 0, "label_set": (),
                    "mapping": {}, "confusion": np.zeros((0, 0), dtype=np.int64),
                    "cluster_idx": np.zeros(0, dtype=np.int64), "n_windows": 0}
        out = cluster_purity_and_nmi(c, labels, n_clusters=n_clusters, seed=seed)
        out["n_windows"] = int(c.shape[0])
        return out

    both = _eval(zero_ac=False, zero_vib=False)
    acoustic_only = _eval(zero_ac=False, zero_vib=True)  # zero vibration → acoustic survives
    vibration_only = _eval(zero_ac=True, zero_vib=False)  # zero acoustic → vibration survives

    return ModalityBalanceResult(
        both=both,
        acoustic_only=acoustic_only,
        vibration_only=vibration_only,
        healthy_segments_used=sorted({s.recording_id for s in healthy}),
        n_clusters=n_clusters,
    )


__all__ = ["ModalityBalanceResult", "run_modality_balance_probe"]
