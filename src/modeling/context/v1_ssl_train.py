"""V1 SSL training loop and cluster-purity sanity gate."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
import torch.utils.data as tud
from tqdm.auto import tqdm

from ...config import describe_device, resolve_device
from ...ingestion.test_dataset_loader import (
    TestDatasetLoader,
)
from ..early_stopping import EarlyStopping, cpu_state_dict
from ..encoders import PerModalityEncoder
from .cluster_metric import cluster_purity_and_nmi
from .v1_ssl_config import (
    V1Result,
    V1SSLConfig,
)
from .v1_ssl_data import (
    _Augmenter,
    _collate,
    _gather_healthy_segments,
    _GroupedBatchSampler,
    _PrecomputedSegment,
    _split_segments_by_recording,
    _WindowedFeatureDataset,
)
from .v1_ssl_model import _nt_xent, _ProjectionHead


def _encode_summary(
    encoder: PerModalityEncoder, batch: dict, device: torch.device
) -> torch.Tensor:
    feat = batch["feat"].to(device)
    xyz = batch["xyz"].to(device)
    dataset_idx = batch["dataset_idx"].to(device)
    _, summary = encoder(feat, xyz, dataset_idx)
    return summary
def train_v1_per_modality(
    loaders: TestDatasetLoader | Iterable[TestDatasetLoader],
    modality: Literal["acoustic", "vibration"],
    cfg: V1SSLConfig | None = None,
) -> V1Result:
    cfg = cfg or V1SSLConfig()
    # Accept either a single loader or any iterable of loaders.  Duck-typed so
    # test stubs that quack like `TestDatasetLoader` (have `list_segments`) work.
    if hasattr(loaders, "list_segments"):
        loaders = [loaders]

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"V1 {modality}: device={describe_device(device)}")

    segments = _gather_healthy_segments(loaders, modality, cfg)
    if not segments:
        raise RuntimeError("V1 SSL: no healthy segments found")

    train_segs, val_segs = _split_segments_by_recording(segments, cfg.val_ratio, cfg.seed)

    train_ds = _WindowedFeatureDataset(train_segs, cfg, enable_mixup=True)
    val_ds = _WindowedFeatureDataset(val_segs, cfg, enable_mixup=False)
    if len(train_ds) == 0:
        raise RuntimeError("V1 SSL: zero training windows after splitting; lower window_seconds")

    pin = device.type == "cuda"
    train_loader = tud.DataLoader(
        train_ds,
        batch_sampler=_GroupedBatchSampler(train_ds, cfg.batch_size, shuffle=True, seed=cfg.seed),
        collate_fn=_collate,
        pin_memory=pin,
    )
    val_loader = tud.DataLoader(
        val_ds,
        batch_sampler=_GroupedBatchSampler(val_ds, cfg.batch_size, shuffle=False, seed=cfg.seed),
        collate_fn=_collate,
        pin_memory=pin,
    )

    encoder = PerModalityEncoder(
        modality=modality,
        feature_dim=cfg.feature_dim,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        acoustic_cnn_width_mult=cfg.acoustic_cnn_width_mult,
    ).to(device)
    projection = _ProjectionHead(cfg.embed_dim, cfg.proj_dim).to(device)
    optim = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Augmenter sees CPU tensors (it runs before the .to(device) move below),
    # so the generator must be on CPU — torch requires generator and target
    # tensor device to match.
    aug_gen = torch.Generator(device="cpu")
    aug_gen.manual_seed(cfg.seed)
    augmenter = _Augmenter(modality, cfg, aug_gen)

    train_history: list[float] = []
    val_history: list[float] = []

    # Early-stop bookkeeping: snapshot the best encoder weights as the val loss
    # improves and restore them at the end (see ``modeling.early_stopping``).
    stopper = EarlyStopping(cfg.patience, cfg.early_stop_min_delta,
                            initial=cpu_state_dict(encoder))
    early_stopped_epoch: int | None = None

    epoch_iter = tqdm(
        range(cfg.epochs),
        desc=f"V1 {modality}",
        unit="epoch",
        leave=False,
    )
    for epoch in epoch_iter:
        encoder.train()
        projection.train()
        epoch_loss = 0.0
        n = 0
        for batch in train_loader:
            view1 = augmenter(batch["feat"]).to(device)
            view2 = augmenter(batch["feat"]).to(device)
            xyz = batch["xyz"].to(device)
            ds_idx = batch["dataset_idx"].to(device)

            _, s1 = encoder(view1, xyz, ds_idx)
            _, s2 = encoder(view2, xyz, ds_idx)
            z1 = projection(s1)
            z2 = projection(s2)
            if z1.shape[0] < 2:
                continue
            loss = _nt_xent(z1, z2, cfg.temperature)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item()) * z1.shape[0]
            n += z1.shape[0]
        train_history.append(epoch_loss / max(1, n))

        encoder.eval()
        projection.eval()
        epoch_val = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                view1 = augmenter(batch["feat"]).to(device)
                view2 = augmenter(batch["feat"]).to(device)
                xyz = batch["xyz"].to(device)
                ds_idx = batch["dataset_idx"].to(device)
                _, s1 = encoder(view1, xyz, ds_idx)
                _, s2 = encoder(view2, xyz, ds_idx)
                z1 = projection(s1)
                z2 = projection(s2)
                if z1.shape[0] < 2:
                    continue
                loss = _nt_xent(z1, z2, cfg.temperature)
                epoch_val += float(loss.item()) * z1.shape[0]
                n_val += z1.shape[0]
        val_history.append(epoch_val / max(1, n_val))
        epoch_iter.set_postfix(
            train=f"{train_history[-1]:.3f}", val=f"{val_history[-1]:.3f}"
        )

        # Early stop on val NT-Xent loss.  Snapshot the encoder (the projection
        # head is discarded post-training) every time the val loss improves
        # past `early_stop_min_delta`.  Break when patience runs out.
        cur_val = val_history[-1]
        if stopper.update(cur_val, lambda: cpu_state_dict(encoder)):
            early_stopped_epoch = epoch + 1
            break

    best_val_loss = stopper.best
    if cfg.restore_best:
        encoder.load_state_dict(stopper.best_snapshot)
    del stopper  # let GC reclaim before sanity-gate eval allocates

    # Sanity gate evaluates on the held-out *labeled* subset only.  D3/D4
    # healthy windows participate in training (they're inside `val_segs`
    # by recording split) but their mode_label is None — including them in
    # the K=3 K-means against the three known modes would give a noisy
    # quality number.  Filter to segments with an explicit mode_label.
    labeled_modes = set(cfg.healthy_modes)
    val_labeled = [
        s for s in val_segs if s.mode_label in labeled_modes
    ]
    sanity = evaluate_sanity_gate(encoder, val_labeled, modality, cfg)

    # Expose recording IDs as fully-qualified `<source_dir_basename>/<recording_id>`
    # so D1's same-`recording_id`-in-different-folders case (e.g., `All/Pump` vs
    # `Pump/Pump`) is disambiguated.  Disjointness of these sets is the
    # held-out-recording invariant.
    def _qualify(seg: _PrecomputedSegment) -> str:
        from pathlib import Path as _P

        return f"{_P(seg.source_dir).name}/{seg.recording_id}"

    return V1Result(
        encoder=encoder,
        projection=projection,
        train_loss_history=train_history,
        val_loss_history=val_history,
        train_recording_ids=sorted({_qualify(s) for s in train_segs}),
        val_recording_ids=sorted({_qualify(s) for s in val_segs}),
        sanity_gate=sanity,
        modality=modality,
        early_stopped_epoch=early_stopped_epoch,
        best_val_loss=best_val_loss,
    )
def evaluate_sanity_gate(
    encoder: PerModalityEncoder,
    segments: list[_PrecomputedSegment],
    modality: Literal["acoustic", "vibration"],
    cfg: V1SSLConfig,
) -> dict:
    """Compute K-means cluster purity vs folder labels on `segments`."""
    if not segments:
        return {"purity": 0.0, "nmi": 0.0, "n_windows": 0, "label_set": tuple()}

    ds = _WindowedFeatureDataset(segments, cfg)
    if len(ds) < 2:
        return {"purity": 0.0, "nmi": 0.0, "n_windows": int(len(ds)), "label_set": tuple()}

    loader = tud.DataLoader(
        ds,
        batch_sampler=_GroupedBatchSampler(ds, cfg.batch_size, shuffle=False, seed=cfg.seed),
        collate_fn=_collate,
    )

    device = next(encoder.parameters()).device
    encoder.eval()
    summaries: list[np.ndarray] = []
    labels: list[str] = []
    with torch.no_grad():
        for batch in loader:
            summary = _encode_summary(encoder, batch, device)
            summaries.append(summary.cpu().numpy())
            labels.extend(batch["mode_label"])
    embeddings = np.concatenate(summaries, axis=0)
    # K=3 against the three healthy operating modes (Pump, Standstill, Turbine).
    # RandomFault is anomaly data, not a fourth mode — it is filtered out by
    # `healthy_modes` upstream of this evaluation.
    metric = cluster_purity_and_nmi(embeddings, labels, n_clusters=3, seed=cfg.seed)
    metric["n_windows"] = int(embeddings.shape[0])
    return metric


__all__ = [
    "V1Result",
    "V1SSLConfig",
    "evaluate_sanity_gate",
    "train_v1_per_modality",
]
