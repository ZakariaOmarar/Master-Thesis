"""V2 SSL fusion training loop and RQ1 cluster-purity evaluation."""

from collections.abc import Iterable

import numpy as np
import torch
import torch.utils.data as tud
from tqdm.auto import tqdm

from ...config import describe_device, resolve_device
from ...ingestion.test_dataset_loader import TestDatasetLoader
from ..early_stopping import EarlyStopping, cpu_state_dict
from .cluster_metric import cluster_purity_and_nmi
from .v2_fusion import V2FusionEncoder
from .v2_ssl_config import (
    V2Result,
    V2SSLConfig,
)
from .v2_ssl_data import (
    _collate,
    _gather_paired_segments,
    _PairedAugmenter,
    _PairedGroupedBatchSampler,
    _PairedSegment,
    _PairedWindowedDataset,
    _split_segments_by_recording,
)
from .v2_ssl_model import _lmm_loss, _nt_xent, _ProjectionHead


def train_v2_fusion(
    loaders: TestDatasetLoader | Iterable[TestDatasetLoader],
    cfg: V2SSLConfig | None = None,
    *,
    v1_acoustic_state_dict: dict | None = None,
    v1_vibration_state_dict: dict | None = None,
) -> V2Result:
    """Train V2 multimodal SSL on healthy paired windows.

    ``v1_*_state_dict`` are optional `PerModalityEncoder.state_dict()` payloads
    from V1 SSL.  When supplied they initialise the V2 per-modality encoders;
    otherwise the encoders start from random init.  V2 itself is fully
    self-supervised regardless — V1 was already SSL, so the chained system is
    label-free end-to-end.
    """
    cfg = cfg or V2SSLConfig()
    if hasattr(loaders, "list_segments"):
        loaders = [loaders]

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"V2: device={describe_device(device)}")

    segments = _gather_paired_segments(loaders, cfg)
    if not segments:
        raise RuntimeError("V2 SSL: no healthy paired segments found")

    train_segs, val_segs = _split_segments_by_recording(segments, cfg.val_ratio, cfg.seed)
    train_ds = _PairedWindowedDataset(train_segs, cfg, enable_mixup=True)
    val_ds = _PairedWindowedDataset(val_segs, cfg, enable_mixup=False)
    if len(train_ds) == 0:
        raise RuntimeError("V2 SSL: zero training windows after splitting; lower window_seconds")

    pin = device.type == "cuda"
    train_loader = tud.DataLoader(
        train_ds,
        batch_sampler=_PairedGroupedBatchSampler(train_ds, cfg.batch_size, shuffle=True, seed=cfg.seed),
        collate_fn=_collate,
        pin_memory=pin,
    )
    val_loader = tud.DataLoader(
        val_ds,
        batch_sampler=_PairedGroupedBatchSampler(val_ds, cfg.batch_size, shuffle=False, seed=cfg.seed),
        collate_fn=_collate,
        pin_memory=pin,
    )

    encoder = V2FusionEncoder(
        feature_dim=cfg.feature_dim,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        context_mode=cfg.context_mode,
        num_context_seeds=cfg.num_context_seeds,
        acoustic_cnn_width_mult=cfg.acoustic_cnn_width_mult,
    ).to(device)
    encoder.load_v1_weights(v1_acoustic_state_dict, v1_vibration_state_dict, strict=True)

    projection = _ProjectionHead(cfg.embed_dim, cfg.proj_dim).to(device)
    optim = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    aug_gen = torch.Generator(device=device)
    aug_gen.manual_seed(cfg.seed)
    augmenter = _PairedAugmenter(cfg, aug_gen)

    train_history: list[float] = []
    train_simclr: list[float] = []
    train_lmm: list[float] = []
    val_history: list[float] = []

    def _maybe_drop_vib(vib: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(vib) if cfg.drop_vibration else vib

    def _modality_dropout(
        ac: torch.Tensor, vib: torch.Tensor, gen: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Independent per-modality dropout.

        - When `cfg.acoustic_dropout_p > 0` or `cfg.vibration_dropout_p > 0`,
          each modality is dropped (zeroed) with its own probability per
          batch.  This is the publication-default regime and lets the
          asymmetry favour the strong acoustic mode-discriminator.
        - Otherwise falls back to the legacy symmetric 50/50 coin-flip
          gated by `cfg.modality_dropout_p` (kept for back-compatibility).
        """
        # Generator and target device must agree — `gen` lives on the same
        # device as `aug_gen` (the training device).  Sample scalars there.
        gd = gen.device
        ap = float(cfg.acoustic_dropout_p)
        vp = float(cfg.vibration_dropout_p)
        if ap > 0.0 or vp > 0.0:
            if ap > 0.0 and float(torch.rand((), generator=gen, device=gd)) < ap:
                ac = torch.zeros_like(ac)
            if vp > 0.0 and float(torch.rand((), generator=gen, device=gd)) < vp:
                vib = torch.zeros_like(vib)
            return ac, vib
        if cfg.modality_dropout_p <= 0.0:
            return ac, vib
        u = float(torch.rand((), generator=gen, device=gd))
        if u >= cfg.modality_dropout_p:
            return ac, vib
        if float(torch.rand((), generator=gen, device=gd)) < 0.5:
            return torch.zeros_like(ac), vib
        return ac, torch.zeros_like(vib)

    # Early-stop bookkeeping: snapshot the best encoder weights as the val loss
    # improves and restore them at the end (see ``modeling.early_stopping``).
    stopper = EarlyStopping(cfg.patience, cfg.early_stop_min_delta,
                            initial=cpu_state_dict(encoder))
    early_stopped_epoch: int | None = None

    epoch_iter = tqdm(
        range(cfg.epochs),
        desc=f"V2 fusion (cma={cfg.cma_weight}, ctx={cfg.context_mode})",
        unit="epoch",
        leave=False,
    )
    for _epoch in epoch_iter:
        encoder.train()
        projection.train()
        loss_sum = simclr_sum = lmm_sum = 0.0
        n = 0
        for batch in train_loader:
            ac = batch["ac_feat"].to(device)
            vib = _maybe_drop_vib(batch["vib_feat"].to(device))
            ac_xyz = batch["ac_xyz"].to(device)
            vib_xyz = batch["vib_xyz"].to(device)
            ds_idx = batch["dataset_idx"].to(device)

            ac1, vib1 = augmenter(ac, vib)
            ac2, vib2 = augmenter(ac, vib)
            vib1 = _maybe_drop_vib(vib1)
            vib2 = _maybe_drop_vib(vib2)
            # Apply modality dropout *after* SimCLR view augmentation but
            # *before* the encoder.  Each view independently risks losing a
            # modality — the contrastive loss then pulls together the two
            # views' c_t even when one of them is single-modality.
            ac1, vib1 = _modality_dropout(ac1, vib1, aug_gen)
            ac2, vib2 = _modality_dropout(ac2, vib2, aug_gen)

            out1 = encoder(ac1, ac_xyz, vib1, vib_xyz, ds_idx, mask_p=cfg.lmm_mask_p)
            out2 = encoder(ac2, ac_xyz, vib2, vib_xyz, ds_idx, mask_p=0.0)

            z1 = projection(out1["context"])
            z2 = projection(out2["context"])
            if z1.shape[0] < 2:
                continue

            simclr = _nt_xent(z1, z2, cfg.temperature)
            lmm_a = _lmm_loss(out1["a_fused"], out1["a_target"], out1["mask_a"])
            lmm_v = _lmm_loss(out1["v_fused"], out1["v_target"], out1["mask_v"])
            lmm = lmm_a + lmm_v
            cma = (
                _nt_xent(out1["a_summary"], out1["v_summary"], cfg.cma_temperature)
                if cfg.cma_weight > 0.0 and out1["a_summary"].shape[0] >= 2
                else torch.zeros((), device=z1.device, dtype=z1.dtype)
            )
            loss = simclr + cfg.lmm_weight * lmm + cfg.cma_weight * cma

            optim.zero_grad()
            loss.backward()
            optim.step()

            B = z1.shape[0]
            loss_sum += float(loss.item()) * B
            simclr_sum += float(simclr.item()) * B
            lmm_sum += float(lmm.item()) * B
            n += B

        denom = max(1, n)
        train_history.append(loss_sum / denom)
        train_simclr.append(simclr_sum / denom)
        train_lmm.append(lmm_sum / denom)

        encoder.eval()
        projection.eval()
        v_loss = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                ac = batch["ac_feat"].to(device)
                vib = _maybe_drop_vib(batch["vib_feat"].to(device))
                ac_xyz = batch["ac_xyz"].to(device)
                vib_xyz = batch["vib_xyz"].to(device)
                ds_idx = batch["dataset_idx"].to(device)

                ac1, vib1 = augmenter(ac, vib)
                ac2, vib2 = augmenter(ac, vib)
                vib1 = _maybe_drop_vib(vib1)
                vib2 = _maybe_drop_vib(vib2)

                out1 = encoder(ac1, ac_xyz, vib1, vib_xyz, ds_idx, mask_p=cfg.lmm_mask_p)
                out2 = encoder(ac2, ac_xyz, vib2, vib_xyz, ds_idx, mask_p=0.0)
                z1 = projection(out1["context"])
                z2 = projection(out2["context"])
                if z1.shape[0] < 2:
                    continue
                simclr = _nt_xent(z1, z2, cfg.temperature)
                lmm = _lmm_loss(out1["a_fused"], out1["a_target"], out1["mask_a"]) + \
                      _lmm_loss(out1["v_fused"], out1["v_target"], out1["mask_v"])
                cma = (
                    _nt_xent(out1["a_summary"], out1["v_summary"], cfg.cma_temperature)
                    if cfg.cma_weight > 0.0 and out1["a_summary"].shape[0] >= 2
                    else torch.zeros((), device=z1.device, dtype=z1.dtype)
                )
                B = z1.shape[0]
                v_loss += float((simclr + cfg.lmm_weight * lmm + cfg.cma_weight * cma).item()) * B
                v_n += B
        val_history.append(v_loss / max(1, v_n))
        epoch_iter.set_postfix(
            train=f"{train_history[-1]:.3f}",
            val=f"{val_history[-1]:.3f}",
            lmm=f"{train_lmm[-1]:.3f}",
        )

        # Early stop on val total-loss.  See V1 trainer for the pattern.
        cur_val = val_history[-1]
        if stopper.update(cur_val, lambda: cpu_state_dict(encoder)):
            early_stopped_epoch = _epoch + 1
            break

    best_val_loss = stopper.best
    if cfg.restore_best:
        encoder.load_state_dict(stopper.best_snapshot)
    del stopper

    # RQ1 cluster purity is computed on labeled-only val segments — D3/D4
    # speed-bucket recordings have no mode_label so including them in the
    # K=3 K-means against the three known modes would be uninterpretable.
    labeled_modes = set(cfg.healthy_modes)
    val_labeled = [s for s in val_segs if s.mode_label in labeled_modes]
    rq1 = evaluate_rq1_purity(encoder, val_labeled, cfg)

    def _qualify(seg: _PairedSegment) -> str:
        from pathlib import Path as _P

        return f"{_P(seg.source_dir).name}/{seg.recording_id}"

    return V2Result(
        encoder=encoder,
        projection=projection,
        train_loss_history=train_history,
        val_loss_history=val_history,
        train_simclr_history=train_simclr,
        train_lmm_history=train_lmm,
        train_recording_ids=sorted({_qualify(s) for s in train_segs}),
        val_recording_ids=sorted({_qualify(s) for s in val_segs}),
        rq1=rq1,
        drop_vibration=cfg.drop_vibration,
        early_stopped_epoch=early_stopped_epoch,
        best_val_loss=best_val_loss,
    )
def evaluate_rq1_purity(
    encoder: V2FusionEncoder,
    segments: list[_PairedSegment],
    cfg: V2SSLConfig,
    *,
    n_clusters: int = 3,
) -> dict:
    """Compute K-means cluster purity on `c_t` for the RQ1 headline number."""
    if not segments:
        return {"purity": 0.0, "nmi": 0.0, "n_windows": 0, "label_set": tuple()}
    ds = _PairedWindowedDataset(segments, cfg)
    if len(ds) < 2:
        return {"purity": 0.0, "nmi": 0.0, "n_windows": int(len(ds)), "label_set": tuple()}
    loader = tud.DataLoader(
        ds,
        batch_sampler=_PairedGroupedBatchSampler(ds, cfg.batch_size, shuffle=False, seed=cfg.seed),
        collate_fn=_collate,
    )

    device = next(encoder.parameters()).device
    encoder.eval()
    contexts: list[np.ndarray] = []
    labels: list[str] = []
    with torch.no_grad():
        for batch in loader:
            ac = batch["ac_feat"].to(device)
            vib = batch["vib_feat"].to(device)
            if cfg.drop_vibration:
                vib = torch.zeros_like(vib)
            out = encoder(
                ac,
                batch["ac_xyz"].to(device),
                vib,
                batch["vib_xyz"].to(device),
                batch["dataset_idx"].to(device),
                mask_p=0.0,
            )
            contexts.append(out["context"].cpu().numpy())
            labels.extend(batch["mode_label"])
    embeddings = np.concatenate(contexts, axis=0)
    metric = cluster_purity_and_nmi(embeddings, labels, n_clusters=n_clusters, seed=cfg.seed)
    metric["n_windows"] = int(embeddings.shape[0])
    return metric


__all__ = [
    "V2Result",
    "V2SSLConfig",
    "evaluate_rq1_purity",
    "train_v2_fusion",
]
