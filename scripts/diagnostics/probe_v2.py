"""Probe V2's internal representations to locate the purity-drop in the
acoustic↔vibration fusion path.

Investigates the hypothesis that, since V1 vibration alone (purity 0.667)
beats V2 fusion (0.627), the cross-attention block must be losing the
vibration signal somewhere between input and `c_t`.

For each held-out window we extract a ladder of representations:

  1.  V1 acoustic    summary (PMA pool — V1's official representation)
  2.  V1 vibration   summary (PMA pool — V1's official representation)
  3.  V1 ac+vib concat (simple early-fusion baseline)
  4.  V2 acoustic   per-modality summary  (= PMA pool of V2.acoustic encoder)
  5.  V2 vibration  per-modality summary  (= PMA pool of V2.vibration encoder)
  6.  V2 fused_a    mean-pool             (post-cross-attention acoustic)
  7.  V2 fused_v    mean-pool             (post-cross-attention vibration)
  8.  V2 fused_a    PMA-pool              (PMA over fused acoustic only)
  9.  V2 fused_v    PMA-pool              (PMA over fused vibration only)
 10.  V2 c_t                              (PMA over [fused_a; fused_v]) — the
                                          official V2 RQ1 representation

Each row is K-means(k=4) → Hungarian-matched to mode labels → cluster purity.

Run with:
    python -m scripts.diagnostics.probe_v2
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.config import resolve_device
from src.modeling.context.cluster_metric import cluster_purity_and_nmi
from src.modeling.context.v1_ssl import V1SSLConfig
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.context.v2_ssl import (
    V2SSLConfig,
    _collate,
    _gather_paired_segments,
    _PairedGroupedBatchSampler,
    _PairedWindowedDataset,
    _split_segments_by_recording,
)
from src.modeling.encoders import PerModalityEncoder
from src.modeling.orchestration.full_run import resolved_loader, v2_config

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "full_run"


def _load_v1_encoder(modality: str, cfg: V1SSLConfig) -> PerModalityEncoder:
    enc = PerModalityEncoder(
        modality=modality,
        feature_dim=cfg.feature_dim,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
    )
    enc.load_state_dict(torch.load(RESULTS / "v1" / f"{modality}.pt", map_location="cpu"))
    enc.eval()
    return enc


def _load_v2_encoder(cfg: V2SSLConfig, ckpt: Path | None = None) -> V2FusionEncoder:
    if ckpt is None:
        ckpt = RESULTS / "v2" / "encoder.pt"
    enc = V2FusionEncoder(
        feature_dim=cfg.feature_dim,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
    )
    enc.load_state_dict(torch.load(ckpt, map_location="cpu"))
    enc.eval()
    return enc


def main(v2_ckpt: Path | None = None, label: str = "v2") -> dict:
    cfg = v2_config(quick=False)  # match the full-epoch run
    device = resolve_device("auto")

    print(f"Loading checkpoints ... (V2 from {v2_ckpt or 'results/full_run/v2/encoder.pt'})")
    v1_acoustic = _load_v1_encoder("acoustic", cfg).to(device)
    v1_vibration = _load_v1_encoder("vibration", cfg).to(device)
    v2 = _load_v2_encoder(cfg, ckpt=v2_ckpt).to(device)

    print("Gathering D1+D2 healthy paired windows ...")
    D1 = resolved_loader("d1.yaml")
    D2 = resolved_loader("d2.yaml")
    segments = _gather_paired_segments([D1, D2], cfg)
    _, val_segs = _split_segments_by_recording(segments, cfg.val_ratio, cfg.seed)
    val_ds = _PairedWindowedDataset(val_segs, cfg)
    print(f"  val windows: {len(val_ds)}")

    sampler = _PairedGroupedBatchSampler(val_ds, cfg.batch_size, shuffle=False, seed=cfg.seed)
    loader = torch.utils.data.DataLoader(val_ds, batch_sampler=sampler, collate_fn=_collate)

    rows: dict[str, list[np.ndarray]] = {
        "v1_acoustic_summary": [],
        "v1_vibration_summary": [],
        "v1_concat": [],
        "v2_acoustic_summary": [],
        "v2_vibration_summary": [],
        "v2_per_modality_concat": [],  # [v2_a_summary; v2_v_summary]
        "v2_fused_a_mean": [],
        "v2_fused_v_mean": [],
        "v2_fused_a_pma": [],
        "v2_fused_v_pma": [],
        "v2_fused_concat_pma": [],  # PMA over [fused_a; fused_v] — official c_t
        "v2_skip_to_summary": [],  # [PMA(fused_a); PMA(fused_v)] — bypass joint PMA
        "v2_context": [],
    }
    labels: list[str] = []

    print("Forwarding all val windows through every probe point ...")
    with torch.no_grad():
        for batch in loader:
            ac = batch["ac_feat"].to(device)
            vib = batch["vib_feat"].to(device)
            ac_xyz = batch["ac_xyz"].to(device)
            vib_xyz = batch["vib_xyz"].to(device)
            ds_idx = batch["dataset_idx"].to(device)

            # ── V1 encoders, run independently ─────────────────────────
            _, v1_a = v1_acoustic(ac, ac_xyz, ds_idx)
            _, v1_v = v1_vibration(vib, vib_xyz, ds_idx)
            rows["v1_acoustic_summary"].append(v1_a.cpu().numpy())
            rows["v1_vibration_summary"].append(v1_v.cpu().numpy())
            rows["v1_concat"].append(torch.cat([v1_a, v1_v], dim=-1).cpu().numpy())

            # ── V2 per-modality (jointly trained but with V1-init) ─────
            v2_a_tokens, v2_a_summary = v2.acoustic(ac, ac_xyz, ds_idx)
            v2_v_tokens, v2_v_summary = v2.vibration(vib, vib_xyz, ds_idx)
            rows["v2_acoustic_summary"].append(v2_a_summary.cpu().numpy())
            rows["v2_vibration_summary"].append(v2_v_summary.cpu().numpy())
            rows["v2_per_modality_concat"].append(
                torch.cat([v2_a_summary, v2_v_summary], dim=-1).cpu().numpy()
            )

            # ── V2 cross-attention fusion + per-modality / joint pools ─
            fused_a, fused_v, context = v2.fuse_and_pool(v2_a_tokens, v2_v_tokens)
            rows["v2_fused_a_mean"].append(fused_a.mean(dim=1).cpu().numpy())
            rows["v2_fused_v_mean"].append(fused_v.mean(dim=1).cpu().numpy())
            # PMA over a single modality's fused tokens (re-use V2's context_pool)
            fa_pma = v2.context_pool(fused_a).squeeze(1)
            fv_pma = v2.context_pool(fused_v).squeeze(1)
            rows["v2_fused_a_pma"].append(fa_pma.cpu().numpy())
            rows["v2_fused_v_pma"].append(fv_pma.cpu().numpy())
            rows["v2_fused_concat_pma"].append(context.cpu().numpy())
            rows["v2_skip_to_summary"].append(
                torch.cat([fa_pma, fv_pma], dim=-1).cpu().numpy()
            )
            rows["v2_context"].append(context.cpu().numpy())

            labels.extend(batch["mode_label"])

    print(f"Total val windows: {len(labels)}")
    print(f"Mode label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    print("\nCluster purity ladder (K-means k=4, Hungarian-matched):")
    print(f"{'Stage':<32} {'purity':>8} {'NMI':>8} {'dim':>6}")
    print("-" * 60)
    out: dict[str, dict] = {}
    for name, chunks in rows.items():
        emb = np.concatenate(chunks, axis=0)
        m = cluster_purity_and_nmi(emb, labels, n_clusters=4, seed=0)
        purity = m["purity"]
        nmi = m["nmi"]
        out[name] = {"purity": float(purity), "nmi": float(nmi), "dim": int(emb.shape[1])}
        print(f"{name:<32} {purity:>8.3f} {nmi:>8.3f} {emb.shape[1]:>6}")

    out_path = RESULTS / f"{label}_probe.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nProbe results written to {out_path}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="V2 encoder .pt to probe")
    parser.add_argument("--label", type=str, default="v2", help="output filename prefix")
    args = parser.parse_args()
    main(v2_ckpt=Path(args.ckpt) if args.ckpt else None, label=args.label)
