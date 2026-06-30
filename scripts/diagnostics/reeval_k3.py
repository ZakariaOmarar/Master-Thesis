"""Re-evaluate trained V2 encoders under the correct setup:
K=3 on Pump+Standstill+Turbine held-out windows only — no RandomFault.

The V1/V2 trainer's `healthy_modes` previously included `RandomFault`, which
let anomaly data into both the SSL training pool and the cluster-purity
evaluation.  This script does the *evaluation* fix (no retraining): for
each existing encoder checkpoint, it gathers held-out windows that come
exclusively from Pump / Standstill / Turbine recordings, runs K-means(k=3)
on `c_t`, Hungarian-matches to those three labels, and reports purity.

Run:
    python -m scripts.diagnostics.reeval_k3
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.modeling.context.cluster_metric import cluster_purity_and_nmi
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.context.v2_ssl import (
    V2SSLConfig,
    _collate,
    _gather_paired_segments,
    _PairedGroupedBatchSampler,
    _PairedWindowedDataset,
)
from src.modeling.orchestration.full_run import resolved_loader, v2_config

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "full_run"
HEALTHY_MODES_K3 = ("Pump", "Standstill", "Turbine")


def _load_encoder(
    path: Path, cfg: V2SSLConfig, context_mode: str = "joint_pma"
) -> V2FusionEncoder | None:
    """Load a V2 encoder; return None if the checkpoint shape doesn't match
    the current architecture (e.g. legacy `n_datasets=4` encoders saved
    before D4 was integrated)."""
    enc = V2FusionEncoder(
        feature_dim=cfg.feature_dim,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        context_mode=context_mode,
    )
    state = torch.load(path, map_location="cpu")
    try:
        enc.load_state_dict(state)
    except RuntimeError as exc:
        print(f"  [SKIP] {path.name}: incompatible checkpoint ({exc.args[0].splitlines()[0][:80]})")
        return None
    enc.eval()
    return enc


def _gather_healthy_only(cfg: V2SSLConfig):
    """Return (cfg with healthy_modes=K3, all healthy paired segments).

    Cluster purity is reported on ALL healthy recordings rather than a
    held-out split because the post-RandomFault-removal healthy pool is only
    ~7 recordings; a 30 % val split leaves 2 recordings that on most seeds
    cover a single mode, making K=3 cluster purity an artifact rather than a
    measurement.  The trade-off is that some windows in this evaluation were
    in the contrastive pool during training; this is clustering-quality
    introspection of the trained representation, not held-out generalisation.
    """
    cfg_clean = V2SSLConfig(**{**asdict(cfg), "healthy_modes": HEALTHY_MODES_K3})
    D1 = resolved_loader("d1.yaml")
    D2 = resolved_loader("d2.yaml")
    segs = _gather_paired_segments([D1, D2], cfg_clean)
    return cfg_clean, segs


@torch.no_grad()
def _purity_k3(encoder: V2FusionEncoder, val_segs, cfg: V2SSLConfig) -> dict:
    ds = _PairedWindowedDataset(val_segs, cfg)
    if len(ds) < 3:
        return {"purity": 0.0, "nmi": 0.0, "n_windows": int(len(ds)), "label_set": tuple()}
    sampler = _PairedGroupedBatchSampler(ds, cfg.batch_size, shuffle=False, seed=cfg.seed)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=_collate)
    contexts: list[np.ndarray] = []
    labels: list[str] = []
    for batch in loader:
        out = encoder(
            batch["ac_feat"], batch["ac_xyz"], batch["vib_feat"], batch["vib_xyz"],
            batch["dataset_idx"], mask_p=0.0,
        )
        contexts.append(out["context"].cpu().numpy())
        labels.extend(batch["mode_label"])
    embeddings = np.concatenate(contexts, axis=0)
    metric = cluster_purity_and_nmi(embeddings, labels, n_clusters=3, seed=0)
    metric["n_windows"] = int(embeddings.shape[0])
    return metric


def main() -> dict:
    cfg = v2_config(quick=False)
    cfg_clean, all_healthy = _gather_healthy_only(cfg)

    rec_counts: dict[str, int] = {}
    for s in all_healthy:
        rec_counts[s.mode_label] = rec_counts.get(s.mode_label, 0) + 1
    print(f"All healthy paired recordings (mode -> #recordings): {rec_counts}")

    checkpoints: dict[str, tuple[Path, str]] = {
        "vanilla_joint_cma0_0": (RESULTS / "v2" / "encoder.pt", "joint_pma"),
        "joint_cma0_5":         (RESULTS / "v2_cma" / "cma_w0_5.pt", "joint_pma"),
        "joint_cma1_0":         (RESULTS / "v2_cma" / "cma_w1_0.pt", "joint_pma"),
        "joint_cma0_1":         (RESULTS / "v2_sweep" / "cma_w0_1_joint.pt", "joint_pma"),
        "joint_cma0_25":        (RESULTS / "v2_sweep" / "cma_w0_25_joint.pt", "joint_pma"),
        "skip_cma0_5":          (RESULTS / "v2_sweep" / "cma_w0_5_skip.pt", "skip"),
        "dual_cma0_5":          (RESULTS / "v2_sweep" / "cma_w0_5_dual.pt", "dual_pma"),
    }

    # V1 standalone baselines: cluster on the per-modality summary directly.
    print("\n[V1 standalone references]")
    for v1_name in ("acoustic", "vibration"):
        sd = torch.load(RESULTS / "v1" / f"{v1_name}.pt", map_location="cpu")
        from src.modeling.encoders import PerModalityEncoder
        enc = PerModalityEncoder(
            modality=v1_name,
            feature_dim=cfg_clean.feature_dim,
            embed_dim=cfg_clean.embed_dim,
            n_heads=cfg_clean.n_heads,
        )
        enc.load_state_dict(sd)
        enc.eval()
        ds = _PairedWindowedDataset(all_healthy, cfg_clean)
        sampler = _PairedGroupedBatchSampler(ds, cfg_clean.batch_size, shuffle=False, seed=cfg_clean.seed)
        loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=_collate)
        summaries: list[np.ndarray] = []
        labels: list[str] = []
        with torch.no_grad():
            for batch in loader:
                feat = batch["ac_feat"] if v1_name == "acoustic" else batch["vib_feat"]
                xyz = batch["ac_xyz"] if v1_name == "acoustic" else batch["vib_xyz"]
                _, summary = enc(feat, xyz, batch["dataset_idx"])
                summaries.append(summary.cpu().numpy())
                labels.extend(batch["mode_label"])
        emb = np.concatenate(summaries, axis=0)
        m = cluster_purity_and_nmi(emb, labels, n_clusters=3, seed=0)
        print(
            f"v1_{v1_name:<22} {float(m['purity']):>8.3f} {float(m['nmi']):>8.3f} {emb.shape[0]:>8}"
        )

    print("\nK=3 cluster purity on ALL Pump+Standstill+Turbine windows (no holdout):")
    print(f"{'variant':<28} {'purity':>8} {'NMI':>8} {'n_win':>8}")
    print("-" * 56)
    rows: dict = {}
    for name, (ckpt, mode) in checkpoints.items():
        if not ckpt.exists():
            print(f"{name:<28} (missing checkpoint at {ckpt})")
            continue
        enc = _load_encoder(ckpt, cfg_clean, context_mode=mode)
        if enc is None:
            continue  # skipped due to incompatible checkpoint
        m = _purity_k3(enc, all_healthy, cfg_clean)
        # Strip non-JSON-serialisable numpy bits before writing.
        m_clean = {
            "purity": float(m["purity"]),
            "nmi": float(m["nmi"]),
            "n_windows": int(m["n_windows"]),
        }
        rows[name] = m_clean
        print(f"{name:<28} {m_clean['purity']:>8.3f} {m_clean['nmi']:>8.3f} {m_clean['n_windows']:>8}")

    out_path = RESULTS / "v2_reeval_k3.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nResults written to {out_path}")
    return rows


if __name__ == "__main__":
    main()
