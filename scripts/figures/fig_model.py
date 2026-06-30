"""Model-driven figures (18, 21, 24 of the figure plan).

  18  t-SNE of the learned embeddings, baseline vs CMA, coloured by mode
  21  NLL score distributions, healthy vs anomaly cohorts, threshold marked
  24  synthetic mode-crossfade timeline with the AND-rule overlay

Figure 18 only forwards the frozen checkpoints of the canonical seed-42 run
(results/runs/20260615_112939__full_pipeline_b5_cma).  Figures 21/24 need
the learned V3 ``pma2`` window pool, which the run did not persist, so the
flow is RETRAINED from the frozen V2 encoder at seed 42 — the same
"seed-42 retrain reaching the headline density fit" protocol the Results
chapter quotes for its threshold-transfer table.  Expect ~20-25 min per
flow on CPU.

Run with:
  python -m scripts.figures.fig_model --part tsne
  python -m scripts.figures.fig_model --part anomaly          # 21 + 24, fusion flow
  python -m scripts.figures.fig_model --part anomaly --with-and  # + unimodal flows
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.figures import style
from scripts.figures.style import (
    ANOMALY,
    HEALTHY,
    INTERMEDIATE,
    LATE_FUSION,
    MODE_COLORS,
    REPO_ROOT,
    save,
)
from src.modeling.anomaly.v3_per_modality import (
    V3AcousticOnlyAdapter,
    V3VibrationOnlyAdapter,
)
from src.modeling.anomaly.v3_trainer import (
    make_transition_segment,
    precompute_paired,
    train_v3_cnf,
)
from src.modeling.context.cluster_metric import cluster_purity_and_nmi
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.context.v2_ssl import (
    _collate,
    _gather_paired_segments,
    _PairedGroupedBatchSampler,
    _PairedWindowedDataset,
)
from src.modeling.encoders import PerModalityEncoder
from src.modeling.orchestration.full_run import resolved_loader
from src.modeling.orchestration.stage_configs import v2_config

# Canonical seed-42 run of the five-seed set the Results tables are built on
# (the full_run of finalize_results_20260617_042101.json).
RUN_DIR = REPO_ROOT / "results" / "runs" / "20260615_112939__full_pipeline_b5_cma"


def _load_v1(modality: str, cfg) -> PerModalityEncoder:
    enc = PerModalityEncoder(
        modality=modality,
        feature_dim=cfg.feature_dim,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        acoustic_cnn_width_mult=cfg.acoustic_cnn_width_mult,
    )
    enc.load_state_dict(torch.load(RUN_DIR / "v1" / f"{modality}.pt", map_location="cpu"))
    return enc.eval()


# ─────────────────────────────────────────────────────────────────────────
# 18 — context t-SNE
# ─────────────────────────────────────────────────────────────────────────
def fig18_tsne() -> None:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from tqdm import tqdm

    cfg = v2_config(quick=False)
    v1_a = _load_v1("acoustic", cfg)
    v2 = V2FusionEncoder.from_checkpoint(RUN_DIR / "v2" / "encoder.pt", cfg)

    print("gathering all labelled D1+D2 windows (strict cohort) ...")
    
    # <-- Added tqdm to the loader resolution
    yaml_files = ["d1.yaml", "d2.yaml"]
    loaders = [resolved_loader(f) for f in tqdm(yaml_files, desc="Resolving loaders", unit="file")]
    
    segments = _gather_paired_segments(loaders, cfg)
    # Strict cohort: every labelled D1+D2 window (no held-out split), so the
    # panel NMI matches the headline strict-cohort number of Table tab:res_rq1
    # (fused c_t ~0.41) rather than the optimistic development/validation split.
    ds = _PairedWindowedDataset(segments, cfg)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_sampler=_PairedGroupedBatchSampler(ds, cfg.batch_size, shuffle=False, seed=0),
        collate_fn=_collate,
    )
    print(f"  {len(ds)} windows")

    v1_emb, cma_emb, ctx_emb, labels = [], [], [], []
    with torch.no_grad():
        # <-- tqdm on the batch extraction loop
        for batch in tqdm(loader, desc="Extracting embeddings", unit="batch"):
            ac, vib = batch["ac_feat"], batch["vib_feat"]
            ac_xyz, vib_xyz, ds_idx = batch["ac_xyz"], batch["vib_xyz"], batch["dataset_idx"]
            _, a1 = v1_a(ac, ac_xyz, ds_idx)
            v1_emb.append(a1.numpy())
            a_tok, a_sum = v2.acoustic(ac, ac_xyz, ds_idx)
            v_tok, _v_sum = v2.vibration(vib, vib_xyz, ds_idx)
            cma_emb.append(a_sum.numpy())
            _fa, _fv, c = v2.fuse_and_pool(a_tok, v_tok)
            ctx_emb.append(c.numpy())
            labels.extend(batch["mode_label"])

    labels = np.asarray(labels)
    keep = np.isin(labels, list(MODE_COLORS))
    sets = {
        "v1": np.concatenate(v1_emb)[keep],
        "cma": np.concatenate(cma_emb)[keep],
        "ctx": np.concatenate(ctx_emb)[keep],
    }
    lab = labels[keep]
    print(f"  {keep.sum()} labelled windows: "
          f"{dict(zip(*np.unique(lab, return_counts=True)))}")

    # <-- Added tqdm to the NMI calculation
    nmi = {
        k: cluster_purity_and_nmi(v, list(lab), n_clusters=3, seed=0)["nmi"]
        for k, v in tqdm(sets.items(), desc="Calculating NMI", unit="set", leave=False)
    }
    
    # <-- tqdm on the t-SNE generation
    proj = {
        k: TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(v)
        for k, v in tqdm(sets.items(), desc="Calculating t-SNE", unit="proj")
    }
    
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(sets["ctx"])

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    panels = [
        ("v1", f"(a) V1 acoustic encoder, by mode\nNMI = {nmi['v1']:.3f}", None),
        ("ctx", f"(b) fused context $c_t$, by mode\nNMI = {nmi['ctx']:.3f}", None),
        ("ctx", "(c) fused context $c_t$,\nlabel-free K-means clusters", km),
    ]
    
    # Using distinct colors to separate K-means clusters from the supervised ground-truth colors
    cluster_colors = np.array(["#9467bd", "#e377c2", "#bcbd22"])
    
    for ax, (key, title, clusters) in zip(axes, panels):
        P = proj[key]
        if clusters is None:
            for mode, color in MODE_COLORS.items():
                m = lab == mode
                ax.scatter(P[m, 0], P[m, 1], s=5, c=color, label=mode, alpha=0.7, lw=0)
        else:
            ax.scatter(P[:, 0], P[:, 1], s=5, c=cluster_colors[clusters], alpha=0.7, lw=0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        
    axes[0].legend(loc="lower left", fontsize=6.5, frameon=True, framealpha=0.8,
                   handletextpad=0.2, markerscale=2.2)
    fig.tight_layout()
    save(fig, "fig18_context_tsne")


# ─────────────────────────────────────────────────────────────────────────
# 21 + 24 — seed-42 retrain of the V3 flows + STREAMING-inference scoring
# (sliding_window_v3_inference with the trained pma2 pool — the calibrated
#  pipeline path; the saved flow.pt files do not persist the pool, and
#  mean-pool scoring of a pma2-trained flow is documented as miscalibrated)
# ─────────────────────────────────────────────────────────────────────────
def _slice_paired(seg, seconds: float, *, tail: bool):
    """Take the first/last ``seconds`` of a paired segment."""
    n_ac = int(round(seconds * seg.acoustic_fs))
    n_vib = int(round(seconds * seg.vibration_fs))
    sl = (lambda T, n: slice(max(0, T - n), T)) if tail else (lambda T, n: slice(0, n))
    a = seg.acoustic_features[..., sl(seg.acoustic_features.shape[-1], n_ac)]
    v = seg.vibration_features[..., sl(seg.vibration_features.shape[-1], n_vib)]
    return replace(seg, acoustic_features=a, vibration_features=v)


def fig21_fig24(with_and: bool) -> None:
    from src.modeling.anomaly.event_detection import sliding_window_v3_inference
    from src.modeling.orchestration.stage_configs import v3_config

    cfg = v2_config(quick=False)
    v3_cfg = v3_config(False)
    v2 = V2FusionEncoder.from_checkpoint(RUN_DIR / "v2" / "encoder.pt", cfg)

    loaders = {d: resolved_loader(f"{d}.yaml") for d in ("d1", "d2", "d3", "d4", "d5")}

    print("retraining the V3-fusion conditional flow (seed-42 retrain) ...")
    res = train_v3_cnf(v2, list(loaders.values()), v2_cfg=cfg, v3_cfg=v3_cfg)
    print(f"  val NLL = {res.val_nll[-1]:.2f}")

    # ── score anomaly cohorts with the calibrated streaming path ─────
    def cohort_sliding(did: str, stride_s: float) -> np.ndarray:
        scores: list[np.ndarray] = []
        for s in loaders[did].list_segments():
            if not s.is_anomaly:
                continue
            p = precompute_paired(s, cfg)
            if p is None:
                continue
            try:
                _t, sc, _cx = sliding_window_v3_inference(
                    v2, res.flow, p, v2_cfg=cfg, inference_stride_s=stride_s,
                    xt_pool=res.xt_pool,
                )
            except Exception as e:
                print(f"    {did}/{s.recording_id} skipped: {e}")
                continue
            scores.append(sc)
        return np.concatenate(scores) if scores else np.zeros(0)

    cohort_scores: dict[str, np.ndarray] = {}
    for did, stride in (("d2", 1.0), ("d3", 0.5), ("d4", 1.0)):
        cohort_scores[did] = cohort_sliding(did, stride)
        print(f"  cohort {did}: {len(cohort_scores[did])} windows scored")

    healthy_scores = np.asarray(res.val_scores)
    thr = res.thresholds
    cohort_alert = {}
    for did, sc in cohort_scores.items():
        cohort_alert[did] = float("nan")  # alert rates belong to the tables

    # — figure 21 —
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    lo = float(min(healthy_scores.min(), *(s.min() for s in cohort_scores.values())))
    hi = float(max(np.percentile(healthy_scores, 99.9),
                   *(np.percentile(s, 99.5) for s in cohort_scores.values())))
    bins = np.linspace(lo, hi, 90)
    ax.hist(healthy_scores, bins=bins, density=True, color=HEALTHY, alpha=0.65,
            label=f"healthy hold-out (n={len(healthy_scores)})")
    for did, color, lbl in [
        ("d4", ANOMALY, "D4 knock recordings (sparse)"),
        ("d2", "#e8a13c", "D2 anomaly recordings"),
        ("d3", "#7b3294", "D3 instrumented hit"),
    ]:
        if did in cohort_scores and cohort_scores[did].size:
            ax.hist(cohort_scores[did], bins=bins, density=True, histtype="step",
                    lw=1.6, color=color,
                    label=f"{lbl} (n={len(cohort_scores[did])})")
    for t in thr.p95:
        ax.axvline(t, color="0.25", lw=1.0, ls="--")
    ax.axvline(np.nan, color="0.25", lw=1.0, ls="--",
               label="per-cluster 95th-pct thresholds")
    ax.set_yscale("log")
    ax.set_xlabel(r"anomaly score $-\log p(x \mid c_t)$")
    ax.set_ylabel("density (log)")
    ax.set_title(
        "Conditional-flow score distributions (seed-42 retrain, streaming inference):\n"
        "healthy mass stays below the per-cluster thresholds, anomaly cohorts cross them",
        fontsize=9,
    )
    ax.legend(loc="upper center", fontsize=6.8, frameon=False)
    fig.tight_layout()
    save(fig, "fig21_nll_distributions")

    # ── figure 24: synthetic mode crossfade ───────────────────────────
    print("building the synthetic mode-crossfade segment ...")
    d1_segs = _gather_paired_segments([loaders["d1"]], cfg)
    pump = next(s for s in d1_segs if s.mode_label == "Pump")
    turb = next(s for s in d1_segs if s.mode_label == "Turbine")
    span_s = 30.0
    trans = make_transition_segment(
        _slice_paired(turb, span_s, tail=True),
        _slice_paired(pump, span_s, tail=False),
        crossfade_seconds=2.0,
        label="TU->PU crossfade",
    )
    stride_tl = 0.5
    t, sc_f, ctx_f = sliding_window_v3_inference(
        v2, res.flow, trans, v2_cfg=cfg, inference_stride_s=stride_tl,
        xt_pool=res.xt_pool,
    )
    alerts_f, clusters_f = thr.alert(ctx_f, sc_f, percentile=v3_cfg.threshold_percentile)
    thr_per_win = thr.p95[clusters_f]

    and_alerts = None
    if with_and:
        print("training the two unimodal flows for the AND overlay ...")
        v1_a, v1_v = _load_v1("acoustic", cfg), _load_v1("vibration", cfg)
        enc_a, enc_v = V3AcousticOnlyAdapter(v1_a), V3VibrationOnlyAdapter(v1_v)
        res_a = train_v3_cnf(enc_a, list(loaders.values()), v2_cfg=cfg, v3_cfg=v3_cfg)
        res_v = train_v3_cnf(enc_v, list(loaders.values()), v2_cfg=cfg, v3_cfg=v3_cfg)
        _ta, sc_a, ctx_a = sliding_window_v3_inference(
            enc_a, res_a.flow, trans, v2_cfg=cfg, inference_stride_s=stride_tl,
            xt_pool=res_a.xt_pool,
        )
        _tv, sc_v, ctx_v = sliding_window_v3_inference(
            enc_v, res_v.flow, trans, v2_cfg=cfg, inference_stride_s=stride_tl,
            xt_pool=res_v.xt_pool,
        )
        al_a, _ = res_a.thresholds.alert(ctx_a, sc_a, percentile=v3_cfg.threshold_percentile)
        al_v, _ = res_v.thresholds.alert(ctx_v, sc_v, percentile=v3_cfg.threshold_percentile)
        n = min(len(al_a), len(al_v))
        and_alerts = np.asarray(al_a[:n], bool) & np.asarray(al_v[:n], bool)

    W = 3.0  # window length: any window starting within W of the fade overlaps it
    fade_start, fade_end = span_s - 2.0, span_s

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.ticklabel_format(useOffset=False, axis="y")
    ax.axvspan(fade_start, fade_end, color="0.90", zorder=0)
    ax.plot(t, sc_f, color=INTERMEDIATE, lw=1.3, label="conditional flow score")
    ax.plot(t, thr_per_win, color="0.25", lw=1.0, ls="--", drawstyle="steps-mid",
            label="per-cluster threshold")
    am = np.asarray(alerts_f, bool)
    ax.scatter(t[am], sc_f[am], s=16, color=ANOMALY, zorder=5, label="flow alert (false positive)")
    if and_alerts is not None:
        y_and = np.full(len(and_alerts), ax.get_ylim()[0])
        ax.scatter(t[: len(and_alerts)][and_alerts], y_and[and_alerts], marker="s",
                   s=18, color=LATE_FUSION, zorder=5, label="AND-rule alert")
        fpr_and = float(np.mean(and_alerts[(t[:len(and_alerts)] >= fade_start - W)
                                           & (t[:len(and_alerts)] <= fade_end + W)]))
    fpr_fade = float(np.mean(am[(t >= fade_start - W) & (t <= fade_end + W)]))
    ax.text(fade_start - 1.0, ax.get_ylim()[1], "Turbine (healthy)", ha="right",
            va="top", fontsize=8, color=MODE_COLORS["Turbine"])
    ax.text(fade_end + 1.0, ax.get_ylim()[1], "Pump (healthy)", ha="left",
            va="top", fontsize=8, color=MODE_COLORS["Pump"])
    txt = f"crossfade FPR (flow) = {fpr_fade:.2f}"
    if and_alerts is not None:
        txt += f"\ncrossfade FPR (AND) = {fpr_and:.2f}"
    ax.text(0.985, 0.04, txt, transform=ax.transAxes, ha="right", fontsize=7.5,
            color="0.2", va="bottom")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$-\log p(x \mid c_t)$")
    ax.set_title(
        "Synthetic TU$\\to$PU crossfade (healthy throughout): the conditional flow\n"
        "fires across the transition; the late-fusion AND rule suppresses it",
        fontsize=9,
    )
    ax.legend(loc="center left", fontsize=6.6, frameon=False)
    fig.tight_layout()
    save(fig, "fig24_crossfade_timeline")


def cfg_window_fallback(cfg) -> float:
    ws = getattr(cfg, "window_scales_seconds", None)
    if isinstance(ws, dict):
        v = ws.get("d1") or next(iter(ws.values()))
        return float(v[0] if isinstance(v, (tuple, list)) else v)
    if isinstance(ws, (tuple, list)) and ws:
        return float(ws[0])
    return 3.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("tsne", "anomaly", "all"), default="all")
    parser.add_argument("--with-and", action="store_true", default=False)
    args = parser.parse_args()
    style.apply_style()
    torch.manual_seed(42)
    if args.part in ("tsne", "all"):
        fig18_tsne()
    if args.part in ("anomaly", "all"):
        fig21_fig24(with_and=args.with_and)


if __name__ == "__main__":
    main()
