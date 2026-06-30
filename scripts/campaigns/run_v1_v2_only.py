"""Phase-B helper — re-train only V1 acoustic + V1 vibration + V2 fusion (+ A1
+ modality probe) under a named intervention, skipping V0/V3/V4/V5.

Each invocation is ~50 minutes of CPU vs ~6 hours for the full pipeline.
Per the approved plan, this is the bisection vehicle for Phase B: each
``--variant`` applies exactly one intervention so the modality-balance probe
can attribute the change unambiguously.

Variants:
  * ``baseline``       — current defaults (post-2026-05-15-fix); reproduces
                         the headline RQ1 numbers without re-running V3-V5.
  * ``b1_sym_dropout`` — symmetric modality dropout
                         (``acoustic_dropout_p = vibration_dropout_p = 0.25``)
  * ``b2_token_ln``    — pre-fusion LayerNorm on each modality's tokens
                         (requires the matching V2FusionEncoder flag — see
                         `tok_norm_before_fusion` in `v2_fusion.py`).
  * ``b3_xmodal_lmm``  — cross-modal LMM: masked positions on modality M
                         are reconstructed from ¬M K/V only.
  * ``b4_resid_alpha`` — learnable residual scaling α (init 0.5) on each
                         direction of BidirectionalCrossAttention.
  * ``b5_cma``         — CMA loss on, ``cma_weight = 0.5``.

Output: ``results/runs/<timestamp>__v1v2_only_<variant>/`` with
``metrics.json`` containing the same V2/V2_a1/modality_probe keys
``full_run.py`` writes, so cross-run comparisons are direct.

Run::

    python -m scripts.campaigns.run_v1_v2_only --variant b5_cma
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

import torch

from src.modeling.context.modality_probe import run_modality_balance_probe

# Re-use the canonical config builders + training helpers from the orchestrator
# so the V1+V2-only path is byte-equivalent except for the named intervention.
from src.modeling.context.v1_ssl import V1SSLConfig, train_v1_per_modality
from src.modeling.context.v2_ssl import V2SSLConfig, train_v2_fusion
from src.modeling.orchestration.full_run import (
    resolved_loader,
    v1_config,
    v2_config,
)

REPO = Path(__file__).resolve().parents[2]


def _apply_variant(
    name: str, v1_cfg: V1SSLConfig, v2_cfg: V2SSLConfig
) -> tuple[V1SSLConfig, V2SSLConfig]:
    """Return (v1_cfg, v2_cfg) with the variant's config knobs flipped.

    Each variant alters at most one mechanism so the cross-variant
    comparison is clean.  Anything that requires *code* changes (B2's
    LayerNorm, B3's cross-modal LMM mask, B4's α scalar) is gated on a
    new V2SSLConfig flag the variant flips; the implementation site for
    each flag is documented in the plan.
    """
    from dataclasses import replace

    if name == "baseline":
        return v1_cfg, v2_cfg
    if name == "b1_sym_dropout":
        return v1_cfg, replace(
            v2_cfg,
            acoustic_dropout_p=0.25,
            vibration_dropout_p=0.25,
            modality_dropout_p=0.0,
        )
    if name == "b5_cma":
        return v1_cfg, replace(v2_cfg, cma_weight=0.5, cma_temperature=0.1)
    if name == "b_dual_pma":
        # Phase A diagnosed the joint PMA pool as the bottleneck: cross-
        # attention puts vibration info into `fused_a` (LMM acoustic
        # reconstruction degrades by Δ=−0.22 when vibration K/V is zeroed),
        # but `c_t = PMA([fused_a; fused_v])` then averages along directions
        # that drop the vibration contribution.  `dual_pma` replaces the
        # joint pool with one PMA per modality + concat-MLP, so vibration's
        # axis is preserved into c_t.  Already a config knob; previously
        # only run with the collapsed pre-fix encoder.  Kept under the
        # `b_*` namespace because it tests a mechanism orthogonal to B1-B5.
        return v1_cfg, replace(v2_cfg, context_mode="dual_pma")
    if name == "b1_plus_dual_pma":
        # B1 (symmetric dropout) freed vibration's training signal but the
        # joint PMA pool still discarded it.  This combines the freed
        # signal with the pool fix — strongest single iteration given the
        # Phase-A + B1 evidence.
        return v1_cfg, replace(
            v2_cfg,
            acoustic_dropout_p=0.25,
            vibration_dropout_p=0.25,
            modality_dropout_p=0.0,
            context_mode="dual_pma",
        )
    # B2/B3/B4 require code changes beyond config flips (a new V2SSLConfig
    # field + the encoder/loss change it triggers).  They are wired
    # just-in-time after Phase-A's verdict so we don't carry dead config
    # surface area.  Each one's implementation site is documented in the
    # approved plan.
    if name in ("b2_token_ln", "b3_xmodal_lmm", "b4_resid_alpha"):
        raise NotImplementedError(
            f"variant {name!r} requires code wiring; see the approved plan's "
            "Phase-B section for the implementation site and re-run after "
            "the relevant edit is in place."
        )
    raise ValueError(f"unknown variant {name!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        required=True,
        choices=(
            "baseline",
            "b1_sym_dropout",
            "b2_token_ln",
            "b3_xmodal_lmm",
            "b4_resid_alpha",
            "b5_cma",
            "b_dual_pma",
            "b1_plus_dual_pma",
        ),
    )
    p.add_argument("--quick", action="store_true", help="3-epoch smoke run, not 12")
    args = p.parse_args()

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / "runs" / f"{timestamp}__v1v2_only_{args.variant}"
    (out_dir / "v1").mkdir(parents=True, exist_ok=True)
    (out_dir / "v2").mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        # Windows cp1252 console can't encode some Unicode (delta, partial,
        # arrows).  Write the rich version to the file; sanitise for stdout.
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        with (out_dir / "run_log.txt").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"REPO = {REPO}")
    log(f"variant = {args.variant}, quick = {args.quick}")
    log(f"out_dir = {out_dir}")

    log("Loading D1, D2, D3, D4 SSL loaders ...")
    SSL_LOADERS = [resolved_loader(f"{d}.yaml") for d in ("d1", "d2", "d3", "d4")]

    v1_cfg = v1_config(args.quick)
    v2_cfg = v2_config(args.quick)
    v1_cfg, v2_cfg = _apply_variant(args.variant, v1_cfg, v2_cfg)
    log(f"V1 config: epochs={v1_cfg.epochs}, n_mels={v1_cfg.n_mels}, "
        f"use_cwt={v1_cfg.use_cwt}, standardize_acoustic={v1_cfg.standardize_acoustic}")
    log(
        f"V2 config: epochs={v2_cfg.epochs}, cma_weight={v2_cfg.cma_weight}, "
        f"context_mode={v2_cfg.context_mode}, "
        f"acoustic_dropout_p={v2_cfg.acoustic_dropout_p}, "
        f"vibration_dropout_p={v2_cfg.vibration_dropout_p}"
    )

    metrics: dict = {"variant": args.variant, "stages": {}}

    # -- V1 acoustic --------------------------------------------------------
    log("V1 acoustic — training on D1+D2+D3+D4 healthy ...")
    t0 = time.time()
    v1_acoustic = train_v1_per_modality(SSL_LOADERS, modality="acoustic", cfg=v1_cfg)
    dt = time.time() - t0
    log(
        f"V1 acoustic done in {dt:.0f}s — "
        f"NMI={v1_acoustic.sanity_gate.get('nmi', 0):.3f} "
        f"ARI={v1_acoustic.sanity_gate.get('ari', 0):.3f} "
        f"purity={v1_acoustic.sanity_gate.get('purity', 0):.3f}"
    )
    torch.save(v1_acoustic.encoder.state_dict(), out_dir / "v1" / "acoustic.pt")
    metrics["stages"]["v1_acoustic"] = {
        "epochs": v1_cfg.epochs,
        "train_loss_final": v1_acoustic.train_loss_history[-1],
        "val_loss_final": v1_acoustic.val_loss_history[-1],
        "sanity_nmi": v1_acoustic.sanity_gate.get("nmi", 0.0),
        "sanity_ari": v1_acoustic.sanity_gate.get("ari", 0.0),
        "sanity_purity": v1_acoustic.sanity_gate.get("purity", 0.0),
        "sanity_n_windows": v1_acoustic.sanity_gate.get("n_windows", 0),
    }

    # -- V1 vibration -------------------------------------------------------
    log("V1 vibration — training on D1+D2+D3+D4 healthy ...")
    t0 = time.time()
    v1_vibration = train_v1_per_modality(SSL_LOADERS, modality="vibration", cfg=v1_cfg)
    dt = time.time() - t0
    log(
        f"V1 vibration done in {dt:.0f}s — "
        f"NMI={v1_vibration.sanity_gate.get('nmi', 0):.3f} "
        f"ARI={v1_vibration.sanity_gate.get('ari', 0):.3f} "
        f"purity={v1_vibration.sanity_gate.get('purity', 0):.3f}"
    )
    torch.save(v1_vibration.encoder.state_dict(), out_dir / "v1" / "vibration.pt")
    metrics["stages"]["v1_vibration"] = {
        "epochs": v1_cfg.epochs,
        "train_loss_final": v1_vibration.train_loss_history[-1],
        "val_loss_final": v1_vibration.val_loss_history[-1],
        "sanity_nmi": v1_vibration.sanity_gate.get("nmi", 0.0),
        "sanity_ari": v1_vibration.sanity_gate.get("ari", 0.0),
        "sanity_purity": v1_vibration.sanity_gate.get("purity", 0.0),
    }

    # -- V2 -----------------------------------------------------------------
    log("V2 — training fusion (inherits V1 weights) ...")
    t0 = time.time()
    v2 = train_v2_fusion(
        SSL_LOADERS,
        cfg=v2_cfg,
        v1_acoustic_state_dict=v1_acoustic.encoder.state_dict(),
        v1_vibration_state_dict=v1_vibration.encoder.state_dict(),
    )
    dt = time.time() - t0
    log(
        f"V2 done in {dt:.0f}s — "
        f"NMI={v2.rq1.get('nmi', 0):.3f} ARI={v2.rq1.get('ari', 0):.3f} "
        f"purity={v2.rq1.get('purity', 0):.3f}"
    )
    torch.save(v2.encoder.state_dict(), out_dir / "v2" / "encoder.pt")
    torch.save(v2.projection.state_dict(), out_dir / "v2" / "projection.pt")
    metrics["stages"]["v2"] = {
        "epochs": v2_cfg.epochs,
        "train_loss_final": v2.train_loss_history[-1],
        "val_loss_final": v2.val_loss_history[-1],
        "train_simclr_final": v2.train_simclr_history[-1],
        "train_lmm_final": v2.train_lmm_history[-1],
        "rq1_nmi": v2.rq1.get("nmi", 0.0),
        "rq1_ari": v2.rq1.get("ari", 0.0),
        "rq1_purity": v2.rq1.get("purity", 0.0),
    }

    # -- V2 A1 (drop_vibration) --------------------------------------------
    log("V2 — A1 ablation (drop_vibration=True) ...")
    from dataclasses import replace
    a1_cfg = replace(v2_cfg, drop_vibration=True)
    t0 = time.time()
    v2_a1 = train_v2_fusion(
        SSL_LOADERS,
        cfg=a1_cfg,
        v1_acoustic_state_dict=v1_acoustic.encoder.state_dict(),
        v1_vibration_state_dict=v1_vibration.encoder.state_dict(),
    )
    dt = time.time() - t0
    log(
        f"V2 A1 done in {dt:.0f}s — "
        f"NMI={v2_a1.rq1.get('nmi', 0):.3f}"
    )
    metrics["stages"]["v2_a1_drop_vibration"] = {
        "rq1_nmi": v2_a1.rq1.get("nmi", 0.0),
        "rq1_ari": v2_a1.rq1.get("ari", 0.0),
        "rq1_purity": v2_a1.rq1.get("purity", 0.0),
    }

    # -- modality probe (the headline metric for Phase B comparisons) ------
    log("V2 modality-balance probe ...")
    try:
        from src.modeling.context.v2_ssl import _gather_labeled_segments

        labeled_segs = _gather_labeled_segments(SSL_LOADERS, v2_cfg)
        probe = run_modality_balance_probe(
            v2.encoder, labeled_segs, v2_cfg=v2_cfg, n_clusters=3, seed=v2_cfg.seed,
        )
        log(
            f"  both NMI={probe.both.get('nmi', 0):.3f}, "
            f"acoustic-only NMI={probe.acoustic_only.get('nmi', 0):.3f}, "
            f"vibration-only NMI={probe.vibration_only.get('nmi', 0):.3f}"
        )
        metrics["stages"]["v2_modality_probe"] = {
            "both": {k: v for k, v in probe.both.items() if k not in ("confusion", "cluster_idx")},
            "acoustic_only": {k: v for k, v in probe.acoustic_only.items() if k not in ("confusion", "cluster_idx")},
            "vibration_only": {k: v for k, v in probe.vibration_only.items() if k not in ("confusion", "cluster_idx")},
            "n_segments": len(probe.healthy_segments_used),
        }
    except Exception as e:
        log(f"modality probe skipped: {type(e).__name__}: {e}")
        metrics["stages"]["v2_modality_probe"] = {"skipped": str(e)}

    # -- summary -----------------------------------------------------------
    both = metrics["stages"].get("v2_modality_probe", {}).get("both", {}).get("nmi", 0.0)
    ac_only = metrics["stages"].get("v2_modality_probe", {}).get("acoustic_only", {}).get("nmi", 0.0)
    delta_nmi = both - ac_only
    log("=" * 60)
    log(f"VARIANT {args.variant!r} HEADLINE: "
        f"probe both={both:.3f}, acoustic_only={ac_only:.3f}, Δ={delta_nmi:+.3f}")
    log("  (Phase-B success bar: Δ ≥ +0.030; V1-acoustic ceiling preserved)")

    metrics["headline"] = {
        "probe_both_nmi": both,
        "probe_acoustic_only_nmi": ac_only,
        "delta_nmi_both_minus_acoustic": delta_nmi,
        "v1_acoustic_sanity_nmi": metrics["stages"]["v1_acoustic"]["sanity_nmi"],
        "v1_vibration_sanity_nmi": metrics["stages"]["v1_vibration"]["sanity_nmi"],
        "v2_a1_sanity_nmi": metrics["stages"]["v2_a1_drop_vibration"]["rq1_nmi"],
        "phase_b_pass": bool(delta_nmi >= 0.030),
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log(f"Wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
