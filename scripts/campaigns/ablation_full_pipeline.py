"""Ablation runner — full-pipeline cell sweep with overfitting-first axes.

One cell = one parameter combination from the ablation grid.  All cells run
against the baseline_v2 infrastructure (early-stop + weight_decay=1e-4 +
head/CNF dropout 0.1 — committed in `full_run.py`'s per-stage builders).

Phases map to runtime flags, not separate cell IDs:
  - Phase 2 / 3 (V1+V2-only triage, ~30 min GPU each):
        --cell <p2_*|p3_*> --skip-v3 --skip-v4
  - Phase 4 (full-pipeline promotion of top-N cells, ~1 h GPU each):
        --cell <winning p2_* or p3_* id>
  - Phase 5 (multi-seed verdict, ~1 h GPU each):
        --cell <winning id> --seed 1337
        --cell <winning id> --seed 2024

Cell IDs (the cartesian axes):
  baseline_v2 — no overrides; reproduces the orchestrator defaults.

  Phase 2 (aug strength × vibration_dropout):
    p2_a0_v3  p2_a0_v5  p2_a0_v7   # aug=mid (9,0.2,12,16)
    p2_a1_v3  p2_a1_v5  p2_a1_v7   # aug=strong (12,0.3,16,20)
    p2_a2_v3  p2_a2_v5  p2_a2_v7   # aug=very-strong (15,0.4,20,24)

  Phase 3 (mixup × embed_dim) — these REQUIRE a `--base-cell` flag set
  to the Phase 2 winner so the mixup/embed_dim variation sits on top of
  the chosen aug × vibration_dropout combination:
    p3_m0_e32  p3_m0_e64  p3_m0_e128   # mixup=0.0
    p3_m2_e32  p3_m2_e64  p3_m2_e128   # mixup=0.2
    p3_m4_e32  p3_m4_e64  p3_m4_e128   # mixup=0.4

  Phase 7a (temperature × cma_weight) — also requires --base-cell <p2_*>:
    p7a_t0_c0  p7a_t0_c5  p7a_t0_c10   # temperature=0.05
    p7a_t1_c0  p7a_t1_c5  p7a_t1_c10   # temperature=0.10 (baseline)
    p7a_t2_c0  p7a_t2_c5  p7a_t2_c10   # temperature=0.20

  Phase 7b (lmm_mask_p × lmm_weight) — also requires --base-cell <p2_*>:
    p7b_lm3_lw1  p7b_lm3_lw2  p7b_lm3_lw3   # lmm_mask=0.3 (baseline)
    p7b_lm5_lw1  p7b_lm5_lw2  p7b_lm5_lw3   # lmm_mask=0.5
    p7b_lm7_lw1  p7b_lm7_lw2  p7b_lm7_lw3   # lmm_mask=0.7

Output: ``results/runs/<ts>__ablation_<cell>_s<seed>[/{v3,v4}_skipped]/``
with ``metrics.json`` + ``cell_config.json`` documenting the overrides
applied.  Use ``scripts/campaigns/analyze_ablation.py`` to aggregate cells.

Run::

    python -m scripts.campaigns.ablation_full_pipeline --cell p2_a1_v5 --skip-v3 --skip-v4
    python -m scripts.campaigns.ablation_full_pipeline --cell p3_m2_e64 --base-cell p2_a1_v5 --skip-v3 --skip-v4
    python -m scripts.campaigns.ablation_full_pipeline --cell p2_a1_v5             # full pipeline
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.modeling.context.modality_probe import run_modality_balance_probe
from src.modeling.context.v1_ssl import V1SSLConfig, train_v1_per_modality
from src.modeling.context.v2_ssl import V2SSLConfig, train_v2_fusion
from src.modeling.orchestration.full_run import (
    resolved_loader,
    v1_config,
    v2_config,
    v3_config,
    v4_config,
)

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Per-cell config mutations
# ---------------------------------------------------------------------------


# Augmentation-strength bundles (tuple = gain_jitter_db, channel_dropout_p,
# spec_augment_freq_mask, spec_augment_time_mask).  Indexed by the `a{N}`
# suffix in cell IDs.  See plan §Phase 2.
_AUG_LEVELS: dict[str, tuple[float, float, int, int]] = {
    "a0": (9.0, 0.2, 12, 16),   # mid (current orchestrator default)
    "a1": (12.0, 0.3, 16, 20),  # strong
    "a2": (15.0, 0.4, 20, 24),  # very-strong
}

_VIB_DROPOUT_LEVELS: dict[str, float] = {
    "v3": 0.3,
    "v5": 0.5,  # current orchestrator default
    "v7": 0.7,
}

_MIXUP_LEVELS: dict[str, float] = {
    "m0": 0.0,
    "m2": 0.2,
    "m4": 0.4,
}

_EMBED_DIM_LEVELS: dict[str, int] = {
    "e32": 32,
    "e64": 64,   # current orchestrator default
    "e128": 128,
}

# Phase 7a — SSL hyperparameter neighborhood around b5_cma.  The NT-Xent
# temperature is set on both V1SSLConfig and V2SSLConfig (V1→V2 weight
# transfer does not enforce parity here, but a mismatch would change the
# contrastive geometry mid-pipeline; sweeping them jointly keeps the
# interpretation clean).  CMA weight applies only to V2 (the cross-modal
# alignment loss lives there).
_TEMPERATURE_LEVELS: dict[str, float] = {
    "t0": 0.05,
    "t1": 0.10,  # current orchestrator default
    "t2": 0.20,
}

_CMA_WEIGHT_LEVELS: dict[str, float] = {
    "c0": 0.0,
    "c5": 0.5,   # b5_cma baseline (current orchestrator default)
    "c10": 1.0,
}

# Phase 7b — Latent Masked Modeling diagnostic.  The audit flagged LMM
# training loss collapsed to 0.012 under (0.3, 1.0) — this sweep diagnoses
# whether LMM is under-pressured (higher mask_p + weight rescues it) or
# structurally broken (no setting rescues it).
_LMM_MASK_LEVELS: dict[str, float] = {
    "lm3": 0.3,  # current orchestrator default
    "lm5": 0.5,
    "lm7": 0.7,
}

_LMM_WEIGHT_LEVELS: dict[str, float] = {
    "lw1": 1.0,  # current orchestrator default
    "lw2": 2.0,
    "lw3": 3.0,
}

# Acoustic-improvement axes (vibration is a settled dead-end for fusion, so the
# representation lever is the acoustic pathway).  These two knobs were never
# swept by the breadth campaign:
#   * acoustic_cnn_width_mult — R1a tested 2× and reverted it because the wider
#     CNN over-fit; now that early-stop + weight-decay + dropout control
#     overfitting, a wider acoustic backbone is worth re-testing.
#   * cwt_n_scales — CWT frequency resolution; finer scales sharpen the
#     100/117 Hz ROW II tone-pair separation (architecture.py CWT docstring).
# Both must match across V1 and V2 (V1→V2 weight transfer enforces parity).
_ACOUSTIC_WIDTH_LEVELS: dict[str, int] = {"w1": 1, "w2": 2}
_CWT_SCALE_LEVELS: dict[str, int] = {"cwt16": 16, "cwt32": 32, "cwt64": 64}


def _apply_p2_cell(
    cell_id: str,
    v1_cfg: V1SSLConfig,
    v2_cfg: V2SSLConfig,
) -> tuple[V1SSLConfig, V2SSLConfig]:
    """Phase 2 cell: ``p2_a{N}_v{M}`` → aug strength × vibration_dropout."""
    parts = cell_id.split("_")
    if len(parts) != 3 or parts[0] != "p2":
        raise ValueError(f"malformed p2 cell id: {cell_id!r}")
    aug_key, vib_key = parts[1], parts[2]
    if aug_key not in _AUG_LEVELS or vib_key not in _VIB_DROPOUT_LEVELS:
        raise ValueError(f"unknown axis level in {cell_id!r}")
    g, c, fm, tm = _AUG_LEVELS[aug_key]
    vib_drop = _VIB_DROPOUT_LEVELS[vib_key]
    v1_cfg = replace(
        v1_cfg,
        gain_jitter_db=g, channel_dropout_p=c,
        spec_augment_freq_mask=fm, spec_augment_time_mask=tm,
    )
    v2_cfg = replace(
        v2_cfg,
        gain_jitter_db=g, channel_dropout_p=c,
        spec_augment_freq_mask=fm, spec_augment_time_mask=tm,
        vibration_dropout_p=vib_drop,
    )
    return v1_cfg, v2_cfg


def _apply_p3_cell(
    cell_id: str,
    v1_cfg: V1SSLConfig,
    v2_cfg: V2SSLConfig,
) -> tuple[V1SSLConfig, V2SSLConfig]:
    """Phase 3 cell: ``p3_m{N}_e{D}`` → mixup_alpha × embed_dim.

    Caller is responsible for first applying the Phase 2 base via
    `_apply_p2_cell(base_cell, ...)` — this function only overlays the
    Phase 3 axes on top.  V1=V2 weight transfer enforces matched
    `embed_dim` (`feature_dim` tracks it), so we tie the two.
    """
    parts = cell_id.split("_")
    if len(parts) != 3 or parts[0] != "p3":
        raise ValueError(f"malformed p3 cell id: {cell_id!r}")
    mix_key, emb_key = parts[1], parts[2]
    if mix_key not in _MIXUP_LEVELS or emb_key not in _EMBED_DIM_LEVELS:
        raise ValueError(f"unknown axis level in {cell_id!r}")
    alpha = _MIXUP_LEVELS[mix_key]
    edim = _EMBED_DIM_LEVELS[emb_key]
    v1_cfg = replace(v1_cfg, mixup_alpha=alpha, feature_dim=edim, embed_dim=edim)
    v2_cfg = replace(v2_cfg, mixup_alpha=alpha, feature_dim=edim, embed_dim=edim)
    return v1_cfg, v2_cfg


def _apply_p7a_cell(
    cell_id: str,
    v1_cfg: V1SSLConfig,
    v2_cfg: V2SSLConfig,
) -> tuple[V1SSLConfig, V2SSLConfig]:
    """Phase 7a cell: ``p7a_t{N}_c{M}`` → temperature × cma_weight."""
    parts = cell_id.split("_")
    if len(parts) != 3 or parts[0] != "p7a":
        raise ValueError(f"malformed p7a cell id: {cell_id!r}")
    t_key, c_key = parts[1], parts[2]
    if t_key not in _TEMPERATURE_LEVELS or c_key not in _CMA_WEIGHT_LEVELS:
        raise ValueError(f"unknown axis level in {cell_id!r}")
    temp = _TEMPERATURE_LEVELS[t_key]
    cma = _CMA_WEIGHT_LEVELS[c_key]
    # Temperature applied to both V1 and V2 (NT-Xent runs in both stages).
    # CMA weight is V2-only (cross-modal alignment loss).
    v1_cfg = replace(v1_cfg, temperature=temp)
    v2_cfg = replace(v2_cfg, temperature=temp, cma_weight=cma)
    return v1_cfg, v2_cfg


def _apply_p7b_cell(
    cell_id: str,
    v1_cfg: V1SSLConfig,
    v2_cfg: V2SSLConfig,
) -> tuple[V1SSLConfig, V2SSLConfig]:
    """Phase 7b cell: ``p7b_lm{N}_lw{M}`` → lmm_mask_p × lmm_weight (V2 only)."""
    parts = cell_id.split("_")
    if len(parts) != 3 or parts[0] != "p7b":
        raise ValueError(f"malformed p7b cell id: {cell_id!r}")
    m_key, w_key = parts[1], parts[2]
    if m_key not in _LMM_MASK_LEVELS or w_key not in _LMM_WEIGHT_LEVELS:
        raise ValueError(f"unknown axis level in {cell_id!r}")
    v2_cfg = replace(
        v2_cfg,
        lmm_mask_p=_LMM_MASK_LEVELS[m_key],
        lmm_weight=_LMM_WEIGHT_LEVELS[w_key],
    )
    return v1_cfg, v2_cfg


def _apply_pa_cell(
    cell_id: str,
    v1_cfg: V1SSLConfig,
    v2_cfg: V2SSLConfig,
) -> tuple[V1SSLConfig, V2SSLConfig]:
    """Acoustic cell: ``pa_w{N}_cwt{M}`` → acoustic_cnn_width_mult × cwt_n_scales.

    Standalone (no base cell): builds on the orchestrator defaults, which are
    the post-baseline_v2 settings (early-stop + wd=1e-4 + dropout).  Width +
    CWT scales are set on both V1 and V2 so the V1→V2 weight transfer matches.
    """
    parts = cell_id.split("_")
    if len(parts) != 3 or parts[0] != "pa":
        raise ValueError(f"malformed pa cell id: {cell_id!r}")
    w_key, cwt_key = parts[1], parts[2]
    if w_key not in _ACOUSTIC_WIDTH_LEVELS or cwt_key not in _CWT_SCALE_LEVELS:
        raise ValueError(f"unknown axis level in {cell_id!r}")
    width = _ACOUSTIC_WIDTH_LEVELS[w_key]
    cwt = _CWT_SCALE_LEVELS[cwt_key]
    v1_cfg = replace(v1_cfg, acoustic_cnn_width_mult=width, cwt_n_scales=cwt, use_cwt=True)
    v2_cfg = replace(v2_cfg, acoustic_cnn_width_mult=width, cwt_n_scales=cwt, use_cwt=True)
    return v1_cfg, v2_cfg


def apply_cell(
    cell_id: str,
    base_cell: str | None,
    v1_cfg: V1SSLConfig,
    v2_cfg: V2SSLConfig,
    v3_cfg,
    v4_cfg,
):
    """Resolve a cell id (+ optional base) into mutated configs."""
    if cell_id == "baseline_v2":
        return v1_cfg, v2_cfg, v3_cfg, v4_cfg
    if cell_id.startswith("p2_"):
        v1_cfg, v2_cfg = _apply_p2_cell(cell_id, v1_cfg, v2_cfg)
        return v1_cfg, v2_cfg, v3_cfg, v4_cfg
    if cell_id.startswith("pa_"):
        v1_cfg, v2_cfg = _apply_pa_cell(cell_id, v1_cfg, v2_cfg)
        return v1_cfg, v2_cfg, v3_cfg, v4_cfg
    # Phase 3 / 7a / 7b all require a Phase 2 base to overlay.  The base cell
    # supplies the aug × vibration_dropout setting that won Phase 2 selection.
    if cell_id.startswith(("p3_", "p7a_", "p7b_")):
        if base_cell is None or not base_cell.startswith("p2_"):
            raise ValueError(
                f"{cell_id!r} requires --base-cell <p2_*> to overlay "
                "the Phase 2 winner's aug × vibration_dropout choice."
            )
        v1_cfg, v2_cfg = _apply_p2_cell(base_cell, v1_cfg, v2_cfg)
        if cell_id.startswith("p3_"):
            v1_cfg, v2_cfg = _apply_p3_cell(cell_id, v1_cfg, v2_cfg)
        elif cell_id.startswith("p7a_"):
            v1_cfg, v2_cfg = _apply_p7a_cell(cell_id, v1_cfg, v2_cfg)
        elif cell_id.startswith("p7b_"):
            v1_cfg, v2_cfg = _apply_p7b_cell(cell_id, v1_cfg, v2_cfg)
        return v1_cfg, v2_cfg, v3_cfg, v4_cfg
    raise ValueError(f"unknown cell id {cell_id!r}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _make_logger(out_dir: Path):
    log_path = out_dir / "run_log.txt"

    def log(msg: str) -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(line + "\n")

    return log


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True, help="Cell ID — see module docstring.")
    p.add_argument(
        "--base-cell", default=None,
        help="For p3_* cells: the Phase 2 cell whose aug × vib_dropout to overlay.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true",
                   help="3-epoch smoke run. NOT recommended for regularization cells.")
    p.add_argument("--skip-v3", action="store_true",
                   help="Phase 2 / 3 budget control: skip V3 (and V4) stages.")
    p.add_argument("--skip-v4", action="store_true",
                   help="Phase 2 / 3 / 4 budget control: skip V4 stage.")
    args = p.parse_args()

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix_bits = [f"s{args.seed}"]
    if args.skip_v3:
        suffix_bits.append("v3skip")
    if args.skip_v4 and not args.skip_v3:
        suffix_bits.append("v4skip")
    if args.quick:
        suffix_bits.append("quick")
    suffix = "_".join(suffix_bits)
    out_dir = REPO / "results" / "runs" / f"{timestamp}__ablation_{args.cell}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v1").mkdir(exist_ok=True)
    (out_dir / "v2").mkdir(exist_ok=True)
    log = _make_logger(out_dir)
    log(f"cell={args.cell} base={args.base_cell} seed={args.seed} "
        f"skip_v3={args.skip_v3} skip_v4={args.skip_v4} quick={args.quick}")

    # Build base configs from the canonical orchestrator builders, then apply
    # the cell mutation.  Seed override happens at the very end so it
    # supersedes the orchestrator's default seed=42.
    v1_cfg = v1_config(args.quick)
    v2_cfg = v2_config(args.quick)
    v3_cfg = v3_config(args.quick)
    v4_cfg = v4_config(args.quick)
    v1_cfg, v2_cfg, v3_cfg, v4_cfg = apply_cell(
        args.cell, args.base_cell, v1_cfg, v2_cfg, v3_cfg, v4_cfg,
    )
    v1_cfg = replace(v1_cfg, seed=args.seed)
    v2_cfg = replace(v2_cfg, seed=args.seed)
    v3_cfg = replace(v3_cfg, seed=args.seed)
    v4_cfg = replace(v4_cfg, seed=args.seed)

    # Persist the resolved cell config alongside metrics — reproducibility.
    (out_dir / "cell_config.json").write_text(json.dumps({
        "cell": args.cell,
        "base_cell": args.base_cell,
        "seed": args.seed,
        "skip_v3": args.skip_v3,
        "skip_v4": args.skip_v4,
        "quick": args.quick,
        "v1_cfg": asdict(v1_cfg),
        "v2_cfg": asdict(v2_cfg),
        "v3_cfg": asdict(v3_cfg),
        "v4_cfg": asdict(v4_cfg),
    }, indent=2, default=str))

    metrics: dict = {
        "cell": args.cell,
        "base_cell": args.base_cell,
        "seed": args.seed,
        "stages": {},
    }

    # Loaders.  V1/V2 use D1-D4 (D5 reserved for anomaly stages).
    log("Loading D1..D4 SSL loaders ...")
    SSL_LOADERS = [resolved_loader(f"{d}.yaml") for d in ("d1", "d2", "d3", "d4")]

    # --- V1 acoustic --------------------------------------------------------
    log("V1 acoustic ...")
    t0 = time.time()
    v1_a = train_v1_per_modality(SSL_LOADERS, modality="acoustic", cfg=v1_cfg)
    log(f"  V1 acoustic {time.time()-t0:.0f}s — sanity NMI={v1_a.sanity_gate.get('nmi',0):.3f} "
        f"early_stopped_epoch={v1_a.early_stopped_epoch} best_val={v1_a.best_val_loss:.3f}")
    torch.save(v1_a.encoder.state_dict(), out_dir / "v1" / "acoustic.pt")
    metrics["stages"]["v1_acoustic"] = {
        "epochs_planned": v1_cfg.epochs,
        "early_stopped_epoch": v1_a.early_stopped_epoch,
        "best_val_loss": v1_a.best_val_loss,
        "train_loss_final": v1_a.train_loss_history[-1],
        "val_loss_final": v1_a.val_loss_history[-1],
        "sanity_nmi": v1_a.sanity_gate.get("nmi", 0.0),
        "sanity_ari": v1_a.sanity_gate.get("ari", 0.0),
        "sanity_purity": v1_a.sanity_gate.get("purity", 0.0),
    }

    # --- V1 vibration -------------------------------------------------------
    log("V1 vibration ...")
    t0 = time.time()
    v1_v = train_v1_per_modality(SSL_LOADERS, modality="vibration", cfg=v1_cfg)
    log(f"  V1 vibration {time.time()-t0:.0f}s — sanity NMI={v1_v.sanity_gate.get('nmi',0):.3f} "
        f"early_stopped_epoch={v1_v.early_stopped_epoch} best_val={v1_v.best_val_loss:.3f}")
    torch.save(v1_v.encoder.state_dict(), out_dir / "v1" / "vibration.pt")
    metrics["stages"]["v1_vibration"] = {
        "epochs_planned": v1_cfg.epochs,
        "early_stopped_epoch": v1_v.early_stopped_epoch,
        "best_val_loss": v1_v.best_val_loss,
        "train_loss_final": v1_v.train_loss_history[-1],
        "val_loss_final": v1_v.val_loss_history[-1],
        "sanity_nmi": v1_v.sanity_gate.get("nmi", 0.0),
        "sanity_purity": v1_v.sanity_gate.get("purity", 0.0),
    }

    # --- V2 fusion ----------------------------------------------------------
    log("V2 fusion ...")
    t0 = time.time()
    v2 = train_v2_fusion(
        SSL_LOADERS, cfg=v2_cfg,
        v1_acoustic_state_dict=v1_a.encoder.state_dict(),
        v1_vibration_state_dict=v1_v.encoder.state_dict(),
    )
    log(f"  V2 {time.time()-t0:.0f}s — RQ1 NMI={v2.rq1.get('nmi',0):.3f} "
        f"early_stopped_epoch={v2.early_stopped_epoch} best_val={v2.best_val_loss:.3f}")
    torch.save(v2.encoder.state_dict(), out_dir / "v2" / "encoder.pt")
    metrics["stages"]["v2"] = {
        "epochs_planned": v2_cfg.epochs,
        "early_stopped_epoch": v2.early_stopped_epoch,
        "best_val_loss": v2.best_val_loss,
        "train_loss_final": v2.train_loss_history[-1],
        "val_loss_final": v2.val_loss_history[-1],
        "rq1_nmi": v2.rq1.get("nmi", 0.0),
        "rq1_purity": v2.rq1.get("purity", 0.0),
    }

    # --- V2 modality probe (always run; cheap, important for selection) ----
    try:
        from src.modeling.context.v2_ssl import _gather_labeled_segments
        labeled_segs = _gather_labeled_segments(SSL_LOADERS, v2_cfg)
        probe = run_modality_balance_probe(
            v2.encoder, labeled_segs, v2_cfg=v2_cfg, n_clusters=3, seed=v2_cfg.seed,
        )
        nmi_both = probe.both.get("nmi", 0.0)
        nmi_ac = probe.acoustic_only.get("nmi", 0.0)
        log(f"  modality probe: both={nmi_both:.3f} ac_only={nmi_ac:.3f} "
            f"Δ={nmi_both - nmi_ac:+.3f}")
        metrics["stages"]["v2_modality_probe"] = {
            "both_nmi": nmi_both,
            "acoustic_only_nmi": nmi_ac,
            "vibration_only_nmi": probe.vibration_only.get("nmi", 0.0),
            "delta_both_minus_acoustic": float(nmi_both - nmi_ac),
            "n_segments": len(probe.healthy_segments_used),
        }
    except Exception as e:
        log(f"  modality probe skipped: {type(e).__name__}: {e}")
        metrics["stages"]["v2_modality_probe"] = {"skipped": f"{type(e).__name__}: {e}"}

    if args.skip_v3 and args.skip_v4:
        log("V3/V4 skipped per --skip-v3 + --skip-v4 (Phase 2/3 triage)")
        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
        log(f"Wrote {metrics_path}")
        return

    # V3 / V4 stages — for cells promoted to full pipeline (Phase 4 / 5),
    # delegate to the orchestrator's main() with the patched configs.  This
    # avoids reimplementing the V3/V4 logic and keeps the canonical-source-
    # of-truth invariant.  Patching is done by monkey-patching the builders
    # via module attributes before the import resolves.
    log("Promoting to full pipeline — running V3/V4 via full_run.main() with patched configs ...")
    import src.modeling.orchestration.full_run as _fr

    # Monkey-patch the builders to return our cell-mutated cfgs.  This is a
    # surgical override; the orchestrator otherwise behaves identically.
    _orig_v1, _orig_v2, _orig_v3, _orig_v4 = _fr.v1_config, _fr.v2_config, _fr.v3_config, _fr.v4_config
    _fr.v1_config = lambda quick: v1_cfg
    _fr.v2_config = lambda quick: v2_cfg
    _fr.v3_config = lambda quick: v3_cfg
    _fr.v4_config = lambda quick, scada_dim=0, unconditional=False: replace(
        v4_cfg, scada_dim=scada_dim, unconditional=unconditional,
    )
    try:
        fr_metrics = _fr.main(quick=args.quick)
        # Merge orchestrator metrics under the same top-level dict.
        metrics["full_run_stages"] = fr_metrics.get("stages", {})
        metrics["full_run_deep_vs_simple"] = fr_metrics.get("deep_vs_simple", {})
        metrics["full_run_timings_s"] = fr_metrics.get("timings_s", {})
    finally:
        _fr.v1_config = _orig_v1
        _fr.v2_config = _orig_v2
        _fr.v3_config = _orig_v3
        _fr.v4_config = _orig_v4

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    log(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
