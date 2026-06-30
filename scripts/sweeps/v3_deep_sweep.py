"""Deep V3 anomaly sweep — Phase 1 of the V3-first deep campaign.

V3 is the GATE for V4 (V4 only fires on V3-flagged windows), so it is tuned
FIRST. This sweep trains V3 against a FROZEN V1/V2 encoder (individual
training — no V1/V2 retraining per cell, ~5-10× cheaper than the full
pipeline) and selects by the **real-anomaly F1** (the B4 metric, vs weak
knock GT) under a **train/val NLL-gap guardrail**.

Selection objective (the train/val gap is a guardrail, not the optimisation
target): maximize ``real_anomaly_f1`` (+ synthetic-AUC@+5dB as tie-break),
**subject to** ``|val_nll − train_nll| ≤ gap_guardrail``. A cell that shrinks
the gap by tanking F1 is rejected.

Cells:
  Regularization (dropout × weight_decay), 12 cells:
    v3_d0_w4 v3_d0_w5 v3_d0_w3   # dropout 0.0
    v3_d1_w4 v3_d1_w5 v3_d1_w3   # dropout 0.1
    v3_d2_w4 v3_d2_w5 v3_d2_w3   # dropout 0.2
    v3_d3_w4 v3_d3_w5 v3_d3_w3   # dropout 0.3
  Capacity (n_layers, hidden_dim), 3 cells:
    v3_cap_small  # (4, 32)
    v3_cap_base   # (6, 64)
    v3_cap_big    # (8, 128)

Run::

    python -m scripts.sweeps.v3_deep_sweep --encoder-run results/runs/<best_encoder_dir>
    python -m scripts.sweeps.v3_deep_sweep --encoder-run <dir> --cell v3_d2_w5
    python -m scripts.sweeps.v3_deep_sweep --encoder-run <dir> --cell v3_d2_w5 --all-paradigms
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from src.modeling.anomaly import V3Config, train_v3_cnf
from src.modeling.anomaly.event_detection import v3_real_anomaly_detection
from src.modeling.anomaly.synthetic_eval import evaluate_synthetic_anomaly_auc
from src.modeling.anomaly.v3_per_modality import (
    V3AcousticOnlyAdapter,
    V3VibrationOnlyAdapter,
)
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.encoders import PerModalityEncoder
from src.modeling.orchestration.full_run import (
    REPO_ROOT,
    resolved_loader,
    v1_config,
    v2_config,
    v3_config,
)

_DROPOUT_LEVELS = {"d0": 0.0, "d1": 0.1, "d2": 0.2, "d3": 0.3}
_WD_LEVELS = {"w4": 1e-4, "w5": 5e-4, "w3": 1e-3}
_CAP_LEVELS = {"small": (4, 32), "base": (6, 64), "big": (8, 128)}


def _all_cells() -> list[str]:
    reg = [f"v3_{d}_{w}" for d in _DROPOUT_LEVELS for w in _WD_LEVELS]
    cap = [f"v3_cap_{c}" for c in _CAP_LEVELS]
    return reg + cap


def _apply_cell(cell_id: str, v3_cfg: V3Config) -> V3Config:
    if cell_id.startswith("v3_cap_"):
        key = cell_id[len("v3_cap_"):]
        if key not in _CAP_LEVELS:
            raise ValueError(f"unknown capacity cell {cell_id!r}")
        n_layers, hidden = _CAP_LEVELS[key]
        return replace(v3_cfg, n_layers=n_layers, hidden_dim=hidden)
    parts = cell_id.split("_")
    if len(parts) != 3 or parts[0] != "v3":
        raise ValueError(f"malformed v3 cell id: {cell_id!r}")
    d_key, w_key = parts[1], parts[2]
    if d_key not in _DROPOUT_LEVELS or w_key not in _WD_LEVELS:
        raise ValueError(f"unknown axis level in {cell_id!r}")
    return replace(v3_cfg, dropout_p=_DROPOUT_LEVELS[d_key], weight_decay=_WD_LEVELS[w_key])


def _load_encoders(encoder_run: Path, v1_cfg, v2_cfg):
    """Load frozen V1 acoustic/vibration + V2 fusion encoders from a run dir."""
    v2 = V2FusionEncoder(
        feature_dim=v2_cfg.feature_dim, embed_dim=v2_cfg.embed_dim,
        n_heads=v2_cfg.n_heads, context_mode=v2_cfg.context_mode,
        num_context_seeds=v2_cfg.num_context_seeds,
        acoustic_cnn_width_mult=v2_cfg.acoustic_cnn_width_mult,
    )
    v2.load_state_dict(torch.load(encoder_run / "v2" / "encoder.pt", map_location="cpu"))
    v2.eval()

    def _v1(modality: str, fname: str) -> PerModalityEncoder:
        enc = PerModalityEncoder(
            modality=modality, feature_dim=v1_cfg.feature_dim,
            embed_dim=v1_cfg.embed_dim, n_heads=v1_cfg.n_heads,
            acoustic_cnn_width_mult=v1_cfg.acoustic_cnn_width_mult,
        )
        enc.load_state_dict(torch.load(encoder_run / "v1" / fname, map_location="cpu"))
        enc.eval()
        return enc

    v1_a = _v1("acoustic", "acoustic.pt")
    v1_v = _v1("vibration", "vibration.pt")
    return v2, v1_a, v1_v


def _evaluate(res, v2_encoder, anom_loaders, anom_segments, v2_cfg, v3_cfg) -> dict:
    """Real-anomaly F1 + synthetic AUC + train/val NLL gap for one V3Result."""
    out: dict = {}
    train_nll = float(res.train_nll[-1]) if res.train_nll else float("nan")
    val_nll = float(res.val_nll[-1]) if res.val_nll else float("nan")
    out["train_nll_final"] = train_nll
    out["val_nll_final"] = val_nll
    out["nll_gap"] = abs(val_nll - train_nll)
    out["early_stopped_epoch"] = res.early_stopped_epoch
    # Real-anomaly detection vs weak knock GT.  Pass the trained xt_pool so
    # inference pooling matches training pooling (calibration fix).
    try:
        real = v3_real_anomaly_detection(
            v2_encoder, res.flow, res.thresholds, anom_segments,
            v2_cfg=v2_cfg, percentile=v3_cfg.threshold_percentile,
            xt_pool=res.xt_pool, device=v3_cfg.device,
            anchor_norm=((res.anchor_mean, res.anchor_std)
                         if getattr(res, "anchor_mean", None) is not None else None),
        )
        out["real_anomaly"] = real
    except Exception as e:
        out["real_anomaly"] = {"skipped": f"{type(e).__name__}: {e}"}
    # Synthetic AUC ladder (uses cached val x/c on the V3Result).
    try:
        if res.val_x is not None and res.val_contexts is not None and res.val_x.shape[0] >= 4:
            auc = evaluate_synthetic_anomaly_auc(
                res.flow, res.val_x, res.val_contexts,
                snr_db_list=(-5.0, 0.0, 5.0, 10.0), n_boot=200, seed=v3_cfg.seed,
            )
            out["synthetic_auc"] = auc.snr_db_to_auc
    except Exception as e:
        out["synthetic_auc"] = {"skipped": f"{type(e).__name__}: {e}"}
    # V3-vs-simple: per-cluster diagonal-Gaussian baseline on the same x/c the
    # flow trained on.  Δ = V3_val_NLL − baseline_val_NLL; Δ<0 ⇒ V3 beats the
    # simple density (earns its complexity).  Thesis "deep-vs-simple" number.
    try:
        from src.modeling.anomaly.kde_baseline import fit_and_score_kde_on_ct
        if (res.train_x is not None and res.train_contexts is not None
                and res.val_x is not None and res.val_contexts is not None):
            base = fit_and_score_kde_on_ct(
                res.train_x, res.train_contexts, res.val_x, res.val_contexts,
                n_clusters=v3_cfg.n_threshold_clusters, seed=v3_cfg.seed,
            )
            out["simple_baseline_val_nll"] = base.val_nll_mean
            out["v3_minus_simple_nll"] = val_nll - base.val_nll_mean
            out["v3_beats_simple"] = bool(val_nll < base.val_nll_mean)
    except Exception as e:
        out["simple_baseline_val_nll"] = None
        out["v3_minus_simple_error"] = f"{type(e).__name__}: {e}"
    return out


def _log(msg: str, log_path: Path) -> None:
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(line + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-run", required=True,
                   help="Run dir with frozen v1/{acoustic,vibration}.pt + v2/encoder.pt")
    p.add_argument("--cell", default=None, help=f"Single cell; omit to run all. {_all_cells()}")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--all-paradigms", action="store_true",
                   help="Also train acoustic + vibration paradigms (default: fusion only).")
    args = p.parse_args()

    encoder_run = Path(args.encoder_run)
    if not (encoder_run / "v2" / "encoder.pt").exists():
        raise SystemExit(f"v2/encoder.pt not found under {encoder_run}")

    cells = [args.cell] if args.cell else _all_cells()
    v1_cfg = v1_config(args.quick)
    v2_cfg = v2_config(args.quick)
    base_v3 = v3_config(args.quick)
    for cid in cells:  # fail fast on typos
        _apply_cell(cid, base_v3)

    print("Loading frozen encoders + D1..D5 anomaly loaders ...")
    v2_enc, v1_a, v1_v = _load_encoders(encoder_run, v1_cfg, v2_cfg)
    anom_ids = ("d1", "d2", "d3", "d4", "d5")
    anom_loaders = []
    for d in anom_ids:
        try:
            anom_loaders.append(resolved_loader(f"{d}.yaml"))
        except Exception as e:
            print(f"  skip {d}: {e}")
    anom_segments = [s for L in anom_loaders for s in L.list_segments() if s.is_anomaly]

    paradigms = [("fusion", v2_enc)]
    if args.all_paradigms:
        paradigms = [
            ("acoustic", V3AcousticOnlyAdapter(v1_a)),
            ("vibration", V3VibrationOnlyAdapter(v1_v)),
            ("fusion", v2_enc),
        ]

    for cell_id in cells:
        v3_cfg = replace(_apply_cell(cell_id, base_v3), seed=args.seed)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "results" / "runs" / f"{ts}__v3deep_{cell_id}_s{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run_log.txt"
        _log(f"cell={cell_id} dropout={v3_cfg.dropout_p} wd={v3_cfg.weight_decay} "
             f"n_layers={v3_cfg.n_layers} hidden={v3_cfg.hidden_dim}", log_path)
        (out_dir / "cell_config.json").write_text(json.dumps({
            "cell": cell_id, "seed": args.seed, "encoder_run": str(encoder_run),
            "v3_cfg": asdict(v3_cfg),
        }, indent=2, default=str))

        cell_metrics: dict = {"cell": cell_id, "seed": args.seed, "paradigms": {}}
        for name, enc in paradigms:
            t0 = time.time()
            try:
                res = train_v3_cnf(enc, anom_loaders, v2_cfg=v2_cfg, v3_cfg=v3_cfg)
            except Exception as e:
                _log(f"  V3-{name} FAILED: {type(e).__name__}: {e}", log_path)
                cell_metrics["paradigms"][name] = {"error": f"{type(e).__name__}: {e}"}
                continue
            ev = _evaluate(res, enc, anom_loaders, anom_segments, v2_cfg, v3_cfg)
            dt = time.time() - t0
            f1 = ev.get("real_anomaly", {}).get("f1") if isinstance(ev.get("real_anomaly"), dict) else None
            _log(f"  V3-{name} {dt:.0f}s — val_nll={ev['val_nll_final']:.3f} "
                 f"gap={ev['nll_gap']:.3f} real_F1={f1 if f1 is None else round(f1,3)} "
                 f"es_epoch={ev['early_stopped_epoch']}", log_path)
            if name == "fusion":
                torch.save(res.flow.state_dict(), out_dir / "flow.pt")
                np.savez(out_dir / "thresholds.npz",
                         centroids=res.thresholds.centroids,
                         p95=res.thresholds.p95, p99=res.thresholds.p99,
                         n_per_cluster=res.thresholds.n_per_cluster)
                # Persist the learned xt_pool so V4 gating (and any re-scoring)
                # uses the same pooling the flow was trained with.  None when
                # the legacy mean-pool path was used.
                if res.xt_pool is not None:
                    torch.save(res.xt_pool.state_dict(), out_dir / "xt_pool.pt")
            cell_metrics["paradigms"][name] = ev
        (out_dir / "metrics.json").write_text(json.dumps(cell_metrics, indent=2, default=str))
        print(f"Wrote {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
