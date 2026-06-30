"""R2.2 — train THREE V3 CNF pipelines (acoustic / vibration / fusion) from
saved V1+V2 weights and emit a comparable per-pipeline metrics JSON.

The three paradigms (per the approved plan):

  * **V3-acoustic** — V1-acoustic encoder wrapped in
    :class:`V3AcousticOnlyAdapter`; vibration input is discarded.
  * **V3-vibration** — V1-vibration encoder wrapped in
    :class:`V3VibrationOnlyAdapter`; acoustic input is discarded.
  * **V3-fusion** — the full V2 fusion encoder (current chained-system
    pipeline).

All three use the same V3SSL config (epochs, batch_size, percentile, …) and
the same `_split_segments_by_recording` train / val_fit / val_eval split so
the per-pipeline numbers are directly comparable.  Each pipeline writes its
own ``v3_<paradigm>/`` artefacts under the run dir, plus a single
``metrics.json`` at the top level with all three pipelines' headlines side
by side for the RQ2 paradigm-comparison table.

Inputs (CLI):
  --from-run <dir>     archived run dir whose v1/ + v2/ encoders feed the
                       three V3 instances (e.g.
                       ``results/runs/20260516_103041__v1v2_only_b5_cma``
                       — the B5 / CMA run had the best V1-acoustic NMI to
                       date and is the natural baseline for RQ2/RQ3 work).

Run::

    python -m scripts.paradigms.run_v3_three_paradigms --from-run results/runs/<id>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.modeling.anomaly.v3_per_modality import (
    V3AcousticOnlyAdapter,
    V3VibrationOnlyAdapter,
)
from src.modeling.anomaly.v3_trainer import V3Config, V3Result, train_v3_cnf
from src.modeling.context.v1_ssl import V1SSLConfig
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.context.v2_ssl import V2SSLConfig
from src.modeling.encoders.per_modality import PerModalityEncoder
from src.modeling.orchestration.full_run import resolved_loader, v1_config, v2_config, v3_config

REPO = Path(__file__).resolve().parents[2]


def _build_v1_encoder(modality: str, v1_cfg: V1SSLConfig) -> PerModalityEncoder:
    return PerModalityEncoder(
        modality=modality,
        feature_dim=v1_cfg.feature_dim,
        embed_dim=v1_cfg.embed_dim,
        n_heads=v1_cfg.n_heads,
        acoustic_cnn_width_mult=v1_cfg.acoustic_cnn_width_mult,
    )


def _build_v2_encoder(v2_cfg: V2SSLConfig) -> V2FusionEncoder:
    return V2FusionEncoder(
        feature_dim=v2_cfg.feature_dim,
        embed_dim=v2_cfg.embed_dim,
        n_heads=v2_cfg.n_heads,
        context_mode=v2_cfg.context_mode,
        num_context_seeds=v2_cfg.num_context_seeds,
        acoustic_cnn_width_mult=v2_cfg.acoustic_cnn_width_mult,
    )


def _load_state(path: Path, target: torch.nn.Module, label: str) -> None:
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = target.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(
            f"  warning: {label} state_dict load — missing={list(missing)[:3]}{'…' if len(missing)>3 else ''}, "
            f"unexpected={list(unexpected)[:3]}{'…' if len(unexpected)>3 else ''}"
        )


def _train_one_pipeline(
    name: str,
    encoder: torch.nn.Module,
    loaders: list,
    v2_cfg: V2SSLConfig,
    v3_cfg: V3Config,
    out_dir: Path,
    log,
) -> V3Result:
    log(f"\n=== V3-{name} training ===")
    t0 = time.time()
    result = train_v3_cnf(encoder, loaders, v2_cfg=v2_cfg, v3_cfg=v3_cfg)
    log(f"V3-{name} done in {time.time()-t0:.0f}s — val NLL final: {result.val_nll[-1]:.3f}")
    pipeline_dir = out_dir / f"v3_{name}"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.flow.state_dict(), pipeline_dir / "flow.pt")
    np.savez(
        pipeline_dir / "thresholds.npz",
        centroids=result.thresholds.centroids,
        p95=result.thresholds.p95,
        p99=result.thresholds.p99,
        n_per_cluster=result.thresholds.n_per_cluster,
    )
    # val_scores / val_contexts are needed downstream by the R2.3 eval to
    # compute per-cohort + late-fusion combinations on a frozen split.
    np.savez(
        pipeline_dir / "val_eval.npz",
        scores=result.val_scores,
        contexts=result.val_contexts,
        labels=np.asarray(result.val_labels, dtype="U64"),
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--from-run", required=True,
        help="archived run dir containing v1/{acoustic,vibration}.pt + v2/encoder.pt",
    )
    ap.add_argument("--quick", action="store_true", help="3-epoch smoke (V1/V2/V3)")
    args = ap.parse_args()

    src_run = Path(args.from_run).resolve()
    if not src_run.exists():
        raise SystemExit(f"--from-run {src_run} does not exist")

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / "runs" / f"{timestamp}__v3_three_paradigms"
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        with (out_dir / "run_log.txt").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"REPO = {REPO}")
    log(f"src_run = {src_run}")
    log(f"out_dir = {out_dir}")

    v1_cfg = v1_config(args.quick)
    v2_cfg = v2_config(args.quick)
    v3_cfg = v3_config(args.quick)
    log(f"V1 config: epochs={v1_cfg.epochs}, acoustic_cnn_width_mult={v1_cfg.acoustic_cnn_width_mult}")
    log(f"V2 config: cma_weight={v2_cfg.cma_weight}, context_mode={v2_cfg.context_mode}")
    log(f"V3 config: epochs={v3_cfg.epochs}, n_threshold_clusters={v3_cfg.n_threshold_clusters}")

    log("Loading D1+D2+D3+D4 SSL loaders ...")
    SSL_LOADERS = [resolved_loader(f"{d}.yaml") for d in ("d1", "d2", "d3", "d4")]

    # Load encoders from the source run.
    log("Building + loading V1 acoustic encoder ...")
    v1_acoustic = _build_v1_encoder("acoustic", v1_cfg)
    _load_state(src_run / "v1" / "acoustic.pt", v1_acoustic, "V1 acoustic")

    log("Building + loading V1 vibration encoder ...")
    v1_vibration = _build_v1_encoder("vibration", v1_cfg)
    _load_state(src_run / "v1" / "vibration.pt", v1_vibration, "V1 vibration")

    log("Building + loading V2 fusion encoder ...")
    v2_encoder = _build_v2_encoder(v2_cfg)
    _load_state(src_run / "v2" / "encoder.pt", v2_encoder, "V2 fusion")

    # Wrap V1s in the V2-API-compatible adapters.
    acoustic_adapter = V3AcousticOnlyAdapter(v1_acoustic)
    vibration_adapter = V3VibrationOnlyAdapter(v1_vibration)

    # Train three V3 instances.
    metrics: dict = {
        "source_run": str(src_run.relative_to(REPO)),
        "v1_cfg": {k: v for k, v in asdict(v1_cfg).items() if not isinstance(v, (list, tuple, dict))},
        "v2_cfg": {k: v for k, v in asdict(v2_cfg).items() if not isinstance(v, (list, tuple, dict))},
        "v3_cfg": {k: v for k, v in asdict(v3_cfg).items() if not isinstance(v, (list, tuple, dict))},
        "pipelines": {},
    }

    pipelines = [
        ("acoustic", acoustic_adapter),
        ("vibration", vibration_adapter),
        ("fusion", v2_encoder),
    ]
    results: dict[str, V3Result] = {}
    for name, enc in pipelines:
        try:
            res = _train_one_pipeline(name, enc, SSL_LOADERS, v2_cfg, v3_cfg, out_dir, log)
            results[name] = res
            metrics["pipelines"][name] = {
                "val_nll_final": float(res.val_nll[-1]),
                "val_nll_min_final": float(res.val_nll_min[-1]) if res.val_nll_min else float("nan"),
                "val_nll_max_final": float(res.val_nll_max[-1]) if res.val_nll_max else float("nan"),
                "p95_per_cluster": res.thresholds.p95.tolist(),
                "p99_per_cluster": res.thresholds.p99.tolist(),
                "n_val_windows": int(res.val_scores.shape[0]),
                "n_threshold_fit_recordings": len(res.threshold_fit_recording_ids),
                "n_val_eval_recordings": len(res.val_recording_ids),
                "n_clusters_fit": int(res.thresholds.centroids.shape[0]),
            }
        except Exception as e:
            log(f"V3-{name} FAILED: {type(e).__name__}: {e}")
            metrics["pipelines"][name] = {"error": f"{type(e).__name__}: {e}"}

    # Headline side-by-side summary.
    log("\n=== V3 three-paradigm headline summary ===")
    log(f"{'pipeline':<10} | {'val_NLL':>10} | {'p95[0]':>8} | {'p95[1]':>8} | {'p95[2]':>8} | {'n_val_w':>8}")
    for name in ("acoustic", "vibration", "fusion"):
        m = metrics["pipelines"].get(name, {})
        if "error" in m:
            log(f"{name:<10} | ERROR: {m['error']}")
            continue
        p95 = m.get("p95_per_cluster", [float("nan")] * 3)
        log(
            f"{name:<10} | {m.get('val_nll_final', float('nan')):>+10.2f} | "
            f"{p95[0]:>+8.2f} | {p95[1] if len(p95) > 1 else float('nan'):>+8.2f} | "
            f"{p95[2] if len(p95) > 2 else float('nan'):>+8.2f} | "
            f"{m.get('n_val_windows', 0):>8d}"
        )

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log(f"\nWrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
