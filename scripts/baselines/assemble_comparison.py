"""Assemble the head-vs-baseline comparison tables the thesis Results need.

The pipeline already computes every piece — but scattered across run files and
never joined: V1/V2 NMI live in the full-pipeline stages, V3 conditioning is in
``v3_fusion_depth`` (with a paired-bootstrap test), the localization paradigms
and their significance are in ``rq3_paradigm_comparison``, the classical V0
multilateration is its own stage, and the standalone prior-work V0 (Khamaisi
trio + KDE) lives under ``results/v0_anomaly/``.  This script crawls those,
pulls the head number and the baseline number for each research question, and
emits one consolidated Markdown + JSON comparison per RQ, **surfacing the
significance the pipeline already computed** rather than inventing new tests.

It recomputes nothing from raw scores (the per-window arrays are not saved in
``metrics.json``); where a comparison is genuinely missing — the unsupervised
RQ1 floor, the SRP-PHAT classical localizer, the V3-gated localization metric —
it says so explicitly, so the gaps are visible rather than silently blank.

Run::

    python -m scripts.baselines.assemble_comparison                 # latest full-pipeline run
    python -m scripts.baselines.assemble_comparison --run <run_dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "results" / "runs"
V0_ANOMALY_DIR = REPO / "results" / "v0_anomaly"
OUT_DIR = REPO / "results" / "comparison"

MISSING = "n/a"


# ---------------------------------------------------------------------------
# Small dict helpers (defensive: runs vary in schema across campaigns)
# ---------------------------------------------------------------------------


def _get(d: Any, *path: str, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return MISSING
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return MISSING
    return f"{f:.{nd}f}"


def _ci(lo, hi, nd: int = 3) -> str:
    if lo is None or hi is None:
        return ""
    try:
        if float(lo) != float(lo) or float(hi) != float(hi):
            return ""
    except (TypeError, ValueError):
        return ""
    return f" [{float(lo):.{nd}f}, {float(hi):.{nd}f}]"


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


def _latest_full_pipeline_run(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if (p / "metrics.json").exists() else None
    cands = sorted(
        glob.glob(str(RUNS_DIR / "*full_pipeline_b5_cma*" / "metrics.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    return Path(cands[0]).parent if cands else None


def _latest_v0_anomaly() -> dict | None:
    cands = sorted(
        glob.glob(str(V0_ANOMALY_DIR / "v0_anomaly_*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not cands:
        return None
    return json.loads(Path(cands[0]).read_text(encoding="utf-8"))


def _v3deep_real_anomaly_f1() -> tuple[float, float, int] | None:
    """Mean ± std real-anomaly F1 across the v3deep sweep runs, if present."""
    f1s: list[float] = []
    for p in glob.glob(str(RUNS_DIR / "*v3deep*" / "metrics.json")):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        for para in _get(d, "paradigms", default={}).values() if isinstance(_get(d, "paradigms"), dict) else []:
            f1 = _get(para, "real_anomaly", "f1") if isinstance(para, dict) else None
            if isinstance(f1, (int, float)) and f1 == f1:
                f1s.append(float(f1))
    if not f1s:
        return None
    import statistics
    return (statistics.mean(f1s), statistics.pstdev(f1s) if len(f1s) > 1 else 0.0, len(f1s))


# ---------------------------------------------------------------------------
# Per-RQ extraction
# ---------------------------------------------------------------------------


def build_rq1(st: dict) -> dict:
    """Operating-context recovery: floor → learned encoder → supervised ceiling."""
    rows = []
    # Unsupervised floor (handcrafted K-means) — present only if full_run was
    # re-run after the floor was added.
    floor = _get(st, "v0", "v0_mode_floor")
    rows.append({
        "method": "K-means / handcrafted (unsup. floor)",
        "role": "baseline (floor)",
        "nmi": _get(floor, "nmi"), "ari": _get(floor, "ari"), "purity": _get(floor, "purity"),
        "status": "present" if floor and "nmi" in (floor or {}) else "RE-RUN full_run for v0_mode_floor",
    })
    rows.append({
        "method": "V1 acoustic (per-modality SSL)", "role": "proposed (unimodal)",
        "nmi": _get(st, "v1_acoustic", "sanity_nmi"), "ari": _get(st, "v1_acoustic", "sanity_ari"),
        "purity": _get(st, "v1_acoustic", "sanity_purity"), "status": "present",
    })
    rows.append({
        "method": "V2 fusion (cross-attention)", "role": "proposed (headline)",
        "nmi": _get(st, "v2", "rq1_nmi"), "ari": _get(st, "v2", "rq1_ari"),
        "purity": _get(st, "v2", "rq1_purity"), "status": "present",
    })
    rows.append({
        "method": "V2 minus vibration (fusion ablation)", "role": "ablation",
        "nmi": _get(st, "v2_a1_drop_vibration", "rq1_nmi"),
        "ari": _get(st, "v2_a1_drop_vibration", "rq1_ari"),
        "purity": _get(st, "v2_a1_drop_vibration", "rq1_purity"), "status": "present",
    })
    # Supervised ceiling (LightGBM) — pooled F1 if present.
    lgbm = {k: v for k, v in _get(st, "v0", default={}).items() if k.startswith("v0_lgbm_")} \
        if isinstance(_get(st, "v0"), dict) else {}
    ceil_f1 = None
    for v in lgbm.values():
        if isinstance(v, dict) and isinstance(v.get("val_macro_f1"), (int, float)):
            ceil_f1 = v["val_macro_f1"]
    rows.append({
        "method": "LightGBM supervised (ceiling, macro-F1)", "role": "baseline (ceiling)",
        "nmi": None, "ari": None, "purity": None, "f1": ceil_f1,
        "status": "present" if ceil_f1 is not None else "RE-RUN full_run for v0_lgbm",
    })
    return {"metric": "mode recovery (NMI / ARI / purity); ceiling reported as macro-F1", "rows": rows}


def build_rq2(st: dict, v0_anom: dict | None) -> dict:
    fd = _get(st, "v3_fusion_depth", default={})
    paired = _get(fd, "v3_vs_a2_paired_test", default={})
    syn = _get(fd, "synthetic_anomaly_auc", default={})
    healthy = _get(fd, "per_cluster_breakdown_healthy", "alert_rate_total")
    f1 = _v3deep_real_anomaly_f1()

    rows = []
    rows.append({
        "method": "V3 conditional flow (proposed)", "role": "proposed (headline)",
        "healthy_alert_rate": healthy,
        "real_anomaly_f1": (f"{f1[0]:.3f}+/-{f1[1]:.3f} (n={f1[2]})" if f1 else None),
        "syn_auc@+5dB": _get(syn, "auc_conditional", "5.0"),
        "status": "present",
    })
    rows.append({
        "method": "V3 unconditional (conditioning ablation)", "role": "ablation",
        "healthy_alert_rate": None,
        "syn_auc@+5dB": _get(syn, "auc_unconditional", "5.0"),
        "status": "present",
    })
    # Standalone prior-work V0 (Khamaisi trio + KDE), acoustic, from results/v0_anomaly.
    if v0_anom:
        for r in v0_anom.get("results", []):
            if r.get("modality") == "acoustic" and "skipped" not in r:
                rows.append({
                    "method": f"V0 {r['model']} (prior-work, acoustic)", "role": "baseline (prior work)",
                    "roc_auc": r.get("roc_auc"),
                    "fpr_in_distribution": r.get("fpr_in_distribution"),
                    "fpr_domain_shift": r.get("fpr_domain_shift"),
                    "status": "present (separate eval axis — unpaired)",
                })
    else:
        rows.append({"method": "V0 prior-work (Khamaisi trio+KDE)", "role": "baseline (prior work)",
                     "status": "RE-RUN scripts.baselines.run_v0_anomaly"})

    significance = {
        "conditioning (V3 vs unconditional)": {
            "delta_nll": _get(paired, "delta_point"),
            "ci95": [_get(paired, "delta_ci95_low"), _get(paired, "delta_ci95_high")],
            "p_value": _get(paired, "p_value_two_sided"),
            "n_paired": _get(paired, "n_paired"),
            "method": _get(paired, "method"),
        }
    }
    return {
        "note": ("Head and prior-work V0 sit on different metric axes (NLL/alert-rate vs "
                 "ROC-AUC/FPR), so they are listed, not paired. The paired test below is the "
                 "scientifically meaningful within-pipeline conditioning ablation."),
        "rows": rows,
        "significance": significance,
    }


def build_rq3(st: dict) -> dict:
    fp = _get(st, "v4_four_paradigms", default={})
    rq3 = _get(st, "rq3_paradigm_comparison", default={})
    loro = _get(rq3, "loro_summary", "aggregate", default={})
    sig = _get(rq3, "significance", default={})
    v0m = _get(st, "v0_multilateration", default={})

    def mae(mode_key, loro_key):
        return _get(fp, mode_key, "val_mae_3d") or _get(loro, loro_key, "micro_mean_mae_m")

    rows = [
        {"method": "V4 fusion (proposed)", "role": "proposed (headline)",
         "holdout_mae_m": mae("fusion", "V4-fusion"),
         "ci95": [_get(fp, "fusion", "val_mae_ci95_low"), _get(fp, "fusion", "val_mae_ci95_high")],
         "status": "present (ungated spatial holdout)"},
        {"method": "V4 acoustic-only", "role": "ablation",
         "holdout_mae_m": mae("acoustic", "V4-acoustic"), "status": "present"},
        {"method": "V4 vibration-only", "role": "ablation",
         "holdout_mae_m": mae("vibration", "V4-vibration"), "status": "present"},
        {"method": "V0 accel multilateration (classical vibration)", "role": "baseline (classical)",
         "holdout_mae_m": _get(v0m, "d2", "mean_error_m"),
         "status": "present (D2 only; D3/D4 unresolved)"},
        {"method": "SRP-PHAT (classical acoustic)", "role": "baseline (classical)",
         "holdout_mae_m": None,
         "status": "RE-RUN: classical_rows empty — evaluate_srp_phat not joined here"},
    ]
    return {
        "note": ("Holdout is a genuine leave-position-out spatial split. The V3-gated MAE was "
                 "UNDEFINED in these stale runs (n_holdout_gated=0 — the legacy interval-overlap "
                 "gate); the direct-path gate (src/modeling/localization/v3_gating.py) fixes it. "
                 "On the cached holdout V3 flags ~100% of the impulsive knock windows, so the gated "
                 "MAE coincides with the ungated MAE; re-run the v4 sweep for the matched number."),
        "rows": rows,
        "significance": {
            "fusion_vs_acoustic": {"delta_m": _get(sig, "fusion_vs_acoustic_mae_delta_m"),
                                   "p_value": _get(sig, "fusion_vs_acoustic_p")},
            "fusion_vs_vibration": {"delta_m": _get(sig, "fusion_vs_vibration_mae_delta_m"),
                                    "p_value": _get(sig, "fusion_vs_vibration_p")},
        },
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_md(payload: dict) -> str:
    L = [f"# Head-vs-baseline comparison ({payload['generated']})", ""]
    L.append(f"Source run: `{payload['source_run']}`")
    L.append(f"Standalone V0: `{payload['source_v0'] or MISSING}`")
    L.append("")

    rq1 = payload["rq1"]
    L += ["## RQ1 - operating-context recovery", "", f"_{rq1['metric']}_", "",
          "| method | role | NMI | ARI | purity | macro-F1 | status |",
          "|---|---|---|---|---|---|---|"]
    for r in rq1["rows"]:
        L.append(f"| {r['method']} | {r['role']} | {_fmt(r.get('nmi'))} | {_fmt(r.get('ari'))} "
                 f"| {_fmt(r.get('purity'))} | {_fmt(r.get('f1'))} | {r['status']} |")

    rq2 = payload["rq2"]
    L += ["", "## RQ2 - anomaly detection under domain shift", "", f"_{rq2['note']}_", "",
          "| method | role | healthy alert | real-anomaly F1 | syn-AUC@+5dB | ROC-AUC | FPR in-dist | FPR shift | status |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rq2["rows"]:
        L.append(f"| {r['method']} | {r['role']} | {_fmt(r.get('healthy_alert_rate'))} "
                 f"| {r.get('real_anomaly_f1') or MISSING} | {_fmt(r.get('syn_auc@+5dB'))} "
                 f"| {_fmt(r.get('roc_auc'))} | {_fmt(r.get('fpr_in_distribution'))} "
                 f"| {_fmt(r.get('fpr_domain_shift'))} | {r['status']} |")
    for name, s in rq2["significance"].items():
        L.append("")
        L.append(f"- **{name}**: dNLL={_fmt(s['delta_point'] if 'delta_point' in s else s.get('delta_nll'))}"
                 f"{_ci(*(s.get('ci95') or [None, None]))}, p={_fmt(s.get('p_value'), 4)}, "
                 f"n={s.get('n_paired', MISSING)} ({s.get('method', '')})")

    rq3 = payload["rq3"]
    L += ["", "## RQ3 - source localization", "", f"_{rq3['note']}_", "",
          "| method | role | holdout MAE (m) | status |", "|---|---|---|---|"]
    for r in rq3["rows"]:
        L.append(f"| {r['method']} | {r['role']} | {_fmt(r.get('holdout_mae_m'))}"
                 f"{_ci(*(r.get('ci95') or [None, None]))} | {r['status']} |")
    for name, s in rq3["significance"].items():
        L.append("")
        L.append(f"- **{name}**: d={_fmt(s.get('delta_m'))} m, p={_fmt(s.get('p_value'), 4)}")
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=None, help="explicit full-pipeline run dir")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    run_dir = _latest_full_pipeline_run(args.run)
    if run_dir is None:
        print("No full_pipeline_b5_cma run with metrics.json found under results/runs/.")
        return 1
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    st = metrics.get("stages", metrics)
    v0_anom = _latest_v0_anomaly()
    v0_src = None
    if v0_anom:
        c = sorted(glob.glob(str(V0_ANOMALY_DIR / "v0_anomaly_*.json")), key=os.path.getmtime)
        v0_src = os.path.basename(c[-1]) if c else None

    payload = {
        "generated": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source_run": run_dir.name,
        "source_v0": v0_src,
        "rq1": build_rq1(st),
        "rq2": build_rq2(st, v0_anom),
        "rq3": build_rq3(st),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = payload["generated"]
    json_out = args.output or (OUT_DIR / f"comparison_{ts}.json")
    md_out = json_out.with_suffix(".md")
    json_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md = _render_md(payload)
    md_out.write_text(md, encoding="utf-8")
    # Console may be cp1252 (Windows); the .md file keeps the Unicode.
    sys.stdout.write(md.encode("ascii", "replace").decode("ascii") + "\n")
    print(f"\nwrote {json_out.relative_to(REPO)} and {md_out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    raise SystemExit(main())
