"""Aggregate `scripts.campaigns.ablation_full_pipeline` cells into a markdown report.

Reads every ``results/runs/*__ablation_*/metrics.json`` + ``cell_config.json``
and emits four tables to ``results/runs/ablation_report_<ts>.md``:

  1. **V1/V2 per-cell metrics** — sanity NMI, RQ1 NMI, modality-probe Δ,
     train/val gaps.  This is the primary Phase 2 / 3 selection table.
  2. **Deep-vs-simple comparison** — V3 CNF vs KDE NLL, V4 fusion vs V0
     multilateration MAE, per cell.  Empty rows for cells that ran with
     --skip-v3 / --skip-v4.
  3. **Phase 2 / 3 axis sweeps** — cells grouped by axis with the
     baseline_v2 row highlighted.  Lets you eyeball the per-axis ordering.
  4. **Phase 5 multi-seed verdict** — when a cell ID appears at multiple
     seeds, report mean ± std across seeds.

Selection guidance (printed at the bottom of the report):
  - V1 / V2 tuple: maximise (sanity_NMI, sanity_purity, modality_probe Δ),
    minimise absolute train/val gap.
  - Deep-vs-simple: cells where `deep_wins=True` for V3 or V4 dominate
    the framing-flip discussion.

Run::

    python -m scripts.campaigns.analyze_ablation
    python -m scripts.campaigns.analyze_ablation --filter "p2_a1_*"   # glob
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "results" / "runs"


def _load_runs(
    filter_glob: str | None,
    campaign_dir: Path | None = None,
) -> list[dict]:
    """Read ablation run dirs' metrics + config; filter by cell id and/or campaign.

    When ``campaign_dir`` is set, the run-dir set is restricted to those listed
    in the campaign's ``state.json``.  This keeps the report scoped to a single
    campaign even when many historical ablation runs live in ``results/runs``.
    Cells from PHASE 6 (`v4_aug_sweep`) live in `*__v4aug_*` dirs and are
    included in the campaign scope too.
    """
    if campaign_dir is not None:
        # Collect run dirs from this campaign's state.json AND, if it is a
        # master thesis campaign, the delegated deep campaign it points to
        # (deep_campaign_dir) — so one --campaign-dir gives a unified report.
        run_dirs: list[Path] = []

        def _collect(cdir: Path) -> None:
            sp = cdir / "state.json"
            if not sp.exists():
                return
            st = json.loads(sp.read_text())
            base = st.get("baseline_v2") or {}
            if base.get("run_dir"):
                run_dirs.append(Path(base["run_dir"]))
            for _k, entry in (st.get("cells") or {}).items():
                if entry.get("run_dir"):
                    run_dirs.append(Path(entry["run_dir"]))
            deep = st.get("deep_campaign_dir")
            if deep:
                _collect(RUNS_DIR / deep)

        _collect(campaign_dir)
        if not run_dirs:
            return []
    else:
        run_dirs = (
            sorted(RUNS_DIR.glob("*__ablation_*"))
            + sorted(RUNS_DIR.glob("*__v4aug_*"))
            + sorted(RUNS_DIR.glob("*__v3deep_*"))
            + sorted(RUNS_DIR.glob("*__v4deep_*"))
        )

    rows: list[dict] = []
    seen_dirs: set[Path] = set()
    for run_dir in run_dirs:
        if run_dir in seen_dirs:
            continue
        seen_dirs.add(run_dir)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        config_path = run_dir / "cell_config.json"
        try:
            metrics = json.loads(metrics_path.read_text())
            cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
        except Exception:
            continue
        # Baseline_v2 dirs (``*__full_pipeline_b5_cma``) have no
        # cell_config.json; synthesise a minimal one.
        cell = cfg.get("cell")
        if cell is None:
            if "__full_pipeline_b5_cma" in run_dir.name:
                cell = "baseline_v2"
            else:
                cell = metrics.get("cell", "?")
        if filter_glob and not fnmatch.fnmatch(cell, filter_glob):
            continue
        rows.append({"dir": run_dir, "cell": cell, "cfg": cfg, "metrics": metrics})
    return rows


def _fmt(v, prec: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "—"
    if isinstance(v, (int, bool)):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def _gap_abs(train: float | None, val: float | None) -> float | None:
    if train is None or val is None:
        return None
    if not isinstance(train, (int, float)) or not isinstance(val, (int, float)):
        return None
    return abs(val - train)


def _table_v1_v2(rows: list[dict]) -> str:
    cols = [
        "cell", "seed", "v1ac_NMI", "v1ac_purity", "v1ac_gap",
        "v1vib_NMI", "v1vib_purity", "v1vib_gap",
        "v2_NMI", "v2_purity", "modΔ", "v2_gap", "v1ac_stop", "v2_stop",
    ]
    out = ["## Table 1 — V1/V2 per-cell metrics", "",
           "| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        st = r["metrics"].get("stages", {})
        v1a = st.get("v1_acoustic", {})
        v1v = st.get("v1_vibration", {})
        v2 = st.get("v2", {})
        probe = st.get("v2_modality_probe", {})
        out.append("| " + " | ".join([
            r["cell"], str(r["cfg"].get("seed", "?")),
            _fmt(v1a.get("sanity_nmi")),
            _fmt(v1a.get("sanity_purity")),
            _fmt(_gap_abs(v1a.get("train_loss_final"), v1a.get("val_loss_final"))),
            _fmt(v1v.get("sanity_nmi")),
            _fmt(v1v.get("sanity_purity")),
            _fmt(_gap_abs(v1v.get("train_loss_final"), v1v.get("val_loss_final"))),
            _fmt(v2.get("rq1_nmi")),
            _fmt(v2.get("rq1_purity")),
            _fmt(probe.get("delta_both_minus_acoustic")),
            _fmt(_gap_abs(v2.get("train_loss_final"), v2.get("val_loss_final"))),
            _fmt(v1a.get("early_stopped_epoch"), prec=0),
            _fmt(v2.get("early_stopped_epoch"), prec=0),
        ]) + " |")
    return "\n".join(out) + "\n"


def _table_deep_vs_simple(rows: list[dict]) -> str:
    cols = [
        "cell", "seed",
        "V3 NLL", "KDE NLL", "Δ (V3-KDE)", "V3 wins",
        "V4 MAE (m)", "V0 multilat MAE (m)", "Δ (V4-V0)", "V4 wins",
    ]
    out = ["## Table 2 — Deep-vs-simple comparison", "",
           "Δ < 0 means the deep model beats the simple baseline.", "",
           "| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        dvs = r["metrics"].get("full_run_deep_vs_simple", {}) or r["metrics"].get("deep_vs_simple", {})
        anom = dvs.get("anomaly", {}) if isinstance(dvs, dict) else {}
        loc = dvs.get("localisation", {}) if isinstance(dvs, dict) else {}
        out.append("| " + " | ".join([
            r["cell"], str(r["cfg"].get("seed", "?")),
            _fmt(anom.get("deep_val_nll_mean")),
            _fmt(anom.get("simple_val_nll_mean")),
            _fmt(anom.get("delta_deep_minus_simple")),
            _fmt(anom.get("deep_wins")),
            _fmt(loc.get("deep_val_mae_m")),
            _fmt(loc.get("simple_val_mae_m")),
            _fmt(loc.get("delta_deep_minus_simple_m")),
            _fmt(loc.get("deep_wins")),
        ]) + " |")
    return "\n".join(out) + "\n"


def _axis_grouping(rows: list[dict]) -> str:
    """Group p2_* and p3_* cells by axis level for eyeball-comparison."""
    out = ["## Table 3 — Phase 2 / 3 axis sweeps", ""]

    # Phase 2 — aug × vibration_dropout grid
    p2_rows = [r for r in rows if r["cell"].startswith("p2_")]
    if p2_rows:
        out.append("### Phase 2: aug strength × vibration_dropout (V2 RQ1 NMI / modality Δ)")
        out.append("")
        out.append("| aug \\ vib_drop | 0.3 (v3) | 0.5 (v5, baseline) | 0.7 (v7) |")
        out.append("|---|---|---|---|")
        for aug_lvl, aug_label in (
            ("a0", "mid (baseline)"), ("a1", "strong"), ("a2", "very-strong"),
        ):
            cells = ["| " + aug_label]
            for vib_lvl in ("v3", "v5", "v7"):
                cell_id = f"p2_{aug_lvl}_{vib_lvl}"
                matched = [r for r in p2_rows if r["cell"] == cell_id]
                if not matched:
                    cells.append("—")
                    continue
                r = matched[0]
                v2 = r["metrics"].get("stages", {}).get("v2", {})
                probe = r["metrics"].get("stages", {}).get("v2_modality_probe", {})
                cells.append(f"NMI={_fmt(v2.get('rq1_nmi'))} / Δ={_fmt(probe.get('delta_both_minus_acoustic'))}")
            out.append(" | ".join(cells) + " |")
        out.append("")

    # Phase 3 — mixup × embed_dim grid
    p3_rows = [r for r in rows if r["cell"].startswith("p3_")]
    if p3_rows:
        out.append("### Phase 3: mixup_alpha × embed_dim (V2 RQ1 NMI / modality Δ)")
        out.append("")
        out.append("| mixup \\ embed_dim | 32 | 64 (baseline) | 128 |")
        out.append("|---|---|---|---|")
        for mix_lvl, mix_label in (("m0", "0.0"), ("m2", "0.2"), ("m4", "0.4")):
            cells = ["| " + mix_label]
            for edim_lvl in ("e32", "e64", "e128"):
                cell_id = f"p3_{mix_lvl}_{edim_lvl}"
                matched = [r for r in p3_rows if r["cell"] == cell_id]
                if not matched:
                    cells.append("—")
                    continue
                r = matched[0]
                v2 = r["metrics"].get("stages", {}).get("v2", {})
                probe = r["metrics"].get("stages", {}).get("v2_modality_probe", {})
                cells.append(f"NMI={_fmt(v2.get('rq1_nmi'))} / Δ={_fmt(probe.get('delta_both_minus_acoustic'))}")
            out.append(" | ".join(cells) + " |")
        out.append("")

    return "\n".join(out) + "\n"


def _multi_seed(rows: list[dict]) -> str:
    """Group by cell id and report mean±std when ≥ 2 seeds present."""
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    out = ["## Table 4 — Multi-seed verdict", ""]
    out.append("Cells with ≥ 2 seeds.  Mean ± std reported across seeds.")
    out.append("")
    out.append("| cell | n_seeds | V2 NMI | V4 MAE (m) | V3 NLL | V3-KDE Δ |")
    out.append("|---|---|---|---|---|---|")
    any_row = False
    for cell, runs in sorted(by_cell.items()):
        if len(runs) < 2:
            continue
        any_row = True

        def _collect(path: list[str], runs=runs) -> list[float]:
            vals: list[float] = []
            for r in runs:
                node = r["metrics"]
                for k in path:
                    if not isinstance(node, dict):
                        node = None
                        break
                    node = node.get(k)
                if isinstance(node, (int, float)) and not (
                    isinstance(node, float) and math.isnan(node)
                ):
                    vals.append(float(node))
            return vals

        def _mean_std(vals: list[float]) -> str:
            if not vals:
                return "—"
            if len(vals) == 1:
                return f"{vals[0]:.3f}"
            return f"{mean(vals):.3f} ± {stdev(vals):.3f}"

        nmi = _collect(["stages", "v2", "rq1_nmi"])
        mae = _collect(["full_run_stages", "v4_four_paradigms", "fusion", "val_mae_3d"])
        v3_nll = _collect(["full_run_deep_vs_simple", "anomaly", "deep_val_nll_mean"])
        d_anom = _collect(["full_run_deep_vs_simple", "anomaly", "delta_deep_minus_simple"])
        out.append(f"| {cell} | {len(runs)} | {_mean_std(nmi)} | {_mean_std(mae)} | "
                   f"{_mean_std(v3_nll)} | {_mean_std(d_anom)} |")
    if not any_row:
        out.append("| (no multi-seed cells yet) | | | | | |")
    return "\n".join(out) + "\n"


def _table_v3_deep(rows: list[dict]) -> str:
    """Deep V3 sweep — real-anomaly detection vs weak knock GT + NLL gap."""
    deep = [r for r in rows if isinstance(r["metrics"].get("paradigms"), dict)]
    out = ["## Table 5 — Deep V3 sweep (real-anomaly detection, gap-guarded)", ""]
    if not deep:
        out.append("_(no v3deep cells loaded)_\n")
        return "\n".join(out)
    cols = ["cell", "seed", "real_P", "real_R", "real_F1", "onset_err_s",
            "syn_AUC@5", "val_NLL", "NLL_gap", "es_epoch"]
    out += ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in deep:
        fus = (r["metrics"].get("paradigms") or {}).get("fusion") or {}
        ra = fus.get("real_anomaly") or {}
        auc = fus.get("synthetic_auc") or {}
        auc5 = auc.get("5.0", auc.get(5.0)) if isinstance(auc, dict) else None
        out.append("| " + " | ".join([
            r["cell"], str(r["cfg"].get("seed", "?")),
            _fmt(ra.get("precision")), _fmt(ra.get("recall")), _fmt(ra.get("f1")),
            _fmt(ra.get("median_onset_error_s")), _fmt(auc5),
            _fmt(fus.get("val_nll_final")), _fmt(fus.get("nll_gap")),
            _fmt(fus.get("early_stopped_epoch"), prec=0),
        ]) + " |")
    return "\n".join(out) + "\n"


def _table_v4_deep(rows: list[dict]) -> str:
    """Deep V4 sweep — spatial-holdout MAE, V3-gated, + train/val MAE gap.

    `mae_gap_m` is `|val_mae - train_mae|` in METRES (computed from a final
    forward pass on train_samples).  The legacy `loss_gap` column is the
    smooth-L1 loss difference in loss_scale-cm units, kept for backward
    compat with older runs that pre-date the train-MAE pass.
    """
    deep = [r for r in rows if "holdout_mae_ungated_m" in r["metrics"]]
    out = ["## Table 6 — Deep V4 sweep (spatial holdout, V3-gated, gap-guarded)", ""]
    if not deep:
        out.append("_(no v4deep cells loaded)_\n")
        return "\n".join(out)
    cols = ["cell", "seed", "train_select", "holdout_MAE_gated", "holdout_MAE_ungated",
            "n_gated", "train_MAE_m", "mae_gap_m", "loss_gap", "es_epoch"]
    out += ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in deep:
        m = r["metrics"]
        out.append("| " + " | ".join([
            r["cell"], str(r["cfg"].get("seed", "?")),
            str(m.get("train_select", "?")),
            _fmt(m.get("holdout_mae_v3gated_m")),
            _fmt(m.get("holdout_mae_ungated_m")),
            _fmt(m.get("n_holdout_gated"), prec=0),
            _fmt(m.get("train_mae_3d_m")),
            _fmt(m.get("train_val_mae_gap_m")),
            _fmt(m.get("train_val_gap_m")),
            _fmt(m.get("early_stopped_epoch"), prec=0),
        ]) + " |")
    return "\n".join(out) + "\n"


def _table_acoustic(rows: list[dict]) -> str:
    """Acoustic-improvement sweep — width_mult × cwt_n_scales grid."""
    aco = [r for r in rows if r["cell"].startswith("pa_")]
    out = ["## Table 7 — Acoustic-improvement sweep (V2 NMI / V1-acoustic NMI)", ""]
    if not aco:
        out.append("_(no acoustic pa_* cells loaded)_\n")
        return "\n".join(out)
    out.append("| width \\ cwt_scales | 16 | 32 (baseline) | 64 |")
    out.append("|---|---|---|---|")
    for w_lvl, w_label in (("w1", "1× (baseline)"), ("w2", "2× (R1a, re-tested)")):
        cells = ["| " + w_label]
        for c_lvl in ("cwt16", "cwt32", "cwt64"):
            cid = f"pa_{w_lvl}_{c_lvl}"
            matched = [r for r in aco if r["cell"] == cid]
            if not matched:
                cells.append("—")
                continue
            st = matched[0]["metrics"].get("stages", {})
            v2 = st.get("v2", {})
            v1a = st.get("v1_acoustic", {})
            cells.append(f"V2={_fmt(v2.get('rq1_nmi'))} / V1ac={_fmt(v1a.get('sanity_nmi'))}")
        out.append(" | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _table_v3_paradigms(rows: list[dict]) -> str:
    """Per-modality V3 anomaly (acoustic / vibration / fusion) — from verdict
    cells run with --all-paradigms (metrics.paradigms has all three)."""
    para_rows = [
        r for r in rows
        if isinstance(r["metrics"].get("paradigms"), dict)
        and len(r["metrics"]["paradigms"]) >= 2
    ]
    out = ["## Table 8 — V3 three paradigms (real-anomaly F1 per modality)", ""]
    if not para_rows:
        out.append("_(no multi-paradigm V3 cells; run the V3 winner with --all-paradigms)_\n")
        return "\n".join(out)
    cols = ["cell", "seed", "acoustic_F1", "vibration_F1", "fusion_F1",
            "acoustic_NLL", "vibration_NLL", "fusion_NLL"]
    out += ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in para_rows:
        pa = r["metrics"]["paradigms"]

        def _f1(name, pa=pa):
            ra = (pa.get(name) or {}).get("real_anomaly") or {}
            return _fmt(ra.get("f1"))

        def _nll(name, pa=pa):
            return _fmt((pa.get(name) or {}).get("val_nll_final"))

        out.append("| " + " | ".join([
            r["cell"], str(r["cfg"].get("seed", "?")),
            _f1("acoustic"), _f1("vibration"), _f1("fusion"),
            _nll("acoustic"), _nll("vibration"), _nll("fusion"),
        ]) + " |")
    return "\n".join(out) + "\n"


def _table_v4_channel_modes(rows: list[dict]) -> str:
    """Per-modality V4 localization (srp / tdoa / vibration / fusion) — from
    verdict cells run with --all-channel-modes (metrics.channel_modes)."""
    cm_rows = [r for r in rows if isinstance(r["metrics"].get("channel_modes"), dict)]
    out = ["## Table 9 — V4 four paradigms (holdout V3-gated MAE per channel mode)", ""]
    if not cm_rows:
        out.append("_(no multi-channel V4 cells; run the V4 winner with --all-channel-modes)_\n")
        return "\n".join(out)
    cols = ["cell", "seed", "srp_only", "tdoa_only", "vibration_only", "fusion(both)"]
    out += ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in cm_rows:
        cm = r["metrics"]["channel_modes"]

        def _mae(mode, cm=cm):
            d = cm.get(mode) or {}
            v = d.get("holdout_mae_v3gated_m")
            if v is None:
                v = d.get("holdout_mae_ungated_m")
            return _fmt(v)

        out.append("| " + " | ".join([
            r["cell"], str(r["cfg"].get("seed", "?")),
            _mae("srp_only"), _mae("tdoa_only"),
            _mae("vibration_only_learned"), _mae("both"),
        ]) + " |")
    return "\n".join(out) + "\n"


def _table_v4_train_select(rows: list[dict]) -> str:
    """V4 training-window-selection ablation: impulse vs V3-gated vs all.

    Shows whether HOW training windows are chosen (offline impulse weak-GT,
    deployment-consistent V3 gating, or no selection) affects the V3-gated
    holdout MAE.  Directly answers the 'does train/serve skew matter?' question.
    """
    # Only show cells that were actually run with ≥2 distinct selectors (the
    # Phase-2b ablation) — every regular v4deep cell carries train_select=
    # "impulse" by default, which is not the ablation.
    from collections import defaultdict as _dd
    sel_by_cell: dict[str, set] = _dd(set)
    for r in rows:
        ts = r["metrics"].get("train_select")
        if ts:
            sel_by_cell[r["cell"]].add(ts)
    ablation_cells = {c for c, sels in sel_by_cell.items() if len(sels) >= 2}
    ts_rows = [r for r in rows
               if r["metrics"].get("train_select") and r["cell"] in ablation_cells]
    out = ["## Table 10 — V4 training-window selection (impulse / v3gated / all)", ""]
    if not ts_rows:
        out.append("_(no train-select ablation cells loaded)_\n")
        return "\n".join(out)
    cols = ["cell", "train_select", "seed", "holdout_MAE_gated", "holdout_MAE_ungated",
            "n_holdout_gated", "train_val_gap_m"]
    out += ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    # Sort so the three modes group together per cell.
    for r in sorted(ts_rows, key=lambda r: (r["cell"], str(r["metrics"].get("train_select")))):
        m = r["metrics"]
        out.append("| " + " | ".join([
            r["cell"], str(m.get("train_select")), str(r["cfg"].get("seed", "?")),
            _fmt(m.get("holdout_mae_v3gated_m")),
            _fmt(m.get("holdout_mae_ungated_m")),
            _fmt(m.get("n_holdout_gated"), prec=0),
            _fmt(m.get("train_val_gap_m")),
        ]) + " |")
    return "\n".join(out) + "\n"


def _table_lopo(campaign_dir: Path | None) -> str:
    """Table 11 — LOPO-CV by position (V4 winner only).

    Reads `<campaign_dir>/lopo/summary.json` + `folds.jsonl` and reports:
      - per-position fold MAE (every labelled position the V4 winner was
        leave-one-position-out evaluated on)
      - aggregate mean ± std per channel mode
    """
    out = ["## Table 11 — LOPO-CV by position (V4 winner)", ""]
    if campaign_dir is None:
        out.append("_(no campaign_dir set; LOPO summary not loaded)_\n")
        return "\n".join(out)
    summary_p = campaign_dir / "lopo" / "summary.json"
    folds_p = campaign_dir / "lopo" / "folds.jsonl"
    if not summary_p.exists():
        out.append("_(no `lopo/summary.json` in campaign dir — Phase 4 did not run)_\n")
        return "\n".join(out)
    try:
        summary = json.loads(summary_p.read_text())
    except Exception:
        out.append("_(failed to parse lopo/summary.json)_\n")
        return "\n".join(out)
    modes = summary.get("channel_modes") or ["both"]
    agg = summary.get("aggregate_per_mode") or {}

    # Per-position rows: parse folds.jsonl if available, else show aggregate
    # only.
    folds: list[dict] = []
    if folds_p.exists():
        for line in folds_p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                folds.append(json.loads(line))
            except Exception:
                continue

    if folds:
        # Group folds by position; columns per mode.
        by_pos: dict[tuple, dict[str, dict]] = defaultdict(dict)
        for f in folds:
            p = tuple(f.get("position_xyz", []))
            by_pos[p][f.get("channel_mode", "?")] = f
        cols = ["position (m)", "n_train_w", "n_val_w"] + [f"{m} MAE (m)" for m in modes]
        out += ["| " + " | ".join(cols) + " |",
                "|" + "|".join(["---"] * len(cols)) + "|"]
        for p in sorted(by_pos.keys()):
            row = by_pos[p]
            ref = next(iter(row.values()))
            cells = [
                f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})" if p else "?",
                _fmt(ref.get("n_train_windows"), prec=0),
                _fmt(ref.get("n_val_windows"), prec=0),
            ]
            for m in modes:
                f = row.get(m)
                if f is None or "error" in f:
                    cells.append("—")
                else:
                    cells.append(_fmt(f.get("val_mae_3d_m")))
            out.append("| " + " | ".join(cells) + " |")
    else:
        out.append("_(folds.jsonl missing; aggregate only)_")
        out.append("")

    # Aggregate footer.
    out.append("")
    out.append("**Aggregate across positions (mean ± std):**")
    out.append("")
    out.append("| mode | n_folds | mean MAE (m) | std MAE (m) | min | max |")
    out.append("|---|---|---|---|---|---|")
    for m in modes:
        a = agg.get(m, {})
        out.append("| " + " | ".join([
            m,
            _fmt(a.get("n_folds"), prec=0),
            _fmt(a.get("mean_mae_m")),
            _fmt(a.get("std_mae_m")),
            _fmt(a.get("min_mae_m")),
            _fmt(a.get("max_mae_m")),
        ]) + " |")
    return "\n".join(out) + "\n"


def _table_v4_position_breakdown(rows: list[dict]) -> str:
    """Table 12 — Per-position MAE on the 5-position holdout (V4 winner).

    Drawn from any row whose metrics carries `val_position_breakdown`
    (added in v4_trainer's final-eval pass).  Selects the row with the
    lowest `holdout_mae_ungated_m` as the winner and prints its
    per-position breakdown with a fail flag for MAE > 0.5 m.
    """
    out = ["## Table 12 — Per-position MAE on 5-position holdout (winner)", ""]
    candidates = [
        r for r in rows
        if isinstance(r["metrics"].get("val_position_breakdown"), dict)
        and r["metrics"]["val_position_breakdown"]
        and r["metrics"].get("holdout_mae_ungated_m") is not None
    ]
    if not candidates:
        out.append("_(no row carries val_position_breakdown — re-run v4_deep_sweep "
                   "with the updated trainer to populate it)_\n")
        return "\n".join(out)
    winner = min(candidates, key=lambda r: r["metrics"]["holdout_mae_ungated_m"])
    pb: dict = winner["metrics"]["val_position_breakdown"]
    cell_id = winner["cell"]
    seed = winner["cfg"].get("seed", "?")
    out.append(f"Winner: **{cell_id}** (seed {seed}) — "
               f"holdout MAE {_fmt(winner['metrics'].get('holdout_mae_ungated_m'))} m")
    out.append("")
    cols = ["position (m)", "n_windows", "MAE (m)", "p95 (m)", "n_outliers (>0.5m)", "fail"]
    out += ["| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|"]
    for key in sorted(pb.keys()):
        entry = pb[key]
        mae = entry.get("mae_3d")
        fail = "YES" if (isinstance(mae, (int, float)) and mae > 0.5) else ""
        out.append("| " + " | ".join([
            key,
            _fmt(entry.get("n"), prec=0),
            _fmt(mae),
            _fmt(entry.get("p95_3d")),
            _fmt(entry.get("n_outliers_gt_0_5m"), prec=0),
            fail,
        ]) + " |")
    return "\n".join(out) + "\n"


def _table_cross_dataset(campaign_dir: Path | None) -> str:
    """Table 13 — Cross-dataset transfer (V4 winner)."""
    out = ["## Table 13 — Cross-dataset transfer (V4 winner)", ""]
    if campaign_dir is None:
        out.append("_(no campaign_dir set; cross-dataset summary not loaded)_\n")
        return "\n".join(out)
    summary_p = campaign_dir / "cross_dataset" / "summary.json"
    if not summary_p.exists():
        out.append("_(no `cross_dataset/summary.json` in campaign dir — Phase 5 did not run)_\n")
        return "\n".join(out)
    try:
        summary = json.loads(summary_p.read_text())
    except Exception:
        out.append("_(failed to parse cross_dataset/summary.json)_\n")
        return "\n".join(out)
    modes = summary.get("channel_modes") or ["both"]
    directions = summary.get("directions") or {}
    cols = ["direction", "n_train_pos", "n_test_pos"] + [f"{m} MAE (m)" for m in modes]
    out += ["| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|"]
    for label in sorted(directions.keys()):
        d = directions[label]
        if "error" in d:
            out.append(f"| {label} | — | — | " + " | ".join(["—"] * len(modes)) + " |")
            continue
        cells = [
            f"{label} ({', '.join(d.get('train_dataset_ids', []))} → "
            f"{', '.join(d.get('test_dataset_ids', []))})",
            _fmt(d.get("n_train_positions"), prec=0),
            _fmt(d.get("n_test_positions"), prec=0),
        ]
        per_mode = d.get("per_channel_mode") or {}
        for m in modes:
            entry = per_mode.get(m, {})
            cells.append(_fmt(entry.get("val_mae_3d_m")))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _guidance(rows: list[dict]) -> str:
    return (
        "## Selection guidance\n\n"
        "- **Phase 2 winner:** pick the cell in Table 3 (Phase 2 grid) with the highest V2 RQ1 NMI AND positive modality-probe Δ.  Cells whose Δ is more negative than baseline_v2 are rejected even if their NMI is higher (acoustic-only collapse).\n"
        "- **Phase 3 winner:** repeat over Table 3 (Phase 3 grid) with the Phase 2 winner already baked into the base cell.\n"
        "- **Phase 4 promotion:** top 3 cells from Phases 2+3 by V2 tuple → re-run without `--skip-v3 --skip-v4` so Tables 2 and 4 fill in.\n"
        "- **Phase 5 verdict:** the Phase 4 winner re-run at seeds {1337, 2024}; Table 4 should show the cell beating baseline_v2 by > 2× the seed std on V3 NLL Δ or V4 MAE.\n"
        "- **Deep-vs-simple framing (Table 2):** if `V3 wins` is False, the thesis primary anomaly result is KDE.  If `V4 wins` is False, the thesis primary localisation result is V0 multilateration.  Run the campaign anyway — the rest of the table is still a contribution.\n"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--filter", default=None,
                   help="Cell-id glob (e.g. 'p2_a1_*'); default: include all ablation runs.")
    p.add_argument(
        "--campaign-dir", default=None,
        help="Path to results/runs/campaign_<ts>; scope report to that campaign's cells only.",
    )
    args = p.parse_args()

    campaign_dir = Path(args.campaign_dir) if args.campaign_dir else None
    rows = _load_runs(args.filter, campaign_dir=campaign_dir)
    if not rows:
        scope = f"campaign {campaign_dir}" if campaign_dir else f"{RUNS_DIR}"
        print(f"No ablation run dirs found in {scope}.")
        return

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    source = f"{campaign_dir}" if campaign_dir else f"{RUNS_DIR}"
    report = [
        f"# Ablation Report ({timestamp})",
        "",
        f"Source: {source}",
        f"Cells loaded: {len(rows)}",
        f"Filter: `{args.filter or '(none)'}`",
        "",
        _table_v1_v2(rows),
        _table_deep_vs_simple(rows),
        _axis_grouping(rows),
        _multi_seed(rows),
        _table_v3_deep(rows),
        _table_v4_deep(rows),
        _table_acoustic(rows),
        _table_v3_paradigms(rows),
        _table_v4_channel_modes(rows),
        _table_v4_train_select(rows),
        _table_lopo(campaign_dir),
        _table_v4_position_breakdown(rows),
        _table_cross_dataset(campaign_dir),
        _guidance(rows),
    ]
    # When scoped to a campaign, write the report INSIDE the campaign dir so
    # multiple campaigns can coexist on disk without report-name collisions.
    if campaign_dir:
        out_path = campaign_dir / "ablation_report.md"
    else:
        out_path = RUNS_DIR / f"ablation_report_{timestamp}.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
