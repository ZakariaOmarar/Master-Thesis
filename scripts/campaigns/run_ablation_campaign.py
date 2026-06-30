"""One-command end-to-end ablation campaign driver.

Sequences all 57 cells of the ablation campaign — Phase 1 baseline →
Phase 2 sweep → Phases 3 / 7a / 7b sweeps (each conditioned on Phase 2
winner) → Phase 4 top-5 promotion to full pipeline → Phase 5 multi-seed
verdict → Phase 6 V4-only augmentation → conditional Phase 8 follow-up if
Phase 1 gates fail.

Single GPU; sequential cells.  No parallelism within the campaign, which
caps total wall-clock at ``--budget-hours``.

State is persisted to ``results/runs/campaign_<ts>/state.json`` after
every cell — re-running the driver with the same ``--resume <campaign_ts>``
flag picks up at the first incomplete cell.  Failed cells are logged
with status="failed" and skipped on resume (the campaign continues
rather than blocking on a single bad cell).

Run::

    python -m scripts.campaigns.run_ablation_campaign                 # fresh 48 h campaign
    python -m scripts.campaigns.run_ablation_campaign --budget-hours 24  # tighter cap
    python -m scripts.campaigns.run_ablation_campaign --resume campaign_20260522_120000
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "results" / "runs"


# ---------------------------------------------------------------------------
# Cell catalogs (must match scripts.campaigns.ablation_full_pipeline)
# ---------------------------------------------------------------------------

PHASE2_CELLS = [
    f"p2_a{a}_v{v}" for a in (0, 1, 2) for v in (3, 5, 7)
]
PHASE3_CELLS = [
    f"p3_m{m}_e{e}" for m in (0, 2, 4) for e in (32, 64, 128)
]
PHASE7A_CELLS = [
    f"p7a_t{t}_c{c}" for t in (0, 1, 2) for c in (0, 5, 10)
]
PHASE7B_CELLS = [
    f"p7b_lm{m}_lw{w}" for m in (3, 5, 7) for w in (1, 2, 3)
]
PHASE6_CELLS = [
    f"v4_pos{p}_srp{s}" for p in (1, 5, 10) for s in ("02", 10, 20)
]
PHASE5_SEEDS = (1337, 2024, 7, 99, 12345, 555)

# Hard timeouts per cell launch.  Generous — a stuck cell shouldn't burn
# more than ~2.5 h before the driver kills it and moves on.
_TIMEOUT_V1V2_ONLY_S = 60 * 60        # 1 h per V1+V2-only cell
_TIMEOUT_FULL_PIPELINE_S = 2.5 * 3600  # 2.5 h per full-pipeline cell
_TIMEOUT_V4_ONLY_S = 30 * 60          # 30 min per V4-only cell


# ---------------------------------------------------------------------------
# Logging + state persistence
# ---------------------------------------------------------------------------


def _now() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


class CampaignLog:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def __call__(self, msg: str) -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        with self.log_path.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(line + "\n")


def _save_state(state_path: Path, state: dict) -> None:
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(state_path)


# ---------------------------------------------------------------------------
# Cell launchers — each returns the path to the cell's run dir or None.
# ---------------------------------------------------------------------------


def _find_run_dir(cell_id: str, seed: int, skip_v3: bool, skip_v4: bool,
                  *, prefix: str = "ablation") -> Path | None:
    """Locate the most recent run dir matching the cell id + seed.

    `ablation_full_pipeline.py` and `v4_aug_sweep.py` both timestamp their
    output dirs, so we can't construct the path ahead of time — we glob
    after the subprocess completes and take the newest match.
    """
    if prefix == "ablation":
        suffix_bits = [f"s{seed}"]
        if skip_v3:
            suffix_bits.append("v3skip")
        if skip_v4 and not skip_v3:
            suffix_bits.append("v4skip")
        suffix = "_".join(suffix_bits)
        pattern = f"*__ablation_{cell_id}_{suffix}"
    elif prefix == "v4aug":
        pattern = f"*__v4aug_{cell_id}_s{seed}"
    else:
        return None
    candidates = sorted(RUNS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _launch_ablation_cell(
    cell_id: str, base_cell: str | None, seed: int,
    skip_v3: bool, skip_v4: bool, log: CampaignLog,
) -> tuple[str, Path | None]:
    """Run one `scripts.campaigns.ablation_full_pipeline` cell as a subprocess.

    Returns (status, run_dir).  Status is "completed", "failed", or "timeout".
    """
    timeout = _TIMEOUT_V1V2_ONLY_S if (skip_v3 and skip_v4) else _TIMEOUT_FULL_PIPELINE_S
    cmd = [
        sys.executable, "-m", "scripts.campaigns.ablation_full_pipeline",
        "--cell", cell_id, "--seed", str(seed),
    ]
    if base_cell:
        cmd += ["--base-cell", base_cell]
    if skip_v3:
        cmd.append("--skip-v3")
    if skip_v4:
        cmd.append("--skip-v4")
    log(f"  launching {cell_id} (seed={seed}, skip_v3={skip_v3}, skip_v4={skip_v4})")
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, cwd=str(REPO), timeout=timeout, check=False)
        status = "completed" if rc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {timeout}s on {cell_id}")
        return "timeout", None
    dt = time.time() - t0
    run_dir = _find_run_dir(cell_id, seed, skip_v3, skip_v4)
    log(f"  {cell_id} {status} in {dt:.0f}s — run_dir={run_dir}")
    return status, run_dir


def _launch_baseline_v2(seed: int, log: CampaignLog) -> tuple[str, Path | None]:
    """Phase 1: run the orchestrator's full_run module directly."""
    cmd = [sys.executable, "-m", "src.modeling.orchestration.full_run"]
    log("  launching baseline_v2 (full pipeline, seed=42 — orchestrator default)")
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, cwd=str(REPO), timeout=_TIMEOUT_FULL_PIPELINE_S, check=False)
        status = "completed" if rc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        log("  TIMEOUT on baseline_v2")
        return "timeout", None
    dt = time.time() - t0
    # baseline_v2 doesn't follow the ablation_* naming; it's the most recent
    # `full_pipeline_b5_cma` run.
    candidates = sorted(
        RUNS_DIR.glob("*__full_pipeline_b5_cma"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    run_dir = candidates[0] if candidates else None
    log(f"  baseline_v2 {status} in {dt:.0f}s — run_dir={run_dir}")
    return status, run_dir


def _launch_v4_aug_cell(
    cell_id: str, baseline_dir: Path, seed: int, log: CampaignLog,
) -> tuple[str, Path | None]:
    cmd = [
        sys.executable, "-m", "scripts.sweeps.v4_aug_sweep",
        "--baseline-run", str(baseline_dir),
        "--cell", cell_id, "--seed", str(seed),
    ]
    log(f"  launching v4 cell {cell_id}")
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, cwd=str(REPO), timeout=_TIMEOUT_V4_ONLY_S, check=False)
        status = "completed" if rc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT on {cell_id}")
        return "timeout", None
    dt = time.time() - t0
    run_dir = _find_run_dir(cell_id, seed, False, False, prefix="v4aug")
    log(f"  {cell_id} {status} in {dt:.0f}s")
    return status, run_dir


# ---------------------------------------------------------------------------
# Auto-selection helpers — programmatic winner picking.
# ---------------------------------------------------------------------------


def _load_metrics(run_dir: Path | None) -> dict | None:
    if run_dir is None:
        return None
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text())
    except Exception:
        return None


def _v1v2_score(metrics: dict) -> tuple[float, float, float, float] | None:
    """Compute the V1+V2 ranking tuple.  Returns None if the cell failed
    or has acoustic-only collapse (modality probe Δ < 0)."""
    if metrics is None:
        return None
    st = metrics.get("stages", {}) or metrics.get("full_run_stages", {})
    v2 = st.get("v2") or {}
    probe = st.get("v2_modality_probe") or {}
    v1a = st.get("v1_acoustic") or {}
    rq1_nmi = v2.get("rq1_nmi")
    delta = probe.get("delta_both_minus_acoustic")
    train = v2.get("train_loss_final")
    val = v2.get("val_loss_final")
    v1a_nmi = v1a.get("sanity_nmi", 0.0)
    if rq1_nmi is None or delta is None or train is None or val is None:
        return None
    if delta < 0:
        # Acoustic-only collapse — exclude from Phase 4 promotion.
        return None
    gap = abs(float(val) - float(train))
    # Sort key (later we sort ascending): primary is -rq1_nmi (so larger NMI
    # wins), secondary -delta, tertiary +gap, tie-break -v1a_nmi.
    return (-float(rq1_nmi), -float(delta), float(gap), -float(v1a_nmi))


def pick_v1v2_winner(
    cell_results: dict[str, tuple[Path | None, dict | None]],
) -> str | None:
    """Pick the highest-scoring V1+V2 cell.  Excludes acoustic-collapse cells."""
    scored: list[tuple[tuple, str]] = []
    for cell, (_run_dir, metrics) in cell_results.items():
        s = _v1v2_score(metrics)
        if s is None:
            continue
        scored.append((s, cell))
    if not scored:
        return None
    scored.sort()
    return scored[0][1]


def pick_top_n_for_full_pipeline(
    cell_results: dict[str, tuple[Path | None, dict | None]], n: int,
) -> list[str]:
    """Top n V1+V2 cells.  Deduplicates by hashing the cell_config.json's
    resolved cfg dict, so structurally-identical cells from different
    sweeps count once."""
    scored: list[tuple[tuple, str, str]] = []
    for cell, (run_dir, metrics) in cell_results.items():
        s = _v1v2_score(metrics)
        if s is None or run_dir is None:
            continue
        cfg_path = run_dir / "cell_config.json"
        try:
            cfg = json.loads(cfg_path.read_text())
            # Hash the V1/V2 cfg dict (V3/V4 cfgs differ between V1+V2-only
            # and full-pipeline runs, so they'd break deduplication).
            cfg_key = json.dumps(
                {"v1_cfg": cfg.get("v1_cfg"), "v2_cfg": cfg.get("v2_cfg")},
                sort_keys=True, default=str,
            )
        except Exception:
            cfg_key = cell  # fall back to cell id
        scored.append((s, cell, cfg_key))
    scored.sort()
    seen_keys: set[str] = set()
    picks: list[str] = []
    for _s, cell, cfg_key in scored:
        if cfg_key in seen_keys:
            continue
        seen_keys.add(cfg_key)
        picks.append(cell)
        if len(picks) >= n:
            break
    return picks


def _phase4_score(metrics: dict) -> tuple[float, float, float, float] | None:
    """Phase 4 full-pipeline ranking — primary is deep_vs_simple V4 Δ
    (negative = V4 beats V0 multilat).  See plan §Selection rules."""
    if metrics is None:
        return None
    # Phase 4 cells use ablation_full_pipeline which embeds full_run results
    # under `full_run_*` keys.
    dvs = metrics.get("full_run_deep_vs_simple") or metrics.get("deep_vs_simple") or {}
    loc = dvs.get("localisation") or {}
    anom = dvs.get("anomaly") or {}
    v4_delta = loc.get("delta_deep_minus_simple_m")
    v4_mae = loc.get("deep_val_mae_m")
    v3_delta = anom.get("delta_deep_minus_simple")
    # V3 synthetic AUC — pulled from the orchestrator's v3_fusion_depth.
    fr_stages = metrics.get("full_run_stages") or metrics.get("stages") or {}
    v3_depth = fr_stages.get("v3_fusion_depth") or {}
    syn = v3_depth.get("synthetic_anomaly_auc") or {}
    syn_auc = (syn.get("auc_conditional") or {}).get("5.0") if isinstance(syn.get("auc_conditional"), dict) else None
    if v4_delta is None or v4_mae is None:
        return None
    # Use large values for missing optional fields so they don't dominate
    # the tuple ranking when absent.
    v3d = float(v3_delta) if v3_delta is not None else 0.0
    aucv = float(syn_auc) if syn_auc is not None else 0.0
    return (float(v4_delta), v3d, float(v4_mae), -aucv)


def pick_phase4_winner(
    cell_results: dict[str, tuple[Path | None, dict | None]],
) -> str | None:
    scored: list[tuple[tuple, str]] = []
    for cell, (_run_dir, metrics) in cell_results.items():
        s = _phase4_score(metrics)
        if s is None:
            continue
        scored.append((s, cell))
    if not scored:
        return None
    scored.sort()
    return scored[0][1]


def check_phase1_gates(metrics: dict) -> dict[str, bool]:
    """Returns {gate_name: passed} for the three Phase 1 absolute gates."""
    dvs = metrics.get("full_run_deep_vs_simple") or metrics.get("deep_vs_simple") or {}
    loc = dvs.get("localisation") or {}
    fr_stages = metrics.get("full_run_stages") or metrics.get("stages") or {}
    v3_depth = fr_stages.get("v3_fusion_depth") or {}
    syn = v3_depth.get("synthetic_anomaly_auc") or {}
    auc_dict = syn.get("auc_conditional") if isinstance(syn.get("auc_conditional"), dict) else {}
    auc_5db = auc_dict.get("5.0") if isinstance(auc_dict, dict) else None
    v4_mae = loc.get("deep_val_mae_m")
    v4_delta = loc.get("delta_deep_minus_simple_m")
    return {
        "v3_synthetic_auc_gt_0.70": (auc_5db is not None and float(auc_5db) > 0.70),
        "v4_mae_lt_1.0m": (v4_mae is not None and float(v4_mae) < 1.0),
        "v4_vs_v0_delta_lt_0.10m": (v4_delta is not None and float(v4_delta) < 0.10),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_phase(
    phase_name: str,
    cells: list[str],
    state: dict,
    state_path: Path,
    log: CampaignLog,
    *,
    base_cell: str | None = None,
    seed: int = 42,
    skip_v3: bool = True,
    skip_v4: bool = True,
    budget_deadline: float | None = None,
) -> dict[str, tuple[Path | None, dict | None]]:
    """Run all cells in a phase sequentially.  Returns {cell: (run_dir, metrics)}
    for completed cells (failed/timeout cells map to (None, None))."""
    log(f"\n=== Phase {phase_name}: {len(cells)} cells ===")
    out: dict[str, tuple[Path | None, dict | None]] = {}
    for cell in cells:
        cell_key = f"{phase_name}:{cell}:s{seed}"
        prior = state.get("cells", {}).get(cell_key)
        if prior and prior.get("status") == "completed":
            run_dir = Path(prior["run_dir"]) if prior.get("run_dir") else None
            metrics = _load_metrics(run_dir)
            if metrics is not None:
                log(f"  resume: {cell} already completed ({run_dir})")
                out[cell] = (run_dir, metrics)
                continue
        if budget_deadline and time.time() > budget_deadline:
            log(f"  BUDGET EXCEEDED — skipping {cell} and remaining cells.")
            state.setdefault("cells", {})[cell_key] = {"status": "skipped_budget"}
            _save_state(state_path, state)
            continue
        status, run_dir = _launch_ablation_cell(
            cell, base_cell, seed, skip_v3, skip_v4, log,
        )
        metrics = _load_metrics(run_dir) if status == "completed" else None
        out[cell] = (run_dir, metrics)
        state.setdefault("cells", {})[cell_key] = {
            "status": status,
            "run_dir": str(run_dir) if run_dir else None,
        }
        _save_state(state_path, state)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget-hours", type=float, default=48.0)
    p.add_argument("--resume", default=None, help="campaign_<ts> dir name to resume.")
    p.add_argument("--skip-followup", action="store_true",
                   help="Disable conditional Phase 8 follow-up if Phase 1 gates fail.")
    args = p.parse_args()

    if args.resume:
        campaign_dir = RUNS_DIR / args.resume
        if not campaign_dir.exists():
            raise SystemExit(f"resume target not found: {campaign_dir}")
    else:
        campaign_dir = RUNS_DIR / f"campaign_{_now()}"
        campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign_dir / "state.json"
    log = CampaignLog(campaign_dir / "campaign.log")
    log(f"Campaign dir: {campaign_dir}")
    log(f"Budget: {args.budget_hours} h; skip_followup={args.skip_followup}")
    start_time = time.time()
    deadline = start_time + args.budget_hours * 3600.0

    state = json.loads(state_path.read_text()) if state_path.exists() else {"cells": {}}
    state["campaign_dir"] = str(campaign_dir)
    state["budget_hours"] = args.budget_hours
    state["started_at"] = state.get("started_at") or _now()
    _save_state(state_path, state)

    # ============================================================== Phase 1
    log("\n=== Phase 1: baseline_v2 ===")
    baseline_prior = state.get("baseline_v2")
    if baseline_prior and baseline_prior.get("status") == "completed":
        baseline_dir = Path(baseline_prior["run_dir"])
        log(f"  resume: baseline_v2 already at {baseline_dir}")
    else:
        status, baseline_dir = _launch_baseline_v2(42, log)
        state["baseline_v2"] = {
            "status": status, "run_dir": str(baseline_dir) if baseline_dir else None,
        }
        _save_state(state_path, state)
        if status != "completed" or baseline_dir is None:
            log("FATAL: baseline_v2 did not complete — aborting campaign.")
            return

    # ============================================================== Phase 2
    p2_results = run_phase(
        "p2", PHASE2_CELLS, state, state_path, log,
        skip_v3=True, skip_v4=True, budget_deadline=deadline,
    )
    p2_winner = pick_v1v2_winner(p2_results)
    log(f"\nPhase 2 winner: {p2_winner}")
    if p2_winner is None:
        log("FATAL: no valid Phase 2 winner — every cell either failed or "
            "had acoustic-only collapse.  Aborting subsequent phases.")
        return
    state["p2_winner"] = p2_winner
    _save_state(state_path, state)

    # ====================================================== Phases 3 / 7a / 7b
    p3_results = run_phase(
        "p3", PHASE3_CELLS, state, state_path, log,
        base_cell=p2_winner, skip_v3=True, skip_v4=True, budget_deadline=deadline,
    )
    p7a_results = run_phase(
        "p7a", PHASE7A_CELLS, state, state_path, log,
        base_cell=p2_winner, skip_v3=True, skip_v4=True, budget_deadline=deadline,
    )
    p7b_results = run_phase(
        "p7b", PHASE7B_CELLS, state, state_path, log,
        base_cell=p2_winner, skip_v3=True, skip_v4=True, budget_deadline=deadline,
    )

    # ============================================================== Phase 4
    log("\n=== Phase 4 selection ===")
    pooled = {**p2_results, **p3_results, **p7a_results, **p7b_results}
    top5 = pick_top_n_for_full_pipeline(pooled, n=5)
    log(f"Phase 4 promoting: {top5}")
    state["phase4_top5"] = top5
    _save_state(state_path, state)
    if not top5:
        log("FATAL: no valid cells to promote to Phase 4 — aborting.")
        return
    # Phase 4 = re-run each cell on full pipeline (no --skip flags).  Cells
    # picked from p3_/p7a_/p7b_ keep the same base_cell (p2_winner); a
    # p2_winner picked here uses no --base-cell.
    p4_results: dict[str, tuple[Path | None, dict | None]] = {}
    for cell in top5:
        base = p2_winner if cell.startswith(("p3_", "p7a_", "p7b_")) else None
        cell_key = f"p4:{cell}:s42"
        prior = state.get("cells", {}).get(cell_key)
        if prior and prior.get("status") == "completed":
            run_dir = Path(prior["run_dir"]) if prior.get("run_dir") else None
            metrics = _load_metrics(run_dir)
            if metrics is not None:
                log(f"  resume: phase4 {cell} already completed ({run_dir})")
                p4_results[cell] = (run_dir, metrics)
                continue
        if time.time() > deadline:
            log("BUDGET EXCEEDED in Phase 4 — emitting partial report.")
            break
        status, run_dir = _launch_ablation_cell(
            cell, base, 42, skip_v3=False, skip_v4=False, log=log,
        )
        metrics = _load_metrics(run_dir) if status == "completed" else None
        p4_results[cell] = (run_dir, metrics)
        state.setdefault("cells", {})[cell_key] = {
            "status": status, "run_dir": str(run_dir) if run_dir else None,
        }
        _save_state(state_path, state)

    # ============================================================== Phase 5
    p4_winner = pick_phase4_winner(p4_results)
    log(f"\nPhase 4 winner (Phase 5 seed lead): {p4_winner}")
    state["p4_winner"] = p4_winner
    _save_state(state_path, state)
    winner_metrics = p4_results.get(p4_winner, (None, None))[1] if p4_winner else None

    if p4_winner and winner_metrics is not None:
        base = p2_winner if p4_winner.startswith(("p3_", "p7a_", "p7b_")) else None
        for seed in PHASE5_SEEDS:
            cell_key = f"p5:{p4_winner}:s{seed}"
            prior = state.get("cells", {}).get(cell_key)
            if prior and prior.get("status") == "completed":
                log(f"  resume: phase5 {p4_winner} s{seed} already completed")
                continue
            if time.time() > deadline:
                log("BUDGET EXCEEDED in Phase 5 — emitting partial report.")
                break
            status, run_dir = _launch_ablation_cell(
                p4_winner, base, seed, skip_v3=False, skip_v4=False, log=log,
            )
            state.setdefault("cells", {})[cell_key] = {
                "status": status, "run_dir": str(run_dir) if run_dir else None,
            }
            _save_state(state_path, state)

    # ============================================================== Phase 6
    log("\n=== Phase 6: V4-only augmentation sweep ===")
    for cell in PHASE6_CELLS:
        cell_key = f"p6:{cell}:s42"
        prior = state.get("cells", {}).get(cell_key)
        if prior and prior.get("status") == "completed":
            log(f"  resume: p6 {cell} already completed")
            continue
        if time.time() > deadline:
            log("BUDGET EXCEEDED in Phase 6 — emitting partial report.")
            break
        status, run_dir = _launch_v4_aug_cell(cell, baseline_dir, 42, log)
        state.setdefault("cells", {})[cell_key] = {
            "status": status, "run_dir": str(run_dir) if run_dir else None,
        }
        _save_state(state_path, state)

    # ============================================================== Phase 8
    if not args.skip_followup and winner_metrics is not None:
        gates = check_phase1_gates(winner_metrics)
        log(f"\nPhase 1 gates on winner: {gates}")
        state["phase1_gates"] = gates
        _save_state(state_path, state)
        failed = [g for g, ok in gates.items() if not ok]
        if failed and time.time() < deadline:
            log(f"=== Phase 8 follow-up triggered (failed gates: {failed}) ===")
            # V4-related failures → re-run v4_aug_sweep against Phase 4 winner's
            # V1/V2/V3 (not baseline_v2).  V3-related failures → no auto cell
            # is defined; flagged in report.
            if any("v4" in g for g in failed):
                p4_winner_dir = p4_results.get(p4_winner, (None, None))[0]
                if p4_winner_dir is not None:
                    for cell in PHASE6_CELLS:
                        cell_key = f"p8:{cell}:s42"
                        if state.get("cells", {}).get(cell_key, {}).get("status") == "completed":
                            continue
                        if time.time() > deadline:
                            log("BUDGET EXCEEDED in Phase 8 — stopping.")
                            break
                        status, run_dir = _launch_v4_aug_cell(cell, p4_winner_dir, 42, log)
                        state.setdefault("cells", {})[cell_key] = {
                            "status": status,
                            "run_dir": str(run_dir) if run_dir else None,
                            "trigger": failed,
                        }
                        _save_state(state_path, state)

    # =========================================================== Final report
    log("\n=== Generating final report ===")
    report_cmd = [
        sys.executable, "-m", "scripts.campaigns.analyze_ablation",
        "--campaign-dir", str(campaign_dir),
    ]
    try:
        subprocess.run(report_cmd, cwd=str(REPO), timeout=600, check=False)
    except Exception as e:
        log(f"analyze_ablation failed: {type(e).__name__}: {e}")
    state["completed_at"] = _now()
    state["wall_clock_hours"] = (time.time() - start_time) / 3600.0
    _save_state(state_path, state)
    log(f"\nCampaign complete in {state['wall_clock_hours']:.2f} h.  Report: "
        f"{campaign_dir}/ablation_report.md (or check most recent ablation_report_*.md)")


if __name__ == "__main__":
    main()
