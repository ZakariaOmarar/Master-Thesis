"""One-command deep V3-first → V4 campaign driver.

Sequences the V3-first deep campaign:

  Phase 1 — deep V3 sweep (frozen encoder) → pick best V3 by gap-guarded
            real-anomaly F1.
  Phase 2 — deep V4 sweep gated by the Phase-1 V3 → pick best V4 by
            gap-guarded V3-gated holdout MAE.
  Phase 3 — multi-seed verdict on the V3 + V4 winners.
  Final  — emit a report.

V3 is tuned FIRST because V4 is gated by V3 (V4 only fires on V3-flagged
windows). All training is INDIVIDUAL against a frozen V1/V2 encoder — no
V1/V2 retraining per cell. Resume-safe via state.json, like
``run_ablation_campaign.py``.

Run on the GPU box (training is not viable on the laptop)::

    python -m scripts.campaigns.run_deep_v3v4_campaign --encoder-run results/runs/<best_encoder_dir>
    python -m scripts.campaigns.run_deep_v3v4_campaign --encoder-run <dir> --resume deepc_<ts>
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

# Cell catalogs (must match the sweep scripts).
PHASE1_V3_CELLS = (
    [f"v3_{d}_{w}" for d in ("d0", "d1", "d2", "d3") for w in ("w4", "w5", "w3")]
    + [f"v3_cap_{c}" for c in ("small", "base", "big")]
)
PHASE2_V4_CELLS = (
    [f"v4_{d}_{w}" for d in ("hd0", "hd1", "hd2", "hd3") for w in ("w4", "w5", "w3")]
    + [f"v4_cap_{c}" for c in ("small", "base", "big")]
    + [f"v4_{r}" for r in ("rs10", "rs20", "rs30")]
    + [f"v4_{p}_{s}" for p in ("pos1", "pos5", "pos10") for s in ("srp02", "srp10", "srp20")]
)
VERDICT_SEEDS = (1337, 2024, 7, 99)

_TIMEOUT_V3_S = 60 * 60
_TIMEOUT_V4_S = 40 * 60
# Phase 4 LOPO-CV trains V4 once per labelled position (~23 folds) on the
# winner cell, so the budget scales linearly with folds × channel modes.
# 4 modes × 23 folds × ~10 min/fold ≈ 15 h; bound generously.
_TIMEOUT_LOPO_S = 16 * 3600
# Phase 5 cross-dataset transfer: train + eval once per direction × per
# channel mode.  2 directions × 4 modes × ~15 min/run ≈ 2 h; bound to 4 h.
_TIMEOUT_CROSS_S = 4 * 3600


def _now() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


class Log:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"[{_dt.datetime.now():%H:%M:%S}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)
        with self.path.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(line + "\n")


def _save(state_path: Path, state: dict) -> None:
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(state_path)


def _find_run_dir(prefix: str, cell: str, seed: int,
                  train_select: str | None = None) -> Path | None:
    """Locate the newest run dir matching `prefix_cell_sSEED[_tsMODE]`.

    `train_select` disambiguates the v4deep_<cell>_s42 family: impulse uses the
    bare suffix, non-impulse modes have `_ts<MODE>` appended (matches the
    convention in v4_deep_sweep.py).  Passing None matches the impulse/default
    naming.
    """
    suffix = "" if train_select in (None, "impulse") else f"_ts{train_select}"
    cands = sorted(RUNS_DIR.glob(f"*__{prefix}_{cell}_s{seed}{suffix}"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _load_metrics(run_dir: Path | None) -> dict | None:
    if run_dir is None:
        return None
    mp = run_dir / "metrics.json"
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except Exception:
        return None


def _launch(cmd: list[str], timeout: float, log: Log) -> str:
    log(f"  $ {' '.join(cmd[2:])}")
    try:
        rc = subprocess.run(cmd, cwd=str(REPO), timeout=timeout, check=False)
        return "completed" if rc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        log("  TIMEOUT")
        return "timeout"


# ---------------------------------------------------------------------------
# Selection (gap-guarded — gap is a guardrail, not the objective)
# ---------------------------------------------------------------------------


def pick_v3_winner(results: dict[str, dict | None], gap_guardrail: float) -> str | None:
    """Max real-anomaly F1 (tie-break synthetic-AUC@5dB), subject to NLL gap
    ≤ guardrail. If no cell passes the guardrail, fall back to the smallest
    gap among the top-F1 third (so we never return nothing)."""
    scored: list[tuple[float, float, float, str]] = []  # (-f1, -auc5, gap, cell)
    for cell, m in results.items():
        if not m:
            continue
        fus = (m.get("paradigms") or {}).get("fusion") or {}
        ra = fus.get("real_anomaly") or {}
        f1 = ra.get("f1")
        gap = fus.get("nll_gap")
        if f1 is None or gap is None:
            continue
        auc = (fus.get("synthetic_auc") or {})
        auc5 = auc.get(5.0, auc.get("5.0", 0.0)) if isinstance(auc, dict) else 0.0
        scored.append((-float(f1), -float(auc5), float(gap), cell))
    if not scored:
        return None
    passing = [t for t in scored if t[2] <= gap_guardrail]
    pool = passing if passing else sorted(scored)[: max(1, len(scored) // 3)]
    pool.sort()
    return pool[0][3]


def pick_v4_winner(results: dict[str, dict | None], gap_guardrail_m: float) -> str | None:
    """Min V3-gated holdout MAE (fallback ungated), subject to train/val gap
    ≤ guardrail_m. Fallback to smallest-gap among top-MAE third."""
    scored: list[tuple[float, float, str]] = []  # (mae, gap, cell)
    for cell, m in results.items():
        if not m:
            continue
        mae = m.get("holdout_mae_v3gated_m")
        if mae is None:
            mae = m.get("holdout_mae_ungated_m")
        gap = m.get("train_val_gap_m")
        if mae is None or gap is None:
            continue
        scored.append((float(mae), float(gap), cell))
    if not scored:
        return None
    passing = [t for t in scored if t[1] <= gap_guardrail_m]
    pool = passing if passing else sorted(scored)[: max(1, len(scored) // 3)]
    pool.sort()
    return pool[0][2]


def _run_phase(prefix: str, cells: list[str], base_cmd, timeout: float,
               state: dict, state_path: Path, log: Log, deadline: float,
               seed: int = 42) -> dict[str, dict | None]:
    out: dict[str, dict | None] = {}
    for cell in cells:
        key = f"{prefix}:{cell}:s{seed}"
        prior = state.get("cells", {}).get(key)
        if prior and prior.get("status") == "completed":
            rd = Path(prior["run_dir"]) if prior.get("run_dir") else None
            m = _load_metrics(rd)
            if m is not None:
                log(f"  resume: {cell} done")
                out[cell] = m
                continue
        if time.time() > deadline:
            log(f"  BUDGET EXCEEDED — skipping {cell}")
            state.setdefault("cells", {})[key] = {"status": "skipped_budget"}
            _save(state_path, state)
            continue
        status = _launch(base_cmd(cell, seed), timeout, log)
        rd = _find_run_dir(prefix, cell, seed)
        out[cell] = _load_metrics(rd) if status == "completed" else None
        state.setdefault("cells", {})[key] = {
            "status": status, "run_dir": str(rd) if rd else None}
        _save(state_path, state)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-run", required=True,
                   help="Run dir with frozen v1/{acoustic,vibration}.pt + v2/encoder.pt")
    p.add_argument("--budget-hours", type=float, default=24.0)
    p.add_argument("--resume", default=None)
    p.add_argument("--v3-gap-guardrail", type=float, default=20.0,
                   help="Max |val_nll-train_nll| for a V3 cell to be eligible (NLL units).")
    p.add_argument("--v4-gap-guardrail-m", type=float, default=0.30,
                   help="Max |val_mae-train_mae| (m) for a V4 cell to be eligible.")
    args = p.parse_args()

    encoder_run = Path(args.encoder_run)
    if not (encoder_run / "v2" / "encoder.pt").exists():
        raise SystemExit(f"v2/encoder.pt not found under {encoder_run}")

    campaign_dir = (RUNS_DIR / args.resume) if args.resume else (RUNS_DIR / f"deepc_{_now()}")
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign_dir / "state.json"
    log = Log(campaign_dir / "campaign.log")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"cells": {}}
    state["encoder_run"] = str(encoder_run)
    state["started_at"] = state.get("started_at") or _now()
    _save(state_path, state)
    deadline = time.time() + args.budget_hours * 3600.0
    log(f"Deep V3→V4 campaign | encoder={encoder_run} | budget={args.budget_hours}h")

    # ---- Phase 1: deep V3 ----
    log("\n=== Phase 1 — deep V3 sweep ===")
    p1 = _run_phase(
        "v3deep", PHASE1_V3_CELLS,
        lambda cell, seed: [sys.executable, "-m", "scripts.sweeps.v3_deep_sweep",
                            "--encoder-run", str(encoder_run), "--cell", cell, "--seed", str(seed)],
        _TIMEOUT_V3_S, state, state_path, log, deadline)
    v3_winner = pick_v3_winner(p1, args.v3_gap_guardrail)
    log(f"Phase 1 V3 winner: {v3_winner}")
    state["v3_winner"] = v3_winner
    _save(state_path, state)
    if v3_winner is None:
        log("FATAL: no V3 winner — aborting.")
        return
    v3_winner_dir = _find_run_dir("v3deep", v3_winner, 42)

    # ---- Phase 2: deep V4 gated by best V3 ----
    # Shared sample cache so the 27 cells precompute V4 samples exactly once
    # (the first cell writes it, the rest load it) instead of 27× redundant
    # SRP-PHAT precompute.  Keyed by encoder so a different encoder-run won't
    # reuse a stale cache.
    samples_cache = campaign_dir / "v4_samples.pkl"
    log("\n=== Phase 2 — deep V4 sweep (gated by Phase-1 V3) ===")
    p2 = _run_phase(
        "v4deep", PHASE2_V4_CELLS,
        lambda cell, seed: [sys.executable, "-m", "scripts.sweeps.v4_deep_sweep",
                            "--encoder-run", str(encoder_run),
                            "--v3-run", str(v3_winner_dir),
                            "--samples-cache", str(samples_cache),
                            "--cell", cell, "--seed", str(seed)],
        _TIMEOUT_V4_S, state, state_path, log, deadline)
    v4_winner = pick_v4_winner(p2, args.v4_gap_guardrail_m)
    log(f"Phase 2 V4 winner: {v4_winner}")
    state["v4_winner"] = v4_winner
    _save(state_path, state)

    # ---- Phase 3: multi-seed verdict ----
    log("\n=== Phase 3 — multi-seed verdict ===")
    if v3_winner:
        for seed in VERDICT_SEEDS:
            if time.time() > deadline:
                log("BUDGET EXCEEDED in Phase 3 (V3 seeds)")
                break
            key = f"v3verdict:{v3_winner}:s{seed}"
            if state.get("cells", {}).get(key, {}).get("status") == "completed":
                continue
            # A4 fix (timeout): the verdict only needs fusion stability
            # across seeds (the per-paradigm story is in the seed-42 grid).
            # Dropping --all-paradigms cuts ~3× off per-seed wall time and
            # keeps the verdict inside _TIMEOUT_V3_S.
            status = _launch(
                [sys.executable, "-m", "scripts.sweeps.v3_deep_sweep",
                 "--encoder-run", str(encoder_run), "--cell", v3_winner,
                 "--seed", str(seed)],
                _TIMEOUT_V3_S, log)
            state.setdefault("cells", {})[key] = {
                "status": status, "run_dir": str(_find_run_dir("v3deep", v3_winner, seed))}
            _save(state_path, state)
    if v4_winner and v3_winner_dir is not None:
        for seed in VERDICT_SEEDS:
            if time.time() > deadline:
                log("BUDGET EXCEEDED in Phase 3 (V4 seeds)")
                break
            key = f"v4verdict:{v4_winner}:s{seed}"
            if state.get("cells", {}).get(key, {}).get("status") == "completed":
                continue
            status = _launch(
                [sys.executable, "-m", "scripts.sweeps.v4_deep_sweep",
                 "--encoder-run", str(encoder_run), "--v3-run", str(v3_winner_dir),
                 "--samples-cache", str(samples_cache),
                 "--all-channel-modes",
                 "--cell", v4_winner, "--seed", str(seed)],
                _TIMEOUT_V4_S, log)
            state.setdefault("cells", {})[key] = {
                "status": status, "run_dir": str(_find_run_dir("v4deep", v4_winner, seed))}
            _save(state_path, state)

    # ---- Phase 2b: training-window-selection ablation ----
    # Does HOW V4 training windows are selected matter?  Run the V4 winner at
    # all three selectors (impulse weak-GT / V3-gated / all-windows) so the
    # thesis can show whether train/serve skew (impulse-train vs V3-deploy) or
    # label noise (all-windows) actually affects the gated holdout MAE.
    if v4_winner and v3_winner_dir is not None:
        log("\n=== Phase 2b — V4 training-window-selection ablation ===")
        for ts_mode in ("impulse", "v3gated", "all"):
            if time.time() > deadline:
                log("BUDGET EXCEEDED in Phase 2b")
                break
            key = f"v4trainsel:{v4_winner}:{ts_mode}:s42"
            if state.get("cells", {}).get(key, {}).get("status") == "completed":
                continue
            status = _launch(
                [sys.executable, "-m", "scripts.sweeps.v4_deep_sweep",
                 "--encoder-run", str(encoder_run), "--v3-run", str(v3_winner_dir),
                 "--samples-cache", str(samples_cache),
                 "--train-select", ts_mode,
                 "--cell", v4_winner, "--seed", "42"],
                _TIMEOUT_V4_S, log)
            # Run-dir naming (A3 fix): v4_deep_sweep tags non-impulse modes
            # with a `_tsMODE` suffix so the three runs never collide.
            state.setdefault("cells", {})[key] = {
                "status": status,
                "run_dir": str(_find_run_dir("v4deep", v4_winner, 42, train_select=ts_mode)),
                "train_select": ts_mode}
            _save(state_path, state)

    # ---- Phase 4: LOPO-CV by position ----
    # Robustness of the V4 winner's holdout MAE across ALL labelled positions
    # (~23), not just the fixed 5-position holdout.  Runs once per channel
    # mode on the V4 winner only.  Output: <campaign_dir>/lopo/summary.json
    # (mean ± std MAE per mode) + folds.jsonl (per-position detail).
    if v4_winner:
        lopo_key = f"lopo:{v4_winner}:s42"
        if state.get("cells", {}).get(lopo_key, {}).get("status") != "completed":
            log("\n=== Phase 4 — LOPO-CV by position (V4 winner) ===")
            if time.time() > deadline:
                log("BUDGET EXCEEDED — skipping Phase 4")
            else:
                lopo_out = campaign_dir / "lopo"
                status = _launch(
                    [sys.executable, "-m", "src.modeling.orchestration.v4_lopo_cv",
                     "--encoder-run", str(encoder_run),
                     "--samples-cache", str(samples_cache),
                     "--all-channel-modes",
                     "--out-dir", str(lopo_out),
                     "--seed", "42"],
                    _TIMEOUT_LOPO_S, log)
                state.setdefault("cells", {})[lopo_key] = {
                    "status": status, "out_dir": str(lopo_out)}
                _save(state_path, state)

    # ---- Phase 5: cross-dataset transfer ----
    # Train on D1-D4, test on D5 only (and the reverse).  Confirms the
    # winner generalizes across recording sessions / rig states, not just
    # across positions within the same session.  V4 winner only.
    if v4_winner:
        cross_key = f"cross_dataset:{v4_winner}:s42"
        if state.get("cells", {}).get(cross_key, {}).get("status") != "completed":
            log("\n=== Phase 5 — Cross-dataset transfer (V4 winner) ===")
            if time.time() > deadline:
                log("BUDGET EXCEEDED — skipping Phase 5")
            else:
                cross_out = campaign_dir / "cross_dataset"
                status = _launch(
                    [sys.executable, "-m", "src.modeling.orchestration.v4_cross_dataset",
                     "--encoder-run", str(encoder_run),
                     "--samples-cache", str(samples_cache),
                     "--all-channel-modes",
                     "--out-dir", str(cross_out),
                     "--seed", "42"],
                    _TIMEOUT_CROSS_S, log)
                state.setdefault("cells", {})[cross_key] = {
                    "status": status, "out_dir": str(cross_out)}
                _save(state_path, state)

    # ---- Final report ----
    log("\n=== Final report ===")
    try:
        subprocess.run(
            [sys.executable, "-m", "scripts.campaigns.analyze_ablation",
             "--campaign-dir", str(campaign_dir)],
            cwd=str(REPO), timeout=600, check=False)
    except Exception as e:
        log(f"analyze_ablation failed: {type(e).__name__}: {e}")
    state["completed_at"] = _now()
    _save(state_path, state)
    log(f"Done. Winners: V3={v3_winner} V4={v4_winner}. "
        f"Report: {campaign_dir}/ablation_report.md")


if __name__ == "__main__":
    main()
