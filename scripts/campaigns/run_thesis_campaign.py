"""Master thesis campaign — one command, end-to-end, produces every result.

Stages (run on the GPU box; training is not viable on the laptop):

  Stage A — Acoustic-improvement sweep (6 cells, V1+V2 only).
            Vibration is a settled dead-end for fusion, so the representation
            lever is the ACOUSTIC pathway.  Sweeps the two acoustic knobs the
            breadth campaign never touched — acoustic_cnn_width_mult × cwt_n_scales
            (R1a's wider CNN is worth re-testing now that overfitting is
            controlled).  Picks the best encoder by V2 mode-NMI + V1-acoustic
            NMI, gap-guarded.

  Stage B — Deep V3 → V4 campaign on that best encoder.  Delegates to
            ``run_deep_v3v4_campaign`` (V3 sweep → best V3 → V4 sweep gated by
            V3 → 5-seed verdict with per-modality V3 paradigms + per-modality
            V4 channel modes → report).

  Final  — one combined report over the whole campaign.

The result set this produces covers the full thesis:
  * RQ1 representation: acoustic-improved encoder, mode-NMI, acoustic-dominance.
  * RQ2 anomaly: V3 real-anomaly P/R/F1, V3-vs-simple-baseline, V3 three
    paradigms (acoustic/vibration/fusion), overfitting controlled.
  * RQ3 localization: V4 spatial-holdout V3-gated MAE vs V0, V4 four paradigms.
  * Robustness: 5-seed verdict (seed 42 + 4) on the V3 and V4 winners.

Run::

    python -m scripts.campaigns.run_thesis_campaign
    python -m scripts.campaigns.run_thesis_campaign --resume thesis_<ts>
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

ACOUSTIC_CELLS = [
    f"pa_{w}_{c}" for w in ("w1", "w2") for c in ("cwt16", "cwt32", "cwt64")
]
_TIMEOUT_V1V2_S = 75 * 60  # acoustic cells: V1+V2 only; width=2 is slower


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


def _find_run_dir(prefix: str, cell: str, seed: int) -> Path | None:
    cands = sorted(RUNS_DIR.glob(f"*__{prefix}_{cell}_s{seed}*"),
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


def pick_acoustic_winner(results: dict[str, tuple[Path | None, dict | None]]) -> str | None:
    """Best acoustic encoder by (V2 rq1_nmi, V1-acoustic sanity_nmi), with the
    V1/V2 train-val gap as a guardrail tiebreak.  Vibration metrics are
    deliberately ignored — the campaign's premise is acoustic-driven.
    """
    scored: list[tuple[float, float, float, str]] = []
    for cell, (_rd, m) in results.items():
        if not m:
            continue
        st = m.get("stages", {})
        v2 = st.get("v2") or {}
        v1a = st.get("v1_acoustic") or {}
        nmi = v2.get("rq1_nmi")
        v1nmi = v1a.get("sanity_nmi", 0.0)
        if nmi is None:
            continue
        gap = abs((v2.get("val_loss_final") or 0.0) - (v2.get("train_loss_final") or 0.0))
        scored.append((-float(nmi), -float(v1nmi), float(gap), cell))
    if not scored:
        return None
    scored.sort()
    return scored[0][3]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget-hours", type=float, default=30.0)
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    campaign_dir = (RUNS_DIR / args.resume) if args.resume else (RUNS_DIR / f"thesis_{_now()}")
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign_dir / "state.json"
    log = Log(campaign_dir / "campaign.log")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"cells": {}}
    state["started_at"] = state.get("started_at") or _now()
    _save(state_path, state)
    deadline = time.time() + args.budget_hours * 3600.0
    log(f"Thesis campaign | budget={args.budget_hours}h | dir={campaign_dir}")

    # ---- Stage A: acoustic-improvement sweep ----
    log("\n=== Stage A — acoustic-improvement sweep (V1+V2) ===")
    aco: dict[str, tuple[Path | None, dict | None]] = {}
    for cell in ACOUSTIC_CELLS:
        key = f"acoustic:{cell}:s42"
        prior = state.get("cells", {}).get(key)
        if prior and prior.get("status") == "completed":
            rd = Path(prior["run_dir"]) if prior.get("run_dir") else None
            aco[cell] = (rd, _load_metrics(rd))
            log(f"  resume: {cell} done")
            continue
        if time.time() > deadline:
            log("  BUDGET EXCEEDED in Stage A")
            break
        cmd = [sys.executable, "-m", "scripts.campaigns.ablation_full_pipeline",
               "--cell", cell, "--seed", "42", "--skip-v3", "--skip-v4"]
        log(f"  $ {' '.join(cmd[2:])}")
        try:
            rc = subprocess.run(cmd, cwd=str(REPO), timeout=_TIMEOUT_V1V2_S, check=False)
            status = "completed" if rc.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
        rd = _find_run_dir("ablation", cell, 42)
        aco[cell] = (rd, _load_metrics(rd) if status == "completed" else None)
        state.setdefault("cells", {})[key] = {"status": status, "run_dir": str(rd) if rd else None}
        _save(state_path, state)

    acoustic_winner = pick_acoustic_winner(aco)
    log(f"Acoustic winner: {acoustic_winner}")
    state["acoustic_winner"] = acoustic_winner
    _save(state_path, state)
    if acoustic_winner is None:
        log("FATAL: no acoustic winner — aborting.")
        return
    encoder_run = aco[acoustic_winner][0]
    state["encoder_run"] = str(encoder_run)
    _save(state_path, state)

    # ---- Stage B: deep V3 -> V4 campaign on the best encoder ----
    log("\n=== Stage B — deep V3 -> V4 campaign on acoustic winner ===")
    remaining_h = max(0.5, (deadline - time.time()) / 3600.0)
    deep_dir_name = f"deepc_{_now()}"
    cmd = [sys.executable, "-m", "scripts.campaigns.run_deep_v3v4_campaign",
           "--encoder-run", str(encoder_run),
           "--budget-hours", f"{remaining_h:.2f}",
           "--resume", deep_dir_name]
    log(f"  $ {' '.join(cmd[2:])}")
    state["deep_campaign_dir"] = deep_dir_name
    _save(state_path, state)
    try:
        subprocess.run(cmd, cwd=str(REPO), timeout=int(remaining_h * 3600) + 600, check=False)
    except subprocess.TimeoutExpired:
        log("  deep campaign timed out (budget).")

    # ---- Final combined report ----
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
    log(f"Thesis campaign complete. Acoustic winner={acoustic_winner}. "
        f"Deep results in results/runs/{deep_dir_name}/ablation_report.md ; "
        f"acoustic report in {campaign_dir}/ablation_report.md")


if __name__ == "__main__":
    main()
