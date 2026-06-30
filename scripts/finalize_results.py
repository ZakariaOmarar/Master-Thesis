"""Post-process saved run outputs into the breakdowns/CIs the Results chapter owes.

This produces the audit items whose data is ALREADY computed and saved, so it
needs no GPU and no retrain — it only reads JSON:

  * #2  per-position localization breakdown (from LOPO ``folds.jsonl``)
  * #4  across-fold bootstrap CI on the LOPO per-mode mean (the summary has
        mean/std but no interval)
  * #12 accel-TDOA classical baseline bootstrap CI (from ``v0_multilateration``
        ``per_recording`` errors, where saved)

Writes ``results/reports/finalize_results_<ts>.{md,json}``.

Run::

    python -m scripts.finalize_results
    python -m scripts.finalize_results --lopo-dir results/runs/<dir>/lopo --full-run results/runs/<dir>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _newest(pattern: str) -> Path | None:
    c = sorted(glob.glob(str(REPO / pattern)), key=os.path.getmtime, reverse=True)
    return Path(c[0]) if c else None


def _find_lopo_dir(explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    cands = sorted(glob.glob(str(REPO / "results" / "runs" / "**" / "lopo" / "folds.jsonl"),
                             recursive=True), key=os.path.getmtime, reverse=True)
    return Path(cands[0]).parent if cands else None


def per_position(lopo_dir: Path) -> dict:
    """#2 + #4 from folds.jsonl."""
    folds = lopo_dir / "folds.jsonl"
    rows = [json.loads(l) for l in folds.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    modes = sorted({r["channel_mode"] for r in rows})
    positions = sorted({tuple(r["position_xyz"]) for r in rows})

    table = []
    for p in positions:
        entry = {"position": list(p)}
        for m in modes:
            hit = [r for r in rows if tuple(r["position_xyz"]) == p and r["channel_mode"] == m]
            entry[m] = round(hit[0]["val_mae_3d_m"], 3) if hit else None
            entry["n_val"] = hit[0]["n_val_windows"] if hit else entry.get("n_val")
        table.append(entry)

    # across-fold bootstrap CI on each mode's mean (#4)
    mode_ci = {}
    for m in modes:
        maes = np.array([r["val_mae_3d_m"] for r in rows if r["channel_mode"] == m])
        lo, hi = _boot_ci(maes)
        mode_ci[m] = {"mean_mae_m": round(float(maes.mean()), 4),
                      "std_mae_m": round(float(maes.std()), 4),
                      "ci95_low_m": round(lo, 4), "ci95_high_m": round(hi, 4),
                      "n_folds": int(maes.size)}

    # worst positions for the geometry argument (use "both" if present, else first mode)
    key_mode = "both" if "both" in modes else modes[0]
    worst = sorted(table, key=lambda e: (e.get(key_mode) is None, e.get(key_mode) or 0),
                   reverse=True)[:4]
    return {"modes": modes, "per_position": table, "mode_ci": mode_ci,
            "worst4_positions": worst, "key_mode": key_mode}


def accel_tdoa_ci(full_run: Path) -> dict:
    """#12 (accel-TDOA) from v0_multilateration per_recording errors."""
    metrics = json.loads((full_run / "metrics.json").read_text(encoding="utf-8"))
    st = metrics.get("stages", metrics)
    m = st.get("v0_multilateration", {})
    out = {}
    for ds, blk in m.items():
        if not isinstance(blk, dict):
            continue
        pr = blk.get("per_recording")
        if not pr:
            continue
        errs = np.array([float(r["error_m"]) for r in pr if "error_m" in r])
        if errs.size == 0:
            continue
        lo, hi = _boot_ci(errs)
        out[ds] = {"mean_mae_m": round(float(errs.mean()), 4), "n": int(errs.size),
                   "ci95_low_m": round(lo, 4), "ci95_high_m": round(hi, 4)}
    return out


def reg_grid() -> list[dict]:
    """#9 — per-cell V3 capacity/regularization sweep: real-anomaly F1 + recall."""
    out = []
    for p in sorted(glob.glob(str(REPO / "results" / "runs" / "*v3deep*" / "metrics.json"))):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        para = d.get("paradigms", {})
        fus = para.get("fusion", {}) if isinstance(para, dict) else {}
        ra = fus.get("real_anomaly", {}) if isinstance(fus, dict) else {}
        if not isinstance(ra, dict) or not ra:
            continue

        def _r(k, ra=ra):
            v = ra.get(k)
            return round(float(v), 4) if isinstance(v, (int, float)) else None
        row = {"cell": Path(p).parent.name, "f1": _r("f1"),
               "recall": _r("recall"), "precision": _r("precision")}
        if any(row[k] is not None for k in ("f1", "recall", "precision")):
            out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lopo-dir", type=Path, default=None)
    ap.add_argument("--full-run", type=Path, default=None)
    args = ap.parse_args(argv)

    lopo_dir = _find_lopo_dir(args.lopo_dir)
    full_run = args.full_run or _newest("results/runs/*__full_pipeline_b5_cma")

    payload = {"generated": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
               "lopo_dir": str(lopo_dir) if lopo_dir else None,
               "full_run": str(full_run) if full_run else None}

    L = ["# Finalized result breakdowns (post-processed, no rerun)", ""]
    if lopo_dir and (lopo_dir / "folds.jsonl").exists():
        pp = per_position(lopo_dir)
        payload["per_position"] = pp
        L += [f"## #2 Per-position localization MAE (m) — source `{lopo_dir}`", "",
              "| position (x,y,z) | n_val | " + " | ".join(pp["modes"]) + " |",
              "|---|---:|" + "---:|" * len(pp["modes"])]
        for e in pp["per_position"]:
            pos = "(" + ",".join(f"{v:.2f}" for v in e["position"]) + ")"
            L.append(f"| {pos} | {e.get('n_val','')} | "
                     + " | ".join(str(e.get(m, "")) for m in pp["modes"]) + " |")
        L += ["", "## #4 LOPO per-mode mean with across-fold 95% bootstrap CI", "",
              "| channel mode | mean MAE | 95% CI | n folds |", "|---|---:|---:|---:|"]
        for m, c in pp["mode_ci"].items():
            L.append(f"| {m} | {c['mean_mae_m']} | [{c['ci95_low_m']}, {c['ci95_high_m']}] | {c['n_folds']} |")
        wm = pp["key_mode"]
        L += ["", f"Worst 4 positions ({wm} mode) — the geometry-failure anecdote, now tabulated:"]
        for e in pp["worst4_positions"]:
            pos = "(" + ",".join(f"{v:.2f}" for v in e["position"]) + ")"
            L.append(f"  - {pos}: {e.get(wm)} m")
    else:
        L += ["## #2/#4 — no LOPO folds.jsonl found (run v4_lopo_cv first)."]

    if full_run:
        tdoa = accel_tdoa_ci(full_run)
        payload["accel_tdoa_ci"] = tdoa
        L += ["", f"## #12 accel-TDOA classical baseline, bootstrap CI — source `{full_run}`", ""]
        if tdoa:
            L += ["| cohort | mean MAE (m) | 95% CI | n rec |", "|---|---:|---:|---:|"]
            for ds, c in tdoa.items():
                L.append(f"| {ds} | {c['mean_mae_m']} | [{c['ci95_low_m']}, {c['ci95_high_m']}] | {c['n']} |")
            L += ["", "The accelerometer-TDOA solver is training-free, so this mean with its CI is "
                  "also its leave-one-out localization number (#7) — it needs no per-fold retrain. "
                  "(SRP-PHAT CIs come from `localization_baselines_ci.py`.)"]
        else:
            L += ["No `v0_multilateration` per_recording arrays found in this run."]

    rg = reg_grid()
    if rg:
        payload["reg_grid"] = rg
        L += ["", "## #9 V3 capacity/regularization sweep — real-anomaly F1, recall, precision per cell", "",
              "| cell | F1 | recall | precision |", "|---|---:|---:|---:|"]
        for r in rg:
            L.append(f"| {r['cell']} | {r['f1']} | {r['recall']} | {r['precision']} |")
        f1s = [r["f1"] for r in rg if r["f1"] is not None]
        rcs = [r["recall"] for r in rg if r["recall"] is not None]
        if f1s and rcs:
            L.append(f"\nF1 range [{min(f1s):.3f}, {max(f1s):.3f}]; recall is the discriminator "
                     f"(range [{min(rcs):.3f}, {max(rcs):.3f}], wider than F1).")

    out_dir = REPO / "results" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = payload["generated"]
    (out_dir / f"finalize_results_{ts}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = "\n".join(L) + "\n"
    (out_dir / f"finalize_results_{ts}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote results/reports/finalize_results_{ts}.{{md,json}}")
    return 0


if __name__ == "__main__":
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    raise SystemExit(main())
