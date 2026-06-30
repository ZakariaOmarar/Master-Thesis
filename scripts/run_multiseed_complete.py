"""One command to finish the multi-seed result set.

After a multi-seed sweep you have one full-pipeline run per encoder seed, but the
localization cross-validation (LOPO + cross-session) and the RQ1 floor/ceiling
were only ever produced for a single run.  This driver fills exactly those gaps
-- the runs the Chapter 5/6 audit flagged as still missing -- and then aggregates
everything into one median/spread report.  It does not re-run anything already
complete.

Per seed run (skipped when its output already exists; ``--force`` to redo):

    * rq1_strict_nmi      strict RQ1 NMI on all labelled D1+D2 windows  (CPU, fast)
    * v4_lopo_cv          leave-one-position-out CV, 4 channel modes    (GPU, ~35 min)
    * v4_cross_dataset    cross-session transfer, 4 channel modes        (GPU, ~5 min)
    * finalize_results    per-position + LOPO bootstrap CIs (JSON only, on the
                          reference seed)

Once (encoder-independent):

    * rq1_mode_refs       K-means floor + LightGBM ceiling for the RQ1 table

Finally:

    * aggregate_multiseed median [min, max] / mean+/-std across seeds ->
                          results/reports/multiseed_complete_<ts>.{json,md}

Already complete in your result set and therefore not re-run here:
run_v0_anomaly, head_domain_shift_fpr, v0_domain_shift_multiseed.

Usage::

    python -m scripts.run_multiseed_complete                 # auto-discover the seeds
    python -m scripts.run_multiseed_complete --dry-run       # print the plan, run nothing
    python -m scripts.run_multiseed_complete --runs A B C    # explicit run dirs
    python -m scripts.run_multiseed_complete --force         # redo even existing CV
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
RUNS_DIR = REPO / "results" / "runs"
V0_DIR = REPO / "results" / "v0_anomaly"
LOG_DIR = REPO / "results" / "logs"

# The thesis multi-seed set (Results.tex:31); stale one-off runs are ignored
# unless overridden with --seeds / --runs.
CANONICAL_SEEDS = (42, 1337, 2024, 7, 99)


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _seed_of(run: Path):
    man = _load(run / "manifest.json") or {}
    return (((man.get("configs") or {}).get("v3_cfg") or {}).get("seed"))


def discover_runs(seeds) -> dict:
    """Newest full-pipeline run per encoder seed (dedups re-runs of a seed),
    restricted to ``seeds`` so stale one-off runs are not swept in."""
    want = set(seeds)
    by_seed: dict = {}
    for c in glob.glob(str(RUNS_DIR / "*__full_pipeline_b5_cma")):
        p = Path(c)
        if not (p / "metrics.json").exists():
            continue
        seed = _seed_of(p)
        if seed is None or seed not in want:
            continue
        if seed not in by_seed or p.stat().st_mtime > by_seed[seed].stat().st_mtime:
            by_seed[seed] = p
    return dict(sorted(by_seed.items()))


def _run(name: str, cmd: list[str], log) -> int:
    banner = (f"\n{'='*70}\n[{_dt.datetime.now():%H:%M:%S}] STEP: {name}\n"
              f"  $ {' '.join(str(c) for c in cmd)}\n{'='*70}")
    print(banner, flush=True)
    log.write(banner + "\n"); log.flush()
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(REPO),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush()
        log.write(line); log.flush()
    proc.wait()
    tag = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    msg = f"[{_dt.datetime.now():%H:%M:%S}] {name}: {tag}"
    print(msg, flush=True); log.write(msg + "\n"); log.flush()
    return proc.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, default=None,
                    help="Explicit run dirs (default: newest run per canonical seed).")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help=f"Encoder seeds to include (default: {CANONICAL_SEEDS}).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run a step even if its output already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan (per seed: run vs skip) and exit.")
    ap.add_argument("--quick", action="store_true",
                    help="Pass --quick to the GPU CV stages (smoke test only).")
    ap.add_argument("--cv-seed", type=int, default=42,
                    help="Seed for the localization-head training inside each CV "
                         "(the encoder seed is fixed by the run). Default 42.")
    args = ap.parse_args(argv)

    # Share the result-neutral feature cache across every per-seed subprocess
    # (LOPO / cross-session reuse the CWT/mel/vibration stacks full_run already
    # extracted).  Keyed on bytes + all params + source hash → never changes a
    # number.  Set HYDRO_FEATURE_CACHE_DIR= (empty) to disable.
    os.environ.setdefault("HYDRO_FEATURE_CACHE_DIR", str(REPO / ".feature_cache"))

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # Resolve the encoder runs.
    if args.runs:
        runs = {(_seed_of(r) if _seed_of(r) is not None else r.name): r.resolve()
                for r in args.runs}
        runs = dict(sorted(runs.items(), key=lambda kv: str(kv[0])))
    else:
        runs = discover_runs(args.seeds or CANONICAL_SEEDS)
    if not runs:
        print("No full_pipeline_b5_cma run with metrics.json found under results/runs/.")
        return 1

    q = ["--quick"] if args.quick else []

    def needs(run: Path, rel: str) -> bool:
        return args.force or not (run / rel).exists()

    # ---- Plan -------------------------------------------------------------
    print("=" * 70)
    print("Multi-seed completion plan")
    print(f"  encoder runs ({len(runs)}):")
    for seed, run in runs.items():
        steps = []
        steps.append("strict-nmi" if needs(run, "rq1_strict_nmi.json") else "strict-nmi(skip)")
        steps.append("LOPO" if needs(run, "lopo/summary.json") else "LOPO(skip)")
        steps.append("cross" if needs(run, "cross_dataset/summary.json") else "cross(skip)")
        print(f"    seed {seed!s:<5} {run.name:42} -> {', '.join(steps)}")
    refs_exist = bool(glob.glob(str(V0_DIR / "rq1_mode_refs_*.json")))
    print(f"  rq1_mode_refs (floor+ceiling): "
          f"{'skip (exists)' if refs_exist and not args.force else 'run (once)'}")
    print("  finalize_results: reference seed | aggregate_multiseed: always")
    print("=" * 70)
    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = (LOG_DIR / f"run_multiseed_complete_{ts}.log").open("w", encoding="utf-8")

    try:
        import torch
        if torch.cuda.is_available():
            names = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
            banner = f"GPU DETECTED: [{names}] - CV stages will run on CUDA."
        else:
            banner = ("*** WARNING: no CUDA device - the LOPO/cross CV will run on CPU "
                      "(slow). Check your torch CUDA build before proceeding. ***")
    except Exception as e:
        banner = f"*** WARNING: could not query torch for GPU ({e}); proceeding. ***"
    print(banner, flush=True); log.write(banner + "\n"); log.flush()

    results: list[tuple[str, str]] = []

    def rec(name, rc):
        results.append((name, "OK" if rc == 0 else f"FAILED ({rc})"))

    # ---- Per-seed localization CV ----------------------------------------
    for seed, run in runs.items():
        if needs(run, "rq1_strict_nmi.json"):
            rec(f"seed{seed} rq1_strict_nmi",
                _run(f"seed {seed}: rq1_strict_nmi",
                     [PY, "-m", "scripts.baselines.rq1_strict_nmi", "--run-dir", run], log))
        else:
            results.append((f"seed{seed} rq1_strict_nmi", "SKIPPED (exists)"))

        if needs(run, "lopo/summary.json"):
            rec(f"seed{seed} v4_lopo_cv",
                _run(f"seed {seed}: v4_lopo_cv (LOPO, 4 modes)",
                     [PY, "-m", "src.modeling.orchestration.v4_lopo_cv",
                      "--encoder-run", run, "--all-channel-modes",
                      "--out-dir", run / "lopo", "--seed", str(args.cv_seed), *q], log))
        else:
            results.append((f"seed{seed} v4_lopo_cv", "SKIPPED (summary.json exists)"))

        if needs(run, "cross_dataset/summary.json"):
            rec(f"seed{seed} v4_cross_dataset",
                _run(f"seed {seed}: v4_cross_dataset (cross-session, 4 modes)",
                     [PY, "-m", "src.modeling.orchestration.v4_cross_dataset",
                      "--encoder-run", run, "--all-channel-modes",
                      "--out-dir", run / "cross_dataset", "--seed", str(args.cv_seed), *q], log))
        else:
            results.append((f"seed{seed} v4_cross_dataset", "SKIPPED (summary.json exists)"))

    # ---- RQ1 floor + ceiling (once, encoder-independent) ------------------
    if refs_exist and not args.force:
        results.append(("rq1_mode_refs", "SKIPPED (exists)"))
    else:
        rec("rq1_mode_refs",
            _run("rq1_mode_refs (K-means floor + LightGBM ceiling)",
                 [PY, "-m", "scripts.baselines.rq1_mode_refs"], log))

    # ---- finalize_results on the reference seed (cheap, JSON only) --------
    ref_run = runs.get(args.cv_seed) or next(iter(runs.values()))
    if (ref_run / "lopo" / "summary.json").exists():
        rec("finalize_results (reference seed)",
            _run(f"finalize_results (seed {_seed_of(ref_run)})",
                 [PY, "-m", "scripts.finalize_results",
                  "--lopo-dir", ref_run / "lopo", "--full-run", ref_run], log))

    # ---- Aggregate across seeds ------------------------------------------
    rec("aggregate_multiseed",
        _run("aggregate_multiseed (median +/- spread across seeds)",
             [PY, "-m", "scripts.aggregate_multiseed",
              "--runs", *[str(r) for r in runs.values()]], log))

    # ---- Summary ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    for name, status in results:
        print(f"  {name:32} {status}")
    print(f"\n  log: {(LOG_DIR / f'run_multiseed_complete_{ts}.log').relative_to(REPO)}")
    print("  report: results/reports/multiseed_complete_*.md")
    print("=" * 70)
    log.close()
    return 0 if all(s == "OK" or s.startswith("SKIPPED") for _, s in results) else 1


if __name__ == "__main__":
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    raise SystemExit(main())
