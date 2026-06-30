"""One-command driver to regenerate the thesis result set on a laptop.

Runs every orchestration the Results chapter depends on, in order, with proper
streamed logging (no buffering surprises), seed 42, and resume-by-skip. Point it
at an existing encoder run to skip the multi-hour full pipeline.

Usage (one command)::

    # Full regeneration from scratch (trains V1/V2/V3/V4 - several hours):
    python -m scripts.run_all_remaining

    # Reuse an encoder run you already have (skips the heavy training):
    python -m scripts.run_all_remaining --encoder-run results/runs/20260608_123657__full_pipeline_b5_cma

    # Re-do even steps whose outputs already exist:
    python -m scripts.run_all_remaining --encoder-run <dir> --force

What each step produces, and the audit item it closes:

    1. full_run --v0-baselines   RQ1/RQ2/RQ3-LORO + V0 baselines, one consistent
                                  set; per-cluster calibration (#1), crossfade
                                  cohorts (#6), SNR-ladder CIs (#8), localization
                                  conditioning ablation (#11).
    2. v4_lopo_cv                 leave-one-position-out, 4 channel modes (#2 base).
    3. v4_cross_dataset           cross-session transfer, 4 channel modes.
    4. head_domain_shift_fpr      RQ2 domain-shift FPR of the heads (tab:res_rq2_shift).
    5. run_v0_anomaly             V0 anomaly recall + FPR (per-cohort).
    6. assemble_comparison        joins head-vs-baseline numbers per RQ.

    7. finalize_results        post-processes saved outputs into per-position (#2),
                               LOPO bootstrap CIs (#4), accel-TDOA CI + the
                               training-free classical LORO number (#7,#12),
                               reg-grid recall (#9). No GPU; reads JSON only.
    8. localization_baselines  SRP-PHAT bootstrap CIs (#12).

Also produced as a by-product of the steps above: late-fusion paradigm in LOPO
(#15, inside v4_lopo_cv) and event-level F1 per pipeline (#16, inside
head_domain_shift_fpr). All training stages use device="auto" -> CUDA when present.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG_DIR = REPO / "results" / "logs"


def _newest_full_run() -> Path | None:
    cands = sorted(
        glob.glob(str(REPO / "results" / "runs" / "*__full_pipeline_b5_cma")),
        key=os.path.getmtime,
        reverse=True,
    )
    for c in cands:
        if (Path(c) / "metrics.json").exists():
            return Path(c)
    return None


def _run(name: str, cmd: list[str], log) -> int:
    """Stream a subprocess to console + log file; return its exit code."""
    banner = f"\n{'='*70}\n[{_dt.datetime.now():%H:%M:%S}] STEP: {name}\n  $ {' '.join(cmd)}\n{'='*70}"
    print(banner, flush=True)
    log.write(banner + "\n"); log.flush()
    proc = subprocess.Popen(
        cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},  # sub-scripts emit utf-8
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder-run", type=Path, default=None,
                    help="Existing full_pipeline run dir (v1/ + v2/); skips step 1.")
    ap.add_argument("--force", action="store_true", help="Re-run steps even if outputs exist.")
    ap.add_argument("--quick", action="store_true", help="Pass --quick to every stage (smoke).")
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="If given (>1 seed), also run the headline multi-seed sweep "
             "(mean +/- std for the stability numbers in section 6.7). "
             "Heavy: one full pipeline run per seed. Example: --seeds 42 1337 2024 7 99.",
    )
    args = ap.parse_args(argv)

    # Enable the result-neutral feature cache for every subprocess (full_run +
    # the CV drivers) so the CWT/mel/vibration stacks are extracted once and
    # reused across stages AND across seeds.  Keyed on input bytes + all params
    # (incl. defaults) + source hash, so it can only speed things up, never
    # change a number.  Set HYDRO_FEATURE_CACHE_DIR= (empty) to disable.
    os.environ.setdefault("HYDRO_FEATURE_CACHE_DIR", str(REPO / ".feature_cache"))

    # Make this driver's own console output encoding-proof on a cp1252 terminal.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_all_remaining_{ts}.log"
    log = log_path.open("w", encoding="utf-8")

    # GPU banner - every training stage uses device="auto", which picks CUDA when
    # present.  Surface it up front so a silent CPU fallback is impossible to miss.
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
            banner = f"GPU DETECTED: {n}x [{names}] - all training stages will run on CUDA."
        else:
            banner = ("*** WARNING: no CUDA device detected - this run will use the CPU "
                      "(very slow). Check your GPU driver / `torch` CUDA build before proceeding. ***")
    except Exception as e:
        banner = f"*** WARNING: could not query torch for GPU ({e}); proceeding. ***"
    print(banner, flush=True)
    log.write(banner + "\n"); log.flush()
    q = ["--quick"] if args.quick else []
    results: list[tuple[str, str]] = []

    def record(name: str, rc: int) -> None:
        results.append((name, "OK" if rc == 0 else f"FAILED ({rc})"))

    # --- Step 1: full pipeline (skip if an encoder run is supplied) ---------
    enc = args.encoder_run
    if enc is None:
        rc = _run("1/6 full_run (RQ1/RQ2/RQ3-LORO + V0)",
                  [PY, "-m", "src.modeling.orchestration.full_run", "--v0-baselines", *q], log)
        record("1 full_run", rc)
        enc = _newest_full_run()
        if rc != 0 or enc is None:
            print("\nfull_run failed or produced no run dir; stopping (later steps need its encoder).")
            log.close(); return 1
    else:
        enc = enc.resolve()
        if not (enc / "metrics.json").exists():
            print(f"--encoder-run {enc} has no metrics.json; is it a finished full_pipeline run?")
            log.close(); return 1
        results.append(("1 full_run", f"SKIPPED (reusing {enc.name})"))
    print(f"\nUsing encoder run: {enc}", flush=True); log.write(f"encoder_run={enc}\n")

    # --- Step 1b: strict RQ1 NMI (the Chapter-6 headline, not the sanity gate) ---
    # The in-pipeline NMI is the model-selection sanity cohort; this reports the
    # strict number on all labelled D1+D2 windows -> <enc>/rq1_strict_nmi.json.
    rc = _run("1b rq1_strict_nmi (strict RQ1 NMI headline)",
              [PY, "-m", "scripts.baselines.rq1_strict_nmi", "--run-dir", str(enc)], log)
    record("1b rq1_strict_nmi", rc)

    # --- Step 2: leave-one-position-out CV ----------------------------------
    lopo_out = enc / "lopo"
    if args.force or not (lopo_out / "summary.json").exists():
        rc = _run("2/6 v4_lopo_cv (LOPO, 4 channel modes)",
                  [PY, "-m", "src.modeling.orchestration.v4_lopo_cv",
                   "--encoder-run", str(enc), "--all-channel-modes",
                   "--out-dir", str(lopo_out), "--seed", "42", *q], log)
        record("2 v4_lopo_cv", rc)
    else:
        results.append(("2 v4_lopo_cv", "SKIPPED (summary.json exists)"))

    # --- Step 3: cross-session transfer -------------------------------------
    cross_out = enc / "cross_dataset"
    if args.force or not (cross_out / "summary.json").exists():
        rc = _run("3/6 v4_cross_dataset (cross-session, 4 channel modes)",
                  [PY, "-m", "src.modeling.orchestration.v4_cross_dataset",
                   "--encoder-run", str(enc), "--all-channel-modes",
                   "--out-dir", str(cross_out), "--seed", "42", *q], log)
        record("3 v4_cross_dataset", rc)
    else:
        results.append(("3 v4_cross_dataset", "SKIPPED (summary.json exists)"))

    # --- Step 3b: leave-one-recording-out CV (LORO headline, fusion) ---------
    if args.force or not (enc / "v4_loocv.json").exists():
        rc = _run("3b v4_loocv (leave-one-recording-out CV)",
                  [PY, "-m", "src.modeling.orchestration.v4_loocv",
                   "--encoder-run", str(enc), *q], log)
        record("3b v4_loocv", rc)
    else:
        results.append(("3b v4_loocv", "SKIPPED (v4_loocv.json exists)"))

    # --- Step 3c: four-paradigm comparison (leave-one-recording-out CV) ------
    para_out = enc / "paradigms"
    if args.force or not (para_out / "metrics.json").exists():
        rc = _run("3c run_v4_three_paradigms (LORO paradigm comparison)",
                  [PY, "-m", "scripts.paradigms.run_v4_three_paradigms",
                   "--encoder-run", str(enc), "--out-dir", str(para_out), *q], log)
        record("3c run_v4_three_paradigms", rc)
    else:
        results.append(("3c run_v4_three_paradigms", "SKIPPED (paradigms/metrics.json exists)"))

    # --- Step 4: head domain-shift FPR (+ multi-seed shift sweep) ------------
    # The shift split is fragile (one split can land an easy/hard campaign in the
    # eval set), so sweep several split seeds for a mean/range -- symmetric with
    # step 9's V0 multiseed. The sweep reuses the single retrain, so it is free.
    head_seeds = ["42", "1337"] if args.quick else ["42", "1337", "2024", "7", "99"]
    rc = _run("4/6 head_domain_shift_fpr (RQ2 domain-shift + multi-seed sweep)",
              [PY, "-m", "scripts.baselines.head_domain_shift_fpr",
               "--from-run", str(enc), "--seed", "42", "--seeds", *head_seeds, *q], log)
    record("4 head_domain_shift_fpr", rc)

    # --- Step 5: V0 anomaly baselines ---------------------------------------
    rc = _run("5/6 run_v0_anomaly (V0 recall + FPR)",
              [PY, "-m", "scripts.baselines.run_v0_anomaly", *q], log)
    record("5 run_v0_anomaly", rc)

    # --- Step 6: assemble the head-vs-baseline comparison -------------------
    rc = _run("6/8 assemble_comparison",
              [PY, "-m", "scripts.baselines.assemble_comparison", "--run", str(enc)], log)
    record("6 assemble_comparison", rc)

    # --- Step 7: post-process saved outputs into breakdowns/CIs (no rerun) --
    rc = _run("7/8 finalize_results (per-position #2, LOPO-CI #4, accel-TDOA CI #12)",
              [PY, "-m", "scripts.finalize_results",
               "--lopo-dir", str(lopo_out), "--full-run", str(enc)], log)
    record("7 finalize_results", rc)

    # --- Step 8: classical localization baselines with bootstrap CIs (#12) --
    rc = _run("8/9 localization_baselines_ci (SRP-PHAT + accel-TDOA CIs)",
              [PY, "-m", "scripts.baselines.localization_baselines_ci", *q], log)
    record("8 localization_baselines_ci", rc)

    # --- Step 9: multi-seed V0 domain-shift (defensible OC-SVM robustness) ---
    ms_args = ["--seeds", "42", "1337"] if args.quick else []  # 2-seed smoke when --quick
    rc = _run("9/9 v0_domain_shift_multiseed (baseline shift-FPR over seeds)",
              [PY, "-m", "scripts.baselines.v0_domain_shift_multiseed", *ms_args], log)
    record("9 v0_domain_shift_multiseed", rc)

    # --- Step 10 (optional): headline multi-seed sweep (section 6.7 mean+/-std) ---
    # Heavy: re-runs the full pipeline once per seed.  The shift-FPR multi-seed
    # (step 4) and the V0 shift multi-seed (step 9) are already done above; this
    # adds the stability spread of the headline metrics (RQ1 NMI, RQ2 F1, RQ3 MAE)
    # into results/runs/multi_seed_summary.json.
    if args.seeds and len(args.seeds) > 1:
        rc = _run("10/11 multi_seed (train one full pipeline per seed)",
                  [PY, "-m", "src.modeling.orchestration.multi_seed",
                   "--seeds", *[str(s) for s in args.seeds], *q], log)
        record("10 multi_seed", rc)

        # --- Step 11: per-seed localization CV spread + cross-seed aggregate --
        # multi_seed trained one encoder per seed; run LOPO + cross-session on
        # each (the localization headline's seed spread) and aggregate into
        # results/reports/multiseed_complete_*.{json,md}.
        rc = _run("11/11 run_multiseed_complete (per-seed LOPO+cross + aggregate)",
                  [PY, "-m", "scripts.run_multiseed_complete",
                   "--seeds", *[str(s) for s in args.seeds], *q], log)
        record("11 run_multiseed_complete", rc)

    # --- Summary ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    for name, status in results:
        print(f"  {name:26} {status}")
    print(f"\n  log: {log_path.relative_to(REPO)}")
    print("\nAll audit items covered:")
    print("  backbone #1,#6,#8,#11 (full_run) | per-position #2 + LOPO CIs #4 (finalize)")
    print("  classical baseline+CIs #7,#12 (finalize + localization_baselines_ci)")
    print("  reg-grid recall #9 (finalize) | late-fusion in LOPO #15 (v4_lopo_cv)")
    print("  RQ2 alert table + event-F1 #16 + domain-shift #20 + head multi-seed shift sweep")
    print("       (head_domain_shift_fpr, live xt_pool)")
    print("  OC-SVM/baseline shift robustness over seeds (v0_domain_shift_multiseed)")
    print("  Outputs: results/reports/finalize_results_*.md, results/v0_anomaly/head_domain_shift_*.json,")
    print("           results/v0_anomaly/v0_domain_shift_multiseed_*.json")
    print("=" * 70)
    log.close()
    return 0 if all(s == "OK" or s.startswith("SKIPPED") for _, s in results) else 1


if __name__ == "__main__":
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    raise SystemExit(main())
