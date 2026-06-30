"""Run the full V1→V5 pipeline for multiple seeds and aggregate the
headline numbers (mean ± std) for the publication tables.

Usage:
    python -m src.modeling.orchestration.multi_seed \
        --quick   # optional, halves epoch counts
    # defaults to the five canonical thesis seeds; override with --seeds.

Each seed runs to completion and is archived under `results/runs/`.
After every seed finishes the script prints the running mean ± std for
the headline metrics so progress is visible during the (possibly hours-
long) sweep.

Final output: `results/runs/multi_seed_summary.json` with mean / std
per metric across all seeds.

Reproducibility: the thesis tables report a distribution over the five
seeds in :data:`THESIS_SEEDS`.  These are the documented defaults so that
``python -m src.modeling.orchestration.multi_seed`` reproduces the reported
multi-seed numbers without any extra arguments.  See ``REPRODUCING.md`` for
the full table → command → artifact mapping.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from .archive import ARCHIVE_ROOT
from .full_run import (
    main as full_run_main,
)

# Canonical thesis seed set.  Every five-seed "median [min, max]" table in the
# Results chapter is computed over exactly these seeds; they are the documented
# default so the reported numbers reproduce with no extra arguments.  Do not
# change without re-running and re-reporting every multi-seed table.
THESIS_SEEDS = (42, 1337, 2024, 7, 99)

HEADLINE_KEYS = (
    "v1_acoustic_purity",
    "v1_vibration_purity",
    "v2_purity",
    "v2_nmi",
    "v2_a1_purity",
    "v3_val_nll",
    "v3_a2_val_nll",
    "v4_mae_3d",
    "v4_p95_3d",
    "v4_a3_mae_3d",
    "v5_1_mae_3d",
)


def _override_seed(seed: int) -> dict:
    """Hot-patch the orchestrator's `_v*_cfg` helpers to use this seed."""
    import src.modeling.orchestration.full_run as fr

    original_v1 = fr.v1_config
    original_v2 = fr.v2_config
    original_v3 = fr.v3_config

    def v1(quick: bool):
        cfg = original_v1(quick)
        return cfg.__class__(**{**asdict(cfg), "seed": seed})

    def v2(quick: bool):
        cfg = original_v2(quick)
        return cfg.__class__(**{**asdict(cfg), "seed": seed})

    def v3(quick: bool):
        cfg = original_v3(quick)
        return cfg.__class__(**{**asdict(cfg), "seed": seed})

    fr.v1_config = v1
    fr.v2_config = v2
    fr.v3_config = v3
    return {"v1": original_v1, "v2": original_v2, "v3": original_v3}


def _restore(originals: dict) -> None:
    import src.modeling.orchestration.full_run as fr

    fr.v1_config = originals["v1"]
    fr.v2_config = originals["v2"]
    fr.v3_config = originals["v3"]


def main(seeds: list[int], quick: bool = False, run_v0_baselines: bool = False) -> dict:
    print(f"Multi-seed sweep: seeds={seeds}, quick={quick}, v0_baselines={run_v0_baselines}")
    per_seed_headlines: list[dict] = []
    per_seed_metrics: list[dict] = []
    sweep_start = time.time()

    for i, seed in enumerate(seeds):
        print(f"\n=== Seed {seed} ({i + 1}/{len(seeds)}) ===")
        t0 = time.time()
        originals = _override_seed(seed)
        try:
            # V0 reference baselines are deterministic references, not part of the
            # per-seed stability story, so run them only on the first seed (their
            # own shift-robustness sweep is `v0_domain_shift_multiseed`).
            metrics = full_run_main(
                quick=quick, run_v0_baselines=(run_v0_baselines and i == 0)
            )
        finally:
            _restore(originals)
        elapsed_min = (time.time() - t0) / 60.0
        print(f"Seed {seed} finished in {elapsed_min:.1f} min")

        per_seed_metrics.append(metrics)
        # Build the headline directly from the metrics full_run returned.  Do not
        # read it back via list_runs(): full_run writes its run dir without
        # updating results/runs/index.json, so the index is stale and
        # `list_runs()[0]` is not this seed's run.
        from .archive import _extract_headline

        headline = _extract_headline(metrics.get("stages", {}))
        per_seed_headlines.append({"seed": seed, **headline})

        # Running aggregate.
        if len(per_seed_headlines) > 1:
            print("Running aggregate (mean ± std):")
            for k in HEADLINE_KEYS:
                vals = [h[k] for h in per_seed_headlines if h.get(k) is not None]
                if not vals:
                    continue
                m = float(np.mean(vals))
                s = float(np.std(vals))
                print(f"  {k:<24} = {m:.4f} ± {s:.4f}  (n={len(vals)})")

    total_elapsed = (time.time() - sweep_start) / 60.0
    print(f"\nMulti-seed sweep done in {total_elapsed:.1f} min total.")

    # Final aggregate.
    summary: dict = {
        "seeds": seeds,
        "quick": quick,
        "total_elapsed_minutes": total_elapsed,
        "per_seed_headlines": per_seed_headlines,
        "aggregate": {},
    }
    for k in HEADLINE_KEYS:
        vals = [h[k] for h in per_seed_headlines if h.get(k) is not None]
        if not vals:
            continue
        summary["aggregate"][k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
            "values": [float(v) for v in vals],
        }

    out_path = ARCHIVE_ROOT / "multi_seed_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Aggregate written to {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(THESIS_SEEDS),
        help="Seeds to run sequentially.  Default: the canonical thesis seeds "
             f"{list(THESIS_SEEDS)}.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use the --quick (halved-epoch) profile in each run.",
    )
    parser.add_argument(
        "--v0-baselines", action="store_true",
        help="Also run the V0 reference baselines on the first seed (reference "
             "rows for the RQ2 baseline tables).",
    )
    args = parser.parse_args()
    main(seeds=args.seeds, quick=args.quick, run_v0_baselines=args.v0_baselines)
