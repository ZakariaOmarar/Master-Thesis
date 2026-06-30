"""RQ1 reference rows: the unsupervised K-means *floor* and the LightGBM
supervised *ceiling*, both on hand-engineered features + mode labels.

These are encoder-independent (they never touch the V1/V2 encoder), so they are
computed once, not per seed -- unlike the learned NMI, which varies by seed.
They are exactly the two reference rows ``assemble_comparison`` flags as
``RE-RUN full_run for v0_mode_floor`` / ``v0_lgbm`` when a multi-seed sweep was
run without ``--v0-baselines``.  Reuses the same functions ``full_run._run_v0``
calls, so the numbers match a ``full_run --v0-baselines`` exactly, without the
multi-hour pipeline retrain.

Run::

    python -m scripts.baselines.rq1_mode_refs

Writes ``results/v0_anomaly/rq1_mode_refs_<ts>.json``.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "v0_anomaly"

# Same labelled-mode cohort full_run uses for RQ1 (D5 carries no mode label).
SSL_IDS = ("d1", "d2", "d3", "d4")
LGBM_IDS = ("d1", "d2")  # only D1/D2 have enough labelled recordings for the ceiling


def main() -> int:
    # Imported lazily so `-h` / import errors are cheap and the heavy deps load
    # only when we actually compute.
    from src.config.dataset_registry import REGISTRY
    from src.modeling.anomaly_baselines import (
        V0ModeConfig,
        cluster_mode_floor,
        train_v0_mode_lgbm,
    )
    from src.modeling.orchestration.full_run import resolved_loader

    loaders = []
    for meta in REGISTRY:
        if meta.id in SSL_IDS and meta.root.exists():
            print(f"  loading {meta.id} ...", flush=True)
            loaders.append(resolved_loader(f"{meta.id}.yaml"))
    if not loaders:
        print("No D1..D4 datasets available (registry roots missing); cannot "
              "compute the RQ1 references.")
        return 1

    out: dict = {
        "generated": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "cohort": "labelled D1..D4 operating-mode windows (K=3)",
        "note": ("encoder-independent RQ1 reference rows: unsupervised K-means "
                 "floor and supervised LightGBM ceiling, on hand-engineered "
                 "features. Computed once; not part of the per-seed story."),
    }

    # --- Floor: unsupervised K-means on hand-engineered features -------------
    try:
        floor = cluster_mode_floor(loaders, V0ModeConfig())
        out["mode_floor"] = {
            "method": "K-means / handcrafted (unsup. floor)",
            "nmi": float(floor.nmi),
            "ari": float(floor.ari),
            "purity": float(floor.purity),
            "n_windows": int(floor.n_windows),
            "n_recordings": int(floor.n_recordings),
            "n_clusters": int(floor.n_clusters),
            "label_set": list(floor.label_set),
        }
        print(f"  floor (K-means): NMI={floor.nmi:.3f} ARI={floor.ari:.3f} "
              f"purity={floor.purity:.3f}", flush=True)
    except Exception as e:
        out["mode_floor"] = {"skipped": f"{type(e).__name__}: {e}"}
        print(f"  floor skipped: {type(e).__name__}: {e}", flush=True)

    # --- Ceiling: supervised LightGBM per labelled campaign ------------------
    out["lgbm_ceiling"] = {}
    for L in loaders:
        if L.spec.id not in LGBM_IDS:
            continue
        try:
            r = train_v0_mode_lgbm(L, V0ModeConfig())
            out["lgbm_ceiling"][L.spec.id] = {
                "val_macro_f1": float(r.val_macro_f1),
                "val_per_class_f1": {str(k): float(v) for k, v in r.val_per_class_f1.items()},
                "n_train_recordings": len(r.train_recording_ids),
                "n_val_recordings": len(r.val_recording_ids),
            }
            print(f"  ceiling {L.spec.id} (LightGBM): macro-F1={r.val_macro_f1:.3f}", flush=True)
        except Exception as e:
            out["lgbm_ceiling"][L.spec.id] = {"skipped": f"{type(e).__name__}: {e}"}
            print(f"  ceiling {L.spec.id} skipped: {type(e).__name__}: {e}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"rq1_mode_refs_{out['generated']}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    raise SystemExit(main())
