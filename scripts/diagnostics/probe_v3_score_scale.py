"""Probe: are the saved V3 head artifacts score-sane under the current code?

Loads the acoustic + fusion pipelines exactly as rq2_three_paradigm_eval does,
scores a healthy D2 cohort, and prints the NLL distribution + alert rate against
a freshly-fit per-cluster threshold.  Sane => healthy NLL p95 ~ the saved
thresholds.npz p95 (~ -240) and alert rate ~ 0.05.  Degenerate => NLL ~ 1e5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.anomaly.threshold import PerClusterThresholds
from src.modeling.anomaly.v3_per_modality import (
    V3AcousticOnlyAdapter,
    V3VibrationOnlyAdapter,
)
from src.modeling.eval.rq2_three_paradigm_eval import (
    _build_loader,
    _build_v1,
    _build_v2,
    _load_state,
    _load_v3,
    _loader,
    _PipelineState,
    _score_cohort_three_paradigms,
    _segments_for,
)
from src.modeling.orchestration.full_run import v1_config, v2_config

RUN = REPO / "results" / "runs" / "20260524_002807__full_pipeline_b5_cma"


def main() -> int:
    v1_cfg = v1_config(False)
    v2_cfg = v2_config(False)
    embed = int(v1_cfg.embed_dim)

    v1_a = _build_v1("acoustic", v1_cfg); _load_state(RUN / "v1" / "acoustic.pt", v1_a); v1_a.eval()
    v1_v = _build_v1("vibration", v1_cfg); _load_state(RUN / "v1" / "vibration.pt", v1_v); v1_v.eval()
    v2 = _build_v2(v2_cfg); _load_state(RUN / "v2" / "encoder.pt", v2); v2.eval()

    flow_a, th_a = _load_v3(RUN / "v3_acoustic", x_dim=embed, c_dim=embed)
    flow_v, th_v = _load_v3(RUN / "v3_vibration", x_dim=embed, c_dim=embed)
    flow_f, th_f = _load_v3(RUN / "v3_fusion", x_dim=embed, c_dim=embed)
    pipelines = [
        _PipelineState("acoustic", V3AcousticOnlyAdapter(v1_a), flow_a, th_a),
        _PipelineState("vibration", V3VibrationOnlyAdapter(v1_v), flow_v, th_v),
        _PipelineState("fusion", v2, flow_f, th_f),
    ]

    loaders = [_loader(d) for d in ("d1", "d2")]
    healthy = _segments_for(loaders, v2_cfg, healthy=True)
    print(f"healthy segments: {len(healthy)}")
    scored = _score_cohort_three_paradigms(pipelines, _build_loader(healthy, v2_cfg))

    for name in ("acoustic", "vibration", "fusion"):
        s = scored[name]["scores"]
        print(f"\n[{name}] n={s.size}  NLL: min={s.min():.1f} p50={np.percentile(s,50):.1f} "
              f"p95={np.percentile(s,95):.1f} max={s.max():.1f}")
        # saved threshold alert rate (the degenerate path)
        alerts_saved = scored[name]["alerts"]
        print(f"  saved-threshold alert rate (healthy) = {alerts_saved.mean():.3f}")
        # fresh threshold alert rate (scale-invariant path)
        ctx = scored[name]["contexts"]
        thr = PerClusterThresholds.fit(ctx, s, n_clusters=3, seed=42)
        fresh, _ = thr.alert(ctx, s, percentile=95)
        print(f"  fresh-threshold alert rate (healthy) = {fresh.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
