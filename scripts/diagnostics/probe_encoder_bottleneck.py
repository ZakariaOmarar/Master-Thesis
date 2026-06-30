"""Probe #1: does the impulsiveness signal survive the encoder (bottleneck test)?

Freezes the trained encoders, takes the embedding that actually feeds the V3
normalizing flow (`x` = the pooled fused/per-modality representation), and asks
a simple Ridge regressor: can `x` linearly predict the raw physical features
(crest factor, peak, temporal kurtosis, energy) of its own modality?

  * Low test-R^2  => the encoder has optimized impulsiveness/energy OUT of the
    representation; the flow is blind to it no matter how good it is.  The fix
    is the encoder (or bypassing it with physical features), not the flow.
  * High test-R^2 => the signal is in `x`; the flow / threshold is the issue.

Targets are computed on the same windows as the embeddings, pooled over all
datasets (healthy + anomaly) so the regressor sees the full impulsiveness range.

Run:  python -m scripts.diagnostics.probe_encoder_bottleneck [run_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

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
    _segments_for,
)
from src.modeling.orchestration.full_run import v1_config, v2_config

DEFAULT_RUN = REPO / "results" / "runs" / "20260616_022513__full_pipeline_b5_cma"
FEATS = ["mean", "std", "max", "crest", "kurt"]


def _winfeats(feat: np.ndarray) -> dict[str, np.ndarray]:
    b = feat.shape[0]
    x = feat.reshape(b, -1, feat.shape[-1]).astype(np.float64)
    env = x.mean(axis=1)
    mu = env.mean(axis=1, keepdims=True)
    sd = env.std(axis=1, keepdims=True) + 1e-8
    return {
        "mean": x.mean(axis=(1, 2)), "std": x.std(axis=(1, 2)),
        "max": x.max(axis=(1, 2)),
        "crest": env.max(axis=1) / (np.abs(env.mean(axis=1)) + 1e-8),
        "kurt": (((env - mu) / sd) ** 4).mean(axis=1) - 3.0,
    }


def main() -> int:
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RUN
    print(f"run = {run.name}")
    v1_cfg, v2_cfg = v1_config(False), v2_config(False)
    embed = int(v1_cfg.embed_dim)

    v1_a = _build_v1("acoustic", v1_cfg); _load_state(run / "v1" / "acoustic.pt", v1_a); v1_a.eval()
    v1_v = _build_v1("vibration", v1_cfg); _load_state(run / "v1" / "vibration.pt", v1_v); v1_v.eval()
    v2 = _build_v2(v2_cfg); _load_state(run / "v2" / "encoder.pt", v2); v2.eval()
    flow_a, th_a, xt_a, anc_a = _load_v3(run / "v3_acoustic", x_dim=embed, c_dim=embed)
    flow_v, th_v, xt_v, anc_v = _load_v3(run / "v3_vibration", x_dim=embed, c_dim=embed)
    flow_f, th_f, xt_f, anc_f = _load_v3(run / "v3_fusion", x_dim=embed, c_dim=embed)
    pipes = [
        _PipelineState("acoustic", V3AcousticOnlyAdapter(v1_a), flow_a, th_a, xt_a, anc_a),
        _PipelineState("vibration", V3VibrationOnlyAdapter(v1_v), flow_v, th_v, xt_v, anc_v),
        _PipelineState("fusion", v2, flow_f, th_f, xt_f, anc_f),
    ]

    loaders = [_loader(d) for d in ("d1", "d2", "d3", "d4")]
    segs = []
    for i in range(4):
        segs += _segments_for([loaders[i]], v2_cfg, healthy=True)
        segs += _segments_for([loaders[i]], v2_cfg, healthy=False)
    print(f"collecting embeddings over {len(segs)} segs ...", flush=True)
    loader = _build_loader(segs, v2_cfg)

    X = {p.name: [] for p in pipes}
    raw = {"ac": {f: [] for f in FEATS}, "vib": {f: [] for f in FEATS}}
    with torch.no_grad():
        for batch in loader:
            ac, axyz = batch["ac_feat"], batch["ac_xyz"]
            vib, vxyz = batch["vib_feat"], batch["vib_xyz"]
            ds = batch["dataset_idx"]
            for p in pipes:
                d = p.encoder(ac, axyz, vib, vxyz, ds, mask_p=0.0)
                fused = torch.cat([d["a_fused"], d["v_fused"]], dim=1)
                x = p.xt_pool(fused) if p.xt_pool is not None else fused.mean(dim=1)
                X[p.name].append(x.cpu().numpy())
            for mod, key in (("ac", "ac_feat"), ("vib", "vib_feat")):
                for f, v in _winfeats(batch[key].numpy()).items():
                    raw[mod][f].append(v)
    X = {k: np.concatenate(v) for k, v in X.items()}
    raw = {m: {f: np.concatenate(v) for f, v in d.items()} for m, d in raw.items()}

    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    def _r2(emb: np.ndarray, y: np.ndarray) -> float:
        Xtr, Xte, ytr, yte = train_test_split(emb, y, test_size=0.3, random_state=0)
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
        return float(r2_score(yte, m.predict(sc.transform(Xte))))

    print("\nTest-R^2 of predicting RAW features from the flow-input embedding x")
    print("(low => encoder discarded that physical signal; high => signal is in x)\n")
    print(f"{'embedding':<10} {'target-mod':<10} " + " ".join(f"{f:>7}" for f in FEATS))
    for pname, mod in (("acoustic", "ac"), ("vibration", "vib"),
                       ("fusion", "ac"), ("fusion", "vib")):
        r2s = [_r2(X[pname], raw[mod][f]) for f in FEATS]
        print(f"{pname:<10} {mod:<10} " + " ".join(f"{r:>7.3f}" for r in r2s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
