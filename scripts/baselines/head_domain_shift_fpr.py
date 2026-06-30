"""Compute the V3 head's domain-shift healthy FPR on the V0 protocol.

The three-paradigm V3 artifacts do not persist the learned ``xt_pool`` (pma2),
so a saved flow cannot be re-scored (NLL blows up; see
``scripts/diagnostics/probe_v3_score_scale.py``).  This script therefore
*retrains* the three V3 pipelines (acoustic / vibration / fusion) from the
frozen V1+V2 encoders of ``--from-run`` and, while ``xt_pool`` is still live in
memory, scores the healthy cohort and computes the same two calibration regimes
``evaluate_v0_anomaly`` reports for the baselines:

  * **FPR in-dist** — threshold fit + evaluated on the same held-out conditions
    (window-parity split).
  * **FPR shift**   — threshold fit on one set of held-out healthy *conditions*
    and evaluated on a disjoint set (recording-level split) — the regime that
    sends the acoustic OC-SVM to 0.95.

Reported for V3-acoustic, V3-vibration, the late-fusion AND of the two, and
V3-fusion, so the head sits on the *identical* axis as
``tab:res_v0_anomaly``'s FPR-shift column.

Run::

    python -m scripts.baselines.head_domain_shift_fpr --from-run results/runs/<id>            # full, seed 42
    python -m scripts.baselines.head_domain_shift_fpr --from-run results/runs/<id> --quick     # 3-epoch smoke
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.paradigms.run_v3_three_paradigms import (
    _build_v1_encoder,
    _build_v2_encoder,
    _load_state,
)
from src.config import resolve_device
from src.modeling.anomaly.event_detection import v3_real_anomaly_detection
from src.modeling.anomaly.threshold import PerClusterThresholds
from src.modeling.anomaly.v3_per_modality import (
    V3AcousticOnlyAdapter,
    V3VibrationOnlyAdapter,
)
from src.modeling.anomaly.v3_trainer import score_segments, train_v3_cnf
from src.modeling.anomaly_baselines.v0_evaluation import (
    _plain_split,
    _wilson_interval,
)
from src.modeling.context.v2_ssl import _precompute_paired
from src.modeling.orchestration.full_run import (
    resolved_loader,
    v1_config,
    v2_config,
    v3_config,
)

DATASETS = ("d1", "d2", "d3", "d4")
PERCENTILE = 95
N_CLUSTERS = 3


def _log(msg: str) -> None:
    line = f"[{_dt.datetime.now():%H:%M:%S}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def _score_healthy_per_recording(encoder, flow, xt_pool, loaders, v2_cfg, device, win_override, anomaly: bool = False, anchor_norm=None):
    """Score every healthy window under one pipeline, tagged by dataset+recording.

    Returns ``(scores, contexts, rec_key, ds_id)`` aligned per window.  Each
    recording is scored on its own so its windows carry a single rec id.

    ``win_override`` is V3's per-dataset ``window_seconds_override`` — it must be
    threaded through so the scored windows match the ones the flow trained on
    (otherwise NLL blows up exactly like the legacy mean-pool mismatch).
    """
    scores_all, ctx_all, rec_all, ds_all = [], [], [], []
    for L in loaders:
        for seg in L.list_segments():
            if bool(seg.is_anomaly) != anomaly:  # healthy when anomaly=False, faults when True
                continue
            ps = _precompute_paired(seg, v2_cfg)
            if ps is None:
                continue
            s, c, _ = score_segments(
                encoder, flow, [ps], v2_cfg=v2_cfg, xt_pool=xt_pool, device=device,
                window_seconds_override=win_override, anchor_norm=anchor_norm,
            )
            if s.size == 0:
                continue
            key = f"{seg.dataset_id}::{seg.recording_id}"
            scores_all.append(np.asarray(s, dtype=np.float64))
            ctx_all.append(np.asarray(c, dtype=np.float64))
            rec_all.extend([key] * s.size)
            ds_all.extend([seg.dataset_id] * s.size)
    return (
        np.concatenate(scores_all) if scores_all else np.zeros(0),
        np.concatenate(ctx_all) if ctx_all else np.zeros((0, 1)),
        np.array(rec_all, dtype=object),
        np.array(ds_all, dtype=object),
    )


def _fpr(thr: PerClusterThresholds, ctx, scores) -> float:
    alerts, _ = thr.alert(ctx, scores, percentile=PERCENTILE)
    return float(alerts.mean()) if alerts.size else float("nan")


def _alerts(thr: PerClusterThresholds, ctx, scores) -> np.ndarray:
    a, _ = thr.alert(ctx, scores, percentile=PERCENTILE)
    return a.astype(bool)


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


def _compute_shift_rows(scored: dict, seed: int) -> tuple[dict, dict]:
    """One domain-shift evaluation at a given split ``seed``.

    Reuses the per-window NLL scores (which are seed-independent); only the
    held-in/held-out recording partition and the per-cluster k-means depend on
    ``seed``. This makes a multi-seed sweep over the shift split nearly free
    once the three flows are trained and scored. Returns ``(rows, comp)`` where
    ``rows`` carries fpr_in_dist / fpr_shift for acoustic/vibration/fusion/AND
    and ``comp`` is the per-dataset composition of the shift-eval cohort.
    """
    rec_keys = sorted(set(scored["acoustic"]["rec"].tolist()))
    shift_fit_recs, shift_eval_recs = _plain_split(rec_keys, 0.5, seed + 1)

    def _mask(rec, ids):
        return np.array([r in ids for r in rec], dtype=bool)

    # Per-pipeline thresholds fit on shift_fit conditions.
    thr = {}
    for name in ("acoustic", "vibration", "fusion"):
        d = scored[name]
        fm = _mask(d["rec"], shift_fit_recs)
        k = max(1, min(N_CLUSTERS, int(fm.sum())))
        thr[name] = PerClusterThresholds.fit(d["ctx"][fm], d["scores"][fm], n_clusters=k, seed=seed)

    # In-dist (window-parity split within shift_fit conditions) for reference.
    rows = {}
    for name in ("acoustic", "vibration", "fusion"):
        d = scored[name]
        fm = np.where(_mask(d["rec"], shift_fit_recs))[0]
        em = _mask(d["rec"], shift_eval_recs)
        in_fit, in_eval = fm[0::2], fm[1::2]
        k = max(1, min(N_CLUSTERS, int(in_fit.size)))
        thr_in = PerClusterThresholds.fit(d["ctx"][in_fit], d["scores"][in_fit], n_clusters=k, seed=seed)
        fpr_in = _fpr(thr_in, d["ctx"][in_eval], d["scores"][in_eval])
        fpr_shift = _fpr(thr[name], d["ctx"][em], d["scores"][em])
        n_eval = int(em.sum())
        ci = _wilson_interval(int(round(fpr_shift * n_eval)), n_eval)
        rows[name] = {"fpr_in_dist": fpr_in, "fpr_shift": fpr_shift,
                      "n_shift_eval": n_eval, "shift_ci95": list(ci)}

    # Late-fusion AND on the shift_eval cohort (each modality under its shift threshold).
    da, dv = scored["acoustic"], scored["vibration"]
    em_a, em_v = _mask(da["rec"], shift_eval_recs), _mask(dv["rec"], shift_eval_recs)
    and_alert = _alerts(thr["acoustic"], da["ctx"][em_a], da["scores"][em_a]) & \
                _alerts(thr["vibration"], dv["ctx"][em_v], dv["scores"][em_v])
    # in-dist AND
    fa = np.where(_mask(da["rec"], shift_fit_recs))[0]
    ia_fit, ia_eval = fa[0::2], fa[1::2]
    ka = max(1, min(N_CLUSTERS, int(ia_fit.size)))
    thr_a_in = PerClusterThresholds.fit(da["ctx"][ia_fit], da["scores"][ia_fit], n_clusters=ka, seed=seed)
    thr_v_in = PerClusterThresholds.fit(dv["ctx"][ia_fit], dv["scores"][ia_fit], n_clusters=ka, seed=seed)
    and_in = _alerts(thr_a_in, da["ctx"][ia_eval], da["scores"][ia_eval]) & \
             _alerts(thr_v_in, dv["ctx"][ia_eval], dv["scores"][ia_eval])
    rows["AND"] = {"fpr_in_dist": float(and_in.mean()) if and_in.size else float("nan"),
                   "fpr_shift": float(and_alert.mean()) if and_alert.size else float("nan"),
                   "n_shift_eval": int(em_a.sum()), "shift_ci95": None}

    # Per-dataset shift_eval composition (transparency: which campaign is the shift).
    ds_eval = da["ds"][em_a]
    comp = {str(u): int((ds_eval == u).sum()) for u in sorted(set(ds_eval.tolist()))}
    return rows, comp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-run", required=True, help="run dir with v1/ + v2/ encoders")
    ap.add_argument("--quick", action="store_true", help="3-epoch smoke")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="If given, also sweep the shift split over these seeds and "
                         "report mean/range/collapse-count (reuses the one retrain).")
    args = ap.parse_args()

    src = Path(args.from_run).resolve()
    if not src.exists():
        raise SystemExit(f"--from-run {src} not found")

    device = resolve_device("auto")
    v1_cfg, v2_cfg, v3_cfg = v1_config(args.quick), v2_config(args.quick), v3_config(args.quick)
    _log(f"device={device}  xt_pool={v3_cfg.xt_pool}  v3.epochs={v3_cfg.epochs}  seed={args.seed}")

    _log("loading frozen V1/V2 encoders ...")
    v1_a = _build_v1_encoder("acoustic", v1_cfg); _load_state(src / "v1" / "acoustic.pt", v1_a, "V1-a")
    v1_v = _build_v1_encoder("vibration", v1_cfg); _load_state(src / "v1" / "vibration.pt", v1_v, "V1-v")
    v2 = _build_v2_encoder(v2_cfg); _load_state(src / "v2" / "encoder.pt", v2, "V2")

    _log("loading SSL loaders (d1-d4) ...")
    loaders = [resolved_loader(f"{d}.yaml") for d in DATASETS]

    pipelines = {
        "acoustic": V3AcousticOnlyAdapter(v1_a),
        "vibration": V3VibrationOnlyAdapter(v1_v),
        "fusion": v2,
    }

    # Anomaly segments for the event-level fifth axis (#16).
    anom_segs = [s for L in loaders for s in L.list_segments() if s.is_anomaly]

    # Train all three; keep flow + live xt_pool, then score healthy.
    scored: dict[str, dict] = {}
    scored_anom: dict[str, dict] = {}
    deployed_thr: dict[str, PerClusterThresholds] = {}
    event_f1: dict[str, dict] = {}
    for name, enc in pipelines.items():
        _log(f"training V3-{name} ...")
        res = train_v3_cnf(enc, loaders, v2_cfg=v2_cfg, v3_cfg=v3_cfg)
        # When the flow was trained with the impulse+spectral anchor, it has
        # +N_ANCHOR input dims; the same standardized anchor must be appended at
        # scoring time or flow.dim (e.g. 72) won't match the scored x (e.g. 64).
        anchor_norm = (
            (res.anchor_mean, res.anchor_std) if res.anchor_mean is not None else None
        )
        _log(f"  V3-{name} val NLL final = {res.val_nll[-1]:+.2f}  "
             f"(xt_pool={'yes' if res.xt_pool is not None else 'mean'}, "
             f"anchor={'yes' if anchor_norm is not None else 'no'})")
        s, c, rec, ds = _score_healthy_per_recording(
            enc, res.flow, res.xt_pool, loaders, v2_cfg, device, v3_cfg.window_seconds_override,
            anchor_norm=anchor_norm,
        )
        _log(f"  scored {s.size} healthy windows; NLL p50={np.percentile(s,50):+.1f} p95={np.percentile(s,95):+.1f}")
        scored[name] = {"scores": s, "ctx": c, "rec": rec, "ds": ds}
        # Anomaly cohorts + the deployed per-cluster threshold, for the RQ2 alert
        # table (tab:res_rq2_alert) that the degenerate full_run stage-8 cannot
        # produce (it scores the saved flow without the live xt_pool).
        deployed_thr[name] = res.thresholds
        sa, ca, _, dsa = _score_healthy_per_recording(
            enc, res.flow, res.xt_pool, loaders, v2_cfg, device,
            v3_cfg.window_seconds_override, anomaly=True, anchor_norm=anchor_norm,
        )
        scored_anom[name] = {"scores": sa, "ctx": ca, "ds": dsa}
        _log(f"  scored {sa.size} anomaly windows (D2/D3/D4 recall via deployed threshold)")
        # #16 event-level precision/recall/F1 vs weak envelope labels (guarded —
        # a failure here must not lose the domain-shift result).
        try:
            ev = v3_real_anomaly_detection(
                enc, res.flow, res.thresholds, anom_segs,
                v2_cfg=v2_cfg, xt_pool=res.xt_pool, device=device,
                anchor_norm=anchor_norm,
            )
            event_f1[name] = {k: ev.get(k) for k in ("precision", "recall", "f1")}
            _log(f"  event-level (#16): P={ev.get('precision')} R={ev.get('recall')} F1={ev.get('f1')}")
        except Exception as e:
            event_f1[name] = {"error": f"{type(e).__name__}: {e}"}
            _log(f"  event-level F1 FAILED ({type(e).__name__}: {e})")

    # Primary shift-split table at the headline seed (drives tab:res_rq2_shift).
    rec_keys = sorted(set(scored["acoustic"]["rec"].tolist()))
    shift_fit_recs, shift_eval_recs = _plain_split(rec_keys, 0.5, args.seed + 1)
    _log(f"\nshift split: fit on {len(shift_fit_recs)} recs, eval on {len(shift_eval_recs)} recs")
    _log(f"  shift_eval recordings: {sorted(shift_eval_recs)}")
    rows, comp = _compute_shift_rows(scored, args.seed)

    # --- Multi-seed shift sweep: the shift FPR is split-fragile (a single split
    #     can put an easy or a hard campaign in the eval set), so report the
    #     mean / range / collapse-count over several split seeds, symmetric with
    #     v0_domain_shift_multiseed for the baselines. Reuses the one retrain
    #     above, so this is seconds, not a per-seed retrain. ---
    shift_multiseed = None
    if args.seeds:
        COLLAPSE = 0.20
        per_pipe: dict[str, list[float]] = {n: [] for n in ("acoustic", "vibration", "fusion", "AND")}
        per_seed_comp: dict[str, dict] = {}
        _log(f"\n=== multi-seed shift sweep ({len(args.seeds)} seeds) ===")
        for s in args.seeds:
            r_s, c_s = _compute_shift_rows(scored, s)
            per_seed_comp[str(s)] = c_s
            for n in per_pipe:
                per_pipe[n].append(float(r_s[n]["fpr_shift"]))
            _log(f"  seed {s:<5} " + "  ".join(
                f"{n}={r_s[n]['fpr_shift']:.3f}" for n in ("acoustic", "vibration", "fusion", "AND")
            ) + f"  eval={sorted(c_s)}")
        shift_multiseed = {"collapse_threshold": COLLAPSE, "seeds": list(args.seeds),
                           "per_seed_eval_composition": per_seed_comp, "pipelines": {}}
        for n, vals in per_pipe.items():
            arr = np.asarray(vals, dtype=float)
            shift_multiseed["pipelines"][n] = {
                "shift_fpr": [float(v) for v in vals],
                "mean": float(np.nanmean(arr)),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "collapse_count": int((arr > COLLAPSE).sum()),
                "n_seeds": int(arr.size),
            }

    # --- RQ2 per-cohort alert table (tab:res_rq2_alert): healthy FPR + D2/D3/D4
    #     recall under each pipeline's DEPLOYED per-cluster threshold, plus the
    #     parameter-free AND / OR late-fusion rules. Guarded so a failure here
    #     cannot lose the domain-shift / event-F1 results computed above. ---
    def _dep_alerts(name, ctx, scores):
        a, _ = deployed_thr[name].alert(np.asarray(ctx), np.asarray(scores), percentile=PERCENTILE)
        return a.astype(bool)

    def _compute_alert_table() -> dict:
        cohorts = ("d2", "d3", "d4")
        tbl: dict[str, dict] = {}
        for name in ("acoustic", "vibration", "fusion"):
            h = _dep_alerts(name, scored[name]["ctx"], scored[name]["scores"])
            row = {"healthy_fpr": float(h.mean()) if h.size else float("nan")}
            a = scored_anom[name]
            for c in cohorts:
                m = a["ds"] == c
                am = _dep_alerts(name, a["ctx"][m], a["scores"][m]) if m.any() else np.zeros(0, bool)
                row[c] = float(am.mean()) if am.size else None
            tbl[name] = row
        # AND / OR: acoustic and vibration alerts aligned per window (same segments/order).
        ha = _dep_alerts("acoustic", scored["acoustic"]["ctx"], scored["acoustic"]["scores"])
        hv = _dep_alerts("vibration", scored["vibration"]["ctx"], scored["vibration"]["scores"])
        nh = min(ha.size, hv.size)
        aa, av = scored_anom["acoustic"], scored_anom["vibration"]
        for rule, op in (("AND", np.logical_and), ("OR", np.logical_or)):
            row = {"healthy_fpr": float(op(ha[:nh], hv[:nh]).mean()) if nh else float("nan")}
            for c in cohorts:
                ma, mv = aa["ds"] == c, av["ds"] == c
                ca_ = _dep_alerts("acoustic", aa["ctx"][ma], aa["scores"][ma]) if ma.any() else np.zeros(0, bool)
                cv_ = _dep_alerts("vibration", av["ctx"][mv], av["scores"][mv]) if mv.any() else np.zeros(0, bool)
                nc = min(ca_.size, cv_.size)
                row[c] = float(op(ca_[:nc], cv_[:nc]).mean()) if nc else None
            tbl[rule] = row
        return tbl

    try:
        alert_table = _compute_alert_table()
    except Exception as e:
        alert_table = {"error": f"{type(e).__name__}: {e}"}
        _log(f"RQ2 alert table FAILED ({type(e).__name__}: {e})")

    out = {
        "generated": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "from_run": str(src.relative_to(REPO)), "seed": args.seed, "quick": args.quick,
        "protocol": "V0 domain-shift mirror: per-cluster p95 threshold fit on shift_fit "
                    "healthy conditions, evaluated on disjoint shift_eval conditions.",
        "shift_eval_datasets": comp,
        "rows": rows,
        "event_f1": event_f1,
        "rq2_alert_table": alert_table,
        "rq2_shift_multiseed": shift_multiseed,
    }
    _log("\n=== HEAD healthy FPR (same axis as tab:res_v0_anomaly) ===")
    _log(f"{'pipeline':<12} {'FPR in-dist':>12} {'FPR shift':>12} {'n_eval':>8}")
    for name in ("acoustic", "vibration", "AND", "fusion"):
        r = rows[name]
        _log(f"{name:<12} {r['fpr_in_dist']:>12.3f} {r['fpr_shift']:>12.3f} {r['n_shift_eval']:>8d}")
    _log(f"shift_eval composition by dataset: {comp}")

    _log("\n=== RQ2 alert table (deployed threshold): healthy FPR + per-cohort recall ===")
    _log(f"{'rule':<12} {'healthyFPR':>10} {'D2':>7} {'D3':>7} {'D4':>7}")
    for nm in ("acoustic", "vibration", "AND", "OR", "fusion"):
        r = alert_table.get(nm, {})
        _log(f"{nm:<12} {_fmt(r.get('healthy_fpr')):>10} {_fmt(r.get('d2')):>7} "
             f"{_fmt(r.get('d3')):>7} {_fmt(r.get('d4')):>7}")

    if shift_multiseed is not None:
        _log("\n=== RQ2 shift FPR across split seeds (head, symmetric with V0 multiseed) ===")
        _log(f"{'pipeline':<12} {'shiftMean':>10} {'min':>7} {'max':>7} {'collapse':>9}")
        for nm in ("acoustic", "vibration", "fusion", "AND"):
            p = shift_multiseed["pipelines"][nm]
            _log(f"{nm:<12} {p['mean']:>10.3f} {p['min']:>7.3f} {p['max']:>7.3f} "
                 f"{p['collapse_count']:>5}/{p['n_seeds']:<3}")

    out_dir = REPO / "results" / "v0_anomaly"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if args.quick else ""
    out_path = out_dir / f"head_domain_shift_{out['generated']}{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _log(f"wrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
