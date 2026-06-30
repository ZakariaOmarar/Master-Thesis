"""Aggregate the multi-seed result set into one median / spread report.

For every headline RQ1/RQ2/RQ3 number this reads each seed's full-pipeline run
(``metrics.json`` + ``rq1_strict_nmi.json``) and its localization CV
(``lopo/summary.json`` + ``cross_dataset/summary.json``), plus the
encoder-independent references (``rq1_mode_refs_*.json``,
``v0_domain_shift_multiseed_*.json``), and reports the median [min, max] and
mean +/- std across seeds.  This is the "report multiseed properly" deliverable:
it never picks a single seed.  Pure JSON crunching -- no GPU, safe to re-run.

Run::

    python -m scripts.aggregate_multiseed                     # auto-discover seeds
    python -m scripts.aggregate_multiseed --runs <dir> <dir>  # explicit run dirs
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import statistics as stats
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "results" / "runs"
V0_DIR = REPO / "results" / "v0_anomaly"
OUT_DIR = REPO / "results" / "reports"

# The thesis multi-seed set (Results.tex:31). Stale one-off runs with other
# seeds are ignored unless the caller overrides with --seeds / --runs.
CANONICAL_SEEDS = (42, 1337, 2024, 7, 99)

SYN_RUNGS = ["-10.0", "-5.0", "0.0", "5.0", "10.0"]
LOPO_MODES = ["tdoa_only", "both", "srp_only", "vibration_only_learned"]
CROSS_MODES = ["tdoa_only", "both", "srp_only", "vibration_only_learned"]
LORO_PARADIGMS = [
    "LF_confidence_gated", "V4-acoustic", "V4-fusion",
    "LF_uniform_avg", "V4-vibration", "LF_weighted_avg",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _get(d, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def summ(vals) -> dict | None:
    xs = [float(v) for v in vals if isinstance(v, (int, float)) and v == v]
    if not xs:
        return None
    return {
        "median": stats.median(xs),
        "mean": stats.fmean(xs),
        "std": stats.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
        "n": len(xs),
        "values": xs,
    }


def _newest(pattern: str) -> Path | None:
    c = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(c[0]) if c else None


def discover_runs(seeds) -> dict[int, Path]:
    """Newest full-pipeline run per encoder seed (dedups re-runs of a seed),
    restricted to ``seeds`` so stale one-off runs are not swept in."""
    want = set(seeds)
    by_seed: dict[int, Path] = {}
    for c in glob.glob(str(RUNS_DIR / "*__full_pipeline_b5_cma")):
        p = Path(c)
        man = _load(p / "manifest.json")
        if man is None or not (p / "metrics.json").exists():
            continue
        seed = _get(man, "configs", "v3_cfg", "seed")
        if seed is None or seed not in want:
            continue
        if seed not in by_seed or p.stat().st_mtime > by_seed[seed].stat().st_mtime:
            by_seed[seed] = p
    return dict(sorted(by_seed.items()))


# --------------------------------------------------------------------------- #
# per-seed extraction
# --------------------------------------------------------------------------- #
def extract_seed(run: Path) -> dict:
    """Flatten one run's headline numbers into {metric_key: value}."""
    st = _get(_load(run / "metrics.json") or {}, "stages", default={})
    strict = _get(_load(run / "rq1_strict_nmi.json") or {}, "strict", default={})
    lopo = _get(_load(run / "lopo" / "summary.json") or {}, "aggregate_per_mode", default={})
    cross = _get(_load(run / "cross_dataset" / "summary.json") or {},
                 "directions", "d1to4_to_d5", "per_channel_mode", default={})
    loro = _get(st, "rq3_paradigm_comparison", "loro_summary", "aggregate", default={})
    fd = _get(st, "v3_fusion_depth", default={})

    v: dict = {}
    # RQ1
    v["rq1_v1ac_sanity"] = _get(st, "v1_acoustic", "sanity_nmi")
    v["rq1_v2fus_sanity"] = _get(st, "v2", "rq1_nmi")
    v["rq1_v2dropvib"] = _get(st, "v2_a1_drop_vibration", "rq1_nmi")
    v["rq1_strict_fusion"] = _get(strict, "v2_fusion", "nmi")
    v["rq1_strict_acoustic"] = _get(strict, "v1_acoustic", "nmi")
    # RQ2
    v["rq2_real_f1"] = _get(st, "v3_real_anomaly", "f1")
    v["rq2_healthy_alert"] = _get(fd, "per_cluster_breakdown_healthy", "alert_rate_total")
    v["rq2_cond_dnll"] = _get(fd, "v3_vs_a2_paired_test", "delta_point")
    v["rq2_valnll_fusion"] = _get(st, "v3_three_paradigms", "fusion", "val_nll_final")
    for r in SYN_RUNGS:
        v[f"rq2_synauc_cond_{r}"] = _get(fd, "synthetic_anomaly_auc", "auc_conditional", r)
        v[f"rq2_synauc_uncond_{r}"] = _get(fd, "synthetic_anomaly_auc", "auc_unconditional", r)
    # RQ3
    for m in LOPO_MODES:
        v[f"rq3_lopo_{m}"] = _get(lopo, m, "mean_mae_m")
    for m in CROSS_MODES:
        v[f"rq3_cross_{m}"] = _get(cross, m, "val_mae_3d_m")
    for para in LORO_PARADIGMS:
        v[f"rq3_loro_{para}"] = _get(loro, para, "macro_mean_mae_m")
    return v


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _fmt(s: dict | None) -> str:
    if s is None:
        return "pending (n=0)"
    return (f"{s['median']:.3f} [{s['min']:.3f}, {s['max']:.3f}]  "
            f"(mean {s['mean']:.3f} +/- {s['std']:.3f}, n={s['n']})")


def render_md(payload: dict) -> str:
    seeds = payload["seeds"]
    agg = payload["aggregate"]
    L = [f"# Multi-seed complete result ({payload['generated']})", "",
         f"Seeds: {seeds}", ""]
    if payload["pending"]:
        L += ["> **Pending (not yet run):**"]
        for line in payload["pending"]:
            L.append(f"> - {line}")
        L.append("")

    def section(title, rows):
        L.append(f"## {title}")
        L.append("")
        L.append("| metric | median [min, max] (mean +/- std, n) |")
        L.append("|---|---|")
        for key, label in rows:
            L.append(f"| {label} | {_fmt(agg.get(key))} |")
        L.append("")

    section("RQ1 - mode discovery (sanity-gate + strict NMI)", [
        ("rq1_v1ac_sanity", "V1 acoustic sanity NMI"),
        ("rq1_v2fus_sanity", "V2 fusion sanity NMI"),
        ("rq1_v2dropvib", "V2 fusion - vibration (ablation) NMI"),
        ("rq1_strict_fusion", "strict K=3 fusion NMI"),
        ("rq1_strict_acoustic", "strict K=3 acoustic NMI"),
    ])
    if payload.get("rq1_refs"):
        L += ["_RQ1 reference rows (encoder-independent, computed once):_  ",
              payload["rq1_refs"], ""]

    section("RQ2 - anomaly head", [
        ("rq2_real_f1", "real-anomaly F1"),
        ("rq2_healthy_alert", "healthy alert rate"),
        ("rq2_cond_dnll", "conditioning dNLL (cond vs uncond)"),
        ("rq2_valnll_fusion", "validation NLL (fusion)"),
    ])
    L.append("### RQ2 - synthetic-AUC ladder (conditional vs unconditional)")
    L.append("")
    L.append("| latent SNR (dB) | conditional | unconditional |")
    L.append("|---|---|---|")
    for r in SYN_RUNGS:
        L.append(f"| {r} | {_fmt(agg.get(f'rq2_synauc_cond_{r}'))} "
                 f"| {_fmt(agg.get(f'rq2_synauc_uncond_{r}'))} |")
    L.append("")

    section("RQ3 - leave-one-position-out MAE (m), by channel mode", [
        (f"rq3_lopo_{m}", m) for m in LOPO_MODES
    ])
    section("RQ3 - cross-session transfer (D2/D3/D4 -> D5) MAE (m), by channel mode", [
        (f"rq3_cross_{m}", m) for m in CROSS_MODES
    ])
    section("RQ3 - leave-one-recording-out macro MAE (m), by paradigm", [
        (f"rq3_loro_{p}", p) for p in LORO_PARADIGMS
    ])

    L += ["## Per-seed raw values", "",
          "See the JSON sibling for the full per-seed table.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, default=None,
                    help="Explicit run dirs (default: newest run per canonical seed).")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help=f"Encoder seeds to include (default: {CANONICAL_SEEDS}).")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.runs:
        runs = {}
        for r in args.runs:
            man = _load(r / "manifest.json")
            seed = _get(man or {}, "configs", "v3_cfg", "seed", default=str(r.name))
            runs[seed] = r
        runs = dict(sorted(runs.items(), key=lambda kv: str(kv[0])))
    else:
        runs = discover_runs(args.seeds or CANONICAL_SEEDS)
    if not runs:
        print("No full_pipeline_b5_cma runs with metrics.json + manifest.json found.")
        return 1

    per_seed = {seed: extract_seed(run) for seed, run in runs.items()}

    # Aggregate every metric key seen.
    keys: list[str] = []
    for v in per_seed.values():
        for k in v:
            if k not in keys:
                keys.append(k)
    aggregate = {k: summ([per_seed[s].get(k) for s in per_seed]) for k in keys}

    # Pending localization CVs (seeds whose lopo/cross summary is absent).
    pending = []
    for seed, run in runs.items():
        miss = [name for name, rel in (("LOPO", "lopo/summary.json"),
                                       ("cross-session", "cross_dataset/summary.json"))
                if not (run / rel).exists()]
        if miss:
            pending.append(f"seed {seed} ({run.name}): {', '.join(miss)}")

    # Encoder-independent RQ1 references (floor / ceiling), newest if present.
    rq1_refs_txt = None
    ref = _newest(str(V0_DIR / "rq1_mode_refs_*.json"))
    if ref:
        d = _load(ref) or {}
        floor = _get(d, "mode_floor", "nmi")
        ceil = {k: _get(v, "val_macro_f1") for k, v in _get(d, "lgbm_ceiling", default={}).items()
                if isinstance(v, dict)}
        rq1_refs_txt = (f"K-means floor NMI={floor:.3f}; " if isinstance(floor, (int, float))
                        else "K-means floor pending; ")
        rq1_refs_txt += "LightGBM ceiling macro-F1 " + (
            ", ".join(f"{k}={vv:.3f}" for k, vv in ceil.items() if isinstance(vv, (int, float)))
            or "pending") + f"  (source: {ref.name})"

    payload = {
        "generated": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "seeds": list(runs.keys()),
        "runs": {str(s): r.name for s, r in runs.items()},
        "pending": pending,
        "rq1_refs": rq1_refs_txt,
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "aggregate": aggregate,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = payload["generated"]
    json_out = args.output or (OUT_DIR / f"multiseed_complete_{ts}.json")
    md_out = json_out.with_suffix(".md")
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_out.write_text(render_md(payload), encoding="utf-8")
    print(render_md(payload))
    print(f"\nwrote {json_out.relative_to(REPO)} and {md_out.relative_to(REPO)}")
    if pending:
        print(f"\n{len(pending)} seed(s) still missing localization CV - run "
              "scripts.run_multiseed_complete to fill them.")
    return 0


if __name__ == "__main__":
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    raise SystemExit(main())
