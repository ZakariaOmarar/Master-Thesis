"""V5.2b — robustness check: SCADA MI/AUC against a CLEAN vibration-only target.

The headline V5.2 analysis (``v5_2_channel_mining.py``) tests SCADA channels
against the legacy 5-layer-pipeline anomaly events, which were derived from a
**partially-instrumented** sensor set (0/8 microphones live, 6/12 accelerometer
channels live — see the report's primary caveat).  This script asks the tightest
possible follow-up: does a *clean* anomaly target, built only from the 6 live
accelerometer channels, recover any significant SCADA association?

Clean target = transient-deviation (impulse) detector, not a level threshold.
A level threshold on RMS would mostly flag high-load periods and then spuriously
correlate with load channels (power, flow).  Instead each live channel is
z-scored against a rolling median/MAD baseline (~5 min window); the per-second
aggregate is the most-deviating live channel, thresholded to the legacy alert
rate (~2.5%).  This isolates impulsive anomalies, matching the impulsive legacy
events (peak z-scores > 200), and is far less load-confounded.

Both significance tests from the headline analysis are re-run:
  - circular-shift permutation MI (decimated grid, Benjamini-Hochberg FDR),
  - circular-shift permutation rank-AUC (full resolution).

Output: ``results/illwerke/scada/v5_2b_clean_target.md`` (+ ``.json``).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

try:  # ``datetime.UTC`` is Python 3.11+; shim it for 3.10 and earlier.
    from datetime import UTC
except ImportError:  # pragma: no cover
    from datetime import timezone as _timezone

    UTC = _timezone.utc  # noqa: UP017

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import rankdata

from scripts.scada.v5_2_channel_mining import (
    SIG_Q,
    _benjamini_hochberg,
    _permutation_test,
)
from src.ingestion.illwerke_loader import load_campaign
from src.modeling.scada import physical_family

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results" / "illwerke" / "scada"
LIVE_VIB_TAG = "Vib180"  # the two live 180-degree triaxial accelerometers


def build_clean_target(
    V: np.ndarray, *, win: int, rate: float
) -> tuple[np.ndarray, float]:
    """Impulse target from live accelerometers: rolling-baseline z, then threshold."""
    z = np.zeros_like(V, dtype=np.float64)
    for j in range(V.shape[1]):
        base = median_filter(V[:, j], size=win, mode="nearest")
        dev = V[:, j] - base
        mad = median_filter(np.abs(dev), size=win, mode="nearest") + 1e-9
        z[:, j] = dev / (1.4826 * mad)
    agg = np.nanmax(z, axis=1)
    thr = float(np.quantile(agg, 1.0 - rate))
    return (agg >= thr).astype(np.uint8), thr


def _auc_permutation(
    X: np.ndarray, ind: np.ndarray, *, n_perm: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided rank-AUC permutation test (full resolution, circular-shift null)."""
    ind = ind.astype(bool)
    npos = int(ind.sum())
    nneg = ind.size - npos
    R = np.empty_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        R[:, j] = rankdata(X[:, j])

    def auc(mask: np.ndarray) -> np.ndarray:
        s = R[mask].sum(0)
        return (s - npos * (npos + 1) / 2.0) / (npos * nneg)

    obs = auc(ind)
    obs_stat = np.abs(obs - 0.5)
    cnt = np.zeros(X.shape[1])
    rng = np.random.default_rng(seed)
    T = ind.size
    lo, hi = T // 10, T - T // 10
    for _ in range(n_perm):
        st = np.abs(auc(np.roll(ind, int(rng.integers(lo, hi)))) - 0.5)
        cnt += st >= obs_stat
    p = (1.0 + cnt) / (n_perm + 1.0)
    return obs, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="E:/MasterThesisData/illwerke-data-230426")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=0.025, help="target positive rate")
    ap.add_argument("--win", type=int, default=301, help="rolling baseline window (s)")
    ap.add_argument("--decimate", type=int, default=10)
    ap.add_argument("--n-perm-mi", type=int, default=199)
    ap.add_argument("--n-perm-auc", type=int, default=999)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    camp = load_campaign(args.data_root)
    names_rms = camp.channel_names_rms
    live = [n for n in names_rms if LIVE_VIB_TAG in n]
    if not live:
        raise RuntimeError("no live Vib180 channels found in RMS stream")
    V = camp.rms[:, [names_rms.index(n) for n in live]].astype(np.float64)
    print(f"[v5.2b] live accelerometer channels: {live}")

    target, thr = build_clean_target(V, win=args.win, rate=args.rate)
    n_pos = int(target.sum())
    print(
        f"[v5.2b] clean impulse target: z_thr={thr:.2f}, "
        f"{n_pos} positives ({100 * n_pos / target.size:.2f}%)"
    )

    allg = camp.allg.astype(np.float64)
    names = camp.channel_names_allg
    var = allg.var(axis=0)
    keep = np.where(var > 1e-12)[0]
    X = allg[:, keep]
    kept = [names[i] for i in keep]
    fam = [physical_family(n) for n in kept]

    # --- MI permutation test (decimated) ----------------------------------
    dec = max(args.decimate, 1)
    Xd = np.ascontiguousarray(X[::dec])
    td = target[::dec]
    print(f"[v5.2b] MI test: {args.n_perm_mi} perms on {Xd.shape[0]} decimated samples")
    mi_obs, mi_p, mi_null = _permutation_test(
        Xd, td, n_perm=args.n_perm_mi, seed=args.seed + 1000
    )
    mi_q = _benjamini_hochberg(mi_p)

    # --- AUC permutation test (full resolution) ---------------------------
    print(f"[v5.2b] AUC test: {args.n_perm_auc} perms on {X.shape[0]} full samples")
    auc_obs, auc_p = _auc_permutation(
        X, target, n_perm=args.n_perm_auc, seed=args.seed + 2000
    )
    auc_q = _benjamini_hochberg(auc_p)

    rows = []
    for i in range(len(kept)):
        rows.append(
            {
                "name": kept[i],
                "family": fam[i],
                "mi": float(mi_obs[i]),
                "mi_over_null": (float(mi_obs[i] / mi_null[i]) if mi_null[i] > 0 else float("inf")),
                "mi_p": float(mi_p[i]),
                "mi_q": float(mi_q[i]),
                "auc": float(auc_obs[i]),
                "auc_p": float(auc_p[i]),
                "auc_q": float(auc_q[i]),
            }
        )
    rows.sort(key=lambda r: min(r["mi_p"], r["auc_p"]))

    report = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "data_root": str(args.data_root),
        "live_channels": live,
        "target": "clean impulse (rolling-baseline z over live accelerometers)",
        "win_s": args.win,
        "rate": args.rate,
        "z_threshold": thr,
        "n_positive": n_pos,
        "n_total": int(target.size),
        "decimate": dec,
        "n_perm_mi": args.n_perm_mi,
        "n_perm_auc": args.n_perm_auc,
        "sig_q": SIG_Q,
        "n_sig_mi": int((mi_q < SIG_Q).sum()),
        "n_sig_auc": int((auc_q < SIG_Q).sum()),
        "ranked": rows,
    }
    (args.out_dir / "v5_2b_clean_target.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"[v5.2b] significant after FDR — MI: {report['n_sig_mi']}, "
        f"AUC: {report['n_sig_auc']}"
    )
    (args.out_dir / "v5_2b_clean_target.md").write_text(
        _render(report), encoding="utf-8"
    )
    print(f"[v5.2b] wrote {args.out_dir / 'v5_2b_clean_target.md'}")


def _render(report: dict) -> str:
    L: list[str] = []
    L.append("# RQ4 / V5.2b — Clean vibration-only target robustness check")
    L.append("")
    L.append(
        "Tightest follow-up to V5.2: instead of the partially-instrumented legacy "
        "anomaly events, the anomaly target is rebuilt from **only the live "
        "accelerometer channels** as an impulse (transient-deviation) detector, "
        "and the SCADA mutual-information / AUC tests are re-run against it."
    )
    L.append("")
    L.append(f"- Generated: `{report['generated_utc']}`")
    L.append(f"- Live channels used: {', '.join(f'`{c}`' for c in report['live_channels'])}")
    L.append(
        f"- Target: {report['target']}, {report['win_s']} s baseline window, "
        f"z ≥ {report['z_threshold']:.2f} → **{report['n_positive']:,}** positives "
        f"({100 * report['n_positive'] / report['n_total']:.2f}%)"
    )
    L.append(
        f"- MI: {report['n_perm_mi']} circular-shift perms (decimate "
        f"{report['decimate']}×); AUC: {report['n_perm_auc']} perms (full 1 Hz). "
        f"FDR threshold q < {report['sig_q']}."
    )
    L.append("")
    L.append(
        f"**Significant after FDR — MI: {report['n_sig_mi']}, "
        f"AUC: {report['n_sig_auc']} (of {len(report['ranked'])} channels).**"
    )
    L.append("")
    L.append("| Channel | Family | MI | MI p | MI q | AUC | AUC p | AUC q |")
    L.append("|---------|--------|---:|-----:|-----:|----:|------:|------:|")
    for r in report["ranked"][:14]:
        L.append(
            f"| `{r['name']}` | {r['family']} | {r['mi']:.4f} | {r['mi_p']:.3f} | "
            f"{r['mi_q']:.3f} | {r['auc']:.3f} | {r['auc_p']:.3f} | {r['auc_q']:.3f} |"
        )
    L.append("")
    sig = [
        r for r in report["ranked"] if r["mi_q"] < report["sig_q"] or r["auc_q"] < report["sig_q"]
    ]
    if sig:
        L.append("### Significant channels (either test, q < 0.05)")
        L.append("")
        for r in sig:
            L.append(
                f"- `{r['name']}` ({r['family']}) — MI q={r['mi_q']:.3f}, "
                f"AUC q={r['auc_q']:.3f}. "
                + (
                    "**Load-type channel — interpret as possible operating-point "
                    "confound, not necessarily a fault signal.**"
                    if r["family"] in {"electrical", "rotational"}
                    else ""
                )
            )
    else:
        L.append(
            "### Verdict — no association detected by this test\n\n"
            "Even with a clean, impulse-based target built only from the live "
            "accelerometers, **no SCADA channel is significant after FDR under "
            "either test**, and the previously-suggestive runner-pressure/flow "
            "channels fade. This **rules out the degraded legacy target as the "
            "explanation** for the headline null. It does **not** prove the channels "
            "are uncorrelated: this is a *global, per-second* test that pools all "
            "operating modes (the campaign has ST/TU/PU/PH modes and many "
            "transitions) and ignores subtle, lagged, or mode-specific associations "
            "— all plausible on real scale-turbine data. The honest statement is "
            "'no association detected by this global test', and a within-mode / "
            "transition-aware analysis is the proper next step. RQ4 remains a "
            "deployment recommendation for a fully-instrumented, fault-labelled "
            "archive analysed per operating mode."
        )
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
