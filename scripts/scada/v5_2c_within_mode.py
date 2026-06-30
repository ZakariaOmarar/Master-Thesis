"""V5.2c — SCADA value for mode identification + within-mode anomaly test.

Two analyses on the real ROW II `ROWII_Allg_M1` SCADA archive:

Part A — SCADA identifies the operating mode (the positive RQ4 contribution).
    The supervisory vector `s_t` exists to tell the detector *what operating
    state the machine is in*, so mode transitions are not mistaken for
    anomalies. This measures the mutual information between each SCADA channel
    and the 4-class operating mode (ST/TU/PU/PH). High MI here is the useful
    result: SCADA is an excellent operating-state descriptor (indeed the legacy
    physics oracle *defines* the mode from these very channels — RPM, power,
    gate, valve, voltage, excitation; see `signal_thresholds.json`).

Part B — within-mode stratified anomaly test (the confound-controlled follow-up).
    The global V5.2 test pooled all operating modes, where between-mode SCADA
    variance can swamp a within-mode anomaly signal. This restricts to each mode
    in turn and re-runs the permutation MI / AUC tests of SCADA channels against
    the anomaly indicator, so the question becomes "within a single operating
    state, does any SCADA channel separate anomaly seconds from normal ones?".

Output: ``results/illwerke/scada/v5_2c_within_mode.md`` (+ ``.json``).
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
from sklearn.feature_selection import mutual_info_classif

from scripts.scada.v5_2_channel_mining import (
    SIG_Q,
    _benjamini_hochberg,
    _permutation_test,
)
from scripts.scada.v5_2b_clean_target import _auc_permutation
from src.ingestion.illwerke_loader import load_allg_campaign
from src.modeling.scada import (
    anomaly_indicator,
    load_anomaly_events,
    physical_family,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE = REPO_ROOT / "results" / "illwerke" / "pipeline"
DEFAULT_OUT = REPO_ROOT / "results" / "illwerke" / "scada"
MIN_POS = 150  # minimum within-mode positive seconds to bother testing a mode


def _state_label_map() -> dict[int, str]:
    mt = json.loads((PIPE / "mode_timeline.json").read_text())
    out: dict[int, str] = {}
    for seg in mt:
        out.setdefault(int(seg["state_id"]), str(seg.get("label", f"S{seg['state_id']}")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="E:/MasterThesisData/illwerke-data-230426")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decimate", type=int, default=10)
    ap.add_argument("--n-perm-mi", type=int, default=199)
    ap.add_argument("--n-perm-auc", type=int, default=999)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ts, allg, names = load_allg_campaign(args.data_root)
    var = allg.var(axis=0)
    keep = np.where(var > 1e-12)[0]
    X = allg[:, keep].astype(np.float64)
    kept = [names[i] for i in keep]
    fam = [physical_family(n) for n in kept]

    oracle = np.load(PIPE / "oracle_labels.npy")
    n = min(len(oracle), X.shape[0])
    X, oracle, ts = X[:n], oracle[:n], ts[:n]
    label_map = _state_label_map()

    events = load_anomaly_events(PIPE / "anomaly_events.json")
    ind_alert = anomaly_indicator(ts, events, severity_set=("alert",))

    report: dict = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "data_root": str(args.data_root),
        "n_total": int(n),
        "seed": args.seed,
        "decimate": args.decimate,
        "n_perm_mi": args.n_perm_mi,
        "n_perm_auc": args.n_perm_auc,
        "sig_q": SIG_Q,
    }

    # ---------------- Part A: SCADA -> operating mode ----------------------
    print("[v5.2c] Part A: MI(SCADA channel, operating mode)")
    mi_mode = mutual_info_classif(X, oracle, random_state=args.seed)
    order = np.argsort(mi_mode)[::-1]
    part_a = [
        {"name": kept[i], "family": fam[i], "mi_mode": float(mi_mode[i])}
        for i in order
    ]
    report["mode_identification"] = {
        "n_modes": int(len(np.unique(oracle))),
        "ranked": part_a,
    }
    print("  top channels by MI with mode:",
          [(r["name"], round(r["mi_mode"], 3)) for r in part_a[:5]])

    # ---------------- Part B: within-mode anomaly test --------------------
    print("[v5.2c] Part B: within-mode stratified anomaly test")
    modes_present, counts = np.unique(oracle, return_counts=True)
    dec = max(args.decimate, 1)
    report["within_mode"] = {}
    for m, cnt in zip(modes_present.tolist(), counts.tolist(), strict=True):
        lbl = label_map.get(m, f"S{m}")
        mask = oracle == m
        ind_m = ind_alert[mask]
        npos = int(ind_m.sum())
        if npos < MIN_POS:
            print(f"  mode {lbl} (id {m}): {npos} positives — SKIP (< {MIN_POS})")
            report["within_mode"][lbl] = {"state_id": m, "n": int(cnt),
                                          "n_positive": npos, "tested": False}
            continue
        Xm = X[mask]
        print(f"  mode {lbl} (id {m}): n={cnt}, positives={npos} — testing...")

        Xd, td = np.ascontiguousarray(Xm[::dec]), ind_m[::dec]
        if td.sum() < 20:  # too few positives survive decimation for MI
            mi_obs = mi_p = mi_q = np.full(Xm.shape[1], np.nan)
        else:
            mi_obs, mi_p, _ = _permutation_test(
                Xd, td, n_perm=args.n_perm_mi, seed=args.seed + 1000 + m
            )
            mi_q = _benjamini_hochberg(mi_p)
        auc_obs, auc_p = _auc_permutation(
            Xm, ind_m, n_perm=args.n_perm_auc, seed=args.seed + 2000 + m
        )
        auc_q = _benjamini_hochberg(auc_p)

        rows = []
        for i in range(len(kept)):
            rows.append({
                "name": kept[i], "family": fam[i],
                "mi": float(mi_obs[i]) if np.isfinite(mi_obs[i]) else None,
                "mi_q": float(mi_q[i]) if np.isfinite(mi_q[i]) else None,
                "auc": float(auc_obs[i]),
                "auc_p": float(auc_p[i]), "auc_q": float(auc_q[i]),
            })
        rows.sort(key=lambda r: (r["auc_q"] if r["mi_q"] is None else min(r["mi_q"], r["auc_q"])))
        n_sig = sum(
            1 for r in rows
            if (r["mi_q"] is not None and r["mi_q"] < SIG_Q) or r["auc_q"] < SIG_Q
        )
        print(f"    -> {n_sig} significant after FDR (either test)")
        report["within_mode"][lbl] = {
            "state_id": m, "n": int(cnt), "n_positive": npos,
            "tested": True, "n_significant": n_sig, "ranked": rows,
        }

    (args.out_dir / "v5_2c_within_mode.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir / "v5_2c_within_mode.md").write_text(_render(report), encoding="utf-8")
    print(f"[v5.2c] wrote {args.out_dir / 'v5_2c_within_mode.md'}")


def _render(report: dict) -> str:
    L: list[str] = []
    L.append("# RQ4 / V5.2c — SCADA mode-identification value + within-mode anomaly test")
    L.append("")
    L.append(f"- Generated: `{report['generated_utc']}`")
    L.append(f"- Grid: {report['n_total']:,} samples @ 1 Hz")
    L.append(
        f"- Significance: MI {report['n_perm_mi']} perms (decimate "
        f"{report['decimate']}×), AUC {report['n_perm_auc']} perms (full); "
        f"FDR q < {report['sig_q']}."
    )

    # Part A
    ma = report["mode_identification"]
    L.append("")
    L.append("## Part A — SCADA identifies the operating mode (positive result)")
    L.append("")
    L.append(
        f"Mutual information between each SCADA channel and the {ma['n_modes']}-state "
        "operating mode. This is the role the supervisory vector `s_t` plays in the "
        "architecture: knowing the operating state lets the detector treat a mode "
        "change as a *context shift*, not an anomaly. The operating mode is in fact "
        "*defined* from these channels by the physics oracle (RPM, power, gate, "
        "valve, voltage, excitation — `signal_thresholds.json`), so strong MI here "
        "is the expected and useful confirmation."
    )
    L.append("")
    L.append("| Channel | Family | MI with mode (nats) |")
    L.append("|---------|--------|--------------------:|")
    for r in ma["ranked"][:12]:
        L.append(f"| `{r['name']}` | {r['family']} | {r['mi_mode']:.3f} |")
    L.append("")

    # Part B
    L.append("## Part B — within-mode stratified anomaly test")
    L.append("")
    L.append(
        "Restricting to one operating mode at a time removes the between-mode "
        "variance that the global V5.2 test could not separate from a within-mode "
        "anomaly signal."
    )
    for lbl, d in report["within_mode"].items():
        L.append("")
        if not d["tested"]:
            L.append(
                f"### Mode `{lbl}` — not tested (n={d['n']:,}, "
                f"{d['n_positive']} anomaly s < threshold)"
            )
            continue
        L.append(
            f"### Mode `{lbl}` — n={d['n']:,}, {d['n_positive']} anomaly s — "
            f"**{d['n_significant']} significant after FDR**"
        )
        L.append("")
        L.append("| Channel | Family | MI | MI q | AUC | AUC q |")
        L.append("|---------|--------|---:|-----:|----:|------:|")
        for r in d["ranked"][:8]:
            mi = "—" if r["mi"] is None else f"{r['mi']:.4f}"
            miq = "—" if r["mi_q"] is None else f"{r['mi_q']:.3f}"
            L.append(
                f"| `{r['name']}` | {r['family']} | {mi} | {miq} | "
                f"{r['auc']:.3f} | {r['auc_q']:.3f} |"
            )

    # Verdict
    any_sig = any(
        d.get("tested") and d.get("n_significant", 0) > 0
        for d in report["within_mode"].values()
    )
    L.append("")
    L.append("## Verdict")
    L.append("")
    if any_sig:
        L.append(
            "Within at least one operating mode, a SCADA channel separates anomaly "
            "seconds from normal ones after FDR — a signal the global test could not "
            "see. See the per-mode tables above for the channels and modes; these are "
            "the concrete supervisory candidates for a field deployment."
        )
    else:
        L.append(
            "No SCADA channel is significant within any individual operating mode "
            "either. The strong, clear result on this archive is **Part A**: SCADA "
            "is an excellent operating-state descriptor. Its value to the pipeline is "
            "therefore as the *context* signal `s_t` (so mode changes are not flagged "
            "as anomalies), not as a standalone anomaly predictor — which is exactly "
            "how the architecture uses it. Detecting the specific (subtle, possibly "
            "transition-bound) anomalies from SCADA alone would need a "
            "fully-instrumented, fault-labelled archive."
        )
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
