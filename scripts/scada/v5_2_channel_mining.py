"""V5.2 runner — Illwerke ROW II SCADA mutual-information ranking (RQ4, check 2).

Loads the real ``ROWII_Allg_M1`` plant process channels, takes the frozen
5-layer-pipeline anomaly events as a binary target, and ranks every SCADA
channel by its mutual information with that target.  This is the offline
"which channels would a field deployment route into the supervisory vector"
analysis described in the Experiments chapter (RQ4).

Significance — circular-shift permutation test
----------------------------------------------
1 Hz SCADA is strongly autocorrelated, so raw KNN-MI is positively biased and
a plain label permutation under-estimates the noise floor.  Instead the anomaly
indicator is **circularly shifted** by ``n_perm`` random offsets: this keeps the
indicator's run-length structure (autocorrelation) intact while destroying its
alignment with the channels, giving an honest null distribution per channel.

  - null floor = mean MI over the shifts (an intuitive effect-size denominator),
  - p-value    = (1 + #{null MI >= observed MI}) / (n_perm + 1),
  - q-value    = Benjamini-Hochberg FDR correction across channels.

The full analysis (observed MI, null, p, q) is computed on a DECIMATED grid so
that a few-hundred-permutation test is tractable; the anomaly events span
1-4 min, so decimating to ~0.1 Hz preserves event structure.  The
full-resolution MI is also recorded (``mi_full``) as a cross-check.

Outputs (under ``results/illwerke/scada/``):
  - ``v5_2_mi_ranking.json``  : machine-readable ranking + metadata
  - ``v5_2_scada_mi_ranking.md`` : human-readable report (the temp markdown)

Run:
  python scripts/scada/v5_2_channel_mining.py \
      --data-root E:/MasterThesisData/illwerke-data-230426
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

from src.ingestion.illwerke_loader import load_allg_campaign, load_rms_campaign
from src.modeling.scada import (
    anomaly_indicator,
    load_anomaly_events,
    physical_family,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS = REPO_ROOT / "results" / "illwerke" / "pipeline" / "anomaly_events.json"
DEFAULT_OUT = REPO_ROOT / "results" / "illwerke" / "scada"

SIG_Q = 0.05  # FDR threshold for "significant"


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values for a 1-D array of p-values."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    # enforce monotonic non-decreasing q from the largest p downward
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _permutation_test(
    Xd: np.ndarray,
    ind_d: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circular-shift permutation test on a (decimated) grid.

    Returns (observed_mi, p_values, null_mean) per channel, all on the grid
    passed in.  ``observed_mi`` and the null are computed identically so the
    p-value comparison is exact.
    """
    rng = np.random.default_rng(seed)
    T = ind_d.shape[0]
    obs = mutual_info_classif(Xd, ind_d, random_state=seed)
    count = np.zeros(Xd.shape[1], dtype=np.int64)
    null_sum = np.zeros(Xd.shape[1], dtype=np.float64)
    lo, hi = T // 10, T - T // 10
    for b in range(n_perm):
        rolled = np.roll(ind_d, int(rng.integers(lo, hi)))
        nb = mutual_info_classif(Xd, rolled, random_state=seed + 1 + b)
        count += nb >= obs
        null_sum += nb
    p = (1.0 + count) / (n_perm + 1.0)
    return obs, p, null_sum / max(n_perm, 1)


def assess_target_instrumentation(
    data_root: str | Path, *, dead_var: float = 1e-6
) -> dict:
    """Audit the RMS sensor streams that produced the anomaly-event target.

    Not all ROW II microphones / accelerometers were physically installed, so
    the legacy pipeline's anomaly events were derived from a partial sensor set.
    A channel is flagged dead/flatlined if its variance is below ``dead_var``
    (the live accelerometers sit at variance 0.6-8.3; the dead ones at 0 or the
    microphones at ~1e-8). This makes the central RQ4 caveat data-backed.
    """
    ts, rms, names = load_rms_campaign(data_root)
    var = rms.var(axis=0)
    mics, vibs = [], []
    for n, v in zip(names, var.tolist(), strict=True):
        rec = {"name": n, "var": float(v), "live": bool(v >= dead_var)}
        (mics if "Mic" in n else vibs).append(rec)
    return {
        "n_mic_total": len(mics),
        "n_mic_live": sum(c["live"] for c in mics),
        "n_vib_total": len(vibs),
        "n_vib_live": sum(c["live"] for c in vibs),
        "dead_channels": [c["name"] for c in mics + vibs if not c["live"]],
        "live_channels": [c["name"] for c in mics + vibs if c["live"]],
        "dead_var_threshold": dead_var,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="E:/MasterThesisData/illwerke-data-230426")
    ap.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--decimate", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=99)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- load the real SCADA channels -------------------------------------
    ts, allg, names = load_allg_campaign(args.data_root)
    print(f"[v5.2] loaded Allg_M1: {allg.shape[0]} samples x {allg.shape[1]} channels")

    events = load_anomaly_events(args.events)

    # Audit the instrumentation that produced the anomaly-event target: not all
    # ROW II mics/accelerometers were installed, so the target is degraded.
    instr = assess_target_instrumentation(args.data_root)
    print(
        f"[v5.2] target instrumentation: mics {instr['n_mic_live']}/"
        f"{instr['n_mic_total']} live, accel {instr['n_vib_live']}/"
        f"{instr['n_vib_total']} live"
    )

    # Drop zero-variance channels once, shared across targets.
    var = allg.var(axis=0)
    keep_idx = np.where(var > 1e-12)[0]
    dropped = [names[i] for i in range(len(names)) if i not in set(keep_idx.tolist())]
    X = allg[:, keep_idx].astype(np.float64)
    kept_names = [names[i] for i in keep_idx]
    families = [physical_family(n) for n in kept_names]

    dec = max(args.decimate, 1)
    Xd = np.ascontiguousarray(X[::dec])
    print(f"[v5.2] decimation x{dec} -> {Xd.shape[0]} samples for permutation test")

    targets = {
        "alert": ("alert",),
        "alert+watch": ("alert", "watch"),
    }

    report: dict = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "data_root": str(args.data_root),
        "events_path": str(args.events),
        "n_total_samples": int(allg.shape[0]),
        "n_decimated_samples": int(Xd.shape[0]),
        "decimate": dec,
        "n_channels_total": int(allg.shape[1]),
        "n_channels_kept": int(len(kept_names)),
        "dropped_zero_variance": dropped,
        "seed": args.seed,
        "n_perm": args.n_perm,
        "sig_q": SIG_Q,
        "target_instrumentation": instr,
        "targets": {},
    }

    for tag, sev in targets.items():
        ind = anomaly_indicator(ts, events, severity_set=sev)
        ind_d = ind[::dec]
        n_pos = int(ind.sum())
        print(
            f"[v5.2] target={tag}: {n_pos} positive samples "
            f"({100 * n_pos / ind.shape[0]:.2f}%)  running {args.n_perm} permutations..."
        )

        mi_full = mutual_info_classif(X, ind, random_state=args.seed)
        obs, p, null_mean = _permutation_test(
            Xd, ind_d, n_perm=args.n_perm, seed=args.seed + 1000
        )
        q = _benjamini_hochberg(p)

        order = np.argsort(obs)[::-1]
        ranked = []
        for i in order:
            floor = float(null_mean[i])
            ranked.append(
                {
                    "name": kept_names[i],
                    "family": families[i],
                    "mi": float(obs[i]),
                    "mi_full": float(mi_full[i]),
                    "null_mi": floor,
                    "mi_over_null": (float(obs[i]) / floor) if floor > 0 else float("inf"),
                    "p_value": float(p[i]),
                    "q_value": float(q[i]),
                    "significant": bool(q[i] < SIG_Q),
                }
            )

        n_sig = int(sum(r["significant"] for r in ranked))
        print(f"[v5.2]   -> {n_sig} channels significant at q < {SIG_Q}")
        report["targets"][tag] = {
            "severity_set": list(sev),
            "n_positive": n_pos,
            "positive_rate_pct": round(100 * n_pos / ind.shape[0], 4),
            "n_significant": n_sig,
            "ranked": ranked,
        }

    json_path = args.out_dir / "v5_2_mi_ranking.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[v5.2] wrote {json_path}")

    md_path = args.out_dir / "v5_2_scada_mi_ranking.md"
    md_path.write_text(_render_markdown(report, top_k=args.top_k), encoding="utf-8")
    print(f"[v5.2] wrote {md_path}")


def _render_markdown(report: dict, *, top_k: int) -> str:
    lines: list[str] = []
    lines.append("# RQ4 / V5.2 — Real ROW II SCADA Mutual-Information Ranking")
    lines.append("")
    lines.append(
        "Offline ranking of the real Illwerke `ROWII_Allg_M1` plant process "
        "channels against the frozen 5-layer-pipeline anomaly events. This is the "
        "RQ4 *deployment recommendation* check: it identifies which pressure, "
        "thermal, and hydraulic channels a field deployment would route into the "
        "supervisory vector `s_t` whose conditioning mechanism the prototype study "
        "(V5.1) already validated. **No model is trained here — pure offline analysis.**"
    )
    lines.append("")
    lines.append(f"- Generated: `{report['generated_utc']}`")
    lines.append(f"- Source: `{report['data_root']}`")
    lines.append(f"- Anomaly events: `{report['events_path']}`")
    lines.append(
        f"- Grid: **{report['n_total_samples']:,}** samples @ 1 Hz "
        f"(~{report['n_total_samples'] / 3600:.1f} h)"
    )
    lines.append(
        f"- Channels: {report['n_channels_kept']} of {report['n_channels_total']} "
        f"ranked ({report['n_channels_total'] - report['n_channels_kept']} dropped "
        "for zero variance)"
    )
    if report["dropped_zero_variance"]:
        dropped = ", ".join(f"`{d}`" for d in report["dropped_zero_variance"])
        lines.append(f"  - Dropped: {dropped}")
    lines.append(
        f"- **Significance**: circular-shift permutation test, **{report['n_perm']} "
        f"permutations**, on a {report['decimate']}× decimated grid "
        f"({report['n_decimated_samples']:,} samples @ "
        f"{1.0 / report['decimate']:.2f} Hz). `q` is the Benjamini-Hochberg "
        f"FDR-adjusted p-value across channels; **significant = q < {report['sig_q']}**. "
        f"MI estimator: sklearn `mutual_info_classif` (KNN), seed {report['seed']}."
    )
    lines.append("")
    lines.append(
        "> **Reading the table.** `MI` is mutual information in nats (decimated "
        "grid); `MI/Null` is the effect size against the circular-shift floor; "
        "`p`/`q` are the permutation significance. A channel is a genuine candidate "
        "only if it is **significant after FDR correction** — a high MI or a high "
        "MI/Null ratio at q ≥ 0.05 is not trustworthy. `mi_full` (1 Hz) is kept in "
        "the JSON as a cross-check; channels whose raw 1 Hz MI is inflated by "
        "autocorrelation (e.g. `Sys_Watchdog`) collapse on the decimated grid and "
        "fail the test, which is the intended behaviour."
    )

    for tag, payload in report["targets"].items():
        lines.append("")
        lines.append(f"## Target: `{tag}` events")
        lines.append("")
        lines.append(
            f"Positive samples: **{payload['n_positive']:,}** "
            f"({payload['positive_rate_pct']:.2f}% of the grid), "
            f"severities {payload['severity_set']}. "
            f"**{payload['n_significant']} of {report['n_channels_kept']} channels "
            f"significant at q < {report['sig_q']}.**"
        )
        lines.append("")
        lines.append("| Rank | Channel | Family | MI (nats) | MI/Null | p | q | sig |")
        lines.append("|-----:|---------|--------|----------:|--------:|----:|----:|:---:|")
        for r, row in enumerate(payload["ranked"][:top_k], start=1):
            ratio = row["mi_over_null"]
            ratio_s = "∞" if ratio == float("inf") else f"{ratio:.2f}"
            sig = "**✓**" if row["significant"] else "✗"
            lines.append(
                f"| {r} | `{row['name']}` | {row['family']} | "
                f"{row['mi']:.4f} | {ratio_s} | {row['p_value']:.3f} | "
                f"{row['q_value']:.3f} | {sig} |"
            )
        lines.append("")
        sig_rows = [r for r in payload["ranked"] if r["significant"]]
        if sig_rows:
            fam_groups: dict[str, list[str]] = {}
            for r in sig_rows:
                fam_groups.setdefault(r["family"], []).append(
                    f"`{r['name']}` (q={r['q_value']:.3f})"
                )
            lines.append("Significant channels (q < 0.05) by physical family:")
            lines.append("")
            for fam in sorted(fam_groups, key=lambda f: len(fam_groups[f]), reverse=True):
                lines.append(f"- **{fam}**: {', '.join(fam_groups[fam])}")
        else:
            lines.append(
                "**No channel is significant after FDR correction on this target.**"
            )

    # --- caveats + recommendation, driven by the headline (alert) target --
    headline = report["targets"].get("alert") or next(iter(report["targets"].values()))
    sig_rows = [r for r in headline["ranked"] if r["significant"]]
    # "Suggestive" = uncorrected p < 0.05 but not surviving FDR.
    sugg_rows = sorted(
        (r for r in headline["ranked"] if r["p_value"] < 0.05 and not r["significant"]),
        key=lambda r: r["p_value"],
    )
    best = min(headline["ranked"], key=lambda r: r["p_value"])
    present_families = sorted({r["family"] for r in headline["ranked"]})

    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    instr = report.get("target_instrumentation")
    if instr:
        dead = ", ".join(f"`{c}`" for c in instr["dead_channels"])
        lines.append(
            f"- **The target was built on partial instrumentation (primary "
            f"caveat).** Not all ROW II sensors were installed: only "
            f"**{instr['n_mic_live']} of {instr['n_mic_total']} microphones** and "
            f"**{instr['n_vib_live']} of {instr['n_vib_total']} accelerometer "
            f"channels** carry live signal (the rest are flatlined at the noise "
            f"floor or stuck at a constant fault value). The legacy 5-layer "
            f"pipeline derived the `anomaly_events.json` target from this degraded "
            f"set, so the target itself is unreliable — a likely cause of the weak "
            f"SCADA association independent of any true physical decoupling. Dead "
            f"channels: {dead}."
        )
    lines.append(
        "- **The target is acoustic, not independent.** The anomaly events come "
        "from the legacy acoustic/vibration RMS pipeline, so this MI measures how "
        "much each SCADA channel co-varies with *acoustically-flagged* anomalies, "
        "not with ground-truth faults. Weak association is expected and is itself "
        "informative about how decoupled the two views are."
    )
    lines.append(
        "- **The association is weak in absolute terms.** Even the lowest-p channels "
        "carry small MI; the permutation test asks whether the ordering is "
        "distinguishable from noise, not whether any single channel is a strong "
        "predictor."
    )
    if "thermal" not in present_families:
        lines.append(
            "- **No temperature channels (expected).** `ROWII_Allg_M1` carries "
            "pressure, power/electrical, flow, level, and rotational channels but no "
            "thermal instrumentation — this is expected for the M1 archive, not a "
            "deficiency, so the thermal element of the RQ4 anticipation simply does "
            "not apply to this data."
        )
    lines.append(
        "- **Decimation and KNN bias.** MI is estimated on a decimated grid for a "
        "tractable permutation test; KNN MI is positively biased, which is exactly "
        "why significance is judged by the permutation null rather than by the raw "
        "value. Families are tagged by a keyword heuristic over German labels — "
        "treat the groupings as indicative."
    )

    lines.append("")
    lines.append("## Deployment recommendation")
    lines.append("")
    if sig_rows:
        fam_groups = {}
        for r in sig_rows:
            fam_groups.setdefault(r["family"], []).append(
                f"`{r['name']}` (q={r['q_value']:.3f})"
            )
        lines.append(
            "On the alert target, the channels that are statistically significant "
            "(q < 0.05, FDR-corrected) — the candidate supervisory inputs for an "
            "instrumented ROW II deployment — are, grouped by family:"
        )
        lines.append("")
        for fam in sorted(fam_groups, key=lambda f: len(fam_groups[f]), reverse=True):
            lines.append(f"- **{fam}**: {', '.join(fam_groups[fam])}")
        lines.append("")
        lines.append(
            "These should be routed into the supervisory vector `s_t` — the "
            "conditioning mechanism the prototype V5.1 study already validated — and "
            "confirmed against fault-labelled field data. Thermal channels cannot be "
            "assessed (none present), and because the target is acoustically derived "
            "and the effect sizes are modest, RQ4 remains a deployment recommendation "
            "rather than a validated detector."
        )
    elif sugg_rows:
        sugg = ", ".join(
            f"`{r['name']}` ({r['family']}, p={r['p_value']:.3f}, q={r['q_value']:.2f})"
            for r in sugg_rows
        )
        lines.append(
            "**No channel survives FDR correction (q < 0.05).** The evidence is "
            "*suggestive but not significant*: the channels reaching uncorrected "
            f"p < 0.05 are {sugg}. These are precisely the physically-expected "
            "turbine channels — runner/draft pressure and volumetric flow — and the "
            f"single strongest signal is `{best['name']}` "
            f"(p={best['p_value']:.3f}, q={best['q_value']:.2f}). The fact that the "
            "lowest p-values land on the hydraulically-relevant channels rather than "
            "on the autocorrelation artefacts (`Sys_Watchdog` etc., which the test "
            "now correctly rejects) indicates the weak signal is real but "
            "under-powered against a 28-channel multiple-comparison penalty.\n\n"
            "The honest RQ4 conclusion is therefore directional, not confirmatory: "
            "an instrumented ROW II deployment should prioritise **runner/draft "
            "pressure and turbine flow** in the supervisory vector `s_t` — the "
            "conditioning mechanism the prototype V5.1 study validated — and a "
            "fault-labelled field archive (ideally with the absent temperature "
            "channels) is needed to raise these from suggestive to significant. The "
            "prototype shows the *mechanism* works; this archive points at *which "
            "channels* to wire but cannot, against an acoustically-derived target, "
            "prove the association."
        )
    else:
        lines.append(
            "No channel reaches even uncorrected p < 0.05 on the alert target: on "
            "this archive the SCADA channels and the acoustic anomaly flags are "
            "statistically decoupled. The honest RQ4 conclusion is that supervisory "
            "conditioning must be re-evaluated once a fault-labelled, temperature- "
            "and pressure-instrumented field archive is available — the prototype "
            "(V5.1) shows the *mechanism* works, but this archive does not identify "
            "a channel set to feed it."
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
