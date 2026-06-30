"""Figure 32: ROW II SCADA timeline excerpt with mode shading + anomaly ticks.

Requires the raw Illwerke archive (the ``Allg_M1`` process channels), which
lives on the external drive, e.g. ``E:/MasterThesisData/illwerke-data-230426``.
The mode timeline and anomaly events are read from the committed pipeline
artefacts in ``results/illwerke/pipeline``.

Run with (on the machine that has the archive):
  python -m scripts.figures.fig_scada --data-root E:/MasterThesisData/illwerke-data-230426
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures import style
from scripts.figures.style import ANOMALY, REPO_ROOT, save

PIPE = REPO_ROOT / "results" / "illwerke" / "pipeline"
DEFAULT_ROOT = Path("E:/MasterThesisData/illwerke-data-230426")

CHANNELS = ["1_P_Ist", "1_Drehzahl_Ist", "1_Leitapparat Stell."]
CHANNEL_LABELS = ["active power (MW)", "speed (rpm)", "guide vane (%)"]
MODE_SHADE = {"ST": "#f0f0f0", "TU": "#d7e8f4", "PU": "#fbe3d5", "PH": "#e8e0f0"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--start-h", type=float, default=20.0, help="excerpt start (hours)")
    ap.add_argument("--hours", type=float, default=8.0, help="excerpt length (hours)")
    args = ap.parse_args()

    if not args.data_root.exists():
        print(
            f"SKIPPED: raw SCADA archive not found at {args.data_root}.\n"
            "fig32_scada_timeline needs the Illwerke Allg_M1 channels; run this\n"
            "script on the machine with the external drive:\n"
            "  python -m scripts.figures.fig_scada --data-root <path-to-illwerke-data>"
        )
        sys.exit(1)

    from src.ingestion.illwerke_loader import load_allg_campaign

    style.apply_style()
    ts_ns, allg, names = load_allg_campaign(args.data_root)
    t_s = (ts_ns - ts_ns[0]) / 1e9
    i0 = int(np.searchsorted(t_s, args.start_h * 3600))
    i1 = int(np.searchsorted(t_s, (args.start_h + args.hours) * 3600))
    t_h = t_s[i0:i1] / 3600.0

    modes = json.loads((PIPE / "mode_timeline.json").read_text())
    events = json.loads((PIPE / "anomaly_events.json").read_text())

    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(6.8, 4.0), sharex=True)
    for ax, ch, lbl in zip(axes, CHANNELS, CHANNEL_LABELS):
        try:
            col = names.index(ch)
        except ValueError:
            ax.text(0.5, 0.5, f"channel {ch!r} not in archive", transform=ax.transAxes,
                    ha="center", fontsize=8, color="0.4")
            continue
        for seg in modes:
            a, b = seg["t_start_s"] / 3600.0, seg["t_end_s"] / 3600.0
            if b < t_h[0] or a > t_h[-1]:
                continue
            ax.axvspan(max(a, t_h[0]), min(b, t_h[-1]),
                       color=MODE_SHADE.get(seg["label"], "white"), zorder=0)
        ax.plot(t_h, allg[i0:i1, col], color="0.2", lw=0.7)
        for ev in events:
            a = ev["t_start_s"] / 3600.0
            if t_h[0] <= a <= t_h[-1]:
                ax.plot([a], [ax.get_ylim()[0]], marker="|", ms=9, mew=1.6,
                        color=ANOMALY, clip_on=False)
        ax.set_ylabel(lbl, fontsize=7.5)
    axes[-1].set_xlabel("time (h)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MODE_SHADE.values()]
    axes[0].legend(handles, MODE_SHADE.keys(), ncol=4, fontsize=7, frameon=False,
                   loc="upper right")
    fig.suptitle(
        "ROW II SCADA excerpt: the process channels track the operating mode almost\n"
        "deterministically (shading = mode timeline; red ticks = legacy anomaly events)",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "fig32_scada_timeline")


if __name__ == "__main__":
    main()
