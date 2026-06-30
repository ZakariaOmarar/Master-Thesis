"""Geometry figures: sensor arrays + knock positions, and the per-position
localization error map (figures 3 and 25 of the figure plan).

Reads the real on-disk geometry (``position.json`` for the circular rig,
``node_position.txt`` for the D2 rig) and the per-position LOPO breakdown
of ``results/reports/finalize_results_20260611_012822.json``.

Run with:  python -m scripts.figures.fig_geometry
"""

from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

from scripts.figures import style
from scripts.figures.style import (
    ACOUSTIC,
    ANOMALY,
    CAMPAIGN_COLORS,
    REPO_ROOT,
    VIBRATION,
    save,
)

FINALIZE = REPO_ROOT / "results" / "reports" / "finalize_results_20260611_012822.json"


# ── geometry loading (meters) ───────────────────────────────────────────
def circular_rig_sensors() -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    records = json.loads(
        (REPO_ROOT / "data" / "third_test_dataset" / "position.json").read_text()
    )
    mic_xyz, mic_ids, vib_xyz, vib_ids = [], [], [], []
    for r in records:
        xyz = np.array([r["x"], r["y"], r["z"]], dtype=float) / 100.0
        if str(r["id"]).startswith("(V)"):
            vib_xyz.append(xyz)
            vib_ids.append(str(r["id"])[3:])
        else:
            mic_xyz.append(xyz)
            mic_ids.append(str(r["id"]))
    return np.array(mic_xyz), mic_ids, np.array(vib_xyz), vib_ids


def d2_rig_sensors() -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    txt = (REPO_ROOT / "data" / "second_test_dataset" / "node_position.txt").read_text(
        encoding="utf-8", errors="ignore"
    )
    mic_xyz, mic_ids, vib_xyz, vib_ids = [], [], [], []
    for line in txt.splitlines():
        m = re.match(
            r"\s*(vibration|microfone)_([A-Z]).*?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
            line,
        )
        if not m:
            continue
        kind, sid = m.group(1), m.group(2)
        xyz = np.array([float(m.group(3)), float(m.group(4)), float(m.group(5))]) / 100.0
        if kind == "vibration":
            vib_xyz.append(xyz)
            vib_ids.append(sid)
        else:
            mic_xyz.append(xyz)
            mic_ids.append(sid)
    return np.array(mic_xyz), mic_ids, np.array(vib_xyz), vib_ids


def circular_rig_knocks() -> dict[str, list[tuple[np.ndarray, bool]]]:
    """Knock positions (m) per campaign on the circular rig; bool = usable."""
    xyz_re = re.compile(
        r"^\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)$"
    )
    out: dict[str, list[tuple[np.ndarray, bool]]] = {"D3": [], "D4": [], "D5": []}
    # D3 instrumented hit: centroid of sensors Fl and Gr (published position).
    out["D3"].append((np.array([0.085, -0.025, 0.080]), True))
    for camp, root in [("D4", "fourth_test_dataset"), ("D5", "fifth_test_dataset")]:
        for p in sorted((REPO_ROOT / "data" / root).rglob("*")):
            if not p.is_dir():
                continue
            m = xyz_re.match(p.name)
            if m is None:
                continue
            xyz = np.array([float(m.group(i)) for i in (1, 2, 3)]) / 100.0
            n_accel = len({f.stem for f in p.glob("vibration_raw_*.csv")})
            usable = camp != "D4" or n_accel == 4
            out[camp].append((xyz, usable))
    return out


D2_KNOCKS = {  # folder-encoded, cm -> m; single-mode = usable labelled position
    "single": [np.array([0.10, 0.00, 0.23]), np.array([0.15, 0.06, 0.15]),
               np.array([0.15, 0.30, 0.15])],
    "dual": [np.array([0.00, 0.17, 0.12]), np.array([0.00, 0.40, 0.15])],
}


def draw_hull(ax, pts_2d: np.ndarray, color: str = "0.45") -> None:
    hull = ConvexHull(pts_2d)
    cycle = np.append(hull.vertices, hull.vertices[0])
    ax.fill(pts_2d[cycle, 0], pts_2d[cycle, 1], color=color, alpha=0.07, zorder=0)
    ax.plot(pts_2d[cycle, 0], pts_2d[cycle, 1], color=color, lw=1.0, ls="--",
            zorder=1, label="sensor convex hull")


def draw_sensors(ax, mic, vib, ix=0, iy=1):
    ax.scatter(mic[:, ix], mic[:, iy], c=ACOUSTIC, marker="o", s=26,
               zorder=4, label="microphone")
    ax.scatter(vib[:, ix], vib[:, iy], c=VIBRATION, marker="^", s=34,
               zorder=4, label="accelerometer")


# ─────────────────────────────────────────────────────────────────────────
# 3 — sensor-array geometry
# ─────────────────────────────────────────────────────────────────────────
def fig03_geometry() -> None:
    mic, mic_ids, vib, vib_ids = circular_rig_sensors()
    knocks = circular_rig_knocks()
    d2_mic, d2_mic_ids, d2_vib, d2_vib_ids = d2_rig_sensors()

    fig, (ax_xy, ax_xz, ax_d2) = plt.subplots(
        1, 3, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1.25, 1.0, 0.95]}
    )

    # — (a) circular rig, top view —
    all_xy = np.vstack([mic[:, :2], vib[:, :2]])
    draw_hull(ax_xy, all_xy)
    draw_sensors(ax_xy, mic, vib)
    # stagger the id labels: sensors sharing (x, y) get stacked offsets
    seen_xy: dict[tuple, int] = {}
    for sid, p, color in (
        [(s, p, ACOUSTIC) for s, p in zip(mic_ids, mic)]
        + [(s, p, VIBRATION) for s, p in zip(vib_ids, vib)]
    ):
        key = (round(p[0], 3), round(p[1], 3))
        k = seen_xy.get(key, 0)
        seen_xy[key] = k + 1
        dx, dy = (4, 2 + 6 * k) if p[0] >= 0.05 else (-4, 2 + 6 * k)
        ax_xy.annotate(sid, (p[0], p[1]), fontsize=5.2, color=color,
                       xytext=(dx, dy), textcoords="offset points",
                       ha="left" if dx > 0 else "right")
    for camp in ("D3", "D4", "D5"):
        for xyz, usable in knocks[camp]:
            ax_xy.scatter(
                xyz[0], xyz[1],
                facecolors=CAMPAIGN_COLORS[camp] if usable else "none",
                edgecolors="k" if usable else CAMPAIGN_COLORS[camp],
                marker="*", s=70, linewidths=0.6, zorder=5,
            )
    ax_xy.set_title("(a) circular rig (D3-D5), top view", fontsize=8.5)
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")

    # — (b) circular rig, side view —
    draw_sensors(ax_xz, mic, vib, ix=0, iy=2)
    for camp in ("D3", "D4", "D5"):
        for xyz, usable in knocks[camp]:
            ax_xz.scatter(
                xyz[0], xyz[2],
                facecolors=CAMPAIGN_COLORS[camp] if usable else "none",
                edgecolors="k" if usable else CAMPAIGN_COLORS[camp],
                marker="*", s=70, linewidths=0.6, zorder=5,
            )
    ax_xz.set_title("(b) circular rig, side view", fontsize=8.5)
    ax_xz.set_xlabel("x (m)")
    ax_xz.set_ylabel("z (m)")

    # — (c) D2 rectangular rig, top view —
    d2_all = np.vstack([d2_mic[:, :2], d2_vib[:, :2]])
    draw_hull(ax_d2, d2_all)
    draw_sensors(ax_d2, d2_mic, d2_vib)
    for p in D2_KNOCKS["single"]:
        ax_d2.scatter(p[0], p[1], facecolors=CAMPAIGN_COLORS["D2"], edgecolors="k",
                      marker="*", s=70, linewidths=0.6, zorder=5)
    for p in D2_KNOCKS["dual"]:
        ax_d2.scatter(p[0], p[1], facecolors="none", edgecolors=CAMPAIGN_COLORS["D2"],
                      marker="*", s=70, linewidths=1.0, zorder=5)
    ax_d2.set_title("(c) D2 rig, top view", fontsize=8.5)
    ax_d2.set_xlabel("x (m)")
    ax_d2.set_ylabel("y (m)")

    for ax in (ax_xy, ax_xz, ax_d2):
        ax.set_aspect("equal", adjustable="datalim")
        ax.axhline(0, color="0.88", lw=0.7, zorder=0)
        ax.axvline(0, color="0.88", lw=0.7, zorder=0)

    handles = [
        Line2D([], [], marker="o", ls="", color=ACOUSTIC, ms=5, label="microphone"),
        Line2D([], [], marker="^", ls="", color=VIBRATION, ms=6, label="accelerometer"),
        Line2D([], [], ls="--", color="0.45", label="convex hull"),
        Line2D([], [], marker="*", ls="", mfc=CAMPAIGN_COLORS["D3"], mec="k", ms=9, label="D3 hit"),
        Line2D([], [], marker="*", ls="", mfc=CAMPAIGN_COLORS["D4"], mec="k", ms=9, label="D4 knock"),
        Line2D([], [], marker="*", ls="", mfc=CAMPAIGN_COLORS["D5"], mec="k", ms=9, label="D5 knock"),
        Line2D([], [], marker="*", ls="", mfc=CAMPAIGN_COLORS["D2"], mec="k", ms=9, label="D2 knock"),
        Line2D([], [], marker="*", ls="", mfc="none", mec="0.4", ms=9, label="excluded"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=6.6,
               columnspacing=1.0, handletextpad=0.3, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, "fig03_sensor_array_geometry")


# ─────────────────────────────────────────────────────────────────────────
# 25 — per-position localization error map
# ─────────────────────────────────────────────────────────────────────────
def fig25_error_map() -> None:
    # Per-position LOPO folds of the canonical seed-42 run (the lopo_dir of
    # finalize_results_20260617_042101.json; tdoa-only mean 0.126 m, with a
    # five-seed median of 0.129 m in Table tab:res_rq3_lopo).
    folds = REPO_ROOT / "results" / "runs" / "20260615_112939__full_pipeline_b5_cma" / "lopo" / "folds.jsonl"
    mode = "both"

    mic, _, vib, _ = circular_rig_sensors()
    d2_mic, _, d2_vib, _ = d2_rig_sensors()
    d2_set = {(0.10, 0.00, 0.23), (0.15, 0.06, 0.15), (0.15, 0.30, 0.15)}

    rows = []
    with open(folds) as f:
        for line in f:
            r = json.loads(line)
            if r["channel_mode"] != mode:
                continue
            xyz = tuple(round(float(v), 2) for v in r["position_xyz"])
            rows.append((xyz, float(r["val_mae_3d_m"]), int(r["n_val_windows"])))

    fig, (ax_c, ax_d2) = plt.subplots(
        1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.5, 1.0]},
        constrained_layout=True,
    )
    vmin, vmax = 0.05, 0.36
    cmap = plt.get_cmap("RdYlGn_r")
    worst = {xyz for xyz, mae, _ in sorted(rows, key=lambda r: -r[1])[:4]}

    for ax, sensors_mic, sensors_vib in ((ax_c, mic, vib), (ax_d2, d2_mic, d2_vib)):
        all_xy = np.vstack([sensors_mic[:, :2], sensors_vib[:, :2]])
        draw_hull(ax, all_xy)
        draw_sensors(ax, sensors_mic, sensors_vib)

    seen: dict[tuple, int] = {}
    for xyz, mae, _n in rows:
        is_d2 = xyz in d2_set
        ax = ax_d2 if is_d2 else ax_c
        # offset duplicate coordinates slightly so both campaigns stay visible
        k = seen.get(xyz, 0)
        seen[xyz] = k + 1
        dx = 0.012 * k
        sc = ax.scatter(
            xyz[0] + dx, xyz[1], c=[mae], cmap=cmap, vmin=vmin, vmax=vmax,
            marker="*", s=240, edgecolors="k", linewidths=0.7, zorder=6,
        )
        if xyz in worst:  # annotate only the geometric failures
            # per-position offsets keep labels clear of markers and axes
            offsets = {
                (-0.20, 0.00, 0.00): (0, 11),
                (0.00, -0.20, 0.00): (0, -15),
                (-0.11, 0.00, 0.00): (0, 11),
                (0.15, 0.30, 0.15): (-22, -3),
                (0.10, 0.00, 0.23): (0, 11),
            }
            off = offsets.get(xyz, (0, -15))
            ax.annotate(f"{mae:.2f} m", (xyz[0] + dx, xyz[1]), xytext=off,
                        textcoords="offset points", ha="center", fontsize=6.6,
                        color=ANOMALY, fontweight="bold", annotation_clip=False)

    ax_c.set_title("(a) circular rig (D3/D4/D5 positions)", fontsize=8.5)
    ax_d2.set_title("(b) D2 rig (3 single-mode positions)", fontsize=8.5)
    for ax in (ax_c, ax_d2):
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")

    cbar = fig.colorbar(sc, ax=[ax_c, ax_d2], shrink=0.85, pad=0.02)
    cbar.set_label("held-out MAE (m), LOPO fusion head", fontsize=8)
    save(fig, "fig25_per_position_error_map")


def main() -> None:
    style.apply_style()
    fig03_geometry()
    fig25_error_map()


if __name__ == "__main__":
    main()
