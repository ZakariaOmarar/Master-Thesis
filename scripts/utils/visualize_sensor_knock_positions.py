"""Visualise sensor and knock positions for the circular-rig campaigns (D3-D5).

The three circular-rig campaigns (D3, D4, D5) share one physical 3D-printed
rig and one ``position.json`` sensor layout. This script reads the real
files on disk and plots, in centimetres on the prototype frame:

  * the 9 microphones and 4 accelerometers (from ``position.json``);
  * every knock / anomaly position, parsed from the folder names
    (``(x, y, z)`` in cm), coloured by campaign;
  * the D3 ``hit_between_A_B`` event, placed at the centroid of the two
    named sensors -- exactly how the ingestion pipeline derives it.

It renders a 3D scatter plus three 2D projections (top XY, front XZ,
side YZ) and prints a numeric summary (per-campaign and combined bounding
boxes, and which knocks fall outside the sensor footprint). Nothing here
loads waveforms, so it runs in well under a second.

Usage:
    python -m scripts.utils.visualize_sensor_knock_positions
    python -m scripts.utils.visualize_sensor_knock_positions --show
    python -m scripts.utils.visualize_sensor_knock_positions --out my_figure.png
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# All three circular-rig campaigns reuse the D3 position.json. D5 ships its
# own copy (identical content); we read whichever the campaign points at.
CAMPAIGNS = {
    "D3": {
        "root": REPO_ROOT / "data" / "third_test_dataset",
        "positions": REPO_ROOT / "data" / "third_test_dataset" / "position.json",
        "colour": "#ff7f0e",
        "marker": "*",
    },
    "D4": {
        "root": REPO_ROOT / "data" / "fourth_test_dataset",
        "positions": REPO_ROOT / "data" / "third_test_dataset" / "position.json",
        "colour": "#d62728",
        "marker": "X",
    },
    "D5": {
        "root": REPO_ROOT / "data" / "fifth_test_dataset",
        "positions": REPO_ROOT / "data" / "fifth_test_dataset" / "position.json",
        "colour": "#9467bd",
        "marker": "P",
    },
}

# Folder named exactly "(x, y, z)" (optional spaces) -> a knock position in cm.
_XYZ_RE = re.compile(
    r"^\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)$"
)
# D3 "hit_between_Fl_Gr_speed1" -> centroid of sensors Fl and Gr.
_HIT_RE = re.compile(r"^hit_between_([A-Za-z]+)_([A-Za-z]+)_speed\d+$")
# A D4 knock folder is "usable" only if exactly 4 accelerometers are present;
# the strict ingestion adapter rejects the (5.5, 4, 8) folder (5 accels).
_EXPECTED_ACCEL = 4


@dataclass(frozen=True)
class Sensors:
    mic_ids: list[str]
    mic_xyz: np.ndarray  # (n_mic, 3) cm
    vib_ids: list[str]
    vib_xyz: np.ndarray  # (n_vib, 3) cm
    lookup: dict[str, np.ndarray]  # normalised id -> xyz


@dataclass(frozen=True)
class Knock:
    xyz: np.ndarray  # (3,) cm
    label: str
    usable: bool


def _norm_id(raw: str) -> str:
    return raw.replace("(V)", "").replace("_", "").upper()


def load_sensors(position_json: Path) -> Sensors:
    records = json.loads(position_json.read_text(encoding="utf-8"))
    mic_ids, mic_xyz, vib_ids, vib_xyz = [], [], [], []
    lookup: dict[str, np.ndarray] = {}
    for r in records:
        sid = str(r["id"])
        xyz = np.array([float(r["x"]), float(r["y"]), float(r["z"])], dtype=float)
        lookup[_norm_id(sid)] = xyz
        if sid.startswith("(V)"):
            vib_ids.append(sid.replace("(V)", ""))
            vib_xyz.append(xyz)
        else:
            mic_ids.append(sid)
            mic_xyz.append(xyz)
    return Sensors(
        mic_ids=mic_ids,
        mic_xyz=np.array(mic_xyz),
        vib_ids=vib_ids,
        vib_xyz=np.array(vib_xyz),
        lookup=lookup,
    )


def _count_accelerometers(folder: Path) -> int:
    """Distinct accelerometer IDs present (raw stream preferred, else peak)."""
    raw = {p.stem.split("vibration_raw_", 1)[-1]
           for p in folder.glob("vibration_raw_*.csv")}
    if raw:
        return len(raw)
    peak = {p.stem.split("vibration_", 1)[-1]
            for p in folder.glob("vibration_*.csv")
            if not p.stem.startswith("vibration_raw_")}
    return len(peak)


def load_knocks(root: Path, sensors: Sensors) -> list[Knock]:
    knocks: list[Knock] = []
    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        name = folder.name

        m_xyz = _XYZ_RE.match(name)
        if m_xyz is not None:
            xyz = np.array([float(m_xyz.group(i)) for i in (1, 2, 3)])
            n_accel = _count_accelerometers(folder)
            usable = n_accel == _EXPECTED_ACCEL
            suffix = "" if usable else f"  [{n_accel} accel: skipped]"
            knocks.append(Knock(xyz=xyz, label=f"{name}{suffix}", usable=usable))
            continue

        m_hit = _HIT_RE.match(name)
        if m_hit is not None:
            a, b = _norm_id(m_hit.group(1)), _norm_id(m_hit.group(2))
            if a in sensors.lookup and b in sensors.lookup:
                xyz = 0.5 * (sensors.lookup[a] + sensors.lookup[b])
                knocks.append(
                    Knock(xyz=xyz, label=f"hit {m_hit.group(1)}-{m_hit.group(2)}", usable=True)
                )
    return knocks


def _bbox(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return xyz.min(axis=0), xyz.max(axis=0)


def print_summary(sensors: Sensors, knocks_by_campaign: dict[str, list[Knock]]) -> None:
    all_sensors = np.vstack([sensors.mic_xyz, sensors.vib_xyz])
    lo, hi = _bbox(all_sensors)
    print("\n=== Sensor footprint (cm) ===")
    print(f"  {len(sensors.mic_ids)} mics + {len(sensors.vib_ids)} accelerometers")
    print(f"  x [{lo[0]:.1f}, {hi[0]:.1f}]  y [{lo[1]:.1f}, {hi[1]:.1f}]  "
          f"z [{lo[2]:.1f}, {hi[2]:.1f}]   extent = {hi - lo}")

    print("\n=== Knock positions (cm) ===")
    for name, knocks in knocks_by_campaign.items():
        print(f"  {name}: {len(knocks)} positions")
        for k in knocks:
            outside = (
                not (lo[0] <= k.xyz[0] <= hi[0] and lo[1] <= k.xyz[1] <= hi[1])
            )
            tag = "  <-- outside sensor footprint" if outside else ""
            print(f"      ({k.xyz[0]:6.1f}, {k.xyz[1]:6.1f}, {k.xyz[2]:5.1f})  "
                  f"{k.label}{tag}")
        if knocks:
            kxyz = np.vstack([k.xyz for k in knocks])
            klo, khi = _bbox(kxyz)
            print(f"      bbox: x [{klo[0]:.1f}, {khi[0]:.1f}]  "
                  f"y [{klo[1]:.1f}, {khi[1]:.1f}]  z [{klo[2]:.1f}, {khi[2]:.1f}]")

    every = np.vstack([k.xyz for ks in knocks_by_campaign.values() for k in ks])
    klo, khi = _bbox(every)
    print(f"\n  combined knock bbox: x [{klo[0]:.1f}, {khi[0]:.1f}]  "
          f"y [{klo[1]:.1f}, {khi[1]:.1f}]  z [{klo[2]:.1f}, {khi[2]:.1f}]   "
          f"extent = {khi - klo}")


def plot(sensors: Sensors, knocks_by_campaign: dict[str, list[Knock]], out: Path, show: bool) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    fig = plt.figure(figsize=(14, 11))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    def draw_sensors(ax, ix, iy, annotate=False):
        ax.scatter(sensors.mic_xyz[:, ix], sensors.mic_xyz[:, iy],
                   c="#1f77b4", marker="o", s=55, label="microphone", zorder=3)
        ax.scatter(sensors.vib_xyz[:, ix], sensors.vib_xyz[:, iy],
                   c="#2ca02c", marker="^", s=80, label="accelerometer", zorder=3)
        if annotate:
            for sid, p in zip(sensors.mic_ids, sensors.mic_xyz):
                ax.annotate(sid, (p[ix], p[iy]), fontsize=7, color="#1f77b4",
                            xytext=(3, 3), textcoords="offset points")
            for sid, p in zip(sensors.vib_ids, sensors.vib_xyz):
                ax.annotate(sid, (p[ix], p[iy]), fontsize=7, color="#2ca02c",
                            xytext=(3, 3), textcoords="offset points")

    def draw_knocks(ax, ix, iy):
        for name, knocks in knocks_by_campaign.items():
            cfg = CAMPAIGNS[name]
            us = np.array([k.xyz for k in knocks if k.usable]).reshape(-1, 3)
            sk = np.array([k.xyz for k in knocks if not k.usable]).reshape(-1, 3)
            if len(us):
                ax.scatter(us[:, ix], us[:, iy], c=cfg["colour"], marker=cfg["marker"],
                           s=90, edgecolors="k", linewidths=0.4, label=f"{name} knock", zorder=4)
            if len(sk):
                ax.scatter(sk[:, ix], sk[:, iy], facecolors="none", edgecolors=cfg["colour"],
                           marker=cfg["marker"], s=110, linewidths=1.4,
                           label=f"{name} knock (skipped)", zorder=4)

    # 3D panel
    ax3d.scatter(sensors.mic_xyz[:, 0], sensors.mic_xyz[:, 1], sensors.mic_xyz[:, 2],
                 c="#1f77b4", marker="o", s=55, label="microphone")
    ax3d.scatter(sensors.vib_xyz[:, 0], sensors.vib_xyz[:, 1], sensors.vib_xyz[:, 2],
                 c="#2ca02c", marker="^", s=80, label="accelerometer")
    for name, knocks in knocks_by_campaign.items():
        cfg = CAMPAIGNS[name]
        for k in knocks:
            face = cfg["colour"] if k.usable else "none"
            ax3d.scatter(*k.xyz, facecolors=face, edgecolors=cfg["colour"],
                         marker=cfg["marker"], s=90, linewidths=1.2)
    # one legend proxy per campaign for the 3D axis
    for name, cfg in CAMPAIGNS.items():
        if knocks_by_campaign.get(name):
            ax3d.scatter([], [], [], c=cfg["colour"], marker=cfg["marker"], s=90,
                         label=f"{name} knock")
    ax3d.set_xlabel("x (cm)")
    ax3d.set_ylabel("y (cm)")
    ax3d.set_zlabel("z (cm)")
    ax3d.set_title("3D view")
    ax3d.legend(loc="upper left", fontsize=7)

    for ax, (ix, iy, xl, yl, title) in zip(
        (ax_xy, ax_xz, ax_yz),
        [(0, 1, "x (cm)", "y (cm)", "Top view (XY)"),
         (0, 2, "x (cm)", "z (cm)", "Front view (XZ)"),
         (1, 2, "y (cm)", "z (cm)", "Side view (YZ)")],
    ):
        draw_sensors(ax, ix, iy, annotate=(ax is ax_xy))
        draw_knocks(ax, ix, iy)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax_xy.legend(loc="best", fontsize=7)

    fig.suptitle("Circular-rig sensor and knock positions (D3-D5), prototype frame",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nFigure written to {out}")
    if show:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "docs" / "final_thesis" / "figures"
                    / "sensor_knock_positions_d3d5.png")
    ap.add_argument("--show", action="store_true", help="open an interactive window too")
    args = ap.parse_args()

    # D3 and D4 share one position.json; load it once, reuse for both.
    sensors_by_json: dict[Path, Sensors] = {}
    knocks_by_campaign: dict[str, list[Knock]] = {}
    sensors = None
    for name, cfg in CAMPAIGNS.items():
        pj = cfg["positions"]
        if pj not in sensors_by_json:
            sensors_by_json[pj] = load_sensors(pj)
        s = sensors_by_json[pj]
        sensors = sensors or s  # all three layouts are identical
        knocks_by_campaign[name] = load_knocks(cfg["root"], s)

    assert sensors is not None
    print_summary(sensors, knocks_by_campaign)
    plot(sensors, knocks_by_campaign, args.out, args.show)


if __name__ == "__main__":
    main()
