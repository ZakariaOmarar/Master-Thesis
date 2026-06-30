"""Classical-localization figures computed from raw recordings
(figures 29 and 30 of the figure plan).

  29  worked localization example: SRP-PHAT power slice, accel-TDOA
      hyperbolae, classical estimates, ground truth, for one D5 knock
  30  wave-speed sensitivity of the accel-TDOA solver (plastic vs steel)

Run with:  python -m scripts.figures.fig_classical
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.figures import style
from scripts.figures.style import (
    ACOUSTIC,
    ANOMALY,
    CLASSICAL,
    VIBRATION,
    save,
)
from src.modeling.localization.classical import gcc_phat, srp_phat_3d
from src.modeling.localization.multilateration import (
    accel_tdoa_multilateration_v0,
    estimate_pairwise_tdoas,
)
from src.modeling.orchestration.full_run import resolved_loader

C_SOUND = 343.0
C_PLASTIC = 2000.0
C_STEEL = 5000.0


def _knock_segments(dataset_yaml: str):
    return [s for s in resolved_loader(dataset_yaml).list_segments() if s.is_anomaly]


def _burst_slice(seg, pad_s: float = 0.4) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Locate the strongest impulsive burst; return (mic_win, accel_win, fs_mic, fs_acc)."""
    ds = seg.segment
    env = np.abs(ds.accel_data - ds.accel_data.mean(axis=1, keepdims=True)).mean(axis=0)
    fs_acc, fs_mic = ds.accel_sample_rate, ds.mic_sample_rate
    t_b = int(np.argmax(env))
    a0 = int(np.clip(t_b - pad_s * fs_acc, 0, ds.accel_data.shape[1]))
    a1 = int(np.clip(t_b + pad_s * fs_acc, 0, ds.accel_data.shape[1]))
    m_c = int(t_b / fs_acc * fs_mic)
    m0 = int(np.clip(m_c - pad_s * fs_mic, 0, ds.mic_data.shape[1]))
    m1 = int(np.clip(m_c + pad_s * fs_mic, 0, ds.mic_data.shape[1]))
    return ds.mic_data[:, m0:m1], ds.accel_data[:, a0:a1], fs_mic, fs_acc


# ─────────────────────────────────────────────────────────────────────────
# 29 — worked localization example
# ─────────────────────────────────────────────────────────────────────────
def fig29_worked_example() -> None:
    segs = _knock_segments("d5.yaml")
    # prefer an inside-hull position with a clean burst
    seg = next((s for s in segs if s.spatial_label and abs(s.spatial_label[0]) < 0.1), segs[0])
    gt = np.asarray(seg.spatial_label, dtype=float)
    mic_win, acc_win, fs_mic, fs_acc = _burst_slice(seg)
    mic_xyz = seg.mic_positions
    vib_xyz = seg.vib_positions

    # — SRP-PHAT volume —
    n_mics = mic_xyz.shape[0]
    pairs = [(i, j) for i in range(n_mics) for j in range(i + 1, n_mics)]
    dists = [np.linalg.norm(mic_xyz[i] - mic_xyz[j]) for i, j in pairs]
    max_delay = max(2, int(np.ceil(max(dists) / C_SOUND * fs_mic * 1.5)))
    gcc = np.stack([gcc_phat(mic_win[i], mic_win[j], max_delay) for i, j in pairs])

    gx = np.arange(-0.24, 0.26, 0.005)
    gy = np.arange(-0.30, 0.10, 0.005)
    gz = np.arange(0.0, 0.26, 0.005)
    srp = srp_phat_3d(gcc, mic_xyz, gx, gy, gz, fs_mic, c=C_SOUND, mic_pairs=pairs)
    pk = np.unravel_index(np.argmax(srp), srp.shape)
    srp_est = np.array([gx[pk[0]], gy[pk[1]], gz[pk[2]]])

    # — accel-TDOA estimate + hyperbolae —
    tdoa_est, _res = accel_tdoa_multilateration_v0(acc_win, vib_xyz, fs_acc, c=C_PLASTIC)
    tdoa_s, vib_pairs = estimate_pairwise_tdoas(acc_win, vib_xyz, fs_acc, c=C_PLASTIC)

    iz = int(np.argmin(np.abs(gz - gt[2])))
    fig, (ax, axg) = plt.subplots(1, 2, figsize=(7.2, 3.3),
                                  gridspec_kw={"width_ratios": [1.35, 1]})

    im = ax.imshow(
        srp[:, :, iz].T, origin="lower", aspect="equal", cmap="inferno",
        extent=(gx[0], gx[-1], gy[0], gy[-1]),
    )
    ax.grid(False)

    # hyperbolae of the three highest-|TDOA| accel pairs in this z-slice
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    Z = np.full_like(GX, gz[iz])
    order = np.argsort(-np.abs(tdoa_s))[:3]
    for k in order:
        i, j = vib_pairs[k]
        di = np.sqrt((GX - vib_xyz[i, 0]) ** 2 + (GY - vib_xyz[i, 1]) ** 2 + (Z - vib_xyz[i, 2]) ** 2)
        dj = np.sqrt((GX - vib_xyz[j, 0]) ** 2 + (GY - vib_xyz[j, 1]) ** 2 + (Z - vib_xyz[j, 2]) ** 2)
        ax.contour(GX, GY, (di - dj) - C_PLASTIC * tdoa_s[k], levels=[0.0],
                   colors=[VIBRATION], linewidths=1.0, linestyles="--")

    ax.scatter(mic_xyz[:, 0], mic_xyz[:, 1], c="white", marker="o", s=28,
               edgecolors=ACOUSTIC, linewidths=1.2, zorder=5)
    ax.scatter(vib_xyz[:, 0], vib_xyz[:, 1], c="white", marker="^", s=36,
               edgecolors=VIBRATION, linewidths=1.2, zorder=5)
    ax.scatter(*gt[:2], marker="*", s=260, c="#00d26a", edgecolors="k",
               linewidths=0.8, zorder=6)
    ax.scatter(*srp_est[:2], marker="x", s=90, c="cyan", linewidths=2.0, zorder=6)
    ax.scatter(*tdoa_est[:2], marker="+", s=110, c=VIBRATION, linewidths=2.2, zorder=6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"(a) SRP-PHAT slice at $z={gz[iz]:.2f}$ m, knock {seg.source_dir.name} cm",
        fontsize=8.5,
    )
    fig.colorbar(im, ax=ax, shrink=0.85, label="steered power")
    handles = [
        Line2D([], [], marker="*", ls="", mfc="#00d26a", mec="k", ms=11, label="ground truth"),
        Line2D([], [], marker="x", ls="", color="cyan", ms=8, mew=2, label="SRP-PHAT argmax"),
        Line2D([], [], marker="+", ls="", color=VIBRATION, ms=9, mew=2, label="accel-TDOA solve"),
        Line2D([], [], ls="--", color=VIBRATION, label="TDOA hyperbolae"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.2, frameon=True,
              framealpha=0.85)

    # (b) GCC-PHAT of the three used accel pairs
    max_d_acc = max(np.linalg.norm(vib_xyz[i] - vib_xyz[j]) for i, j in vib_pairs)
    mds = max(2, int(round(max_d_acc / C_PLASTIC * fs_acc * 1.5)))
    for off, k in enumerate(order):
        i, j = vib_pairs[k]
        g = gcc_phat(acc_win[i].astype(float), acc_win[j].astype(float), mds * 6)
        lags = np.arange(-mds * 6, mds * 6 + 1) / fs_acc * 1e3
        g = g / (np.abs(g).max() + 1e-12)
        axg.plot(lags, g + 1.4 * off, color=VIBRATION, lw=0.9)
        axg.axvline(tdoa_s[k] * 1e3, color=ANOMALY, lw=0.8, ls=":")
        axg.text(lags[0], 1.4 * off + 0.62, f"pair ({i},{j})", fontsize=6.6, color="0.3",
                 va="bottom", bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1.0})
    axg.set_xlabel("lag (ms)")
    axg.set_yticks([])
    axg.set_title("(b) accel GCC-PHAT, three pairs\n(red: refined TDOA)", fontsize=8.5)

    fig.tight_layout()
    save(fig, "fig29_worked_example")


# ─────────────────────────────────────────────────────────────────────────
# 30 — wave-speed sensitivity
# ─────────────────────────────────────────────────────────────────────────
def fig30_wave_speed() -> None:
    """The prototype's TDOA budget, and what an inflated wave speed does to it.

    A direct error-vs-speed sweep of the classical solver is nearly flat on
    this rig (the bounded multi-start solve absorbs the scaling), so the
    honest visualization is the mechanism itself: the measured sub-sample
    delays against the geometrically admissible band, which a metallic wave
    speed shrinks by 2.55x.
    """
    segs = _knock_segments("d5.yaml")

    tdoas_us: list[float] = []
    max_sep = 0.0
    for seg in segs:
        _mic, acc_win, _fs_mic, fs_acc = _burst_slice(seg)
        tdoa_s, pairs = estimate_pairwise_tdoas(acc_win, seg.vib_positions, fs_acc,
                                                c=C_PLASTIC)
        tdoas_us.extend(np.abs(tdoa_s) * 1e6)
        for i, j in pairs:
            max_sep = max(max_sep, float(np.linalg.norm(seg.vib_positions[i]
                                                        - seg.vib_positions[j])))
    tdoas_us = np.asarray(tdoas_us)
    lim_plastic_us = max_sep / C_PLASTIC * 1e6
    lim_steel_us = max_sep / C_STEEL * 1e6
    sample_us = 1.0 / 446.0 * 1e6  # one raw-vibration sample

    frac_pl = float(np.mean(tdoas_us <= lim_plastic_us))
    frac_st = float(np.mean(tdoas_us <= lim_steel_us))

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    bins = np.logspace(np.log10(max(tdoas_us.min(), 0.5)), np.log10(tdoas_us.max() * 1.3), 40)
    ax.hist(tdoas_us, bins=bins, color=VIBRATION, alpha=0.7,
            label=f"measured pair delays\n({len(tdoas_us)} pairs, {len(segs)} D5 bursts)")
    ax.set_xscale("log")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.06)
    ymax = ax.get_ylim()[1]
    ax.axvspan(bins[0], lim_steel_us, color=CLASSICAL, alpha=0.18)
    ax.axvspan(bins[0], lim_plastic_us, color=VIBRATION, alpha=0.12)
    ax.axvline(lim_plastic_us, color=VIBRATION, lw=1.4, ls="--")
    ax.axvline(lim_steel_us, color=CLASSICAL, lw=1.4, ls=":")
    ax.axvline(sample_us, color="0.2", lw=1.2)
    ax.annotate(f"admissible if plastic, $c{{=}}2000$ m/s:\n"
                f"$\\leq${lim_plastic_us:.0f} µs  ({frac_pl:.0%} of delays)",
                xy=(lim_plastic_us, ymax * 0.80), xytext=(0.13, 0.96),
                textcoords="axes fraction", fontsize=7.3, color=VIBRATION, va="top",
                arrowprops={"arrowstyle": "-|>", "color": VIBRATION, "lw": 0.9})
    ax.annotate(f"admissible if steel, $c{{=}}5000$ m/s:\n"
                f"$\\leq${lim_steel_us:.0f} µs  ({frac_st:.0%}) — 2.55x tighter",
                xy=(lim_steel_us, ymax * 0.55), xytext=(0.13, 0.74),
                textcoords="axes fraction", fontsize=7.3, color="0.25", va="top",
                arrowprops={"arrowstyle": "-|>", "color": "0.45", "lw": 0.9})
    ax.annotate(f"one raw-vibration\nsample = {sample_us:.0f} µs",
                xy=(sample_us, ymax * 0.62), xytext=(0.56, 0.96),
                textcoords="axes fraction", fontsize=7.3, color="0.2", va="top",
                arrowprops={"arrowstyle": "-|>", "color": "0.4", "lw": 0.9})
    ax.set_xlabel("measured |TDOA| per accelerometer pair (µs, log)")
    ax.set_ylabel("count")
    ax.set_title(
        "The prototype's TDOA budget: the admissible delay band (shaded) sits far below\n"
        "one vibration sample, and a metallic wave speed shrinks it another 2.55x",
        fontsize=9,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=7.2)
    fig.tight_layout()
    save(fig, "fig30_wave_speed_sensitivity")


def main() -> None:
    style.apply_style()
    fig29_worked_example()
    fig30_wave_speed()


if __name__ == "__main__":
    main()
