"""Schematic / conceptual figures (no model, no signal data).

Renders (thesis figure plan numbering):
   1  domain-shift motivation timeline (conceptual)
   2  annotated-photo placeholder for the prototype
   4  campaign overview card strip (D1..D5, from tab:datasets/durations)
   5  preprocessing pipeline flowchart with tensor shapes
  17  split-protocol diagram (recording-level splits + 3 localization protocols)
  34  paradigm-map summary (demand -> winning paradigm)
  35  one-page synthesis (pipeline -> three general principles)

Run with:  python -m scripts.figures.fig_schematics
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.figures import style
from scripts.figures.style import (
    ACOUSTIC,
    ANOMALY,
    CAMPAIGN_COLORS,
    CLASSICAL,
    INTERMEDIATE,
    LATE_FUSION,
    MODE_COLORS,
    VIBRATION,
    save,
)


# ── drawing helpers ─────────────────────────────────────────────────────
def box(ax, x, y, w, h, text, *, fc="#f0f0f0", ec="0.35", fs=8, lw=1.0,
        text_color="0.1", weight="normal", rounding=0.06):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw, zorder=2,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=text_color, fontweight=weight, zorder=3)
    return p


def arrow(ax, p0, p1, *, color="0.3", lw=1.2, style="-|>", shrink=2.0,
          connectionstyle="arc3,rad=0.0", ls="-"):
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
        shrinkA=shrink, shrinkB=shrink, connectionstyle=connectionstyle,
        linestyle=ls, zorder=1.5,
    )
    ax.add_patch(a)
    return a


def blank_axes(figsize, xlim=(0, 10), ylim=(0, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.grid(False)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────
# 1 — domain-shift motivation
# ─────────────────────────────────────────────────────────────────────────
def fig01_domain_shift() -> None:
    rng = np.random.default_rng(7)
    t = np.linspace(0, 100, 1200)
    tu_end, pu_start = 44, 52

    base = np.where(t < tu_end, 1.0, np.where(t > pu_start, 2.9, 0.0))
    ramp = (t >= tu_end) & (t <= pu_start)
    base[ramp] = 1.0 + (2.9 - 1.0) * (t[ramp] - tu_end) / (pu_start - tu_end)
    spike = 2.6 * np.exp(-0.5 * ((t - (tu_end + pu_start) / 2) / 1.9) ** 2)
    score = base + spike + 0.14 * rng.standard_normal(t.size)
    score = np.convolve(score, np.ones(5) / 5, mode="same")
    t, score = t[5:-5], score[5:-5]  # drop smoothing edge artefacts

    thr = 2.05

    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.axvspan(tu_end, pu_start, color="0.92", zorder=0)
    above = score > thr
    ax.plot(t, score, color=ACOUSTIC, lw=1.2, label="anomaly score of a static detector")
    ax.fill_between(t, thr, score, where=above, color=ANOMALY, alpha=0.45,
                    label="false alarms (machine healthy throughout)")
    ax.axhline(thr, color="0.2", lw=1.1, ls="--")
    ax.text(1.5, thr + 0.1, "static threshold (fit on TU)", fontsize=8, color="0.2")

    ax.text(tu_end / 2, 4.55, "Turbine (TU)", ha="center", fontsize=9,
            color=MODE_COLORS["Turbine"], fontweight="bold")
    ax.text((tu_end + pu_start) / 2, 4.55, "transition", ha="center", fontsize=8, color="0.35")
    ax.text((pu_start + 100) / 2, 4.55, "Pump (PU)", ha="center", fontsize=9,
            color=MODE_COLORS["Pump"], fontweight="bold")

    ax.annotate("synchronized\nfalse-alarm spike", xy=(46.5, 4.1), xytext=(28, 3.6),
                fontsize=8, color=ANOMALY, ha="center",
                arrowprops={"arrowstyle": "-|>", "color": ANOMALY, "lw": 1.0})
    ax.annotate("healthy PU baseline sits above\nthe TU-fit threshold: permanent alarm",
                xy=(83, 3.0), xytext=(60, 1.0), fontsize=8, color="0.15",
                arrowprops={"arrowstyle": "-|>", "color": "0.4", "lw": 1.0})

    ax.set_xlabel("time")
    ax.set_ylabel("anomaly score")
    ax.set_ylim(0, 5.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    fig.tight_layout()
    save(fig, "fig01_domain_shift_motivation")


# ─────────────────────────────────────────────────────────────────────────
# 2 — annotated photographs of the two rigs
# ─────────────────────────────────────────────────────────────────────────
def fig02_photo() -> None:
    from scripts.figures.style import REPO_ROOT

    img_rect = plt.imread(REPO_ROOT / "data" / "second_test_dataset" / "1000052773.jpg")
    img_circ = plt.imread(REPO_ROOT / "data" / "third_test_dataset" / "1000052877.jpg")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.2, 4.3), gridspec_kw={"width_ratios": [1.25, 0.75]}
    )

    def marker(ax, x_frac, y_frac, num, img):
        h, w = img.shape[:2]
        x, y = x_frac * w, y_frac * h
        
        # [FIXED] Reduced markersize (ms) from 22 to 18 for a slightly smaller circle
        ax.plot(x, y, marker="o", ms=18, mfc="white", mec="0.1",
                mew=1.2, alpha=0.85, zorder=5)
        
        # [FIXED] Reduced font size slightly to 8.0 to comfortably fit the smaller circle
        ax.text(x, y, num, ha="center", va="center", fontsize=8.0,
                fontweight="bold", color="0.1", zorder=6)

    ax_a.imshow(img_rect)
    for xf, yf, n in [(0.13, 0.50, "1"), (0.67, 0.70, "2"), (0.14, 0.72, "3"),
                      (0.42, 0.43, "3"), (0.55, 0.08, "4")]:
        marker(ax_a, xf, yf, n, img_rect)
    ax_a.set_title("(a) rectangular rig (D1, D2)", fontsize=10)

    ax_b.imshow(img_circ)
    for xf, yf, n in [(0.45, 0.42, "5"), (0.10, 0.75, "3"), (0.68, 0.46, "3"),
                      (0.85, 0.18, "4")]:
        marker(ax_b, xf, yf, n, img_circ)
    ax_b.set_title("(b) circular 3D-printed rig (D3-D5)", fontsize=10)

    for ax in (ax_a, ax_b):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    # Split into two lines for readability
    key = ("1  inlet valve    2  pump / drive    3  sensor breakouts on the casing (mics + accelerometers)\n"
           "4  acquisition boards (shared trigger)    5  circular casing (~10 cm)")
    
    # Bottom legend formatting
    fig.text(0.5, 0.02, key, ha="center", fontsize=9.5, color="0.2")
    
    # Expanded bottom margin (rect parameter) to fit the double-line legend
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, "fig02_prototype_photo", png_only=True)


# ─────────────────────────────────────────────────────────────────────────
# 4 — campaign overview cards
# ─────────────────────────────────────────────────────────────────────────
def fig04_campaign_overview() -> None:
    cards = [
        ("D1", "4 mics + 4 accel", "peak vib (~4 Hz)",
         "modes labeled:\nPU / ST / TU", "39 min healthy\n11 min anomaly", "0 knock pos."),
        ("D2", "5 mics + 5 accel", "peak vib (4 Hz)",
         "modes labeled:\nPU / ST / TU", "20 min healthy\n8 min anomaly", "3 knock pos.\n(single-mode)"),
        ("D3", "9 mics + 4 accel", "peak vib (16 Hz)",
         "speed{1,2,3}\n(fan-noise level)", "7 min healthy\n0.6 min anomaly", "1 knock pos.\n(midpoint)"),
        ("D4", "9 mics + 4 accel", "raw vib (~376 Hz)",
         "speed{1,2,3}\n(fan-noise level)", "33 min healthy\n77 min anomaly\n(sparse)", "6 knock pos."),
        ("D5", "9 mics + 4 accel", "raw vib (~446 Hz)",
         "flat healthy pool\n(no speed token)", "4 min healthy\n27 min anomaly", "6 knock pos."),
    ]
    fig, ax = blank_axes((7.4, 3.4), xlim=(0, 25.4), ylim=(0, 11))

    # timeline spine
    arrow(ax, (0.4, 0.7), (25.0, 0.7), color="0.4", lw=1.4)
    ax.text(24.9, 0.18, "campaign order", ha="right", fontsize=7.5, color="0.4")

    w, gap = 4.6, 0.4
    for i, (name, sensors, vib, labels, dur, spatial) in enumerate(cards):
        x = 0.4 + i * (w + gap)
        c = CAMPAIGN_COLORS[name]
        box(ax, x, 1.4, w, 9.0, "", fc="white", ec=c, lw=1.6, rounding=0.12)
        ax.add_patch(FancyBboxPatch((x, 9.35), w, 1.05,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=c, ec=c, zorder=2))
        ax.text(x + w / 2, 9.88, name, ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", zorder=3)
        rows = [
            (sensors, "0.1", "bold"),
            (vib, VIBRATION if "raw" in vib else "0.35", "normal"),
            (labels, MODE_COLORS["Turbine"] if "modes" in labels else "0.35", "normal"),
            (dur, "0.1", "normal"),
            (spatial, ANOMALY if "knock" in spatial and not spatial.startswith("0") else "0.45", "normal"),
        ]
        y = 8.55
        for txt, color, weight in rows:
            ax.text(x + w / 2, y, txt, ha="center", va="center", fontsize=7.3,
                    color=color, fontweight=weight, zorder=3)
            y -= 1.55
        ax.plot([x + w / 2], [0.7], "o", color=c, ms=7, zorder=3)

    # group braces
    ax.text(0.4 + (2 * w + gap) / 2, 0.18 + 10.9 - 10.9, "", fontsize=1)  # noop spacing
    for (i0, i1, label) in [(0, 1, "mode detection"), (2, 3, "fault detection + localization"),
                            (4, 4, "localization")]:
        x0 = 0.4 + i0 * (w + gap)
        x1 = 0.4 + i1 * (w + gap) + w
        ax.annotate("", xy=(x0, 10.78), xytext=(x1, 10.78),
                    arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 1.0})
        ax.text((x0 + x1) / 2, 10.92, label, ha="center", fontsize=7.3, color="0.35")
    save(fig, "fig04_campaign_overview")


# ─────────────────────────────────────────────────────────────────────────
# 5 — preprocessing pipeline flowchart
# ─────────────────────────────────────────────────────────────────────────
def fig05_preprocessing() -> None:
    # [FIXED] Increased figure size and xlim to comfortably fit the larger fonts
    fig, ax = blank_axes((11.5, 5.5), xlim=(0, 34), ylim=(0, 13))
    fs_universal = 9.5
    # [FIXED] All fs (fontsizes) bumped from 7.2 to 8.5
    box(ax, 0.5, 5.0, 4.0, 3.0, "recording\ndirectory\n(WAVs + CSVs)", fc="#f7f7f7", fs=fs_universal)

    box(ax, 6.0, 8.3, 5.5, 3.0, "audio load\n16 kHz, stack,\nstrict channel check", fc="#e8f0fa",
        ec=ACOUSTIC, fs=fs_universal)
    box(ax, 6.0, 1.7, 5.5, 3.0, "vibration parse\npeak (D1-D3) /\nraw DMA (D4-D5)", fc="#e9f6e9",
        ec=VIBRATION, fs=fs_universal)

    box(ax, 13.0, 5.0, 5.0, 3.0, "cross-modal\nsync check\n(4 gates, $\\pm$0.5 s)", fc="#fdf3e3",
        ec="0.4", fs=fs_universal)

    box(ax, 19.5, 8.3, 6.0, 3.0, "acoustic features\nlog-mel 96 + CWT 64\n($n_{fft}$ 4096, hop 2048)",
        fc="#e8f0fa", ec=ACOUSTIC, fs=fs_universal)
    box(ax, 19.5, 1.7, 6.0, 3.0, "vibration features\namplitude + envelope\n+ impulsiveness",
        fc="#e9f6e9", ec=VIBRATION, fs=fs_universal)

    box(ax, 27.0, 5.0, 6.0, 3.0, "windowing\n50% overlap,\noctave scales,\nshape-keyed batches",
        fc="#f0eaf7", ec=INTERMEDIATE, fs=fs_universal)

    # ── ARROWS (Snapped exactly to center-edges and corners) ──
    # 1. Recording -> Load/Parse
    arrow(ax, (4.5, 6.5), (6.0, 8.3), connectionstyle="arc3,rad=0.2", color="0.4")
    arrow(ax, (4.5, 6.5), (6.0, 4.7), connectionstyle="arc3,rad=-0.2", color="0.4")
    
    # 2. Load/Parse -> Sync Check
    arrow(ax, (11.5, 9.8), (13.0, 8.0), connectionstyle="arc3,rad=-0.2", color="0.4")
    arrow(ax, (11.5, 3.2), (13.0, 5.0), connectionstyle="arc3,rad=0.2", color="0.4")
    
    # [FIXED] Texts shifted to x=8.75 to center perfectly over/under the first column of boxes
    ax.text(8.75, 11.7, r"$(n_{mic}, T)$ @ 16 kHz", fontsize=fs_universal, color=ACOUSTIC, ha="center")
    ax.text(8.75, 1.0, r"$(n_{vib}, T_{vib})$ @ 4-446 Hz", fontsize=fs_universal, color=VIBRATION, ha="center")

    # 3. Sync Check -> Features
    arrow(ax, (18.0, 6.5), (19.5, 8.3), connectionstyle="arc3,rad=0.2", color="0.4")
    arrow(ax, (18.0, 6.5), (19.5, 4.7), connectionstyle="arc3,rad=-0.2", color="0.4")
    
    # 4. Features -> Windowing
    arrow(ax, (25.5, 9.8), (27.0, 8.0), connectionstyle="arc3,rad=-0.2", color="0.4")
    arrow(ax, (25.5, 3.2), (27.0, 5.0), connectionstyle="arc3,rad=0.2", color="0.4")
    
    # [FIXED] Texts shifted to x=22.5 to center perfectly over/under the second column of boxes
    ax.text(22.5, 11.7, r"$(n_{mic}, 2, 96, T_{ac})$ @ 7.81 Hz", fontsize=fs_universal, color=ACOUSTIC, ha="center")
    ax.text(22.5, 1.0, r"$(n_{vib}, 3, T_{vib})$", fontsize=fs_universal, color=VIBRATION, ha="center")

    # 5. Output Tuple
    arrow(ax, (30.0, 5.0), (30.0, 3.2), color="0.4")
    ax.text(30.0, 2.0, "per-window tuple:\nfeatures + positions (m)\n+ campaign id + labels",
            ha="center", fontsize=fs_universal, color="0.3")

    save(fig, "fig05_preprocessing_pipeline")


# ─────────────────────────────────────────────────────────────────────────
# 17 — split protocol
# ─────────────────────────────────────────────────────────────────────────
def fig17_split_protocol() -> None:
    # [FIXED] Significantly scaled up the canvas to (11.5, 6.2) and grid to (42, 24) 
    # to allow for much larger text without cramping the layout.
    fig, ax = blank_axes((11.5, 6.2), xlim=(0, 42), ylim=(0, 24))
    
    LBL_X, ROW_X = 0.5, 11.5  

    # — anomaly-detection split: healthy pool —
    # [FIXED] Font sizes globally boosted (e.g., headers from 8.8 to 11.0)
    ax.text(LBL_X, 22.8, "RQ2 anomaly detection", fontsize=11.0, color="0.25", fontweight="bold")
    
    segs = [
        # (x_offset, width, facecolor, edgecolor, text)
        (ROW_X, 9.5, "#dbe9f6", ACOUSTIC, "train (healthy only)\nencoder + flow fit"),
        (ROW_X + 10.2, 8.0, "#fdf3e3", "#c08a2d", "threshold fit\n(held-out healthy)"),
        (ROW_X + 19.0, 10.0, "#e9f6e9", VIBRATION, "evaluation\n(disjoint healthy\n+ anomaly cohorts)"),
    ]
    for x, w, fc, ec, txt in segs:
        # [FIXED] Box fonts bumped from 7.2 to 9.5
        box(ax, x, 19.0, w, 3.0, txt, fc=fc, ec=ec, fs=9.5)
        
    ax.text(ROW_X + 14.5, 18.0, "fit and evaluation sets are disjoint by construction",
            fontsize=9.0, color=ANOMALY, ha="center", va="top")
            
    ax.text(LBL_X, 20.5, "threshold transfer: fit on\none operating condition,\nevaluate on a disjoint one",
            fontsize=9.5, color="0.35", va="center")

    # — localization protocols —
    ax.text(LBL_X, 16.0, "RQ3 localization: three protocols, increasing strictness",
            fontsize=11.0, color="0.25", fontweight="bold")

    def cells(y, n, held, label, cw=1.05, gap=0.15, h=1.4, x0=ROW_X):
        for i in range(n):
            fc = "#f2f2f2" if i not in held else "#fbdcdc"
            ec = "0.55" if i not in held else ANOMALY
            box(ax, x0 + i * (cw + gap), y, cw, h, "", fc=fc, ec=ec, fs=8, rounding=0.04)
        ax.text(LBL_X, y + h/2, label, fontsize=9.5, color="0.2", va="center")
        return x0 + n * (cw + gap)

    # [FIXED] Recalculated cells to perfectly align at the exact same ending X-coordinate (28.5)
    xe = cells(13.0, 10, {3}, "LORO: hold out one\nrecording per fold", cw=1.5, gap=0.2, h=1.8)
    ax.text(xe + 0.8, 13.9, "same position may still\nappear on both sides",
            fontsize=9.0, color="0.35", va="center")

    xe = cells(9.8, 16, {6}, "LOPO: hold out one of\nthe 16 labelled positions", cw=0.9, gap=0.1625, h=1.8)
    ax.text(xe + 0.8, 10.7, "head retrained\nfrom scratch per fold",
            fontsize=9.0, color="0.35", va="center")

    # cross-session
    y = 6.4
    h = 1.8
    for i, (name, c) in enumerate([("D2", CAMPAIGN_COLORS["D2"]), ("D3", CAMPAIGN_COLORS["D3"]),
                                   ("D4", CAMPAIGN_COLORS["D4"])]):
        box(ax, ROW_X + i * 3.4, y, 2.8, h, name, fc="white", ec=c, fs=10.5)
        
    box(ax, ROW_X + 12.8, y, 2.8, h, "D5", fc="#fbdcdc", ec=ANOMALY, fs=10.5)
    arrow(ax, (ROW_X + 10.0, y + h/2), (ROW_X + 12.2, y + h/2), color="0.3", lw=1.5)
    
    ax.text(LBL_X, y + h/2, "cross-session: train on D2/D3/D4,\ntest on unseen D5",
            fontsize=9.5, color="0.2", va="center")
            
    ax.text(ROW_X + 16.4, y + h/2, "test session never seen at training time (strongest claim)",
            fontsize=9.0, color="0.35", va="center")

    # label discipline footer
    box(ax, 0.5, 0.8, 41.0, 4.2,
        "label discipline:  self-supervised stages and thresholds see healthy data only;  "
        "mode labels enter at evaluation only;\nspatial labels supervise and score the localization "
        "head only;  anomaly windows reach V4 only through the V3 alert gate",
        fc="#f7f7f7", ec="0.5", fs=9.0)
        
    save(fig, "fig17_split_protocol")


# ─────────────────────────────────────────────────────────────────────────
# 34 — paradigm map
# ─────────────────────────────────────────────────────────────────────────
def fig34_paradigm_map() -> None:
    fig, ax = blank_axes((6.8, 3.8), xlim=(0, 28), ylim=(0, 16))

    # Five-seed verdicts (Results ch. tables).  The earlier "worst-case stability
    # -> intermediate fusion" row is dropped: that ordering reversed under the
    # multi-seed reruns, so intermediate fusion no longer owns a demand cleanly.
    demands = [
        ("no false alarms on healthy data", "healthy both-fire 0.003; transfer FPR 0.004"),
        ("average localization accuracy", "LORO macro MAE 0.181 m"),
        ("transfer to unseen positions / sessions", "LOPO 0.129 m; D5 0.113 m"),
        ("label-free mode discovery", "strict NMI 0.41, matches floor"),
    ]
    winners = [
        ("Late fusion: AND rule", LATE_FUSION),
        ("Late fusion: confidence gate", LATE_FUSION),
        ("Unimodal: accelerometer TDOA", VIBRATION),
        ("Unimodal: acoustic encoder", ACOUSTIC),
    ]
    y = 12.4
    for (d, v), (wname, wc) in zip(demands, winners):
        box(ax, 0.4, y, 11.8, 2.4, f"{d}\n{v}", fc="#f7f7f7", ec="0.45", fs=7.2)
        box(ax, 16.4, y, 11.0, 2.4, wname, fc=wc, ec=wc, fs=7.5, text_color="white",
            weight="bold")
        arrow(ax, (12.2, y + 1.2), (16.4, y + 1.2), color=wc, lw=1.6)
        y -= 3.0

    ax.text(6.3, 15.6, "operational demand (measured)", fontsize=9, ha="center",
            fontweight="bold", color="0.2")
    ax.text(21.9, 15.6, "winning fusion paradigm", fontsize=9, ha="center",
            fontweight="bold", color="0.2")
    save(fig, "fig34_paradigm_map")


# ─────────────────────────────────────────────────────────────────────────
# 35 — one-page synthesis: the thesis pipeline distilled to three principles
# ─────────────────────────────────────────────────────────────────────────
def fig35_synthesis() -> None:
    fig, ax = blank_axes((5.2, 6.6), xlim=(0, 20), ylim=(0, 26))

    # Vertical pipeline: each learned stage in its concept colour, terminating
    # in the three general principles the four research questions converge on.
    cx, w, h = 10.0, 13.0, 2.3
    stages = [
        ("Context learning\nlabel-free, acoustic-trunk", ACOUSTIC),
        ("Context conditioning\nFiLM on operating context", INTERMEDIATE),
        ("Detection\nAND-rule gate + impulse-aware recall", ANOMALY),
        ("Localization\naccelerometer-TDOA geometric transfer", VIBRATION),
        ("Deployment decisions\ndemand-specific operating points", CLASSICAL),
    ]
    y = 23.0
    centers = []
    for text, c in stages:
        box(ax, cx - w / 2, y, w, h, text, fc=c, ec=c, fs=7.6,
            text_color="white", weight="bold")
        centers.append(y + h / 2)
        y -= 3.4

    for top, bot in zip(centers[:-1], centers[1:]):
        arrow(ax, (cx, top - h / 2), (cx, bot + h / 2), color="0.45", lw=1.6)

    # Terminal node: the three principles, set apart.
    py_top = y + h
    arrow(ax, (cx, centers[-1] - h / 2), (cx, py_top), color="0.45", lw=1.6)
    pbox_h = 5.4
    box(ax, cx - w / 2 - 1.4, py_top - pbox_h, w + 2.8, pbox_h, "",
        fc="#f4f4f4", ec="0.3", lw=1.3)
    ax.text(cx, py_top - 0.9, "Cross-cutting observations", ha="center",
            va="center", fontsize=8.6, fontweight="bold", color="0.15")
    principles = [
        "no universal winner",
        "modality roles are asymmetric",
        "complexity must earn its place",
    ]
    yp = py_top - 2.0
    for p in principles:
        ax.text(cx - w / 2 - 0.4, yp, "•", ha="center", va="center",
                fontsize=9, color="0.2")
        ax.text(cx - w / 2 + 0.4, yp, p, ha="left", va="center", fontsize=8,
                color="0.15")
        yp -= 1.15
    save(fig, "fig35_synthesis")


def main() -> None:
    style.apply_style()
    fig01_domain_shift()
    fig02_photo()
    fig04_campaign_overview()
    fig05_preprocessing()
    fig17_split_protocol()
    fig34_paradigm_map()
    fig35_synthesis()


if __name__ == "__main__":
    main()
