"""Architecture diagrams (figures 9-16 of the figure plan).

   9  high-level end-to-end system overview (Chapter 1; the abstract
       companion that fig10 then zooms into at the model level)
  10  full V1->V4 system block diagram (frozen/trainable colour-coded)
  11  per-modality encoder detail (2D / 1D CNN + attentive statistics pool)
  12  cross-modal fusion + context vector
  13  self-supervised training objectives (V1 contrastive, V2 joint)
  14  conditional normalizing-flow anomaly head
  15  localization head (SRP prior + TDOA tokens + bounded residual)
  16  chained streaming inference

Run with:  python -m scripts.figures.fig_architecture
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from scripts.figures import style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from scripts.figures.fig_schematics import arrow, blank_axes, box
from scripts.figures.style import (
    ACOUSTIC,
    ANOMALY,
    CLASSICAL,
    INTERMEDIATE,
    LATE_FUSION,
    VIBRATION,
    save,
)

AC_BG, VIB_BG, CTX_BG, ALERT_BG, GRAY_BG = "#e8f0fa", "#e9f6e9", "#f0eaf7", "#fdebd9", "#f2f2f2"


def stage_banner(ax, x, w, y, text, color):
    ax.add_patch(plt.Rectangle((x, y), w, 0.9, fc=color, alpha=0.14, ec="none", zorder=0))
    ax.text(x + w / 2, y + 0.45, text, ha="center", va="center", fontsize=8,
            color=color, fontweight="bold", zorder=1)


# ─────────────────────────────────────────────────────────────────────────
# 9 — high-level end-to-end system overview (Chapter 1)
# ─────────────────────────────────────────────────────────────────────────
# extra palette unique to the Chapter-1 overview (the rest come from style.py)
LOCALIZE = "#0f8b8d"    # teal  -> localization head + position output
DETECT_HDR = "#3a4a63"  # slate-blue -> neutral header for the combined stage
LOC_BG = "#e3f3f3"      # light teal fill for the localization head


# ─────────────────────────────────────────────────────────────────────────
def fig09_system_overview() -> None:
    # fig09 renders on a fixed equal-aspect canvas with thicker, more rounded
    # boxes than the other schematics, so it uses these local helpers rather
    # than the shared fig_schematics ones (whose signatures/styling differ).
    def blank_axes(figsize, xlim, ylim):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
        return fig, ax

    def box(ax, x, y, w, h, text, fc, ec, fs=9.5, tc="#1a1a1a"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.28",
            linewidth=1.9, facecolor=fc, edgecolor=ec, zorder=2,
            mutation_aspect=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3)

    def arrow(ax, start, end, color="0.3", lw=1.5, connectionstyle="arc3,rad=0",
              ls="-"):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=15,
            lw=lw, color=color, ls=ls, connectionstyle=connectionstyle,
            shrinkA=0, shrinkB=0, zorder=5,
            capstyle="round", joinstyle="round"))

    fig, ax = blank_axes((12.8, 5.6), xlim=(0, 56), ylim=(0, 24))

    # subsystem banners, each tagged with the chapter that details it
    def banner(x, w, text, color, alpha=0.16):
        ax.add_patch(plt.Rectangle((x, 21.4), w, 1.3, fc=color, alpha=alpha,
                                   ec="none", zorder=0))
        ax.text(x + w / 2, 22.05, text, ha="center", va="center", fontsize=10.0,
                color=color, fontweight="bold", zorder=1)

    banner(0.8, 15.4, "Sensing & data  (Ch. 3)", CLASSICAL)
    banner(16.6, 15.6, "Representation  (Ch. 4 · V1-V2)", INTERMEDIATE)
    banner(32.6, 18.6, "Detection & localization  (Ch. 4 · V3-V4)",
           DETECT_HDR, alpha=0.20)

    # evaluation lens spanning the two learned subsystems
    ax.plot([16.6, 48.2], [19.7, 19.7], color="0.55", lw=0.9, ls=":")
    ax.text(32.4, 20.1, "Evaluation (Ch. 5-6): unimodal · late-fusion · "
            "intermediate-fusion compared at each learned stage", ha="center",
            va="bottom", fontsize=9.0, color="0.4", style="italic")

    # ── sensing & data pipeline ──────────────────────────────────────────
    box(ax, 1.0, 14.0, 6.4, 3.0, "microphone array\n(4-9 channels)", fc=AC_BG,
        ec=ACOUSTIC, fs=9.5)
    box(ax, 1.0, 6.2, 6.4, 3.0, "accelerometer array\n(4-5 channels)", fc=VIB_BG,
        ec=VIBRATION, fs=9.5)
    box(ax, 9.0, 6.2, 6.4, 10.8,
        "preprocessing\npipeline\n\ningest · sync\nfeatures · window",
        fc=GRAY_BG, ec="0.4", fs=9.5)
    ax.text(12.2, 5.6, "uniform per-window tensors\n(two streams, time-aligned)",
            ha="center", va="top", fontsize=8.5, color="0.4")
    arrow(ax, (7.4, 15.5), (9.0, 14.4), color=ACOUSTIC, lw=1.5,
          connectionstyle="arc3,rad=-0.12")
    arrow(ax, (7.4, 7.7), (9.0, 9.0), color=VIBRATION, lw=1.5,
          connectionstyle="arc3,rad=0.12")

    # ── multimodal representation ────────────────────────────────────────
    box(ax, 17.2, 14.0, 6.6, 3.0, "acoustic encoder\n2D-CNN $\\to$ tokens",
        fc=AC_BG, ec=ACOUSTIC, fs=9.5)
    box(ax, 17.2, 6.2, 6.6, 3.0, "vibration encoder\n1D-CNN $\\to$ tokens",
        fc=VIB_BG, ec=VIBRATION, fs=9.5)
    box(ax, 25.6, 8.6, 5.6, 5.0, "cross-modal\nfusion", fc=CTX_BG,
        ec=INTERMEDIATE, fs=9.8)
    ax.add_patch(Circle((34.0, 11.1), 1.5, fc=INTERMEDIATE, ec="none", zorder=3))
    ax.text(34.0, 11.1, "$c_t$", ha="center", va="center", fontsize=17,
            color="white", fontweight="bold", zorder=4)
    ax.text(28.6, 6.9, "context vector $c_t$\n(operating state)", ha="center",
            va="top", fontsize=8.5, color=INTERMEDIATE)

    arrow(ax, (15.4, 14.4), (17.2, 15.5), color=ACOUSTIC, lw=1.5,
          connectionstyle="arc3,rad=0.12")
    arrow(ax, (15.4, 9.0), (17.2, 7.7), color=VIBRATION, lw=1.5,
          connectionstyle="arc3,rad=-0.12")
    arrow(ax, (23.8, 15.5), (25.8, 12.8), color=ACOUSTIC, lw=1.5,
          connectionstyle="arc3,rad=0.2")
    arrow(ax, (23.8, 7.7), (25.8, 9.4), color=VIBRATION, lw=1.5,
          connectionstyle="arc3,rad=-0.2")
    arrow(ax, (31.2, 11.1), (32.45, 11.1), color=INTERMEDIATE, lw=1.7)

    # ── detection & localization ─────────────────────────────────────────
    box(ax, 35.8, 14.0, 7.4, 3.2, "conditional\nanomaly head\nflow · FiLM($c_t$)",
        fc="#fbe9e9", ec=ANOMALY, fs=9.5)
    # localization head in its own teal (distinct from the red anomaly path)
    box(ax, 35.8, 5.2, 7.4, 3.4, "localization head\nSRP + TDOA · FiLM($c_t$)",
        fc=LOC_BG, ec=LOCALIZE, fs=9.5)
    ax.add_patch(plt.Polygon([[45.2, 15.6], [46.6, 17.1], [48.0, 15.6], [46.6, 14.1]],
                             fc=ALERT_BG, ec=LATE_FUSION, lw=1.6, zorder=3))
    ax.text(46.6, 15.6, "alert?", ha="center", va="center", fontsize=9.0,
            color=LATE_FUSION, fontweight="bold", zorder=4)

    # context vector conditions the anomaly head ...
    arrow(ax, (35.4, 11.9), (37.4, 14.0), color=INTERMEDIATE, lw=1.5,
          connectionstyle="arc3,rad=-0.15")
    ax.text(33.7, 13.4, "feature + $c_t$", ha="center", va="center",
            fontsize=8.3, color="0.4")

    # the two inputs to the localization head converge at its TOP-LEFT:
    #   (1) purple context vector  c_t
    arrow(ax, (34.7, 10.0), (37.6, 8.62), color=INTERMEDIATE, lw=1.5,
          connectionstyle="arc3,rad=0.16")
    #   (2) orange gate from the alert decision, entering just beside it
    arrow(ax, (46.6, 14.1), (38.9, 8.62), color=LATE_FUSION, lw=1.8,
          connectionstyle="arc3,rad=0.34")
    ax.text(44.6, 11.3, "only on alert", ha="center", va="center", fontsize=8.5,
            color=LATE_FUSION, fontweight="bold")

    # anomaly -> alert
    arrow(ax, (43.2, 15.6), (45.2, 15.6), color=ANOMALY, lw=1.6)

    # optional field-only supervisory pathway, merged at the context vector
    box(ax, 30.4, 1.9, 7.2, 2.6, "supervisory $s_t$\n(field only · RQ4)",
        fc="white", ec="0.55", fs=8.8)
    arrow(ax, (34.0, 4.5), (34.0, 9.55), color="0.55", lw=1.3, ls=":")

    # ── operator-facing outputs ──────────────────────────────────────────
    box(ax, 49.2, 14.0, 6.2, 3.0, "alert events\n(timeline)", fc="white",
        ec=ANOMALY, fs=9.5)
    box(ax, 49.2, 5.2, 6.2, 3.0, "source position\n$(x,y,z)$ m", fc="white",
        ec=LOCALIZE, fs=9.5)
    arrow(ax, (48.0, 15.6), (49.2, 15.5), color=ANOMALY, lw=1.6)
    arrow(ax, (43.2, 6.9), (49.2, 6.7), color=LOCALIZE, lw=1.6)

    # ── cross-cutting design commitments ─────────────────────────────────
    box(ax, 6.0, 0.25, 42.0, 1.2,
        "Design commitments across every stage:   channel-agnostic   ·   "
        "label-free context   ·   exact-likelihood scoring",
        fc="white", ec="0.55", fs=9.2)

    save(fig, "fig09_system_overview")

# ─────────────────────────────────────────────────────────────────────────
# 10 — full system block diagram
# ─────────────────────────────────────────────────────────────────────────
def fig10_system() -> None:
    fig, ax = blank_axes((10.0, 4.7), xlim=(0, 50), ylim=(0, 21))

    # stage banners
    stage_banner(ax, 6.6, 9.0, 19.4, "V1 — encoders", ACOUSTIC)
    stage_banner(ax, 16.2, 9.6, 19.4, "V2 — fusion + context", INTERMEDIATE)
    stage_banner(ax, 26.4, 10.4, 19.4, "V3 — conditional anomaly", ANOMALY)
    stage_banner(ax, 37.4, 12.4, 19.4, "V4 — localization", VIBRATION)

    # inputs (reduced fs to 6.6 to prevent overflow)
    box(ax, 0.3, 13.2, 5.2, 3.6, "acoustic window\n$(n_{mic},2,F,T)$", fc=AC_BG,
        ec=ACOUSTIC, fs=6.6)
    box(ax, 0.3, 4.6, 5.2, 3.6, "vibration window\n$(n_{vib},3,T)$", fc=VIB_BG,
        ec=VIBRATION, fs=6.6)

    # V1 encoders (reduced fs to 6.6)
    box(ax, 7.0, 13.2, 8.2, 4.4,
        "2D CNN +\nattentive stats pool\n$\\to$ token per mic", fc=AC_BG, ec=ACOUSTIC, fs=6.6)
    box(ax, 7.0, 3.8, 8.2, 4.4,
        "1D CNN +\nattentive stats pool\n$\\to$ token per accel.", fc=VIB_BG, ec=VIBRATION, fs=6.6)
    ax.text(11.1, 11.0, "+ position, modality,\ndataset embeddings", fontsize=6.6,
            ha="center", va="center", color="0.35")

    # V2 fusion (reduced fs to 6.6)
    box(ax, 16.6, 7.2, 8.4, 7.4,
        "bidirectional\ncross-attention\nacoustic $\\leftrightarrow$ vibration\n\nPMA pool\n(2 seeds)",
        fc=CTX_BG, ec=INTERMEDIATE, fs=6.6)
    ax.add_patch(Circle((27.6, 10.9), 1.35, fc=INTERMEDIATE, ec="none", zorder=3))
    ax.text(27.6, 10.9, "$c_t$", ha="center", va="center", fontsize=12, color="white",
            fontweight="bold", zorder=4)
    
    # context vector text on the right side of the circle
    ax.text(29.5, 10.9, "context vector\n(operating state)", ha="left", va="center",
            fontsize=6.8, color=INTERMEDIATE)

    # V3: flow on top, threshold below, alert diamond to the right (reduced fs to 6.6)
    box(ax, 27.0, 14.4, 8.6, 4.0,
        "conditional flow\n6 coupling layers\nFiLM($c_t$) $\\Rightarrow$ exact NLL",
        fc="#fbe9e9", ec=ANOMALY, fs=6.6)
    box(ax, 29.8, 3.6, 8.0, 3.6, "per-cluster threshold\n95th pct, K-means($c_t$)",
        fc=GRAY_BG, ec="0.4", fs=6.6)
    ax.add_patch(plt.Polygon([[38.4, 9.2], [40.4, 10.9], [38.4, 12.6], [36.4, 10.9]],
                             fc=ALERT_BG, ec=LATE_FUSION, lw=1.4, zorder=2))
    ax.text(38.4, 10.9, "alert?", ha="center", va="center", fontsize=7.4,
            color=LATE_FUSION, fontweight="bold", zorder=3)

    # V4 (reduced fs to 6.6)
    box(ax, 38.6, 14.4, 11.0, 4.0,
        "localization head\nSRP volume + TDOA tokens\n+ FiLM($c_t$) residual",
        fc=VIB_BG, ec=VIBRATION, fs=6.6)
    box(ax, 43.0, 8.0, 6.4, 3.4, "$\\hat{p}\\,(x,y,z)$\nposition (m)", fc="white",
        ec="0.2", fs=6.8)

    # arrows — left to right along the two input rows, then up/down to heads
    arrow(ax, (5.5, 15.0), (7.0, 15.4), lw=1.4)
    arrow(ax, (5.5, 6.4), (7.0, 6.0), lw=1.4)
    arrow(ax, (15.2, 15.4), (16.6, 13.2), connectionstyle="arc3,rad=0.18", color=ACOUSTIC, lw=1.4)
    arrow(ax, (15.2, 6.0), (16.6, 8.6), connectionstyle="arc3,rad=-0.18", color=VIBRATION, lw=1.4)
    arrow(ax, (25.0, 10.9), (26.2, 10.9), color=INTERMEDIATE, lw=1.5)

    # x into the flow (from the fusion box top), c_t into flow + threshold
    arrow(ax, (24.0, 14.6), (27.0, 16.2), connectionstyle="arc3,rad=0.22", color="0.4", lw=1.3)
    ax.text(23.2, 17.2, "pooled feature $x$", fontsize=6.8, color="0.35",
            ha="center", va="bottom")
    arrow(ax, (28.4, 12.1), (29.6, 14.35), color=INTERMEDIATE, lw=1.3,
          connectionstyle="arc3,rad=-0.15")
    arrow(ax, (28.4, 9.7), (30.6, 7.25), color=INTERMEDIATE, lw=1.3,
          connectionstyle="arc3,rad=0.15")

    # flow -> alert, threshold -> alert, alert -> V4
    arrow(ax, (35.6, 15.4), (38.3, 12.8), connectionstyle="arc3,rad=-0.25", color=ANOMALY, lw=1.4)
    
    # [FIXED] Pushed Y up to 15.8 and shifted X to 36.5 to confidently clear the top of the arc
    ax.text(36.5, 15.8, "NLL", fontsize=7.0, color=ANOMALY, ha="center")
    
    # threshold arrow starting from edge of box
    arrow(ax, (36.2, 7.2), (38.3, 9.1), connectionstyle="arc3,rad=-0.2", color="0.4", lw=1.3)
    
    arrow(ax, (39.4, 12.4), (41.4, 14.35), color=LATE_FUSION, lw=1.5)
    ax.text(41.4, 12.9, "only on\nalert", fontsize=6.8, color=LATE_FUSION,
            ha="left", va="center")
    arrow(ax, (44.0, 14.4), (45.6, 11.45), color=VIBRATION, lw=1.4)

    # legend
    box(ax, 0.3, 0.4, 23.6, 2.4,
        "training: each stage trains with all stages above it frozen;\nV1/V2 self-supervised, "
        "V3 healthy windows only, V4 alert-gated windows",
        fc="white", ec="0.6", fs=6.8)
    for x0, c, t in [(25.6, ACOUSTIC, "acoustic"), (30.0, VIBRATION, "vibration"),
                     (34.6, INTERMEDIATE, "context"), (38.9, ANOMALY, "anomaly"),
                     (43.0, LATE_FUSION, "decision")]:
        ax.add_patch(plt.Rectangle((x0, 1.25), 1.0, 0.8, fc=c, ec="none"))
        ax.text(x0 + 1.25, 1.65, t, fontsize=7.0, va="center", color="0.25")
    save(fig, "fig10_system_block_diagram")

# ─────────────────────────────────────────────────────────────────────────
# 11 — encoder detail
# ─────────────────────────────────────────────────────────────────────────
def fig11_encoders() -> None:
    fig, ax = blank_axes((6.8, 3.9), xlim=(0, 30), ylim=(0, 16))

    def conv_stack(x0, y0, widths, color, bg, kernel, pool):
        bx = x0
        h0, dh, w = 2.6, 0.55, 2.6
        for i, c in enumerate(widths):
            box(ax, bx, y0, w, h0 + i * dh, f"{c}", fc=bg, ec=color, fs=7)
            if i < len(widths) - 1:
                arrow(ax, (bx + w, y0 + 1.1), (bx + w + 0.7, y0 + 1.1), color=color, lw=1.0)
            bx += w + 0.7
        ax.text(x0 + (len(widths) * (w + 0.7) - 0.7) / 2, y0 - 1.1,
                f"{kernel} conv blocks, {pool}", fontsize=6.6, ha="center", color="0.35")
        return bx - 0.7

    # acoustic column
    ax.text(7.6, 15.3, "acoustic encoder (per microphone)", fontsize=8.5,
            color=ACOUSTIC, fontweight="bold", ha="center")
    box(ax, 0.2, 10.2, 3.7, 3.4, "$(2,F,T)$\nlog-mel\n+ CWT", fc=AC_BG, ec=ACOUSTIC, fs=6.8)
    arrow(ax, (3.9, 11.9), (4.7, 11.9), color=ACOUSTIC)
    conv_stack(4.7, 10.2, [32, 64, 128], ACOUSTIC, AC_BG, r"$3\times3$", r"$2\times2$ max-pool")

    # vibration column
    ax.text(7.6, 7.1, "vibration encoder (per accelerometer)", fontsize=8.5,
            color=VIBRATION, fontweight="bold", ha="center")
    box(ax, 0.2, 2.0, 3.7, 3.4, "$(3,T)$\namp + env\n+ impuls.", fc=VIB_BG, ec=VIBRATION, fs=6.8)
    arrow(ax, (3.9, 3.7), (4.7, 3.7), color=VIBRATION)
    conv_stack(4.7, 2.0, [32, 64, 128], VIBRATION, VIB_BG, "$k=5$ 1D", "stride pool")

    # shared ASP block — arrows land on the box's left edge, label sits above
    box(ax, 16.4, 5.2, 7.8, 5.6,
        "attentive statistics pool\n$\\alpha=\\mathrm{softmax}(\\phi(h))$\n"
        "$\\mu=\\sum_m \\alpha_m h_m$\n$\\sigma_c=\\sqrt{\\sum_m \\alpha_m h_{m,c}^2-\\mu_c^2}$",
        fc=GRAY_BG, ec="0.35", fs=7.0)
    ax.text(20.3, 11.5, "shared design, separate weights", fontsize=6.8, ha="center",
            va="bottom", color="0.35")
    arrow(ax, (14.0, 11.9), (16.35, 9.4), connectionstyle="arc3,rad=0.3", color=ACOUSTIC)
    arrow(ax, (14.0, 3.7), (16.35, 6.6), connectionstyle="arc3,rad=-0.3", color=VIBRATION)

    box(ax, 25.6, 6.4, 3.6, 3.4, "$[\\mu \\Vert \\sigma]$\n$\\to$ linear\n$\\to$ 128-d", fc="white",
        ec="0.25", fs=7.0)
    arrow(ax, (24.2, 8.1), (25.6, 8.1), color="0.3")
    ax.text(27.4, 5.0, "one token per channel;\n2nd moment keeps\nshort knocks visible",
            fontsize=6.4, ha="center", va="top", color="0.35")
    save(fig, "fig11_encoder_detail")


# ─────────────────────────────────────────────────────────────────────────
# 12 — fusion + context
# ─────────────────────────────────────────────────────────────────────────
def fig12_fusion() -> None:
    fig, ax = blank_axes((6.8, 3.9), xlim=(0, 30), ylim=(0, 16))

    # token construction
    box(ax, 0.2, 9.6, 6.6, 4.6,
        "token = [feature 128\n+ position $(x,y,z)$ m\n+ modality emb.\n+ dataset emb.]\n$\\to$ projection",
        fc=AC_BG, ec=ACOUSTIC, fs=6.3)
    box(ax, 0.2, 1.8, 6.6, 4.6,
        "token = [feature 128\n+ position $(x,y,z)$ m\n+ modality emb.\n+ dataset emb.]\n$\\to$ projection",
        fc=VIB_BG, ec=VIBRATION, fs=6.3)
    ax.text(3.5, 14.7, "acoustic tokens ($n_{mic}$)", fontsize=7.5, color=ACOUSTIC,
            ha="center", fontweight="bold")
    ax.text(3.5, 6.9, "vibration tokens ($n_{vib}$)", fontsize=7.5, color=VIBRATION,
            ha="center", fontweight="bold")

    # MAB self-attention
    box(ax, 8.2, 10.4, 4.4, 3.0, "MAB\nself-attention", fc=AC_BG, ec=ACOUSTIC, fs=7)
    box(ax, 8.2, 2.6, 4.4, 3.0, "MAB\nself-attention", fc=VIB_BG, ec=VIBRATION, fs=7)
    arrow(ax, (6.8, 11.9), (8.2, 11.9), color=ACOUSTIC)
    arrow(ax, (6.8, 4.1), (8.2, 4.1), color=VIBRATION)

    # bidirectional cross-attention
    box(ax, 14.0, 5.6, 7.6, 5.6, "", fc=CTX_BG, ec=INTERMEDIATE, fs=7)
    ax.text(17.8, 11.7, "bidirectional cross-attention", fontsize=7.4,
            ha="center", va="bottom", color=INTERMEDIATE, fontweight="bold")
    ax.text(17.8, 9.9, "Q: acoustic | K,V: vibration", fontsize=6.4, ha="center", color=ACOUSTIC)
    ax.text(17.8, 8.4, "Q: vibration | K,V: acoustic", fontsize=6.4, ha="center", color=VIBRATION)
    ax.text(17.8, 6.6, "two MABs,\nno shared weights", fontsize=6.0, ha="center",
            va="center", color="0.4")
    arrow(ax, (12.6, 11.9), (14.4, 10.4), connectionstyle="arc3,rad=0.2", color=ACOUSTIC)
    arrow(ax, (12.6, 4.1), (14.4, 6.8), connectionstyle="arc3,rad=-0.2", color=VIBRATION)

    # PMA pool + context
    box(ax, 22.8, 7.0, 4.2, 3.4, "PMA pool\n(2 seeds,\naveraged)",
        fc=GRAY_BG, ec="0.35", fs=6.8)
    arrow(ax, (21.6, 8.7), (22.8, 8.7), color=INTERMEDIATE)
    ax.add_patch(Circle((28.7, 8.7), 1.05, fc=INTERMEDIATE, ec="none", zorder=3))
    ax.text(28.7, 8.7, "$c_t$", ha="center", va="center", fontsize=10, color="white",
            fontweight="bold", zorder=4)
    arrow(ax, (27.0, 8.7), (27.6, 8.7), color=INTERMEDIATE)
    ax.text(28.7, 7.3, "operating-\nstate code", fontsize=6.2, ha="center",
            va="top", color=INTERMEDIATE)

    ax.text(18.5, 1.0, "permutation-invariant: any channel count, any array geometry;\n"
                       "the same weights serve the 4+4, 5+5 and 9+4 rigs",
            fontsize=7.0, ha="center", va="center", color="0.3", style="italic")
    save(fig, "fig12_fusion_context")


# ─────────────────────────────────────────────────────────────────────────
# 13 — self-supervised objectives
# ─────────────────────────────────────────────────────────────────────────
def fig13_ssl() -> None:
    fig, ax = blank_axes((6.8, 3.8), xlim=(0, 30), ylim=(0, 16))

    # (a) V1
    ax.text(6.6, 15.2, "(a) V1: per-modality contrastive", fontsize=8.5,
            fontweight="bold", color="0.2", ha="center")
    box(ax, 0.3, 10.6, 3.6, 2.4, "window", fc=GRAY_BG, ec="0.4", fs=7)
    box(ax, 5.3, 12.4, 4.0, 2.0, "view 1\n(SpecAugment)", fc=AC_BG, ec=ACOUSTIC, fs=6.6)
    box(ax, 5.3, 8.8, 4.0, 2.0, "view 2\n(dropout, gain)", fc=AC_BG, ec=ACOUSTIC, fs=6.6)
    arrow(ax, (3.9, 12.4), (5.3, 13.3), connectionstyle="arc3,rad=0.2")
    arrow(ax, (3.9, 11.2), (5.3, 9.9), connectionstyle="arc3,rad=-0.2")
    box(ax, 10.5, 10.6, 2.6, 2.4, "enc.", fc="white", ec="0.25", fs=7)
    arrow(ax, (9.3, 13.4), (10.7, 12.6), connectionstyle="arc3,rad=0.15")
    arrow(ax, (9.3, 9.8), (10.7, 10.9), connectionstyle="arc3,rad=-0.15")
    ax.plot([14.6], [13.0], "o", color=ACOUSTIC, ms=8)
    ax.plot([14.6], [10.2], "o", color=ACOUSTIC, ms=8)
    ax.plot([17.0], [12.6], "o", color="0.6", ms=8)
    ax.annotate("", xy=(14.6, 12.6), xytext=(14.6, 10.6),
                arrowprops={"arrowstyle": "<|-|>", "color": HEALTHY_GREEN, "lw": 1.4})
    ax.annotate("", xy=(16.6, 12.7), xytext=(15.1, 12.9),
                arrowprops={"arrowstyle": "<|-|>", "color": ANOMALY, "lw": 1.2})
    ax.text(14.0, 11.6, "attract", fontsize=6.2, color=HEALTHY_GREEN, rotation=90,
            ha="center", va="center")
    ax.text(17.0, 11.9, "repel\n(other windows)", fontsize=6.2, color=ANOMALY,
            ha="center", va="top")
    ax.text(15.4, 8.6, "NT-Xent loss", fontsize=6.8, color="0.3", ha="center")

    # (b) V2
    ax.text(24.0, 15.2, "(b) V2: joint fusion objectives", fontsize=8.5,
            fontweight="bold", color="0.2", ha="center")
    box(ax, 19.6, 11.6, 8.8, 2.6,
        "contrastive on $c_t$\n(two augmented views)", fc=CTX_BG, ec=INTERMEDIATE, fs=6.8)
    box(ax, 19.6, 8.0, 8.8, 2.9,
        "latent masked modelling:\nmask tokens, regress fused\noutput to pre-mask embedding",
        fc=GRAY_BG, ec="0.4", fs=6.6)
    box(ax, 19.6, 4.4, 8.8, 2.9,
        "cross-modal alignment (CMA):\nacoustic and vibration\nsummaries agree",
        fc="#eef7fa", ec="#3b8ea5", fs=6.6)
    ax.text(24.0, 3.6, "positives: same window across modalities;\nnegatives: other windows in batch",
            fontsize=6.4, ha="center", va="top", color="0.35")

    ax.text(15.0, 1.4,
            "labels are never used: invariance to nuisance augmentation + cross-modal agreement\n"
            "shape $c_t$ into an operating-state code",
            fontsize=7.0, ha="center", color="0.3", style="italic")
    save(fig, "fig13_ssl_objectives")


# ─────────────────────────────────────────────────────────────────────────
# 14 — conditional flow head
# ─────────────────────────────────────────────────────────────────────────
def fig14_flow() -> None:
    fig, ax = blank_axes((6.8, 3.6), xlim=(0, 30), ylim=(0, 14))

    box(ax, 0.2, 8.6, 4.0, 2.8, "pooled fused\nfeature $x$", fc=GRAY_BG, ec="0.35", fs=7)
    # coupling layers
    x0 = 5.6
    for i in range(6):
        box(ax, x0 + i * 2.5, 8.6, 2.1, 2.8, f"$f_{{{i + 1}}}$", fc="#fbe9e9", ec=ANOMALY, fs=8)
        if i < 5:
            arrow(ax, (x0 + i * 2.5 + 2.1, 10.0), (x0 + (i + 1) * 2.5, 10.0), color=ANOMALY, lw=1.0)
    arrow(ax, (4.2, 10.0), (x0, 10.0), color="0.3")
    box(ax, 21.6, 8.6, 3.4, 2.8, "$z \\sim \\mathcal{N}(0, I)$", fc="white", ec="0.25", fs=7.5)
    arrow(ax, (x0 + 5 * 2.5 + 2.1, 10.0), (21.6, 10.0), color=ANOMALY, lw=1.0)

    # FiLM conditioning
    ax.add_patch(Circle((12.9, 4.6), 1.0, fc=INTERMEDIATE, ec="none", zorder=3))
    ax.text(12.9, 4.6, "$c_t$", ha="center", va="center", fontsize=9.5, color="white",
            fontweight="bold", zorder=4)
    for i in range(6):
        arrow(ax, (12.9, 5.6), (x0 + i * 2.5 + 1.05, 8.55), color=INTERMEDIATE, lw=0.8,
              connectionstyle=f"arc3,rad={(i - 2.5) * 0.07:.2f}", style="-|>")
    ax.text(12.9, 2.9, "FiLM: $h \\leftarrow (1+\\gamma(\\tilde{c}))\\odot h + \\beta(\\tilde{c})$,"
                       "  $\\gamma,\\beta$ zero-init", fontsize=7.0, ha="center", color=INTERMEDIATE)

    ax.text(13.9, 12.45, "affine coupling, alternating masks; scale bounded "
                         "$s = s_{max}\\tanh(\\cdot)$", fontsize=6.8, ha="center",
            color="0.35", va="bottom")

    # score + threshold
    box(ax, 26.0, 8.6, 3.8, 2.8, "score\n$-\\log p(x|c_t)$", fc="#fbe9e9", ec=ANOMALY, fs=7)
    arrow(ax, (25.0, 10.0), (26.0, 10.0), color=ANOMALY)
    box(ax, 22.4, 3.4, 7.2, 3.0,
        "per-cluster threshold:\nK-means($c_t$), $K{=}3$;\n95th pct of healthy score",
        fc=ALERT_BG, ec=LATE_FUSION, fs=6.8)
    arrow(ax, (27.9, 8.55), (27.4, 6.5), color="0.35")
    ax.text(22.0, 1.6, "alert when score exceeds the threshold of its nearest cluster",
            fontsize=6.6, color="0.35", ha="center", va="center")
    ax.text(0.3, 13.45, "trained on healthy windows only (max. likelihood)",
            fontsize=6.8, color="0.35", ha="left", va="bottom")
    save(fig, "fig14_conditional_flow_head")


# ─────────────────────────────────────────────────────────────────────────
# 15 — localization head
# ─────────────────────────────────────────────────────────────────────────
def fig15_localization() -> None:
    fig, ax = blank_axes((7.2, 4.2), xlim=(0, 32), ylim=(0, 17))

    # acoustic pathway
    ax.text(7.0, 16.2, "acoustic pathway", fontsize=8, color=ACOUSTIC, fontweight="bold", ha="center")
    box(ax, 0.2, 12.6, 4.4, 2.6, "mic pairs\nGCC-PHAT", fc=AC_BG, ec=ACOUSTIC, fs=7)
    box(ax, 5.6, 12.6, 5.0, 2.6, "SRP-PHAT volume\n(fixed 3D grid, m)", fc=AC_BG, ec=ACOUSTIC, fs=7)
    box(ax, 11.6, 12.6, 4.4, 2.6, "3D CNN\nlogit map", fc=AC_BG, ec=ACOUSTIC, fs=7)
    box(ax, 17.0, 12.6, 5.4, 2.6, "soft-argmax\n$\\hat{p}_0$ (init, m)", fc="white", ec=ACOUSTIC, fs=7)
    for xa, xb in [(4.6, 5.6), (10.6, 11.6), (16.0, 17.0)]:
        arrow(ax, (xa, 13.9), (xb, 13.9), color=ACOUSTIC, lw=1.0)

    # vibration pathway
    ax.text(7.0, 10.9, "structure-borne pathway", fontsize=8, color=VIBRATION,
            fontweight="bold", ha="center")
    box(ax, 0.2, 7.0, 4.4, 3.0, "accel pairs\nGCC delays", fc=VIB_BG, ec=VIBRATION, fs=7)
    box(ax, 5.6, 7.0, 6.0, 3.0, "8-d TDOA tokens\npath diff via\n$c \\approx 2000$ m/s",
        fc=VIB_BG, ec=VIBRATION, fs=6.6)
    box(ax, 12.6, 7.0, 4.8, 3.0, "set-transformer\npool (any #pairs)", fc=VIB_BG,
        ec=VIBRATION, fs=6.8)
    for xa, xb in [(4.6, 5.6), (11.6, 12.6)]:
        arrow(ax, (xa, 8.5), (xb, 8.5), color=VIBRATION, lw=1.0)

    # residual head
    box(ax, 20.0, 5.8, 7.0, 4.6,
        "FiLM($c_t$) residual MLP\n$\\Delta = r\\tanh(\\mathrm{MLP}(\\cdot))$\n$r = 0.20$ m bound",
        fc=CTX_BG, ec=INTERMEDIATE, fs=7)
    arrow(ax, (17.4, 8.5), (20.0, 8.4), color=VIBRATION)
    arrow(ax, (19.7, 13.9), (22.4, 10.5), connectionstyle="arc3,rad=-0.25", color=ACOUSTIC)
    ax.add_patch(Circle((18.4, 4.0), 0.95, fc=INTERMEDIATE, ec="none", zorder=3))
    ax.text(18.4, 4.0, "$c_t$", ha="center", va="center", fontsize=9, color="white",
            fontweight="bold", zorder=4)
    arrow(ax, (19.3, 4.4), (20.6, 5.7), color=INTERMEDIATE)

    box(ax, 28.2, 7.4, 3.6, 3.0, "$\\hat{p} = \\hat{p}_0 + \\Delta$", fc="white", ec="0.2", fs=7.5)
    arrow(ax, (27.0, 8.4), (28.2, 8.6), color="0.3")
    arrow(ax, (22.4, 13.9), (29.6, 10.5), connectionstyle="arc3,rad=-0.3", color=ACOUSTIC, ls=":")
    ax.text(27.6, 13.4, "$\\hat{p}_0$", fontsize=7, color=ACOUSTIC)

    ax.text(7.6, 3.4,
            "burst-aware: SRP computed on the\nhighest-energy sub-window;\n"
            "zero-init FiLM $\\Rightarrow$ unconditional ablation\n= soft-argmax + plain residual",
            fontsize=6.4, color="0.35", ha="center", va="top")
    ax.text(27.2, 4.6, "confidence gate (late fusion):\ntrust the pipeline that moved\n"
                       "its own estimate less", fontsize=6.4, color=LATE_FUSION,
            ha="center", va="top")
    save(fig, "fig15_localization_head")


# ─────────────────────────────────────────────────────────────────────────
# 16 — streaming inference
# ─────────────────────────────────────────────────────────────────────────
def fig16_streaming() -> None:
    fig, ax = blank_axes((7.2, 3.4), xlim=(0, 32), ylim=(0, 13))

    # always-on region
    ax.add_patch(plt.Rectangle((0.2, 4.6), 21.0, 7.6, fc="#eef4ee", ec="none", zorder=0))
    ax.text(10.7, 11.6, "runs on every window", fontsize=7.5, color="0.35", ha="center")
    ax.add_patch(plt.Rectangle((21.8, 4.6), 9.9, 7.6, fc="#fdf2f2", ec="none", zorder=0))
    ax.text(26.7, 11.6, "runs only on alert", fontsize=7.5, color=ANOMALY, ha="center")

    # stream
    rng = np.random.default_rng(3)
    xs = np.linspace(0.8, 4.6, 90)
    ax.plot(xs, 8.6 + 0.5 * np.sin(xs * 7) + 0.22 * rng.standard_normal(90), color="0.45", lw=0.8)
    ax.text(2.7, 10.3, "window stream\n(fine inference stride)", fontsize=6.6, ha="center", color="0.35")

    box(ax, 5.6, 7.4, 4.6, 2.6, "V1+V2 encoder\n$\\to c_t$, $x$", fc=CTX_BG, ec=INTERMEDIATE, fs=7)
    box(ax, 11.4, 7.4, 4.4, 2.6, "V3 score\n$-\\log p(x|c_t)$", fc="#fbe9e9", ec=ANOMALY, fs=7)
    box(ax, 16.8, 7.4, 4.0, 2.6, "hysteresis\nevent rule", fc=ALERT_BG, ec=LATE_FUSION, fs=7)
    box(ax, 22.6, 7.4, 4.4, 2.6, "V4 localization\n$\\hat{p}(x,y,z)$", fc=VIB_BG, ec=VIBRATION, fs=7)
    box(ax, 28.0, 7.4, 3.6, 2.6, "operator\nevent", fc="white", ec="0.25", fs=7)

    arrow(ax, (4.7, 8.7), (5.6, 8.7))
    arrow(ax, (10.2, 8.7), (11.4, 8.7))
    arrow(ax, (15.8, 8.7), (16.8, 8.7))
    arrow(ax, (20.8, 8.7), (22.6, 8.7), color=ANOMALY)
    arrow(ax, (27.0, 8.7), (28.0, 8.7))
    ax.text(21.7, 9.4, "alert", fontsize=6.4, color=ANOMALY, ha="center")

    # score trace with hysteresis
    xs2 = np.linspace(5.6, 20.4, 240)
    sc = 1.3 + 0.18 * rng.standard_normal(240)
    sc += 1.9 * np.exp(-0.5 * ((xs2 - 14.6) / 0.7) ** 2)
    ax.plot(xs2, sc + 0.6, color="0.4", lw=0.9)
    ax.axhline(3.0, xmin=5.6 / 32, xmax=20.4 / 32, color=ANOMALY, lw=0.8, ls="--")
    ax.axhline(2.45, xmin=5.6 / 32, xmax=20.4 / 32, color=LATE_FUSION, lw=0.8, ls=":")
    ax.text(5.45, 3.1, "open", fontsize=6.0, color=ANOMALY, ha="right", va="center")
    ax.text(5.45, 2.35, "close", fontsize=6.0, color=LATE_FUSION, ha="right", va="center")
    band = (sc + 0.6) > 3.0
    if band.any():
        x_on = xs2[band].min()
        x_off = xs2[band].max()
        ax.axvspan(x_on, x_off, ymin=0.06, ymax=0.34, color=ANOMALY, alpha=0.15)
        ax.text((x_on + x_off) / 2, 4.55, "event", fontsize=6.2, color=ANOMALY,
                ha="center", va="bottom")
    ax.text(5.8, 0.2, "score timeline $\\to$ discrete events (Schmitt trigger)",
            fontsize=6.4, color="0.35", ha="left", va="bottom")
    save(fig, "fig16_streaming_inference")


HEALTHY_GREEN = "#2e7d32"


def main() -> None:
    style.apply_style()
    fig09_system_overview()
    fig10_system()
    fig11_encoders()
    fig12_fusion()
    fig13_ssl()
    fig14_flow()
    fig15_localization()
    fig16_streaming()


if __name__ == "__main__":
    main()
