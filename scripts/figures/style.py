"""Shared style for every thesis figure.

One colour per concept, everywhere:
  * modality   — acoustic blue, vibration green (matches the existing
                 sensor-position figure: mics blue, accelerometers green);
  * paradigm   — unimodal inherits its modality colour, late fusion orange,
                 intermediate fusion purple, classical/non-learned grey;
  * mode       — Okabe-Ito colours, colour-blind safe;
  * campaign   — D3/D4/D5 keep the colours of the existing geometry figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "docs" / "final_thesis" / "figures"

# ── modality / paradigm ─────────────────────────────────────────────────
ACOUSTIC = "#1f77b4"
VIBRATION = "#2ca02c"
LATE_FUSION = "#ff7f0e"
INTERMEDIATE = "#9467bd"
CLASSICAL = "#7f7f7f"
ANOMALY = "#d62728"
HEALTHY = "#4d8f4d"

PARADIGM_COLORS = {
    "acoustic": ACOUSTIC,
    "vibration": VIBRATION,
    "late": LATE_FUSION,
    "intermediate": INTERMEDIATE,
    "classical": CLASSICAL,
}

# ── operating modes (Okabe-Ito) ─────────────────────────────────────────
MODE_COLORS = {
    "Pump": "#D55E00",       # vermillion
    "Turbine": "#0072B2",    # blue
    "Standstill": "#999999", # grey
}

# ── campaigns ───────────────────────────────────────────────────────────
CAMPAIGN_COLORS = {
    "D1": "#6baed6",
    "D2": "#74c476",
    "D3": "#ff7f0e",
    "D4": "#d62728",
    "D5": "#9467bd",
}

# ── channel modes of the localization head ─────────────────────────────
CHANNEL_MODE_COLORS = {
    "tdoa_only": VIBRATION,
    "srp_only": ACOUSTIC,
    "both": INTERMEDIATE,
    "vibration_only_learned": "#98df8a",
}
CHANNEL_MODE_LABELS = {
    "tdoa_only": "tdoa-only",
    "srp_only": "srp-only (acoustic)",
    "both": "both (fusion)",
    "vibration_only_learned": "vibration-only (learned)",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str, *, png_only: bool = False) -> None:
    """Save a figure as PDF (vector, for LaTeX) and PNG (preview)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png)
    if not png_only:
        fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {png.relative_to(REPO_ROOT)}" + ("" if png_only else " (+.pdf)"))
