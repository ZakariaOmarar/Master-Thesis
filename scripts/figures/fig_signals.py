"""Raw-signal evidence figures, computed from the real recordings on disk
(figures 6, 7, 8, 9 of the figure plan).

   6  cross-modal synchronization evidence (envelope cross-correlation)
   7  per-mode acoustic signatures (log-mel + CWT, ST/TU/PU)
   8  vibration feature triplet, healthy vs knock (D4 raw waveform)
   9  acoustic PSD with the ROW II characteristic frequencies marked

Loads recordings through the production ingestion pipeline
(``TestDatasetLoader``) and the production feature extractors, so the
figures show exactly what the model sees.

Run with:  python -m scripts.figures.fig_signals
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from scripts.figures import style
from scripts.figures.style import (
    ACOUSTIC,
    ANOMALY,
    MODE_COLORS,
    VIBRATION,
    save,
)
from src.features.acoustic_representations import compute_cwt_scalogram
from src.features.audio_spectral import compute_log_mel_spectrogram
from src.features.vibration_temporal import compute_vibration_input_stack
from src.ingestion.sync_verification import (
    _decimate_to_rate,
    _hilbert_envelope_mean,
    _normalised_xcorr,
    _vibration_envelope_mean,
    verify_paired_sync,
)
from src.modeling.orchestration.full_run import resolved_loader

# Characteristic ROW II frequencies (Results ch., tab:characteristic_frequencies)
# (frequency, label, annotation row 0/1 — staggered to avoid collisions)
CHARACTERISTIC_HZ = [
    (5.87, "shaft fund.\n5.87 Hz", 0),
    (43.75, "runner-blade\npassing (7x)", 1),
    (50.0, "mains\n50 Hz", 0),
    (100.0, "rotor-pole\npassing (16x)", 1),
    (117.3, "guide-vane\npassing (20x)", 0),
]


def _segments(dataset_yaml: str):
    return resolved_loader(dataset_yaml).list_segments()




# ─────────────────────────────────────────────────────────────────────────
# 7 — per-mode acoustic signatures
# ─────────────────────────────────────────────────────────────────────────
def fig07_mode_signatures() -> None:
    wanted = {"Standstill": None, "Turbine": None, "Pump": None}
    for seg in _segments("d2.yaml"):
        if seg.mode_label in wanted and wanted[seg.mode_label] is None and not seg.is_anomaly:
            wanted[seg.mode_label] = seg
    missing = [m for m, s in wanted.items() if s is None]
    if missing:  # fall back to D1 for any missing mode
        for seg in _segments("d1.yaml"):
            if seg.mode_label in missing and wanted[seg.mode_label] is None and not seg.is_anomaly:
                wanted[seg.mode_label] = seg

    slice_s = 12.0
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.8), constrained_layout=True)
    for col, mode in enumerate(["Standstill", "Turbine", "Pump"]):
        seg = wanted[mode]
        ds = seg.segment
        fs = ds.mic_sample_rate
        T = ds.mic_data.shape[1]
        n = min(int(slice_s * fs), T)
        start = max(0, (T - n) // 2)
        x = ds.mic_data[0, start: start + n]

        mel = compute_log_mel_spectrogram(x, fs)
        ax = axes[0, col]
        im0 = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma",
                        extent=(0, n / fs, 0, mel.shape[0]),
                        vmin=mel.max() - 80, vmax=mel.max())
        ax.set_title(mode, fontsize=9, color=MODE_COLORS[mode], fontweight="bold")
        if col == 0:
            ax.set_ylabel("log-mel band\n(20 Hz - 8 kHz)", fontsize=7.5)
        ax.set_xticks([])
        ax.grid(False)

        cwt = compute_cwt_scalogram(x, fs)
        ax = axes[1, col]
        im1 = ax.imshow(cwt, origin="lower", aspect="auto", cmap="magma",
                        extent=(0, n / fs, 0, cwt.shape[0]))
        if col == 0:
            ax.set_ylabel("CWT scale\n(20 - 250 Hz, log)", fontsize=7.5)
        ax.set_xlabel("time (s)", fontsize=7.5)
        ax.grid(False)

    fig.colorbar(im0, ax=axes[0, :], shrink=0.85, pad=0.01, label="dB")
    fig.colorbar(im1, ax=axes[1, :], shrink=0.85, pad=0.01, label="log power")
    save(fig, "fig07_mode_signatures")


# ─────────────────────────────────────────────────────────────────────────
# 8 — vibration triplet, healthy vs knock
# ─────────────────────────────────────────────────────────────────────────
def fig08_vibration_triplet() -> None:
    segs = _segments("d4.yaml")
    healthy = next(s for s in segs if not s.is_anomaly)
    knock = next(s for s in segs if s.is_anomaly)

    win_s = 6.0
    rows = ["amplitude (z)", "Hilbert envelope (z)", "impulsiveness\n(excess kurtosis)"]
    fig, axes = plt.subplots(3, 2, figsize=(6.8, 3.9), sharex="col", constrained_layout=True)

    for col, (seg, title, color) in enumerate(
        [(healthy, "healthy (D4, speed bucket)", VIBRATION),
         (knock, f"knock recording (D4, {knock.source_dir.name})", ANOMALY)]
    ):
        ds = seg.segment
        fs = ds.accel_sample_rate
        stack = compute_vibration_input_stack(ds.accel_data, sample_rate=fs)
        ch = stack[0]  # first accelerometer
        n = int(win_s * fs)
        if col == 0:
            start = max(0, (ch.shape[1] - n) // 2)
        else:  # center the slice on the strongest impulsive event
            start = int(np.clip(np.argmax(ch[2]) - n // 2, 0, ch.shape[1] - n))
        t = np.arange(n) / fs
        for row in range(3):
            ax = axes[row, col]
            ax.plot(t, ch[row, start: start + n], color=color, lw=0.7)
            if row == 0:
                ax.set_title(title, fontsize=8.5, color=color)
            if col == 0:
                ax.set_ylabel(rows[row], fontsize=7.2)
        axes[2, col].set_xlabel("time (s)", fontsize=8)

    # share y-limits per row so the contrast is honest
    for row in range(3):
        lims = [axes[row, c].get_ylim() for c in range(2)]
        lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
        for c in range(2):
            axes[row, c].set_ylim(lo, hi)

    save(fig, "fig08_vibration_triplet")




def main() -> None:
    style.apply_style()
    fig07_mode_signatures()
    fig08_vibration_triplet()


if __name__ == "__main__":
    main()
