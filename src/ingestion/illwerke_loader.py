"""Load and synchronize Illwerke ROWII UDBF data streams.

Two sources:
  ROWII_Allg_M1/          -- plant process variables at 10 Hz (ground-truth validator)
  ROWII_Anomalie_Erkennung/RMS/  -- acoustic + vibration RMS at 1 Hz (model input)

Both are stored as Gantner UDBF .dat files, one sub-folder per day (YYYYMMDD).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .udbf_reader import UDBFFile, concat_udbf, read_udbf_folder

# ---------------------------------------------------------------------------
# Public contract
# ---------------------------------------------------------------------------


class IllwerkeCampaign(NamedTuple):
    """Synchronized, aligned arrays for one campaign period."""

    rms: np.ndarray
    """Shape (T, 16) float32 — acoustic/vibration RMS at 1 Hz."""
    allg: np.ndarray
    """Shape (T, N_allg) float32 — plant process variables decimated to 1 Hz."""
    timestamps_ns: np.ndarray
    """Shape (T,) int64 — absolute nanosecond timestamps (1 Hz grid)."""
    channel_names_rms: list[str]
    """16 channel names in column order of `rms`."""
    channel_names_allg: list[str]
    """N_allg channel names in column order of `allg`."""


# ---------------------------------------------------------------------------
# RMS source names (sub-folder prefixes inside RMS/)
# ---------------------------------------------------------------------------

_RMS_SOURCES: list[str] = [
    "RmsGeneratorMic",
    "RmsGeneratorVib",
    "RmsTurbineMic",
    "RmsTurbineVib",
]

_ALLG_SOURCE = "ROWII_Allg_M1"
_RMS_ROOT = "ROWII_Anomalie_Erkennung/RMS"

_ALLG_SAMPLE_HZ = 10
_RMS_SAMPLE_HZ = 1
_TARGET_HZ = 1
_ALLG_DECIMATE = _ALLG_SAMPLE_HZ // _TARGET_HZ  # 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _day_dirs(root: Path, days: Sequence[date | str] | None) -> list[Path]:
    """Return sorted day sub-directories matching YYYYMMDD pattern."""
    if days is not None:
        out: list[Path] = []
        for d in days:
            name = d.strftime("%Y%m%d") if isinstance(d, date) else str(d)
            p = root / name
            if p.is_dir():
                out.append(p)
            else:
                warnings.warn(f"Day directory not found: {p}", stacklevel=2)
        return sorted(out)
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())


def _load_one_source(source_dirs: list[Path]) -> UDBFFile:
    """Read and concatenate all .dat files across a list of directories."""
    all_files: list[UDBFFile] = []
    for d in source_dirs:
        files = read_udbf_folder(d, pattern="*.dat", sort=True)
        all_files.extend(files)
    if not all_files:
        raise FileNotFoundError(f"No .dat files found in {source_dirs}")
    return concat_udbf(all_files)


def _to_second_grid(
    udbf: UDBFFile, target_hz: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a UDBFFile onto a uniform integer-second grid.

    Returns (timestamps_ns_grid, data_grid) where data_grid is (T, C) float32.
    Uses nearest-neighbour lookup per grid point — sufficient for mode detection
    where temporal resolution > 1 s is irrelevant.
    """
    ts = udbf.timestamps_ns.astype(np.int64)
    # Build grid: one point per second from first to last timestamp
    t_start = int(ts[0])
    t_end = int(ts[-1])
    step_ns = int(1_000_000_000 // target_hz)
    grid_ns = np.arange(t_start, t_end + step_ns, step_ns, dtype=np.int64)

    # Nearest-neighbour assignment: for each grid point find closest raw sample
    idx = np.searchsorted(ts, grid_ns)
    idx = np.clip(idx, 0, len(ts) - 1)
    # Check both neighbours and pick the closer one
    idx_prev = np.maximum(idx - 1, 0)
    use_prev = np.abs(ts[idx_prev] - grid_ns) < np.abs(ts[idx] - grid_ns)
    idx = np.where(use_prev, idx_prev, idx)

    return grid_ns, udbf.data[idx].astype(np.float32)


def _median_decimate(data: np.ndarray, factor: int) -> np.ndarray:
    """Decimate (T, C) by computing block median of size `factor`."""
    T, C = data.shape
    T_trim = (T // factor) * factor
    return np.median(data[:T_trim].reshape(T_trim // factor, factor, C), axis=1).astype(
        np.float32
    )


def _align_arrays(
    ts_a: np.ndarray,
    data_a: np.ndarray,
    ts_b: np.ndarray,
    data_b: np.ndarray,
    tolerance_ns: int = 2_000_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the overlapping time range and align two 1-Hz grids.

    Returns (common_timestamps_ns, aligned_a, aligned_b).
    Both grids must already be at 1 Hz (1-second steps). Uses set intersection
    of second-rounded timestamps.
    """
    # Round to nearest second for matching
    sec_a = (ts_a // 1_000_000_000).astype(np.int64)
    sec_b = (ts_b // 1_000_000_000).astype(np.int64)

    common_sec = np.intersect1d(sec_a, sec_b)
    if len(common_sec) == 0:
        raise ValueError(
            "RMS and Allg_M1 streams share no overlapping seconds. "
            "Check that the campaign date ranges match."
        )

    mask_a = np.isin(sec_a, common_sec)
    mask_b = np.isin(sec_b, common_sec)

    aligned_a = data_a[mask_a]
    aligned_b = data_b[mask_b]
    common_ts = ts_a[mask_a]

    # Sanity: lengths must match after masking
    if aligned_a.shape[0] != aligned_b.shape[0]:
        n = min(aligned_a.shape[0], aligned_b.shape[0])
        aligned_a = aligned_a[:n]
        aligned_b = aligned_b[:n]
        common_ts = common_ts[:n]

    return common_ts, aligned_a, aligned_b


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_rms_campaign(
    data_root: str | Path,
    days: Sequence[date | str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all RMS streams and merge into a single (T, 16) array at 1 Hz.

    Parameters
    ----------
    data_root:
        Root of the Illwerke data, e.g. ``E:/MasterThesisData/illwerke-data-230426``.
    days:
        Iterable of ``date`` objects or ``"YYYYMMDD"`` strings to restrict the
        campaign period.  ``None`` loads all available days.

    Returns
    -------
    timestamps_ns : (T,) int64
    rms_data      : (T, 16) float32
    channel_names : list of 16 channel name strings
    """
    data_root = Path(data_root)
    rms_root = data_root / _RMS_ROOT

    per_source_ts: list[np.ndarray] = []
    per_source_data: list[np.ndarray] = []
    all_ch_names: list[str] = []

    for src_name in _RMS_SOURCES:
        src_dirs = _day_dirs(
            rms_root / src_name if (rms_root / src_name).exists() else rms_root, days
        )
        # Handle the layout: files live directly in the day directories
        # with names like RmsGeneratorMic__0_YYYY-MM-DD_HH-MM-SS_000000.dat
        if not src_dirs:
            # Try flat layout: all day dirs are directly under rms_root
            day_dirs = _day_dirs(rms_root, days)
            src_files: list[UDBFFile] = []
            for dd in day_dirs:
                files = read_udbf_folder(dd, pattern=f"{src_name}*.dat", sort=True)
                src_files.extend(files)
        else:
            src_files = []
            for dd in src_dirs:
                files = read_udbf_folder(dd, pattern=f"{src_name}*.dat", sort=True)
                src_files.extend(files)

        if not src_files:
            # Fallback: search all day dirs for this prefix
            all_day_dirs = _day_dirs(rms_root, days)
            for dd in all_day_dirs:
                files = read_udbf_folder(dd, pattern=f"{src_name}*.dat", sort=True)
                src_files.extend(files)

        if not src_files:
            raise FileNotFoundError(
                f"No files found for RMS source '{src_name}' under {rms_root}"
            )

        merged = concat_udbf(src_files)
        ts_grid, data_grid = _to_second_grid(merged, target_hz=_RMS_SAMPLE_HZ)

        per_source_ts.append(ts_grid)
        per_source_data.append(data_grid)  # (T_i, 4)
        all_ch_names.extend(merged.channel_names)

    # Intersect timestamps across all four sources and concatenate columns
    common_ts = per_source_ts[0]
    for ts_i in per_source_ts[1:]:
        sec_c = (common_ts // 1_000_000_000).astype(np.int64)
        sec_i = (ts_i // 1_000_000_000).astype(np.int64)
        common_sec = np.intersect1d(sec_c, sec_i)
        common_ts = common_ts[np.isin(sec_c, common_sec)]

    aligned_sources: list[np.ndarray] = []
    common_sec = (common_ts // 1_000_000_000).astype(np.int64)
    for ts_i, data_i in zip(per_source_ts, per_source_data):
        sec_i = (ts_i // 1_000_000_000).astype(np.int64)
        mask = np.isin(sec_i, common_sec)
        aligned_sources.append(data_i[mask])

    rms_data = np.concatenate(aligned_sources, axis=1)  # (T, 16)
    return common_ts, rms_data, all_ch_names


def load_allg_campaign(
    data_root: str | Path,
    days: Sequence[date | str] | None = None,
    forced_drop_channels: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Allg_M1 plant process variables, decimated to 1 Hz.

    Parameters
    ----------
    data_root:
        Root of the Illwerke data.
    days:
        Campaign days.  ``None`` loads all available days.
    forced_drop_channels:
        Channel names to remove from the returned array (e.g. uninstalled
        sensors that may be present in the .dat files but carry no signal).

    Returns
    -------
    timestamps_ns : (T,) int64
    allg_data     : (T, N_ch) float32
    channel_names : list of channel name strings
    """
    data_root = Path(data_root)
    allg_root = data_root / _ALLG_SOURCE

    day_dirs = _day_dirs(allg_root, days)
    all_files: list[UDBFFile] = []
    for dd in day_dirs:
        files = read_udbf_folder(dd, pattern="*.dat", sort=True)
        all_files.extend(files)

    if not all_files:
        raise FileNotFoundError(f"No Allg_M1 .dat files found under {allg_root}")

    merged = concat_udbf(all_files)

    # Decimate 10 Hz → 1 Hz with median filter (anti-aliasing)
    ts_grid, data_10hz = _to_second_grid(merged, target_hz=_ALLG_SAMPLE_HZ)

    # Now reduce to 1 Hz by taking every _ALLG_DECIMATE-th point
    # (already on a 0.1-s grid — just take every 10th)
    ts_1hz = ts_grid[::_ALLG_DECIMATE]
    data_1hz = _median_decimate(data_10hz, factor=_ALLG_DECIMATE)

    # Trim to equal length (rounding artefacts)
    n = min(len(ts_1hz), data_1hz.shape[0])
    ts_out, data_out, names_out = ts_1hz[:n], data_1hz[:n], merged.channel_names

    # Drop explicitly excluded channels (uninstalled / disconnected sensors).
    if forced_drop_channels:
        drop_set = set(forced_drop_channels)
        keep_idx = [i for i, ch in enumerate(names_out) if ch not in drop_set]
        data_out = data_out[:, keep_idx]
        names_out = [names_out[i] for i in keep_idx]

    return ts_out, data_out, names_out


def load_campaign(
    data_root: str | Path,
    days: Sequence[date | str] | None = None,
    forced_drop_channels: Sequence[str] | None = None,
    allg_drop_channels: Sequence[str] | None = None,
) -> IllwerkeCampaign:
    """Load, synchronize, and return a complete ``IllwerkeCampaign``.

    Parameters
    ----------
    data_root:
        Root of the Illwerke data directory.
    days:
        Campaign days to include.  ``None`` loads all.
    forced_drop_channels:
        Channel names to exclude from the RMS stream.  The keep_mask is applied
        separately in ``fit_feature_transform``.
    allg_drop_channels:
        Channel names to exclude from the Allg_M1 stream (uninstalled sensors
        that may appear in the .dat files).

    Returns
    -------
    IllwerkeCampaign
        Both streams aligned to a common 1-Hz time grid.
    """
    ts_rms, rms_data, ch_rms = load_rms_campaign(data_root, days)
    ts_allg, allg_data, ch_allg = load_allg_campaign(
        data_root, days, forced_drop_channels=allg_drop_channels
    )

    common_ts, rms_aligned, allg_aligned = _align_arrays(
        ts_rms, rms_data, ts_allg, allg_data
    )

    return IllwerkeCampaign(
        rms=rms_aligned,
        allg=allg_aligned,
        timestamps_ns=common_ts,
        channel_names_rms=ch_rms,
        channel_names_allg=ch_allg,
    )
