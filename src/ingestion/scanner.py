"""Directory scanner for recordings containing WAV and vibration CSV files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# NOTE on the `sensor` capture group: `[^_]+` matches a single underscore-free
# token.  For D3 stereo filenames like `recorded_D_l_Pump.wav` this matches
# `sensor="D"` and `recording="l_Pump"` — the `D_l` stereo-channel ID is not
# preserved by the scanner.  This is harmless in practice because:
#   - The scanner only uses `sensor` for flat-layout *recording grouping*, where
#     the per-recording sensor identity is rebuilt downstream by
#     `test_dataset_loader._sensor_id` from the full filename.
#   - All four current campaigns (D1-D4) ship as bundled directories (one
#     folder per recording with all sensor files inside), so the flat-layout
#     code path is exercised only by `tests/unit/test_ingestion_dual_layout.py`
#     with single-token sensor names ("B", "C", "D", "E").
# If a future campaign uses flat-layout with multi-token sensor names, replace
# the `sensor` group with a layout-aware split that recognises the trailing
# mode token (matching `_sensor_id`'s convention).
_RECORDED_RE = re.compile(
    r"^recorded_(?P<sensor>[^_]+)_(?P<recording>.+)\.wav$", re.IGNORECASE
)
_VIBRATION_RE = re.compile(
    r"^vibration_(?P<sensor>[^_]+)_(?P<recording>.+)\.csv$", re.IGNORECASE
)


@dataclass(frozen=True)
class RecordingGroup:
    """A resolved recording unit containing matched mic and vibration files."""

    recording_id: str
    source_dir: Path
    mic_files: tuple[Path, ...]
    vibration_files: tuple[Path, ...]


class RecordingScanner:
    """Find recording folders that contain both mic WAV and vibration CSV files."""

    def __init__(
        self,
        root_dir: Path,
        mic_glob: str = "recorded_*.wav",
        vibration_glob: str = "vibration_*.csv",
    ) -> None:
        self._root_dir = Path(root_dir)
        self._mic_glob = mic_glob
        self._vibration_glob = vibration_glob

        if not self._root_dir.exists() or not self._root_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self._root_dir}")

    def scan(self) -> list[Path]:
        """Return sorted recording directories with required file types."""
        candidates: list[Path] = []

        if self._is_recording_dir(self._root_dir):
            candidates.append(self._root_dir)

        for child in sorted(self._root_dir.iterdir()):
            if child.is_dir() and self._is_recording_dir(child):
                candidates.append(child)

        return sorted(set(candidates), key=lambda p: p.name)

    def scan_groups(self) -> list[RecordingGroup]:
        """Return resolved recording groups from root and immediate subdirectories.

        Supports two layouts:
        1) Bundled directories: one folder contains all files for a recording.
        2) Flat grouped files: files named as
           ``recorded_<sensor>_<recording>.wav`` and
           ``vibration_<sensor>_<recording>.csv`` and grouped by ``<recording>``.
        """
        groups: list[RecordingGroup] = []

        groups.extend(self._scan_directory_groups(self._root_dir))
        for child in sorted(self._root_dir.iterdir()):
            if child.is_dir():
                groups.extend(self._scan_directory_groups(child))

        return sorted(
            groups,
            key=lambda g: (str(g.source_dir).lower(), g.recording_id.lower()),
        )

    def _is_recording_dir(self, directory: Path) -> bool:
        has_mic = any(directory.glob(self._mic_glob))
        has_vibration = any(directory.glob(self._vibration_glob))
        return has_mic and has_vibration

    def _scan_directory_groups(self, directory: Path) -> list[RecordingGroup]:
        mic_files = sorted(directory.glob(self._mic_glob))
        vibration_files = sorted(directory.glob(self._vibration_glob))

        if not mic_files or not vibration_files:
            return []

        grouped = self._group_flat_layout(directory, mic_files, vibration_files)
        if grouped:
            return grouped

        return [
            RecordingGroup(
                recording_id=directory.name,
                source_dir=directory,
                mic_files=tuple(mic_files),
                vibration_files=tuple(vibration_files),
            )
        ]

    def _group_flat_layout(
        self,
        directory: Path,
        mic_files: list[Path],
        vibration_files: list[Path],
    ) -> list[RecordingGroup]:
        mic_by_recording: dict[str, list[Path]] = {}
        vib_by_recording: dict[str, list[Path]] = {}

        for path in mic_files:
            match = _RECORDED_RE.match(path.name)
            if match is None:
                return []
            recording_id = match.group("recording")
            mic_by_recording.setdefault(recording_id, []).append(path)

        for path in vibration_files:
            match = _VIBRATION_RE.match(path.name)
            if match is None:
                return []
            recording_id = match.group("recording")
            vib_by_recording.setdefault(recording_id, []).append(path)

        common_ids = sorted(set(mic_by_recording) & set(vib_by_recording))
        if not common_ids:
            return []

        groups: list[RecordingGroup] = []
        for recording_id in common_ids:
            groups.append(
                RecordingGroup(
                    recording_id=recording_id,
                    source_dir=directory,
                    mic_files=tuple(sorted(mic_by_recording[recording_id])),
                    vibration_files=tuple(sorted(vib_by_recording[recording_id])),
                )
            )
        return groups
