"""Segment loader wrappers for thesis WAV + CSV recordings."""

from __future__ import annotations

from pathlib import Path

from ..data import DataSegment
from .adapters import WavVibrationAdapter
from .scanner import RecordingGroup


class SegmentLoader:
    """Load DataSegment objects from recording directories."""

    def __init__(self, adapter: WavVibrationAdapter | None = None) -> None:
        self._adapter = adapter if adapter is not None else WavVibrationAdapter()

    def load(self, recording: Path | RecordingGroup) -> DataSegment:
        if isinstance(recording, RecordingGroup):
            return self.load_group(recording)
        return self._adapter.read_recording_directory(recording)

    def load_group(self, group: RecordingGroup) -> DataSegment:
        return self._adapter.read_recording_files(
            recording_dir=group.source_dir,
            mic_files=group.mic_files,
            vibration_files=group.vibration_files,
            recording_id=group.recording_id,
        )

    def load_many(self, recordings: list[Path | RecordingGroup]) -> list[DataSegment]:
        return [self.load(item) for item in recordings]
