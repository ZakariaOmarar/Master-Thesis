"""Ingestion utilities for thesis WAV + vibration CSV datasets and Gantner UDBF files."""

from .adapters import WavVibrationAdapter
from .illwerke_loader import (
    IllwerkeCampaign,
    load_allg_campaign,
    load_campaign,
    load_rms_campaign,
)
from .loader import SegmentLoader
from .scanner import RecordingGroup, RecordingScanner
from .udbf_reader import UDBFFile, concat_udbf, read_udbf, read_udbf_folder

__all__ = [
    "IllwerkeCampaign",
    "RecordingGroup",
    "RecordingScanner",
    "SegmentLoader",
    "UDBFFile",
    "WavVibrationAdapter",
    "concat_udbf",
    "load_allg_campaign",
    "load_campaign",
    "load_rms_campaign",
    "read_udbf",
    "read_udbf_folder",
]
