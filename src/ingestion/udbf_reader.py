"""Minimal reader for Gantner UDBF (Universal Data Binary File) format.

Supports the specific ROWII variants produced by the Illwerke plant DAQ system
(Gantner Instruments DAQ firmware V2.17/V2.18).

File structure:
    [binary header]
    [JSON metadata block]
    [channel definition blocks]  -- each: binary fields + name (length-prefixed) + UUID
    [separator: 0x00 + 0x2A* N]
    [data records: uint64 timestamp_ns + N_ch * float32 values]

Timestamp units: nanoseconds relative to Gantner epoch.
Start datetime is parsed from the filename and used for absolute timestamps.
"""

from __future__ import annotations

import json
import re
import struct
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

_UUID_RE = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# Matches both Allg (ROWII_Allg_M1__2026-04-15_00-00-00.dat)
# and RMS/FFT  (RmsGeneratorMic__0_2026-04-15_00-00-00_000000.dat) filenames.
_FNAME_DT_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")


class UDBFFile(NamedTuple):
    """Parsed content of one UDBF file."""

    source_name: str
    """DeviceLocation / SourceName from JSON metadata."""
    channel_names: list[str]
    """Channel names in order."""
    timestamps_ns: np.ndarray
    """Absolute nanosecond timestamps (int64), aligned to file start via filename."""
    data: np.ndarray
    """Shape (n_records, n_channels), dtype float32."""
    file_start: datetime
    """UTC datetime parsed from filename."""
    sample_interval_ns: int
    """Nominal interval between samples in nanoseconds."""


def _find_json_block(data: bytes) -> tuple[int, int, dict]:
    """Return (json_start, json_end, metadata_dict)."""
    start = data.find(ord("{"))
    if start == -1:
        raise ValueError("No JSON block found in UDBF file")
    depth = 0
    for i in range(start, len(data)):
        if data[i] == ord("{"):
            depth += 1
        elif data[i] == ord("}"):
            depth -= 1
            if depth == 0:
                end = i
                break
    else:
        raise ValueError("Unterminated JSON block in UDBF file")
    meta = json.loads(data[start : end + 1].decode("ascii", errors="replace"))
    return start, end, meta


def _parse_channel_names(data: bytes, after: int) -> list[str]:
    """Parse Gantner UDBF channel names from the binary header section.

    After the JSON metadata block there is a 35-byte preamble (timestamps +
    sample rate as float64) followed by:
        uint16 LE  n_channels
        For each channel:
            uint16 LE  name_len  (byte count including null terminator)
            bytes      name      (null-terminated ASCII)
            ...variable metadata ending with a UUID...
            bytes      uuid      (36 ASCII chars + null)
            (next channel name_len follows immediately)

    The metadata between name and UUID is variable length, so we scan forward
    for each UUID boundary using the regex.
    """
    PREAMBLE_SIZE = (
        35  # 9-byte marker + 2-byte tag + 8-byte ts + 8-byte scale + 8-byte sample_rate
    )
    pos = after + PREAMBLE_SIZE

    if pos + 2 > len(data):
        return []

    n_channels = struct.unpack_from("<H", data, pos)[0]
    pos += 2

    # Sanity check
    if n_channels == 0 or n_channels > 500:
        uuids_found = list(_UUID_RE.finditer(data, after))
        n_channels = len(uuids_found)
        pos = after + PREAMBLE_SIZE + 2  # reset

    names: list[str] = []
    for ch_idx in range(n_channels):
        if pos + 2 > len(data):
            break

        name_len = struct.unpack_from("<H", data, pos)[0]
        pos += 2

        if name_len < 2 or name_len > 80 or pos + name_len > len(data):
            break

        name_bytes = data[pos : pos + name_len]
        try:
            name = name_bytes.rstrip(b"\x00").decode("ascii")
            if not name or not all(0x20 <= ord(c) <= 0x7E for c in name):
                name = f"ch{ch_idx}"
        except (UnicodeDecodeError, ValueError):
            name = f"ch{ch_idx}"

        names.append(name)
        pos += name_len

        # Advance past variable metadata to just after the UUID null terminator.
        uuid_match = _UUID_RE.search(data, pos, pos + 300)
        if uuid_match:
            pos = uuid_match.end() + 1  # +1 for null terminator
        else:
            break

    return names if names else [f"ch{i}" for i in range(n_channels)]


def _find_data_start(data: bytes, after_last_uuid_end: int) -> int:
    """Skip null bytes and asterisk separators to find data start offset."""
    pos = after_last_uuid_end
    while pos < len(data) and data[pos] in (0, 0x2A):
        pos += 1
    return pos


def _parse_filename_start(path: Path) -> datetime:
    """Extract file-start UTC datetime from Gantner filename convention.

    Filename pattern: ..._YYYY-MM-DD_HH-MM-SS[_...].dat
    Treated as LOCAL time; returned as naive UTC-equivalent for consistency.
    """
    m = _FNAME_DT_RE.search(path.stem)
    if not m:
        raise ValueError(f"Cannot parse datetime from filename: {path.name}")
    date_part = m.group(1)  # YYYY-MM-DD
    time_part = m.group(2).replace("-", ":")  # HH:MM:SS
    return datetime.fromisoformat(f"{date_part}T{time_part}")


def read_udbf(path: str | Path) -> UDBFFile:
    """Read a Gantner UDBF file and return parsed channel data.

    Parameters
    ----------
    path:
        Path to the .dat file.

    Returns
    -------
    UDBFFile
        Named tuple with channel_names, timestamps_ns, data, etc.
    """
    path = Path(path)
    with path.open("rb") as fh:
        raw = fh.read()

    _, json_end, meta = _find_json_block(raw)

    uuids = list(_UUID_RE.finditer(raw, json_end + 1))
    if not uuids:
        raise ValueError(f"No channel UUIDs found in {path.name}")

    last_uuid_end = uuids[-1].end()
    channel_names = _parse_channel_names(raw, json_end + 1)

    data_start = _find_data_start(raw, last_uuid_end)
    n_channels = len(channel_names)

    # Record layout: 8-byte uint64 timestamp + n_channels * float32
    record_size = 8 + n_channels * 4
    data_section = raw[data_start:]
    n_records = len(data_section) // record_size

    if n_records == 0:
        raise ValueError(f"No data records found in {path.name}")

    # Parse as structured numpy array for efficiency
    dtype = np.dtype([("ts", "<u8")] + [(f"c{i}", "<f4") for i in range(n_channels)])
    expected_bytes = n_records * record_size
    records = np.frombuffer(data_section[:expected_bytes], dtype=dtype)

    timestamps_raw = records["ts"].astype(np.int64)
    data_arr = np.column_stack(
        [records[f"c{i}"].astype(np.float32) for i in range(n_channels)]
    )

    # Build absolute timestamps from filename + relative offset
    file_start = _parse_filename_start(path)
    # Convert raw timestamp offset to nanoseconds from file start
    ts_relative_ns = timestamps_raw - timestamps_raw[0]
    sample_interval_ns = (
        int(np.round(np.median(np.diff(ts_relative_ns)))) if n_records > 1 else 0
    )
    # Absolute: file_start in nanoseconds (treat file_start as t=0)
    timestamps_abs_ns = ts_relative_ns

    source_name = meta.get("SourceName", meta.get("DeviceLocation", path.stem))

    return UDBFFile(
        source_name=source_name,
        channel_names=channel_names,
        timestamps_ns=timestamps_abs_ns,
        data=data_arr,
        file_start=file_start,
        sample_interval_ns=sample_interval_ns,
    )


def read_udbf_folder(
    folder: str | Path,
    pattern: str = "*.dat",
    sort: bool = True,
) -> list[UDBFFile]:
    """Read all UDBF files matching *pattern* in *folder*.

    Files are concatenated in chronological order if *sort* is True.
    """
    folder = Path(folder)
    files = list(folder.glob(pattern))
    if sort:
        files = sorted(files)
    return [read_udbf(f) for f in files]


def concat_udbf(files: list[UDBFFile]) -> UDBFFile:
    """Concatenate a list of UDBFFile objects (same channel layout assumed)."""
    if not files:
        raise ValueError("Empty file list")
    if len(files) == 1:
        return files[0]

    # Validate channel names match
    ref_channels = files[0].channel_names
    for f in files[1:]:
        if f.channel_names != ref_channels:
            raise ValueError(
                f"Channel name mismatch: {f.source_name} has {f.channel_names} "
                f"vs expected {ref_channels}"
            )

    # Build continuous timestamps: each file's ts is relative to its own start.
    # We offset each file's timestamps by its file_start relative to first file.
    ref_start = files[0].file_start
    all_ts = []
    all_data = []
    for f in files:
        # Seconds offset from reference file start
        offset_s = (f.file_start - ref_start).total_seconds()
        offset_ns = int(offset_s * 1_000_000_000)
        all_ts.append(f.timestamps_ns + offset_ns)
        all_data.append(f.data)

    return UDBFFile(
        source_name=files[0].source_name,
        channel_names=ref_channels,
        timestamps_ns=np.concatenate(all_ts),
        data=np.vstack(all_data),
        file_start=ref_start,
        sample_interval_ns=files[0].sample_interval_ns,
    )
