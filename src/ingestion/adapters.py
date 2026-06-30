"""Format adapters for thesis multimodal recordings (WAV + vibration CSV)."""

from __future__ import annotations

import csv
import warnings
from collections.abc import Iterable, Sequence
from datetime import datetime

try:  # ``datetime.UTC`` exists only on Python 3.11+; shim it for 3.10 and earlier.
    from datetime import UTC
except ImportError:  # pragma: no cover - exercised on Python <= 3.10
    from datetime import timezone as _timezone

    UTC = _timezone.utc  # noqa: UP017
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ..config.constants import (
    ACCEL_COUNT,
    ACCEL_SAMPLE_RATE_TARGET,
    MIC_SAMPLE_RATE,
)
from ..data import DataSegment
from ..exceptions import IngestionError

_TIME_KEYS = ("esp_time_us", "timestamp_us", "time_us", "timestamp", "time")
_AMPLITUDE_KEYS = ("amplitude", "amp", "fft_amplitude", "peak_amplitude")
_FREQUENCY_KEYS = ("frequency", "freq", "dominant_frequency", "peak_frequency")
_RAW_VIBRATION_PREFIX = "vibration_raw_"
_PC_TIME_COLUMN = "pc_time"


def resolve_vibration_format(
    paths: Iterable[Path], vibration_format: str
) -> str:
    """Resolve ``vibration_format`` to a concrete ``"peak"`` or ``"raw"`` value.

    ``"auto"`` (the new default) prefers raw when any ``vibration_raw_*.csv``
    is present in ``paths``, falling back to peak otherwise.  Centralised so
    the rule lives in exactly one place and the loader + adapter always
    agree on what "auto" resolves to for a given recording.
    """
    if vibration_format == "auto":
        has_raw = any(Path(p).stem.startswith(_RAW_VIBRATION_PREFIX) for p in paths)
        return "raw" if has_raw else "peak"
    if vibration_format in ("peak", "raw"):
        return vibration_format
    raise IngestionError(
        f"vibration_format must be 'auto', 'peak', or 'raw', got {vibration_format!r}"
    )


def filter_vibration_csv_paths(
    paths: Iterable[Path], vibration_format: str
) -> tuple[Path, ...]:
    """Partition vibration CSVs into raw-waveform vs peak-amplitude streams.

    Both ``vibration_<sensor>.csv`` (peak; D1/D2/D3) and
    ``vibration_raw_<sensor>.csv`` (raw; D4/D5) match the same
    ``vibration_*.csv`` scanner glob, so any code that loads one format
    must filter out the other.  Centralised here so the filter rule (the
    literal ``vibration_raw_`` prefix) lives in exactly one place; both
    :class:`WavVibrationAdapter` and
    :class:`src.ingestion.test_dataset_loader.TestDatasetLoader` call this
    rather than reimplementing the predicate.

    ``vibration_format="auto"`` (the global default) prefers raw whenever
    any ``vibration_raw_*.csv`` is present in the input list.
    """
    paths_list = list(paths)
    resolved = resolve_vibration_format(paths_list, vibration_format)
    if resolved == "raw":
        return tuple(p for p in paths_list if p.stem.startswith(_RAW_VIBRATION_PREFIX))
    return tuple(p for p in paths_list if not p.stem.startswith(_RAW_VIBRATION_PREFIX))


def _sensor_id_from_raw_csv(path: Path) -> str:
    """`vibration_raw_E.csv` -> `E`; `vibration_raw_D.csv` -> `D`."""
    return path.stem.split(_RAW_VIBRATION_PREFIX, 1)[-1]


class WavVibrationAdapter:
    """Read one recording directory into a DataSegment.

    Expected files by default:
    - `recorded_*.wav` microphone files
    - `vibration_*.csv` vibration files with timestamp, amplitude, frequency columns
    """

    def __init__(
        self,
        expected_mic_count: int | None = None,
        expected_accel_count: int = ACCEL_COUNT,
        mic_glob: str = "recorded_*.wav",
        vibration_glob: str = "vibration_*.csv",
        accel_target_sr: int = ACCEL_SAMPLE_RATE_TARGET,
        allowed_mic_counts: tuple[int, ...] = (4, 9),
        vibration_format: str = "auto",
        accel_sr_overrides: dict[str, int] | None = None,
        sync_correct: bool = False,
        sync_correct_kwargs: dict | None = None,
    ) -> None:
        """Build the adapter.

        ``sync_correct`` (default False) is an opt-in flag that runs
        :func:`src.ingestion.sync_verification.auto_sync_paired_recording`
        on every loaded segment.  Default is False because:
          * The historical orchestrator-side correction loop in
            ``full_run.py`` already calls auto-sync (although see TODO in
            that file: the in-place mutation fails on the frozen
            DataSegment, so the orchestrator path is currently a no-op
            and this flag is the only working entry point).
          * The V0 baseline scripts published in `results/` did not
            sync-correct; flipping the default would invalidate their
            numbers without warning.
        Pass ``sync_correct=True`` for any new training / eval pipeline
        where you want sync-aligned segments out of the box, and
        ``sync_correct_kwargs`` to override the gating thresholds
        (``max_offset_s``, ``confidence_floor``, ``drift_tolerance_s``,
        ``min_offset_to_correct_s``, ``min_envelope_kurtosis``).
        """
        if vibration_format not in ("auto", "peak", "raw"):
            raise IngestionError(
                f"vibration_format must be 'auto', 'peak', or 'raw', got {vibration_format!r}"
            )
        self._expected_mic_count = expected_mic_count
        self._expected_accel_count = expected_accel_count
        self._mic_glob = mic_glob
        self._vibration_glob = vibration_glob
        self._accel_target_sr = accel_target_sr
        self._allowed_mic_counts = tuple(
            sorted({int(c) for c in allowed_mic_counts})
        )
        self._vibration_format = vibration_format
        self._accel_sr_overrides = dict(accel_sr_overrides or {})
        self._sync_correct = bool(sync_correct)
        self._sync_correct_kwargs = dict(sync_correct_kwargs or {})

        if self._expected_mic_count is None and not self._allowed_mic_counts:
            raise IngestionError(
                "allowed_mic_counts must be non-empty when expected_mic_count is None"
            )

        if self._expected_mic_count is not None and int(self._expected_mic_count) <= 0:
            raise IngestionError("expected_mic_count must be positive")

    def read_recording_directory(self, recording_dir: Path) -> DataSegment:
        """Read one recording folder and return a synchronized DataSegment."""
        recording_dir = Path(recording_dir)
        if not recording_dir.exists() or not recording_dir.is_dir():
            raise IngestionError(f"Recording directory not found: {recording_dir}")

        mic_files = sorted(recording_dir.glob(self._mic_glob))
        vibration_files = sorted(recording_dir.glob(self._vibration_glob))
        return self.read_recording_files(
            recording_dir=recording_dir,
            mic_files=mic_files,
            vibration_files=vibration_files,
            recording_id=recording_dir.name,
        )

    def read_recording_files(
        self,
        recording_dir: Path,
        mic_files: list[Path] | tuple[Path, ...],
        vibration_files: list[Path] | tuple[Path, ...],
        recording_id: str | None = None,
    ) -> DataSegment:
        """Read one recording from explicit mic and vibration file lists."""
        recording_dir = Path(recording_dir)

        mic_data, mic_sr, mic_paths = self._read_wav_channels(mic_files, recording_dir)
        vib_amp, vib_freq, vib_sr_raw, vib_paths = self._read_vibration_channels(
            vibration_files,
            recording_dir,
        )

        # `_read_vibration_channels` returns either a single shared rate (peak
        # streams; shared firmware clock) or per-channel rates (raw streams;
        # each accelerometer on its own ESP32 + DMA timer).  Compute the
        # effective stream duration off the SLOWEST channel — taking the
        # min ensures we never extrapolate beyond what any single channel
        # actually captured.
        per_channel_rates: np.ndarray | None
        if isinstance(vib_sr_raw, np.ndarray):
            per_channel_rates = vib_sr_raw
            channel_durations = vib_amp.shape[1] / per_channel_rates
            vib_duration = float(channel_durations.min())
        else:
            per_channel_rates = None
            vib_duration = vib_amp.shape[1] / float(vib_sr_raw)

        mic_duration = mic_data.shape[1] / mic_sr
        common_duration = min(mic_duration, vib_duration)

        mic_samples = int(round(common_duration * mic_sr))
        mic_data = mic_data[:, :mic_samples]

        actual_duration = mic_samples / mic_sr
        accel_samples = max(1, int(round(actual_duration * self._accel_target_sr)))

        if per_channel_rates is not None:
            # Raw streams: each channel resampled from its OWN rate to the
            # target, so sample n of every output channel lands at wall-clock
            # n / target_sr regardless of per-board clock jitter.
            accel_data = _resample_channels_per_channel_rate(
                vib_amp,
                per_channel_rates,
                accel_samples,
                float(self._accel_target_sr),
            )
            vib_freq_resampled = _resample_channels_per_channel_rate(
                vib_freq,
                per_channel_rates,
                accel_samples,
                float(self._accel_target_sr),
            )
        else:
            accel_data = _resample_channels(
                vib_amp, float(vib_sr_raw), accel_samples, self._accel_target_sr
            )
            vib_freq_resampled = _resample_channels(
                vib_freq, float(vib_sr_raw), accel_samples, self._accel_target_sr
            )

        # Opt-in cross-modal sync correction.  Runs the four-gate
        # auto-sync pipeline (envelope-kurtosis, audit confidence,
        # stability-across-sub-segments, offset-magnitude) and mutates
        # the local mic_data / accel_data arrays before the DataSegment
        # is built.  We do it HERE, not on the constructed DataSegment,
        # because DataSegment is frozen — any post-construction mutation
        # attempt raises FrozenInstanceError silently when caught.
        sync_report_payload: dict | None = None
        if self._sync_correct:
            from .sync_verification import auto_sync_paired_recording

            mic_data, accel_data, _report = auto_sync_paired_recording(
                mic_data,
                accel_data,
                mic_fs=float(mic_sr),
                accel_fs=float(self._accel_target_sr),
                **self._sync_correct_kwargs,
            )
            # Auto-sync drops leading samples from whichever stream lagged.
            # The two streams may now have slightly different durations;
            # re-truncate both to the shorter common duration so the
            # DataSegment duration contract (mic_samples ≈ accel_samples ·
            # mic_sr / accel_sr) holds within the ±1 sample tolerance of
            # ``__post_init__``.
            mic_dur = mic_data.shape[1] / float(mic_sr)
            vib_dur = accel_data.shape[1] / float(self._accel_target_sr)
            common = float(min(mic_dur, vib_dur))
            mic_data = mic_data[:, : int(round(common * mic_sr))]
            accel_data = accel_data[:, : int(round(common * self._accel_target_sr))]
            vib_freq_resampled = vib_freq_resampled[:, : accel_data.shape[1]]
            sync_report_payload = {
                "applied": bool(_report.applied),
                "reason": str(_report.reason),
                "applied_offset_s": float(_report.applied_offset_s),
                "audit_offset_s": float(_report.audit.offset_s),
                "audit_confidence": float(_report.audit.confidence),
                "acoustic_envelope_kurtosis": float(
                    _report.acoustic_envelope_kurtosis
                ),
                "stability_is_stable": bool(_report.stability.is_stable),
                "stability_n_high_conf": int(
                    _report.stability.n_sub_segments_high_conf
                ),
                "stability_drift_slope_s_per_s": float(
                    _report.stability.drift_slope_s_per_s
                ),
                "residual_uncertainty_s": float(
                    _report.residual_offset_uncertainty_s
                ),
            }

        # Wall-clock start time: parsed from the vibration CSV's `pc_time`
        # column when present (D4 raw), else the earliest WAV/CSV mtime,
        # else load time.  Replaces the previous unconditional
        # `datetime.now()` which gave every DataSegment loaded in a single
        # training session approximately the same `start_time`.
        start_time, start_time_source = _infer_recording_start_time(
            vib_paths, mic_paths
        )

        # Metadata's `vibration_sample_rate_raw` reports the median across
        # channels so the schema stays a single float for both peak and raw
        # paths; `vibration_sample_rate_per_channel` carries the full
        # per-channel vector for raw streams (None for peak).
        if per_channel_rates is not None:
            vib_sr_meta = float(np.median(per_channel_rates))
            per_channel_rates_meta: list[float] | None = per_channel_rates.tolist()
        else:
            vib_sr_meta = float(vib_sr_raw)
            per_channel_rates_meta = None

        metadata = {
            "source": "wav_vibration_csv",
            "recording_dir": str(recording_dir),
            "recording_id": (
                recording_id if recording_id is not None else recording_dir.name
            ),
            "mic_files": [str(p) for p in mic_paths],
            "vibration_files": [str(p) for p in vib_paths],
            "mic_sample_rate_original": mic_sr,
            "vibration_sample_rate_raw": vib_sr_meta,
            "vibration_sample_rate_per_channel": per_channel_rates_meta,
            "vibration_frequencies": vib_freq_resampled,
            "start_time_source": start_time_source,
            "sync_correction": sync_report_payload,
        }

        return DataSegment.from_arrays(
            mic_data=mic_data,
            accel_data=accel_data,
            start_time=start_time,
            mic_sr=mic_sr,
            accel_sr=self._accel_target_sr,
            metadata=metadata,
        )

    def _validate_mic_file_count(self, found_count: int, recording_dir: Path) -> None:
        if self._expected_mic_count is not None:
            if found_count != self._expected_mic_count:
                raise IngestionError(
                    "Expected "
                    f"{self._expected_mic_count} WAV files, found {found_count} in {recording_dir}"
                )
            return

        if found_count not in self._allowed_mic_counts:
            raise IngestionError(
                "Expected WAV count in "
                f"{self._allowed_mic_counts}, found {found_count} in {recording_dir}"
            )

    def _read_wav_channels(
        self,
        wav_paths: list[Path] | tuple[Path, ...],
        recording_dir: Path,
    ) -> tuple[np.ndarray, int, list[Path]]:
        wav_paths = sorted(Path(p) for p in wav_paths)
        self._validate_mic_file_count(len(wav_paths), recording_dir)

        channels: list[np.ndarray] = []
        sample_rate: int | None = None

        for wav_path in wav_paths:
            sr, data = wavfile.read(wav_path)
            if sample_rate is None:
                sample_rate = int(sr)
            elif int(sr) != sample_rate:
                raise IngestionError(
                    f"Sample-rate mismatch: {wav_path.name} has {sr}, expected {sample_rate}"
                )

            if data.ndim != 1:
                raise IngestionError(
                    f"WAV must be mono, got shape {data.shape} in {wav_path.name}"
                )

            if data.dtype == np.int16:
                arr = data.astype(np.float64) / 32768.0
            elif np.issubdtype(data.dtype, np.integer):
                max_abs = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
                arr = data.astype(np.float64) / float(max_abs)
            else:
                # Float WAVs are conventionally in [-1, +1] (IEEE 754).  A
                # peak well outside that range means the writing toolchain
                # stored integer-scale values in a float container (sox /
                # ffmpeg `-c:a pcm_f32le` from int16 sources is the canonical
                # case).  Detect the source bit-depth from the magnitude and
                # normalise; warn loudly so any recording-protocol drift is
                # surfaced rather than silently boosting features by ~32 000×.
                arr = data.astype(np.float64)
                abs_max = float(np.max(np.abs(arr))) if arr.size else 0.0
                if abs_max > 1.5:
                    if abs_max < 32768.0 * 1.5:
                        assumed_bit_depth, scale = 16, 32768.0
                    elif abs_max < 2147483648.0 * 1.5:
                        assumed_bit_depth, scale = 32, 2147483648.0
                    else:
                        raise IngestionError(
                            f"Float WAV {wav_path.name} has |max|={abs_max:.3e}; "
                            f"value range is not consistent with any standard "
                            f"PCM bit depth — file may be corrupted"
                        )
                    warnings.warn(
                        f"Float WAV {wav_path.name} has |max|={abs_max:.1f}, "
                        f"outside the conventional [-1, +1] range; rescaling "
                        f"by 1/{scale:.0f} on the assumption that the writer "
                        f"stored {assumed_bit_depth}-bit PCM values in a float "
                        f"container",
                        stacklevel=2,
                    )
                    arr = arr / scale

            channels.append(arr)

        assert sample_rate is not None
        if sample_rate != MIC_SAMPLE_RATE:
            # Keep strict default expectation for thesis recording protocol.
            raise IngestionError(
                f"Expected mic sample rate {MIC_SAMPLE_RATE} Hz, got {sample_rate} Hz"
            )

        min_len = min(len(ch) for ch in channels)
        mic_data = np.stack([ch[:min_len] for ch in channels])
        return mic_data, sample_rate, wav_paths

    def _read_vibration_channels(
        self,
        csv_paths: list[Path] | tuple[Path, ...],
        recording_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray, float, list[Path]]:
        csv_paths = sorted(Path(p) for p in csv_paths)
        # Filter raw vs peak files based on configured vibration_format.  The
        # `vibration_*.csv` glob in the scanner matches both `vibration_D.csv`
        # (peak) and `vibration_raw_D.csv` (raw); see `filter_vibration_csv_paths`.
        # "auto" resolves to "raw" when raw files are present, else "peak".
        resolved_format = resolve_vibration_format(csv_paths, self._vibration_format)
        csv_paths = list(filter_vibration_csv_paths(csv_paths, resolved_format))

        if len(csv_paths) != self._expected_accel_count:
            raise IngestionError(
                f"Expected {self._expected_accel_count} vibration {resolved_format} CSV files, "
                f"found {len(csv_paths)} in {recording_dir}"
            )

        if resolved_format == "raw":
            return self._read_raw_vibration_channels(csv_paths)

        amp_channels: list[np.ndarray] = []
        freq_channels: list[np.ndarray] = []
        timestamp_channels: list[np.ndarray] = []

        for csv_path in csv_paths:
            amps, freqs, timestamps = _read_vibration_csv(csv_path)
            amp_channels.append(amps)
            freq_channels.append(freqs)
            timestamp_channels.append(timestamps)

        min_len = min(len(ch) for ch in amp_channels)
        amp_data = np.stack([ch[:min_len] for ch in amp_channels])
        freq_data = np.stack([ch[:min_len] for ch in freq_channels])

        # Infer the peak-stream sample rate from the MEDIAN dt across
        # (channels, samples).  Inferring from a single channel's
        # timestamps (the legacy behaviour) lets a single dropped row
        # or clock anomaly on channel 0 bias the entire recording's
        # rate.  The peak streams are shared-clock per firmware design,
        # so the median across channels is the right population
        # statistic for the underlying acquisition rate.
        all_dts: list[int] = []
        for ts in timestamp_channels:
            ts_trim = ts[:min_len]
            if ts_trim.size > 1:
                all_dts.extend(np.diff(ts_trim).tolist())
        if all_dts:
            dt_us = float(np.median(all_dts))
            vib_sr_raw = (
                1_000_000.0 / dt_us if dt_us > 0 else float(self._accel_target_sr)
            )
        else:
            vib_sr_raw = float(self._accel_target_sr)

        if vib_sr_raw <= 0:
            vib_sr_raw = float(self._accel_target_sr)

        return amp_data, freq_data, vib_sr_raw, csv_paths

    def _read_raw_vibration_channels(
        self,
        csv_paths: list[Path],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
        """Raw-waveform path: read every channel's ``vibration_raw_*.csv`` and
        return aligned waveforms together with **per-channel** native rates.

        Each accelerometer in the D4 acquisition rig has its own ESP32, each
        ESP32 has its own DMA timer, so the inferred per-channel sample rates
        can differ by a few percent.  This function returns each channel's
        own rate as an ``(n_channels,)`` array; the caller in
        :meth:`read_recording_files` uses
        :func:`_resample_channels_per_channel_rate` to place every channel
        on the same target time grid — preserving inter-channel timing
        alignment that a single-rate resample would silently destroy.

        The returned ``freq_data`` is a zero array of matching shape so the
        downstream metadata schema (which carries the per-window dominant
        frequency from the peak stream) stays uniform.  The "raw" channel
        unit is the embedded ADC count value; downstream features apply
        per-channel zero-mean centring (`compute_vibration_input_stack`)
        which absorbs any channel-specific bias.

        Raises ``IngestionError`` if per-channel rates differ by more than
        ~10 % — the operational tolerance of the four-ESP32 D4 rig (one
        accelerometer board runs at ~404 Hz, the other three at ~376 Hz,
        a ~7.4 % spread driven by the boards' uncalibrated DMA timers).
        Below 10 % the per-channel resampling below correctly absorbs the
        clock divergence into a common 376 Hz output grid; above 10 %
        indicates a real firmware fault worth investigating before
        training.
        """
        waveforms: list[np.ndarray] = []
        rates: list[float] = []
        for csv_path in csv_paths:
            wav, sr_inferred = _read_vibration_raw_csv(csv_path)
            # If the dataset declares a per-sensor override (e.g. D5 sensor E
            # at 471 Hz vs the 446 Hz dataset-wide rate), use that instead of
            # the timestamp-based inference.  Overrides are firmware-documented
            # rates from `scripts/utils/derive_dataset_sampling_rate.py` and are more
            # reliable than per-recording timestamp inference on short clips.
            sensor_id = _sensor_id_from_raw_csv(csv_path)
            sr = float(self._accel_sr_overrides.get(sensor_id, sr_inferred))
            waveforms.append(wav)
            rates.append(sr)
        rates_arr = np.asarray(rates, dtype=np.float64)
        median_rate = float(np.median(rates_arr))
        max_dev = float(np.max(np.abs(rates_arr - median_rate) / median_rate))
        if max_dev > 0.10:
            raise IngestionError(
                f"Raw vibration channel rates differ by {max_dev * 100:.1f}% "
                f"from the median (rates={rates}, median={median_rate:.2f} Hz). "
                f"Per-channel rates above ~10 % apart indicate a firmware "
                f"clock fault — investigate before training."
            )
        min_len = min(len(w) for w in waveforms)
        amp_data = np.stack([w[:min_len] for w in waveforms])
        freq_data = np.zeros_like(amp_data)
        return amp_data, freq_data, rates_arr, csv_paths


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in fieldnames:
            return name
    return None


def _read_first_pc_time(path: Path) -> datetime | None:
    """Return the first row's ``pc_time`` as a tz-aware UTC datetime, or None.

    The D4 raw-waveform CSVs (and any future board firmware that writes
    wall-clock to ``pc_time``) carry Unix epoch seconds in the ``pc_time``
    column on every DMA flush.  Used as the highest-priority source for
    :func:`_infer_recording_start_time`.  Returns None when the column is
    missing, empty, or unparseable rather than raising — the caller will
    fall back to filesystem mtime.
    """
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or _PC_TIME_COLUMN not in reader.fieldnames:
                return None
            for row in reader:
                raw = (row.get(_PC_TIME_COLUMN) or "").strip()
                if not raw:
                    continue
                t = float(raw)
                return datetime.fromtimestamp(t, tz=UTC)
    except (OSError, ValueError):
        return None
    return None


def _infer_recording_start_time(
    csv_paths: Sequence[Path], wav_paths: Sequence[Path]
) -> tuple[datetime, str]:
    """Best-effort wall-clock start time + provenance tag.

    Returns ``(datetime_utc, source)`` where ``source`` is one of
    ``"pc_time"``, ``"file_mtime"``, or ``"load_time"``.  Priority:

      1. The first valid ``pc_time`` value from any vibration CSV
         (Unix-epoch seconds, written by the D4 raw firmware on every
         DMA flush) — the most accurate source when present.
      2. The earliest filesystem mtime across the recording's WAV and
         CSV files.  Always available for real files; a slightly
         conservative upper bound on the true capture time (firmware
         tends to write at recording end, so mtime ≈ start + duration),
         but vastly more meaningful than load-time wall clock.
      3. ``datetime.now()`` as last resort — used only when the recording
         is being synthesised in memory and neither pc_time nor mtime is
         meaningful (e.g. unit tests with tmp_path WAVs).

    This replaces the previous unconditional ``datetime.now()`` behaviour,
    which gave every DataSegment loaded in a single training session
    approximately the same ``start_time`` regardless of when the
    underlying data was actually captured.
    """
    for path in csv_paths:
        t = _read_first_pc_time(path)
        if t is not None:
            return t, "pc_time"
    paths = list(csv_paths) + list(wav_paths)
    mtimes: list[float] = []
    for p in paths:
        try:
            mtimes.append(p.stat().st_mtime)
        except OSError:
            continue
    if mtimes:
        return datetime.fromtimestamp(min(mtimes), tz=UTC), "file_mtime"
    return datetime.now(UTC), "load_time"


def _read_vibration_raw_csv(path: Path) -> tuple[np.ndarray, float]:
    """Parse a raw-waveform vibration CSV (D4-format).

    Each row is one DMA batch with header
        ``pc_time, esp_time_us, s0, s1, …, s127``
    where the trailing entries are zero-padded; the actual sample count is
    inferred per file as the maximum index whose column is non-zero across
    all rows.  The effective ADC rate is ``samples_per_batch / batch_period``
    inferred from the median of `diff(esp_time_us)`.

    Returns ``(waveform_1d, sample_rate_hz)``.  The waveform is the
    concatenation of all batches' real samples, in row order.
    """
    import pandas as pd

    df = pd.read_csv(path)
    if "esp_time_us" not in df.columns:
        raise IngestionError(f"Raw vibration CSV {path.name} missing esp_time_us column")
    sample_cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
    if not sample_cols:
        raise IngestionError(f"Raw vibration CSV {path.name} has no s* sample columns")
    sample_cols.sort(key=lambda c: int(c[1:]))
    samples = df[sample_cols].to_numpy(dtype=np.float64)  # (n_batches, 128)

    # Per-row trailing-zero count varies; treat the max non-zero column index
    # across all rows as the effective batch size.  Stricter: per-row trim.
    nonzero_mask = samples != 0.0
    last_real_per_row = nonzero_mask.cumsum(axis=1).argmax(axis=1)  # idx of last non-zero
    real_per_row = (last_real_per_row + 1) * nonzero_mask.any(axis=1)
    batch_size = int(np.median(real_per_row[real_per_row > 0]))
    if batch_size <= 0:
        raise IngestionError(f"Raw vibration CSV {path.name} has no usable samples")

    # Estimate ADC rate from median batch period.
    ts = df["esp_time_us"].to_numpy(dtype=np.int64)
    if ts.size >= 2:
        dt_us = float(np.median(np.diff(ts)))
        sample_rate = batch_size / (dt_us / 1_000_000.0) if dt_us > 0 else float(batch_size)
    else:
        sample_rate = float(batch_size)

    # Concatenate the first `batch_size` samples of every row.
    waveform = samples[:, :batch_size].reshape(-1)
    return waveform.astype(np.float64), float(sample_rate)


def _read_vibration_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a peak-amplitude vibration CSV.

    Expected columns (any of the synonyms in ``_TIME_KEYS``, ``_AMPLITUDE_KEYS``,
    ``_FREQUENCY_KEYS``): a timestamp, the peak amplitude, and the dominant
    frequency.  The timestamp is **required** — without it the downstream
    rate-inference step cannot tell a 4 Hz D1 stream from a 16 Hz D3 stream,
    and a silent default would silently misalign the entire recording.
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise IngestionError(f"CSV has no header: {path}")

        fieldnames = [f.strip() for f in reader.fieldnames]
        time_col = _pick_column(fieldnames, _TIME_KEYS)
        amp_col = _pick_column(fieldnames, _AMPLITUDE_KEYS)
        freq_col = _pick_column(fieldnames, _FREQUENCY_KEYS)

        if amp_col is None or freq_col is None:
            raise IngestionError(
                f"CSV {path.name} must include amplitude and frequency columns "
                f"(found {fieldnames})"
            )
        if time_col is None:
            # Silent fallback would assume 4 Hz spacing (250 ms per row) and
            # silently misalign any non-D1 stream.  Reject explicitly.
            raise IngestionError(
                f"CSV {path.name} has no recognised timestamp column "
                f"(expected one of {_TIME_KEYS}); rate inference cannot "
                f"proceed without timestamps"
            )

        timestamps: list[int] = []
        amplitudes: list[float] = []
        frequencies: list[float] = []

        for i, row in enumerate(reader):
            try:
                amplitudes.append(float(row[amp_col]))
                frequencies.append(float(row[freq_col]))
                raw_t_str = row.get(time_col, "")
                if raw_t_str == "":
                    raise IngestionError(
                        f"CSV {path.name} row {i} has empty {time_col!r} value; "
                        f"every row must carry a timestamp for rate inference"
                    )
                raw_t = float(raw_t_str)
                t_us = int(raw_t if "us" in time_col else raw_t * 1_000_000.0)
                timestamps.append(t_us)
            except (TypeError, ValueError) as exc:
                raise IngestionError(
                    f"Invalid numeric row in {path.name}: {exc}"
                ) from exc

    if len(amplitudes) == 0:
        raise IngestionError(f"CSV {path.name} contains no data rows")

    return (
        np.asarray(amplitudes, dtype=np.float64),
        np.asarray(frequencies, dtype=np.float64),
        np.asarray(timestamps, dtype=np.int64),
    )


def _resample_channels(
    data: np.ndarray, src_rate: float, n_out: int, dst_rate: int
) -> np.ndarray:
    """Resample every channel from a single shared ``src_rate`` to ``dst_rate``.

    Used for peak-amplitude vibration streams (D1/D2/D3) where the firmware
    samples all channels off a single shared clock — every channel's sample
    n therefore corresponds to wall-clock n / src_rate, and one source rate
    is the correct model.

    For raw-waveform vibration streams (D4) each accelerometer has its own
    DMA timer on its own ESP32 and the per-channel rates diverge by up to a
    few percent; use :func:`_resample_channels_per_channel_rate` instead so
    each channel is placed on the target time grid using its own rate.
    """
    n_channels, n_in = data.shape
    t_in = np.arange(n_in) / src_rate
    t_out = np.arange(n_out) / dst_rate
    t_out = np.clip(t_out, t_in[0], t_in[-1])

    out = np.empty((n_channels, n_out), dtype=np.float64)
    for ch in range(n_channels):
        out[ch] = np.interp(t_out, t_in, data[ch])
    return out


def _resample_channels_per_channel_rate(
    data: np.ndarray, src_rates: np.ndarray, n_out: int, dst_rate: float
) -> np.ndarray:
    """Resample each channel from its OWN source rate to a common ``dst_rate``.

    Unlike :func:`_resample_channels` (which assumes one shared source rate
    across channels), this places sample n of every output channel at
    wall-clock ``n / dst_rate`` using **that channel's own** source rate to
    interpolate.  Without this, a single median rate applied to all channels
    introduces an inter-channel timing drift of
    ``(src_rates[i] - median) * t`` seconds at time t — for raw D4 vibration
    with ~5 % per-board clock jitter, that's tens of milliseconds drift
    inside a 10 min recording, which destroys any inter-channel TDOA
    estimate the V4 structure-borne head depends on.

    Args:
        data: ``(n_channels, n_in)`` waveform; channel rows share the same
            length but their samples represent slightly different wall-clock
            spacings.
        src_rates: ``(n_channels,)`` per-channel native sample rates in Hz.
        n_out: number of output samples per channel.
        dst_rate: target sample rate in Hz.  Each output sample n lands at
            wall-clock ``n / dst_rate`` regardless of which channel it
            belongs to.
    """
    n_channels, n_in = data.shape
    if src_rates.shape != (n_channels,):
        raise IngestionError(
            f"src_rates must have shape ({n_channels},); got {src_rates.shape}"
        )
    t_out = np.arange(n_out) / float(dst_rate)
    out = np.empty((n_channels, n_out), dtype=np.float64)
    for ch in range(n_channels):
        sr = float(src_rates[ch])
        if sr <= 0:
            raise IngestionError(f"channel {ch} has non-positive rate {sr}")
        t_in = np.arange(n_in) / sr
        t_clip = np.clip(t_out, t_in[0], t_in[-1])
        out[ch] = np.interp(t_clip, t_in, data[ch])
    return out
