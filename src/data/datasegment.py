"""DataSegment - the universal data contract carried through the entire pipeline.

Every step from ingestion through feature extraction operates on DataSegment
objects. The frozen dataclass design enforces immutability: preprocessing steps
return a new DataSegment rather than modifying in place, which prevents
accidental state mutation across pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..exceptions import (
    ChannelCountError,
    DataContractError,
    DataIntegrityError,
    SampleRateError,
)


@dataclass(frozen=True)
class DataSegment:
    """Immutable multimodal segment holding synchronized acoustic and vibration samples.

    Each DataSegment represents one analysis window from a single recording, with
    all mic and accelerometer channels time-aligned to the same start_time. The
    __post_init__ validator catches shape inconsistencies, non-finite values, and
    sample-count mismatches early, so downstream code can trust the invariants.

    Attributes:
        mic_data: Acoustic samples, shape (n_mic_channels, n_mic_samples).
        accel_data: Vibration amplitude samples, shape (n_accel_channels, n_accel_samples).
        mic_sample_rate: Audio sample rate, 16000 Hz for the ROW II dataset.
        accel_sample_rate: Resampled vibration rate, nominally 4 Hz.
        start_time: UTC wall-clock time of the first sample in the window.
        duration_s: Segment duration in seconds.
        channel_names: Ordered names for all channels (mic channels first, accel last).
        metadata: Provenance dict for downstream use, e.g. window_index,
            is_transition_window, recording_id, transition_mask.
    """

    mic_data: np.ndarray
    accel_data: np.ndarray
    mic_sample_rate: int
    accel_sample_rate: int
    start_time: datetime
    duration_s: float
    channel_names: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.mic_sample_rate, int) or self.mic_sample_rate <= 0:
            raise SampleRateError(f"Invalid mic_sample_rate: {self.mic_sample_rate!r}")
        if not isinstance(self.accel_sample_rate, int) or self.accel_sample_rate <= 0:
            raise SampleRateError(
                f"Invalid accel_sample_rate: {self.accel_sample_rate!r}"
            )

        if self.mic_data.ndim != 2:
            raise ChannelCountError(
                "mic_data must be 2D with shape (channels, samples)"
            )
        if self.accel_data.ndim != 2:
            raise ChannelCountError(
                "accel_data must be 2D with shape (channels, samples)"
            )

        expected_mic = int(round(self.duration_s * self.mic_sample_rate))
        expected_accel = int(round(self.duration_s * self.accel_sample_rate))
        if abs(self.mic_data.shape[1] - expected_mic) > 1:
            raise DataContractError(
                f"mic samples {self.mic_data.shape[1]} inconsistent with duration {self.duration_s}s"
            )
        if abs(self.accel_data.shape[1] - expected_accel) > 1:
            raise DataContractError(
                f"accel samples {self.accel_data.shape[1]} inconsistent with duration {self.duration_s}s"
            )

        total_channels = self.mic_data.shape[0] + self.accel_data.shape[0]
        if len(self.channel_names) != total_channels:
            raise ChannelCountError(
                f"channel_names length {len(self.channel_names)} does not match total channels {total_channels}"
            )

        if not np.all(np.isfinite(self.mic_data)):
            raise DataIntegrityError("mic_data contains NaN or Inf")
        if not np.all(np.isfinite(self.accel_data)):
            raise DataIntegrityError("accel_data contains NaN or Inf")

    @property
    def n_mic_channels(self) -> int:
        return int(self.mic_data.shape[0])

    @property
    def n_accel_channels(self) -> int:
        return int(self.accel_data.shape[0])

    @property
    def n_mic_samples(self) -> int:
        return int(self.mic_data.shape[1])

    @property
    def n_accel_samples(self) -> int:
        return int(self.accel_data.shape[1])

    @property
    def mic_channel_names(self) -> tuple[str, ...]:
        return self.channel_names[: self.n_mic_channels]

    @property
    def accel_channel_names(self) -> tuple[str, ...]:
        return self.channel_names[self.n_mic_channels :]

    @classmethod
    def from_arrays(
        cls,
        mic_data: np.ndarray,
        accel_data: np.ndarray,
        start_time: datetime,
        # fmt: skip
        mic_sr: int,
        accel_sr: int,
        metadata: dict[str, Any] | None = None,
    ) -> "DataSegment":
        """Convenience constructor that infers duration and generates default channel names.

        Useful in ingestion adapters and tests where the full set of DataSegment
        parameters is not known in advance. Duration is inferred from mic_data length
        and mic_sr; channel names default to mic_0..mic_N, accel_0..accel_M.
        """
        mic = np.asarray(mic_data, dtype=np.float64)
        accel = np.asarray(accel_data, dtype=np.float64)

        if mic.ndim != 2:
            raise ChannelCountError("mic_data must be 2D")
        if accel.ndim != 2:
            raise ChannelCountError("accel_data must be 2D")

        duration_s = mic.shape[1] / mic_sr

        names = tuple(
            [f"mic_{i}" for i in range(mic.shape[0])]
            + [f"accel_{i}" for i in range(accel.shape[0])]
        )

        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        return cls(
            mic_data=mic,
            accel_data=accel,
            mic_sample_rate=mic_sr,
            accel_sample_rate=accel_sr,
            start_time=start_time,
            duration_s=duration_s,
            channel_names=names,
            metadata=metadata if metadata is not None else {},
        )
