from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from src.data import DataSegment


def test_datasegment_from_arrays_basic() -> None:
    mic_sr = 16_000
    accel_sr = 4
    duration_s = 2.0

    mic = np.random.default_rng(0).normal(0, 0.01, size=(9, int(mic_sr * duration_s)))
    accel = np.random.default_rng(1).normal(
        0, 0.1, size=(4, int(accel_sr * duration_s))
    )

    seg = DataSegment.from_arrays(
        mic_data=mic,
        accel_data=accel,
        start_time=datetime(2026, 3, 26, tzinfo=UTC),
        mic_sr=mic_sr,
        accel_sr=accel_sr,
        metadata={"state_code": "TU"},
    )

    assert seg.n_mic_channels == 9
    assert seg.n_accel_channels == 4
    assert seg.n_mic_samples == int(mic_sr * duration_s)
    assert seg.n_accel_samples == int(accel_sr * duration_s)
    assert seg.metadata["state_code"] == "TU"
