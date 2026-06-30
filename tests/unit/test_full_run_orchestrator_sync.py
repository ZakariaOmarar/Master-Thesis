"""Regression test for the 2026-05-18 orchestrator-sync fix.

Before the fix:
  ``full_run.resolved_loader`` returned a vanilla TestDatasetLoader,
  and the orchestrator's sync-correction block called
  ``s.segment.mic_data = mic_corr`` post-load.  Because DataSegment is
  ``@dataclass(frozen=True)``, that assignment raised
  ``FrozenInstanceError`` on every recording — silently caught by a
  bare ``except Exception`` that counted the failure as ``n_skipped``.
  Net effect: **no recording was ever sync-corrected in production**;
  the V0/V1/V2/V3/V4 downstream stages all consumed unaligned data.

After the fix:
  ``resolved_loader`` constructs the loader with
  ``sync_correct=True`` so the adapter applies the four-gate sync
  correction at load time (before the frozen DataSegment is built),
  and the orchestrator's audit loop just reads
  ``segment.metadata['sync_correction']``.  No post-load mutation.

This test pins both invariants:
  1. ``resolved_loader`` propagates ``vibration_format`` (so D4's
     ``"raw"`` format isn't silently demoted to ``"peak"``).
  2. ``resolved_loader`` returns a loader whose loaded segments
     carry a populated ``metadata['sync_correction']`` payload — i.e.
     sync correction was actually attempted (applied or gated, but
     never silently skipped).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from src.ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.requires_data


def _write_wav_int16(path: Path, sr: int = 16_000, duration_s: float = 1.0) -> None:
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    x = (0.2 * np.sin(2 * np.pi * 220.0 * t) * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, x)


def _write_peak_vibration_csv(path: Path, n_rows: int = 16) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["esp_time_us", "amplitude", "frequency"]
        )
        writer.writeheader()
        for i in range(n_rows):
            writer.writerow(
                {
                    "esp_time_us": 1_000_000 + i * 250_000,
                    "amplitude": 100.0 + i,
                    "frequency": 125.0,
                }
            )


def test_resolved_loader_propagates_vibration_format_and_sync_correct() -> None:
    """The fix in ``full_run.resolved_loader``: sync_correct=True is on by
    default, with the orchestrator's historical four-gate thresholds, and
    ``vibration_format`` is preserved when the spec is reconstructed.
    """
    from src.modeling.orchestration.full_run import resolved_loader

    # D4 is the dataset where the old `resolved_loader` silently dropped
    # the raw-vibration format and fell back to peak. Under the new
    # `vibration_format: auto` policy, the YAML declares "auto" and the
    # adapter resolves it to "raw" at file-read time (since D4 ships only
    # `vibration_raw_*.csv`). Both invariants matter — pin both.
    L = resolved_loader("d4.yaml")
    assert L.spec.vibration_format == "auto"
    from src.ingestion.adapters import resolve_vibration_format
    # Verify the resolution policy lands on "raw" for D4's actual CSVs.
    sample_csvs = list((L.spec.root / "speed1").rglob("vibration*.csv"))
    assert sample_csvs, "no vibration CSVs found under D4/speed1 (test premise broken)"
    assert resolve_vibration_format(sample_csvs, "auto") == "raw"
    # And the adapter inside the loader must have sync_correct enabled —
    # this is the lever that prevents the FrozenInstanceError regression.
    assert L._adapter._sync_correct is True
    # The orchestrator-historical kwargs must round-trip into the adapter
    # so the chapter 6 audit-table semantics carry over.
    assert L._adapter._sync_correct_kwargs["max_offset_s"] == 0.5
    assert L._adapter._sync_correct_kwargs["confidence_floor"] == 1.5
    assert L._adapter._sync_correct_kwargs["drift_tolerance_s"] == 0.010
    assert L._adapter._sync_correct_kwargs["min_offset_to_correct_s"] == 0.001
    assert L._adapter._sync_correct_kwargs["n_sub_segments"] == 5


def test_loader_with_sync_correct_populates_metadata_per_segment(tmp_path: Path) -> None:
    """End-to-end: load segments via TestDatasetLoader(sync_correct=True) and
    verify each segment's metadata carries a populated sync-correction
    report.  Pre-fix, this metadata didn't exist; the orchestrator's
    post-load mutation silently failed and the only signal of the broken
    pipeline was n_skipped == n_total in the metrics.json — a metric a
    reader could easily miss.
    """
    # Build a minimal D1-style dataset under tmp_path so we can configure
    # the loader without depending on real-data presence.
    root = tmp_path / "mini_d1"
    root.mkdir()
    pump_dir = root / "Pump"
    pump_dir.mkdir()
    for sensor in "BCDE":
        _write_wav_int16(pump_dir / f"recorded_{sensor}.wav", duration_s=2.0)
        _write_peak_vibration_csv(pump_dir / f"vibration_{sensor}.csv", n_rows=8)

    spec = DatasetSpec(
        id="d1",
        root=root,
        n_mics=4,
        n_vibrations=4,
        accel_target_sr=4,
        position_source="default",
        label_scheme="d1_mode",
    )
    L = TestDatasetLoader(
        spec,
        sync_correct=True,
        sync_correct_kwargs=dict(
            max_offset_s=0.5,
            n_sub_segments=5,
            confidence_floor=1.5,
            drift_tolerance_s=0.010,
            min_offset_to_correct_s=0.001,
            use_fractional_shift=True,
        ),
    )
    segments = L.list_segments()
    assert len(segments) >= 1

    for s in segments:
        report = s.segment.metadata.get("sync_correction")
        # The pre-fix bug looked exactly like report=None (because the
        # mutation path silently no-op'd and never stored anything).  The
        # whole point of the fix is that every segment now carries a
        # non-None report.
        assert report is not None, (
            "sync_correction metadata is None — the load-time sync "
            "correction was not applied; the orchestrator pattern has "
            "regressed to the silent no-op."
        )
        # Every audit field the chapter 6 sync diagnostic depends on must
        # be present; if one disappears, the metrics aggregation in
        # `full_run.py` would silently skip that field instead of
        # erroring.
        for key in (
            "applied",
            "reason",
            "applied_offset_s",
            "audit_offset_s",
            "audit_confidence",
            "acoustic_envelope_kurtosis",
            "stability_is_stable",
            "stability_n_high_conf",
            "stability_drift_slope_s_per_s",
            "residual_uncertainty_s",
        ):
            assert key in report, f"sync_correction report missing {key!r}"


def test_orchestrator_audit_block_aggregates_metadata_without_mutation(
    tmp_path: Path,
) -> None:
    """Reproduce the orchestrator's audit-aggregation walk in isolation and
    verify it produces non-zero counts (the pre-fix counts were
    ``n_applied = 0, n_skipped = N`` for every dataset because every
    in-place mutation raised FrozenInstanceError).
    """
    root = tmp_path / "mini_d1"
    root.mkdir()
    pump_dir = root / "Pump"
    pump_dir.mkdir()
    for sensor in "BCDE":
        _write_wav_int16(pump_dir / f"recorded_{sensor}.wav", duration_s=2.0)
        _write_peak_vibration_csv(pump_dir / f"vibration_{sensor}.csv", n_rows=8)

    spec = DatasetSpec(
        id="d1",
        root=root,
        n_mics=4,
        n_vibrations=4,
        accel_target_sr=4,
        position_source="default",
        label_scheme="d1_mode",
    )
    L = TestDatasetLoader(
        spec,
        sync_correct=True,
        sync_correct_kwargs=dict(max_offset_s=0.5, confidence_floor=1.5),
    )

    # Mirror the orchestrator's audit walk.
    n_applied = 0
    n_rejected_flat_envelope = 0
    n_rejected_low_conf = 0
    n_rejected_drift = 0
    n_rejected_below_floor = 0
    n_skipped = 0
    for s in L.list_segments():
        report = s.segment.metadata.get("sync_correction")
        if report is None:
            n_skipped += 1
            continue
        reason = (report.get("reason") or "").lower()
        if report.get("applied"):
            n_applied += 1
        elif "near-gaussian" in reason:
            n_rejected_flat_envelope += 1
        elif "stability" in reason or "drift" in reason:
            n_rejected_drift += 1
        elif "confidence" in reason or "uninformative" in reason:
            n_rejected_low_conf += 1
        elif "below" in reason or "already aligned" in reason:
            n_rejected_below_floor += 1

    n_total = (
        n_applied
        + n_rejected_flat_envelope
        + n_rejected_low_conf
        + n_rejected_drift
        + n_rejected_below_floor
        + n_skipped
    )
    # Pre-fix invariant that was BROKEN:  n_skipped == n_total because the
    # mutation always FrozenInstanceError'd.  Post-fix invariant: every
    # recording lands in one of the *categorised* buckets (applied or one
    # of the four rejected reasons), and n_skipped stays at 0.
    assert n_total >= 1
    assert n_skipped == 0, (
        f"n_skipped={n_skipped} of {n_total} — at least one segment did "
        f"not produce a categorised sync-correction outcome; the silent "
        f"no-op regression may have returned."
    )
    # On synthetic steady-tone audio the four-gate pipeline is expected to
    # land in `near-gaussian` (envelope kurtosis below 1).  Just check that
    # at least one recording was categorised — the specific bucket depends
    # on the synthetic content and is not load-bearing for this test.
    categorised = (
        n_applied + n_rejected_flat_envelope + n_rejected_low_conf
        + n_rejected_drift + n_rejected_below_floor
    )
    assert categorised >= 1
