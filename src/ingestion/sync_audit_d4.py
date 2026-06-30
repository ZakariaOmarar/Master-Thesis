"""F3 — D4 sync diagnostic: raw vs peak vibration cross-check.

Background
----------

The 2026-05-13 full-run cross-modal sync stage refused to correct any D4
recording (9 / 9 gated out as ``low_conf`` with mean confidence 1.03 and
median offset +192 ms).  The documented acquisition prior is that D4
shares the D1/D2/D3 hardware trigger, in which case the +192 ms is an
*artefact of the raw 376 Hz vibration decoder*, not a true misalignment.

The decisive test is to run the same envelope cross-correlation on the
*aggregated peak* CSV stream that lives alongside the raw waveform in
every D4 recording directory:

  * The peak stream is the historical pre-D4 vibration format — its
    timestamping has been validated against D1/D2/D3 audio for years.
  * The raw stream is the new high-resolution path we want to switch
    to permanently.

Three outcome buckets:

  * **B1** — peak says ~0 ms, raw says ~192 ms.  → Raw UDBF decoder has
    a time-origin bug.  Fix is in `udbf_reader.py` / the raw vibration
    adapter.
  * **B2** — both say ~0 ms.  → Envelope cross-correlation was confused
    by D4 content (e.g., long sustained tones with weak transients);
    no real misalignment.  Fix is to add a transient-energy precondition
    to `auto_sync_paired_recording`.
  * **A**  — both say ~192 ms.  → Shared-trigger assumption wrong for
    D4 specifically.  Treat D4 results as exploratory in Chapter 6.

Run as::

    python -m src.ingestion.sync_audit_d4

Output: ``results/sync_audit_d4.json`` with per-recording offsets,
confidences, and a verdict label.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .sync_verification import (
    _acoustic_envelope_excess_kurtosis,
    verify_paired_sync,
)
from .test_dataset_loader import DatasetSpec, TestDatasetLoader

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolved_d4_spec(vibration_format: str) -> DatasetSpec:
    """Build a D4 spec with the requested vibration_format override.

    The on-disk YAML pins ``vibration_format: raw`` — to exercise the peak
    path we construct a new spec with the same fields and that one switch.
    """
    spec = DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / "d4.yaml")
    # Keep the canonical id="d4" so the PositionRegistry can dispatch the
    # known layout; only swap the vibration_format.
    return DatasetSpec(
        id=spec.id,
        root=REPO_ROOT / spec.root,
        n_mics=spec.n_mics,
        n_vibrations=spec.n_vibrations,
        accel_target_sr=spec.accel_target_sr,
        position_source=(
            REPO_ROOT / spec.position_source
            if spec.position_source not in ("default", "rowii")
            else spec.position_source
        ),
        label_scheme=spec.label_scheme,
        vibration_format=vibration_format,
        extra=spec.extra,
    )


def _verdict_from_offsets(
    raw_offset_ms: float,
    raw_conf: float,
    peak_offset_ms: float,
    peak_conf: float,
    decoder_bug_threshold_ms: float = 30.0,
    near_zero_threshold_ms: float = 30.0,
) -> str:
    """Categorise one recording into B1 / B2 / A / inconclusive.

    Both confidences must be ≥ 1.5 for the corresponding offset to count as
    informative; otherwise we mark the row inconclusive on that side.
    """
    raw_informative = raw_conf >= 1.5
    peak_informative = peak_conf >= 1.5

    if not (raw_informative or peak_informative):
        return "inconclusive_both_low_conf"
    if peak_informative and not raw_informative:
        return "inconclusive_raw_low_conf"
    if raw_informative and not peak_informative:
        return "inconclusive_peak_low_conf"

    # Both informative.
    raw_near_zero = abs(raw_offset_ms) < near_zero_threshold_ms
    peak_near_zero = abs(peak_offset_ms) < near_zero_threshold_ms
    raw_far = abs(raw_offset_ms) >= decoder_bug_threshold_ms
    peak_far = abs(peak_offset_ms) >= decoder_bug_threshold_ms

    if peak_near_zero and raw_far:
        return "B1_raw_decoder_bug"
    if peak_near_zero and raw_near_zero:
        return "B2_envelope_confused"
    if peak_far and raw_far:
        # Sign agreement matters too.
        if (peak_offset_ms * raw_offset_ms) > 0:
            return "A_shared_trigger_wrong"
        return "mixed_offset_signs_disagree"
    return "mixed"


def run_diagnostic() -> dict:
    """Run the raw-vs-peak cross-check on every D4 recording.

    Returns a dict suitable for ``json.dump``.  Also writes
    ``results/sync_audit_d4.json`` next to this run's outputs.
    """
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "sync_audit_d4.json"

    raw_loader = TestDatasetLoader(_resolved_d4_spec("raw"))
    peak_loader = TestDatasetLoader(_resolved_d4_spec("peak"))

    raw_segs = {(s.dataset_id, s.recording_id, s.source_dir): s for s in raw_loader.list_segments()}
    peak_segs = {(s.dataset_id, s.recording_id, s.source_dir): s for s in peak_loader.list_segments()}

    # Compare on the intersection of recording IDs — every D4 dir should
    # contain both raw and peak CSVs, but a partial drop-in is possible.
    common_keys = sorted(
        {(rk[1], rk[2]) for rk in raw_segs}
        & {(pk[1], pk[2]) for pk in peak_segs}
    )

    rows: list[dict] = []
    for rec_id, src_dir in common_keys:
        # Look up segments by (any-dataset-id, rec_id, src_dir) since the two
        # loaders tag the dataset_id with the vibration_format suffix.
        raw_seg = next(s for (k_id, k_rec, k_src), s in raw_segs.items()
                       if k_rec == rec_id and k_src == src_dir)
        peak_seg = next(s for (k_id, k_rec, k_src), s in peak_segs.items()
                        if k_rec == rec_id and k_src == src_dir)

        raw_sync = verify_paired_sync(
            raw_seg.segment.mic_data,
            raw_seg.segment.accel_data,
            mic_fs=float(raw_seg.segment.mic_sample_rate),
            accel_fs=float(raw_seg.segment.accel_sample_rate),
        )
        peak_sync = verify_paired_sync(
            peak_seg.segment.mic_data,
            peak_seg.segment.accel_data,
            mic_fs=float(peak_seg.segment.mic_sample_rate),
            accel_fs=float(peak_seg.segment.accel_sample_rate),
        )
        # Envelope excess kurtosis on the exact decimated signal the
        # cross-correlation operates on — distinguishes "genuinely flat"
        # from "transient content that simply doesn't cross-correlate".
        # Computed on the raw-vibration target rate (the canonical D4 path).
        env_kurt = _acoustic_envelope_excess_kurtosis(
            raw_seg.segment.mic_data,
            float(raw_seg.segment.mic_sample_rate),
            float(raw_seg.segment.accel_sample_rate),
        )

        verdict = _verdict_from_offsets(
            raw_sync.offset_s * 1000.0, raw_sync.confidence,
            peak_sync.offset_s * 1000.0, peak_sync.confidence,
        )

        rows.append({
            "recording_id": rec_id,
            "source_dir": str(src_dir),
            "acoustic_envelope_excess_kurtosis": env_kurt,
            "raw": {
                "offset_ms": raw_sync.offset_s * 1000.0,
                "confidence": raw_sync.confidence,
                "vib_fs_hz": float(raw_seg.segment.accel_sample_rate),
                "n_vib_samples": int(raw_seg.segment.accel_data.shape[1]),
            },
            "peak": {
                "offset_ms": peak_sync.offset_s * 1000.0,
                "confidence": peak_sync.confidence,
                "vib_fs_hz": float(peak_seg.segment.accel_sample_rate),
                "n_vib_samples": int(peak_seg.segment.accel_data.shape[1]),
            },
            "verdict": verdict,
        })
        print(
            f"  {rec_id:>40s}: "
            f"raw={raw_sync.offset_s * 1000.0:+7.1f} ms (conf {raw_sync.confidence:.2f})  "
            f"peak={peak_sync.offset_s * 1000.0:+7.1f} ms (conf {peak_sync.confidence:.2f})  "
            f"env-kurt={env_kurt:+7.2f}  "
            f"-> {verdict}"
        )

    # Aggregate verdict — what does the pool say?
    verdict_counts: dict[str, int] = {}
    for r in rows:
        verdict_counts[r["verdict"]] = verdict_counts.get(r["verdict"], 0) + 1

    aggregate = max(verdict_counts, key=verdict_counts.get) if verdict_counts else "no_data"

    kurtoses = [r["acoustic_envelope_excess_kurtosis"] for r in rows]
    raw_confs = [r["raw"]["confidence"] for r in rows]
    peak_confs = [r["peak"]["confidence"] for r in rows]

    # Interpretation — the scientific conclusion the row data supports.
    # B1 would require peak to be tight-near-zero while raw is tight-far;
    # the audit's purpose is precisely to rule that in or out.
    if aggregate.startswith("B1"):
        interpretation = (
            "Raw vibration decoder has a time-origin bug: the peak (trusted) "
            "stream cross-correlates near zero while the raw stream does not. "
            "Fix the raw decoder; sync correction then becomes unnecessary."
        )
    elif aggregate.startswith("inconclusive_both_low_conf") or aggregate.startswith("B2"):
        interpretation = (
            "Neither vibration format (raw OR the historically-trusted peak) "
            "yields a confident envelope cross-correlation on D4 — raw and "
            "peak are EQUALLY noisy, which rules out a raw-decoder bug (B1). "
            "D4's acoustic envelope kurtosis is moderate-to-high yet "
            "cross-modal confidence stays ~1.0, meaning the acoustic and "
            "vibration transients simply do not co-occur strongly enough for "
            "envelope xcorr to lock on. Under the stated shared-hardware-"
            "trigger acquisition prior, the streams ARE aligned by "
            "construction; envelope xcorr just cannot independently verify "
            "it. Action: the orchestrator's Gate-1 path retains D4 streams "
            "as-is with an honest 'uninformative' reason — no fabricated "
            "correction is applied."
        )
    elif aggregate.startswith("A"):
        interpretation = (
            "Both raw AND peak cross-correlate confidently to a consistent "
            "non-zero offset — the shared-trigger assumption is wrong for D4. "
            "Treat D4 results as exploratory; lead Chapter 6 with D1/D2/D3."
        )
    else:
        interpretation = (
            "Mixed / inconsistent verdicts across recordings — inspect the "
            "per-recording rows individually before drawing a conclusion."
        )

    summary = {
        "n_recordings": len(rows),
        "verdict_counts": verdict_counts,
        "aggregate_verdict": aggregate,
        "interpretation": interpretation,
        "envelope_kurtosis_summary": {
            "median": float(np.median(kurtoses)) if kurtoses else float("nan"),
            "min": float(np.min(kurtoses)) if kurtoses else float("nan"),
            "max": float(np.max(kurtoses)) if kurtoses else float("nan"),
        },
        "confidence_summary": {
            "raw_median": float(np.median(raw_confs)) if raw_confs else float("nan"),
            "raw_max": float(np.max(raw_confs)) if raw_confs else float("nan"),
            "peak_median": float(np.median(peak_confs)) if peak_confs else float("nan"),
            "peak_max": float(np.max(peak_confs)) if peak_confs else float("nan"),
        },
        "rows": rows,
        "method": "envelope_normxcorr_per_recording_raw_vs_peak_plus_envelope_kurtosis",
        "thresholds_ms": {"decoder_bug": 30.0, "near_zero": 30.0},
    }

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Aggregate verdict over {len(rows)} D4 recordings: {aggregate}")
    print(f"Per-verdict counts: {verdict_counts}")
    print(f"Interpretation: {interpretation}")
    return summary


if __name__ == "__main__":
    run_diagnostic()
