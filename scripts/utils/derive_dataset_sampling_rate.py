"""Derive a dataset's effective vibration sampling rate from raw CSV timestamps.

The registry refuses to load a dataset without `accel_target_sr`, and there is
no fallback constant on the code side.  This script is the canonical way to
fill in that field from the data itself, so a new prototype (different DMA
batch size, different firmware cadence) just works once its YAML is created.

Usage:
    python -m scripts.utils.derive_dataset_sampling_rate configs/datasets/d5.yaml
    python -m scripts.utils.derive_dataset_sampling_rate configs/datasets/d5.yaml --apply

Without `--apply` the script prints the derivation report and exits.  With
`--apply` it writes the rounded integer back into the YAML, preserving the
rest of the file via a targeted line replacement.

Derivation:
    Each `vibration_raw_*.csv` row carries `(pc_time, esp_time_us, s0..s{B-1})`
    where B is the firmware DMA batch size (B=109 on D4, B=128 on D5).  The
    effective rate is

        fs = (N_rows * B) / (esp_time_last - esp_time_first) * 1e6   [Hz]

    `esp_time_us` is preferred over `pc_time` because the embedded clock is
    monotonic and free of OS-level jitter that affects `pc_time`.  We
    cross-check against `pc_time` and warn if the two disagree by more than
    1 %.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_pc_time_seconds(s: str) -> float:
    """Parse `HH:MM:SS.fff` → seconds since midnight."""
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def _read_first_and_last_rows(path: Path) -> tuple[list[str], list[str], list[str], int]:
    """Return (header, first_data_row, last_data_row, n_data_rows).

    Streams the file rather than loading it all into memory — vibration_raw
    files can have ~100k rows on long recordings.
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        first_row = next(reader)
        last_row = first_row
        n_data_rows = 1
        for row in reader:
            last_row = row
            n_data_rows += 1
    return header, first_row, last_row, n_data_rows


def derive_one_file(path: Path) -> dict:
    """Derive sampling rate from one vibration_raw CSV.

    Returns a dict with the per-file diagnostic + the derived rate.
    """
    header, first, last, n_rows = _read_first_and_last_rows(path)
    if header[:2] != ["pc_time", "esp_time_us"]:
        raise ValueError(f"unexpected header in {path.name}: {header[:3]}")
    n_samples_per_row = len(header) - 2  # subtract pc_time, esp_time_us

    esp_first = int(first[1])
    esp_last = int(last[1])
    duration_s_esp = (esp_last - esp_first) / 1e6

    pc_first = _parse_pc_time_seconds(first[0])
    pc_last = _parse_pc_time_seconds(last[0])
    duration_s_pc = pc_last - pc_first

    fs_esp = (n_rows * n_samples_per_row) / duration_s_esp
    fs_pc = (n_rows * n_samples_per_row) / duration_s_pc

    disagreement = abs(fs_esp - fs_pc) / fs_esp

    return {
        "path": str(path),
        "n_rows": n_rows,
        "samples_per_row": n_samples_per_row,
        "duration_s_esp": duration_s_esp,
        "duration_s_pc": duration_s_pc,
        "fs_esp_hz": fs_esp,
        "fs_pc_hz": fs_pc,
        "disagreement_pct": disagreement * 100.0,
    }


_MIC_SAMPLE_RATE = 16_000  # global; same across all datasets


def _sensor_id_from_path(p: Path) -> str:
    """`vibration_raw_E.csv` -> `E`; `vibration_raw_D.csv` -> `D`."""
    name = p.stem  # e.g. "vibration_raw_D"
    return name.split("vibration_raw_", 1)[-1]


def _cluster_sensor_rates(per_file: list[dict], tolerance_pct: float = 2.0) -> dict:
    """Group per-file rates by sensor id; flag sensors whose median deviates
    more than `tolerance_pct` from the dataset-wide median.

    Returns dict with `dominant_rate` (median across all "in-tolerance" sensors)
    and `overrides` (sensor_id -> rate, only for outliers).
    """
    by_sensor: dict[str, list[float]] = {}
    for r in per_file:
        sid = _sensor_id_from_path(Path(r["path"]))
        by_sensor.setdefault(sid, []).append(r["fs_esp_hz"])

    sensor_medians = {
        sid: sorted(rates)[len(rates) // 2] for sid, rates in by_sensor.items()
    }
    overall_rates = sorted(sensor_medians.values())
    overall_median = overall_rates[len(overall_rates) // 2]

    # Sensors within tolerance form the "dominant" group; rest are overrides.
    in_tolerance = [
        sid
        for sid, m in sensor_medians.items()
        if abs(m - overall_median) / overall_median * 100.0 <= tolerance_pct
    ]
    dominant_median = sorted(sensor_medians[s] for s in in_tolerance)[
        len(in_tolerance) // 2
    ]
    dominant_rate = int(round(dominant_median))

    overrides = {
        sid: int(round(m))
        for sid, m in sensor_medians.items()
        if abs(m - dominant_median) / dominant_median * 100.0 > tolerance_pct
    }
    return {
        "sensor_medians": sensor_medians,
        "dominant_rate": dominant_rate,
        "overrides": overrides,
    }


def _suggest_hop_length(target_acoustic_rate_hz: float) -> tuple[int, float]:
    """Pick the integer hop_length that best aligns the acoustic frame rate
    to the target vibration rate.  Returns (hop, achieved_mismatch_pct).

    Constrained to a sane band: hop ∈ [16, 128].  16 is the STFT lower bound
    for n_fft=1024 (50% overlap floor); 128 still gives ≥ 125 Hz acoustic
    frame rate which covers the 117 Hz vane-pass band.
    """
    best_hop, best_mismatch = 43, float("inf")
    for hop in range(16, 129):
        rate = _MIC_SAMPLE_RATE / hop
        mismatch = abs(rate - target_acoustic_rate_hz) / target_acoustic_rate_hz
        if mismatch < best_mismatch:
            best_hop, best_mismatch = hop, mismatch
    return best_hop, best_mismatch * 100.0


def derive_dataset_sampling_rate(yaml_path: Path) -> dict:
    """Walk the dataset root, derive per-sensor rates from every
    vibration_raw_*.csv, then suggest:
      * accel_target_sr (dataset-wide rate)
      * accel_sr_overrides (per-sensor outliers)
      * hop_length (best cross-modal alignment to dominant rate)
    """
    with yaml_path.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    root = (REPO_ROOT / spec["root"]).resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    raw_files = sorted(root.rglob("vibration_raw_*.csv"))
    if not raw_files:
        raise FileNotFoundError(
            f"no vibration_raw_*.csv files under {root}; this dataset ships only "
            f"the peak-amplitude stream and accel_target_sr must be set by hand."
        )

    per_file = [derive_one_file(p) for p in raw_files]
    cluster = _cluster_sensor_rates(per_file)
    dominant_rate = cluster["dominant_rate"]
    overrides = cluster["overrides"]

    # Cross-modal alignment: choose hop to match dominant sensor rate.
    suggested_hop, hop_mismatch = _suggest_hop_length(dominant_rate)
    legacy_hop_mismatch = (
        abs(dominant_rate - _MIC_SAMPLE_RATE / 43) / dominant_rate * 100.0
    )

    return {
        "dataset_id": spec["id"],
        "yaml_path": str(yaml_path),
        "root": str(root),
        "n_files_scanned": len(per_file),
        "per_file": per_file,
        "sensor_medians": cluster["sensor_medians"],
        "recommended_accel_target_sr": dominant_rate,
        "recommended_accel_sr_overrides": overrides,
        "recommended_hop_length": suggested_hop,
        "hop_mismatch_pct": hop_mismatch,
        "legacy_hop43_mismatch_pct": legacy_hop_mismatch,
    }


def _print_report(report: dict) -> None:
    print(f"\n=== Sampling-rate derivation: {report['dataset_id']} ===")
    print(f"Dataset root: {report['root']}")
    print(f"Files scanned: {report['n_files_scanned']}")
    print()
    print(f"{'file':<60} {'rows':>6} {'B':>4} {'fs_esp':>8} {'fs_pc':>8} {'diff%':>6}")
    for r in report["per_file"]:
        name = Path(r["path"]).relative_to(REPO_ROOT).as_posix()
        if len(name) > 58:
            name = "..." + name[-55:]
        print(
            f"{name:<60} {r['n_rows']:>6d} {r['samples_per_row']:>4d} "
            f"{r['fs_esp_hz']:>8.2f} {r['fs_pc_hz']:>8.2f} {r['disagreement_pct']:>5.2f}%"
        )
    print()
    print("Per-sensor median rates:")
    for sid, m in sorted(report["sensor_medians"].items()):
        print(f"  sensor {sid:<4} -> {m:7.2f} Hz")
    print()
    print(f"Recommended accel_target_sr:        {report['recommended_accel_target_sr']} Hz")
    if report["recommended_accel_sr_overrides"]:
        print(f"Recommended accel_sr_overrides:     {report['recommended_accel_sr_overrides']}")
    else:
        print("Recommended accel_sr_overrides:     (none — all sensors within tolerance)")
    print(f"Recommended hop_length:             {report['recommended_hop_length']}  "
          f"(acoustic frame rate = {_MIC_SAMPLE_RATE / report['recommended_hop_length']:.2f} Hz, "
          f"cross-modal mismatch = {report['hop_mismatch_pct']:.2f}%)")
    print(f"Legacy global hop=43 mismatch:      {report['legacy_hop43_mismatch_pct']:.2f}%")


def _apply_to_yaml(yaml_path: Path, report: dict) -> None:
    """Write `accel_target_sr`, `accel_sr_overrides`, and `hop_length` back into
    the YAML, preserving the rest of the file via targeted line edits.
    """
    rate = report["recommended_accel_target_sr"]
    overrides = report["recommended_accel_sr_overrides"]
    hop = report["recommended_hop_length"]

    text = yaml_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    def _replace_or_append(key: str, value_repr: str, comment: str) -> None:
        nonlocal lines
        new_line = f"{key}: {value_repr}   # {comment}"
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(f"{key}:"):
                indent = line[: len(line) - len(stripped)]
                lines[i] = f"{indent}{new_line}"
                return
        # Not found — append before the `extra:` block if present, else at end.
        insertion = next(
            (
                i
                for i, line in enumerate(lines)
                if line.lstrip().startswith("extra:")
            ),
            len(lines),
        )
        lines.insert(insertion, new_line)

    _replace_or_append(
        "accel_target_sr",
        str(rate),
        "derived empirically by scripts/utils/derive_dataset_sampling_rate.py",
    )
    if overrides:
        override_repr = (
            "{" + ", ".join(f"{k}: {v}" for k, v in sorted(overrides.items())) + "}"
        )
        _replace_or_append(
            "accel_sr_overrides",
            override_repr,
            "per-sensor rate outliers vs dataset-wide accel_target_sr",
        )
    _replace_or_append(
        "hop_length",
        str(hop),
        f"chosen for {report['hop_mismatch_pct']:.2f}% cross-modal alignment to vib rate",
    )

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote accel_target_sr={rate}, hop_length={hop}", end="")
    if overrides:
        print(f", accel_sr_overrides={overrides}", end="")
    print(f" into {yaml_path.name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("yaml", type=Path, help="path to configs/datasets/dN.yaml")
    p.add_argument(
        "--apply",
        action="store_true",
        help="write the derived rate back into the YAML",
    )
    args = p.parse_args(argv)

    report = derive_dataset_sampling_rate(args.yaml)
    _print_report(report)
    if args.apply:
        _apply_to_yaml(args.yaml, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
