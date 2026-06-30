"""Results archival for the thesis run history.

Each `python -m src.modeling.orchestration.full_run` invocation produces
a `metrics.json` in `results/full_run/` (overwritten every time).  This
module **archives** that file (and key checkpoints) into a timestamped
sub-directory under `results/runs/`, so every run accumulates as a
permanent record for the master's-thesis Chapter 6 tables.

Each archive contains:
  - `metrics.json` — the run's full metric dump
  - `manifest.json` — timestamp, git commit, host, config snapshot
  - `v1_acoustic.pt`, `v1_vibration.pt`, `v2_encoder.pt`, `v3_flow.pt`,
    `v3_thresholds.npz`, `v4_head.pt`, `v5_1_head_speed.pt` — the
    encoder/flow/head state_dicts so the run can be re-evaluated later.
  - `console.log` — copied from full_run if present

Aggregate `results/runs/index.json` indexes every archive with its key
metrics for quick scanning.
"""

from __future__ import annotations

import datetime
import json
import shutil
import socket
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path

ARCHIVE_ROOT = Path(__file__).resolve().parents[3] / "results" / "runs"


def _git_commit() -> str:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return rev.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return bool(out.decode().strip())
    except Exception:
        return False


def archive_run(
    full_run_dir: Path,
    run_label: str | None = None,
    extra_manifest: dict | None = None,
) -> Path:
    """Copy `results/full_run/` into a timestamped archive subdirectory.

    Args:
        full_run_dir: source directory (typically `results/full_run/`)
        run_label: optional human-readable suffix (e.g. "seed42",
            "modality_dropout_off"); defaults to the timestamp alone.
        extra_manifest: extra fields to drop into `manifest.json`.

    Returns the absolute path of the new archive directory.
    """
    full_run_dir = Path(full_run_dir)
    if not full_run_dir.exists():
        raise FileNotFoundError(full_run_dir)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{ts}__{run_label}" if run_label else ts
    dst = ARCHIVE_ROOT / label
    dst.mkdir(parents=True, exist_ok=False)

    # Copy metrics + log if present.
    for fname in (
        "metrics.json",
        "run_log.txt",
        "full_console.log",
        "v2_reeval_k3.json",
        "v3_cohort_diagnostic.json",
        "v2_probe.json",
        "INVESTIGATION.md",
        "INVESTIGATION_CORRECTED.md",
        "SUMMARY.md",
    ):
        src = full_run_dir / fname
        if src.exists():
            shutil.copy2(src, dst / fname)

    # Copy the small checkpoint files (skip the large probe duplicates).
    for sub in ("v1", "v2", "v3", "v4", "v5_1"):
        src_sub = full_run_dir / sub
        if src_sub.is_dir():
            for f in src_sub.iterdir():
                if f.is_file():
                    (dst / sub).mkdir(exist_ok=True)
                    shutil.copy2(f, dst / sub / f.name)

    manifest = {
        "timestamp": ts,
        "label": label,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "host": socket.gethostname(),
        "source": str(full_run_dir),
    }
    if extra_manifest:
        # Coerce dataclass -> dict so manifest is JSON-serialisable.
        cleaned: dict = {}
        for k, v in extra_manifest.items():
            if is_dataclass(v):
                cleaned[k] = asdict(v)
            else:
                cleaned[k] = v
        manifest["extra"] = cleaned
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))

    _update_index(dst)
    return dst


def _update_index(archive_dir: Path) -> None:
    """Append a row to `results/runs/index.json` summarising the new archive."""
    index_path = ARCHIVE_ROOT / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {"runs": []}

    metrics_path = archive_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    manifest = json.loads((archive_dir / "manifest.json").read_text())
    stages = metrics.get("stages", {})

    summary = {
        "label": manifest["label"],
        "timestamp": manifest["timestamp"],
        "commit": manifest["git_commit"],
        "dirty": manifest["git_dirty"],
        "headline": _extract_headline(stages),
    }
    index["runs"].append(summary)
    # Keep newest first.
    index["runs"].sort(key=lambda r: r["timestamp"], reverse=True)
    index_path.write_text(json.dumps(index, indent=2))


def _extract_headline(stages: dict) -> dict:
    """Pluck the headline numbers from `metrics["stages"]` for the index."""
    headline: dict = {}
    if "v1_acoustic" in stages:
        headline["v1_acoustic_purity"] = stages["v1_acoustic"].get("sanity_purity")
    if "v1_vibration" in stages:
        headline["v1_vibration_purity"] = stages["v1_vibration"].get("sanity_purity")
    if "v2" in stages:
        headline["v2_purity"] = stages["v2"].get("rq1_purity")
        headline["v2_nmi"] = stages["v2"].get("rq1_nmi")
    if "v2_a1_drop_vibration" in stages:
        headline["v2_a1_purity"] = stages["v2_a1_drop_vibration"].get("rq1_purity")
    # V3/V4 stage names: the orchestrator reports the per-paradigm stages
    # `v3_three_paradigms` / `v4_four_paradigms`; the fusion paradigm is the
    # headline.  (Older runs used flat `v3`/`v4`; support both.)
    v3 = stages.get("v3_three_paradigms", stages.get("v3"))
    if isinstance(v3, dict):
        fus = v3.get("fusion", v3)
        headline["v3_val_nll"] = fus.get("val_nll_final")
    if "v3_a2_unconditional" in stages:
        headline["v3_a2_val_nll"] = stages["v3_a2_unconditional"].get("val_nll_final")
    if "v3_rq2_transition_fpr" in stages:
        headline["v3_transition_fpr"] = stages["v3_rq2_transition_fpr"]
    if "v3_cohort_validation" in stages:
        headline["v3_cohort_alerts"] = {
            k: v.get("alert_rate") for k, v in stages["v3_cohort_validation"].items()
        }
    v4 = stages.get("v4_four_paradigms", stages.get("v4"))
    if isinstance(v4, dict):
        fus = v4.get("fusion", v4)
        headline["v4_mae_3d"] = fus.get("val_mae_3d")
        headline["v4_p95_3d"] = fus.get("val_p95_3d")
    if "v4_a3_unconditional" in stages:
        headline["v4_a3_mae_3d"] = stages["v4_a3_unconditional"].get("val_mae_3d")
    if "v5_1" in stages:
        headline["v5_1_mae_3d"] = stages["v5_1"].get("val_mae_3d")
    return headline


def list_runs() -> list[dict]:
    """Return the run-history index (most-recent first)."""
    index_path = ARCHIVE_ROOT / "index.json"
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text()).get("runs", [])


__all__ = ["ARCHIVE_ROOT", "archive_run", "list_runs"]
