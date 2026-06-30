"""Unit tests for the head-vs-baseline comparison assembler.

Pure-synthetic: feeds a minimal stages dict mirroring the real run schema and
checks each RQ extractor pulls the right head/baseline numbers and surfaces the
pipeline's existing paired-bootstrap significance.  No recordings or runs needed.
"""

from __future__ import annotations

from scripts.baselines.assemble_comparison import (
    _get,
    build_rq1,
    build_rq2,
    build_rq3,
)

STAGES = {
    "v1_acoustic": {"sanity_nmi": 0.70, "sanity_ari": 0.62, "sanity_purity": 0.76},
    "v2": {"rq1_nmi": 0.995, "rq1_ari": 0.999, "rq1_purity": 0.999},
    "v2_a1_drop_vibration": {"rq1_nmi": 0.979},
    "v3_fusion_depth": {
        "v3_vs_a2_paired_test": {
            "delta_point": 0.0105, "delta_ci95_low": 0.0100, "delta_ci95_high": 0.0110,
            "p_value_two_sided": 0.0, "n_paired": 527,
            "method": "paired_percentile_bootstrap_1000",
        },
        "synthetic_anomaly_auc": {
            "auc_conditional": {"5.0": 0.55}, "auc_unconditional": {"5.0": 0.33},
        },
        "per_cluster_breakdown_healthy": {"alert_rate_total": 0.074},
    },
    "v4_four_paradigms": {
        "fusion": {"val_mae_3d": 0.226, "val_mae_ci95_low": 0.225, "val_mae_ci95_high": 0.227},
        "acoustic": {"val_mae_3d": 0.226},
        "vibration": {"val_mae_3d": 0.542},
    },
    "rq3_paradigm_comparison": {
        "significance": {
            "fusion_vs_acoustic_mae_delta_m": -0.0005, "fusion_vs_acoustic_p": 0.014,
            "fusion_vs_vibration_mae_delta_m": -0.317, "fusion_vs_vibration_p": 0.0,
        },
    },
    "v0_multilateration": {"d2": {"mean_error_m": 0.491, "n_successful": 5}},
}


def test_get_descends_nested() -> None:
    # Regression: the validation loop must descend, not check every key at top level.
    assert _get(STAGES, "v2", "rq1_nmi") == 0.995
    assert _get(STAGES, "v2", "missing", default="x") == "x"
    assert _get(STAGES, "nope", default=7) == 7


def test_rq1_brackets_floor_encoder_ceiling() -> None:
    rq1 = build_rq1(STAGES)
    by_method = {r["method"]: r for r in rq1["rows"]}
    assert by_method["V2 fusion (cross-attention)"]["nmi"] == 0.995
    assert by_method["V1 acoustic (per-modality SSL)"]["nmi"] == 0.70
    assert by_method["V2 minus vibration (fusion ablation)"]["nmi"] == 0.979
    # Floor + ceiling absent in this synthetic run → flagged for a re-run, not silently blank.
    assert "RE-RUN" in by_method["K-means / handcrafted (unsup. floor)"]["status"]
    assert "RE-RUN" in by_method["LightGBM supervised (ceiling, macro-F1)"]["status"]


def test_rq2_surfaces_conditioning_paired_test() -> None:
    rq2 = build_rq2(STAGES, v0_anom=None)
    head = next(r for r in rq2["rows"] if r["role"] == "proposed (headline)")
    assert head["healthy_alert_rate"] == 0.074
    assert head["syn_auc@+5dB"] == 0.55
    sig = rq2["significance"]["conditioning (V3 vs unconditional)"]
    assert sig["p_value"] == 0.0
    assert sig["n_paired"] == 527
    assert sig["delta_nll"] == 0.0105


def test_rq2_lists_standalone_v0_unpaired() -> None:
    v0 = {"results": [
        {"modality": "acoustic", "model": "ocsvm", "roc_auc": 0.64,
         "fpr_in_distribution": 0.05, "fpr_domain_shift": 0.95},
        {"modality": "vibration", "model": "kde", "roc_auc": 0.87},  # filtered out (not acoustic)
    ]}
    rq2 = build_rq2(STAGES, v0_anom=v0)
    v0_rows = [r for r in rq2["rows"] if "prior work" in r["role"]]
    assert len(v0_rows) == 1 and v0_rows[0]["roc_auc"] == 0.64
    assert "unpaired" in v0_rows[0]["status"]


def test_rq3_head_beats_classical_with_significance() -> None:
    rq3 = build_rq3(STAGES)
    by = {r["method"]: r for r in rq3["rows"]}
    assert by["V4 fusion (proposed)"]["holdout_mae_m"] == 0.226
    assert by["V0 accel multilateration (classical vibration)"]["holdout_mae_m"] == 0.491
    assert "RE-RUN" in by["SRP-PHAT (classical acoustic)"]["status"]
    # Honest finding: fusion barely beats acoustic, strongly beats vibration.
    assert rq3["significance"]["fusion_vs_acoustic"]["delta_m"] == -0.0005
    assert rq3["significance"]["fusion_vs_vibration"]["p_value"] == 0.0
    assert "UNDEFINED" in rq3["note"]
