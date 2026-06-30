# `scripts/` — runnable experiment drivers

Everything here is a **command-line entry point**, not importable library code
(the library lives under [`src/`](../src)). Scripts are grouped by role:

```
scripts/
├── campaigns/          multi-stage end-to-end drivers (produce the results)
├── paradigms/          RQ2 / RQ3 paradigm-comparison trainers
├── sweeps/             single-stage hyperparameter sweeps
├── diagnostics/        one-off investigations (run, never imported)
└── utils/              small reusable tools
```

Run from the repository root, module form preferred:

```bash
python -m scripts.campaigns.run_thesis_campaign
python scripts/utils/derive_dataset_sampling_rate.py configs/datasets/d5.yaml
```

All of them import shared per-stage configuration (`_v1_cfg`, `_v2_cfg`, …) from
[`src/modeling/orchestration/stage_configs.py`](../src/modeling/orchestration/stage_configs.py),
re-exported through `full_run`, so that scripts and the canonical pipeline never drift.

> **Reproducing the headline results does not require any script here.**
> The canonical run is `python -m src.modeling.orchestration.full_run`
> (smoke: `--quick`); `python -m src.modeling.orchestration.multi_seed`
> produces the mean ± std numbers in the thesis tables. The scripts below are
> the supporting investigations, sweeps, and diagnostics around that run.

---

## `campaigns/` — multi-stage drivers

Sequence many training runs into one end-to-end experiment.

| Script | Purpose |
|---|---|
| `run_thesis_campaign.py` | Master campaign: acoustic-improvement sweep → full pipeline → multi-seed verdict → report. One command, produces every result. |
| `run_ablation_campaign.py` | End-to-end ablation campaign (57 cells): baseline → per-phase sweeps → top-K promotion → multi-seed → conditional follow-ups. |
| `run_deep_v3v4_campaign.py` | V3-first deep campaign: deep V3 sweep → V3-gated deep V4 sweep → multi-seed verdict on the winners. |
| `ablation_full_pipeline.py` | Single-cell runner used by the campaigns: one parameter combination through the full pipeline. |
| `run_v1_v2_only.py` | Retrain only V1+V2 (+A1+modality probe) under one named intervention (~50 min CPU vs ~6 h full). |
| `analyze_ablation.py` | Aggregates campaign cell outputs (`results/runs/*__ablation_*/`) into a Markdown report. |

## `paradigms/` — RQ2 / RQ3 paradigm comparisons

| Script | Purpose |
|---|---|
| `run_v3_three_paradigms.py` | Trains V3 acoustic / vibration / fusion CNF heads from saved V1+V2 weights; emits comparable per-pipeline metrics. |
| `run_v4_three_paradigms.py` | Trains the V4 localization paradigms (SRP-PHAT, accel-multilateration, learned heads) for the RQ3 comparison. |

## `sweeps/` — hyperparameter sweeps

| Script | Purpose |
|---|---|
| `v3_deep_sweep.py` | Deep V3 anomaly sweep against a frozen V1/V2 encoder; selects by gap-guarded real-anomaly F1. |
| `v4_deep_sweep.py` | Deep V4 localization sweep against a frozen V2 encoder; evaluates on held-out positions, V3-gated. |
| `v4_aug_sweep.py` | V4-only augmentation sweep (target-position noise × SRP-volume noise) reusing one cached set of V4 samples. |

## `diagnostics/` — one-off investigations

Run, never imported. These live outside `src/` so they stay out of the importable
library.

| Script | Purpose |
|---|---|
| `probe_v2.py` | Probes V2's internal representations to locate where the acoustic↔vibration fusion loses cluster purity. |
| `reeval_k3.py` | Re-evaluates trained V2 encoders under the corrected K=3 (Pump/Standstill/Turbine) held-out setup, no retraining. |
| `train_v2_cma.py` | Trains V2 with the cross-modal-alignment loss and compares RQ1 purity against vanilla V2. |
| `v2_sweep.py` | V2 architectural sweep: CMA-weight grid × context-aggregation variants. |
| `v3_diagnostic.py` | Dumps per-cohort V3 anomaly-score distributions for the Chapter 6 histograms. |

## `utils/` — small tools

| Script | Purpose |
|---|---|
| `derive_dataset_sampling_rate.py` | Canonical way to fill a new dataset's `accel_target_sr` from raw CSV timestamps. `--apply` writes it back into the YAML. |
| `visualize_sensor_knock_positions.py` | Plots sensor + knock positions for a dataset (figure helper). |
