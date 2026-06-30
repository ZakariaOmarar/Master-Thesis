# Reproducing the thesis results

This document is the traceability map an examiner needs: for every headline
table in the Results chapter, it names the **command**, the **seeds**, and the
**artifact file** the number is read from.

## Prerequisites

1. Environment: `pip install -r requirements-lock.txt` (frozen, Python 3.11.9)
   then `pip install -e ".[dev,dl]"`.
2. Recordings under `data/` (D1–D5). These are **not** redistributed with the
   repository (see README → Datasets). Without them only the data-free test
   suite runs; no headline number can be regenerated.
3. All result artifacts land under `results/` (git-ignored). Each run records
   its git commit, dirty flag, host, and full configs in `manifest.json`.

## Canonical seeds

The thesis reports every stochastic quantity as a distribution over **five
seeds**:

```
THESIS_SEEDS = (42, 1337, 2024, 7, 99)
```

defined once in [`src/modeling/orchestration/multi_seed.py`](src/modeling/orchestration/multi_seed.py)
and used as the documented default of the multi-seed driver. The deep
localization campaign realizes the same set as base seed `42` (Phase 1/2 grid)
plus `VERDICT_SEEDS = (1337, 2024, 7, 99)`
([`scripts/campaigns/run_deep_v3v4_campaign.py`](scripts/campaigns/run_deep_v3v4_campaign.py)).

## Table → command → seeds → artifact

| Thesis result | Command | Seeds | Artifact |
|---|---|---|---|
| RQ1 mode-discovery NMI (Table `res_rq1`) | `python -m src.modeling.orchestration.multi_seed` | THESIS_SEEDS | `results/runs/multi_seed_summary.json` (`v2_nmi`); per-seed `results/runs/<ts>__full_pipeline_b5_cma/metrics.json` → `stages.v2.rq1_nmi` |
| RQ2 per-cohort alert rates / specificity (Tables `res_rq2_alert`, `res_rq2_spec`) | `python -m src.modeling.orchestration.multi_seed` | THESIS_SEEDS | per-seed `…/rq2_paradigm_comparison.json` (written by Stage 8) |
| RQ2 conditioning ΔNLL + latent-SNR AUC (Tables in §res_rq2_cond) | same run | THESIS_SEEDS | per-seed `metrics.json` → `stages.v3_fusion_depth.v3_vs_a2_paired_test` and `…synthetic_anomaly_auc` |
| RQ2 domain-shift FPR (Table `res_rq2_shift`) | `python -m src.modeling.orchestration.multi_seed` + `python -m scripts.baselines.v0_domain_shift_multiseed` | THESIS_SEEDS | `metrics.json` per-cohort breakdown + baseline script output under `results/v0_anomaly/` |
| RQ2 reference baselines (Tables `res_v0`, `res_v0_anomaly`) | `python -m src.modeling.orchestration.full_run --v0-baselines` (seed 42) | 42 | `metrics.json` → `stages.v0` |
| RQ3 LORO paradigm comparison (Table `res_rq3_loro`) | full_run Stage 9 per seed | THESIS_SEEDS | per-seed `…/rq3_paradigm_comparison.json` |
| RQ3 LOPO by position (Table `res_rq3_lopo`) | `python -m src.modeling.orchestration.v4_lopo_cv --encoder-run <run> --v3-run <run> --all-channel-modes --seed <s>` (once per seed) | THESIS_SEEDS | `results/lopo/summary.json` (`aggregate_per_mode`); per-fold `results/lopo/folds.jsonl` |
| RQ3 cross-session → D5 (Table `res_rq3_cross`) | `python -m src.modeling.orchestration.v4_cross_dataset --encoder-run <run> --v3-run <run> --all-channel-modes --seed <s>` (once per seed) | THESIS_SEEDS | `results/cross_dataset/summary.json` |
| RQ4a prototype stand-in (null) | full_run Stage 7 | THESIS_SEEDS | `metrics.json` → `stages.v5_1` |
| RQ4b real SCADA MI ranking (Tables `res_rq4_mode`, `res_rq4_within`) | `python -m scripts.scada.v5_2_channel_mining` | n/a (deterministic + permutation null) | `results/illwerke/scada/…` |
| Appendix B impulse-aware flow (recall/ROC-AUC) | `python -m scripts.train_deep_impulse_flow` → `python -m scripts.aggregate_deep_impulse` | THESIS_SEEDS | `results/` deep-impulse aggregate JSON |

`<run>` is a `results/runs/<ts>__full_pipeline_b5_cma/` directory produced by a
seed's `full_run`; the LOPO and cross-session drivers reuse that run's saved V2
encoder (`v2/encoder.pt`) and fusion V3 so they do not retrain the upstream
stages. The deep campaign
[`run_deep_v3v4_campaign.py`](scripts/campaigns/run_deep_v3v4_campaign.py) chains
Phases 1–5 end-to-end across the five seeds in one invocation.

## Statistical reporting

Paired significance tests (V3 vs A2 NLL, V4 vs A3 MAE) and MAE confidence
intervals resample at the **recording level** (block bootstrap), not the window
level, because per-window observations within a recording are correlated; see
`groups=` in
[`src/modeling/eval/statistics.py`](src/modeling/eval/statistics.py). The
recorded `method` / `n_recordings` fields in each `*_paired_test` block of
`metrics.json` state the resampling unit used.

## Notes

* `--quick` halves epoch counts at every stage for a ~25 min CPU smoke run; it
  is **not** the reported configuration.
* Determinism is best-effort (Python/NumPy/Torch RNGs pinned; BLAS scheduling
  variance bounded, not eliminated). The five-seed distribution — not any single
  run — is the unit of reporting.
