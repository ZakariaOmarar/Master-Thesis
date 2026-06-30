"""Reusable orchestration for V0–V5 pipeline runs.

This package holds the pipeline machinery that other code imports and that
defines the canonical run:

  * ``full_run``      — the end-to-end thesis orchestrator (Stages 0–9) and the
                        single source of truth for the per-stage configs
                        (``v1_config``, ``v2_config``, … reused by ``scripts/``).
  * ``multi_seed``    — multi-seed driver that aggregates ``full_run`` into the
                        mean ± std numbers reported in the thesis tables.
  * ``archive``       — archives each run's ``metrics.json`` into the timestamped
                        ``results/runs/`` history.
  * ``v4_loocv`` / ``v4_lopo_cv`` / ``v4_cross_dataset`` — V4 cross-validation
                        and transfer estimators.

One-off investigation/sweep drivers (probe_v2, reeval_k3, train_v2_cma,
v2_sweep, v3_diagnostic) live under ``scripts/`` — they are run, not imported.
"""
