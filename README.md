# CMSC_Net_Hydro

Multimodal acoustic + vibration anomaly detection and source localization for
reversible Francis pump-turbines. This is the implementation accompanying a
Master's thesis; it realizes the chained **V0 → V5** label-free pipeline.

Experiments run on a **3D-printed circular scale rig** modelled after the
Rodundwerk II (ROW II) machine — the reference sensor geometry in
[`src/config/constants.py`](src/config/constants.py) is ROW II's, and the same
ingestion contract accepts the full-scale recordings when they are delivered
(see the `illwerke_raw` stub config).

## Thesis claim

> A single label-free model in which an unsupervised operational-context
> vector, learned by self-supervised pretraining on fused acoustic-vibration
> streams, simultaneously conditions an anomaly detection head and an
> anomaly-gated source-localization head.

The chained system (mode → anomaly → gated-localization) is the contribution.
Chapter 6 reports four severing ablations against the chained system, not a
bake-off of published architectures.

## Architecture

```
   WAV / vibration  →  per-modality CNN encoders + Set-Transformer pool
                           (sensor-pos + modality + dataset embeddings)
                                       │
                                       ▼
                       Bidirectional cross-attention (1 block)
                                       │
                                       ▼
                  fused tokens z_t  →  c_t = PMA(z_t)        ── continuous mode label (cluster/Hungarian)
                                       │             │
                                       │      FiLM(c) ┐
                                       ▼              │
                       Conditional Normalizing Flow ◀─┘   ── continuous anomaly score s_t
                                       │
                                       ▼  (gate: s_t > per-cluster 99 % threshold)
                                       │
                       Cross3D 3-D CNN on SRP-PHAT + accel-TDOA, FiLM(c [+ s])
                                       ▼
                                    (x, y, z)            ── only on alert windows
```

| Iter | What it delivers |
|---|---|
| **V0**  | Reference baselines: LSTM-AE on log-mel (RQ2), LightGBM on hand-engineered features (RQ1 upper bound), classical SRP-PHAT (RQ3). |
| **V1**  | Per-modality SSL warmup (contrastive) + cluster-purity sanity gate. Label-free; V2 inherits its weights. |
| **V2**  | Bidirectional cross-attention fusion + multimodal SSL (contrastive + Latent Masked Modeling); inherits V1. |
| **V3**  | Conditional Normalizing Flow anomaly head + per-cluster percentile thresholds + synthetic-transition stress test + A2 ablation. |
| **V4**  | Anomaly-gated Cross3D localization head + accel-TDOA + FiLM conditioning + A3 ablation. |
| **V5**  | RQ4a noise-robustness conditioning + RQ4b Illwerke SCADA channel-mining analysis. |
| **streaming** | Gated runtime emitting `(mode, anomaly_score, alert_flag, (x,y,z) | None)` per window. |

Every stage is implemented and covered by the smoke tests. The end-to-end run
is driven by [`src/modeling/orchestration/full_run.py`](src/modeling/orchestration/full_run.py).

## Datasets

All datasets were captured on the 3D-printed prototype rig. Each is registered
by a single YAML under [`configs/datasets/`](configs/datasets) and loaded
through the `DatasetRegistry`.

| ID | Folder | Sensors | Labels | Spatial GT | Role |
|----|--------|---------|--------|:---:|------|
| `d1` | `data/first_test_dataset`  | 4 mics + 4 vib (peak) | Pump / Standstill / Turbine / RandomFault | — | RQ1, RQ2 |
| `d2` | `data/second_test_dataset` | 5 mics + 5 vib (peak) | …same + `pos_(x,y,z)_*` | ✔ (5 pos) | RQ1, RQ2, RQ3 |
| `d3` | `data/third_test_dataset`  | 9 mics + 4 accel (peak) | `speed{1,2,3}` + `hit_between_Fl_Gr_speed1` | ✔ (1 hit) | RQ2, RQ3, RQ4a |
| `d4` | `data/fourth_test_dataset` | 9 mics + 4 accel (**raw ≈376 Hz**) | `speed{1,2,3}` + `RandomFault_knock_*` `(x,y,z)` | ✔ (7 pos) | RQ1–RQ4a |
| `d5` | `data/fifth_test_dataset`  | 9 mics + 4 accel (raw ≈446 Hz) | `healthy/` + `knock/(x,y,z)` | ✔ (6 pos) | V3, V4, inference |
| `illwerke_raw` | `data/illwerke_raw` *(future)* | 9 mics + 4 accel | TBD | TBD | drop-in via config when delivered |

> **`speed{1,2,3}` are healthy recordings with three levels of added acoustic
> noise — an augmentation/domain-shift knob, not an operating mode.** They are
> pooled as healthy across datasets and must never be used as a mode label.

Adding a dataset is a single YAML edit; see the schema and inline comments in
[`configs/datasets/d1.yaml`](configs/datasets/d1.yaml). For a sensor that needs
its vibration sampling rate derived from raw timestamps, use
[`scripts/utils/derive_dataset_sampling_rate.py`](scripts/utils/derive_dataset_sampling_rate.py).

The recordings themselves are not redistributed with this repository.

## Configuration — where settings actually live

Two layers, by design:

1. **Per-dataset facts** (sensor counts, sampling rates, position source, label
   scheme, window scales) live in [`configs/datasets/*.yaml`](configs/datasets)
   and are the canonical source of truth, loaded via `DatasetSpec.from_yaml`
   and `src/config/dataset_registry.py`.
2. **Per-stage model hyperparameters** (V1–V5 trainer settings) live in the
   Python builders `_v1_cfg`, `_v2_cfg`, `_v3_cfg`, `_v4_cfg` in
   [`stage_configs.py`](src/modeling/orchestration/stage_configs.py) (re-exported
   from [`full_run.py`](src/modeling/orchestration/full_run.py)). These are the
   single source shared by the orchestrator and every driver under `scripts/`;
   change them there, not at the caller.

Architecture-wide constants (the empirically-selected acoustic features
`n_fft=4096, hop=2048, n_mels=96`; sync windows) live in
[`src/config/architecture.py`](src/config/architecture.py).

## Layout

```
src/
├── config/         physical constants (ROW II geometry), architecture config, dataset registry, device
├── data/           DataSegment — the immutable universal data contract
├── exceptions.py   project-wide exception hierarchy
├── ingestion/      generic WAV+CSV reader, per-dataset loader, positions, sync verification/correction
├── features/       log-mel + CWT (acoustic) and amplitude/envelope/kurtosis (vibration) encoder inputs
└── modeling/
    ├── encoders/        per-modality CNNs, Set-Transformer (MAB/PMA), pooling
    ├── fusion/          V2 bidirectional cross-attention block
    ├── context/         V1/V2 SSL trainers, V2 fusion encoder, cluster-purity metric, modality probe
    ├── anomaly/         V3 RealNVP CNF + FiLM, per-cluster thresholds, event detection, synthetic eval
    ├── localization/    GCC-/SRP-PHAT, multilateration, V4 features + Cross3D head + trainer + temporal
    ├── scada/           V5.1 speed conditioning, V5.2 Illwerke MI channel mining
    ├── streaming/       gated V2→V3→V4 runtime + cost/quality study
    ├── eval/            RQ2/RQ3 paradigm evals, fusion forensics, bootstrap statistics
    ├── anomaly_baselines/  V0 LSTM-AE, LightGBM mode classifier, SRP-PHAT baseline
    └── orchestration/   full_run (end-to-end), multi_seed, archive, V4 cross-validation drivers

configs/datasets/   per-dataset registration YAMLs (d1–d5 + illwerke_raw stub) — the only configs code loads
scripts/            runnable experiment drivers (campaigns, sweeps, diagnostics) — see scripts/README.md
tests/              pytest suite (tests/unit/ = focused unit tests; tests/ = smoke tests; 219 run without data, 109 marked requires_data)
docs/, results/     thesis text and run artifacts — NOT version-controlled (see .gitignore)
```

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; or `source .venv/bin/activate`
pip install -r requirements-lock.txt                 # exact pinned reproduction environment
pip install -e ".[dev,dl]"                           # editable install + dev/DL extras
```

`requirements-lock.txt` is the frozen environment that produced the reported
results (Python 3.11.9, numpy 2.x); `pyproject.toml` carries the looser
supported ranges for day-to-day development.

## Running the pipeline

The recordings under `data/` are required. The whole V0–V5 chain runs from one
entry point:

```bash
# End-to-end run; add --quick for a ~25 min CPU smoke run:
python -m src.modeling.orchestration.full_run

# Repeat across the five canonical thesis seeds (42, 1337, 2024, 7, 99) for the
# median [min, max] numbers in the thesis tables — these are the defaults:
python -m src.modeling.orchestration.multi_seed
```

Each run writes a timestamped directory under `results/runs/`. The sweeps,
ablations, and diagnostics in [`scripts/`](scripts/README.md) are supporting
investigations around this run, not prerequisites for it.

The canonical seed set lives in `multi_seed.THESIS_SEEDS`. The exact
command → seed → artifact mapping for every reported table is documented in
[`REPRODUCING.md`](REPRODUCING.md).

## Tests

```bash
# Synthetic-only — needs no recordings, runs anywhere:
python -m pytest -m "not requires_data" -q

# Full suite — data-dependent tests auto-skip when data/ is absent:
python -m pytest -q
```

Tests that exercise the real recordings are marked `@pytest.mark.requires_data`
and skip cleanly on a checkout without a `data/` directory (see
[`tests/conftest.py`](tests/conftest.py)).

## Linting & type-checking

```bash
make check       # ruff + pyright + the data-free test suite (what CI runs)
ruff check       # lint only
pyright          # type-check src/ only
```

`ruff` (config in `[tool.ruff]`) is deliberately scoped: `pep8-naming` is left
off because the modelling code uses maths/ML conventions (`X`, `W`,
`B, T, C = x.shape`), and a few rules are ignored with inline rationale.

The package ships `py.typed` and is type-checked with `pyright` in *basic* mode
(config in `[tool.pyright]`); a handful of numpy/torch-driven categories are
relaxed with documented reasons. The same lint, type-check, and data-free test
gate runs in [CI](.github/workflows/ci.yml) on every push and pull request;
`make check` runs the identical gate locally before pushing.

## Notes on the previous repo state

A prior iteration implemented an Illwerke-specific 5-layer physics pipeline and
a Plotly.js dashboard. It was removed when the thesis architecture pivoted to
the V0–V5 chained label-free system above; the removal is recorded in the git
history (the commit titled "Remove the old Illwerke pipeline and dashboard").
The frozen Illwerke pipeline outputs under `results/illwerke/` still feed the
V5.2 SCADA-mining analysis; like all of `results/`, they are kept locally and
are not tracked in git.
