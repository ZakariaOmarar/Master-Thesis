"""Dynamic dataset registry — single source of truth for per-dataset metadata.

Scans `configs/datasets/*.yaml` at import time and exposes a uniform lookup
keyed by dataset id (`d1`, `d2`, ..., `d5`, `illwerke_raw`).  Replaces the
hardcoded per-dataset dicts that previously lived in
`src/config/architecture.py` and the `_DATASET_INDEX` constants in
`v1_ssl.py` / `v2_ssl.py`.

Design contract:
    * The YAML is the single source of truth.  Code never carries per-dataset
      fallback constants (no `{"d1": 4, "d2": 4, ...}` dicts).
    * Adding a future dataset is a YAML edit.  The registry rescan on import
      picks it up without touching any other module.
    * Indices are alphabetical-sorted by canonical id → deterministic across
      machines, stable when a dataset is appended in alphabetical order.
      (V1/V2 embedding tables resize accordingly; pre-existing checkpoints
      must be retrained — acceptable per the per-dataset-hop migration
      already in flight.)

Optional fields the registry tolerates as `None`:
    * `position_path` — required when `position_source != "default"`
    * `accel_sr_overrides` — per-sensor SR outliers (empty dict if absent)
    * `aliases` — secondary names that resolve to the same metadata
    * `hop_length` — removed 2026-05-21: empirical sweep (chapter 3 §3.4.2)
      showed hop_length contributes <0.005 ROC AUC and a single global
      `ACOUSTIC_FEATURES.hop_length=2048` is now used for all datasets.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = REPO_ROOT / "configs" / "datasets"


# Position source enum — every value must have a parser in src/ingestion/positions.py.
_VALID_POSITION_SOURCES = frozenset(
    {
        "default",  # D1-style synthesized geometry
        "d2_node_position_txt",  # D2 / future rectangular rigs sharing the .txt layout
        "d3_position_json",  # D3, D4, D5 / future circular rigs sharing position.json
        "rowii",  # Illwerke ROW II from src/config/constants.SENSOR_LAYOUT
    }
)


@dataclass(frozen=True)
class DatasetMetadata:
    """Resolved per-dataset metadata.  Built from one configs/datasets/*.yaml."""

    id: str
    index: int                       # alphabetical-sorted index for embedding lookups
    root: Path                       # absolute path (REPO_ROOT-prefixed)
    n_mics: int
    n_vibrations: int
    accel_target_sr: int             # REQUIRED; no fallback constant
    vibration_format: Literal["auto", "peak", "raw"]
    position_source: str             # element of _VALID_POSITION_SOURCES
    position_path: Path | None       # absolute path; None iff position_source == "default" / "rowii"
    label_scheme: str
    window_scales_seconds: tuple[float, ...]
    v3_window_seconds: float
    v4_window_seconds: float
    accel_sr_overrides: dict[str, int] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    description: str | None = None
    extra: dict = field(default_factory=dict)


class DatasetRegistry:
    """Scans `configs/datasets/*.yaml` and exposes lookup by id (incl. aliases)."""

    def __init__(self, configs_dir: Path | None = None):
        self._configs_dir = Path(configs_dir) if configs_dir is not None else _CONFIGS_DIR
        self._by_id: dict[str, DatasetMetadata] = {}
        self._by_index: list[DatasetMetadata] = []
        self._alias_to_canonical: dict[str, str] = {}
        self._load()

    # ---------------------------------------------------------------- loading

    def _load(self) -> None:
        if not self._configs_dir.exists():
            raise FileNotFoundError(
                f"dataset configs directory not found: {self._configs_dir}"
            )
        yaml_paths = sorted(self._configs_dir.glob("*.yaml"))
        if not yaml_paths:
            raise FileNotFoundError(
                f"no YAML configs under {self._configs_dir}"
            )

        # First pass: parse all YAMLs into intermediate dicts so we can
        # assign deterministic indices by sorted canonical id.
        parsed: list[dict] = []
        for path in yaml_paths:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if not isinstance(data, dict) or "id" not in data:
                raise ValueError(f"{path} is not a valid dataset config (missing 'id')")
            data["_source_path"] = path
            parsed.append(data)

        parsed.sort(key=lambda d: d["id"])

        for index, data in enumerate(parsed):
            meta = self._build_metadata(data, index)
            if meta.id in self._by_id:
                raise ValueError(
                    f"duplicate dataset id {meta.id!r} (sources: "
                    f"{self._by_id[meta.id].extra.get('_source_path')} and "
                    f"{data['_source_path']})"
                )
            self._by_id[meta.id] = meta
            self._by_index.append(meta)
            for alias in meta.aliases:
                if alias in self._alias_to_canonical or alias in self._by_id:
                    raise ValueError(
                        f"alias {alias!r} for {meta.id!r} clashes with an existing id/alias"
                    )
                self._alias_to_canonical[alias] = meta.id

    @staticmethod
    def _build_metadata(data: dict, index: int) -> DatasetMetadata:
        source = data["_source_path"]

        def require(key: str):
            if key not in data or data[key] is None:
                raise ValueError(
                    f"{source.name}: required field {key!r} missing or null"
                )
            return data[key]

        if "accel_target_sr" not in data or data["accel_target_sr"] in (None, 0):
            raise ValueError(
                f"{source.name}: accel_target_sr is missing or 0 — run "
                f"`python -m scripts.utils.derive_dataset_sampling_rate {source.relative_to(REPO_ROOT)} --apply` "
                f"to populate it from data."
            )

        position_source = require("position_source")
        if position_source not in _VALID_POSITION_SOURCES:
            raise ValueError(
                f"{source.name}: position_source={position_source!r} not in "
                f"{sorted(_VALID_POSITION_SOURCES)}"
            )

        raw_position_path = data.get("position_path")
        position_path: Path | None
        if raw_position_path in (None, "null", ""):
            if position_source not in ("default", "rowii"):
                raise ValueError(
                    f"{source.name}: position_path required when "
                    f"position_source={position_source!r}"
                )
            position_path = None
        else:
            position_path = (REPO_ROOT / str(raw_position_path)).resolve()

        vibration_format = data.get("vibration_format", "auto")
        if vibration_format not in ("auto", "peak", "raw"):
            raise ValueError(
                f"{source.name}: vibration_format={vibration_format!r} not in "
                f"['auto', 'peak', 'raw']"
            )

        window_scales = tuple(float(x) for x in require("window_scales_seconds"))
        if not window_scales:
            raise ValueError(f"{source.name}: window_scales_seconds is empty")

        extra = dict(data.get("extra", {}))
        description = extra.get("description") if isinstance(extra.get("description"), str) else None

        return DatasetMetadata(
            id=str(data["id"]),
            index=index,
            root=(REPO_ROOT / str(require("root"))).resolve(),
            n_mics=int(require("n_mics")),
            n_vibrations=int(require("n_vibrations")),
            accel_target_sr=int(data["accel_target_sr"]),
            vibration_format=vibration_format,
            position_source=position_source,
            position_path=position_path,
            label_scheme=str(require("label_scheme")),
            window_scales_seconds=window_scales,
            v3_window_seconds=float(require("v3_window_seconds")),
            v4_window_seconds=float(require("v4_window_seconds")),
            accel_sr_overrides={
                str(k): int(v) for k, v in (data.get("accel_sr_overrides") or {}).items()
            },
            aliases=tuple(data.get("aliases", []) or []),
            description=description,
            extra={**extra, "_source_path": str(source)},
        )

    # ---------------------------------------------------------------- lookup

    def _canonical(self, dataset_id: str) -> str:
        if dataset_id in self._by_id:
            return dataset_id
        if dataset_id in self._alias_to_canonical:
            return self._alias_to_canonical[dataset_id]
        raise KeyError(
            f"unknown dataset_id {dataset_id!r} (known: {sorted(self._by_id)}, "
            f"aliases: {sorted(self._alias_to_canonical)})"
        )

    def get(self, dataset_id: str) -> DatasetMetadata:
        return self._by_id[self._canonical(dataset_id)]

    def index_of(self, dataset_id: str) -> int:
        return self.get(dataset_id).index

    def all_ids(self) -> list[str]:
        return [m.id for m in self._by_index]

    def all_metadata(self) -> list[DatasetMetadata]:
        return list(self._by_index)

    def has(self, dataset_id: str) -> bool:
        try:
            self._canonical(dataset_id)
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        return len(self._by_index)

    def __iter__(self) -> Iterator[DatasetMetadata]:
        return iter(self._by_index)

    def __contains__(self, dataset_id: str) -> bool:
        return self.has(dataset_id)


REGISTRY = DatasetRegistry()


__all__ = ["REGISTRY", "DatasetMetadata", "DatasetRegistry"]
