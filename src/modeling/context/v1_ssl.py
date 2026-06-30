"""V1 per-modality SSL warmup — SimCLR-style contrastive on healthy windows.

Trains the acoustic OR vibration encoder independently on healthy windows of
D1+D2.  No labels touch the training loop.  Mode labels appear only at the
sanity-gate evaluation step (cluster purity, computed by `cluster_metric`).

Design choices:
  - Per-modality SimCLR contrastive (acoustic and vibration trained
    separately).  No cross-modal anything yet — that arrives in V2.
  - NT-Xent loss between two augmented views of each anchor window.
  - Augmentations applied in feature space: SpecAugment + channel dropout +
    gain jitter.  We deliberately skip pre-feature time-domain augmentations
    so feature extraction can be amortised with a one-time precomputation.
  - Per-recording train/val split (recordings disjoint between splits).

`train_v1_per_modality` is the single entry point.  It returns a `V1Result`
dataclass containing the trained encoder, projection head, sanity-gate
metrics, and the train/val recording IDs.
"""
from .v1_ssl_config import (
    V1Result,
    V1SSLConfig,
    _dataset_idx,
    _registry_window_scales,
)
from .v1_ssl_data import (
    _Augmenter,
    _collate,
    _gather_healthy_segments,
    _gather_labeled_segments,
    _GroupedBatchSampler,
    _precompute_segment,
    _PrecomputedSegment,
    _resolve_segment_scales,
    _split_segments_by_recording,
    _time_split_segment,
    _WindowedFeatureDataset,
)
from .v1_ssl_model import _nt_xent, _ProjectionHead
from .v1_ssl_train import (
    _encode_summary,
    evaluate_sanity_gate,
    train_v1_per_modality,
)

__all__ = [
    "V1Result",
    "V1SSLConfig",
    "_Augmenter",
    "_GroupedBatchSampler",
    "_PrecomputedSegment",
    "_ProjectionHead",
    "_WindowedFeatureDataset",
    "_collate",
    "_dataset_idx",
    "_encode_summary",
    "_gather_healthy_segments",
    "_gather_labeled_segments",
    "_nt_xent",
    "_precompute_segment",
    "_registry_window_scales",
    "_resolve_segment_scales",
    "_split_segments_by_recording",
    "_time_split_segment",
    "evaluate_sanity_gate",
    "train_v1_per_modality",
]
