"""V2 multimodal SSL trainer — bidirectional cross-attention + Latent Masked Modeling.

Inherits V1 weights for both per-modality encoders and trains the fusion
block + context PMA jointly under two equally-weighted losses:

  1. **SimCLR contrastive on `c_t`** — two augmented views of each paired
     window pass through the full pipeline; the NT-Xent loss pulls the two
     views' context vectors together.
  2. **Latent Masked Modeling (LMM)** — at training time, a fraction of
     per-modality tokens (acoustic and vibration, independently) are replaced
     by a learned mask token before the cross-attention block.  The fused
     output at masked positions is regressed back to the **pre-mask
     per-modality tokens** via a cosine-similarity loss.  This is *not* raw
     cross-modal reconstruction (no scalogram pixels or CSV values are
     decoded) — it stays entirely in latent space, per the supervisor's note.

RQ1 metric: K-means(k=4) on `c_t` Hungarian-matched to D1+D2 folder labels.
The same evaluation function powers the A1 ablation (vibration-severed):
when `cfg.drop_vibration=True`, the vibration-feature inputs are zeroed at
the input — the cross-attention block + PMA still run, so the architecture
stays identical, but no information flows from the vibration branch.
"""


from .v2_ssl_config import (
    V2Result,
    V2SSLConfig,
    _dataset_idx,
    _registry_window_scales,
)
from .v2_ssl_data import (
    _collate,
    _gather_labeled_segments,
    _gather_paired_segments,
    _PairedAugmenter,
    _PairedGroupedBatchSampler,
    _PairedSegment,
    _PairedWindowedDataset,
    _precompute_paired,
    _resolve_paired_segment_scales,
    _split_segments_by_recording,
    _time_split_paired_segment,
)
from .v2_ssl_model import _lmm_loss, _nt_xent, _ProjectionHead
from .v2_ssl_train import evaluate_rq1_purity, train_v2_fusion

__all__ = [
    "V2Result",
    "V2SSLConfig",
    "_PairedAugmenter",
    "_PairedGroupedBatchSampler",
    "_PairedSegment",
    "_PairedWindowedDataset",
    "_ProjectionHead",
    "_collate",
    "_dataset_idx",
    "_gather_labeled_segments",
    "_gather_paired_segments",
    "_lmm_loss",
    "_nt_xent",
    "_precompute_paired",
    "_registry_window_scales",
    "_resolve_paired_segment_scales",
    "_split_segments_by_recording",
    "_time_split_paired_segment",
    "evaluate_rq1_purity",
    "train_v2_fusion",
]
