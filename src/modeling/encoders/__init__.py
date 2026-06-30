"""V1+ encoder building blocks.

Two layers, both reused by V2:
  - `set_transformer`: channel-agnostic Set-Transformer pool (MAB + PMA) plus a
    `ChannelTokenEnricher` that augments per-channel features with sensor-coord,
    modality, and dataset embeddings.  Channel-agnosticism is a core design
    requirement: one model must serve datasets with differing sensor counts.
  - `per_modality`:    plain CNN backbones (2-D for acoustic, 1-D for vibration)
    bundled with the Set-Transformer pool into a `PerModalityEncoder`.  Outputs
    both per-channel tokens (for V2 cross-attention) and a single PMA summary
    (for V1 contrastive + cluster purity).
"""

from .per_modality import (
    Acoustic2DCNN,
    PerModalityEncoder,
    Vibration1DCNN,
)
from .set_transformer import (
    MAB,
    PMA,
    ChannelTokenEnricher,
)

__all__ = [
    "MAB",
    "PMA",
    "Acoustic2DCNN",
    "ChannelTokenEnricher",
    "PerModalityEncoder",
    "Vibration1DCNN",
]
