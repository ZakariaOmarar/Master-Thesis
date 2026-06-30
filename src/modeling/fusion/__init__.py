"""Cross-modal fusion building blocks for V2+.

`BidirectionalCrossAttention` is the single fusion block stacked between the
two `PerModalityEncoder` token streams to produce fused tokens `z_t` from
which the context vector `c_t = PMA(z_t)` is pooled.
"""

from .cross_attention import BidirectionalCrossAttention

__all__ = ["BidirectionalCrossAttention"]
