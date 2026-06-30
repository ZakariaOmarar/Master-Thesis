"""V2 fusion: a single bidirectional cross-attention block.

Takes two per-modality token sequences from `PerModalityEncoder` (acoustic and
vibration), and produces fused versions where each modality has attended over
the other.  Implementation is two `MAB` cross-attention passes — one per
direction — sharing no weights.

By design this is **one block**.  Twins-Transformer dual-branch and
multi-scale spatiotemporal cross-attention are deferred.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..encoders.set_transformer import MAB


class BidirectionalCrossAttention(nn.Module):
    """One bidirectional cross-attention block.

    forward:
      acoustic_q  ──cross-attn over──▶ vibration_kv  →  fused_acoustic
      vibration_q ──cross-attn over──▶ acoustic_kv   →  fused_vibration

    Each direction is one `MAB` (pre-norm transformer block).  Output shapes
    match the inputs.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.a_from_v = MAB(dim, num_heads=num_heads, dropout=dropout)
        self.v_from_a = MAB(dim, num_heads=num_heads, dropout=dropout)

    def forward(
        self,
        acoustic_tokens: torch.Tensor,  # (B, N_a, D)
        vibration_tokens: torch.Tensor,  # (B, N_v, D)
        acoustic_key_padding_mask: torch.Tensor | None = None,
        vibration_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if acoustic_tokens.shape[-1] != vibration_tokens.shape[-1]:
            raise ValueError(
                "BidirectionalCrossAttention requires both modalities to share embed_dim; "
                f"got {acoustic_tokens.shape[-1]} vs {vibration_tokens.shape[-1]}"
            )
        fused_acoustic = self.a_from_v(
            acoustic_tokens, vibration_tokens, key_padding_mask=vibration_key_padding_mask
        )
        fused_vibration = self.v_from_a(
            vibration_tokens, acoustic_tokens, key_padding_mask=acoustic_key_padding_mask
        )
        return fused_acoustic, fused_vibration

    def forward_with_attn(
        self,
        acoustic_tokens: torch.Tensor,
        vibration_tokens: torch.Tensor,
        acoustic_key_padding_mask: torch.Tensor | None = None,
        vibration_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Diagnostic forward — returns the cross-attention weights in both
        directions alongside the fused token sequences.

        Returns ``(fused_acoustic, fused_vibration, attn_a_from_v, attn_v_from_a)``
        where each attention tensor has shape ``(B, N_q, N_kv)`` (head-averaged).
        Used by `src/modeling/eval/fusion_forensics.py` — the regular `forward`
        path is unchanged and remains the training/inference entry point.
        """
        if acoustic_tokens.shape[-1] != vibration_tokens.shape[-1]:
            raise ValueError(
                "BidirectionalCrossAttention requires both modalities to share embed_dim"
            )
        fused_acoustic, attn_a_from_v = self.a_from_v.forward_with_attn(
            acoustic_tokens, vibration_tokens, key_padding_mask=vibration_key_padding_mask
        )
        fused_vibration, attn_v_from_a = self.v_from_a.forward_with_attn(
            vibration_tokens, acoustic_tokens, key_padding_mask=acoustic_key_padding_mask
        )
        return fused_acoustic, fused_vibration, attn_a_from_v, attn_v_from_a


__all__ = ["BidirectionalCrossAttention"]
