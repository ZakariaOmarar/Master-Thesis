"""Set-Transformer building blocks (Lee et al. 2019) used as the channel pool.

Three components:

  - `MAB`: Multi-head Attention Block, the basic transformer block from the
     paper (LayerNorm → MultiheadAttention → residual → FFN → residual).
  - `PMA`: Pooling by Multi-head Attention, with `num_seeds` learned seed
     queries that attend over the input set.  V1 / V2 use `num_seeds=1` to
     produce a single summary vector per modality (`c_modality_t`).
  - `ChannelTokenEnricher`: builds per-channel tokens by concatenating
     `[feature, position_xyz, modality_embedding, dataset_embedding]` and
     projecting to the shared embedding dimension.

Permutation-invariance over the channel set is what makes the same trained
model handle D1 (4+4), D2 (5+5), D3 (9+4), and a future Illwerke array — the
core channel-agnosticism requirement of this work.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...config.dataset_registry import REGISTRY


def _default_n_datasets() -> int:
    """Dataset embedding table size = number of canonical ids in the registry."""
    return len(REGISTRY)


class MAB(nn.Module):
    """Multi-head Attention Block: pre-norm transformer block with cross-attention.

    `forward(q, kv)` lets the block do either self-attention (q == kv) or
    cross-attention (q != kv).  PMA below uses cross-attention (seeds → tokens).
    """

    def __init__(self, dim: int, num_heads: int = 4, ff_mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.attn(q_n, kv_n, kv_n, key_padding_mask=key_padding_mask, need_weights=False)
        x = q + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x

    def forward_with_attn(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Diagnostic forward — returns (output, attention_weights).

        Attention weights are averaged over heads by `MultiheadAttention` when
        `average_attn_weights=True` (the default).  Shape: ``(B, N_q, N_kv)``.
        Used only by `src/modeling/eval/fusion_forensics.py`; the regular
        training forward path uses `need_weights=False` for speed.
        """
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, attn_weights = self.attn(
            q_n, kv_n, kv_n,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        x = q + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x, attn_weights


class PMA(nn.Module):
    """Pooling by Multi-head Attention.

    `num_seeds` learned query tokens attend over the input set; the output is
    the corresponding `num_seeds` summary vectors.  For V1's per-modality
    summary we use `num_seeds=1`, which makes PMA a learned weighted average
    over the channel-token sequence.

    PMA is preferred over a naive temporal mean: the learned seeds let the
    network discover which frames + channels carry the most context.
    """

    def __init__(self, dim: int, num_seeds: int = 1, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_seeds = num_seeds
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MAB(dim, num_heads=num_heads, dropout=dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: (B, N, dim) → (B, num_seeds, dim)"""
        B = x.shape[0]
        seeds = self.seeds.expand(B, -1, -1)
        return self.mab(seeds, x, key_padding_mask=key_padding_mask)


class ChannelTokenEnricher(nn.Module):
    """Project per-channel features + sensor-coord + modality + dataset embeddings.

    The output `(B, N, embed_dim)` token sequence is the input to MAB and PMA.

    Modality and dataset are passed as integer indices (0/1 for mic/vibration,
    0/1/2/3 for D1/D2/D3/illwerke).  Indices outside the embedding range are
    rejected by the underlying `nn.Embedding`.
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int = 128,
        n_modalities: int = 2,
        n_datasets: int | None = None,
        modality_emb_dim: int = 16,
        dataset_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        if n_datasets is None:
            n_datasets = _default_n_datasets()
        self.modality_embed = nn.Embedding(n_modalities, modality_emb_dim)
        self.dataset_embed = nn.Embedding(n_datasets, dataset_emb_dim)
        in_dim = feature_dim + 3 + modality_emb_dim + dataset_emb_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        features: torch.Tensor,  # (B, N, feature_dim)
        xyz: torch.Tensor,  # (B, N, 3)
        modality_idx: int,  # scalar int in [0, n_modalities)
        dataset_idx: torch.Tensor,  # (B,) long
    ) -> torch.Tensor:
        if features.shape[:2] != xyz.shape[:2]:
            raise ValueError(
                f"features {tuple(features.shape)} and xyz {tuple(xyz.shape)} must agree on (B, N)"
            )
        B, N, _ = features.shape
        device = features.device

        mod_emb = self.modality_embed(
            torch.full((B, N), modality_idx, dtype=torch.long, device=device)
        )
        ds_emb = self.dataset_embed(dataset_idx.long().to(device)).unsqueeze(1).expand(-1, N, -1)
        cat = torch.cat([features, xyz.to(device), mod_emb, ds_emb], dim=-1)
        return self.proj(cat)


__all__ = ["MAB", "PMA", "ChannelTokenEnricher"]
