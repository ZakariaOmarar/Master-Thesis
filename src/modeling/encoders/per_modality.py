"""Per-modality encoder: plain CNN backbone + Set-Transformer pool.

Two backbones, both deliberately plain (no CBAM, no BiLSTM, no
Twins-Transformer) so that the contribution is the fusion and conditioning,
not backbone tuning:

  - `Acoustic2DCNN`: 2-D CNN on `(2, F, T)` log-mel + CWT input from
    `src/features/audio_spectral.py`.
  - `Vibration1DCNN`: 1-D CNN on `(3, T)` amplitude + Hilbert envelope +
    rolling-kurtosis input from `src/features/vibration_temporal.py`.

`PerModalityEncoder` wires a backbone to the channel-agnostic Set-Transformer
pool (`ChannelTokenEnricher` → MAB → PMA(num_seeds=1)).  It returns both:

  - per-channel tokens, shape `(B, N, embed_dim)`, for V2's cross-attention
    fusion to consume after V1 weight transfer; and
  - a single PMA summary, shape `(B, embed_dim)`, for V1's contrastive loss
    and the cluster-purity sanity gate.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ...config.architecture import ENCODER
from .pooling import AttentiveStatsPool1d, AttentiveStatsPool2d
from .set_transformer import MAB, PMA, ChannelTokenEnricher

PoolType = Literal["avg", "asp"]
_DEFAULT_POOL: PoolType = ENCODER.pool_type  # type: ignore[assignment]
_DEFAULT_POOL_REDUCTION = ENCODER.pool_reduction


def _norm2d(num_channels: int, norm: str) -> nn.Module:
    """Normalisation layer for the 2-D acoustic CNN — see `_norm` rationale."""
    if norm == "group":
        groups = min(8, max(1, num_channels // 4))
        return nn.GroupNorm(num_groups=groups, num_channels=num_channels)
    return nn.BatchNorm2d(num_channels)


def _norm1d(num_channels: int, norm: str) -> nn.Module:
    """Normalisation layer for the 1-D vibration CNN — see `_norm` rationale."""
    if norm == "group":
        groups = min(8, max(1, num_channels // 4))
        return nn.GroupNorm(num_groups=groups, num_channels=num_channels)
    return nn.BatchNorm1d(num_channels)


# Normalisation choice (F7).  The 2026-05-14 run swapped BatchNorm → GroupNorm
# for channel-count invariance, but this collapsed the V1 acoustic contrastive
# encoder: the NT-Xent loss returned to its degenerate fixed point (≈ ln(N))
# and the embeddings went to effective rank 1.9 / cosine 0.99999.  BatchNorm's
# cross-sample batch coupling is load-bearing for SimCLR-style contrastive
# pretraining — removing it removes the implicit anti-collapse pressure.  The
# default is therefore BACK to "batch"; "group" is retained as an opt-in knob
# (the channel-count-invariance concern F7 raised is real, but the fix is a
# variance-regularised SSL objective, not a bare norm swap — see Chapter 7).
_DEFAULT_NORM = ENCODER.norm


class Acoustic2DCNN(nn.Module):
    """Plain 2-D CNN backbone applied per microphone.

    Input:  `(B, N_mic, 2, F, T)` — channel 0 is log-mel, channel 1 is CWT
    Output: `(B, N_mic, feature_dim)` — per-mic feature vector

    ``width_mult`` (R1a, 2026-05-16): scales every conv-block channel count by
    the same factor.  Default 1 reproduces the published 32/64/128 backbone;
    ``width_mult=2`` gives 64/128/256 (~3.7× param count) which the R1
    acoustic-improvement experiment targets.  V1 and V2 encoders must use
    the same value so that V1→V2 weight transfer (`load_v1_weights`)
    sees compatible shapes.

    ``pool_type`` (2026-05-19): pooling kind at the tail of the CNN.

      * ``"asp"`` (publication default) — Attentive Statistics Pooling
        (Okabe et al., Interspeech 2018) over the (F, T) feature map.
        Concatenates an attention-weighted mean and standard deviation
        per channel; the std term preserves transient knock signatures
        that the legacy ``AdaptiveAvgPool2d(1)`` diluted at fine hop
        lengths (hop=43 ⇒ ~745 frames per 2 s window; a 50 ms knock is
        ~2.7 % of the frames, so a first-moment-only summary is
        dominated by background).
      * ``"avg"`` — legacy ``AdaptiveAvgPool2d(1)``; retained for
        ablation and backward compatibility with hop=512 checkpoints.

    V1 and V2 must use the same ``pool_type`` and ``pool_reduction``
    for V1→V2 weight transfer to succeed.
    """

    def __init__(
        self,
        in_channels: int = 2,
        feature_dim: int = ENCODER.feature_dim,
        norm: str = _DEFAULT_NORM,
        width_mult: int = ENCODER.acoustic_cnn_width_mult,
        pool_type: PoolType = _DEFAULT_POOL,
        pool_reduction: int = _DEFAULT_POOL_REDUCTION,
    ) -> None:
        super().__init__()
        if width_mult < 1:
            raise ValueError(f"width_mult must be ≥ 1, got {width_mult}")
        if pool_type not in ("avg", "asp"):
            raise ValueError(f"pool_type must be 'avg' or 'asp'; got {pool_type!r}")
        c1 = ENCODER.cnn_c1 * width_mult
        c2 = ENCODER.cnn_c2 * width_mult
        c3 = ENCODER.cnn_c3 * width_mult
        if pool_type == "asp":
            pool_layer: nn.Module = AttentiveStatsPool2d(c3, reduction=pool_reduction)
            proj_in_dim = 2 * c3
        else:
            pool_layer = nn.AdaptiveAvgPool2d(1)
            proj_in_dim = c3
        self.pool_type = pool_type
        self.pool_reduction = int(pool_reduction)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            _norm2d(c1, norm),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            _norm2d(c2, norm),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            _norm2d(c3, norm),
            nn.GELU(),
            pool_layer,
        )
        self.proj = nn.Linear(proj_in_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Acoustic2DCNN expects (B, N, C, F, T); got {tuple(x.shape)}")
        B, N, C, F, T = x.shape
        flat = x.reshape(B * N, C, F, T)
        h = self.cnn(flat).flatten(start_dim=1)  # (B*N, c3) for avg; (B*N, 2*c3) for asp
        h = self.proj(h)  # (B*N, feature_dim)
        return h.reshape(B, N, -1)


class Vibration1DCNN(nn.Module):
    """Plain 1-D CNN backbone applied per vibration channel.

    Input:  `(B, N_vib, 3, T)` — channels are amplitude / envelope / kurtosis
    Output: `(B, N_vib, feature_dim)` — per-vibration feature vector

    ``pool_type`` mirrors :class:`Acoustic2DCNN` — see that class for the
    rationale.  Both backbones must use the same value (V1→V2 transfer
    enforces this via :func:`v2_fusion.load_v1_weights`).
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = ENCODER.feature_dim,
        norm: str = _DEFAULT_NORM,
        pool_type: PoolType = _DEFAULT_POOL,
        pool_reduction: int = _DEFAULT_POOL_REDUCTION,
    ) -> None:
        super().__init__()
        if pool_type not in ("avg", "asp"):
            raise ValueError(f"pool_type must be 'avg' or 'asp'; got {pool_type!r}")
        # Normalisation: see `_DEFAULT_NORM` rationale above (F7).
        c3 = ENCODER.cnn_c3
        if pool_type == "asp":
            pool_layer: nn.Module = AttentiveStatsPool1d(c3, reduction=pool_reduction)
            proj_in_dim = 2 * c3
        else:
            pool_layer = nn.AdaptiveAvgPool1d(1)
            proj_in_dim = c3
        self.pool_type = pool_type
        self.pool_reduction = int(pool_reduction)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            _norm1d(32, norm),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            _norm1d(64, norm),
            nn.GELU(),
            nn.Conv1d(64, c3, kernel_size=5, padding=2),
            _norm1d(c3, norm),
            nn.GELU(),
            pool_layer,
        )
        self.proj = nn.Linear(proj_in_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Vibration1DCNN expects (B, N, C, T); got {tuple(x.shape)}")
        B, N, C, T = x.shape
        flat = x.reshape(B * N, C, T)
        h = self.cnn(flat).flatten(start_dim=1)
        h = self.proj(h)
        return h.reshape(B, N, -1)


class PerModalityEncoder(nn.Module):
    """CNN backbone → ChannelTokenEnricher → MAB → PMA(1).

    Returns `(tokens, summary)` so V2 can consume the token sequence and V1
    can consume the summary.
    """

    def __init__(
        self,
        modality: Literal["acoustic", "vibration"],
        feature_dim: int = ENCODER.feature_dim,
        embed_dim: int = ENCODER.embed_dim,
        n_heads: int = ENCODER.n_heads,
        n_modalities: int = 2,
        n_datasets: int | None = None,
        norm: str = _DEFAULT_NORM,
        acoustic_cnn_width_mult: int = ENCODER.acoustic_cnn_width_mult,
        pool_type: PoolType = _DEFAULT_POOL,
        pool_reduction: int = _DEFAULT_POOL_REDUCTION,
    ) -> None:
        super().__init__()
        self.modality = modality
        self.pool_type = pool_type
        self.pool_reduction = int(pool_reduction)
        if modality == "acoustic":
            self.backbone: nn.Module = Acoustic2DCNN(
                in_channels=2,
                feature_dim=feature_dim,
                norm=norm,
                width_mult=acoustic_cnn_width_mult,
                pool_type=pool_type,
                pool_reduction=pool_reduction,
            )
            self.modality_idx = 0
        elif modality == "vibration":
            # Vibration is intentionally kept at the published width — the
            # R1 experiment changes only the acoustic backbone.  Pass
            # `acoustic_cnn_width_mult` here to make accidental swaps loud.
            self.backbone = Vibration1DCNN(
                in_channels=3,
                feature_dim=feature_dim,
                norm=norm,
                pool_type=pool_type,
                pool_reduction=pool_reduction,
            )
            self.modality_idx = 1
        else:
            raise ValueError(f"unknown modality {modality!r}")

        self.enricher = ChannelTokenEnricher(
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            n_modalities=n_modalities,
            n_datasets=n_datasets,
        )
        self.self_attn = MAB(embed_dim, num_heads=n_heads)
        self.pma = PMA(embed_dim, num_seeds=1, num_heads=n_heads)
        self.embed_dim = embed_dim

    def forward(
        self,
        x: torch.Tensor,  # (B, N, ...) feature tensor for this modality
        xyz: torch.Tensor,  # (B, N, 3) sensor positions in metres
        dataset_idx: torch.Tensor,  # (B,) long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)  # (B, N, feature_dim)
        tokens = self.enricher(feats, xyz, self.modality_idx, dataset_idx)  # (B, N, embed_dim)
        tokens = self.self_attn(tokens, tokens)  # one self-attention pass
        summary = self.pma(tokens).squeeze(1)  # (B, embed_dim)
        return tokens, summary


__all__ = ["Acoustic2DCNN", "PerModalityEncoder", "Vibration1DCNN"]
