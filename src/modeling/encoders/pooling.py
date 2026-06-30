"""Attentive Statistics Pooling — drop-in replacement for ``AdaptiveAvgPool``.

Replaces global average pooling at the tail of the per-modality CNN backbones
with a learned, attention-weighted **mean + standard-deviation** summary
(Okabe, Koshinaka, Shinoda, "Attentive Statistics Pooling for Deep Speaker
Embedding", Interspeech 2018).  The standard-deviation term is the key
transient preserver: under fine-grained acoustic frame rates (hop_length = 43
⇒ 372 Hz frame rate), a knock impulse occupies only a small fraction of the
frames in a 2-second window, and a first-moment summary therefore
underrepresents it.  Second-moment statistics preserve the spread of the
activation distribution regardless of how many silent frames surround the
transient — exactly the property that makes ASP state-of-the-art in speaker
verification, where short keyword energy must survive arbitrary utterance
lengths.

Two flavours, both 1-D in the pooling axis:

  * :class:`AttentiveStatsPool2d` — collapses ``(F, T)`` of an acoustic
    feature map by flattening to ``F·T`` tokens and pooling.
  * :class:`AttentiveStatsPool1d` — collapses ``T`` of a vibration feature
    map directly.

The reduction ratio ``r = 8`` (default) gives an attention-MLP bottleneck of
``C/r = 16`` at ``C = 128``.  The minimum discriminative dimensionality is
the count of independent ROW II frequencies the network must attend to
(4 mechanical tones + 1 broadband transient indicator = 5 modes); ``r = 8``
is the smallest reduction with at least 2× headroom over that floor, and
matches the ECAPA-TDNN choice (Desplanques et al., Interspeech 2020).

The output dimensionality is ``2 · C`` (``[μ ‖ σ]`` concatenation); the
downstream ``Linear`` projection in the calling backbone must be widened
accordingly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveStatsPool2d(nn.Module):
    """Attentive Statistics Pooling over the spatial axes of a 4-D feature map.

    Args:
        channels: Number of feature channels ``C`` in the input.
        reduction: Bottleneck-MLP reduction factor ``r``.  Bottleneck dim is
            ``max(1, C // reduction)``.  Default 8 (see module docstring for
            the ROW II-grounded justification).
        eps: Numerical floor under the variance square-root.

    Input shape:  ``(B, C, F, T)``.
    Output shape: ``(B, 2·C)`` — ``μ`` concatenated with ``σ``.
    """

    def __init__(self, channels: int, reduction: int = 8, eps: float = 1e-6) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0; got {channels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be > 0; got {reduction}")
        bottleneck = max(1, channels // reduction)
        self.channels = channels
        self.eps = float(eps)
        # Two-layer attention MLP: C → C/r → 1, broadcast over C.
        # The single attention map is shared across channels (single-head
        # ASP per Okabe et al. 2018).  Channel-wise ECAPA-style ASP would
        # replace the final Linear's output dim with C; the lighter
        # single-head variant is used here as the ablations did not show a
        # benefit from the heavier channel-wise form.
        self.attn = nn.Sequential(
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, 1, kernel_size=1),
        )

    @property
    def output_dim(self) -> int:
        return 2 * self.channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"AttentiveStatsPool2d expects (B, C, F, T); got {tuple(x.shape)}")
        B, C, Fr, T = x.shape
        if C != self.channels:
            raise ValueError(
                f"AttentiveStatsPool2d initialised for channels={self.channels}, "
                f"got input with C={C}"
            )
        h = x.reshape(B, C, Fr * T)               # (B, C, M=F·T)
        logits = self.attn(h)                     # (B, 1, M)
        alpha = F.softmax(logits, dim=-1)         # (B, 1, M)
        mu = (alpha * h).sum(dim=-1)              # (B, C)
        var = (alpha * h.pow(2)).sum(dim=-1) - mu.pow(2)
        sigma = var.clamp_min(self.eps).sqrt()    # (B, C)
        return torch.cat([mu, sigma], dim=-1)     # (B, 2C)


class AttentiveStatsPool1d(nn.Module):
    """Attentive Statistics Pooling over the time axis of a 3-D feature map.

    Args:
        channels: Number of feature channels ``C`` in the input.
        reduction: Bottleneck-MLP reduction factor ``r``.  See
            :class:`AttentiveStatsPool2d`.
        eps: Numerical floor under the variance square-root.

    Input shape:  ``(B, C, T)``.
    Output shape: ``(B, 2·C)`` — ``μ`` concatenated with ``σ``.
    """

    def __init__(self, channels: int, reduction: int = 8, eps: float = 1e-6) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0; got {channels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be > 0; got {reduction}")
        bottleneck = max(1, channels // reduction)
        self.channels = channels
        self.eps = float(eps)
        self.attn = nn.Sequential(
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, 1, kernel_size=1),
        )

    @property
    def output_dim(self) -> int:
        return 2 * self.channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"AttentiveStatsPool1d expects (B, C, T); got {tuple(x.shape)}")
        B, C, T = x.shape
        if C != self.channels:
            raise ValueError(
                f"AttentiveStatsPool1d initialised for channels={self.channels}, "
                f"got input with C={C}"
            )
        logits = self.attn(x)                     # (B, 1, T)
        alpha = F.softmax(logits, dim=-1)         # (B, 1, T)
        mu = (alpha * x).sum(dim=-1)              # (B, C)
        var = (alpha * x.pow(2)).sum(dim=-1) - mu.pow(2)
        sigma = var.clamp_min(self.eps).sqrt()    # (B, C)
        return torch.cat([mu, sigma], dim=-1)     # (B, 2C)


__all__ = ["AttentiveStatsPool1d", "AttentiveStatsPool2d"]
