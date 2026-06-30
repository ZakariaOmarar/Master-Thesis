"""V3 anomaly trainer model pieces: learnable x_t pool and anchor helpers."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..encoders.set_transformer import PMA


def _fit_anchor_norm(cache: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Healthy mean/std of the cached per-window anchor (for standardization)."""
    A = np.concatenate([e["anchor"] for e in cache], axis=0)
    return A.mean(axis=0), A.std(axis=0) + 1e-8
def _augment_anchor(x: torch.Tensor, anchor: np.ndarray,
                    anchor_norm: tuple[np.ndarray, np.ndarray] | None) -> torch.Tensor:
    """Concatenate the standardized impulse+spectral anchor to a pooled x."""
    if anchor_norm is None:
        return x
    mean, std = anchor_norm
    a = torch.as_tensor((anchor - mean) / std, dtype=x.dtype, device=x.device)
    return torch.cat([x, a], dim=1)
class _XtPool(nn.Module):
    """Learnable channel-token pool for V3's `x_t` extraction.

    Replaces the legacy ``fused.mean(dim=1)`` (single first-moment average
    over the `(N_a + N_v)` channel-token axis) with two learned attention
    seeds (PMA, Lee et al. ICML 2019), concatenated and projected back to
    ``embed_dim``.

    Motivation (2026-05-19):

      The legacy mean pool was the **second** dilution stage for V3 at
      ``hop_length=43``: a knock signature that survived the per-mic
      ``AdaptiveAvgPool2d`` only contributes ``1/(N_a + N_v)`` to the
      channel-mean.  PMA-2 lets V3 jointly model a "stationary-mode-
      consistent" attention pattern (uniform weights) and a "transient-
      event-localized" attention pattern (mass on whichever mics see the
      knock peak) — two seeds is the minimum count strictly greater than
      V2's existing PMA-1 context pool, so the V3 ``x_t`` carries
      complementary information rather than a rescaled copy of ``c_t``.

      Trained jointly with the conditional flow on the frozen V2 encoder
      output.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.pma = PMA(embed_dim, num_seeds=2, num_heads=num_heads)
        self.proj = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        # fused: (B, N_a + N_v, embed_dim)
        pooled = self.pma(fused)                 # (B, 2, embed_dim)
        flat = pooled.flatten(start_dim=1)       # (B, 2 * embed_dim)
        return self.proj(flat)                   # (B, embed_dim)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
