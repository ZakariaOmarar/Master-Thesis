"""Deep impulse-aware anomaly flow: a learned raw front-end plus an anchored flow.

The SSL/CMA encoders tend to discard the impulsive and spectral cues a knock
produces; a Ridge bottleneck probe shows the embedding cannot predict crest
factor (R² close to 0). This model addresses that directly:

  - A 1-D CNN reads the raw waveform window per modality and is trained
    end-to-end with a conditional normalizing flow on the healthy
    negative-log-likelihood. Because the objective is one-class density rather
    than contrastive/CMA, transients are preserved instead of collapsed.
  - The hand-crafted impulse and spectral features are concatenated to the
    learned embedding as a recall anchor. They cannot be optimised away, so a
    knock still registers even when a new campaign looks different, and they
    keep the one-class objective from collapsing to a trivial solution.
  - A learned low-dimensional context head and the flow's context-conditional
    base normalise the healthy density per operating regime, so one global
    threshold transfers across campaigns.

The anomaly score is -log p([cnn_emb ⊕ anchor] | context). The model is fit on
healthy windows only, and the per-modality scores are sum-fused.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .cnf_head import ConditionalRealNVP


@dataclass
class DeepImpulseConfig:
    """All hyperparameters for the deep impulse-aware anomaly detector.

    Single source of truth shared by training (`train_deep_impulse_flow.py`)
    and the hyperparameter search (`search_deep_impulse_flow.py`).  Feature
    knobs (n_mels/win_s/stride_s) define the on-disk feature cache, so the HP
    search holds them fixed and sweeps the model/training knobs.
    """
    # --- features (define the feature cache; fixed during HP search) ---
    n_mels: int = 64
    n_t: int = 64
    win_s: float = 1.0
    stride_s: float = 0.5
    n_anchor: int = 16
    # --- model ---
    emb_dim: int = 32
    ctx_dim: int = 8
    dropout: float = 0.1
    flow_layers: int = 6
    flow_hidden: int = 64
    # --- training ---
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    val_frac: float = 0.15
    augment: str = "strong"            # "none" | "light" | "strong"
    target_fpr: float = 0.05
    seed: int = 0
    # --- few-shot adaptation to a new campaign ---
    adapt_frac: float = 0.0
    adapt_epochs: int = 10
    adapt_lr_mult: float = 0.05


class RawCNN1D(nn.Module):
    """Compact 1-D CNN front-end on a fixed-length raw window -> embedding.

    Small-kernel strided convs keep it sensitive to sharp transients; modest
    depth/width suits the prototype-scale data (guards overfitting).
    """

    def __init__(self, in_len: int, emb_dim: int = 32, in_ch: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, 9, stride=4, padding=4), nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 32, 9, stride=4, padding=4), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 48, 7, stride=4, padding=3), nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, emb_dim, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.emb_dim = emb_dim

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if w.dim() == 2:
            w = w.unsqueeze(1)  # (B, L) -> (B, 1, L)
        return self.net(w).squeeze(-1)  # (B, emb_dim)


class SpectroCNN(nn.Module):
    """Compact 2-D CNN over a (freq, time) spectrogram -> embedding.

    A spectrogram keeps frequency content the raw-downsample path discards
    (essential for SPECTRAL anomalies like D3) while still exposing transients
    as vertical stripes (impulsive D2/D4).  Dropout guards the small-data regime.
    """

    def __init__(self, emb_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.GELU(),
            nn.Conv2d(48, emb_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.emb_dim = emb_dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if s.dim() == 3:
            s = s.unsqueeze(1)  # (B, F, T) -> (B, 1, F, T)
        return self.net(s).flatten(1)  # (B, emb_dim)


class DeepImpulseFlow(nn.Module):
    """Learned front-end + anchored conditional flow (one-class, per modality).

    front="spectro" (recommended): 2-D CNN over a (freq,time) spectrogram —
    preserves spectral (D3) and transient (D2/D4) cues.  front="raw1d": 1-D CNN
    over a fixed-length raw window.

    `anchor` = standardized hand-crafted impulse+spectral features (recall
    guarantee + anti-collapse; cannot be optimised away).  `context` = learned
    low-dim regime descriptor for the flow's conditional base.
    """

    def __init__(self, n_anchor: int, *, front: str = "spectro", in_len: int = 8192,
                 emb_dim: int = 32, ctx_dim: int = 8, dropout: float = 0.1,
                 flow_layers: int = 6, flow_hidden: int = 64) -> None:
        super().__init__()
        self.front = front
        if front == "spectro":
            self.cnn = SpectroCNN(emb_dim, dropout)
        elif front == "raw1d":
            self.cnn = RawCNN1D(in_len, emb_dim)
        else:
            raise ValueError(f"unknown front {front!r}")
        self.ctx_head = nn.Sequential(nn.Linear(emb_dim, ctx_dim), nn.Tanh())
        self.flow = ConditionalRealNVP(
            dim=emb_dim + n_anchor, c_dim=ctx_dim,
            n_layers=flow_layers, hidden_dim=flow_hidden, conditional_base=True,
        )
        self.n_anchor = n_anchor

    @classmethod
    def from_config(cls, cfg: DeepImpulseConfig) -> DeepImpulseFlow:
        return cls(cfg.n_anchor, front="spectro", emb_dim=cfg.emb_dim,
                   ctx_dim=cfg.ctx_dim, dropout=cfg.dropout,
                   flow_layers=cfg.flow_layers, flow_hidden=cfg.flow_hidden)

    def log_prob(self, front_in: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        emb = self.cnn(front_in)
        ctx = self.ctx_head(emb)
        return self.flow.log_prob(torch.cat([emb, anchor], dim=1), ctx)

    def anomaly_score(self, front_in: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(front_in, anchor)


__all__ = ["DeepImpulseFlow", "RawCNN1D"]
