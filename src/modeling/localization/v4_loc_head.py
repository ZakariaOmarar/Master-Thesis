"""V4 conditional localization head — heatmap soft-argmax + FiLM residual on (c [+ s]).

Two design moves over the original Cross3D global-pool regressor:

1. **Heatmap soft-argmax.**  The 3-D CNN now produces a per-voxel logit map
   over the same SRP-PHAT grid the front-end built.  A soft-argmax over the
   logit volume yields a continuous `(x, y, z)` initial estimate in metres
   without throwing the spatial structure of the SRP volume away (the
   original `AdaptiveAvgPool3d(1)` collapsed it).  This is the integral-
   regression formulation (Sun et al., 2018, "Integral human pose
   regression"); it is differentiable end-to-end and gives sub-voxel
   precision on a fixed grid.

2. **FiLM-conditioned residual.**  A small MLP on `[global_feat, tdoa_feat]`
   produces a small additive correction `Δ(x, y, z)`, FiLM-modulated by
   `c [+ s]`.  The conditional path corrects the unconditional spatial
   prior; the A3 ablation (`unconditional=True` / zero `c`) leaves the
   soft-argmax intact and exposes the FiLM-contributed gain in isolation.

The chained-system invariant is preserved: with all-zero `c` the FiLM
modulation is identity (γ, β are zero-init linear projections of `c`), so
the head degrades to (soft-argmax + unconditional residual MLP) — that
remains the A3 lower bound.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.set_transformer import MAB, PMA


class HeatmapCross3D(nn.Module):
    """3-D CNN producing per-voxel logits and a global feature.

    No spatial pooling inside the CNN trunk — we keep the full grid
    resolution for the heatmap.  A separate global-mean pool feeds the
    FiLM-residual head.

    Input:  `(B, 1, Nx, Ny, Nz)` — single-channel SRP-PHAT power volume.
    Output:
      - `logits`:        `(B, Nx, Ny, Nz)`  — per-voxel logits (pre-softmax).
      - `global_feat`:   `(B, feature_dim)` — global summary for the residual.
    """

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
        )
        self.logit_head = nn.Conv3d(64, 1, kernel_size=1)
        self.global_proj = nn.Linear(64, feature_dim)

    def forward(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if volume.ndim == 4:
            volume = volume.unsqueeze(1)  # (B, Nx, Ny, Nz) → (B, 1, Nx, Ny, Nz)
        if volume.ndim != 5:
            raise ValueError(f"volume must be (B, 1, Nx, Ny, Nz); got {tuple(volume.shape)}")
        h = self.cnn(volume)  # (B, 64, Nx, Ny, Nz)
        logits = self.logit_head(h).squeeze(1)  # (B, Nx, Ny, Nz)
        global_feat = h.mean(dim=(2, 3, 4))  # (B, 64)
        global_feat = self.global_proj(global_feat)  # (B, feature_dim)
        return logits, global_feat


def soft_argmax_3d(
    logits: torch.Tensor,
    grid_coords: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Soft-argmax of a 3-D logit volume over a fixed Cartesian grid.

    Args:
      logits: `(B, Nx, Ny, Nz)` per-voxel logits.
      grid_coords: `(Nx, Ny, Nz, 3)` voxel-centre coordinates in metres.
      temperature: positive scalar; higher → smoother (more weight on tails),
        lower → sharper (closer to argmax).  `1.0` is the standard choice.

    Returns:
      `(B, 3)` soft-argmax position in metres.
    """
    B = logits.shape[0]
    flat_logits = logits.reshape(B, -1) / float(max(temperature, 1e-6))
    weights = F.softmax(flat_logits, dim=-1)  # (B, V)
    coords = grid_coords.reshape(-1, 3).to(weights.device).to(weights.dtype)  # (V, 3)
    return weights @ coords  # (B, 3)


class TDOASetEncoder(nn.Module):
    """Per-pair MLP → self-attention → PMA(1) over structure-borne TDOA tokens.

    Input:  `(B, n_pairs, 8)` per-pair features
            `[path_diff_m, pos_i (3), pos_j (3), distance_m]`
            (cf. `v4_features.compute_accel_tdoa_tokens`).
    Output: `(B, feature_dim)` channel-agnostic TDOA summary.
    """

    def __init__(self, feature_dim: int = 64, n_heads: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.self_attn = MAB(feature_dim, num_heads=n_heads)
        self.pma = PMA(feature_dim, num_seeds=1, num_heads=n_heads)
        self.feature_dim = feature_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != 8:
            raise ValueError(f"tokens must be (B, n_pairs, 8); got {tuple(tokens.shape)}")
        if tokens.shape[1] == 0:
            B = tokens.shape[0]
            return torch.zeros(B, self.feature_dim, device=tokens.device, dtype=tokens.dtype)
        h = self.in_proj(tokens)
        h = self.self_attn(h, h)
        return self.pma(h).squeeze(1)


class FiLMResidualHead(nn.Module):
    """FiLM-conditioned residual MLP on `[global_feat, tdoa_feat, init_xyz]`
    → Δ(x, y, z).

    The residual is added to the soft-argmax initial estimate.  Three design
    choices that distinguish this head from the prior bounded-only version:

    1. **`init_xyz` is concatenated into the input feature.**  The MLP can
       now learn *how much to trust the soft-argmax estimate* depending on
       where it lies in the grid — corner-of-grid soft-argmax estimates are
       systematically biased toward the grid centre (the softmax-weighted
       centroid pulls inward), and passing the estimate as a feature lets
       the head learn to apply a larger correction when the estimate is
       near the grid boundary.

    2. **Residual bound widened to ±`residual_scale_m`** (default 0.20 m).
       The previous ±0.05 m cap was insufficient to correct the soft-argmax
       centre-bias on corner positions of the V4 cohort
       (e.g. ground-truth `(-0.20, 0, 0)` vs. soft-argmax pulling inward by
       ~ 10–15 cm).  The tanh squash keeps
       the residual from blowing up while allowing full grid-extent
       corrections when warranted.

    3. **FiLM γ/β zero-init + final-layer zero-init** preserve the A3
       ablation invariant: at init the head returns the soft-argmax
       prediction exactly, regardless of `c` / `s`.  Under `unconditional`
       the FiLM is the identity; the MLP still runs on the per-window
       features and can learn an unconditional correction, which is the
       A3 lower bound.
    """

    def __init__(
        self,
        in_dim: int,
        c_dim: int,
        s_dim: int = 0,
        hidden_dim: int = 128,
        residual_scale: float = 0.20,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        cond_dim = c_dim + s_dim
        # The MLP sees the feature stack + the soft-argmax init position
        # (3 extra dims).  See class-level docstring for the rationale.
        self.in_proj = nn.Linear(in_dim + 3, hidden_dim)
        self.film_gamma = nn.Linear(cond_dim, hidden_dim)
        self.film_beta = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)
        # Dropout slots after each GELU.  `nn.Dropout(0.0)` is a no-op so the
        # default keeps the head byte-equivalent to pre-fix behaviour.
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 3),
        )
        self.residual_scale = float(residual_scale)
        self.s_dim = s_dim
        self.c_dim = c_dim
        # Zero-init the final layer so the residual starts at exactly zero
        # — training begins as pure soft-argmax.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        s: torch.Tensor | None = None,
        init_xyz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if init_xyz is None:
            init_xyz = torch.zeros(x.shape[0], 3, device=x.device, dtype=x.dtype)
        h_in = torch.cat([x, init_xyz], dim=-1)
        h = self.in_proj(h_in)
        if self.s_dim > 0:
            if s is None:
                s = torch.zeros(x.shape[0], self.s_dim, device=x.device, dtype=x.dtype)
            cond = torch.cat([c, s], dim=-1)
        else:
            cond = c
        h = h * (1.0 + self.film_gamma(cond)) + self.film_beta(cond)
        out = self.head(h)  # (B, 3) raw
        return torch.tanh(out) * self.residual_scale  # (B, 3) bounded


class V4LocalizationHead(nn.Module):
    """Heatmap soft-argmax + TDOA-set + FiLM residual → (x, y, z) in metres.

    The head holds a fixed `grid_coords` buffer derived from the SRP-PHAT
    `GridSpec` it was constructed with.  All training / inference samples
    must come from the same grid.

    With `unconditional=True` (or all-zero `c`), the conditional residual
    MLP still runs on `[global_feat, tdoa_feat]` but the FiLM modulation is
    identity — this is the A3 ablation invariant.
    """

    def __init__(
        self,
        grid_coords: torch.Tensor,
        cnn_feature_dim: int = 128,
        tdoa_feature_dim: int = 64,
        c_dim: int = 128,
        s_dim: int = 0,
        hidden_dim: int = 128,
        n_heads_tdoa: int = 2,
        residual_scale_m: float = 0.20,
        soft_argmax_temperature: float = 1.0,
        head_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if grid_coords.ndim != 4 or grid_coords.shape[-1] != 3:
            raise ValueError(
                f"grid_coords must be (Nx, Ny, Nz, 3); got {tuple(grid_coords.shape)}"
            )
        # Persistent so it survives `state_dict()` round-trips.
        self.register_buffer("grid_coords", grid_coords.float(), persistent=True)
        self.soft_argmax_temperature = float(soft_argmax_temperature)

        self.cnn = HeatmapCross3D(feature_dim=cnn_feature_dim)
        self.tdoa = TDOASetEncoder(feature_dim=tdoa_feature_dim, n_heads=n_heads_tdoa)
        self.residual = FiLMResidualHead(
            in_dim=cnn_feature_dim + tdoa_feature_dim,
            c_dim=c_dim,
            s_dim=s_dim,
            hidden_dim=hidden_dim,
            residual_scale=residual_scale_m,
            dropout_p=head_dropout_p,
        )
        self.c_dim = c_dim
        self.s_dim = s_dim

    def forward(
        self,
        volume: torch.Tensor,
        tdoa_tokens: torch.Tensor,
        c: torch.Tensor,
        s: torch.Tensor | None = None,
        *,
        unconditional: bool = False,
        return_components: bool = False,
        external_init_xyz: torch.Tensor | None = None,
    ) -> torch.Tensor | dict:
        """Localisation forward pass.

        ``external_init_xyz`` (R3.3 / 2026-05-16): when provided, OVERRIDES
        the acoustic soft-argmax as the spatial init.  Used by
        ``channel_mode="vibration_only_learned"`` to bootstrap the head
        from a classical accel-TDOA multilateration estimate (mirror of
        the acoustic init from `srp_phat_3d` → `soft_argmax_3d`), so the
        vibration pipeline has a meaningful spatial prior instead of the
        grid-centroid collapse that the older ``tdoa_only`` mode produces.
        Shape must be ``(B, 3)`` in metres.  The 3-D CNN still runs
        (`logits` / `global_feat` are computed and returned), but only
        `global_feat` propagates to the residual MLP — the soft-argmax of
        the (zeroed) volume is discarded.
        """
        logits, global_feat = self.cnn(volume)
        if external_init_xyz is not None:
            if external_init_xyz.shape != (volume.shape[0], 3):
                raise ValueError(
                    f"external_init_xyz must be (B, 3); got "
                    f"{tuple(external_init_xyz.shape)} vs B={volume.shape[0]}"
                )
            init_xyz = external_init_xyz.to(device=volume.device, dtype=volume.dtype)
        else:
            init_xyz = soft_argmax_3d(
                logits, self.grid_coords, temperature=self.soft_argmax_temperature
            )
        tdoa_feat = self.tdoa(tdoa_tokens)
        feat = torch.cat([global_feat, tdoa_feat], dim=-1)
        if unconditional:
            c = torch.zeros_like(c)
            if s is not None:
                s = torch.zeros_like(s)
        # Pass init_xyz into the residual MLP so it can learn corner-bias
        # corrections; see FiLMResidualHead docstring.
        delta = self.residual(feat, c, s, init_xyz=init_xyz)
        pred = init_xyz + delta
        if return_components:
            return {
                "pred": pred,
                "init_xyz": init_xyz,
                "delta": delta,
                "logits": logits,
            }
        return pred


__all__ = [
    "FiLMResidualHead",
    "HeatmapCross3D",
    "TDOASetEncoder",
    "V4LocalizationHead",
    "soft_argmax_3d",
]
