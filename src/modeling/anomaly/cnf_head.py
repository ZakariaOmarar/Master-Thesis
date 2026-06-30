"""V3 anomaly head — RealNVP Conditional Normalizing Flow with FiLM conditioning.

Architecture choice: RealNVP affine coupling on a fixed-dim latent,
FiLM-conditioned on the V2 context vector `c_t`.  Glow's learnable 1×1
inv-conv is deferred — RealNVP is sufficient on small data and gives an exact
log-likelihood, which is exactly the streaming runtime's required
`anomaly_score = -log p(x|c)`.

Layout:
  - `FiLMMLP`         : MLP with FiLM(c) modulation between hidden layers
  - `FiLMCoupling`    : RealNVP affine coupling layer with FiLM-conditioned
                        scale (s) and translate (t) networks
  - `ConditionalRealNVP`: stack of `FiLMCoupling` layers with alternating
                        masks; provides `log_prob(x, c)` and `anomaly_score(x, c)`
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMMLP(nn.Module):
    """MLP with FiLM(c) modulation between hidden layers.

    Each FiLM block computes ``h ← h * (1 + γ(c̃)) + β(c̃)`` where ``c̃ = LN(c)``
    is the layer-normalised conditioner.  Normalising `c` before FiLM
    stabilises training when the upstream PMA pool's output norm drifts
    (which it does — the V2 fusion + PMA stack has no LayerNorm on its
    output).  γ and β are learned linear projections of `c̃`, zero-init so
    the model starts at the identity FiLM (= unconditional flow).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        c_dim: int,
        n_hidden: int = 2,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.cond_norm = nn.LayerNorm(c_dim)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
        )
        self.film_gamma = nn.ModuleList(
            [nn.Linear(c_dim, hidden_dim) for _ in range(n_hidden + 1)]
        )
        self.film_beta = nn.ModuleList(
            [nn.Linear(c_dim, hidden_dim) for _ in range(n_hidden + 1)]
        )
        # Initialise FiLM γ/β projections small so init is close to identity
        for g, b in zip(self.film_gamma, self.film_beta):
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)
            nn.init.zeros_(b.weight)
            nn.init.zeros_(b.bias)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        # Coupling MLP dropout — applied after each GELU between the FiLM
        # blocks.  `nn.Dropout(0.0)` is a no-op so the default keeps the
        # coupling byte-equivalent to pre-fix behaviour.
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c_n = self.cond_norm(c)
        h = self.in_proj(x)
        h = h * (1.0 + self.film_gamma[0](c_n)) + self.film_beta[0](c_n)
        h = F.gelu(h)
        h = self.dropout(h)
        for i, layer in enumerate(self.hidden):
            h = layer(h)
            h = h * (1.0 + self.film_gamma[i + 1](c_n)) + self.film_beta[i + 1](c_n)
            h = F.gelu(h)
            h = self.dropout(h)
        return self.out_proj(h)


class FiLMCoupling(nn.Module):
    """RealNVP affine coupling layer with FiLM-conditioned s and t networks.

    Given a binary `mask` (1 = passthrough, 0 = transformed):
        x_a = x * mask
        z   = x_a + (1 - mask) * (x * exp(s(x_a, c)) + t(x_a, c))
    where ``s = tanh(scale_net) * scale_max`` is bounded for
    invertibility / stability.  `scale_max` controls the per-layer
    multiplicative range: with `scale_max=1.0` the per-layer Jacobian
    factor is bounded in [e⁻¹, e¹] ≈ [0.37, 2.72]; with `scale_max=2.0`
    it spans [e⁻², e²] ≈ [0.14, 7.4], which is the standard RealNVP
    setting (Dinh et al., 2017) and gives the flow more expressive power
    on small-data regimes where each layer has to do more work.

    log-det-Jacobian = sum of `s` over the transformed positions.
    """

    def __init__(
        self,
        dim: int,
        c_dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
        n_hidden: int = 2,
        scale_max: float = 2.0,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if mask.numel() != dim:
            raise ValueError(f"mask numel {mask.numel()} must equal dim {dim}")
        self.register_buffer("mask", mask.float())
        self.scale_net = FiLMMLP(dim, dim, hidden_dim, c_dim, n_hidden, dropout_p=dropout_p)
        self.translate_net = FiLMMLP(dim, dim, hidden_dim, c_dim, n_hidden, dropout_p=dropout_p)
        self.scale_max = float(scale_max)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask
        inv_mask = 1.0 - mask
        x_a = x * mask
        s = (torch.tanh(self.scale_net(x_a, c)) * self.scale_max) * inv_mask
        t = self.translate_net(x_a, c) * inv_mask
        z = x_a + inv_mask * (x * torch.exp(s) + t)
        # Per-coupling log-det clamp (F6 — defensive bound on the worst-case
        # batch contribution).  |log_det| is *already* bounded by
        # `scale_max * (dim/2)` by construction (s = tanh(.) * scale_max), so
        # the clamp at ±50 is a defensive belt-and-suspenders against the
        # combination of large `dim` and `scale_max` blowing up the
        # accumulated log-det when summed across n_layers couplings.
        log_det = s.sum(dim=-1).clamp(-50.0, 50.0)
        return z, log_det


class ConditionalRealNVP(nn.Module):
    """Conditional Normalizing Flow on a fixed-size latent x, FiLM-conditioned on c.

    Base distribution: standard normal ``N(0, I)``, or — when
    ``conditional_base=True`` — a **context-conditional** Gaussian
    ``N(μ(c), diag(σ(c)²))`` whose location/scale are a small MLP of `c`.
    The conditional base lets the "centre of normal" move with the operating
    regime, so the score ``-log p(x|c)`` is regime-normalised by construction
    instead of leaking each regime's offset into the score.  This is the fix
    for the cross-regime confound that masked single-regime faults (e.g. D2's
    continuous knock separated at fusion-AUC 0.80 against the pooled healthy
    baseline but 0.93 against its own regime; the FiLM-only coupling barely
    moved off the unconditional flow, ΔNLL≈0.055).  The base head is zero-init
    so it starts *exactly* at ``N(0, I)``: training only adds regime structure,
    and a `flow.pt` saved without the head loads (strict=False) back to the
    plain ``N(0, I)`` behaviour.

    Anomaly score: ``-log p(x|c) = -[log p_z(z|c) + log|det J|]``.
    """

    # Defensive bound on the conditional log-σ so the base can't collapse
    # (σ→0 ⇒ exploding NLL) or flatten (σ→∞ ⇒ vanishing sensitivity).
    _LOG_SIGMA_MIN = -3.0
    _LOG_SIGMA_MAX = 3.0

    def __init__(
        self,
        dim: int,
        c_dim: int,
        n_layers: int = 6,
        hidden_dim: int = 64,
        n_hidden_per_net: int = 2,
        scale_max: float = 2.0,
        dropout_p: float = 0.0,
        conditional_base: bool = True,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be ≥ 1")
        self.dim = dim
        self.c_dim = c_dim
        self.scale_max = float(scale_max)
        self.conditional_base = bool(conditional_base)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % 2 :: 2] = 1.0  # alternating mask across layers
            self.layers.append(
                FiLMCoupling(
                    dim, c_dim, hidden_dim, mask,
                    n_hidden=n_hidden_per_net, scale_max=scale_max,
                    dropout_p=dropout_p,
                )
            )
        if self.conditional_base:
            # c → (μ, log σ); LayerNorm(c) mirrors the FiLM conditioner so the
            # head is robust to the upstream PMA pool's output-norm drift.
            self.base_net = nn.Sequential(
                nn.LayerNorm(c_dim),
                nn.Linear(c_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * dim),
            )
            # Zero-init the output projection ⇒ μ=0, log σ=0 ⇒ N(0, I) at init.
            nn.init.zeros_(self.base_net[-1].weight)
            nn.init.zeros_(self.base_net[-1].bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, ld = layer(z, c)
            log_det = log_det + ld
        return z, log_det

    def _base_params(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-window base mean/log-σ from the conditioner (clamped)."""
        mu, log_sigma = self.base_net(c).chunk(2, dim=-1)
        return mu, log_sigma.clamp(self._LOG_SIGMA_MIN, self._LOG_SIGMA_MAX)

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x, c)
        if self.conditional_base:
            mu, log_sigma = self._base_params(c)
            u = (z - mu) * torch.exp(-log_sigma)
            log_p_z = (
                -0.5 * (u * u).sum(dim=-1)
                - log_sigma.sum(dim=-1)
                - 0.5 * self.dim * math.log(2.0 * math.pi)
            )
        else:
            log_p_z = (
                -0.5 * (z * z).sum(dim=-1)
                - 0.5 * self.dim * math.log(2.0 * math.pi)
            )
        return log_p_z + log_det

    def anomaly_score(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x, c)


__all__ = ["ConditionalRealNVP", "FiLMCoupling", "FiLMMLP"]
