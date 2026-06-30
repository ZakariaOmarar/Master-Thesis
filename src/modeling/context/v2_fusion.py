"""V2 fusion encoder: two PerModalityEncoders + bidirectional cross-attention + PMA.

Takes paired acoustic and vibration windows, produces:
  - per-modality pre-fusion tokens (LMM targets);
  - per-modality post-fusion tokens `z_t` (for LMM prediction);
  - a single context vector `c_t` for SimCLR + RQ1 cluster purity + V3 FiLM
    conditioning.

`c_t` aggregation supports three modes (selected by `context_mode`):
  - ``joint_pma`` : `c_t = PMA([fused_a; fused_v])`  — original V2 behaviour
  - ``skip``       : `c_t = MLP([PMA(joint); a_summary; v_summary])` — adds a
                     skip-connection from the per-modality PMA summaries to
                     the context vector, so the strong per-modality signal
                     observed at probe time is preserved through fusion.
  - ``dual_pma``   : `c_t = MLP([PMA_a(fused_a); PMA_v(fused_v)])` — replaces
                     the joint pool with two modality-specific PMAs whose
                     seed queries learn to read each modality individually.

The `apply_mask` helper replaces a random subset of tokens with a learned
mask token so V2's Latent Masked Modeling loss can predict the missing-token
embeddings from the cross-attention output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn

from ...config.architecture import ENCODER
from ..encoders import PMA, PerModalityEncoder
from ..fusion import BidirectionalCrossAttention

if TYPE_CHECKING:
    from .v2_ssl import V2SSLConfig

ContextMode = Literal["joint_pma", "skip", "dual_pma"]


class V2FusionEncoder(nn.Module):
    """Acoustic + vibration → cross-attention → context vector."""

    def __init__(
        self,
        feature_dim: int = ENCODER.feature_dim,
        embed_dim: int = ENCODER.embed_dim,
        n_heads: int = ENCODER.n_heads,
        n_modalities: int = 2,
        n_datasets: int | None = None,
        context_mode: ContextMode = "joint_pma",
        num_context_seeds: int = ENCODER.num_context_seeds,
        acoustic_cnn_width_mult: int = ENCODER.acoustic_cnn_width_mult,
        pool_type: str = ENCODER.pool_type,
        pool_reduction: int = ENCODER.pool_reduction,
    ) -> None:
        super().__init__()
        if context_mode not in ("joint_pma", "skip", "dual_pma"):
            raise ValueError(f"unknown context_mode {context_mode!r}")
        if num_context_seeds < 1:
            raise ValueError(f"num_context_seeds must be ≥ 1; got {num_context_seeds}")
        # V1→V2 weight transfer requires identical pool shapes on both
        # backbones, so the pool kwargs are propagated to both encoders
        # in lock-step.  `load_v1_weights(strict=True)` will raise if the
        # caller mismatches V1 / V2 pool configs.
        self.acoustic = PerModalityEncoder(
            modality="acoustic",
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_modalities=n_modalities,
            n_datasets=n_datasets,
            acoustic_cnn_width_mult=acoustic_cnn_width_mult,
            pool_type=pool_type,
            pool_reduction=pool_reduction,
        )
        self.vibration = PerModalityEncoder(
            modality="vibration",
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_modalities=n_modalities,
            n_datasets=n_datasets,
            pool_type=pool_type,
            pool_reduction=pool_reduction,
        )
        self.pool_type = pool_type
        self.pool_reduction = int(pool_reduction)
        self.fusion = BidirectionalCrossAttention(embed_dim, num_heads=n_heads)
        # Multi-seed PMA: `num_context_seeds` independent learned queries
        # attend over the fused tokens.  Their outputs are mean-pooled into
        # the single context vector `c_t`, preserving downstream c_dim while
        # giving the pool more capacity to summarise the joint sequence
        # (Lee et al., 2019, Set Transformer §3.2: "more seeds yield richer
        # multi-aspect summaries").  Default raised from 1 → 2.
        self.context_pool = PMA(embed_dim, num_seeds=num_context_seeds, num_heads=n_heads)
        self.num_context_seeds = num_context_seeds
        self.context_mode = context_mode
        if context_mode == "skip":
            self.skip_proj = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        elif context_mode == "dual_pma":
            self.context_pool_a = PMA(embed_dim, num_seeds=num_context_seeds, num_heads=n_heads)
            self.context_pool_v = PMA(embed_dim, num_seeds=num_context_seeds, num_heads=n_heads)
            self.dual_proj = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        self.mask_token = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.embed_dim = embed_dim

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        cfg: V2SSLConfig,
        *,
        map_location: str = "cpu",
    ) -> V2FusionEncoder:
        """Build an encoder sized from ``cfg`` and load its weights, in eval mode.

        Shared constructor for the post-hoc V4 analyses (the cross-validation
        drivers and the augmentation sweep) that all reuse a V2 encoder trained
        by ``full_run``. Sizing the encoder from ``cfg`` in one place keeps those
        callers from drifting out of sync with the orchestrator's V2 config.
        """
        encoder = cls(
            feature_dim=cfg.feature_dim,
            embed_dim=cfg.embed_dim,
            n_heads=cfg.n_heads,
            context_mode=cfg.context_mode,
            num_context_seeds=cfg.num_context_seeds,
            acoustic_cnn_width_mult=cfg.acoustic_cnn_width_mult,
        )
        encoder.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
        encoder.eval()
        return encoder

    # -- weight loading ----------------------------------------------------

    def load_v1_weights(
        self,
        acoustic_state_dict: dict | None,
        vibration_state_dict: dict | None,
        strict: bool = True,
    ) -> None:
        """Initialize the per-modality encoders from V1 SSL checkpoints.

        Either argument may be `None` to skip that modality (e.g., to keep
        random init for the modality whose V1 run is not yet available).
        """
        if acoustic_state_dict is not None:
            self.acoustic.load_state_dict(acoustic_state_dict, strict=strict)
        if vibration_state_dict is not None:
            self.vibration.load_state_dict(vibration_state_dict, strict=strict)

    # -- masking helper ----------------------------------------------------

    def apply_mask(
        self,
        tokens: torch.Tensor,
        mask_p: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace a random `mask_p` fraction of tokens with the learned mask token.

        Returns `(masked_tokens, mask)` where `mask` is a `(B, N)` boolean
        tensor flagging which positions were replaced.  At least one token per
        sample is always left unmasked so the cross-attention has something to
        attend to.
        """
        B, N, D = tokens.shape
        if mask_p <= 0.0 or N == 0:
            return tokens, torch.zeros(B, N, dtype=torch.bool, device=tokens.device)

        rand = torch.rand(B, N, device=tokens.device, generator=generator)
        mask = rand < mask_p
        # Guarantee ≥1 unmasked token per sample.
        all_masked = mask.all(dim=1)
        if all_masked.any():
            mask[all_masked, 0] = False

        masked_tokens = torch.where(
            mask.unsqueeze(-1),
            self.mask_token.view(1, 1, D).to(tokens.dtype),
            tokens,
        )
        return masked_tokens, mask

    # -- forward -----------------------------------------------------------

    def encode_modalities(
        self,
        ac_feat: torch.Tensor,
        ac_xyz: torch.Tensor,
        vib_feat: torch.Tensor,
        vib_xyz: torch.Tensor,
        dataset_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_tokens, _ = self.acoustic(ac_feat, ac_xyz, dataset_idx)
        v_tokens, _ = self.vibration(vib_feat, vib_xyz, dataset_idx)
        return a_tokens, v_tokens

    def encode_modalities_with_summaries(
        self,
        ac_feat: torch.Tensor,
        ac_xyz: torch.Tensor,
        vib_feat: torch.Tensor,
        vib_xyz: torch.Tensor,
        dataset_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like `encode_modalities` but also returns the per-modality PMA summaries.

        The summaries are needed by the cross-modal alignment (CMA) loss in V2's
        trainer.  They're cheap (PMA(num_seeds=1)) and live inside the
        per-modality encoders, so this is a strict superset of the original.
        """
        a_tokens, a_summary = self.acoustic(ac_feat, ac_xyz, dataset_idx)
        v_tokens, v_summary = self.vibration(vib_feat, vib_xyz, dataset_idx)
        return a_tokens, v_tokens, a_summary, v_summary

    def fuse_and_pool(
        self,
        a_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        a_summary: torch.Tensor | None = None,
        v_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run cross-attention and aggregate `c_t` per the configured `context_mode`.

        ``a_summary`` and ``v_summary`` are required when `context_mode="skip"`;
        ignored otherwise.  They come from `encode_modalities_with_summaries`
        and are the per-modality PMA pools the skip-connection feeds into c_t.
        """
        fused_a, fused_v = self.fusion(a_tokens, v_tokens)
        # PMA returns (B, num_seeds, embed_dim); reduce to (B, embed_dim) by
        # mean over seeds.  When num_seeds == 1 this is a no-op equivalent
        # to the previous `.squeeze(1)` and preserves checkpoint compatibility
        # for runs that didn't override the seed count.
        def _pool(p: PMA, x: torch.Tensor) -> torch.Tensor:
            return p(x).mean(dim=1)

        if self.context_mode == "joint_pma":
            all_tokens = torch.cat([fused_a, fused_v], dim=1)
            context = _pool(self.context_pool, all_tokens)
        elif self.context_mode == "skip":
            if a_summary is None or v_summary is None:
                raise ValueError("context_mode='skip' requires a_summary and v_summary")
            all_tokens = torch.cat([fused_a, fused_v], dim=1)
            joint = _pool(self.context_pool, all_tokens)
            context = self.skip_proj(torch.cat([joint, a_summary, v_summary], dim=-1))
        elif self.context_mode == "dual_pma":
            ca = _pool(self.context_pool_a, fused_a)
            cv = _pool(self.context_pool_v, fused_v)
            context = self.dual_proj(torch.cat([ca, cv], dim=-1))
        else:  # pragma: no cover — guarded in __init__
            raise AssertionError(self.context_mode)
        return fused_a, fused_v, context

    def forward(
        self,
        ac_feat: torch.Tensor,
        ac_xyz: torch.Tensor,
        vib_feat: torch.Tensor,
        vib_xyz: torch.Tensor,
        dataset_idx: torch.Tensor,
        mask_p: float = 0.0,
        mask_generator: torch.Generator | None = None,
    ) -> dict:
        """Run the full V2 forward pass.

        Returns a dict with:
          - ``a_target``, ``v_target``: pre-mask per-modality tokens (LMM targets)
          - ``a_fused``, ``v_fused``: post-fusion tokens
          - ``mask_a``, ``mask_v``: boolean masks (`(B, N_a)`, `(B, N_v)`)
          - ``context``: pooled `c_t` of shape `(B, embed_dim)`
        """
        a_target, v_target, a_summary, v_summary = self.encode_modalities_with_summaries(
            ac_feat, ac_xyz, vib_feat, vib_xyz, dataset_idx
        )
        if mask_p > 0.0:
            a_input, mask_a = self.apply_mask(a_target, mask_p, mask_generator)
            v_input, mask_v = self.apply_mask(v_target, mask_p, mask_generator)
        else:
            a_input, v_input = a_target, v_target
            mask_a = torch.zeros(a_target.shape[:2], dtype=torch.bool, device=a_target.device)
            mask_v = torch.zeros(v_target.shape[:2], dtype=torch.bool, device=v_target.device)

        fused_a, fused_v, context = self.fuse_and_pool(
            a_input, v_input, a_summary=a_summary, v_summary=v_summary
        )
        return {
            "a_target": a_target,
            "v_target": v_target,
            "a_summary": a_summary,
            "v_summary": v_summary,
            "a_fused": fused_a,
            "v_fused": fused_v,
            "mask_a": mask_a,
            "mask_v": mask_v,
            "context": context,
        }


__all__ = ["V2FusionEncoder"]
