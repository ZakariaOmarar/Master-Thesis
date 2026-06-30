"""R2.1 — Per-modality encoder adapters for V3 (anomaly) training.

`train_v3_cnf` ([src/modeling/anomaly/v3_trainer.py](../anomaly/v3_trainer.py))
is duck-typed on its encoder.  It calls
``encoder(ac, ac_xyz, vib, vib_xyz, ds_idx, mask_p=...)`` and reads dict
keys ``a_fused`` / ``v_fused`` / ``context`` from the output, then mean-pools
``cat([a_fused, v_fused], dim=1)`` to form V3's input ``x`` and uses
``context`` as V3's FiLM conditioner ``c``.

That gives us a clean way to fit two more V3 instances *without* touching
the trainer body: wrap a ``PerModalityEncoder`` (V1) so it presents the
V2-style dict-returning interface, with the other-modality slots zeroed.

- ``V3AcousticOnlyAdapter`` consumes the acoustic input, ignores vibration,
  returns ``{"a_fused": tokens_a, "v_fused": zeros, "context": summary_a}``.
- ``V3VibrationOnlyAdapter`` is symmetric.

V3's ``_extract_xc`` then yields ``x = mean(tokens_modality)`` (the other
half of the concatenation is zero, so the mean is half the per-modality
mean by construction — matches the dim contract) and ``c = summary_modality``.
Both adapters compose with the existing `_PairedWindowedDataset` paired
loader because they take the same inputs as V2 — the unused modality's
input is simply dropped inside the adapter.

The adapters carry no trainable parameters of their own; all parameters
live in the wrapped ``PerModalityEncoder``.  Set the wrapped encoder to
``eval()`` and freeze its parameters in the V3 trainer the same way the
V2 fusion encoder is frozen at line 162-164 of v3_trainer.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..encoders.per_modality import PerModalityEncoder


@dataclass
class _PerModalityXC:
    """What V3 consumes from `_extract_xc` for one paired batch.

    Kept private — exposed only for adapter unit tests; production V3 path
    walks the encoder.forward output dict directly.
    """

    x: torch.Tensor  # (B, embed_dim) — V3 CNF input
    c: torch.Tensor  # (B, embed_dim) — V3 FiLM conditioner


class V3AcousticOnlyAdapter(nn.Module):
    """V2-API-compatible wrapper around a V1-acoustic ``PerModalityEncoder``.

    Forward signature matches `V2FusionEncoder.forward` so the V3 trainer
    (``_extract_xc`` in `v3_trainer.py`) consumes it transparently.
    Vibration inputs are accepted (for API uniformity with paired loaders)
    but **discarded** inside the adapter — vibration plays no role in this
    pipeline's c_t or x.

    Returns a dict with the same keys V3 reads:

      - ``a_fused``: the acoustic per-channel tokens (after the
        per-modality Set-Transformer self-attention pass), shape
        ``(B, N_ac, embed_dim)``.  Same as ``V2FusionEncoder.fuse_and_pool``'s
        first return value would be **if** the cross-attention were the
        identity — which is the cleanest "no cross-modal mixing" baseline.
      - ``v_fused``: a length-1 token of zeros, shape ``(B, 1, embed_dim)``.
        Length 1 (not 0) so V3's ``torch.cat([a_fused, v_fused], dim=1)``
        and the subsequent ``mean(dim=1)`` are well-defined (an empty
        sequence would produce NaN mean) but contributes 1 / (N_ac + 1)
        of the average, which the projection-head freezing makes harmless.
        Tests verify the mean-pool x is dominated by acoustic.
      - ``context``: the acoustic PMA summary, shape ``(B, embed_dim)``.

    `mask_p` is accepted for V2 API compatibility but ignored — V3
    consumes `out["context"]` and doesn't ask the encoder to mask LMM
    tokens (V3 runs in eval mode through the encoder).
    """

    def __init__(self, acoustic_encoder: PerModalityEncoder) -> None:
        super().__init__()
        if acoustic_encoder.modality != "acoustic":
            raise ValueError(
                f"V3AcousticOnlyAdapter requires modality='acoustic'; "
                f"got {acoustic_encoder.modality!r}"
            )
        self.encoder = acoustic_encoder
        self.embed_dim = acoustic_encoder.embed_dim

    @property
    def context_mode(self) -> str:  # for V3 trainer logging parity
        return "acoustic_only"

    def forward(
        self,
        ac_feat: torch.Tensor,
        ac_xyz: torch.Tensor,
        vib_feat: torch.Tensor,
        vib_xyz: torch.Tensor,
        dataset_idx: torch.Tensor,
        mask_p: float = 0.0,
        mask_generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        del vib_feat, vib_xyz, mask_p, mask_generator  # ignored
        tokens_a, summary_a = self.encoder(ac_feat, ac_xyz, dataset_idx)
        B = tokens_a.shape[0]
        zero_v = tokens_a.new_zeros((B, 1, self.embed_dim))
        return {
            "a_fused": tokens_a,
            "v_fused": zero_v,
            "context": summary_a,
            # Targets / summaries kept for API parity with V2 — V3 doesn't
            # read them but downstream eval helpers (e.g.
            # `_PairedSegment`-walking code) sometimes touch them.
            "a_target": tokens_a,
            "v_target": zero_v,
            "a_summary": summary_a,
            "v_summary": tokens_a.new_zeros((B, self.embed_dim)),
            "mask_a": torch.zeros(tokens_a.shape[:2], dtype=torch.bool, device=tokens_a.device),
            "mask_v": torch.zeros(zero_v.shape[:2], dtype=torch.bool, device=zero_v.device),
        }


class V3VibrationOnlyAdapter(nn.Module):
    """V2-API-compatible wrapper around a V1-vibration ``PerModalityEncoder``.

    Symmetric to ``V3AcousticOnlyAdapter`` — acoustic inputs are accepted
    and discarded; ``a_fused`` is a single zero-vector token, ``v_fused``
    is the per-channel vibration token sequence, ``context`` is the
    vibration PMA summary.
    """

    def __init__(self, vibration_encoder: PerModalityEncoder) -> None:
        super().__init__()
        if vibration_encoder.modality != "vibration":
            raise ValueError(
                f"V3VibrationOnlyAdapter requires modality='vibration'; "
                f"got {vibration_encoder.modality!r}"
            )
        self.encoder = vibration_encoder
        self.embed_dim = vibration_encoder.embed_dim

    @property
    def context_mode(self) -> str:
        return "vibration_only"

    def forward(
        self,
        ac_feat: torch.Tensor,
        ac_xyz: torch.Tensor,
        vib_feat: torch.Tensor,
        vib_xyz: torch.Tensor,
        dataset_idx: torch.Tensor,
        mask_p: float = 0.0,
        mask_generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        del ac_feat, ac_xyz, mask_p, mask_generator  # ignored
        tokens_v, summary_v = self.encoder(vib_feat, vib_xyz, dataset_idx)
        B = tokens_v.shape[0]
        zero_a = tokens_v.new_zeros((B, 1, self.embed_dim))
        return {
            "a_fused": zero_a,
            "v_fused": tokens_v,
            "context": summary_v,
            "a_target": zero_a,
            "v_target": tokens_v,
            "a_summary": tokens_v.new_zeros((B, self.embed_dim)),
            "v_summary": summary_v,
            "mask_a": torch.zeros(zero_a.shape[:2], dtype=torch.bool, device=zero_a.device),
            "mask_v": torch.zeros(tokens_v.shape[:2], dtype=torch.bool, device=tokens_v.device),
        }


__all__ = [
    "V3AcousticOnlyAdapter",
    "V3VibrationOnlyAdapter",
]
