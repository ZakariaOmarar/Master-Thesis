"""V2 SSL projection head, NT-Xent contrastive loss, and latent masked-modeling loss."""


import torch
import torch.nn as nn
import torch.nn.functional as F


class _ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    B = z1.shape[0]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.t()) / temperature
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)
    return F.cross_entropy(sim, targets)
def _lmm_loss(
    fused: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Cosine-similarity loss between fused-token output and pre-mask target,
    averaged over masked positions only.  Returns 0.0 when no position is
    masked in the batch (so the gradient simply skips this term).
    """
    if mask is None or not mask.any():
        return torch.zeros((), device=fused.device, dtype=fused.dtype)
    p = F.normalize(fused[mask], dim=-1)
    t = F.normalize(target[mask], dim=-1)
    return (1.0 - (p * t).sum(-1)).mean()


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
