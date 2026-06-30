"""Unit tests for the 2026-05-19 Attentive Statistics Pooling primitives.

Each test pins one invariant of `AttentiveStatsPool{1,2}d`:

  * **Shape invariance across input lengths** — ASP must collapse any
    `T'` (or `F' × T'`) to a fixed `(B, 2·C)` summary so the encoder is
    natively scale-invariant.  This is the load-bearing property that
    makes the multi-scale training + per-stage window override design
    work at all.
  * **Gradient flow** — every parameter must receive a non-zero gradient
    on a single backward pass, so the attention MLP can co-optimise
    with the rest of the encoder under SimCLR/V3 losses.
  * **Equivalence on flat input** — an all-equal feature map must give
    `σ = 0` and `μ = h`, so the pool's behaviour degenerates to the
    legacy mean-pool when the input has no temporal variation.  This
    is the regression check that catches future refactors silently
    breaking the std term.
  * **Parameter count** — locks in the ECAPA-style attention MLP shape
    (`C → C/r → 1`) so a future refactor doesn't inflate the pool's
    parameter budget without an audit.
"""

from __future__ import annotations

import pytest
import torch

from src.modeling.encoders.pooling import (
    AttentiveStatsPool1d,
    AttentiveStatsPool2d,
)

# ---------------------------------------------------------------------------
# Shape invariance across input lengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [10, 100, 1000])
def test_asp1d_shape_invariance_across_T(T: int) -> None:
    pool = AttentiveStatsPool1d(channels=128, reduction=8).eval()
    x = torch.randn(4, 128, T)
    out = pool(x)
    assert out.shape == (4, 2 * 128)
    assert torch.all(torch.isfinite(out))


@pytest.mark.parametrize(
    "Fr, T",
    [(8, 10), (8, 100), (8, 1000), (16, 500)],
)
def test_asp2d_shape_invariance_across_FT(Fr: int, T: int) -> None:
    pool = AttentiveStatsPool2d(channels=128, reduction=8).eval()
    x = torch.randn(4, 128, Fr, T)
    out = pool(x)
    assert out.shape == (4, 2 * 128)
    assert torch.all(torch.isfinite(out))


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_asp1d_gradients_flow_to_every_parameter() -> None:
    pool = AttentiveStatsPool1d(channels=64, reduction=8)
    x = torch.randn(2, 64, 50, requires_grad=False)
    out = pool(x)
    out.sum().backward()
    for name, p in pool.named_parameters():
        assert p.grad is not None, f"parameter {name!r} has no gradient"
        assert torch.any(p.grad != 0), f"parameter {name!r} has all-zero gradient"


def test_asp2d_gradients_flow_to_every_parameter() -> None:
    pool = AttentiveStatsPool2d(channels=64, reduction=8)
    x = torch.randn(2, 64, 8, 50, requires_grad=False)
    out = pool(x)
    out.sum().backward()
    for name, p in pool.named_parameters():
        assert p.grad is not None, f"parameter {name!r} has no gradient"
        assert torch.any(p.grad != 0), f"parameter {name!r} has all-zero gradient"


# ---------------------------------------------------------------------------
# Degeneracy: flat input ⇒ σ=0, μ=h
# ---------------------------------------------------------------------------


def test_asp1d_flat_input_gives_zero_sigma_and_constant_mu() -> None:
    pool = AttentiveStatsPool1d(channels=32, reduction=8).eval()
    # All-equal feature map at value 0.7 across (B, C, T).
    x = torch.full((3, 32, 64), 0.7)
    out = pool(x)
    # First 32 dims are μ — all 0.7.
    assert torch.allclose(out[:, :32], torch.full((3, 32), 0.7), atol=1e-5)
    # Last 32 dims are σ — clamped to eps^(1/2) ≈ 1e-3 on flat input,
    # but must be very small relative to μ.
    assert float(out[:, 32:].abs().max()) < 1e-2


def test_asp2d_flat_input_gives_zero_sigma_and_constant_mu() -> None:
    pool = AttentiveStatsPool2d(channels=32, reduction=8).eval()
    x = torch.full((3, 32, 8, 64), -0.3)
    out = pool(x)
    assert torch.allclose(out[:, :32], torch.full((3, 32), -0.3), atol=1e-5)
    assert float(out[:, 32:].abs().max()) < 1e-2


# ---------------------------------------------------------------------------
# Parameter count (locks the ECAPA-style attention-MLP shape)
# ---------------------------------------------------------------------------


def test_asp_parameter_count_at_c128_r8() -> None:
    """At C=128, r=8 the attention MLP is C → 16 → 1 with biases — 2225 params.

    Locks in the published ECAPA-style attention-MLP shape against silent
    refactors that would inflate the pool's parameter budget (e.g. if a
    future maintainer accidentally widens the bottleneck).
    """
    pool1d = AttentiveStatsPool1d(channels=128, reduction=8)
    pool2d = AttentiveStatsPool2d(channels=128, reduction=8)
    expected = (128 * 16 + 16) + (16 * 1 + 1)  # 2065
    n1 = sum(p.numel() for p in pool1d.parameters())
    n2 = sum(p.numel() for p in pool2d.parameters())
    assert n1 == expected
    assert n2 == expected


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_asp1d_rejects_non_positive_channels() -> None:
    with pytest.raises(ValueError, match="channels"):
        AttentiveStatsPool1d(channels=0)


def test_asp1d_rejects_non_positive_reduction() -> None:
    with pytest.raises(ValueError, match="reduction"):
        AttentiveStatsPool1d(channels=128, reduction=0)


def test_asp2d_rejects_wrong_input_rank() -> None:
    pool = AttentiveStatsPool2d(channels=8)
    with pytest.raises(ValueError, match=r"expects.*B, C, F, T"):
        pool(torch.randn(4, 8, 10))


def test_asp1d_rejects_wrong_input_rank() -> None:
    pool = AttentiveStatsPool1d(channels=8)
    with pytest.raises(ValueError, match=r"expects.*B, C, T"):
        pool(torch.randn(4, 8, 10, 5))
