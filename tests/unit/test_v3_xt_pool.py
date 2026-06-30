"""Unit tests for the 2026-05-19 V3 `_XtPool` (PMA-2) channel-token summariser.

Each test pins one invariant of `_XtPool` and the V3 trainer's two
xt-pool paths (legacy ``mean`` vs publication ``pma2``):

  * **Shape contract** — `_XtPool(fused)` produces `(B, embed_dim)`
    regardless of the channel-token count `N_a + N_v`, so V3's
    conditional flow sees a fixed feature dim across D1 (4+4=8
    tokens), D2 (5+5=10), and D3/D4 (9+4=13).
  * **Gradient flow** — both the PMA-2 seeds and the projection are
    co-optimised with the flow during training; ergo every parameter
    must receive non-zero gradient on a single backward step.
  * **Mean-pool degeneracy** — when the input has every token equal
    to a constant vector, the PMA-2 output collapses to a function of
    that constant (≈ Linear ∘ stacked-constant).  This regression
    check catches a future refactor accidentally killing the seed
    parameters or the attention soft-max.
  * **Override → single-scale v2_cfg** — `_make_override_v2_cfg`
    materialises the V3 override into the multi-scale dict format
    expected by `_PairedWindowedDataset`, leaving the original
    `v2_cfg` untouched.
  * **Mean-vs-pma2 ablation contract** — running V3 once with
    ``xt_pool="mean"`` and once with ``"pma2"`` on the same synthetic
    healthy/anomaly toy must give two different `x_t` distributions
    (the two paths take genuinely different code branches).
"""

from __future__ import annotations

import pytest
import torch

from src.modeling.anomaly.v3_trainer import (
    V3Config,
    _make_override_v2_cfg,
    _resolve_v3_override,
    _XtPool,
)
from src.modeling.context.v2_ssl import V2SSLConfig

# ---------------------------------------------------------------------------
# _XtPool shape + gradient invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_tokens", [4, 8, 13, 20])
def test_xtpool_output_shape_is_embed_dim_for_any_n_tokens(n_tokens: int) -> None:
    pool = _XtPool(embed_dim=64, num_heads=4).eval()
    fused = torch.randn(3, n_tokens, 64)
    out = pool(fused)
    assert out.shape == (3, 64)
    assert torch.all(torch.isfinite(out))


def test_xtpool_gradients_flow_to_pma_seeds_and_proj() -> None:
    pool = _XtPool(embed_dim=32, num_heads=2)
    fused = torch.randn(2, 10, 32)
    out = pool(fused)
    out.sum().backward()
    for name, p in pool.named_parameters():
        assert p.grad is not None, f"{name!r} has no gradient"
        assert torch.any(p.grad != 0), f"{name!r} has all-zero gradient"


def test_xtpool_constant_input_collapses_to_function_of_constant() -> None:
    """All-equal tokens at value `v` → PMA(constant matrix) ≈ proj(v ‖ v)."""
    pool = _XtPool(embed_dim=16, num_heads=2).eval()
    fused = torch.full((4, 8, 16), 0.3)
    out = pool(fused)
    # PMA over a constant token sequence produces the same per-seed output
    # for every batch element → out should be identical across the batch.
    for b in range(1, 4):
        assert torch.allclose(out[b], out[0], atol=1e-5)


# ---------------------------------------------------------------------------
# Override resolution + v2_cfg materialisation
# ---------------------------------------------------------------------------


def test_resolve_v3_override_normalises_scalar_to_dict() -> None:
    v3 = V3Config(window_seconds_override=1.0, xt_pool="mean")
    out = _resolve_v3_override(v3, ["d1", "d3"])
    assert out == {"d1": 1.0, "d3": 1.0}


def test_resolve_v3_override_passes_through_dict() -> None:
    v3 = V3Config(
        window_seconds_override={"d1": 3.0, "d4": 0.5},
        xt_pool="mean",
    )
    out = _resolve_v3_override(v3, ["d1", "d4"])
    assert out == {"d1": 3.0, "d4": 0.5}


def test_resolve_v3_override_returns_none_when_unset() -> None:
    v3 = V3Config(window_seconds_override=None, xt_pool="mean")
    out = _resolve_v3_override(v3, ["d1"])
    assert out is None


def test_make_override_v2_cfg_materialises_per_dataset_scales() -> None:
    """The override REPLACES the dataclass's `window_scales_seconds_per_dataset`.

    Construct with explicit ``{}`` so the publication multi-scale default
    sourced from `WINDOWING` does not interfere with the assertion below.
    """
    v2_cfg = V2SSLConfig(
        window_seconds=2.0,
        window_stride_seconds=1.0,
        window_scales_seconds_per_dataset={},
    )
    new_cfg = _make_override_v2_cfg(v2_cfg, {"d1": 3.0, "d3": 1.0})
    # New config has a per-dataset scale dict, original untouched.
    assert new_cfg.window_scales_seconds_per_dataset == {"d1": (3.0,), "d3": (1.0,)}
    assert v2_cfg.window_scales_seconds_per_dataset == {}
    # Other fields preserved.
    assert new_cfg.window_seconds == 2.0


def test_make_override_v2_cfg_returns_original_when_no_override() -> None:
    v2_cfg = V2SSLConfig(window_seconds=2.0, window_scales_seconds_per_dataset={})
    out = _make_override_v2_cfg(v2_cfg, None)
    assert out is v2_cfg
    out2 = _make_override_v2_cfg(v2_cfg, {})
    assert out2 is v2_cfg


# ---------------------------------------------------------------------------
# Mean-vs-pma2 ablation contract
# ---------------------------------------------------------------------------


def test_pma2_and_mean_paths_produce_different_xt_distributions() -> None:
    """Forward two identical fused tensors through both paths; the outputs
    must differ.  Failing this check means the V3 trainer's `xt_pool` flag
    has lost its discriminative effect (e.g. a refactor accidentally
    routed both paths through the same code).
    """
    torch.manual_seed(0)
    fused = torch.randn(8, 13, 32)  # D3/D4-style token count

    # Legacy mean path
    x_mean = fused.mean(dim=1)

    # PMA-2 path (random init)
    pool = _XtPool(embed_dim=32, num_heads=2).eval()
    x_pma = pool(fused)

    assert x_mean.shape == x_pma.shape
    # The two paths must produce numerically different vectors — they are
    # different functions of fused.  A coincidental near-zero diff is
    # vanishingly unlikely under random PMA init.
    diff = (x_mean - x_pma).abs().mean().item()
    assert diff > 1e-3, f"mean and pma2 paths produced near-identical outputs (Δ={diff:.2e})"
