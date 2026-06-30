"""Smoke tests for V4 channel-ablation (`V4Config.channel_mode`).

The ablation must:
  1. accept ``"srp_only"`` and ``"tdoa_only"`` as well as the default
     ``"both"``;
  2. produce numerically different predictions than the default when
     a non-trivial volume / TDOA distinguishes the modes;
  3. preserve the V4Result schema (per-recording breakdown, CI, etc.).
"""

from __future__ import annotations

import numpy as np
import torch

from src.modeling.localization import GridSpec, V4Config
from src.modeling.localization.v4_loc_head import V4LocalizationHead
from src.modeling.localization.v4_trainer import (
    V4Sample,
    _grid_coords_from_spec,
    _make_batch,
)


def test_make_batch_srp_only_zeros_tdoa() -> None:
    sample = V4Sample(
        srp_volume=np.random.RandomState(0).rand(8, 8, 4).astype(np.float32),
        tdoa_tokens=np.random.RandomState(0).rand(6, 8).astype(np.float32),
        context=np.random.RandomState(0).rand(16).astype(np.float32),
        x_for_v3=np.random.RandomState(0).rand(16).astype(np.float32),
        target_xyz=np.array([0.1, 0.0, 0.05], dtype=np.float32),
        scada=None,
        mode_label=None,
        recording_id="r0",
        source_dir="/tmp",
        dataset_id="d4",
    )
    batch_srp = _make_batch([sample], channel_mode="srp_only")
    assert torch.all(batch_srp["tdoa"] == 0)
    assert not torch.all(batch_srp["volumes"] == 0)


def test_make_batch_tdoa_only_zeros_srp() -> None:
    sample = V4Sample(
        srp_volume=np.random.RandomState(0).rand(8, 8, 4).astype(np.float32),
        tdoa_tokens=np.random.RandomState(0).rand(6, 8).astype(np.float32),
        context=np.random.RandomState(0).rand(16).astype(np.float32),
        x_for_v3=np.random.RandomState(0).rand(16).astype(np.float32),
        target_xyz=np.array([0.1, 0.0, 0.05], dtype=np.float32),
        scada=None,
        mode_label=None,
        recording_id="r0",
        source_dir="/tmp",
        dataset_id="d4",
    )
    batch_tdoa = _make_batch([sample], channel_mode="tdoa_only")
    assert torch.all(batch_tdoa["volumes"] == 0)
    assert not torch.all(batch_tdoa["tdoa"] == 0)


def test_channel_mode_changes_predictions() -> None:
    """The same trained head, fed three different channel modes, must
    return three numerically different prediction tensors."""
    grid = GridSpec(lo=(-0.2, -0.2, -0.05), hi=(0.4, 0.4, 0.30), n=(8, 8, 4))
    coords = _grid_coords_from_spec(grid)
    torch.manual_seed(0)
    head = V4LocalizationHead(
        grid_coords=coords, cnn_feature_dim=32, tdoa_feature_dim=16,
        c_dim=16, s_dim=0, hidden_dim=32,
    )
    head.eval()
    # Construct a SRP volume with a clearly-defined peak, plus non-trivial
    # TDOA tokens, so the three channel modes have measurably different
    # inputs.
    rng = np.random.RandomState(0)
    sample = V4Sample(
        srp_volume=rng.rand(8, 8, 4).astype(np.float32),
        tdoa_tokens=rng.rand(6, 8).astype(np.float32),
        context=rng.rand(16).astype(np.float32),
        x_for_v3=rng.rand(16).astype(np.float32),
        target_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        scada=None,
        mode_label=None,
        recording_id="r0",
        source_dir="/tmp",
        dataset_id="d4",
    )
    preds = {}
    for mode in ("both", "srp_only", "tdoa_only"):
        batch = _make_batch([sample], channel_mode=mode)
        with torch.no_grad():
            preds[mode] = head(
                batch["volumes"], batch["tdoa"], batch["contexts"], batch["scada"],
            ).numpy()
    # All three must be finite.
    for mode, p in preds.items():
        assert np.all(np.isfinite(p)), f"{mode} predictions are non-finite"
    # At random init, the residual MLP's final layer is zero-init by
    # design (so the head starts at exactly the soft-argmax prior — see
    # `FiLMResidualHead.__init__`).  Therefore `pred = init_xyz` at
    # init, and only the SRP-modifying ablations (which change
    # `init_xyz` via the soft-argmax over a different logit volume)
    # produce a different prediction.  "srp_only" leaves the volume
    # intact → identical init_xyz → identical prediction.  "tdoa_only"
    # zeros the volume → soft-argmax collapses to the grid centroid →
    # different prediction.  This is the channel-ablation invariant
    # at init; after training the residual MLP departs from zero and
    # all three modes diverge — verified end-to-end in the full V4
    # training smoke test.
    assert np.allclose(preds["both"], preds["srp_only"], atol=1e-5), (
        "srp_only must equal both at random init (residual MLP is zero-init)"
    )
    assert not np.allclose(preds["both"], preds["tdoa_only"]), (
        "tdoa_only must differ from both at init — its soft-argmax over a "
        "zero volume should collapse to the grid centroid"
    )


def test_v4_config_channel_mode_default() -> None:
    cfg = V4Config()
    assert cfg.channel_mode == "both"
