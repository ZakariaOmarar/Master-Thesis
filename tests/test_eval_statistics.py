"""Smoke tests for `src/modeling/eval/statistics.py`.

These cover the small-sample machinery the thesis defence rests on:

  * **Percentile bootstrap CI on a mean** — must include the true mean
    on synthetic Gaussian data at the nominal coverage rate.
  * **Paired bootstrap test** — when A and B are dependent draws with a
    known positive Δ, the test must report `direction=A>B` with a CI
    that excludes 0 and a near-zero p-value.  When A and B are
    identical, the test must be inconclusive.
  * **Wilson-score CI on a proportion** — exact for the canonical
    "0 successes out of N" edge case (Wald CI fails here).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.modeling.eval import (
    binomial_proportion_ci,
    paired_bootstrap_test,
    percentile_bootstrap_ci,
)


def test_percentile_bootstrap_ci_covers_true_mean() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.16, 0.05, size=300)
    ci = percentile_bootstrap_ci(x, n_boot=1000, seed=0)
    # True mean (0.16) must be inside the CI on a 300-sample draw with σ=0.05.
    assert ci.ci_low < 0.16 < ci.ci_high
    # Point estimate must equal the sample mean within fp tolerance.
    assert abs(ci.point - float(x.mean())) < 1e-9
    # CI half-width should be on the order of σ/√n.
    half_width = 0.5 * (ci.ci_high - ci.ci_low)
    assert half_width < 0.02  # σ/√n ≈ 0.003 → CI half-width ≈ 0.006


def test_percentile_bootstrap_ci_degenerate_small_n() -> None:
    # n=1: point estimate fine, CI is degenerate (low == high).
    ci = percentile_bootstrap_ci(np.array([0.5]), n_boot=10, seed=0)
    assert ci.point == 0.5
    assert ci.ci_low == 0.5
    assert ci.ci_high == 0.5
    assert ci.n_boot == 0  # bootstrap not run on n=1


def test_paired_bootstrap_test_detects_known_positive_delta() -> None:
    """When B = A + ε with ε ~ N(0.02, 0.01), the paired test must
    report Δ = mean(A) − mean(B) ≈ −0.02 with CI excluding 0,
    direction = "A<B" (i.e. A is better under lower-is-better)."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.16, 0.05, size=300)
    b = a + rng.normal(0.02, 0.01, size=300)
    test = paired_bootstrap_test(a, b, n_boot=1000, seed=0)
    assert test.direction == "A<B"
    assert test.delta_ci_high < 0.0
    assert test.delta_point < 0.0
    assert test.p_value_two_sided < 0.01


def test_paired_bootstrap_test_inconclusive_when_a_equals_b() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.16, 0.05, size=300)
    b = a.copy()
    test = paired_bootstrap_test(a, b, n_boot=1000, seed=0)
    assert test.direction == "inconclusive"
    assert abs(test.delta_point) < 1e-9
    # Two-sided p-value should be exactly 1.0 when Δ = 0 deterministically.
    assert test.p_value_two_sided > 0.99


def test_paired_bootstrap_test_handles_higher_is_better() -> None:
    """For AUC-like metrics where higher is better, the direction
    labels invert."""
    rng = np.random.default_rng(0)
    auc_v3 = rng.uniform(0.85, 0.95, size=50)
    auc_a2 = auc_v3 - rng.uniform(0.05, 0.15, size=50)
    test = paired_bootstrap_test(
        auc_v3, auc_a2, lower_is_better=False, n_boot=500, seed=0,
    )
    # V3 has higher AUC → A (V3) > B (A2) → direction = "A>B"
    assert test.direction == "A>B"
    assert test.delta_point > 0.0
    assert test.delta_ci_low > 0.0


def test_wilson_proportion_ci_handles_zero_successes() -> None:
    """Wald CI mishandles the 0/N edge case (returns (0, 0)).  Wilson
    must give a non-degenerate upper bound."""
    p, lo, hi = binomial_proportion_ci(0, 100)
    assert p == 0.0
    assert lo < 1e-9  # Wilson lower bound is at the boundary modulo fp noise
    assert hi > 0.0  # Wilson gives a meaningful upper bound
    assert hi < 0.05  # but tight enough to be informative


def test_wilson_proportion_ci_matches_v3_healthy_rate() -> None:
    """Sanity: 85 successes in 1659 trials (the recent V3 healthy
    hold-out alert rate) gives p̂ = 0.0512 with Wilson CI tight around
    the construction target 5 %."""
    p, lo, hi = binomial_proportion_ci(85, 1659)
    assert abs(p - 85.0 / 1659.0) < 1e-9
    # Target rate (5 %) must lie inside the CI.
    assert lo < 0.05 < hi
    # CI width must be small (< 3 pp half-width on n=1659).
    half_width = 0.5 * (hi - lo)
    assert half_width < 0.03


def test_paired_bootstrap_test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap_test(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# Block (cluster) bootstrap — guards against window-level pseudoreplication.
# ---------------------------------------------------------------------------


def test_grouped_ci_is_wider_than_window_level_on_correlated_data() -> None:
    """Pseudoreplication demonstration: when 300 windows come from 5 recordings
    with strong between-recording variance and negligible within-recording
    variance, the naive window-level CI is falsely tight.  The recording-level
    block bootstrap (groups=) must produce a substantially wider, honest CI.
    The point estimate is identical (same statistic of the same data)."""
    rng = np.random.default_rng(0)
    group_means = rng.normal(0.16, 0.05, size=5)
    groups = np.repeat(np.arange(5), 60)
    values = np.concatenate([gm + rng.normal(0.0, 0.001, size=60) for gm in group_means])

    ci_window = percentile_bootstrap_ci(values, n_boot=1000, seed=0)
    ci_block = percentile_bootstrap_ci(values, n_boot=1000, seed=0, groups=groups)

    assert abs(ci_window.point - ci_block.point) < 1e-12  # same point estimate
    width_window = ci_window.ci_high - ci_window.ci_low
    width_block = ci_block.ci_high - ci_block.ci_low
    assert width_block > 3.0 * width_window  # honest CI is much wider
    assert ci_block.n_groups == 5
    assert ci_block.method == "grouped_percentile_bootstrap"
    assert ci_window.n_groups is None


def test_grouped_paired_test_preserves_sign_but_widens_interval() -> None:
    """A recording-consistent positive Δ (B always worse) must keep
    direction='A<B' under both resamplings, but the block bootstrap — which
    treats the recording, not the window, as the independent unit — must give
    a wider Δ interval than the pseudoreplicated window-level test."""
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(6), 40)
    a = rng.normal(0.16, 0.02, size=240)
    group_offsets = rng.uniform(0.01, 0.05, size=6)  # per-recording, B worse
    b = a + np.repeat(group_offsets, 40)

    res_w = paired_bootstrap_test(a, b, n_boot=1000, seed=0)
    res_g = paired_bootstrap_test(a, b, n_boot=1000, seed=0, groups=groups)

    assert res_w.direction == "A<B"
    assert res_g.direction == "A<B"  # conclusion (sign) preserved
    assert res_g.n_groups == 6
    assert res_g.method == "grouped_paired_percentile_bootstrap"
    width_w = res_w.delta_ci_high - res_w.delta_ci_low
    width_g = res_g.delta_ci_high - res_g.delta_ci_low
    assert width_g > width_w  # honest, recording-level interval is wider


def test_grouped_ci_single_group_is_degenerate() -> None:
    """One recording carries no between-group information → degenerate CI,
    not a falsely tight window-level one."""
    vals = np.array([0.1, 0.2, 0.3, 0.4])
    ci = percentile_bootstrap_ci(vals, groups=np.zeros(4, dtype=int), seed=0)
    assert ci.n_groups == 1
    assert ci.ci_low == ci.point == ci.ci_high
    assert ci.n_boot == 0


def test_grouped_paired_single_group_is_inconclusive() -> None:
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.2, 0.3, 0.4])
    res = paired_bootstrap_test(a, b, groups=np.zeros(3, dtype=int), seed=0)
    assert res.n_groups == 1
    assert res.direction == "inconclusive"
    assert np.isnan(res.p_value_two_sided)
    assert res.delta_point < 0.0  # point Δ still computed


def test_grouped_bootstrap_raises_on_group_length_mismatch() -> None:
    with pytest.raises(ValueError, match="must match"):
        percentile_bootstrap_ci(np.array([1.0, 2.0, 3.0]), groups=np.array([0, 1]))
