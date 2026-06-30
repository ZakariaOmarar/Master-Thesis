"""Statistical utilities shared across the evaluation surface.

These are the canonical small-sample tests the thesis defence rests on.
Each function is anchored to a citation a reviewer will recognise, and
each docstring states (a) what is being tested, (b) why this particular
form is appropriate for our data scale, and (c) the master's-thesis
question the result answers.

Functions:

  * `percentile_bootstrap_ci`  — single-sample percentile-bootstrap CI on
     an arbitrary scalar statistic of a 1-D error array.  Used for V4 MAE
     uncertainty; identical to the inline implementation already living in
     `v4_trainer.py` and factored out here so other modules can reuse it.

  * `paired_bootstrap_test`    — paired bootstrap on Δ = stat(A) − stat(B)
     for two metric arrays evaluated on the *same* underlying samples
     (V3 vs A2 NLL per held-out window; V4 vs A3 Euclidean error per
     val window).  Returns the CI on Δ and a bootstrap-style two-sided
     p-value.  This is the rigorous answer to "is the conditioning win
     statistically meaningful?"

  * `binomial_proportion_ci`   — Wilson-score interval on a proportion
     (used for V3 alert rates so the healthy hold-out rate is reported as
     `5.1 % ± Wilson CI` rather than a point estimate).

Citations:
  - Efron, B. (1979). "Bootstrap methods: Another look at the
    jackknife." *Annals of Statistics* 7(1).
  - Efron, B. & Tibshirani, R. (1993). *An Introduction to the
    Bootstrap*.  Chapman & Hall.  §6 (single-sample) and §16 (paired).
  - Wilson, E. B. (1927). "Probable inference, the law of succession,
    and statistical inference."  *J. Am. Stat. Assoc.* 22(158).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np


def _group_index_lists(groups: np.ndarray, n: int) -> list[np.ndarray]:
    """Return one index array per unique group label, for the block bootstrap.

    The *cluster* (a.k.a. block) bootstrap resamples whole groups with
    replacement and pools every member of each chosen group.  This is the
    correct resampling unit when the elementary observations are *not*
    independent — e.g. per-window errors that share a recording, or per-knock
    NLLs from the same physical position.  Resampling the windows directly
    (the naive bootstrap) treats correlated observations as independent draws,
    which is statistical *pseudoreplication*: it overstates the effective
    sample size and produces anti-conservative (too narrow) intervals and
    too-small p-values (Davison & Hinkley 1997, §3.8; Cameron et al. 2008,
    "Bootstrap-based improvements for inference with clustered errors").
    """
    groups = np.asarray(groups).reshape(-1)
    if groups.shape[0] != n:
        raise ValueError(
            f"groups length {groups.shape[0]} must match values length {n}"
        )
    uniq = np.unique(groups)
    return [np.flatnonzero(groups == g) for g in uniq]


@dataclass(frozen=True)
class BootstrapCI:
    """Result of a percentile-bootstrap CI computation."""

    point: float
    ci_low: float
    ci_high: float
    n_boot: int
    method: str = "percentile_bootstrap"
    n_groups: int | None = None


def percentile_bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = lambda x: float(np.mean(x)),
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> BootstrapCI:
    """Percentile bootstrap CI on a scalar statistic of a 1-D sample.

    Why percentile bootstrap (Efron & Tibshirani 1993, §13): we cannot
    assume normality of the underlying error distribution on a 3-D MAE
    over ~ 300 windows from ~ 3 recordings — the per-recording
    correlation structure is unknown and the sample is small.  The
    percentile bootstrap is the simplest non-parametric CI that respects
    whatever distributional shape the data actually has.

    For the V4 MAE use case this answers "given the val errors we
    observed, what is the range of MAE values consistent with them at
    95 % confidence?"

    Args:
      values: 1-D array of sample values (e.g. per-window 3-D Euclidean
        errors for V4 MAE).
      statistic: scalar functional; default is the arithmetic mean.
      n_boot: number of bootstrap resamples.  1000 is the small-sample
        standard; for publication numbers Davison & Hinkley (1997) §5
        recommend ≥ 1000.
      alpha: two-tailed confidence level (default 0.05 → 95 % CI).
      seed: RNG seed for reproducibility.
      groups: optional per-observation group label (same length as
        ``values``).  When given, a **block (cluster) bootstrap** resamples
        whole groups with replacement instead of individual observations —
        the statistically correct unit when observations within a group are
        correlated (per-window errors sharing a recording, per-knock errors
        sharing a position).  The naive window-level bootstrap (``groups=None``)
        is *pseudoreplicated* on such data and yields anti-conservative
        intervals; pass the recording/position id here to get an honest CI.
        With fewer than two distinct groups the CI is degenerate (a single
        block carries no between-group information).
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(values.shape[0])
    if n < 2:
        point = float(statistic(values)) if n == 1 else float("nan")
        return BootstrapCI(point=point, ci_low=point, ci_high=point, n_boot=0)
    rng = np.random.default_rng(int(seed))

    if groups is not None:
        block_idx = _group_index_lists(groups, n)
        n_groups = len(block_idx)
        point = float(statistic(values))
        if n_groups < 2:
            # One block ⇒ no between-group variability to resample.  Report the
            # point estimate with a degenerate interval rather than a falsely
            # tight window-level one.
            return BootstrapCI(
                point=point, ci_low=point, ci_high=point, n_boot=0,
                method="grouped_percentile_bootstrap", n_groups=n_groups,
            )
        boots = np.empty(int(n_boot), dtype=np.float64)
        for i in range(int(n_boot)):
            chosen = rng.integers(0, n_groups, size=n_groups)
            idx = np.concatenate([block_idx[g] for g in chosen])
            boots[i] = float(statistic(values[idx]))
        low = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
        high = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
        return BootstrapCI(
            point=point, ci_low=low, ci_high=high, n_boot=int(n_boot),
            method="grouped_percentile_bootstrap", n_groups=n_groups,
        )

    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(statistic(values[idx]))
    low = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return BootstrapCI(
        point=float(statistic(values)),
        ci_low=low,
        ci_high=high,
        n_boot=int(n_boot),
    )


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Result of a paired bootstrap test on Δ = stat(A) − stat(B)."""

    delta_point: float
    delta_ci_low: float
    delta_ci_high: float
    p_value_two_sided: float
    n_boot: int
    direction: Literal["A<B", "A>B", "inconclusive"]
    method: str = "paired_percentile_bootstrap"
    n_groups: int | None = None


def paired_bootstrap_test(
    a_values: np.ndarray,
    b_values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = lambda x: float(np.mean(x)),
    *,
    lower_is_better: bool = True,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> PairedBootstrapResult:
    """Paired bootstrap on Δ = statistic(A) − statistic(B) when A and B
    are evaluated on the same underlying samples (paired observations).

    Used twice in this thesis:

    * **V3 vs A2 NLL** (RQ2 significance test).  For each held-out
      healthy window i, we have `nll_V3[i]` and `nll_A2[i]` from the
      conditional and unconditional flow respectively, scored on the
      *same* window.  Δ = mean NLL gap is the metric of interest;
      the paired bootstrap quantifies its 95 % CI.

    * **V4 vs A3 3-D MAE** (RQ3 significance test).  For each val
      window i, we have `err_V4[i] = ‖pred_V4 - target‖` and
      `err_A3[i] = ‖pred_A3 - target‖` from the conditional and
      unconditional V4 head, on the *same* window.

    The paired form (resample window indices, not the metric values
    independently) is the correct test because the two metric arrays
    are not independent: both are computed from the same window's
    features.  Independence-assuming unpaired tests (e.g. Welch's t)
    would inflate the apparent uncertainty.

    Returns 95 % CI on Δ and a two-sided p-value defined as
    ``2 * min(P(Δ_boot ≤ 0), P(Δ_boot ≥ 0))`` (Davison & Hinkley
    1997 §5.4) — the bootstrap analogue of a paired t-test p-value.

    Args:
      a_values: per-sample metric for system A (e.g. V4 errors).
      b_values: per-sample metric for system B (e.g. A3 errors).
        Must have the same length as `a_values`; index i in both refers
        to the same underlying sample.
      statistic: scalar functional; default is the mean.
      lower_is_better: when True (default), reports `direction = "A<B"`
        if Δ ≤ 0 with CI excluding 0 → A wins.  Set False for higher-
        is-better metrics (e.g. AUC).
      n_boot: number of bootstrap resamples.
      alpha: two-tailed confidence level (default 0.05).
      seed: RNG seed.
      groups: optional per-observation group label (same length as the
        value arrays).  When given, a **block (cluster) bootstrap** resamples
        whole groups (e.g. recordings, positions) with replacement rather than
        individual paired windows.  This is the correct unit when the paired
        observations are autocorrelated within a group — e.g. the V3-vs-A2
        per-window NLLs that share a recording, or the V4-vs-A3 per-knock
        errors that share a position.  The naive ``groups=None`` test treats
        every window as an independent paired draw, which is pseudoreplication
        and inflates significance (understates the p-value).  Pass the
        recording/position id to obtain an honest, recording-level test.
        With fewer than two distinct groups the CI/p-value are undefined and
        the result is reported ``inconclusive``.
    """
    a = np.asarray(a_values, dtype=np.float64).reshape(-1)
    b = np.asarray(b_values, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(
            f"paired_bootstrap_test requires equal-length arrays; "
            f"got A:{a.shape} vs B:{b.shape}"
        )
    n = int(a.shape[0])
    if n < 2:
        return PairedBootstrapResult(
            delta_point=float("nan"),
            delta_ci_low=float("nan"),
            delta_ci_high=float("nan"),
            p_value_two_sided=float("nan"),
            n_boot=0,
            direction="inconclusive",
        )

    delta_point = float(statistic(a)) - float(statistic(b))
    rng = np.random.default_rng(int(seed))

    n_groups: int | None = None
    method = "paired_percentile_bootstrap"
    if groups is not None:
        block_idx = _group_index_lists(groups, n)
        n_groups = len(block_idx)
        method = "grouped_paired_percentile_bootstrap"
        if n_groups < 2:
            # A single block carries no between-group information; the
            # recording-level test is undefined.  Report the point estimate
            # but flag the result inconclusive rather than fabricate a CI.
            return PairedBootstrapResult(
                delta_point=delta_point,
                delta_ci_low=float("nan"),
                delta_ci_high=float("nan"),
                p_value_two_sided=float("nan"),
                n_boot=0,
                direction="inconclusive",
                method=method,
                n_groups=n_groups,
            )
        boots = np.empty(int(n_boot), dtype=np.float64)
        for i in range(int(n_boot)):
            chosen = rng.integers(0, n_groups, size=n_groups)
            idx = np.concatenate([block_idx[g] for g in chosen])
            boots[i] = float(statistic(a[idx])) - float(statistic(b[idx]))
    else:
        boots = np.empty(int(n_boot), dtype=np.float64)
        for i in range(int(n_boot)):
            idx = rng.integers(0, n, size=n)
            boots[i] = float(statistic(a[idx])) - float(statistic(b[idx]))
    low = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))

    # Two-sided bootstrap p-value (Davison & Hinkley 1997 §5.4).
    p_le_0 = float((boots <= 0.0).mean())
    p_ge_0 = float((boots >= 0.0).mean())
    p_two = 2.0 * min(p_le_0, p_ge_0)
    p_two = min(1.0, max(0.0, p_two))

    if lower_is_better:
        if high < 0.0:
            direction: Literal["A<B", "A>B", "inconclusive"] = "A<B"
        elif low > 0.0:
            direction = "A>B"
        else:
            direction = "inconclusive"
    else:
        # higher_is_better: A wins iff Δ = stat(A) − stat(B) > 0 with CI excluding 0
        if low > 0.0:
            direction = "A>B"  # in higher-is-better, A > B means A is better
        elif high < 0.0:
            direction = "A<B"
        else:
            direction = "inconclusive"

    return PairedBootstrapResult(
        delta_point=delta_point,
        delta_ci_low=low,
        delta_ci_high=high,
        p_value_two_sided=p_two,
        n_boot=int(n_boot),
        direction=direction,
        method=method,
        n_groups=n_groups,
    )


def binomial_proportion_ci(
    successes: int,
    n: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Wilson-score CI on a binomial proportion.

    Used for V3 alert-rate reporting.  Wilson's interval (1927) is the
    standard small-sample CI for a proportion; it correctly handles the
    edge cases where the point estimate is at 0 or 100 % (which a naive
    Wald interval mishandles).

    Returns ``(p_hat, lower, upper)``.  For the V3 healthy hold-out
    case, n is large (~ 1600 windows) and p_hat ≈ 0.05, so the Wilson
    interval is narrow and the calibration target (5.0 %) sits well
    within it.  On the smaller D3-hit cohort (n = 37) the Wilson CI is
    appreciably wider and the reporting acknowledges this.
    """
    from scipy.stats import norm

    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p_hat = successes / n
    z = float(norm.ppf(1.0 - alpha / 2.0))
    denom = 1.0 + (z ** 2) / n
    centre = (p_hat + (z ** 2) / (2.0 * n)) / denom
    half_width = (z * np.sqrt(p_hat * (1.0 - p_hat) / n + (z ** 2) / (4.0 * n ** 2))) / denom
    lower = max(0.0, centre - half_width)
    upper = min(1.0, centre + half_width)
    return float(p_hat), float(lower), float(upper)


__all__ = [
    "BootstrapCI",
    "PairedBootstrapResult",
    "binomial_proportion_ci",
    "paired_bootstrap_test",
    "percentile_bootstrap_ci",
]
