"""Evaluation utilities shared across the V0–V5 stages.

Currently exposes the small-sample statistical machinery the thesis
defence rests on: percentile-bootstrap CIs and paired bootstrap tests
(Efron & Tibshirani 1993), plus Wilson-score CIs on proportions
(Wilson 1927) for V3 alert-rate reporting.
"""

from .statistics import (
    BootstrapCI,
    PairedBootstrapResult,
    binomial_proportion_ci,
    paired_bootstrap_test,
    percentile_bootstrap_ci,
)

__all__ = [
    "BootstrapCI",
    "PairedBootstrapResult",
    "binomial_proportion_ci",
    "paired_bootstrap_test",
    "percentile_bootstrap_ci",
]
