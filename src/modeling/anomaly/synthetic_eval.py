"""V3 synthetic-anomaly ROC-AUC — controlled validation for RQ2.

The field-collection protocol does not provide per-window anomaly
labels (Chapter 3 §3.4.6), so V3 is currently validated via per-cohort
alert rates (qualitative ranking) and held-out healthy alert rate
(quantitative calibration target ≈ 5 %).  Neither produces a
classical ROC curve.

This module closes that gap by constructing a **controlled
synthetic-anomaly cohort**: held-out healthy paired-feature windows
are corrupted with additive Gaussian noise at calibrated
signal-to-noise ratios, producing window-level ground-truth labels
(clean = 0, corrupted = 1) on which the V3 anomaly score
``s_t = −log p(x | c)`` is a real classifier.  The resulting
ROC-AUC is the standard small-sample-AD evaluation curve
(Kawaguchi et al., DCASE 2021; Khamaisi et al. 2025 ROC AUC ≈ 0.998
for OC-SVM at ROW II were computed on a similar synthetic
construction).

Why feature-space noise rather than waveform-space noise
---------------------------------------------------------

Two reasons:

1. **Cost**: V3 consumes V2-encoded features ``(x, c)``.  Corrupting
   the waveform requires re-running V2 forward on the corrupted
   waveform; corrupting the features lets the same V2 encoder pass
   the clean and the noisy view, so AUC is computed on V3 alone and
   does not conflate V2 robustness with V3 sensitivity.

2. **Interpretation**: a Gaussian perturbation on the V2 latent is
   precisely the operational definition of "a healthy window pushed
   off the healthy manifold by a fixed Euclidean amount".  V3's
   anomaly score should rise monotonically with that perturbation;
   AUC quantifies how reliably it does so.

The trade-off is that this AUC measures **V3's sensitivity to
off-manifold perturbations in latent space**, not its sensitivity
to real-world acoustic / vibrational anomalies.  The latter is
already validated by the per-cohort alert rates (D3 hit 78 %,
D2 RF 59 %, D4 RF 11 % vs healthy 5 %).  The two evaluations are
complementary, not redundant.

Citation
--------

* Kawaguchi, Y. et al. (2021). "Description and discussion on
  DCASE 2021 challenge task 2: unsupervised anomalous sound
  detection for machine condition monitoring under domain
  shifted conditions."
* Khamaisi, S. et al. (2025).  "Noise: Acoustic Anomaly Detection
  on Hydroelectric Plants at ROW II."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...config import resolve_device
from .cnf_head import ConditionalRealNVP


@dataclass
class SyntheticAnomalyAUC:
    """Result of a synthetic-anomaly ROC-AUC evaluation.

    ``snr_db_to_auc`` is the headline curve: ROC-AUC of the V3 anomaly
    score as a discriminator between clean held-out healthy windows
    and synthetically-corrupted versions of the same windows at the
    indicated SNR (lower SNR ⇒ louder noise ⇒ easier discrimination).
    """

    snr_db_to_auc: dict[float, float]
    snr_db_to_n_clean: dict[float, int]
    snr_db_to_n_corrupted: dict[float, int]
    snr_db_to_auc_ci_low: dict[float, float]
    snr_db_to_auc_ci_high: dict[float, float]
    seed: int


def _roc_auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Mann-Whitney U-statistic form of ROC-AUC.

    Why this form: it is exact (no histogram binning), handles ties
    correctly via 0.5 weighting, and matches sklearn's
    ``roc_auc_score`` to machine precision while not adding a sklearn
    dependency to this module.

    Reference: Hanley & McNeil (1982), "The meaning and use of the
    area under a receiver operating characteristic (ROC) curve."
    *Radiology* 143(1).
    """
    pos = np.asarray(scores_pos, dtype=np.float64).reshape(-1)
    neg = np.asarray(scores_neg, dtype=np.float64).reshape(-1)
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Combined ranks; tie-aware.
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)
    # Average ranks within tie groups.
    sorted_vals = combined[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos_ranks = float(ranks[:n_pos].sum())
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def evaluate_synthetic_anomaly_auc(
    flow: ConditionalRealNVP,
    healthy_x: np.ndarray,
    healthy_c: np.ndarray,
    *,
    snr_db_list: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0),
    n_boot: int = 1000,
    seed: int = 0,
    device: torch.device | str = "auto",
) -> SyntheticAnomalyAUC:
    """Compute V3 ROC-AUC against feature-space synthetic anomalies.

    Args:
      flow: trained V3 ConditionalRealNVP.
      healthy_x: ``(N, D)`` clean V2 mean-pool features (V3's `x` input)
        for held-out healthy windows.  Source: `V3Result.val_*` or the
        cohort-time forward pass.
      healthy_c: ``(N, C)`` matching V2 PMA context vectors.
      snr_db_list: per-window SNR ladder.  Noise scale at each SNR is
        ``σ_noise = σ_x · 10^(-SNR_dB / 20)`` where ``σ_x`` is the
        empirical standard deviation of `healthy_x` over the window
        and feature axes — i.e. SNR is defined in **latent-space dB**,
        which is what we mean by "off-manifold perturbation of
        controlled magnitude".  Lower SNR ⇒ larger perturbation.
      n_boot: bootstrap resamples for the AUC CI.
      seed: RNG seed for reproducible noise realisations.
      device: torch device for the flow forward pass.

    Returns:
      `SyntheticAnomalyAUC` with the AUC and 95 % CI per SNR.
    """
    device = resolve_device(device)
    flow = flow.to(device).eval()
    rng = np.random.default_rng(int(seed))

    healthy_x = np.asarray(healthy_x, dtype=np.float64)
    healthy_c = np.asarray(healthy_c, dtype=np.float64)
    if healthy_x.shape[0] != healthy_c.shape[0]:
        raise ValueError(
            f"healthy_x and healthy_c must have the same N; "
            f"got {healthy_x.shape[0]} vs {healthy_c.shape[0]}"
        )
    if healthy_x.shape[0] < 4:
        return SyntheticAnomalyAUC(
            snr_db_to_auc={s: float("nan") for s in snr_db_list},
            snr_db_to_n_clean={s: int(healthy_x.shape[0]) for s in snr_db_list},
            snr_db_to_n_corrupted=dict.fromkeys(snr_db_list, 0),
            snr_db_to_auc_ci_low={s: float("nan") for s in snr_db_list},
            snr_db_to_auc_ci_high={s: float("nan") for s in snr_db_list},
            seed=int(seed),
        )

    # σ_x measured over the whole healthy pool, treated as scalar.  This
    # is the relevant per-feature noise scale that "1 dB of SNR" maps
    # to in latent space.  Using a single scalar (rather than per-dim
    # σ) keeps the noise isotropic — the cleanest "off-manifold push"
    # interpretation.
    sigma_x = float(np.std(healthy_x))
    if sigma_x <= 0.0:
        return SyntheticAnomalyAUC(
            snr_db_to_auc={s: float("nan") for s in snr_db_list},
            snr_db_to_n_clean={s: int(healthy_x.shape[0]) for s in snr_db_list},
            snr_db_to_n_corrupted=dict.fromkeys(snr_db_list, 0),
            snr_db_to_auc_ci_low={s: float("nan") for s in snr_db_list},
            snr_db_to_auc_ci_high={s: float("nan") for s in snr_db_list},
            seed=int(seed),
        )

    # Clean scores: V3 NLL on the held-out healthy pool.
    with torch.no_grad():
        clean_scores = (
            flow.anomaly_score(
                torch.from_numpy(healthy_x.astype(np.float32)).to(device),
                torch.from_numpy(healthy_c.astype(np.float32)).to(device),
            )
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    auc_per_snr: dict[float, float] = {}
    n_clean: dict[float, int] = {}
    n_corr: dict[float, int] = {}
    ci_low_per_snr: dict[float, float] = {}
    ci_high_per_snr: dict[float, float] = {}

    for snr_db in snr_db_list:
        # σ_noise = σ_x · 10^(-SNR_dB / 20)
        sigma_noise = sigma_x * (10.0 ** (-float(snr_db) / 20.0))
        noise = rng.normal(0.0, sigma_noise, size=healthy_x.shape)
        corrupted_x = healthy_x + noise
        with torch.no_grad():
            corrupted_scores = (
                flow.anomaly_score(
                    torch.from_numpy(corrupted_x.astype(np.float32)).to(device),
                    torch.from_numpy(healthy_c.astype(np.float32)).to(device),
                )
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        auc = _roc_auc(corrupted_scores, clean_scores)
        auc_per_snr[float(snr_db)] = float(auc)
        n_clean[float(snr_db)] = int(clean_scores.size)
        n_corr[float(snr_db)] = int(corrupted_scores.size)

        # Bootstrap CI on AUC (resample paired window indices because
        # clean[i] and corrupted[i] are paired observations of the same
        # underlying window).
        n = int(clean_scores.size)
        if n >= 4 and n_boot > 0:
            boot_aucs = np.empty(int(n_boot), dtype=np.float64)
            for b in range(int(n_boot)):
                idx = rng.integers(0, n, size=n)
                boot_aucs[b] = _roc_auc(corrupted_scores[idx], clean_scores[idx])
            ci_low_per_snr[float(snr_db)] = float(np.percentile(boot_aucs, 2.5))
            ci_high_per_snr[float(snr_db)] = float(np.percentile(boot_aucs, 97.5))
        else:
            ci_low_per_snr[float(snr_db)] = float("nan")
            ci_high_per_snr[float(snr_db)] = float("nan")

    return SyntheticAnomalyAUC(
        snr_db_to_auc=auc_per_snr,
        snr_db_to_n_clean=n_clean,
        snr_db_to_n_corrupted=n_corr,
        snr_db_to_auc_ci_low=ci_low_per_snr,
        snr_db_to_auc_ci_high=ci_high_per_snr,
        seed=int(seed),
    )


__all__ = [
    "SyntheticAnomalyAUC",
    "evaluate_synthetic_anomaly_auc",
]
