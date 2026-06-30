"""RQ2 evaluation harness for the V0 anomaly baselines.

Runs every V0 scorer (LSTM-AE, K-means, OC-SVM, KDE) through the *same*
evaluation protocol the proposed conditional head is held to, so the reported
gain of the head is relative to a credible prior-work reference rather than to
nothing.

Protocol (mirrors `train_v3_cnf`, which pools `ANOM_LOADERS` and fits its
thresholds on a held-out cohort):

  * **Pool across campaigns.**  The prototype corpus carries only a handful of
    *healthy recordings per campaign* (1–4), too few for a recording-level
    held-out split on its own, so healthy windows are pooled across the
    requested datasets exactly as the head's training is.
  * **Recording-level split.**  Healthy recordings are split into ``train``
    (fits the scorer) and a ``heldout`` pool the scorer never sees.  The held-out
    pool then drives two calibration regimes, both using the head's per-cluster
    percentile thresholding rule
    (:class:`~src.modeling.anomaly.threshold.PerClusterThresholds`) so the
    comparison isolates the *scoring* mechanism, not the thresholding.

Reported numbers:
  * **Detection** — within-campaign healthy-vs-anomaly ROC-AUC (Khamaisi's
    headline number) with a bootstrap CI; shows the V0 scorer detects anomalies.
  * **Calibration contrast** — the held-out healthy false-positive rate when the
    threshold is calibrated *in-distribution* (a window split of the conditions
    under test) versus *under domain shift* (a disjoint set of held-out
    conditions), each with a Wilson CI.  An in-distribution FPR near the target
    with a domain-shift FPR far above it is the unconditional baseline's failure
    mode and the precise gap the conditional head closes.
  * **Supplementary** — anomaly alert rate under the in-distribution threshold
    and the controlled synthetic-anomaly ROC-AUC ladder (kept in ``details``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from ...config import resolve_device
from ...ingestion.test_dataset_loader import TestDatasetLoader, TestDatasetSegment
from ..anomaly.synthetic_eval import _roc_auc
from ..anomaly.threshold import PerClusterThresholds
from .density_scorers import _aggregate_windows, build_scorer
from .lstm_ae import (
    V0Config,
    extract_log_mel_windows,
    extract_vibration_temporal_windows,
    fit_lstm_ae_on_windows,
)

# Models that produce a flat-feature score (everything except the AE).
CLASSICAL_MODELS = ("kmeans", "ocsvm", "kde")
ALL_MODELS = ("lstm_ae", *CLASSICAL_MODELS)
MODALITIES = ("acoustic", "vibration")


# ---------------------------------------------------------------------------
# Window bank — one slice of signal, both representations, plus metadata
# ---------------------------------------------------------------------------


@dataclass
class WindowBank:
    """Per-window features + metadata for one pooled corpus / modality.

    ``seq`` is the ``(N, T, F)`` sequence the LSTM-AE ingests; ``feats`` is the
    ``(N, 2F)`` time-aggregated flat vector the classical scorers ingest and
    that also serves as the shared K-means context for thresholding.  Every row
    of ``seq`` and ``feats`` describes the identical window of signal.
    """

    seq: np.ndarray
    feats: np.ndarray
    rec_id: np.ndarray
    dataset_id: np.ndarray
    is_anomaly: np.ndarray
    mode: np.ndarray
    op_cond: np.ndarray

    @property
    def n(self) -> int:
        return int(self.feats.shape[0])


def _extract_seq(segment: TestDatasetSegment, cfg: V0Config, modality: str) -> np.ndarray:
    if modality == "acoustic":
        return extract_log_mel_windows(segment, cfg)
    if modality == "vibration":
        return extract_vibration_temporal_windows(segment, cfg)
    raise ValueError(f"unknown modality {modality!r}")


def build_window_bank(
    segments: list[TestDatasetSegment], cfg: V0Config, modality: str
) -> WindowBank:
    """Slide windows over every segment, collecting both representations.

    The flat ``feats`` (time-mean+std) are a fixed ``2F`` width and always
    concatenate.  The ``seq`` windows are ``(N, T, F)``; ``T`` is constant only
    when every segment shares a sample rate (true for the 16 kHz microphones,
    but *not* for the vibration stream, whose accelerometer rate varies across
    campaigns).  When the frame counts disagree, ``seq`` is left empty and the
    sequence-consuming LSTM-AE raises a clear error, while the flat-feature
    scorers (K-means / OC-SVM / KDE) still run on the pooled corpus.
    """
    seqs: list[np.ndarray] = []
    feats: list[np.ndarray] = []
    rec_ids: list[str] = []
    ds_ids: list[str] = []
    is_anom: list[bool] = []
    modes: list[str] = []
    op_conds: list[str] = []
    frames: int | None = None
    seq_consistent = True
    for s in segments:
        seq = _extract_seq(s, cfg, modality)
        if seq.shape[0] == 0:
            continue
        if frames is None:
            frames = seq.shape[1]
        elif seq.shape[1] != frames:
            seq_consistent = False
        seqs.append(seq)
        feats.append(_aggregate_windows(seq))
        n = seq.shape[0]
        rec_ids.extend([s.recording_id] * n)
        ds_ids.extend([s.dataset_id] * n)
        is_anom.extend([bool(s.is_anomaly)] * n)
        modes.extend([s.mode_label or ""] * n)
        op_conds.extend([s.op_condition or ""] * n)
    f = cfg.effective_feature_dim
    if not seqs:
        empty_o = np.array([], dtype=object)
        return WindowBank(
            seq=np.zeros((0, 1, f), dtype=np.float32),
            feats=np.zeros((0, 2 * f), dtype=np.float32),
            rec_id=empty_o, dataset_id=empty_o, is_anomaly=np.array([], dtype=bool),
            mode=empty_o, op_cond=empty_o,
        )
    # seq concatenates only when every window shares a frame count; otherwise it
    # is left empty (flat-feature scorers don't need it).
    seq_arr = (
        np.concatenate(seqs, axis=0)
        if seq_consistent
        else np.zeros((0, 1, f), dtype=np.float32)
    )
    return WindowBank(
        seq=seq_arr,
        feats=np.concatenate(feats, axis=0),
        rec_id=np.array(rec_ids, dtype=object),
        dataset_id=np.array(ds_ids, dtype=object),
        is_anomaly=np.array(is_anom, dtype=bool),
        mode=np.array(modes, dtype=object),
        op_cond=np.array(op_conds, dtype=object),
    )


# ---------------------------------------------------------------------------
# Recording-level splits (stratified by operating condition)
# ---------------------------------------------------------------------------


def _plain_split(
    unique: list[str], val_ratio: float, seed: int
) -> tuple[set[str], set[str]]:
    """Deterministic recording-level split (matches V3's `_split_segments_by_recording`).

    No stratification: each campaign holds essentially one healthy recording per
    operating condition, so a held-out recording is necessarily a *different*
    condition than the fit pool.  That is exactly the regime the conditional head
    must survive, and the unconditional V0's drift across it is the finding — not
    an artefact to be engineered away.
    """
    recs = sorted(set(unique))
    np.random.default_rng(seed).shuffle(recs)
    n_val = max(1, int(round(len(recs) * val_ratio)))
    val, train = recs[:n_val], recs[n_val:]
    if not train:  # tiny corpus — keep ≥ 1 fit recording
        train = [val.pop()]
    return set(train), set(val)


# ---------------------------------------------------------------------------
# Generic synthetic-anomaly ROC-AUC (model-agnostic mirror of synthetic_eval)
# ---------------------------------------------------------------------------


@dataclass
class SyntheticAUCResult:
    snr_db_to_auc: dict[float, float]
    snr_db_to_auc_ci_low: dict[float, float]
    snr_db_to_auc_ci_high: dict[float, float]
    n_clean: int
    seed: int


def evaluate_synthetic_anomaly_auc_v0(
    score_fn: Callable[[np.ndarray], np.ndarray],
    healthy_x: np.ndarray,
    *,
    snr_db_list: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0),
    n_boot: int = 500,
    seed: int = 0,
) -> SyntheticAUCResult:
    """ROC-AUC of ``score_fn`` separating clean vs noise-corrupted windows.

    A model-agnostic mirror of
    :func:`src.modeling.anomaly.synthetic_eval.evaluate_synthetic_anomaly_auc`:
    held-out healthy inputs are perturbed by isotropic Gaussian noise of scale
    ``σ_noise = σ_x · 10^(-SNR_dB / 20)`` (``σ_x`` the global std of ``healthy_x``)
    in the scorer's *native* input space, yielding window-level ground truth on
    which the anomaly score is a genuine binary discriminator.  ``healthy_x`` may
    be ``(N, D)`` flat features or ``(N, T, F)`` sequence windows; the noise is
    added in whichever shape is given.
    """
    rng = np.random.default_rng(int(seed))
    healthy_x = np.asarray(healthy_x, dtype=np.float64)
    n = healthy_x.shape[0]
    nan = {float(s): float("nan") for s in snr_db_list}
    if n < 4:
        return SyntheticAUCResult(dict(nan), dict(nan), dict(nan), n, int(seed))
    sigma_x = float(np.std(healthy_x))
    if sigma_x <= 0.0:
        return SyntheticAUCResult(dict(nan), dict(nan), dict(nan), n, int(seed))

    clean_scores = np.asarray(score_fn(healthy_x), dtype=np.float64).reshape(-1)
    auc: dict[float, float] = {}
    ci_low: dict[float, float] = {}
    ci_high: dict[float, float] = {}
    for snr_db in snr_db_list:
        sigma_noise = sigma_x * (10.0 ** (-float(snr_db) / 20.0))
        corrupted = healthy_x + rng.normal(0.0, sigma_noise, size=healthy_x.shape)
        corrupt_scores = np.asarray(score_fn(corrupted), dtype=np.float64).reshape(-1)
        auc[float(snr_db)] = _roc_auc(corrupt_scores, clean_scores)
        if n >= 4 and n_boot > 0:
            boot = np.empty(int(n_boot), dtype=np.float64)
            for b in range(int(n_boot)):
                idx = rng.integers(0, n, size=n)
                boot[b] = _roc_auc(corrupt_scores[idx], clean_scores[idx])
            ci_low[float(snr_db)] = float(np.percentile(boot, 2.5))
            ci_high[float(snr_db)] = float(np.percentile(boot, 97.5))
        else:
            ci_low[float(snr_db)] = float("nan")
            ci_high[float(snr_db)] = float("nan")
    return SyntheticAUCResult(auc, ci_low, ci_high, n, int(seed))


# ---------------------------------------------------------------------------
# Per-model score functions
# ---------------------------------------------------------------------------


def _ae_score_fn(
    model, mean: np.ndarray, std: np.ndarray, device
) -> Callable[[np.ndarray], np.ndarray]:
    """Native (sequence-space) reconstruction-MSE scorer for the LSTM-AE."""

    def _fn(seq: np.ndarray) -> np.ndarray:
        norm = (np.asarray(seq, dtype=np.float32) - mean) / std
        x = torch.from_numpy(norm.astype(np.float32)).to(device)
        with torch.no_grad():
            return model.reconstruction_score(x).cpu().numpy().astype(np.float64)

    return _fn


def _fit_model(
    model_name: str,
    bank: WindowBank,
    rec_key: np.ndarray,
    healthy_mask: np.ndarray,
    train_mask: np.ndarray,
    split_ids: tuple[set[str], set[str]],
    cfg: V0Config,
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, str]:
    """Fit one V0 model on healthy-fit windows; return ``(score_fn, all_scores, kind)``.

    ``kind`` is ``"seq"`` for the AE (native input is the sequence window) or
    ``"feats"`` for the classical scorers.  ``split_ids`` is the shared
    ``(train_ids, val_ids)`` composite-recording split threaded into the AE so it
    trains on exactly the windows the threshold and the classical scorers see;
    ``rec_key`` holds the matching per-window ``dataset::recording`` keys.
    """
    if model_name == "lstm_ae":
        if bank.seq.shape[0] != bank.n:
            raise RuntimeError(
                "LSTM-AE cannot pool sequence windows whose frame counts differ "
                "(vibration accel rates vary across campaigns); run the AE on a "
                "single sample-rate corpus or use a flat-feature scorer"
            )
        result = fit_lstm_ae_on_windows(
            bank.seq[healthy_mask],
            list(rec_key[healthy_mask]),
            cfg,
            split=split_ids,
        )
        device = resolve_device(cfg.device)
        score_fn = _ae_score_fn(
            result.model, result.standardiser_mean, result.standardiser_std, device
        )
        return score_fn, score_fn(bank.seq), "seq"

    scorer = build_scorer(model_name, seed=cfg.seed)
    scorer.fit(bank.feats[train_mask])
    return scorer.score, scorer.score(bank.feats), "feats"


# ---------------------------------------------------------------------------
# Breakdown helpers
# ---------------------------------------------------------------------------


def _group_alert_rate(
    keys: np.ndarray, alerts: np.ndarray, *, skip_empty: bool = True
) -> dict[str, dict[str, float | int]]:
    """Alert rate broken down by an arbitrary per-window key (mode / op / dataset)."""
    out: dict[str, dict[str, float | int]] = {}
    uniq = sorted({k for k in keys.tolist() if not (skip_empty and k == "")})
    for key in uniq:
        mask = keys == key
        n_k = int(mask.sum())
        n_alerts = int(alerts[mask].sum())
        out[str(key)] = {
            "n": n_k,
            "n_alerts": n_alerts,
            "alert_rate": float(n_alerts / n_k) if n_k else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# Uncertainty estimates (Experiments §5.1: Wilson on rates, bootstrap on AUC)
# ---------------------------------------------------------------------------


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion ``k / n`` (95 % at z=1.96)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _roc_auc_ci(
    pos: np.ndarray, neg: np.ndarray, *, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """ROC-AUC of ``pos`` vs ``neg`` with a percentile-bootstrap 95 % CI."""
    auc = _roc_auc(pos, neg)
    if n_boot <= 0 or pos.size < 2 or neg.size < 2:
        return auc, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        pi = rng.integers(0, pos.size, pos.size)
        ni = rng.integers(0, neg.size, neg.size)
        boots[b] = _roc_auc(pos[pi], neg[ni])
    return auc, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _fpr_under_threshold(
    thr: PerClusterThresholds,
    feats: np.ndarray,
    scores: np.ndarray,
    percentile: int,
) -> tuple[float, tuple[float, float], int, int]:
    """False-positive rate of a healthy cohort under a fitted threshold + Wilson CI."""
    alerts, _ = thr.alert(feats, scores, percentile=percentile)
    n = int(alerts.size)
    k = int(alerts.sum())
    rate = float(k / n) if n else float("nan")
    return rate, _wilson_interval(k, n), k, n


# ---------------------------------------------------------------------------
# Top-level: evaluate one (corpus, modality, model)
# ---------------------------------------------------------------------------


@dataclass
class V0AnomalyResult:
    """One V0 scorer's RQ2 numbers over a pooled corpus / modality.

    The two headline stories:
      * ``roc_auc`` — within-campaign healthy-vs-anomaly detection AUC (Khamaisi's
        reported number); shows the V0 scorer is a *competent detector*.
      * ``fpr_in_distribution`` vs ``fpr_domain_shift`` — the same scorer's healthy
        false-positive rate when its threshold is calibrated to the operating
        condition under test versus to *other* conditions.  The jump between them
        is the unconditional baseline's domain-shift failure, the gap the
        conditional head is built to close.
    """

    dataset_ids: list[str]
    modality: str
    model: str
    n_clusters: int
    percentile: int
    roc_auc: float
    roc_auc_ci: tuple[float, float]
    fpr_in_distribution: float
    fpr_in_ci: tuple[float, float]
    fpr_domain_shift: float
    fpr_shift_ci: tuple[float, float]
    anomaly_alert_rate: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset_ids": self.dataset_ids,
            "modality": self.modality,
            "model": self.model,
            "n_clusters": self.n_clusters,
            "percentile": self.percentile,
            "roc_auc": self.roc_auc,
            "roc_auc_ci": list(self.roc_auc_ci),
            "fpr_in_distribution": self.fpr_in_distribution,
            "fpr_in_ci": list(self.fpr_in_ci),
            "fpr_domain_shift": self.fpr_domain_shift,
            "fpr_shift_ci": list(self.fpr_shift_ci),
            "anomaly_alert_rate": self.anomaly_alert_rate,
            **self.details,
        }


def _all_segments(loaders: list[TestDatasetLoader]) -> list[TestDatasetSegment]:
    out: list[TestDatasetSegment] = []
    for loader in loaders:
        out.extend(loader.list_segments())
    return out


def evaluate_v0_anomaly(
    loaders: TestDatasetLoader | list[TestDatasetLoader],
    model_name: str,
    modality: str = "acoustic",
    cfg: V0Config | None = None,
    *,
    percentile: int = 95,
    n_clusters: int = 3,
    threshold_fit_ratio: float = 0.5,
    snr_db_list: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0),
    n_boot: int = 500,
) -> V0AnomalyResult:
    """Full RQ2 evaluation of one V0 scorer over a pooled corpus / modality.

    ``loaders`` is one loader or a list pooled into a single corpus (the way the
    proposed head is trained on `ANOM_LOADERS`).  ``model_name`` is one of
    ``lstm_ae`` / ``kmeans`` / ``ocsvm`` / ``kde``.  The vibration modality
    forces a 3-channel feature dimension on the LSTM-AE.
    """
    if model_name not in ALL_MODELS:
        raise ValueError(f"unknown model {model_name!r}; expected one of {ALL_MODELS}")
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}; expected one of {MODALITIES}")
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]

    cfg = cfg or V0Config()
    if modality == "vibration" and cfg.feature_dim is None:
        cfg = V0Config(**{**cfg.__dict__, "feature_dim": 3})

    dataset_ids = [loader.spec.id for loader in loaders]
    bank = build_window_bank(_all_segments(loaders), cfg, modality)
    if bank.n == 0:
        raise RuntimeError(f"no windows for {dataset_ids}/{modality}")

    healthy = ~bank.is_anomaly
    if not healthy.any():
        raise RuntimeError(f"no healthy windows for {dataset_ids}/{modality}")

    # Recording-level split (plain, mirrors `train_v3_cnf`): the scorer trains on
    # `train`; the rest (`heldout`) is never seen by the scorer and supplies both
    # calibration regimes below.  Recording ids are not unique across campaigns
    # (D3 and D4 both name a recording `speed1`), so split on a composite
    # `dataset::recording` key to avoid collapsing — and leaking between — them.
    rec_key = np.array(
        [f"{ds}::{r}" for ds, r in zip(bank.dataset_id, bank.rec_id)], dtype=object
    )
    healthy_recs = sorted(set(rec_key[healthy].tolist()))
    if len(healthy_recs) < 3:
        raise RuntimeError(
            f"only {len(healthy_recs)} healthy recording(s) in {dataset_ids}/{modality}; "
            f"the held-out calibration split needs >= 3 — pool more datasets"
        )
    train_ids, heldout_ids = _plain_split(healthy_recs, cfg.val_ratio, cfg.seed)
    if len(heldout_ids) < 2:
        raise RuntimeError(
            f"only {len(heldout_ids)} held-out healthy recording(s) for "
            f"{dataset_ids}/{modality}; the domain-shift split needs >= 2 — "
            f"raise val_ratio or pool more datasets"
        )

    def _mask(ids: set[str]) -> np.ndarray:
        keep = np.array([k in ids for k in rec_key], dtype=bool)
        return healthy & keep

    train_mask = _mask(train_ids)
    heldout_mask = _mask(heldout_ids)
    anom_mask = bank.is_anomaly

    # Fit the scorer on healthy-fit windows; score everything.
    score_fn, all_scores, native_kind = _fit_model(
        model_name, bank, rec_key, healthy, train_mask, (train_ids, heldout_ids), cfg
    )

    # ----------------------------------------------------------------------
    # Calibration in two regimes, both on held-out healthy the scorer never
    # trained on, isolating the effect of the operating-condition shift.
    # ----------------------------------------------------------------------
    heldout_idx = np.where(heldout_mask)[0]  # ascending → grouped by recording

    # (a) IN-DISTRIBUTION: fit the threshold on half of every held-out
    #     recording's windows and score the other half — the conditions under
    #     test are present on both sides, so a competent detector calibrates to
    #     near the target FPR.  Interleaving by (sorted) window index spreads
    #     each recording across both halves.
    in_fit_idx, in_eval_idx = heldout_idx[0::2], heldout_idx[1::2]
    if in_fit_idx.size < n_clusters or in_eval_idx.size == 0:
        raise RuntimeError(
            f"too few held-out windows ({heldout_idx.size}) to calibrate "
            f"{dataset_ids}/{modality}; pool more datasets"
        )
    k_eff = max(1, min(n_clusters, int(in_fit_idx.size)))
    thr_in = PerClusterThresholds.fit(
        bank.feats[in_fit_idx], all_scores[in_fit_idx], n_clusters=k_eff, seed=cfg.seed
    )
    fpr_in, fpr_in_ci, _, _ = _fpr_under_threshold(
        thr_in, bank.feats[in_eval_idx], all_scores[in_eval_idx], percentile
    )

    # (b) DOMAIN-SHIFT: fit the threshold on one set of held-out *conditions*
    #     and score a disjoint set — the conditions under test are unseen by the
    #     threshold, the exact regime the conditional head must survive.
    shift_fit_recs, shift_eval_recs = _plain_split(
        sorted(heldout_ids), threshold_fit_ratio, cfg.seed + 1
    )
    sf_mask, se_mask = _mask(shift_fit_recs), _mask(shift_eval_recs)
    if int(sf_mask.sum()) >= n_clusters and se_mask.any():
        thr_shift = PerClusterThresholds.fit(
            bank.feats[sf_mask], all_scores[sf_mask], n_clusters=k_eff, seed=cfg.seed
        )
        fpr_shift, fpr_shift_ci, _, _ = _fpr_under_threshold(
            thr_shift, bank.feats[se_mask], all_scores[se_mask], percentile
        )
        shift_by_dataset = _group_alert_rate(
            bank.dataset_id[se_mask],
            thr_shift.alert(bank.feats[se_mask], all_scores[se_mask], percentile=percentile)[0],
        )
    else:
        fpr_shift, fpr_shift_ci, shift_by_dataset = float("nan"), (float("nan"), float("nan")), {}

    # Headline detection — within-campaign healthy-vs-anomaly ROC-AUC (Khamaisi's
    # reported number, e.g. OC-SVM ≈ 0.998 at ROW II).  Negatives are a campaign's
    # own healthy windows, positives its labelled anomaly windows, so the AUC is a
    # clean within-condition detection measure, free of the cross-condition shift
    # that contaminates a pooled-negative construction.  Caveat: anomaly recordings
    # carry unlabelled healthy windows and some campaign-healthy windows were in
    # the scorer's training pool, so this is the standard (slightly optimistic)
    # unsupervised-AD detection AUC — exactly Khamaisi's construction.
    roc_auc_by_dataset: dict[str, float] = {}
    pooled_pos: list[np.ndarray] = []
    pooled_neg: list[np.ndarray] = []
    for ds in sorted(set(bank.dataset_id[anom_mask].tolist())) if anom_mask.any() else []:
        ds_mask = bank.dataset_id == ds
        pos, neg = all_scores[anom_mask & ds_mask], all_scores[healthy & ds_mask]
        if pos.size and neg.size:
            roc_auc_by_dataset[str(ds)] = _roc_auc(pos, neg)
            pooled_pos.append(pos)
            pooled_neg.append(neg)
    if pooled_pos:
        roc_auc, roc_lo, roc_hi = _roc_auc_ci(
            np.concatenate(pooled_pos), np.concatenate(pooled_neg),
            n_boot=n_boot, seed=cfg.seed,
        )
    else:
        roc_auc, roc_lo, roc_hi = float("nan"), float("nan"), float("nan")

    # Supplementary — anomaly alert rate under the in-distribution threshold, and
    # the controlled synthetic-anomaly ROC-AUC ladder.
    if anom_mask.any():
        anom_alerts, _ = thr_in.alert(
            bank.feats[anom_mask], all_scores[anom_mask], percentile=percentile
        )
        anomaly_rate = float(anom_alerts.mean())
        anomaly_by_dataset = _group_alert_rate(bank.dataset_id[anom_mask], anom_alerts)
    else:
        anomaly_rate, anomaly_by_dataset = float("nan"), {}
    native_eval = bank.seq[in_eval_idx] if native_kind == "seq" else bank.feats[in_eval_idx]
    syn = evaluate_synthetic_anomaly_auc_v0(
        score_fn, native_eval, snr_db_list=snr_db_list, n_boot=n_boot, seed=cfg.seed
    )

    return V0AnomalyResult(
        dataset_ids=dataset_ids,
        modality=modality,
        model=model_name,
        n_clusters=k_eff,
        percentile=percentile,
        roc_auc=roc_auc,
        roc_auc_ci=(roc_lo, roc_hi),
        fpr_in_distribution=fpr_in,
        fpr_in_ci=fpr_in_ci,
        fpr_domain_shift=fpr_shift,
        fpr_shift_ci=fpr_shift_ci,
        anomaly_alert_rate=anomaly_rate,
        details={
            "n_windows_total": bank.n,
            "n_train": int(train_mask.sum()),
            "n_heldout": int(heldout_mask.sum()),
            "n_anomaly": int(anom_mask.sum()),
            "n_train_recordings": len(train_ids),
            "n_heldout_recordings": len(heldout_ids),
            "roc_auc_by_dataset": roc_auc_by_dataset,
            "anomaly_alert_by_dataset": anomaly_by_dataset,
            "domain_shift_fpr_by_dataset": shift_by_dataset,
            "synthetic_auc": {str(k): v for k, v in syn.snr_db_to_auc.items()},
        },
    )


__all__ = [
    "ALL_MODELS",
    "CLASSICAL_MODELS",
    "MODALITIES",
    "SyntheticAUCResult",
    "V0AnomalyResult",
    "WindowBank",
    "build_window_bank",
    "evaluate_synthetic_anomaly_auc_v0",
    "evaluate_v0_anomaly",
]
