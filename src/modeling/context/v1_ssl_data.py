"""V1 SSL data pipeline: windowed feature dataset, sampler, augmentation, splits."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.utils.data as tud

from ...features.audio_spectral import compute_encoder_input_stack
from ...features.vibration_temporal import compute_vibration_input_stack
from ...ingestion.test_dataset_loader import (
    TestDatasetLoader,
    TestDatasetSegment,
)
from .v1_ssl_config import (
    V1SSLConfig,
    _dataset_idx,
)


@dataclass
class _PrecomputedSegment:
    """One segment's full-duration feature stack + per-channel xyz + label."""

    features: np.ndarray  # acoustic: (n_mics, 2, F, T_frames) ; vibration: (n_vib, 3, T)
    xyz: np.ndarray  # (N, 3)
    dataset_idx: int
    dataset_id: str  # canonical string id used to resolve per-dataset window scales
    mode_label: str
    recording_id: str
    source_dir: str
    feature_fs: float  # frames per second along the last feature axis
def _precompute_segment(
    s: TestDatasetSegment,
    modality: Literal["acoustic", "vibration"],
    cfg: V1SSLConfig,
) -> _PrecomputedSegment | None:
    if modality == "acoustic":
        if cfg.use_cwt:
            features = compute_encoder_input_stack(
                s.segment.mic_data,
                fs=int(s.segment.mic_sample_rate),
                n_mels=cfg.n_mels,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                cwt_n_scales=cfg.cwt_n_scales,
                standardize=cfg.standardize_acoustic,
            )
        else:
            # Fast path for smoke tests — log-mel only, duplicated as channel 1.
            from ...features.audio_spectral import compute_log_mel_spectrogram

            mels = []
            for ch in range(s.segment.n_mic_channels):
                m = compute_log_mel_spectrogram(
                    s.segment.mic_data[ch],
                    fs=int(s.segment.mic_sample_rate),
                    n_fft=cfg.n_fft,
                    hop_length=cfg.hop_length,
                    n_mels=cfg.n_mels,
                )
                mels.append(np.stack([m, m], axis=0).astype(np.float32))
            features = np.stack(mels, axis=0)
        xyz = s.mic_positions.astype(np.float32)
        feature_fs = float(s.segment.mic_sample_rate) / float(cfg.hop_length)
    elif modality == "vibration":
        features = compute_vibration_input_stack(
            s.segment.accel_data,
            sample_rate=float(s.segment.accel_sample_rate),
            kurtosis_window_seconds=cfg.vib_kurtosis_window_seconds,
            min_kurtosis_samples=cfg.vib_min_kurtosis_samples,
            crest_factor_window_seconds=cfg.vib_crest_factor_window_seconds,
            min_crest_factor_samples=cfg.vib_min_crest_factor_samples,
            standardize=cfg.standardize_vibration,
        )
        xyz = s.vib_positions.astype(np.float32)
        feature_fs = float(s.segment.accel_sample_rate)
    else:
        raise ValueError(modality)

    if features.shape[-1] < 2:
        return None  # too short for any window

    return _PrecomputedSegment(
        features=features.astype(np.float32),
        xyz=xyz,
        dataset_idx=_dataset_idx(s.dataset_id),
        dataset_id=str(s.dataset_id),
        mode_label=s.mode_label or "Unknown",
        recording_id=s.recording_id,
        source_dir=str(s.source_dir),
        feature_fs=feature_fs,
    )


# ---------------------------------------------------------------------------
# Dataset + augmentation
# ---------------------------------------------------------------------------
def _resolve_segment_scales(
    cfg: V1SSLConfig, dataset_id: str
) -> tuple[tuple[float, ...], float]:
    """Return the (scales, stride_ratio) for one segment.

    Priority:
      1. Per-dataset override in ``cfg.window_scales_seconds_per_dataset``.
      2. Global ``cfg.window_scales_seconds`` tuple if non-empty.
      3. Legacy fallback ``(cfg.window_seconds,)`` with stride derived
         from ``cfg.window_stride_seconds / cfg.window_seconds`` — this
         is the byte-equivalent path for pre-2026-05-19 configs.
    """
    per_ds = cfg.window_scales_seconds_per_dataset or {}
    if per_ds.get(dataset_id):
        scales = tuple(float(s) for s in per_ds[dataset_id])
        return scales, float(cfg.window_stride_ratio)
    if cfg.window_scales_seconds:
        scales = tuple(float(s) for s in cfg.window_scales_seconds)
        return scales, float(cfg.window_stride_ratio)
    legacy_ratio = (
        float(cfg.window_stride_seconds) / float(cfg.window_seconds)
        if cfg.window_seconds > 0
        else 0.5
    )
    return (float(cfg.window_seconds),), legacy_ratio
class _WindowedFeatureDataset(tud.Dataset):
    """Yields (feature_window, xyz, dataset_idx, mode_label) per index.

    Window size in frames is computed *per segment* from its `feature_fs`
    AND per scale from `cfg.window_scales_seconds[_per_dataset]`, so a
    2-second window over D4 raw vibration (~376 Hz) keeps all 752 samples
    and a 2-second window over D1 peak vibration (4 Hz) keeps all 8.

    When the config defines multiple scales, every (segment, scale) pair
    materialises its own window list; the grouped batch sampler then
    buckets by ``(channel_count, n_frames)`` so every batch is
    single-dataset × single-scale and ``torch.stack`` never trips.
    """

    def __init__(
        self,
        segments: list[_PrecomputedSegment],
        cfg: V1SSLConfig,
        *,
        enable_mixup: bool = False,
    ) -> None:
        self.cfg = cfg
        self._segments = segments
        # Mixup is opt-in per dataset instance; the trainer enables it on the
        # TRAIN dataset only, never on val.  Reading `cfg.mixup_alpha > 0`
        # alone would silently blend val windows too.
        self._mixup_enabled = bool(enable_mixup and cfg.mixup_alpha > 0.0)

        if not segments:
            self._refs: list[tuple[int, int, int]] = []
            self._partner_pool: dict[tuple, list[int]] = {}
            return

        self._refs = []
        for si, seg in enumerate(segments):
            scales, stride_ratio = _resolve_segment_scales(cfg, seg.dataset_id)
            T = int(seg.features.shape[-1])
            for scale_s in scales:
                n_frames = max(2, int(round(scale_s * seg.feature_fs)))
                stride = max(1, int(round(scale_s * stride_ratio * seg.feature_fs)))
                if T < n_frames:
                    continue
                for start in range(0, T - n_frames + 1, stride):
                    self._refs.append((si, start, n_frames))

        # Partner-pool index for cross-recording mixup.  Bucket key is
        # `(dataset_idx, n_ch, n_frames, mode_label)` — same as the grouped
        # batch sampler's stack-safety key plus mode_label so we only mix
        # within-mode (preserves the contrastive task's semantic content).
        # Sampling at __getitem__ filters this bucket to refs from a
        # different recording.
        self._partner_pool = {}
        if self._mixup_enabled:
            for ref_i, (si, _start, n_frames) in enumerate(self._refs):
                seg = self._segments[si]
                n_ch = int(seg.features.shape[0])
                key = (int(seg.dataset_idx), n_ch, n_frames, seg.mode_label)
                self._partner_pool.setdefault(key, []).append(ref_i)

    def __len__(self) -> int:
        return len(self._refs)

    def _sample_mixup_partner(self, idx: int) -> int | None:
        """Pick a ref index from a different recording, same bucket.

        Returns None if no valid partner exists — caller then skips mixup
        for this anchor (rather than mixing same-recording windows, which
        would defeat the "cross-recording" purpose).
        """
        import random as _r

        si, _start, n_frames = self._refs[idx]
        seg = self._segments[si]
        n_ch = int(seg.features.shape[0])
        key = (int(seg.dataset_idx), n_ch, n_frames, seg.mode_label)
        candidates = self._partner_pool.get(key)
        if not candidates or len(candidates) < 2:
            return None
        anchor_rec = seg.recording_id
        # Cheap rejection sampling — small bucket → fast; large bucket → expect
        # to hit a different recording in ≤ 3 tries.
        for _ in range(6):
            cand = candidates[_r.randrange(len(candidates))]
            if cand == idx:
                continue
            cand_seg = self._segments[self._refs[cand][0]]
            if cand_seg.recording_id != anchor_rec:
                return cand
        return None

    def __getitem__(self, idx: int):
        si, start, n_frames = self._refs[idx]
        seg = self._segments[si]
        feat = seg.features[..., start : start + n_frames]
        feat_t = torch.from_numpy(np.ascontiguousarray(feat))
        if self._mixup_enabled:
            partner = self._sample_mixup_partner(idx)
            if partner is not None:
                import random as _r

                p_si, p_start, _ = self._refs[partner]
                p_seg = self._segments[p_si]
                p_feat = p_seg.features[..., p_start : p_start + n_frames]
                lam = _r.betavariate(self.cfg.mixup_alpha, self.cfg.mixup_alpha)
                # Symmetric beta keeps mixup variance moderate; max(lam, 1-lam)
                # ensures the anchor is always the dominant component, which
                # preserves the "anchor's labels apply" invariant the
                # contrastive loss assumes.
                lam = max(lam, 1.0 - lam)
                feat_t = lam * feat_t + (1.0 - lam) * torch.from_numpy(
                    np.ascontiguousarray(p_feat)
                )
        return {
            "feat": feat_t,
            "xyz": torch.from_numpy(seg.xyz),
            "dataset_idx": int(seg.dataset_idx),
            "mode_label": seg.mode_label,
            "recording_id": seg.recording_id,
        }
def _collate(batch: list[dict]) -> dict:
    feats = torch.stack([b["feat"] for b in batch], dim=0).float()
    xyz = torch.stack([b["xyz"] for b in batch], dim=0).float()
    dataset_idx = torch.tensor([b["dataset_idx"] for b in batch], dtype=torch.long)
    mode_labels = [b["mode_label"] for b in batch]
    rec_ids = [b["recording_id"] for b in batch]
    return {
        "feat": feats,
        "xyz": xyz,
        "dataset_idx": dataset_idx,
        "mode_label": mode_labels,
        "recording_id": rec_ids,
    }
class _GroupedBatchSampler(tud.Sampler[list[int]]):
    """Yields batches whose samples share the same per-segment channel count.

    Without this, a `D1` (4-mic) sample and a `D2` (5-mic) sample collide in
    `torch.stack`.  The Set-Transformer pool itself is channel-agnostic at
    forward time; the constraint is only at the batch-tensor boundary.
    """

    def __init__(
        self,
        dataset: _WindowedFeatureDataset,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        # Bucket by (channels, frame count) — D1/D2 (4 Hz vib) and D4 (376 Hz)
        # have different per-window frame counts even when channel counts match.
        groups: dict[tuple[int, int], list[int]] = {}
        for i, (si, _start, n_frames) in enumerate(dataset._refs):
            n_ch = int(dataset._segments[si].features.shape[0])
            groups.setdefault((n_ch, n_frames), []).append(i)
        self._groups = groups

    def __iter__(self):
        rng = np.random.default_rng(self.seed if not self.shuffle else None)
        all_batches: list[list[int]] = []
        for _key, idxs in self._groups.items():
            ids = list(idxs)
            if self.shuffle:
                rng.shuffle(ids)
            for i in range(0, len(ids), self.batch_size):
                chunk = ids[i : i + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                all_batches.append(chunk)
        if self.shuffle:
            rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(v) // self.batch_size for v in self._groups.values())
        return sum(
            (len(v) + self.batch_size - 1) // self.batch_size for v in self._groups.values()
        )
class _Augmenter:
    """Apply gain jitter + channel dropout + SpecAugment in feature space.

    All augmentations operate per-sample, per-channel, and produce a single
    augmented copy.  The contrastive loop calls this twice per anchor to
    build the two SimCLR views.
    """

    def __init__(self, modality: str, cfg: V1SSLConfig, generator: torch.Generator) -> None:
        self.modality = modality
        self.cfg = cfg
        self.gen = generator

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: acoustic (B, N, 2, F, T) or vibration (B, N, 3, T)
        cfg = self.cfg
        x = x.clone()

        # Gain jitter — multiplicative scalar per (sample, channel)
        if cfg.gain_jitter_db > 0:
            db = (
                torch.rand(*x.shape[:2], 1, generator=self.gen, device=x.device)
                * (2 * cfg.gain_jitter_db)
                - cfg.gain_jitter_db
            )
            scale = 10.0 ** (db / 20.0)
            while scale.ndim < x.ndim:
                scale = scale.unsqueeze(-1)
            x = x * scale

        # Channel dropout — zero out a fraction of channels per sample
        if cfg.channel_dropout_p > 0:
            drop = (
                torch.rand(*x.shape[:2], generator=self.gen, device=x.device)
                < cfg.channel_dropout_p
            )
            keep = (~drop).float()
            while keep.ndim < x.ndim:
                keep = keep.unsqueeze(-1)
            x = x * keep

        # SpecAugment-style masks (acoustic only, on F and T)
        if self.modality == "acoustic" and (cfg.spec_augment_freq_mask > 0 or cfg.spec_augment_time_mask > 0):
            B, N, C, Ff, Tt = x.shape
            if cfg.spec_augment_freq_mask > 0 and Ff > cfg.spec_augment_freq_mask:
                start = torch.randint(0, Ff - cfg.spec_augment_freq_mask + 1, (B, N), generator=self.gen, device=x.device)
                width = torch.randint(1, cfg.spec_augment_freq_mask + 1, (B, N), generator=self.gen, device=x.device)
                for b in range(B):
                    for n in range(N):
                        s_, w_ = int(start[b, n]), int(width[b, n])
                        x[b, n, :, s_ : s_ + w_, :] = 0.0
            if cfg.spec_augment_time_mask > 0 and Tt > cfg.spec_augment_time_mask:
                start = torch.randint(0, Tt - cfg.spec_augment_time_mask + 1, (B, N), generator=self.gen, device=x.device)
                width = torch.randint(1, cfg.spec_augment_time_mask + 1, (B, N), generator=self.gen, device=x.device)
                for b in range(B):
                    for n in range(N):
                        s_, w_ = int(start[b, n]), int(width[b, n])
                        x[b, n, :, :, s_ : s_ + w_] = 0.0

        return x


# ---------------------------------------------------------------------------
# Projection head + NT-Xent
# ---------------------------------------------------------------------------
def _time_split_segment(
    seg: _PrecomputedSegment, val_ratio: float
) -> tuple[_PrecomputedSegment | None, _PrecomputedSegment | None]:
    """Split a single segment in time: first ``(1 − val_ratio)`` of frames →
    train pseudo-segment, last ``val_ratio`` of frames → val pseudo-segment.

    Used by `_split_segments_by_recording` when a `mode_label` group
    has only one recording — without this, that mode never appears in
    val and the K = 3 sanity-gate cannot evaluate it.  Time-splitting
    the same recording trades a small temporal-correlation leak (no
    different from any within-recording train/val split in the
    machine-condition-monitoring literature) for full coverage of the
    K = 3 mode hypothesis at evaluation time.  The pseudo-segments
    share `mode_label`, `xyz`, `feature_fs`, `dataset_idx`, and
    `source_dir`, and differ only in `features` (sliced along the last
    axis) and `recording_id` (suffixed `__train_half` / `__val_half`).
    Returns ``(None, None)`` if the segment is too short to split.
    """
    T = int(seg.features.shape[-1])
    if T < 4:
        return None, None
    n_val = max(2, int(round(T * val_ratio)))
    n_val = min(n_val, T - 2)
    if n_val < 2 or T - n_val < 2:
        return None, None
    train_feats = seg.features[..., : T - n_val]
    val_feats = seg.features[..., T - n_val :]
    train_seg = _PrecomputedSegment(
        features=train_feats,
        xyz=seg.xyz,
        dataset_idx=seg.dataset_idx,
        dataset_id=seg.dataset_id,
        mode_label=seg.mode_label,
        recording_id=f"{seg.recording_id}__train_half",
        source_dir=seg.source_dir,
        feature_fs=seg.feature_fs,
    )
    val_seg = _PrecomputedSegment(
        features=val_feats,
        xyz=seg.xyz,
        dataset_idx=seg.dataset_idx,
        dataset_id=seg.dataset_id,
        mode_label=seg.mode_label,
        recording_id=f"{seg.recording_id}__val_half",
        source_dir=seg.source_dir,
        feature_fs=seg.feature_fs,
    )
    return train_seg, val_seg
def _split_segments_by_recording(
    segments: list[_PrecomputedSegment], val_ratio: float, seed: int
) -> tuple[list[_PrecomputedSegment], list[_PrecomputedSegment]]:
    """Stratified-by-mode-label held-out split at the *recording* level.

    Three regimes per `mode_label` group:

    * ``count ≥ 2`` recordings — standard recording-level stratified split:
      each labeled mode contributes ⌊val_ratio · count⌋ recordings (≥ 1)
      to val with the rest in train.  No within-recording leakage.

    * ``count == 1`` recordings — the recording is **time-split**: first
      ``(1 − val_ratio)`` of its feature frames go into a "train half"
      pseudo-segment, the last ``val_ratio`` into a "val half" pseudo-
      segment.  This guarantees the mode appears in val for the K = 3
      sanity-gate (the alternative — putting the single recording
      entirely in train — was the structural reason Standstill never
      appeared in val before this fix).
      The temporal-correlation cost is the same as any within-recording
      split in the CBM literature and is documented as a known limit.

    * ``count == 0`` recordings — nothing to do.

    Recordings with `mode_label = None` (D3 / D4 healthy speed-bucket
    recordings — mode unrecorded by protocol) are split at the recording
    level without stratification.
    """
    rng = np.random.default_rng(seed)
    rec_to_label: dict[tuple, str | None] = {}
    rec_to_seg: dict[tuple, _PrecomputedSegment] = {}
    for s in segments:
        key = (s.dataset_idx, s.recording_id, s.source_dir)
        if key not in rec_to_label:
            rec_to_label[key] = s.mode_label
            rec_to_seg[key] = s

    by_label: dict[str | None, list] = {}
    for key, lbl in rec_to_label.items():
        by_label.setdefault(lbl, []).append(key)

    train_keys: set = set()
    val_keys: set = set()
    time_split_train: list[_PrecomputedSegment] = []
    time_split_val: list[_PrecomputedSegment] = []
    for recs in by_label.values():
        recs_shuffled = list(recs)
        rng.shuffle(recs_shuffled)
        if len(recs_shuffled) == 1:
            # Time-split the single recording so this mode appears in val.
            only_key = recs_shuffled[0]
            tr, va = _time_split_segment(rec_to_seg[only_key], val_ratio)
            if tr is not None and va is not None:
                time_split_train.append(tr)
                time_split_val.append(va)
            else:
                # Segment too short to split — fall back to train-only.
                train_keys.update(recs_shuffled)
            continue
        n_val_for_label = max(1, int(round(len(recs_shuffled) * val_ratio)))
        # Keep at least one in train per label too.
        n_val_for_label = min(n_val_for_label, len(recs_shuffled) - 1)
        val_keys.update(recs_shuffled[:n_val_for_label])
        train_keys.update(recs_shuffled[n_val_for_label:])

    train = [s for s in segments if (s.dataset_idx, s.recording_id, s.source_dir) in train_keys]
    val = [s for s in segments if (s.dataset_idx, s.recording_id, s.source_dir) in val_keys]
    train.extend(time_split_train)
    val.extend(time_split_val)
    return train, val
def _gather_healthy_segments(
    loaders: Iterable[TestDatasetLoader],
    modality: Literal["acoustic", "vibration"],
    cfg: V1SSLConfig,
) -> list[_PrecomputedSegment]:
    """Collect every healthy (non-anomaly) recording across all loaders.

    Healthy = `s.is_anomaly is False`.  This includes:
      - D1 / D2 with explicit mode labels (Pump / Standstill / Turbine)
      - D3 / D4 speed-bucket recordings whose mode label is unknown
        (the campaign protocol did not record which of the three modes
        the unit was in; the encoder discovers it via K-means at inference).
    The label-leakage invariant is preserved: SSL training never reads
    `mode_label`; only the cluster-purity sanity gate does, and only for
    segments where `mode_label is not None`.
    """
    out: list[_PrecomputedSegment] = []
    for loader in loaders:
        for s in loader.list_segments():
            if s.is_anomaly:
                continue
            pre = _precompute_segment(s, modality, cfg)
            if pre is not None:
                out.append(pre)
    return out
def _gather_labeled_segments(
    loaders: Iterable[TestDatasetLoader],
    modality: Literal["acoustic", "vibration"],
    cfg: V1SSLConfig,
) -> list[_PrecomputedSegment]:
    """Collect only segments that have an explicit `mode_label` *and* are
    not anomaly recordings.  Used by the K=3 cluster-purity sanity gate.

    On the current four-campaign setup this yields D1 + D2 mode folders,
    i.e. recordings whose Pump / Standstill / Turbine label is ground truth.
    D3 / D4 speed buckets and all RandomFault recordings are excluded.
    """
    out: list[_PrecomputedSegment] = []
    healthy = set(cfg.healthy_modes)
    for loader in loaders:
        for s in loader.list_segments():
            if s.is_anomaly:
                continue
            if s.mode_label is None or s.mode_label not in healthy:
                continue
            pre = _precompute_segment(s, modality, cfg)
            if pre is not None:
                out.append(pre)
    return out
