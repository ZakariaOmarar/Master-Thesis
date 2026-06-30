"""V2 SSL paired data pipeline: dataset, sampler, augmentation, splits."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.data as tud

from ...features.audio_spectral import compute_encoder_input_stack, compute_log_mel_spectrogram
from ...features.vibration_temporal import compute_vibration_input_stack
from ...ingestion.test_dataset_loader import TestDatasetLoader, TestDatasetSegment
from .v2_ssl_config import (
    V2SSLConfig,
    _dataset_idx,
)


@dataclass
class _PairedSegment:
    """One segment's full-duration acoustic + vibration features (paired)."""

    acoustic_features: np.ndarray  # (n_mics, 2, F, T_ac)
    acoustic_xyz: np.ndarray  # (n_mics, 3)
    acoustic_fs: float
    vibration_features: np.ndarray  # (n_vib, 3, T_vib)
    vibration_xyz: np.ndarray  # (n_vib, 3)
    vibration_fs: float
    dataset_idx: int
    dataset_id: str  # canonical string id, used to resolve per-dataset scales
    mode_label: str
    recording_id: str
    source_dir: str
def _precompute_paired(
    s: TestDatasetSegment, cfg: V2SSLConfig
) -> _PairedSegment | None:
    # Acoustic — hop_length, n_fft, n_mels come from cfg (defaults to
    # ACOUSTIC_FEATURES per chapter 3 §3.4.2 grid sweep).
    if cfg.use_cwt:
        acoustic_features = compute_encoder_input_stack(
            s.segment.mic_data,
            fs=int(s.segment.mic_sample_rate),
            n_mels=cfg.n_mels,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            cwt_n_scales=cfg.cwt_n_scales,
            standardize=cfg.standardize_acoustic,
        )
    else:
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
        acoustic_features = np.stack(mels, axis=0)

    vibration_features = compute_vibration_input_stack(
        s.segment.accel_data,
        sample_rate=float(s.segment.accel_sample_rate),
        kurtosis_window_seconds=cfg.vib_kurtosis_window_seconds,
        min_kurtosis_samples=cfg.vib_min_kurtosis_samples,
        crest_factor_window_seconds=cfg.vib_crest_factor_window_seconds,
        min_crest_factor_samples=cfg.vib_min_crest_factor_samples,
        standardize=cfg.standardize_vibration,
    )

    if acoustic_features.shape[-1] < 2 or vibration_features.shape[-1] < 2:
        return None

    return _PairedSegment(
        acoustic_features=acoustic_features.astype(np.float32),
        acoustic_xyz=s.mic_positions.astype(np.float32),
        acoustic_fs=float(s.segment.mic_sample_rate) / float(cfg.hop_length),
        vibration_features=vibration_features.astype(np.float32),
        vibration_xyz=s.vib_positions.astype(np.float32),
        vibration_fs=float(s.segment.accel_sample_rate),
        dataset_idx=_dataset_idx(s.dataset_id),
        dataset_id=str(s.dataset_id),
        mode_label=s.mode_label or "Unknown",
        recording_id=s.recording_id,
        source_dir=str(s.source_dir),
    )


# ---------------------------------------------------------------------------
# Paired windowed dataset
# ---------------------------------------------------------------------------
def _resolve_paired_segment_scales(
    cfg: V2SSLConfig, dataset_id: str
) -> tuple[tuple[float, ...], float]:
    """Resolve `(scales_in_seconds, stride_ratio)` for one paired segment.

    Per-dataset override → global tuple → legacy single-scale fallback —
    same precedence as :func:`v1_ssl._resolve_segment_scales`.
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
class _PairedWindowedDataset(tud.Dataset):
    """Yields paired acoustic + vibration windows aligned by wall-clock start.

    Per-segment frame counts (acoustic + vibration) are computed from the
    segment's own feature rates so a 2-second window over D4 raw vibration
    (~376 Hz) keeps all 752 samples while a 2-second window over D1 peak
    vibration (4 Hz) keeps all 8.  When `cfg.window_scales_seconds[_per_dataset]`
    is set, every (segment, scale) pair materialises its own window list;
    both `n_ac` AND `n_vib` are computed from the same per-batch scale so
    the wall-clock pairing (sample 0 of mic ↔ sample 0 of vib) is preserved
    inside each window.  The grouped batch sampler buckets by
    `(n_mics, n_vib, frames_ac, frames_vib)` so that a single `torch.stack`
    never has to reconcile different sample counts.
    """

    def __init__(
        self,
        segments: list[_PairedSegment],
        cfg: V2SSLConfig,
        *,
        enable_mixup: bool = False,
    ) -> None:
        self.cfg = cfg
        self._segments = segments
        self._mixup_enabled = bool(enable_mixup and cfg.mixup_alpha > 0.0)
        if not segments:
            self._refs: list[tuple[int, int, int, int, int]] = []
            self._partner_pool: dict[tuple, list[int]] = {}
            return

        self._refs = []
        for si, seg in enumerate(segments):
            scales, stride_ratio = _resolve_paired_segment_scales(cfg, seg.dataset_id)
            T_ac = int(seg.acoustic_features.shape[-1])
            T_vib = int(seg.vibration_features.shape[-1])
            for scale_s in scales:
                n_ac = max(2, int(round(scale_s * seg.acoustic_fs)))
                n_vib = max(2, int(round(scale_s * seg.vibration_fs)))
                stride_ac = max(1, int(round(scale_s * stride_ratio * seg.acoustic_fs)))
                if T_ac < n_ac or T_vib < n_vib:
                    continue
                for start_ac in range(0, T_ac - n_ac + 1, stride_ac):
                    t_start = start_ac / max(seg.acoustic_fs, 1e-9)
                    start_vib = int(round(t_start * seg.vibration_fs))
                    if start_vib + n_vib > T_vib:
                        continue
                    self._refs.append((si, start_ac, start_vib, n_ac, n_vib))

        # Partner-pool index keyed by (dataset_idx, n_mics, n_vib, n_ac, n_vib,
        # mode_label) — the V2 grouped batch sampler's stack-safety key plus
        # mode_label.  Both modalities use the same partner so the cross-modal
        # pairing is preserved.
        self._partner_pool = {}
        if self._mixup_enabled:
            for ref_i, (si, _sa, _sv, n_ac, n_vib) in enumerate(self._refs):
                seg = self._segments[si]
                key = (
                    int(seg.dataset_idx),
                    int(seg.acoustic_features.shape[0]),
                    int(seg.vibration_features.shape[0]),
                    n_ac, n_vib, seg.mode_label,
                )
                self._partner_pool.setdefault(key, []).append(ref_i)

    def __len__(self) -> int:
        return len(self._refs)

    def _sample_mixup_partner(self, idx: int) -> int | None:
        import random as _r

        si, _sa, _sv, n_ac, n_vib = self._refs[idx]
        seg = self._segments[si]
        key = (
            int(seg.dataset_idx),
            int(seg.acoustic_features.shape[0]),
            int(seg.vibration_features.shape[0]),
            n_ac, n_vib, seg.mode_label,
        )
        candidates = self._partner_pool.get(key)
        if not candidates or len(candidates) < 2:
            return None
        anchor_rec = seg.recording_id
        for _ in range(6):
            cand = candidates[_r.randrange(len(candidates))]
            if cand == idx:
                continue
            cand_seg = self._segments[self._refs[cand][0]]
            if cand_seg.recording_id != anchor_rec:
                return cand
        return None

    def __getitem__(self, idx: int):
        si, sa, sv, n_ac, n_vib = self._refs[idx]
        seg = self._segments[si]
        ac = seg.acoustic_features[..., sa : sa + n_ac]
        vib = seg.vibration_features[..., sv : sv + n_vib]
        ac_t = torch.from_numpy(np.ascontiguousarray(ac))
        vib_t = torch.from_numpy(np.ascontiguousarray(vib))
        if self._mixup_enabled:
            partner = self._sample_mixup_partner(idx)
            if partner is not None:
                import random as _r

                p_si, p_sa, p_sv, _, _ = self._refs[partner]
                p_seg = self._segments[p_si]
                p_ac = p_seg.acoustic_features[..., p_sa : p_sa + n_ac]
                p_vib = p_seg.vibration_features[..., p_sv : p_sv + n_vib]
                lam = _r.betavariate(self.cfg.mixup_alpha, self.cfg.mixup_alpha)
                lam = max(lam, 1.0 - lam)
                ac_t = lam * ac_t + (1.0 - lam) * torch.from_numpy(np.ascontiguousarray(p_ac))
                vib_t = lam * vib_t + (1.0 - lam) * torch.from_numpy(np.ascontiguousarray(p_vib))
        return {
            "ac_feat": ac_t,
            "vib_feat": vib_t,
            "ac_xyz": torch.from_numpy(seg.acoustic_xyz),
            "vib_xyz": torch.from_numpy(seg.vibration_xyz),
            "dataset_idx": int(seg.dataset_idx),
            "mode_label": seg.mode_label,
            "recording_id": seg.recording_id,
        }
class _PairedGroupedBatchSampler(tud.Sampler[list[int]]):
    """Group paired-window indices by (n_mics, n_vib) so each batch is stackable."""

    def __init__(
        self,
        dataset: _PairedWindowedDataset,
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
        # Bucket by (n_mics, n_vib, frames_ac, frames_vib) — D4 raw vibration
        # (376 Hz) and D1/D2 peak vibration (4 Hz) produce different per-window
        # frame counts even when their channel counts match.
        groups: dict[tuple[int, int, int, int], list[int]] = {}
        for i, (si, _sa, _sv, n_ac, n_vib) in enumerate(dataset._refs):
            seg = dataset._segments[si]
            key = (
                int(seg.acoustic_features.shape[0]),
                int(seg.vibration_features.shape[0]),
                int(n_ac),
                int(n_vib),
            )
            groups.setdefault(key, []).append(i)
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
def _collate(batch: list[dict]) -> dict:
    ac_feat = torch.stack([b["ac_feat"] for b in batch], dim=0).float()
    vib_feat = torch.stack([b["vib_feat"] for b in batch], dim=0).float()
    ac_xyz = torch.stack([b["ac_xyz"] for b in batch], dim=0).float()
    vib_xyz = torch.stack([b["vib_xyz"] for b in batch], dim=0).float()
    dataset_idx = torch.tensor([b["dataset_idx"] for b in batch], dtype=torch.long)
    return {
        "ac_feat": ac_feat,
        "vib_feat": vib_feat,
        "ac_xyz": ac_xyz,
        "vib_xyz": vib_xyz,
        "dataset_idx": dataset_idx,
        "mode_label": [b["mode_label"] for b in batch],
        "recording_id": [b["recording_id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Augmentation (paired)
# ---------------------------------------------------------------------------
class _PairedAugmenter:
    """Apply the V1 augmentation set to the paired (acoustic, vibration) batch.

    Each modality is augmented independently — there is no cross-modal
    coupling at the augmentation step.  The two SimCLR views are generated by
    calling this object twice per anchor.
    """

    def __init__(self, cfg: V2SSLConfig, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.gen = generator

    def _gain_and_dropout(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
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
        if cfg.channel_dropout_p > 0:
            drop = (
                torch.rand(*x.shape[:2], generator=self.gen, device=x.device)
                < cfg.channel_dropout_p
            )
            keep = (~drop).float()
            while keep.ndim < x.ndim:
                keep = keep.unsqueeze(-1)
            x = x * keep
        return x

    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 2, F, T)
        cfg = self.cfg
        if cfg.spec_augment_freq_mask <= 0 and cfg.spec_augment_time_mask <= 0:
            return x
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

    def __call__(
        self, ac: torch.Tensor, vib: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ac = self._spec_augment(self._gain_and_dropout(ac.clone()))
        vib = self._gain_and_dropout(vib.clone())
        return ac, vib


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------
def _time_split_paired_segment(
    seg: _PairedSegment, val_ratio: float
) -> tuple[_PairedSegment | None, _PairedSegment | None]:
    """Paired-segment analogue of `v1_ssl._time_split_segment`.

    Splits a single `_PairedSegment` along its time axes (acoustic and
    vibration sliced consistently in wall-clock seconds) so that
    single-recording mode-label groups still produce both a train and a
    val pseudo-segment.  See the v1 docstring for the methodological
    rationale; the temporal-leakage caveat applies equally here.
    """
    T_ac = int(seg.acoustic_features.shape[-1])
    T_vib = int(seg.vibration_features.shape[-1])
    if T_ac < 4 or T_vib < 4:
        return None, None

    # Anchor the split in wall-clock time so the acoustic and vibration
    # halves cover the same physical interval of the recording.
    t_total_s = T_ac / max(seg.acoustic_fs, 1e-9)
    t_val_s = t_total_s * float(val_ratio)
    n_val_ac = max(2, int(round(t_val_s * seg.acoustic_fs)))
    n_val_vib = max(2, int(round(t_val_s * seg.vibration_fs)))
    n_val_ac = min(n_val_ac, T_ac - 2)
    n_val_vib = min(n_val_vib, T_vib - 2)
    if n_val_ac < 2 or n_val_vib < 2 or T_ac - n_val_ac < 2 or T_vib - n_val_vib < 2:
        return None, None

    train_seg = _PairedSegment(
        acoustic_features=seg.acoustic_features[..., : T_ac - n_val_ac].copy(),
        acoustic_xyz=seg.acoustic_xyz,
        acoustic_fs=seg.acoustic_fs,
        vibration_features=seg.vibration_features[..., : T_vib - n_val_vib].copy(),
        vibration_xyz=seg.vibration_xyz,
        vibration_fs=seg.vibration_fs,
        dataset_idx=seg.dataset_idx,
        dataset_id=seg.dataset_id,
        mode_label=seg.mode_label,
        recording_id=f"{seg.recording_id}__train_half",
        source_dir=seg.source_dir,
    )
    val_seg = _PairedSegment(
        acoustic_features=seg.acoustic_features[..., T_ac - n_val_ac :].copy(),
        acoustic_xyz=seg.acoustic_xyz,
        acoustic_fs=seg.acoustic_fs,
        vibration_features=seg.vibration_features[..., T_vib - n_val_vib :].copy(),
        vibration_xyz=seg.vibration_xyz,
        vibration_fs=seg.vibration_fs,
        dataset_idx=seg.dataset_idx,
        dataset_id=seg.dataset_id,
        mode_label=seg.mode_label,
        recording_id=f"{seg.recording_id}__val_half",
        source_dir=seg.source_dir,
    )
    return train_seg, val_seg
def _split_segments_by_recording(
    segments: list[_PairedSegment], val_ratio: float, seed: int
) -> tuple[list[_PairedSegment], list[_PairedSegment]]:
    """Stratified-by-mode-label held-out split at the recording level.

    Mirrors `v1_ssl._split_segments_by_recording`: modes with ≥ 2
    recordings use the standard recording-level stratified split; modes
    with exactly 1 recording are **time-split** so they still appear in
    val (see `_time_split_paired_segment`).  Unlabeled D3 / D4
    speed-bucket recordings (mode_label = None) follow the
    ≥ 2 / == 1 branching like any other label.
    """
    rng = np.random.default_rng(seed)
    rec_to_label: dict[tuple, str | None] = {}
    rec_to_seg: dict[tuple, _PairedSegment] = {}
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
    time_split_train: list[_PairedSegment] = []
    time_split_val: list[_PairedSegment] = []
    for recs in by_label.values():
        recs_shuffled = list(recs)
        rng.shuffle(recs_shuffled)
        if len(recs_shuffled) == 1:
            only_key = recs_shuffled[0]
            tr, va = _time_split_paired_segment(rec_to_seg[only_key], val_ratio)
            if tr is not None and va is not None:
                time_split_train.append(tr)
                time_split_val.append(va)
            else:
                train_keys.update(recs_shuffled)
            continue
        n_val_for_label = max(1, int(round(len(recs_shuffled) * val_ratio)))
        n_val_for_label = min(n_val_for_label, len(recs_shuffled) - 1)
        val_keys.update(recs_shuffled[:n_val_for_label])
        train_keys.update(recs_shuffled[n_val_for_label:])

    if not train_keys and not time_split_train:
        train_keys = {val_keys.pop()}
    train = [s for s in segments if (s.dataset_idx, s.recording_id, s.source_dir) in train_keys]
    val = [s for s in segments if (s.dataset_idx, s.recording_id, s.source_dir) in val_keys]
    train.extend(time_split_train)
    val.extend(time_split_val)
    return train, val
def _gather_paired_segments(
    loaders: Iterable[TestDatasetLoader], cfg: V2SSLConfig
) -> list[_PairedSegment]:
    """Collect every healthy (non-anomaly) paired segment across all loaders.

    Healthy = `s.is_anomaly is False`.  Includes D3 / D4 speed-bucket
    recordings whose mode is unknown — they contribute SSL training signal
    even without a mode label.  The label-leakage invariant is preserved
    because SSL never reads `mode_label`.
    """
    out: list[_PairedSegment] = []
    for loader in loaders:
        for s in loader.list_segments():
            if s.is_anomaly:
                continue
            pre = _precompute_paired(s, cfg)
            if pre is not None:
                out.append(pre)
    return out
def _gather_labeled_segments(
    loaders: Iterable[TestDatasetLoader], cfg: V2SSLConfig
) -> list[_PairedSegment]:
    """Collect non-anomaly segments with an explicit mode_label — used by
    the K=3 RQ1 cluster-purity evaluation."""
    out: list[_PairedSegment] = []
    healthy = set(cfg.healthy_modes)
    for loader in loaders:
        for s in loader.list_segments():
            if s.is_anomaly:
                continue
            if s.mode_label is None or s.mode_label not in healthy:
                continue
            pre = _precompute_paired(s, cfg)
            if pre is not None:
                out.append(pre)
    return out
