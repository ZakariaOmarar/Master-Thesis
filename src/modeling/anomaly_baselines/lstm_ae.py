"""V0 LSTM-AE anomaly baseline on log-mel windows (Khamaisi reference).

Self-contained: reads via `TestDatasetLoader`, computes log-mel via
`src/features/audio_spectral.py`, fits an LSTM-AE on healthy windows, scores
RandomFault windows by per-window reconstruction MSE.  No coupling to the
existing CNF latent cache.

Per-window scoring is converted to per-mode anomaly thresholds following the
same 95th/99th-percentile logic as `src/modeling/mode/p5_anomaly/per_mode_baseline.py`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud

from ...config import describe_device, resolve_device
from ...features.audio_spectral import compute_log_mel_spectrogram
from ...ingestion.test_dataset_loader import (
    TestDatasetLoader,
    TestDatasetSegment,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V0Config:
    """Hyperparameters for the V0 LSTM-AE baseline."""

    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    window_seconds: float = 1.0  # one log-mel window
    window_overlap: float = 0.5  # 50 % overlap between consecutive windows
    hidden_dim: int = 128
    latent_dim: int = 32
    n_layers: int = 2
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.2
    healthy_modes: tuple[str, ...] = ("Pump", "Standstill", "Turbine", "Healthy")
    anomaly_modes: tuple[str, ...] = ("RandomFault",)
    seed: int = 42
    device: str = "auto"

    # R2.4 / 2026-05-16 — when provided, OVERRIDES ``n_mels`` as the LSTM
    # input feature dimension.  Used by the vibration-only V0 baseline,
    # which feeds 3-channel ``[amplitude, envelope, kurtosis]`` features
    # to the same trainer.  ``None`` (default) preserves the acoustic
    # behaviour: input dim = ``n_mels``.
    feature_dim: int | None = None

    # Vibration extractor parameters (only consulted when the caller passes
    # ``extract_vibration_temporal_windows`` as the extract_fn).  These
    # mirror `compute_vibration_input_stack`'s physical-time knobs; the
    # channel-2 statistic auto-selects between excess kurtosis (>=31
    # samples/window) and crest factor based on the segment's
    # accel_sample_rate.
    vib_kurtosis_window_seconds: float = 0.10
    vib_min_kurtosis_samples: int = 31
    vib_crest_factor_window_seconds: float = 1.0
    vib_min_crest_factor_samples: int = 4

    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_feature_dim(self) -> int:
        return self.feature_dim if self.feature_dim is not None else self.n_mels


# ---------------------------------------------------------------------------
# Feature extraction → sliding windows
# ---------------------------------------------------------------------------


def extract_vibration_temporal_windows(
    segment: TestDatasetSegment,
    cfg: V0Config,
    *,
    pool: str = "mean",
) -> np.ndarray:
    """V0 vibration-only feature extractor — mirror of `extract_log_mel_windows`.

    Computes the three-channel ``[amplitude, envelope, kurtosis]`` stack
    via :func:`src.features.vibration_temporal.compute_vibration_input_stack`,
    pools across accelerometer channels (mean/max), slides windows over the
    pooled time-series.  Returns ``(n_windows, frames_per_window, 3)``.

    Window cadence is computed at the *accelerometer* sample rate (slower
    than the acoustic ``mic_sample_rate``) so the window seconds knob means
    the same wall-clock duration as the acoustic extractor.
    """
    from ...features.vibration_temporal import compute_vibration_input_stack

    fs_v = float(segment.segment.accel_sample_rate)
    if fs_v <= 0:
        return np.zeros((0, 1, 3), dtype=np.float32)
    stack = compute_vibration_input_stack(
        segment.segment.accel_data,
        sample_rate=fs_v,
        kurtosis_window_seconds=cfg.vib_kurtosis_window_seconds,
        min_kurtosis_samples=cfg.vib_min_kurtosis_samples,
        crest_factor_window_seconds=cfg.vib_crest_factor_window_seconds,
        min_crest_factor_samples=cfg.vib_min_crest_factor_samples,
    )  # (n_accel, 3, T_vib)
    if pool == "mean":
        pooled = stack.mean(axis=0)  # (3, T_vib)
    elif pool == "max":
        pooled = stack.max(axis=0)
    else:
        raise ValueError(f"unknown pool {pool!r}")

    frames_per_window = max(1, int(round(cfg.window_seconds * fs_v)))
    step = max(1, int(round(frames_per_window * (1.0 - cfg.window_overlap))))
    T_vib = pooled.shape[1]
    if T_vib < frames_per_window:
        return np.zeros((0, frames_per_window, 3), dtype=np.float32)
    windows = []
    for start in range(0, T_vib - frames_per_window + 1, step):
        windows.append(pooled[:, start : start + frames_per_window].T)  # (T, 3)
    if not windows:
        return np.zeros((0, frames_per_window, 3), dtype=np.float32)
    return np.stack(windows, axis=0).astype(np.float32)


def extract_log_mel_windows(
    segment: TestDatasetSegment,
    cfg: V0Config,
    *,
    pool: str = "mean",
) -> np.ndarray:
    """Compute log-mel for every mic, pool across mics, then slide windows.

    Returns ``(n_windows, n_frames_per_window, n_mels)`` float32.

    The pooling strategy is mean over mic channels, so the V0 baseline ingests a
    single mel-spectrogram per recording.  This is the simplest unconditional
    baseline; the channel-aware fusion arrives in V1+.
    """
    fs = int(segment.segment.mic_sample_rate)
    # V0 baselines deliberately use `cfg.hop_length=512` (coarse 31.25 Hz
    # acoustic frame rate) for fast baseline computation.  They do not go
    # through V2's cross-attention, so cross-modal grid alignment (the
    # registry's per-dataset hop, see `hop_for_dataset` in v2_ssl) does not
    # apply here.  Keep V0 on its own STFT params.
    hop = cfg.hop_length
    mels = []
    for ch in range(segment.segment.n_mic_channels):
        m = compute_log_mel_spectrogram(
            segment.segment.mic_data[ch],
            fs=fs,
            n_fft=cfg.n_fft,
            hop_length=hop,
            n_mels=cfg.n_mels,
        )
        mels.append(m)
    arr = np.stack(mels, axis=0)  # (n_mics, n_mels, n_frames)
    if pool == "mean":
        pooled = arr.mean(axis=0)
    elif pool == "max":
        pooled = arr.max(axis=0)
    else:
        raise ValueError(f"unknown pool {pool!r}")
    # pooled: (n_mels, n_frames)

    frames_per_window = max(1, int(round(cfg.window_seconds * fs / hop)))
    step = max(1, int(round(frames_per_window * (1.0 - cfg.window_overlap))))
    n_frames = pooled.shape[1]
    if n_frames < frames_per_window:
        return np.zeros((0, frames_per_window, cfg.n_mels), dtype=np.float32)

    windows = []
    for start in range(0, n_frames - frames_per_window + 1, step):
        windows.append(pooled[:, start : start + frames_per_window].T)  # (T, F)
    if not windows:
        return np.zeros((0, frames_per_window, cfg.n_mels), dtype=np.float32)
    return np.stack(windows, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LSTMAutoencoderV0(nn.Module):
    """Two-layer LSTM encoder + decoder, latent bottleneck.

    The encoder consumes a mel sequence ``(B, T, F)`` and produces a single
    latent ``(B, D_lat)`` from the last hidden state.  The decoder unrolls T
    steps from a repeated latent and reconstructs ``(B, T, F)``.

    Reconstruction error is the per-window mean squared error.
    """

    def __init__(self, cfg: V0Config) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.effective_feature_dim
        self.encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.from_latent = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.decoder = nn.LSTM(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.output = nn.Linear(cfg.hidden_dim, in_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)  # h_n: (n_layers, B, hidden)
        return self.to_latent(h_n[-1])  # (B, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        T = x.shape[1]
        seed = self.from_latent(z)  # (B, hidden)
        decoder_in = seed.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden)
        out, _ = self.decoder(decoder_in)
        return self.output(out)  # (B, T, F)

    @torch.no_grad()
    def reconstruction_score(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.forward(x)
        return ((x_hat - x) ** 2).mean(dim=(1, 2))  # (B,)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    model: LSTMAutoencoderV0
    train_loss_history: list[float]
    val_loss_history: list[float]
    standardiser_mean: np.ndarray
    standardiser_std: np.ndarray
    healthy_train_recordings: list[str]
    healthy_val_recordings: list[str]


def _gather_healthy_windows(
    segments: Iterable[TestDatasetSegment],
    cfg: V0Config,
    extract_fn=extract_log_mel_windows,
) -> tuple[np.ndarray, list[str]]:
    """Collect (n_windows, T, F) plus the recording id of each window.

    ``extract_fn`` (added in R2.4): callable ``(segment, cfg) -> ndarray``
    that yields the per-segment windows.  Default is `extract_log_mel_windows`
    (V0 acoustic baseline).  Pass `extract_vibration_temporal_windows` for
    the V0 vibration baseline.

    Healthy = `is_anomaly=False` (covers D1/D2 mode folders **and** D3/D4
    speed-bucket recordings whose mode is unrecorded — both contribute
    valid healthy training material for the V0 reference model).
    """
    all_windows: list[np.ndarray] = []
    rec_ids_per_window: list[str] = []
    for s in segments:
        if s.is_anomaly:
            continue
        w = extract_fn(s, cfg)
        if w.shape[0] == 0:
            continue
        all_windows.append(w)
        rec_ids_per_window.extend([s.recording_id] * w.shape[0])
    if not all_windows:
        return np.zeros((0, 0, cfg.effective_feature_dim), dtype=np.float32), []
    return np.concatenate(all_windows, axis=0), rec_ids_per_window


def _split_by_recording(
    rec_ids: list[str], val_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(rec_ids))
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_ratio)))
    val_ids = unique[:n_val]
    train_ids = unique[n_val:]
    if not train_ids:
        # Tiny dataset (e.g. D1 with 4 recordings + val_ratio=0.5): keep ≥ 1 train.
        train_ids = [val_ids.pop()]
    return train_ids, val_ids


def train_v0_lstm_ae(
    loader: TestDatasetLoader,
    cfg: V0Config | None = None,
    extract_fn=extract_log_mel_windows,
    *,
    split: tuple[set[str], set[str]] | None = None,
) -> TrainResult:
    """Train the V0 LSTM-AE on healthy recordings of one dataset.

    ``extract_fn`` (added in R2.4): per-segment feature extractor.  Default
    `extract_log_mel_windows` is the V0 acoustic baseline; passing
    `extract_vibration_temporal_windows` gives the V0 vibration baseline
    using the same trainer + model.  The model's input dim follows
    ``cfg.effective_feature_dim`` so the caller must set
    ``cfg.feature_dim = 3`` when using the vibration extractor.

    ``split`` (optional): an explicit ``(train_recording_ids,
    val_recording_ids)`` pair.  When supplied it overrides the internal
    recording split, so a caller (e.g. the RQ2 harness in `v0_evaluation`)
    can train the AE on exactly the same fit pool the other V0 scorers and
    the threshold use — avoiding cross-model leakage.
    """
    cfg = cfg or V0Config()
    segments = loader.list_segments()
    windows, rec_ids = _gather_healthy_windows(segments, cfg, extract_fn=extract_fn)
    return fit_lstm_ae_on_windows(windows, rec_ids, cfg, split=split)


def fit_lstm_ae_on_windows(
    windows: np.ndarray,
    rec_ids: list[str],
    cfg: V0Config | None = None,
    *,
    split: tuple[set[str], set[str]] | None = None,
) -> TrainResult:
    """Train the V0 LSTM-AE on a pre-extracted ``(windows, rec_ids)`` corpus.

    Splitting :func:`train_v0_lstm_ae` into "gather" and "fit" lets the RQ2
    harness pool healthy windows across *several* datasets (the way the
    proposed head is trained on `ANOM_LOADERS`) and feed them here with an
    explicit recording split, while the single-loader entry point stays a thin
    wrapper.  ``windows`` is ``(N, T, F)`` with ``F == cfg.effective_feature_dim``.
    """
    cfg = cfg or V0Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if windows.shape[0] == 0:
        raise RuntimeError("no healthy windows found for V0 training")

    if split is not None:
        train_ids, val_ids = set(split[0]), set(split[1])
    else:
        train_ids, val_ids = _split_by_recording(rec_ids, cfg.val_ratio, cfg.seed)
    train_mask = np.array([r in train_ids for r in rec_ids], dtype=bool)
    val_mask = np.array([r in val_ids for r in rec_ids], dtype=bool)

    train_x = windows[train_mask]
    val_x = windows[val_mask]

    # Standardise per-mel-bin on the training data, apply to val.
    mean = train_x.mean(axis=(0, 1))
    std = train_x.std(axis=(0, 1)) + 1e-6
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    device = resolve_device(cfg.device)
    print(f"V0 LSTM-AE: device={describe_device(device)}")
    model = LSTMAutoencoderV0(cfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()

    pin = device.type == "cuda"
    train_loader = tud.DataLoader(
        tud.TensorDataset(torch.from_numpy(train_x)),
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=pin,
    )
    val_loader = tud.DataLoader(
        tud.TensorDataset(torch.from_numpy(val_x)),
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=pin,
    )

    train_history: list[float] = []
    val_history: list[float] = []

    for _epoch in range(cfg.epochs):
        model.train()
        epoch_train = 0.0
        n_train = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optim.zero_grad()
            xb_hat = model(xb)
            loss = loss_fn(xb_hat, xb)
            loss.backward()
            optim.step()
            epoch_train += float(loss.item()) * xb.shape[0]
            n_train += xb.shape[0]
        train_history.append(epoch_train / max(1, n_train))

        model.eval()
        epoch_val = 0.0
        n_val = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                xb_hat = model(xb)
                loss = loss_fn(xb_hat, xb)
                epoch_val += float(loss.item()) * xb.shape[0]
                n_val += xb.shape[0]
        val_history.append(epoch_val / max(1, n_val))

    return TrainResult(
        model=model,
        train_loss_history=train_history,
        val_loss_history=val_history,
        standardiser_mean=mean.astype(np.float32),
        standardiser_std=std.astype(np.float32),
        healthy_train_recordings=sorted(train_ids),
        healthy_val_recordings=sorted(val_ids),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_recordings(
    model: LSTMAutoencoderV0,
    standardiser_mean: np.ndarray,
    standardiser_std: np.ndarray,
    segments: Iterable[TestDatasetSegment],
    cfg: V0Config,
) -> list[dict]:
    """Score every window of every segment.

    Returns a list of records, one per recording, with per-window scores plus
    the parsed mode/op-condition/spatial labels.
    """
    device = next(model.parameters()).device
    model.eval()
    records: list[dict] = []
    for s in segments:
        windows = extract_log_mel_windows(s, cfg)
        if windows.shape[0] == 0:
            continue
        norm = (windows - standardiser_mean) / standardiser_std
        x = torch.from_numpy(norm).to(device)
        with torch.no_grad():
            scores = model.reconstruction_score(x).cpu().numpy()
        records.append(
            {
                "dataset_id": s.dataset_id,
                "recording_id": s.recording_id,
                "mode_label": s.mode_label,
                "op_condition": s.op_condition,
                "spatial_label": s.spatial_label,
                "n_windows": int(scores.shape[0]),
                "scores": scores.astype(np.float32),
            }
        )
    return records


__all__ = [
    "LSTMAutoencoderV0",
    "TrainResult",
    "V0Config",
    "extract_log_mel_windows",
    "score_recordings",
    "train_v0_lstm_ae",
]
