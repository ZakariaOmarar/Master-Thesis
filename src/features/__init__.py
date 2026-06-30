"""Feature extractors for the V0–V5 pipeline.

  - `acoustic_representations`: STFT / CWT / MFCC primitives (CWT used by `audio_spectral`).
  - `audio_spectral`:            log-mel + CWT stack for the V1 acoustic encoder.
  - `vibration_temporal`:        amplitude + envelope + rolling kurtosis for the V1 vibration encoder.
"""

from .acoustic_representations import (
    build_cwt_mfcc_encoder_input,
    compute_cwt_scalogram,
    compute_cwt_scalogram_stack,
    compute_log_stft_spectrogram,
    compute_mfcc_stack,
    compute_mfcc_with_deltas,
    compute_stft_stack,
)
from .audio_spectral import (
    compute_encoder_input_stack,
    compute_log_mel_spectrogram,
)
from .vibration_temporal import compute_vibration_input_stack

__all__ = [
    "build_cwt_mfcc_encoder_input",
    "compute_cwt_scalogram",
    "compute_cwt_scalogram_stack",
    "compute_encoder_input_stack",
    "compute_log_mel_spectrogram",
    "compute_log_stft_spectrogram",
    "compute_mfcc_stack",
    "compute_mfcc_with_deltas",
    "compute_stft_stack",
    "compute_vibration_input_stack",
]
