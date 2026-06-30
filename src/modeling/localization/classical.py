"""Classical (signal-processing) localization primitives.

Pure-NumPy GCC-PHAT and SRP-PHAT shared by two consumers:

  * the V0 ``srp_phat_baseline`` (the RQ3 classical reference), and
  * V4 acoustic feature extraction (``v4_features.compute_srp_phat_volume``),
    which seeds the learned Cross3D head with an SRP-PHAT prior.

Both functions are dataset-agnostic: the caller supplies the mic geometry, the
candidate grid, and the mic-pair index list, so nothing here is tied to a
specific rig. Sensor geometry lives in ``src/config/constants.py`` (the ROW II
reference array) and in each dataset's YAML; this module holds only the math.

Public API:
  gcc_phat      — single-pair GCC-PHAT cross-correlation
  srp_phat_3d   — vectorised SRP-PHAT power map over a 3-D candidate grid
"""

from __future__ import annotations

import numpy as np


def gcc_phat(
    x_i: np.ndarray,
    x_j: np.ndarray,
    max_delay_samples: int,
    n_fft: int | None = None,
    oversample: int = 1,
    phat_beta: float = 1.0,
    linear: bool = False,
) -> np.ndarray:
    """Compute the GCC-PHAT cross-correlation vector for one mic pair.

    Args:
        x_i: 1D float array of audio samples for mic i.
        x_j: 1D float array of audio samples for mic j.
        max_delay_samples: Half-width of the output in *input* samples.
        n_fft: FFT size. Defaults to len(x_i); increase to next power-of-two
            for efficiency on non-power-of-two window lengths.
        oversample: Integer GCC up-sampling factor (default 1 = unchanged).
            ``oversample = O`` interpolates the cross-correlation to a lag grid
            ``O×`` finer than one input sample by zero-padding the one-sided
            cross-spectrum before the inverse FFT — standard frequency-domain
            sinc interpolation.  The returned vector then has length
            ``2 * max_delay_samples * O + 1`` and a lag step of ``1/O`` samples,
            so the SRP-PHAT peak can resolve below the input-sample grid (the
            16 kHz acoustic path's main precision bottleneck; the accel
            multilateration already does this with parabolic interpolation).
            The caller must pass ``fs * O`` to :func:`srp_phat_3d` so the
            steered-delay indexing matches the finer grid.  pysoundlocalization
            (Haller & Länzlinger) uses ``interp = 16`` by default.
        phat_beta: PHAT whitening exponent in ``G / |G|**β`` (default 1.0 = full
            PHAT).  ``β < 1`` keeps some magnitude weighting, more robust than
            full whitening at low SNR (full PHAT amplifies empty noise bins);
            ``β = 0`` is plain cross-correlation.
        linear: when True, zero-pad to ``len(x_i) + len(x_j)`` so the inverse
            transform is a **linear** (non-circular) correlation — the
            pysoundlocalization convention, which removes the wrap-around
            artefact circular correlation introduces when the impulse sits near
            a crop edge.  Default False preserves the legacy circular form.

    Returns:
        gcc: float32 array of shape (L,), where ``L = 2 * max_delay_samples * O
            + 1``.  Index ``max_delay_samples * O`` corresponds to zero lag.
    """
    oversample = max(1, int(oversample))
    if linear:
        base_n = len(x_i) + len(x_j)
    elif n_fft is None:
        base_n = len(x_i)
    else:
        base_n = n_fft

    Xi = np.fft.rfft(x_i, n=base_n)
    Xj = np.fft.rfft(x_j, n=base_n)
    G = Xi * np.conj(Xj)
    mag = np.abs(G)
    if phat_beta != 1.0:
        mag = mag ** float(phat_beta)
    G_phat = G / (mag + 1e-8)  # PHAT whitening (β-weighted)

    if oversample > 1:
        # Zero-pad the one-sided spectrum to upsample the IFFT in time → a
        # lag grid O× finer.  Multiply by O to preserve the peak amplitude
        # (irfft divides by the output length).
        n_out = base_n * oversample
        n_bins_out = n_out // 2 + 1
        G_pad = np.zeros(n_bins_out, dtype=G_phat.dtype)
        G_pad[: G_phat.shape[0]] = G_phat
        gcc_full = np.fft.irfft(G_pad, n=n_out) * oversample
        half = max_delay_samples * oversample
    else:
        gcc_full = np.fft.irfft(G_phat, n=base_n)
        half = max_delay_samples
    gcc_full = np.fft.fftshift(gcc_full)  # center at zero lag

    center = len(gcc_full) // 2
    gcc = gcc_full[center - half : center + half + 1]
    return gcc.astype(np.float32)  # shape (L,)


def srp_phat_3d(
    gcc_stack: np.ndarray,
    mic_xyz: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    fs: float,
    c: float = 343.0,
    *,
    mic_pairs: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Vectorised SRP-PHAT over a 3-D candidate grid.

    For each candidate point the expected TDOA of every mic pair indexes into
    the pair's GCC-PHAT vector; the summed responses form the steered-response
    power map. Dataset-agnostic — the caller passes the mic geometry and pairs.

    Args:
        gcc_stack: (n_pairs, L) averaged GCC-PHAT stack.
        mic_xyz:   (n_mics, 3) mic positions in metres.
        grid_x/y/z: 1-D coordinate arrays defining the search grid (metres).
        fs:        Sample rate (Hz).
        c:         Speed of sound (m/s).
        mic_pairs: List of (i, j) index pairs. Defaults to all unique pairs
            derived from ``mic_xyz``.

    Returns:
        float32 array of shape (Nx, Ny, Nz).
    """
    if mic_pairs is None:
        n_mics = mic_xyz.shape[0]
        mic_pairs = [(i, j) for i in range(n_mics) for j in range(i + 1, n_mics)]
    max_delay = gcc_stack.shape[1] // 2
    L = gcc_stack.shape[1]

    gx, gy, gz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid_pts = np.stack([gx, gy, gz], axis=-1)  # (Nx, Ny, Nz, 3)
    srp = np.zeros((len(grid_x), len(grid_y), len(grid_z)), dtype=np.float32)

    for k, (i, j) in enumerate(mic_pairs):
        pi = mic_xyz[i]
        pj = mic_xyz[j]
        di = np.linalg.norm(grid_pts - pi, axis=-1)
        dj = np.linalg.norm(grid_pts - pj, axis=-1)
        tdoa_idx = np.round((di - dj) / c * fs).astype(np.int32) + max_delay
        valid = (tdoa_idx >= 0) & (tdoa_idx < L)
        idx_safe = np.clip(tdoa_idx, 0, L - 1)
        srp += (gcc_stack[k, idx_safe] * valid).astype(np.float32)

    return srp
