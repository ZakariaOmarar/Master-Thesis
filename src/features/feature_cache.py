"""Disk memoization for the expensive, deterministic feature extractors.

The CWT acoustic stack (`compute_encoder_input_stack`), the log-mel spectrogram
(`compute_log_mel_spectrogram`), and the vibration stack
(`compute_vibration_input_stack`) are *pure functions* of their input waveform
and a handful of scalar parameters.  Yet the full pipeline recomputes them many
times over: V1 acoustic, V1 vibration, V2, the V2 A1 ablation (identical
features — `drop_vibration` is applied later, in the forward pass), the V4
sample builders, and — across a multi-seed sweep — every seed re-extracts the
*same* per-recording features (features do not depend on the model seed).  Each
extraction of a 96-mel / 64-scale CWT over a multi-minute recording costs
seconds.

This module caches those outputs to disk **only when** the environment variable
``HYDRO_FEATURE_CACHE_DIR`` is set, so the default behaviour is byte-identical to
no caching.  It is engineered to be *provably result-neutral* — a cache hit
returns exactly what the function would have computed, never an approximation —
because the key is a SHA-256 over:

  * a cache-format version constant,
  * the function's qualified name **and a hash of its own source code** (so
    editing the extractor's body invalidates its entries automatically).  Note
    this hash covers the *wrapped* function only, not helper functions it calls
    (e.g. ``compute_cwt_scalogram``); if you change a callee's maths, bump
    ``_CACHE_VERSION`` below (or delete the cache dir) to invalidate, and
  * the **fully-bound** call arguments *including defaults* (so a changed default
    — e.g. ``cwt_min_freq_hz`` — produces a different key instead of silently
    colliding with a stale entry), each ndarray hashed by shape + dtype + raw
    bytes and every scalar by ``repr``.

Any change to the input waveform, any parameter, any default, or the extractor's
source forces a recompute.  A stale hit is therefore impossible short of a hash
collision, and the cached value is the function's own output.

Enable it for a run with, e.g.::

    HYDRO_FEATURE_CACHE_DIR=.feature_cache python -m src.modeling.orchestration.full_run
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

# Bump only for a change to the on-disk format or the keying scheme itself; the
# per-function source hash already invalidates entries when an extractor changes.
_CACHE_VERSION = 2


def _cache_dir() -> Path | None:
    """Return the cache directory from the env var, or None (caching off)."""
    raw = os.environ.get("HYDRO_FEATURE_CACHE_DIR")
    if not raw:
        return None
    return Path(raw)


def _hash_obj(h, obj: object) -> None:
    """Fold one argument into the running hash, exactly and recursively."""
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        h.update(b"ndarray|")
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    elif isinstance(obj, (tuple, list)):
        h.update(b"seq|")
        for x in obj:
            _hash_obj(h, x)
    elif isinstance(obj, dict):
        h.update(b"dict|")
        for k in sorted(obj, key=repr):
            h.update(repr(k).encode())
            _hash_obj(h, obj[k])
    else:
        h.update(("scalar|" + repr(obj)).encode())


def _make_key(qualname: str, src_hash: str, bound_arguments: dict) -> str:
    h = hashlib.sha256()
    h.update(f"v{_CACHE_VERSION}|{qualname}|{src_hash}|".encode())
    for k in sorted(bound_arguments):
        h.update(repr(k).encode())
        _hash_obj(h, bound_arguments[k])
    return h.hexdigest()


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    """Write `arr` to `path` atomically so a crash never leaves a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a .npy-suffixed temp so np.save writes it in place (no extra append)
    # and there is exactly one temp file to rename — no orphan left behind.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npy")
    os.close(fd)
    try:
        np.save(tmp, arr, allow_pickle=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def disk_cached_feature(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Memoize a pure ``(*arrays, **params) -> np.ndarray`` extractor to disk.

    No-op unless ``HYDRO_FEATURE_CACHE_DIR`` is set.  Only ``np.ndarray`` return
    values are cached (a ``None`` return — e.g. a too-short signal — is passed
    through uncached).  A corrupt cache file is treated as a miss and overwritten.
    If the call cannot be bound to the signature (unexpected), caching is skipped
    and the function runs normally — never a wrong answer, at worst a recompute.
    """
    try:
        src_hash = hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()[:16]
    except Exception:
        src_hash = "nosrc"
    try:
        sig: inspect.Signature | None = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        cdir = _cache_dir()
        if cdir is None or sig is None:
            return fn(*args, **kwargs)
        # Bind to the signature and fill defaults so the key reflects every
        # effective parameter, not just the ones the caller passed explicitly.
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError:
            return fn(*args, **kwargs)  # binding failed → run uncached, safely
        key = _make_key(fn.__qualname__, src_hash, dict(bound.arguments))
        path = cdir / f"{key}.npy"
        if path.exists():
            try:
                return np.load(path, allow_pickle=False)
            except Exception:
                pass  # corrupt / partial → recompute and overwrite
        out = fn(*args, **kwargs)
        if isinstance(out, np.ndarray):
            try:
                _atomic_save(path, out)
            except Exception:
                pass  # cache is best-effort; never fail the computation
        return out

    return wrapper


__all__ = ["disk_cached_feature"]
