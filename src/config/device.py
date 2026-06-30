"""Device resolution: honour an explicit device, otherwise pick CUDA when available."""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve a device spec to a concrete torch.device.

    - "auto" / None / "cuda" without a GPU available -> falls back to "cpu".
    - Any other explicit string (e.g. "cuda:1", "cpu", "mps") is honoured.

    Also enables TF32 matmul on Ampere+ when a CUDA device is selected — free
    speedup for SSL/CNF training that costs no measurable accuracy.
    """
    if device is None:
        device = "auto"
    if isinstance(device, torch.device):
        spec = device.type if device.index is None else f"{device.type}:{device.index}"
    else:
        spec = str(device).strip().lower()

    if spec in ("auto", ""):
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif spec.startswith("cuda") and not torch.cuda.is_available():
        resolved = torch.device("cpu")
    else:
        resolved = torch.device(spec)

    if resolved.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    return resolved


def describe_device(device: torch.device) -> str:
    """One-line human-readable device description for log lines."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = f"cuda:{idx}"
        return f"cuda:{idx} ({name})"
    return device.type
