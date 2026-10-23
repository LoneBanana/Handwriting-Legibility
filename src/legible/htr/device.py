"""Device selection shared by training, evaluation, and serving."""

from __future__ import annotations

import torch


def pick_device(preference: str = "auto") -> torch.device:
    """``auto`` prefers CUDA, then Apple's MPS, then the CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
