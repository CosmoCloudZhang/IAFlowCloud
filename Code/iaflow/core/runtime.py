"""
Shared runtime selection and reproducibility helpers.
"""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["resolve_device", "set_reproducibility"]


def set_reproducibility(
    seed: int,
    deterministic: bool = False,
) -> None:
    """
    Seed Python, NumPy, and PyTorch without silently forcing slow kernels.
    
    Arguments:
        seed (int):
            Shared random seed.
        deterministic (bool):
            Whether PyTorch should prefer deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_device(
    requested: str = "auto",
) -> torch.device:
    """
    Resolve an explicit or automatic PyTorch compute device.
    
    Arguments:
        requested (str):
            One of auto, cpu, mps, or cuda.
    
    Returns:
        device (torch.device):
            Available device selected for the run.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    
    if requested == "mps" and not (
        torch.backends.mps.is_built() and torch.backends.mps.is_available()
    ):
        raise RuntimeError("Apple MPS was requested but is unavailable.")
    return device
