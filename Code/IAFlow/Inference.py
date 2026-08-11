"""NumPy-facing inference helpers for trained autoencoders."""

from __future__ import annotations

import numpy as np
import torch

from .Models import Conv1dAutoEncoder
from .Data import NormalizationStats

__all__ = ["encode_A_theta", "reconstruct_A_theta"]


def _prepare_A_theta(
    values: np.ndarray,
    model: Conv1dAutoEncoder,
    normalization: NormalizationStats,
) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float32)
    was_single = array.ndim == 2
    if was_single:
        array = array[None, ...]
    expected = model.input_shape
    if array.ndim != 3 or tuple(array.shape[1:]) != expected:
        raise ValueError(f"A_theta must have shape {expected} or (batch, {expected[0]}, {expected[1]}).")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("A_theta must contain finite, strictly positive values.")
    log_values = np.log10(array)
    return normalization.normalize(log_values), was_single


@torch.inference_mode()
def encode_A_theta(
    values: np.ndarray,
    model: Conv1dAutoEncoder,
    normalization: NormalizationStats,
    *,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Encode one or more physical positive ``A_theta`` surfaces."""
    normalized, was_single = _prepare_A_theta(values, model, normalization)
    selected_device = torch.device(device) if device is not None else next(model.parameters()).device
    tensor = torch.from_numpy(normalized).to(selected_device)
    latent = model.encode(tensor).cpu().numpy()
    return latent[0] if was_single else latent


@torch.inference_mode()
def reconstruct_A_theta(
    values: np.ndarray,
    model: Conv1dAutoEncoder,
    normalization: NormalizationStats,
    *,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Round-trip one or more physical positive ``A_theta`` surfaces."""
    normalized, was_single = _prepare_A_theta(values, model, normalization)
    selected_device = torch.device(device) if device is not None else next(model.parameters()).device
    tensor = torch.from_numpy(normalized).to(selected_device)
    reconstructed = model(tensor).cpu().numpy()
    reconstructed_log10 = normalization.denormalize(reconstructed)
    physical = np.power(10.0, reconstructed_log10, dtype=np.float32)
    return physical[0] if was_single else physical
