"""
NumPy-facing inference helpers for trained autoencoder.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

from .model import AutoEncoder
from ..core.data import NormalizationStats

__all__ = ["encode_A_theta", "reconstruct_A_theta"]


@contextmanager
def _inference_context(
    model: AutoEncoder,
    device: str | torch.device | None,
) -> Iterator[torch.device]:
    """
    Select an inference device while preserving model device and mode.
    
    Arguments:
        model (AutoEncoder):
            Model temporarily placed in evaluation mode.
        device (str or torch.device or None):
            Optional device used for this inference call.
    
    Returns:
        selected_device (collections.abc.Iterator[torch.device]):
            Context yielding the device holding the model during inference.
    """
    parameter = next(model.parameters())
    original_device = parameter.device
    selected_device = torch.device(device) if device is not None else original_device
    was_training = model.training
    moved = selected_device != original_device
    try:
        if moved:
            model.to(selected_device)
        model.eval()
        yield selected_device
    finally:
        model.train(was_training)
        if moved:
            model.to(original_device)


def _prepare_A_theta(
    values: np.ndarray,
    model: AutoEncoder,
    normalization: NormalizationStats,
) -> tuple[np.ndarray, bool]:
    """
    Check and normalize one or more physical surfaces.
    
    Arguments:
        values (numpy.ndarray):
            One surface or a batch of surfaces.
        model (AutoEncoder):
            Model defining the required input shape.
        normalization (NormalizationStats):
            Training-only transform paired with the model.
    
    Returns:
        result (tuple[numpy.ndarray, bool]):
            Normalized batch and a flag recording whether the input was unbatched.
    """
    array = np.asarray(values, dtype=np.float32)
    was_single = array.ndim == 2
    if was_single:
        array = array[None, ...]
    expected = model.input_shape
    if array.ndim != 3 or tuple(array.shape[1:]) != expected:
        raise ValueError(
            f"Surface must have shape {expected} or "
            f"(batch, {expected[0]}, {expected[1]})."
        )
    
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("Surface must contain finite, strictly positive values.")
    log_values = np.log10(array)
    return normalization.normalize(log_values), was_single


@torch.inference_mode()
def encode_A_theta(
    values: np.ndarray,
    model: AutoEncoder,
    normalization: NormalizationStats,
    *,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """
    Encode one or more physical surfaces.
    
    Arguments:
        values (numpy.ndarray):
            One surface or a batch of surfaces.
        model (AutoEncoder):
            Trained autoencoder used for encoding.
        normalization (NormalizationStats):
            Training-only transform paired with the model.
        device (str or torch.device or None):
            Optional inference device overriding the model's current device.
    
    Returns:
        latent (numpy.ndarray):
            One latent vector or a batch of latent vectors.
    """
    normalized, was_single = _prepare_A_theta(values, model, normalization)
    with _inference_context(model, device) as selected_device:
        tensor = torch.from_numpy(normalized).to(selected_device)
        latent = model.encode(tensor).cpu().numpy()
    return latent[0] if was_single else latent


@torch.inference_mode()
def reconstruct_A_theta(
    values: np.ndarray,
    model: AutoEncoder,
    normalization: NormalizationStats,
    *,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """
    Reconstruct one or more physical surfaces.
    
    Arguments:
        values (numpy.ndarray):
            One surface or a batch of surfaces.
        model (AutoEncoder):
            Trained autoencoder used for the round trip.
        normalization (NormalizationStats):
            Training-only transform paired with the model.
        device (str or torch.device or None):
            Optional inference device overriding the model's current device.
    
    Returns:
        reconstructed (numpy.ndarray):
            Physical-space reconstruction with the same rank as values.
    """
    normalized, was_single = _prepare_A_theta(values, model, normalization)
    with _inference_context(model, device) as selected_device:
        tensor = torch.from_numpy(normalized).to(selected_device)
        reconstructed = model(tensor).cpu().numpy()
    reconstructed_log10 = normalization.denormalize(reconstructed)
    physical = np.power(10.0, reconstructed_log10, dtype=np.float32)
    return physical[0] if was_single else physical
