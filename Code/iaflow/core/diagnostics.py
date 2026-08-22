"""
Reusable full-split tail diagnostics for trained surface autoencoder.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import NormalizationStats
from .metrics import relative_error_from_log10_residual
from .model import AutoEncoder

__all__ = ["evaluate_tail_diagnostics"]


@torch.inference_mode()
def evaluate_tail_diagnostics(
    model: AutoEncoder,
    loader: DataLoader,
    normalization: NormalizationStats,
    device: torch.device,
    *,
    source_indices: np.ndarray,
    worst_count: int = 12,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Compute mean and worst error maps plus the worst reconstructed surfaces.
    
    Arguments:
        model (AutoEncoder):
            Trained surface autoencoder.
        loader (torch.utils.data.DataLoader):
            Ordered loader for the complete validation split.
        normalization (NormalizationStats):
            Training-only normalization paired with the model.
        device (torch.device):
            Device used for inference.
        source_indices (numpy.ndarray):
            Source-row indices in the same order as loader samples.
        worst_count (int):
            Number of largest per-surface maximum errors to retain.
        show_progress (bool):
            Whether to display an evaluation progress bar.
    
    Returns:
        diagnostics (dict[str, Any]):
            Scalar tails, error maps, and worst-surface arrays.
    """
    indices = np.asarray(source_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) != len(loader.dataset):
        raise ValueError("source_indices must match the ordered diagnostic dataset.")
    if worst_count <= 0:
        raise ValueError("worst_count must be positive.")
    model.eval()
    shape = normalization.mean.shape
    relative_sum = np.zeros(shape, dtype=np.float64)
    relative_maximum = np.zeros(shape, dtype=np.float32)
    surface_rms_values: list[np.ndarray] = []
    surface_maximum_values: list[np.ndarray] = []
    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    position = 0
    for target in tqdm(
        loader,
        desc="diagnostics",
        leave=False,
        disable=not show_progress,
    ):
        target = target.to(device, non_blocking=device.type == "cuda")
        prediction = model(target)
        log_residual = (prediction - target) * normalization.scale
        relative = relative_error_from_log10_residual(log_residual)
        relative_values = relative.cpu().numpy()
        relative_sum += relative_values.sum(axis=0, dtype=np.float64)
        relative_maximum = np.maximum(
            relative_maximum,
            relative_values.max(axis=0),
        )
        flattened = relative_values.reshape(len(relative_values), -1)
        rms = np.sqrt(np.mean(np.square(flattened), axis=1))
        maximum = np.max(flattened, axis=1)
        surface_rms_values.append(rms)
        surface_maximum_values.append(maximum)
        target_log10 = normalization.denormalize(target.cpu().numpy())
        prediction_log10 = normalization.denormalize(prediction.cpu().numpy())
        for offset in np.argpartition(
            maximum,
            max(0, len(maximum) - min(worst_count, len(maximum))),
        )[-worst_count:]:
            candidates.append(
                (
                    float(maximum[offset]),
                    position + int(offset),
                    target_log10[offset],
                    prediction_log10[offset],
                )
            )
        candidates = sorted(candidates, key=lambda item: item[0], reverse=True)[
            :worst_count
        ]
        position += len(target)
    if position != len(indices):
        raise RuntimeError("Diagnostic loader did not produce every requested surface.")
    surface_rms = np.concatenate(surface_rms_values)
    surface_maximum = np.concatenate(surface_maximum_values)
    worst_positions = np.asarray([item[1] for item in candidates], dtype=np.int64)
    return {
        "number_of_surfaces": position,
        "surface_relative_rmse_p95": float(np.quantile(surface_rms, 0.95)),
        "surface_relative_rmse_p99": float(np.quantile(surface_rms, 0.99)),
        "surface_relative_maximum_p95": float(
            np.quantile(surface_maximum, 0.95)
        ),
        "surface_relative_maximum_p99": float(
            np.quantile(surface_maximum, 0.99)
        ),
        "maximum_relative_error": float(np.max(surface_maximum)),
        "mean_relative_error_map": (relative_sum / position).astype(np.float32),
        "maximum_relative_error_map": relative_maximum,
        "surface_relative_rmse": surface_rms.astype(np.float32),
        "surface_relative_maximum": surface_maximum.astype(np.float32),
        "worst_split_positions": worst_positions,
        "worst_source_indices": indices[worst_positions],
        "worst_target_log10": np.stack([item[2] for item in candidates]),
        "worst_prediction_log10": np.stack([item[3] for item in candidates]),
    }
