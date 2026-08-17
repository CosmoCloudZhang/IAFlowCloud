"""
Evaluation routines shared by scripts and training.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .architectures import AutoEncoder
from .data import NormalizationStats
from .metrics import ReconstructionMetrics, ReconstructionObjective

__all__ = [
    "evaluate_autoencoder",
    "evaluate_reconstruction_objective",
]


@torch.inference_mode()
def evaluate_reconstruction_objective(
    model: AutoEncoder,
    loader: DataLoader,
    normalization: NormalizationStats,
    device: torch.device,
    *,
    show_progress: bool = True,
) -> dict[str, float]:
    """
    Compute the lightweight reconstruction objective for model selection.
    
    Arguments:
        model (AutoEncoder):
            Trained autoencoder to evaluate.
        loader (torch.utils.data.DataLoader):
            Ordered loader for one stored data split.
        normalization (NormalizationStats):
            Training-only normalization paired with the model.
        device (torch.device):
            Device used for model inference.
        show_progress (bool):
            Whether to display the evaluation progress bar.
    
    Returns:
        metrics (dict[str, float]):
            Normalized loss, log10-space loss, and variance recovery.
    """
    model.eval()
    accumulator = ReconstructionObjective(normalization.scale)
    for target in tqdm(loader, desc="validate", leave=False, disable=not show_progress):
        target = target.to(device, non_blocking=device.type == "cuda")
        accumulator.update(target, model(target))
    return accumulator.compute()


@torch.inference_mode()
def evaluate_autoencoder(
    model: AutoEncoder,
    loader: DataLoader,
    normalization: NormalizationStats,
    device: torch.device,
    *,
    show_progress: bool = True,
) -> dict[str, float]:
    """
    Compute reconstruction metrics for one fixed split.
    
    Arguments:
        model (AutoEncoder):
            Trained autoencoder to evaluate.
        loader (torch.utils.data.DataLoader):
            Ordered loader for one stored data split.
        normalization (NormalizationStats):
            Training-only normalization paired with the model.
        device (torch.device):
            Device used for model inference.
        show_progress (bool):
            Whether to display the evaluation progress bar.
    
    Returns:
        metrics (dict[str, float]):
            Normalized, log10-space, and physical-space reconstruction metrics.
    """
    model.eval()
    accumulator = ReconstructionMetrics(normalization.scale)
    for target in tqdm(loader, desc="evaluate", leave=False, disable=not show_progress):
        target = target.to(device, non_blocking=device.type == "cuda")
        accumulator.update(target, model(target))
    return accumulator.compute()
