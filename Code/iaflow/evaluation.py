"""
Evaluation routines shared by scripts and training.
"""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from .architectures import Conv1dAutoEncoder
from .data import NormalizationStats
from .metrics import ReconstructionMetrics

__all__ = ["evaluate_autoencoder"]


@torch.inference_mode()
def evaluate_autoencoder(
    model: Conv1dAutoEncoder,
    loader,
    normalization: NormalizationStats,
    device: torch.device,
    *,
    show_progress: bool = True,
) -> dict[str, float]:
    """
    Compute reconstruction metrics for one fixed split.
    
    Arguments:
        model (Conv1dAutoEncoder):
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
