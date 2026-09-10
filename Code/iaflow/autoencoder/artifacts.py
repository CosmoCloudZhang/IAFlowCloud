"""
Safe, portable model and result artifact helpers.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from ..core.artifacts import portable_path, save_checkpoint, save_json
from .model import AutoEncoder, build_autoencoder
from .config import (
    ExperimentConfig,
    check_input_shape,
    check_model_config,
)
from ..core.data import NormalizationStats, check_surface_cache

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "build_checkpoint",
    "check_checkpoint_data_compatibility",
    "load_compatible_autoencoder_checkpoint",
    "load_autoencoder_checkpoint",
    "portable_path",
    "save_checkpoint",
    "save_json",
]

CHECKPOINT_FORMAT_VERSION = "2.0"

_DATA_PROVENANCE_KEYS = {
    "source_structure_sha256",
    "source_size",
    "target_dataset",
    "transform",
    "normalization",
    "input_shape",
}


def build_checkpoint(
    model: AutoEncoder,
    normalization: NormalizationStats,
    *,
    epoch: int,
    validation_metrics: dict[str, float],
    experiment_config: dict[str, Any],
    data_provenance: dict[str, Any],
    training_state: dict[str, Any],
    rng_state: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    """
    Build a weights-only-compatible PyTorch checkpoint dictionary.
    
    Arguments:
        model (AutoEncoder):
            Autoencoder whose parameters and architecture are stored.
        normalization (NormalizationStats):
            Training-only normalization bundled with the model.
        epoch (int):
            Last completed training epoch.
        validation_metrics (dict[str, float]):
            Validation metrics measured at epoch.
        experiment_config (dict[str, Any]):
            Original resolved experiment configuration.
        data_provenance (dict[str, Any]):
            Compact cache and source-dataset provenance.
        training_state (dict[str, Any]):
            Best-model and early-stopping state.
        rng_state (dict[str, Any]):
            Random-number and training-loader state for exact resume.
        optimizer (torch.optim.Optimizer or None):
            Optional optimizer whose state enables resume.
        scheduler (Any or None):
            Optional learning-rate scheduler.
        scaler (torch.amp.GradScaler or None):
            Optional mixed-precision gradient scaler.
    
    Returns:
        checkpoint (dict[str, Any]):
            Complete checkpoint payload accepted by torch.load with weights_only.
    """
    checkpoint: dict[str, Any] = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": (
            dict(vars(model.config))
            if not isinstance(model.config, dict)
            else dict(model.config)
        ),
        "input_shape": list(model.input_shape),
        "model_state_dict": model.state_dict(),
        "normalization": {
            "mean": torch.from_numpy(np.array(normalization.mean, copy=True)),
            "scale": normalization.scale,
            "count": normalization.count,
        },
        "epoch": int(epoch),
        "validation_metrics": dict(validation_metrics),
        "experiment_config": experiment_config,
        "data_provenance": dict(data_provenance),
        "training_state": dict(training_state),
        "rng_state": rng_state,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    return checkpoint


def _checkpoint_normalization(
    checkpoint: dict[str, Any],
) -> NormalizationStats:
    """
    Restore checked normalization statistics from a checkpoint payload.
    
    Arguments:
        checkpoint (dict[str, Any]):
            Loaded autoencoder checkpoint payload.
    
    Returns:
        normalization (NormalizationStats):
            Checked training-only normalization stored with the model.
    """
    stored = checkpoint.get("normalization")
    if not isinstance(stored, dict):
        raise ValueError("Autoencoder checkpoint normalization must be a mapping.")
    try:
        mean = stored["mean"]
        scale = stored["scale"]
        count = stored["count"]
    except KeyError as error:
        raise ValueError(
            f"Autoencoder checkpoint normalization is missing {error.args[0]!r}."
        ) from error
    if not isinstance(mean, torch.Tensor):
        raise ValueError("Autoencoder checkpoint normalization mean must be a tensor.")
    return NormalizationStats(
        mean=mean.cpu().numpy(),
        scale=float(scale),
        count=int(count),
    )


def check_checkpoint_data_compatibility(
    checkpoint: dict[str, Any],
    cached_normalization: NormalizationStats,
    metadata: dict[str, Any],
) -> None:
    """
    Check checkpoint normalization and provenance against one prepared cache.
    
    Arguments:
        checkpoint (dict[str, Any]):
            Loaded autoencoder checkpoint payload.
        cached_normalization (NormalizationStats):
            Training-only normalization loaded from the current cache.
        metadata (dict[str, Any]):
            Checked current cache manifest.
    """
    checkpoint_normalization = _checkpoint_normalization(checkpoint)
    normalization_matches = (
        checkpoint_normalization.count == cached_normalization.count
        and np.allclose(checkpoint_normalization.mean, cached_normalization.mean)
        and np.isclose(checkpoint_normalization.scale, cached_normalization.scale)
    )
    if not normalization_matches:
        raise ValueError("Checkpoint and cache normalization statistics do not match.")
    
    stored_provenance = checkpoint.get("data_provenance")
    provenance_matches = isinstance(stored_provenance, dict) and all(
        stored_provenance.get(key) == metadata.get(key)
        for key in _DATA_PROVENANCE_KEYS
    )
    if not provenance_matches:
        raise ValueError("Checkpoint data provenance does not match the current prepared data.")


def load_autoencoder_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[AutoEncoder, NormalizationStats, dict[str, Any]]:
    """
    Load a checkpoint without permitting arbitrary pickled objects.
    
    Arguments:
        path (str or pathlib.Path):
            Checkpoint file to load.
        device (str or torch.device):
            Device that receives the reconstructed model.
    
    Returns:
        result (tuple):
            Model, normalization statistics, and raw checkpoint payload.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Autoencoder checkpoint payload must be a mapping.")
    if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported autoencoder checkpoint format.")
    stored_model_config = checkpoint.get("model_config")
    if not isinstance(stored_model_config, dict):
        raise ValueError("Autoencoder checkpoint model configuration must be a mapping.")
    model_config = SimpleNamespace(**check_model_config(stored_model_config))
    input_shape = check_input_shape(
        checkpoint.get("input_shape"),
        "checkpoint.input_shape",
    )
    model = build_autoencoder(model_config, input_shape)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    
    normalization = _checkpoint_normalization(checkpoint)
    return model, normalization, checkpoint


def load_compatible_autoencoder_checkpoint(
    path: str | Path,
    config: ExperimentConfig,
    *,
    device: str | torch.device = "cpu",
) -> tuple[AutoEncoder, NormalizationStats, dict[str, Any], dict[str, Any]]:
    """
    Load a checkpoint after verifying it belongs to the prepared data.
    
    Arguments:
        path (str or pathlib.Path):
            Checkpoint file to load.
        config (ExperimentConfig):
            Current experiment and source-data configuration.
        device (str or torch.device):
            Device that receives the reconstructed model.
    
    Returns:
        result (tuple):
            Model, normalization, checkpoint payload, and cache metadata.
    """
    metadata = check_surface_cache(config)
    model, normalization, checkpoint = load_autoencoder_checkpoint(path, device=device)
    cached = NormalizationStats.load(
        config.resolve_path(config.data.cache_directory) / metadata["normalization_file"]
    )
    check_checkpoint_data_compatibility(checkpoint, cached, metadata)
    return model, normalization, checkpoint, metadata
