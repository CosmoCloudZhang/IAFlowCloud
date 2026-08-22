"""
Checkpoint construction and compatibility checks for additive PCA-AE runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import PCAAutoEncoder
from ..core.data import NormalizationStats, check_surface_cache
from .config import PCAAEExperimentConfig
from .data import (
    CoefficientNormalization,
    check_pca_ae_cache,
    load_pca_ae_transform,
)

__all__ = [
    "PCA_AE_CHECKPOINT_FORMAT_VERSION",
    "build_pca_autoencoder",
    "build_pca_ae_checkpoint",
    "load_compatible_pca_autoencoder_checkpoint",
]

PCA_AE_CHECKPOINT_FORMAT_VERSION = "1.0"


def _normalizations(
    config: PCAAEExperimentConfig,
    coefficient_metadata: dict[str, Any],
) -> tuple[NormalizationStats, CoefficientNormalization]:
    """
    Load the surface and coefficient normalizations selected by a config.
    
    Arguments:
        config (PCAAEExperimentConfig):
            Checked resolved PCA-AE configuration.
        coefficient_metadata (dict[str, Any]):
            Checked coefficient-cache manifest.
    
    Returns:
        result (tuple[NormalizationStats, CoefficientNormalization]):
            Surface and coefficient normalization objects.
    """
    surface_metadata = check_surface_cache(config)
    surface_directory = config.resolve_path(config.data.cache_directory)
    surface_normalization = NormalizationStats.load(
        surface_directory / surface_metadata["normalization_file"]
    )
    coefficient_directory = config.resolve_path(
        config.model.coefficient_cache_directory
    )
    coefficient_normalization = CoefficientNormalization.load(
        coefficient_directory
        / coefficient_metadata["coefficient_normalization_file"]
    )
    return surface_normalization, coefficient_normalization


def build_pca_autoencoder(
    config: PCAAEExperimentConfig,
) -> tuple[
    PCAAutoEncoder,
    NormalizationStats,
    CoefficientNormalization,
    dict[str, Any],
]:
    """
    Build a PCA-AE model from authenticated transform and cache artifacts.
    
    Arguments:
        config (PCAAEExperimentConfig):
            Checked resolved PCA-AE configuration.
    
    Returns:
        result (tuple):
            Model, surface normalization, coefficient normalization, and cache metadata.
    """
    coefficient_metadata = check_pca_ae_cache(config)
    transform = load_pca_ae_transform(config)
    surface_normalization, coefficient_normalization = _normalizations(
        config,
        coefficient_metadata,
    )
    model = PCAAutoEncoder(
        config.model,
        config.data.input_shape,
        transform,
        surface_normalization,
        coefficient_normalization,
    )
    return (
        model,
        surface_normalization,
        coefficient_normalization,
        coefficient_metadata,
    )


def build_pca_ae_checkpoint(
    model: PCAAutoEncoder,
    surface_normalization: NormalizationStats,
    coefficient_normalization: CoefficientNormalization,
    *,
    epoch: int,
    validation_metrics: dict[str, float],
    experiment_config: dict[str, Any],
    coefficient_provenance: dict[str, Any],
    training_state: dict[str, Any],
    rng_state: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    """
    Build one weights-only-compatible PCA-AE checkpoint payload.
    
    Arguments:
        model (PCAAutoEncoder):
            PCA-AE model whose trainable and frozen state is saved.
        surface_normalization (NormalizationStats):
            Training-only surface normalization.
        coefficient_normalization (CoefficientNormalization):
            Training-only coefficient normalization.
        epoch (int):
            Last completed epoch.
        validation_metrics (dict[str, float]):
            Fast validation objective measured at the epoch.
        experiment_config (dict[str, Any]):
            Original resolved PCA-AE configuration.
        coefficient_provenance (dict[str, Any]):
            Checked coefficient-cache manifest.
        training_state (dict[str, Any]):
            Early-stopping and best-model state.
        rng_state (dict[str, Any]):
            Reproducibility state for exact resume.
        optimizer (torch.optim.Optimizer):
            Optimizer whose state enables resume.
        scheduler (Any or None):
            Optional learning-rate scheduler.
        scaler (torch.amp.GradScaler or None):
            Optional CUDA mixed-precision scaler.
    
    Returns:
        checkpoint (dict[str, Any]):
            Complete PCA-AE checkpoint mapping.
    """
    checkpoint: dict[str, Any] = {
        "checkpoint_format_version": PCA_AE_CHECKPOINT_FORMAT_VERSION,
        "model_family": "PCA_AE",
        "model_config": dict(model.config),
        "input_shape": list(model.input_shape),
        "model_state_dict": model.state_dict(),
        "pca_transform_sha256": model.transform_sha256,
        "surface_normalization": {
            "mean": torch.from_numpy(
                np.array(surface_normalization.mean, copy=True)
            ),
            "scale": surface_normalization.scale,
            "count": surface_normalization.count,
        },
        "coefficient_normalization": {
            "mean": torch.from_numpy(
                np.array(coefficient_normalization.mean, copy=True)
            ),
            "scale": torch.from_numpy(
                np.array(coefficient_normalization.scale, copy=True)
            ),
            "count": coefficient_normalization.count,
        },
        "epoch": int(epoch),
        "validation_metrics": dict(validation_metrics),
        "experiment_config": experiment_config,
        "coefficient_provenance": dict(coefficient_provenance),
        "training_state": dict(training_state),
        "rng_state": rng_state,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    return checkpoint


def _check_checkpoint_identity(
    checkpoint: dict[str, Any],
    config: PCAAEExperimentConfig,
    model: PCAAutoEncoder,
    surface_normalization: NormalizationStats,
    coefficient_normalization: CoefficientNormalization,
    coefficient_metadata: dict[str, Any],
) -> None:
    """
    Check a PCA-AE checkpoint against current config and prepared artifacts.
    
    Arguments:
        checkpoint (dict[str, Any]):
            Loaded checkpoint payload.
        config (PCAAEExperimentConfig):
            Current resolved PCA-AE configuration.
        model (PCAAutoEncoder):
            Newly built compatible model.
        surface_normalization (NormalizationStats):
            Current surface normalization.
        coefficient_normalization (CoefficientNormalization):
            Current coefficient normalization.
        coefficient_metadata (dict[str, Any]):
            Current checked coefficient-cache manifest.
    """
    if checkpoint.get("checkpoint_format_version") != PCA_AE_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported PCA-AE checkpoint format.")
    if checkpoint.get("model_family") != "PCA_AE":
        raise ValueError("Checkpoint is not a PCA-AE checkpoint.")
    if checkpoint.get("model_config") != dict(config.model):
        raise ValueError("PCA-AE checkpoint model configuration does not match.")
    if tuple(checkpoint.get("input_shape", ())) != config.data.input_shape:
        raise ValueError("PCA-AE checkpoint input shape does not match.")
    if checkpoint.get("pca_transform_sha256") != model.transform_sha256:
        raise ValueError("PCA-AE checkpoint uses a different frozen PCA basis.")
    stored_provenance = checkpoint.get("coefficient_provenance")
    if stored_provenance != coefficient_metadata:
        raise ValueError("PCA-AE checkpoint coefficient provenance does not match.")
    stored_surface = checkpoint.get("surface_normalization", {})
    stored_coefficient = checkpoint.get("coefficient_normalization", {})
    surface_matches = (
        isinstance(stored_surface, dict)
        and isinstance(stored_surface.get("mean"), torch.Tensor)
        and int(stored_surface.get("count", -1)) == surface_normalization.count
        and np.isclose(float(stored_surface.get("scale", np.nan)), surface_normalization.scale)
        and np.allclose(
            stored_surface["mean"].cpu().numpy(),
            surface_normalization.mean,
        )
    )
    coefficient_matches = (
        isinstance(stored_coefficient, dict)
        and isinstance(stored_coefficient.get("mean"), torch.Tensor)
        and isinstance(stored_coefficient.get("scale"), torch.Tensor)
        and int(stored_coefficient.get("count", -1))
        == coefficient_normalization.count
        and np.allclose(
            stored_coefficient["mean"].cpu().numpy(),
            coefficient_normalization.mean,
        )
        and np.allclose(
            stored_coefficient["scale"].cpu().numpy(),
            coefficient_normalization.scale,
        )
    )
    if not surface_matches or not coefficient_matches:
        raise ValueError("PCA-AE checkpoint normalization does not match its caches.")


def load_compatible_pca_autoencoder_checkpoint(
    path: str | Path,
    config: PCAAEExperimentConfig,
    *,
    device: str | torch.device = "cpu",
) -> tuple[
    PCAAutoEncoder,
    NormalizationStats,
    CoefficientNormalization,
    dict[str, Any],
    dict[str, Any],
]:
    """
    Load one PCA-AE checkpoint after strict artifact compatibility checks.
    
    Arguments:
        path (str or pathlib.Path):
            PCA-AE checkpoint path.
        config (PCAAEExperimentConfig):
            Current resolved PCA-AE configuration.
        device (str or torch.device):
            Device receiving the loaded model.
    
    Returns:
        result (tuple):
            Model, two normalizations, checkpoint, and coefficient metadata.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("PCA-AE checkpoint payload must be a mapping.")
    (
        model,
        surface_normalization,
        coefficient_normalization,
        coefficient_metadata,
    ) = build_pca_autoencoder(config)
    _check_checkpoint_identity(
        checkpoint,
        config,
        model,
        surface_normalization,
        coefficient_normalization,
        coefficient_metadata,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return (
        model,
        surface_normalization,
        coefficient_normalization,
        checkpoint,
        coefficient_metadata,
    )
