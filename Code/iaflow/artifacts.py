"""Safe, portable model and result artifact helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .Models import Conv1dAutoEncoder
from .Models import build_autoencoder
from .Config import ExperimentConfig
from .Data import NormalizationStats, validate_surface_cache

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "build_checkpoint",
    "load_compatible_autoencoder_checkpoint",
    "load_autoencoder_checkpoint",
    "portable_path",
    "save_checkpoint",
    "save_json",
]

CHECKPOINT_FORMAT_VERSION = "2.0"


def portable_path(path: str | Path, project_root: str | Path) -> str:
    """Return a path relative to the project for portable artifact metadata."""
    return os.path.relpath(Path(path).resolve(), Path(project_root).resolve())


def save_json(values: dict[str, Any] | list[Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(values, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, destination)


def build_checkpoint(
    model: Conv1dAutoEncoder,
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
    """Build a weights-only-compatible PyTorch checkpoint dictionary."""
    checkpoint: dict[str, Any] = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": dict(vars(model.config)) if not isinstance(model.config, dict) else dict(model.config),
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


def save_checkpoint(checkpoint: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)


def load_autoencoder_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[Conv1dAutoEncoder, NormalizationStats, dict[str, Any]]:
    """Load a checkpoint without permitting arbitrary pickled objects."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported autoencoder checkpoint format.")
    model_config = SimpleNamespace(**checkpoint["model_config"])
    input_shape = tuple(int(value) for value in checkpoint["input_shape"])
    model = build_autoencoder(model_config, input_shape)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    stored_normalization = checkpoint["normalization"]
    normalization = NormalizationStats(
        mean=stored_normalization["mean"].cpu().numpy(),
        scale=float(stored_normalization["scale"]),
        count=int(stored_normalization["count"]),
    )
    return model, normalization, checkpoint


def load_compatible_autoencoder_checkpoint(
    path: str | Path,
    config: ExperimentConfig,
    *,
    device: str | torch.device = "cpu",
) -> tuple[Conv1dAutoEncoder, NormalizationStats, dict[str, Any], dict[str, Any]]:
    """Load a checkpoint after verifying it belongs to the current prepared data."""
    metadata = validate_surface_cache(config)
    model, normalization, checkpoint = load_autoencoder_checkpoint(path, device=device)
    cached = NormalizationStats.load(
        config.resolve_path(config.data.cache_directory) / metadata["normalization_file"]
    )
    if (
        normalization.count != cached.count
        or not np.allclose(normalization.mean, cached.mean)
        or not np.isclose(normalization.scale, cached.scale)
    ):
        raise ValueError("Checkpoint and cache normalization statistics do not match.")

    stored_provenance = checkpoint.get("data_provenance")
    provenance_keys = {
        "source_structure_sha256",
        "source_size",
        "target_dataset",
        "transform",
        "normalization",
        "input_shape",
    }
    if not isinstance(stored_provenance, dict) or any(
        stored_provenance.get(key) != metadata.get(key) for key in provenance_keys
    ):
        raise ValueError("Checkpoint data provenance does not match the current prepared data.")
    return model, normalization, checkpoint, metadata
