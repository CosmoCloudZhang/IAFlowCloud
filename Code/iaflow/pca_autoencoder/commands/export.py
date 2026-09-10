"""
Export ordered PCA-AE latent representations for the later flow stage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Subset

from ...core.artifacts import portable_path
from ...core.data import SPLIT_NAMES, CachedSurfaceDataset, build_dataloader
from ...core.runtime import resolve_device
from ..artifacts import (
    load_compatible_pca_autoencoder_checkpoint,
)
from ..config import (
    PCAAEExperimentConfig,
    load_resolved_pca_ae_config,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse PCA-AE latent-export arguments.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--maximum-samples", type=int)
    return parser.parse_args()


def _write_metadata(
    destination: h5py.File,
    checkpoint_path: Path,
    config: PCAAEExperimentConfig,
    checkpoint: dict[str, Any],
    exported_splits: tuple[str, ...],
    maximum_samples: int | None,
) -> None:
    """
    Write portable PCA-AE model and split provenance.
    
    Arguments:
        destination (h5py.File):
            Open latent-export destination.
        checkpoint_path (pathlib.Path):
            Selected PCA-AE checkpoint.
        config (PCAAEExperimentConfig):
            Resolved PCA-AE configuration.
        checkpoint (dict[str, Any]):
            Loaded checkpoint payload.
        exported_splits (tuple[str, ...]):
            Ordered stored splits included in the export.
        maximum_samples (int or None):
            Optional per-split smoke limit.
    """
    destination.attrs["checkpoint"] = portable_path(
        checkpoint_path,
        config.project_root,
    )
    destination.attrs["checkpoint_epoch"] = int(checkpoint["epoch"])
    destination.attrs["model_family"] = "PCA_AE"
    destination.attrs["pca_rank"] = int(config.model.pca_rank)
    destination.attrs["pca_transform_sha256"] = checkpoint[
        "pca_transform_sha256"
    ]
    destination.attrs["latent_dim"] = int(config.model.latent_dim)
    destination.attrs["source_dataset"] = str(config.data.source_path)
    destination.attrs["source_target"] = config.data.target_dataset
    destination.attrs["transform"] = config.data.transform
    destination.attrs["normalization"] = config.data.normalization
    destination.attrs["model_config"] = json.dumps(checkpoint["model_config"])
    destination.attrs["exported_splits"] = json.dumps(list(exported_splits))
    destination.attrs["maximum_samples_per_split"] = (
        -1 if maximum_samples is None else maximum_samples
    )


def _export_split(
    destination: h5py.Group,
    split: str,
    config: PCAAEExperimentConfig,
    model: torch.nn.Module,
    device: torch.device,
    maximum_samples: int | None,
) -> None:
    """
    Encode and write one ordered PCA-AE latent split.
    
    Arguments:
        destination (h5py.Group):
            Parent group receiving the split.
        split (str):
            Train, validation, or test split.
        config (PCAAEExperimentConfig):
            Resolved PCA-AE configuration.
        model (torch.nn.Module):
            Selected PCA-AE model.
        device (torch.device):
            Inference device.
        maximum_samples (int or None):
            Optional leading-sample smoke limit.
    """
    complete = CachedSurfaceDataset(config, split)
    sample_count = (
        len(complete)
        if maximum_samples is None
        else min(maximum_samples, len(complete))
    )
    dataset = Subset(complete, range(sample_count))
    loader = build_dataloader(
        dataset,
        config.data,
        training=False,
        seed=config.training.seed,
        device=device,
    )
    group = destination.create_group(split)
    latent_dataset = group.create_dataset(
        "latent",
        shape=(sample_count, config.model.latent_dim),
        dtype=np.float32,
        chunks=(min(4096, sample_count), config.model.latent_dim),
        compression="gzip",
        shuffle=True,
    )
    group.create_dataset(
        "source_indices",
        data=np.asarray(complete.source_indices[:sample_count]),
        compression="gzip",
        shuffle=True,
    )
    position = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        latent = model.encode(batch).cpu().numpy()
        latent_dataset[position : position + len(latent)] = latent
        position += len(latent)
    if position != sample_count:
        raise RuntimeError(f"Incomplete PCA-AE latent export for {split!r}.")
    values = latent_dataset[:]
    group.attrs["mean"] = np.mean(values, axis=0)
    group.attrs["standard_deviation"] = np.std(values, axis=0)
    if config.model.latent_dim > 1:
        group.attrs["covariance"] = np.cov(values, rowvar=False)


def _copy_coordinates(
    destination: h5py.File,
    config: PCAAEExperimentConfig,
) -> None:
    """
    Copy authoritative surface coordinates into the PCA-AE export.
    
    Arguments:
        destination (h5py.File):
            Open latent-export destination.
        config (PCAAEExperimentConfig):
            Resolved PCA-AE configuration.
    """
    with h5py.File(config.resolve_path(config.data.source_path), "r") as source:
        coordinates = destination.create_group("coordinates")
        coordinates.create_dataset("z", data=source["coordinates/z"][:])
        coordinates.create_dataset("k", data=source["coordinates/k"][:])


@torch.inference_mode()
def main() -> None:
    """
    Export PCA-AE latents to an atomic HDF5 artifact.
    """
    arguments = parse_arguments()
    if arguments.maximum_samples is not None and arguments.maximum_samples <= 0:
        raise ValueError("--maximum-samples must be positive.")
    run_directory = arguments.run_directory.expanduser().resolve()
    config = load_resolved_pca_ae_config(run_directory)
    output = (
        config.resolve_path(arguments.output)
        if arguments.output is not None
        else run_directory / "Latents.hdf5"
    )
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"PCA-AE latent output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    device = resolve_device(arguments.device)
    checkpoint_path = run_directory / "Best.pt"
    model, _surface, _coefficients, checkpoint, _metadata = (
        load_compatible_pca_autoencoder_checkpoint(
            checkpoint_path,
            config,
            device=device,
        )
    )
    exported_splits = (
        tuple(SPLIT_NAMES) if arguments.include_test else tuple(SPLIT_NAMES[:2])
    )
    try:
        with h5py.File(temporary, "w") as destination:
            _write_metadata(
                destination,
                checkpoint_path,
                config,
                checkpoint,
                exported_splits,
                arguments.maximum_samples,
            )
            splits = destination.create_group("splits")
            for split in exported_splits:
                _export_split(
                    splits,
                    split,
                    config,
                    model,
                    device,
                    arguments.maximum_samples,
                )
            _copy_coordinates(destination, config)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Exported PCA-AE latent representations: {output}")


if __name__ == "__main__":
    main()
