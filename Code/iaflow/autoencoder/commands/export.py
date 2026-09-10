"""
Export ordered latent representations for the later normalizing-flow stage.
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
from ..artifacts import load_compatible_autoencoder_checkpoint
from ..config import (
    ExperimentConfig,
    load_resolved_experiment_config,
)
from ..model import AutoEncoder


def parse_arguments() -> argparse.Namespace:
    """
    Parse checkpoint, output, device, and test-inclusion arguments.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help="Run containing ResolvedConfig.json and Best.pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional promoted destination; defaults to Latents.hdf5 in the run.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include test latents only after the final checkpoint has been selected.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--maximum-samples",
        type=int,
        help="Optional per-split leading subset for smoke checks only.",
    )
    return parser.parse_args()


def _write_export_metadata(
    destination: h5py.File,
    checkpoint_path: Path,
    config: ExperimentConfig,
    checkpoint: dict[str, Any],
    model: AutoEncoder,
    exported_splits: tuple[str, ...],
    maximum_samples: int | None,
) -> None:
    """
    Write portable model, source, and split provenance to the latent file.
    
    Arguments:
        destination (h5py.File):
            Open latent-export HDF5 destination.
        checkpoint_path (pathlib.Path):
            Resolved checkpoint used for encoding.
        config (ExperimentConfig):
            Checked experiment configuration.
        checkpoint (dict[str, Any]):
            Loaded checkpoint payload.
        model (AutoEncoder):
            Loaded autoencoder defining the latent dimension.
        exported_splits (tuple[str, ...]):
            Ordered stored splits included in the file.
        maximum_samples (int or None):
            Optional per-split smoke limit.
    """
    destination.attrs["checkpoint"] = portable_path(
        checkpoint_path,
        config.project_root,
    )
    destination.attrs["checkpoint_epoch"] = int(checkpoint["epoch"])
    destination.attrs["latent_dim"] = model.config.latent_dim
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
    splits_group: h5py.Group,
    split: str,
    config: ExperimentConfig,
    model: AutoEncoder,
    device: torch.device,
    maximum_samples: int | None,
) -> None:
    """
    Encode one stored split and write ordered latents with diagnostics.
    
    Arguments:
        splits_group (h5py.Group):
            Parent HDF5 group receiving the split output.
        split (str):
            Stored train, validation, or test split.
        config (ExperimentConfig):
            Checked experiment configuration.
        model (AutoEncoder):
            Loaded autoencoder used for encoding.
        device (torch.device):
            Device used for batched inference.
        maximum_samples (int or None):
            Optional leading-sample limit for smoke checks.
    """
    complete_dataset = CachedSurfaceDataset(config, split)
    sample_count = (
        len(complete_dataset)
        if maximum_samples is None
        else min(maximum_samples, len(complete_dataset))
    )
    dataset = Subset(complete_dataset, range(sample_count))
    loader = build_dataloader(
        dataset,
        config.data,
        training=False,
        seed=config.training.seed,
        device=device,
    )
    group = splits_group.create_group(split)
    latent_dataset = group.create_dataset(
        "latent",
        shape=(len(dataset), model.config.latent_dim),
        dtype=np.float32,
        chunks=(min(4096, len(dataset)), model.config.latent_dim),
        compression="gzip",
        shuffle=True,
    )
    group.create_dataset(
        "source_indices",
        data=np.asarray(complete_dataset.source_indices[:sample_count]),
        compression="gzip",
        shuffle=True,
    )
    position = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        latent = model.encode(batch).cpu().numpy()
        latent_dataset[position : position + len(latent)] = latent
        position += len(latent)
    if position != len(dataset):
        raise RuntimeError(f"Incomplete latent export for split '{split}'.")
    
    latent_values = latent_dataset[:]
    group.attrs["mean"] = np.mean(latent_values, axis=0)
    group.attrs["standard_deviation"] = np.std(latent_values, axis=0)
    if model.config.latent_dim > 1:
        group.attrs["covariance"] = np.cov(latent_values, rowvar=False)


def _copy_coordinates(
    destination: h5py.File,
    config: ExperimentConfig,
) -> None:
    """
    Copy authoritative redshift and wavenumber coordinates to the export.
    
    Arguments:
        destination (h5py.File):
            Open latent-export HDF5 destination.
        config (ExperimentConfig):
            Checked experiment and source-data configuration.
    """
    source_path = config.resolve_path(config.data.source_path)
    with h5py.File(source_path, "r") as source:
        coordinates = destination.create_group("coordinates")
        coordinates.create_dataset("z", data=source["coordinates/z"][:])
        coordinates.create_dataset("k", data=source["coordinates/k"][:])


@torch.inference_mode()
def main() -> None:
    """
    Export ordered latent arrays and source indices to an atomic HDF5 file.
    """
    arguments = parse_arguments()
    if arguments.maximum_samples is not None and arguments.maximum_samples <= 0:
        raise ValueError("--maximum-samples must be positive.")
    run_directory = arguments.run_directory.expanduser().resolve()
    config = load_resolved_experiment_config(run_directory)
    output = (
        config.resolve_path(arguments.output)
        if arguments.output is not None
        else run_directory / "Latents.hdf5"
    )
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    
    device = resolve_device(arguments.device)
    checkpoint_path = run_directory / "Best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint_path}")
    model, _normalization, checkpoint, _ = load_compatible_autoencoder_checkpoint(
        checkpoint_path,
        config,
        device=device,
    )
    
    try:
        with h5py.File(temporary, "w") as destination:
            exported_splits = (
                tuple(SPLIT_NAMES) if arguments.include_test else tuple(SPLIT_NAMES[:2])
            )
            _write_export_metadata(
                destination,
                checkpoint_path,
                config,
                checkpoint,
                model,
                exported_splits,
                arguments.maximum_samples,
            )
            splits_group = destination.create_group("splits")
            for split in exported_splits:
                _export_split(
                    splits_group,
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
    print(f"Exported latent representations: {output}")


if __name__ == "__main__":
    main()
