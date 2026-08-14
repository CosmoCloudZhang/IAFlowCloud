"""
Export ordered latent representations for the later normalizing-flow stage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from iaflow.artifacts import load_compatible_autoencoder_checkpoint, portable_path
from iaflow.config import load_experiment_config
from iaflow.data import (
    SPLIT_NAMES,
    CachedSurfaceDataset,
    build_dataloader,
)
from iaflow.training import resolve_device


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "Config" / "NLA" / "AutoEncoderConv1D.yml"


def parse_arguments() -> argparse.Namespace:
    """
    Parse checkpoint, output, device, and test-inclusion arguments.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "Data" / "NLA" / "Latents" / "AutoEncoderLatents.hdf5",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include test latents only after the final checkpoint has been selected.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    """
    Export ordered latent arrays and source indices to an atomic HDF5 file.
    """
    arguments = parse_arguments()
    config = load_experiment_config(arguments.config, project_root=PROJECT_ROOT)
    output = arguments.output.expanduser()
    if not output.is_absolute():
        output = config.project_root / output
    output = output.resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    
    device = resolve_device(arguments.device)
    checkpoint_path = arguments.checkpoint.expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = config.project_root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    model, _normalization, checkpoint, _ = load_compatible_autoencoder_checkpoint(
        checkpoint_path,
        config,
        device=device,
    )
    
    try:
        with h5py.File(temporary, "w") as destination:
            destination.attrs["checkpoint"] = portable_path(
                checkpoint_path, config.project_root
            )
            destination.attrs["checkpoint_epoch"] = int(checkpoint["epoch"])
            destination.attrs["latent_dim"] = model.config.latent_dim
            destination.attrs["source_dataset"] = str(config.data.source_path)
            destination.attrs["source_target"] = config.data.target_dataset
            destination.attrs["transform"] = config.data.transform
            destination.attrs["normalization"] = config.data.normalization
            destination.attrs["model_config"] = json.dumps(checkpoint["model_config"])
            exported_splits = SPLIT_NAMES if arguments.include_test else SPLIT_NAMES[:2]
            destination.attrs["exported_splits"] = json.dumps(list(exported_splits))
            
            splits_group = destination.create_group("splits")
            for split in exported_splits:
                dataset = CachedSurfaceDataset(config, split)
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
                    data=np.asarray(dataset.source_indices),
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
            
            source_path = config.resolve_path(config.data.source_path)
            with h5py.File(source_path, "r") as source:
                coordinates = destination.create_group("coordinates")
                coordinates.create_dataset("z", data=source["coordinates/z"][:])
                coordinates.create_dataset("k", data=source["coordinates/k"][:])
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Exported latent representations: {output}")


if __name__ == "__main__":
    main()
