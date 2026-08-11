"""Export ordered latent representations for the later normalizing-flow stage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from IAFlow.Artifacts import load_autoencoder_checkpoint
from IAFlow.Config import load_experiment_config
from IAFlow.Data import (
    SPLIT_NAMES,
    CachedSurfaceDataset,
    NormalizationStats,
    build_dataloader,
    validate_surface_cache,
)
from IAFlow.Train import resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "Config" / "AutoEncoderConv1D.yml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "Data" / "ML" / "AutoEncoderLatents.hdf5",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    arguments = parse_arguments()
    output = arguments.output.expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)

    config = load_experiment_config(arguments.config, project_root=PROJECT_ROOT)
    device = resolve_device(arguments.device)
    metadata = validate_surface_cache(config)
    cache_directory = config.resolve_path(config.data.cache_directory)
    cached_normalization = NormalizationStats.load(
        cache_directory / metadata["normalization_file"]
    )
    model, normalization, checkpoint = load_autoencoder_checkpoint(
        arguments.checkpoint,
        device=device,
    )
    if not np.allclose(normalization.mean, cached_normalization.mean) or not np.isclose(
        normalization.scale, cached_normalization.scale
    ):
        raise ValueError("Checkpoint and cache normalization statistics do not match.")

    try:
        with h5py.File(temporary, "w") as destination:
            destination.attrs["schema_version"] = "1.0"
            destination.attrs["checkpoint"] = str(arguments.checkpoint.resolve())
            destination.attrs["checkpoint_epoch"] = int(checkpoint["epoch"])
            destination.attrs["latent_dim"] = model.config.latent_dim
            destination.attrs["source_dataset"] = str(config.resolve_path(config.data.source_path))
            destination.attrs["source_target"] = config.data.target_dataset
            destination.attrs["transform"] = config.data.transform
            destination.attrs["normalization"] = config.data.normalization
            destination.attrs["model_config"] = json.dumps(checkpoint["model_config"])

            splits_group = destination.create_group("splits")
            for split in SPLIT_NAMES:
                dataset = CachedSurfaceDataset(cache_directory, split)
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
