"""Evaluate a selected autoencoder checkpoint on one stored split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from IAFlow.Artifacts import load_autoencoder_checkpoint, save_json
from IAFlow.Config import load_experiment_config
from IAFlow.Data import (
    CachedSurfaceDataset,
    NormalizationStats,
    build_dataloader,
    validate_surface_cache,
)
from IAFlow.Train import evaluate_autoencoder, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "Config" / "AutoEncoderConv1D.yml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
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

    dataset = CachedSurfaceDataset(cache_directory, arguments.split)
    if arguments.maximum_samples is not None:
        if arguments.maximum_samples <= 0:
            raise ValueError("--maximum-samples must be positive.")
        dataset = Subset(dataset, range(min(arguments.maximum_samples, len(dataset))))
    loader = build_dataloader(
        dataset,
        config.data,
        training=False,
        seed=config.training.seed,
        device=device,
    )
    metrics = evaluate_autoencoder(
        model,
        loader,
        normalization,
        device,
        show_progress=not arguments.no_progress,
    )
    result = {
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": arguments.split,
        "device": str(device),
        "metrics": metrics,
    }
    output = arguments.output
    if output is None:
        output = arguments.checkpoint.resolve().parent / f"{arguments.split.capitalize()}Metrics.json"
    save_json(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
