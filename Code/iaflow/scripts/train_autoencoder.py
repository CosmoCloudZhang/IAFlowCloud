"""
Train and validation-select a configurable Conv1D autoencoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iaflow.config import load_experiment_config
from iaflow.training import fit_autoencoder


def parse_arguments() -> argparse.Namespace:
    """
    Parse experiment overrides, smoke limits, and resume arguments.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Continue the same run from Last.pt; --epochs is the new total.",
    )
    parser.add_argument("--maximum-train-samples", type=int)
    parser.add_argument("--maximum-validation-samples", type=int)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    Apply command-line overrides, recheck the config, and train the model.
    """
    arguments = parse_arguments()
    config = load_experiment_config(arguments.config)
    if arguments.device is not None:
        config.training.device = arguments.device
    
    if arguments.epochs is not None:
        if arguments.epochs <= 0:
            raise ValueError("--epochs must be positive.")
        config.training.epochs = arguments.epochs
    
    if arguments.seed is not None:
        if arguments.seed < 0:
            raise ValueError("--seed cannot be negative.")
        config.training.seed = arguments.seed
    
    if arguments.latent_dim is not None:
        if arguments.latent_dim <= 0:
            raise ValueError("--latent-dim must be positive.")
        config.model.latent_dim = arguments.latent_dim
    config.check()
    
    summary = fit_autoencoder(
        config,
        run_directory=arguments.run_directory,
        prepare_data=arguments.prepare_data,
        overwrite_cache=arguments.overwrite_cache,
        maximum_train_samples=arguments.maximum_train_samples,
        maximum_validation_samples=arguments.maximum_validation_samples,
        resume_checkpoint=arguments.resume,
        show_progress=not arguments.no_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
