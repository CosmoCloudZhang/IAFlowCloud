"""
Train and validation-select one additive PCA-AE coefficient autoencoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...core.workflows import apply_training_overrides, resolve_run_directory
from ..config import load_pca_ae_experiment_template
from ..training import fit_pca_autoencoder


def parse_arguments() -> argparse.Namespace:
    """
    Parse PCA-AE template, runtime, smoke, and resume arguments.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--maximum-train-samples", type=int)
    parser.add_argument("--maximum-validation-samples", type=int)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    Resolve runtime axes and train the selected PCA-AE model.
    """
    arguments = parse_arguments()
    if arguments.latent_dim <= 0:
        raise ValueError("--latent-dim must be positive.")
    template = load_pca_ae_experiment_template(arguments.config)
    run_directory = resolve_run_directory(
        template,
        arguments.latent_dim,
        explicit=arguments.run_directory,
        resume=arguments.resume,
        family="PCA-AE",
    )
    config = template.resolve(arguments.latent_dim, run_directory)
    apply_training_overrides(
        config,
        device=arguments.device,
        epochs=arguments.epochs,
        seed=arguments.seed,
    )
    summary = fit_pca_autoencoder(
        config,
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
