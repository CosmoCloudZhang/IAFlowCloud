"""
Train and validation-select a resolved Conv1D or Conv2D autoencoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iaflow.config import load_experiment_template
from iaflow.runs import propose_run_directory
from iaflow.training import fit_autoencoder


def parse_arguments() -> argparse.Namespace:
    """
    Parse template, runtime axes, smoke limits, and resume arguments.
    
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
    parser.add_argument(
        "--resume",
        type=Path,
        help="Continue the same run from Last.pt; --epochs is the new total.",
    )
    parser.add_argument("--maximum-train-samples", type=int)
    parser.add_argument("--maximum-validation-samples", type=int)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _run_directory(
    arguments: argparse.Namespace,
    template: object,
) -> Path:
    """
    Resolve a new or resumed concrete run directory.
    
    Arguments:
        arguments (argparse.Namespace):
            Parsed run-directory and resume arguments.
        template (ExperimentTemplate):
            Reusable architecture-depth template.
    
    Returns:
        directory (pathlib.Path):
            Concrete run directory used in the resolved configuration.
    """
    if arguments.resume is not None:
        resume = template.resolve_path(arguments.resume)
        if not resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        if arguments.run_directory is not None:
            explicit = template.resolve_path(arguments.run_directory)
            if explicit != resume.parent:
                raise ValueError("--run-directory must contain the resume checkpoint.")
        return resume.parent
    return propose_run_directory(
        template,
        arguments.latent_dim,
        explicit=arguments.run_directory,
    )


def main() -> None:
    """
    Resolve runtime axes, apply allowed overrides, and train the model.
    """
    arguments = parse_arguments()
    if arguments.latent_dim <= 0:
        raise ValueError("--latent-dim must be positive.")
    template = load_experiment_template(arguments.config)
    run_directory = _run_directory(arguments, template)
    config = template.resolve(arguments.latent_dim, run_directory)
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
    config.check()
    
    summary = fit_autoencoder(
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
