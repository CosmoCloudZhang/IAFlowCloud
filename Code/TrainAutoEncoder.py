"""Train and validation-select a configurable Conv1D autoencoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from IAFlow.Config import load_experiment_config
from IAFlow.Train import fit_autoencoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "Config" / "AutoEncoderConv1D.yml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--epochs", type=int)
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
    arguments = parse_arguments()
    config = load_experiment_config(arguments.config, project_root=PROJECT_ROOT)
    if arguments.device is not None:
        config.training.device = arguments.device
    if arguments.epochs is not None:
        if arguments.epochs <= 0:
            raise ValueError("--epochs must be positive.")
        config.training.epochs = arguments.epochs
    if arguments.latent_dim is not None:
        if arguments.latent_dim <= 0:
            raise ValueError("--latent-dim must be positive.")
        config.model.latent_dim = arguments.latent_dim

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
