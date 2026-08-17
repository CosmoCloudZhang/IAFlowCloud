"""
Resolve and shape-check AE templates across runtime latent dimensions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from iaflow.architectures import build_autoencoder
from iaflow.config import load_experiment_template


def parse_arguments() -> argparse.Namespace:
    """
    Parse template paths and runtime latent dimensions.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="Template file or directory; may be supplied more than once.",
    )
    parser.add_argument(
        "--latent-dims",
        type=int,
        nargs="+",
        default=(2, 4, 6, 8, 10),
    )
    return parser.parse_args()


def _template_paths(
    requested: list[Path],
) -> list[Path]:
    """
    Expand template files and directories into an ordered unique list.
    
    Arguments:
        requested (list[pathlib.Path]):
            Explicit template paths or directories.
    
    Returns:
        paths (list[pathlib.Path]):
            Ordered unique YAML template files.
    """
    paths: set[Path] = set()
    for value in requested:
        path = value.expanduser().resolve()
        if path.is_dir():
            paths.update(path.rglob("Depth*.yaml"))
        elif path.is_file():
            paths.add(path)
        else:
            raise FileNotFoundError(f"Configuration path not found: {path}")
    if not paths:
        raise ValueError("No Depth*.yaml experiment templates were selected.")
    return sorted(paths)


def main() -> None:
    """
    Resolve every template-dimension pair and verify exact model round trips.
    """
    arguments = parse_arguments()
    if any(dimension <= 0 for dimension in arguments.latent_dims):
        raise ValueError("--latent-dims must contain only positive integers.")
    results = []
    for path in _template_paths(arguments.config):
        template = load_experiment_template(path)
        for latent_dim in arguments.latent_dims:
            validation_run = (
                Path(template.output.root_directory)
                / f"Latent{latent_dim:02d}"
                / "ConfigValidation"
            )
            config = template.resolve(latent_dim, validation_run)
            model = build_autoencoder(config.model, config.data.input_shape)
            sample = torch.zeros((2, *config.data.input_shape), dtype=torch.float32)
            with torch.inference_mode():
                latent = model.encode(sample)
                reconstructed = model(sample)
            if latent.shape != (2, latent_dim):
                raise RuntimeError(f"Invalid latent shape for {path}: {latent.shape}")
            if reconstructed.shape != sample.shape:
                raise RuntimeError(
                    f"Invalid reconstruction shape for {path}: {reconstructed.shape}"
                )
            results.append(
                {
                    "config": str(path.relative_to(template.project_root)),
                    "architecture": config.model.name,
                    "latent_dim": latent_dim,
                    "number_of_parameters": model.number_of_parameters,
                    "encoded_shape": model.architecture_summary()["encoded_shape"],
                }
            )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
