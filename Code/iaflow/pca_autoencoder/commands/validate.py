"""
Resolve and shape-check PCA-AE templates across latent dimensions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ...core.data import NormalizationStats
from ...core.workflows import template_paths
from ..config import load_pca_ae_experiment_template
from ..data import (
    CoefficientNormalization,
    load_pca_ae_transform,
)
from ..model import PCAAutoEncoder


def parse_arguments() -> argparse.Namespace:
    """
    Parse PCA-AE template paths and runtime latent dimensions.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument(
        "--latent-dims",
        type=int,
        nargs="+",
        default=(2, 4, 6, 8, 10),
    )
    return parser.parse_args()


def main() -> None:
    """
    Validate every selected PCA-AE template and latent dimension.
    """
    arguments = parse_arguments()
    if any(dimension <= 0 for dimension in arguments.latent_dims):
        raise ValueError("--latent-dims must contain positive integers.")
    results = []
    for path in template_paths(arguments.config, family="PCA-AE"):
        template = load_pca_ae_experiment_template(path)
        transform = load_pca_ae_transform(template)
        coefficient_scale = np.sqrt(
            transform.explained_variance
            * (transform.training_count - 1)
            / transform.training_count
        )
        surface_normalization = NormalizationStats(
            np.zeros(template.data.input_shape, dtype=np.float32),
            1.0,
            transform.training_count,
        )
        coefficient_normalization = CoefficientNormalization(
            np.zeros(transform.rank, dtype=np.float32),
            coefficient_scale,
            transform.training_count,
        )
        for latent_dim in arguments.latent_dims:
            validation_run = (
                Path(template.output.root_directory)
                / f"Latent{latent_dim:02d}"
                / "ConfigValidation"
            )
            config = template.resolve(latent_dim, validation_run)
            model = PCAAutoEncoder(
                config.model,
                config.data.input_shape,
                transform,
                surface_normalization,
                coefficient_normalization,
            )
            sample = torch.zeros((2, *config.data.input_shape), dtype=torch.float32)
            with torch.inference_mode():
                latent = model.encode(sample)
                reconstructed = model(sample)
            if latent.shape != (2, latent_dim) or reconstructed.shape != sample.shape:
                raise RuntimeError(f"Invalid PCA-AE round trip for {path}.")
            results.append(
                {
                    "config": str(path.relative_to(template.project_root)),
                    "architecture": "PCA_AE",
                    "pca_rank": transform.rank,
                    "latent_dim": latent_dim,
                    "number_of_parameters": model.number_of_parameters,
                }
            )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
