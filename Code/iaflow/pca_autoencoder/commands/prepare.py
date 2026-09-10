"""
Prepare the additive PCA-AE coefficient cache without altering direct-AE runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ...core.data import check_surface_cache, prepare_surface_cache
from ..config import load_pca_ae_experiment_template
from ..data import prepare_pca_ae_cache


def parse_arguments() -> argparse.Namespace:
    """
    Parse PCA-AE cache configuration and overwrite arguments.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate only the PCA-AE coefficient cache.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Check the shared surface cache and prepare PCA-AE-only products.
    """
    arguments = parse_arguments()
    template = load_pca_ae_experiment_template(arguments.config)
    try:
        check_surface_cache(template)
    except FileNotFoundError:
        prepare_surface_cache(template, overwrite=False)
    metadata = prepare_pca_ae_cache(template, overwrite=arguments.overwrite)
    cache_directory = template.resolve_path(
        template.model.coefficient_cache_directory
    )
    print(f"PCA-AE coefficient cache ready: {cache_directory}")
    print(f"Coefficient shape: {tuple(metadata['coefficient_shape'])}")
    print(f"Split sizes: {metadata['split_sizes']}")


if __name__ == "__main__":
    main()
