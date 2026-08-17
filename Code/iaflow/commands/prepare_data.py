"""
Prepare the source-ordered ML cache from the authoritative HDF5 dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iaflow.config import load_experiment_template
from iaflow.data import prepare_surface_cache


def parse_arguments() -> argparse.Namespace:
    """
    Parse cache-configuration and overwrite arguments.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing cache after re-reading the full source dataset.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Build or check the configured cache and report its resolved dimensions.
    """
    arguments = parse_arguments()
    template = load_experiment_template(arguments.config)
    metadata = prepare_surface_cache(template, overwrite=arguments.overwrite)
    cache_directory = template.resolve_path(template.data.cache_directory)
    print(f"Cache ready: {cache_directory}")
    print(f"Split sizes: {metadata['split_sizes']}")
    print(f"Input shape: {tuple(metadata['input_shape'])}")


if __name__ == "__main__":
    main()
