"""
Prepare the source-ordered ML cache from the authoritative HDF5 dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iaflow.config import load_experiment_config
from iaflow.data import prepare_surface_cache

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "Config" / "NLA" / "AutoEncoderConv1D.yml"


def parse_arguments() -> argparse.Namespace:
    """
    Parse cache-configuration and overwrite arguments.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing cache after re-reading the full source dataset.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Build or validate the configured cache and report its resolved dimensions.
    """
    arguments = parse_arguments()
    config = load_experiment_config(arguments.config, project_root=PROJECT_ROOT)
    metadata = prepare_surface_cache(config, overwrite=arguments.overwrite)
    cache_directory = config.resolve_path(config.data.cache_directory)
    print(f"Prepared cache: {cache_directory}")
    print(f"Split sizes: {metadata['split_sizes']}")
    print(f"Input shape: {tuple(metadata['input_shape'])}")


if __name__ == "__main__":
    main()
