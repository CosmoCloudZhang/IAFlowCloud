"""
Model-agnostic HDF5 schema checks for IA surface datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .data_split import SPLIT_NAMES, validate_dataset_split_indices

__all__ = ["SurfaceDatasetDescription", "read_surface_dataset_description"]


@dataclass(frozen=True, slots=True)
class SurfaceDatasetDescription:
    """
    Store validated information required to read a spectral target from HDF5.
    """
    
    number_of_models: int
    target_shape: tuple[int, int]
    split_indices: dict[str, np.ndarray]


def read_surface_dataset_description(
    source_path: str | Path,
    target_dataset: str,
    expected_shape: tuple[int, int],
) -> SurfaceDatasetDescription:
    """
    Validate the required HDF5 paths, target shape, and stored data splits.
    
    The scientific HDF5 file is validated from its contents rather than a
    schema-version attribute.
    
    Arguments:
        source_path (str or pathlib.Path):
            Path to the authoritative HDF5 surface dataset.
        target_dataset (str):
            HDF5 path of the three-dimensional target array.
        expected_shape (tuple[int, int]):
            Expected shape of one target surface.
    
    Returns:
        description (SurfaceDatasetDescription):
            Validated model count, target shape, and split indices.
    """
    path = Path(source_path)
    
    with h5py.File(path, "r") as source:
        if target_dataset not in source:
            raise KeyError(f"HDF5 target '{target_dataset}' does not exist.")
        target = source[target_dataset]
        if target.ndim != 3 or tuple(target.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected target shape (N, {expected_shape[0]}, {expected_shape[1]}), "
                f"found {target.shape}."
            )
        split_indices: dict[str, np.ndarray] = {}
        for split in SPLIT_NAMES:
            path_in_file = f"splits/{split}"
            if path_in_file not in source:
                raise KeyError(f"Missing source split '{path_in_file}'.")
            split_indices[split] = np.asarray(source[path_in_file][:], dtype=np.int64)
    
        number_of_models = int(target.shape[0])
        target_shape = tuple(int(value) for value in target.shape[1:])
    
    return SurfaceDatasetDescription(
        number_of_models=number_of_models,
        target_shape=target_shape,
        split_indices=validate_dataset_split_indices(split_indices, number_of_models),
    )
