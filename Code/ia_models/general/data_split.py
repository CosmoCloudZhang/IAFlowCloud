"""
Reproducible train, validation, and test partitions for scientific samples.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SPLIT_NAMES", "build_dataset_split_indices", "validate_dataset_split_indices"]

SPLIT_NAMES = ("train", "validation", "test")


def _integer_scalar(
    value,
    name,
):
    """
    Return a strictly integral scalar.
    
    Arguments:
        value (int or numpy.integer):
            Candidate integer scalar. Floating-point, string, and boolean
            values are rejected rather than coerced.
        name (str):
            Name of the input.
    
    Returns:
        scalar (int):
            Validated Python integer.
    """
    array = np.asarray(value)
    
    if array.ndim != 0 or array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be an integer scalar.")
    
    return int(array)


def validate_dataset_split_indices(
    split_indices: dict[str, object],
    number_of_models: int,
) -> dict[str, np.ndarray]:
    """
    Validate and source-order a complete, non-overlapping sample partition.
    
    Arguments:
        split_indices (dict[str, object]):
            Candidate indices for the train, validation, and test splits.
        number_of_models (int):
            Total number of source models that the splits must cover.
    
    Returns:
        source_ordered_split_indices (dict[str, numpy.ndarray]):
            Validated train, validation, and test index arrays. Each array has
            dtype int64 and is sorted by source-row index.
    """
    number_of_models = _integer_scalar(number_of_models, "number_of_models")
    
    if number_of_models < len(SPLIT_NAMES):
        raise ValueError("At least three models are required for data splits.")
    if set(split_indices) != set(SPLIT_NAMES):
        raise ValueError(f"split_indices must contain exactly {list(SPLIT_NAMES)}.")
    
    source_ordered_split_indices: dict[str, np.ndarray] = {}
    for split_name in SPLIT_NAMES:
        split_index_array = np.asarray(split_indices[split_name])
        if split_index_array.ndim != 1 or len(split_index_array) == 0:
            raise ValueError(f"The {split_name} split must be a non-empty vector.")
        if split_index_array.dtype.kind not in {"i", "u"}:
            raise ValueError(f"The {split_name} split must contain integer indices.")
        split_index_array = split_index_array.astype(np.int64, copy=False)
        if np.any(split_index_array < 0) or np.any(split_index_array >= number_of_models):
            raise ValueError(f"The {split_name} split contains an invalid index.")
        source_ordered_indices = np.sort(split_index_array)
        if np.any(source_ordered_indices[1:] == source_ordered_indices[:-1]):
            raise ValueError(f"The {split_name} split contains duplicate indices.")
        source_ordered_split_indices[split_name] = source_ordered_indices
    
    combined_split_indices = np.concatenate(
        [
            source_ordered_split_indices[split_name]
            for split_name in SPLIT_NAMES
        ]
    )
    if len(combined_split_indices) != number_of_models or not np.array_equal(
        np.sort(combined_split_indices),
        np.arange(number_of_models, dtype=np.int64),
    ):
        raise ValueError("Dataset splits must cover every model exactly once.")
    
    return source_ordered_split_indices


def build_dataset_split_indices(
    number_of_models: int,
    random_seed: int,
    *,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
) -> dict[str, np.ndarray]:
    """
    Build a reproducible train, validation, and test partition.
    
    Arguments:
        number_of_models (int):
            Total number of models to partition.
        random_seed (int):
            Seed controlling random split membership.
        train_fraction (float):
            Fraction assigned to the training split.
        validation_fraction (float):
            Fraction assigned to the validation split.
    
    Returns:
        split_indices (dict[str, numpy.ndarray]):
            Complete, mutually exclusive train, validation, and test index
            arrays sorted by source-row index.
    """
    number_of_models = _integer_scalar(number_of_models, "number_of_models")
    train_fraction = float(train_fraction)
    validation_fraction = float(validation_fraction)
    test_fraction = 1.0 - train_fraction - validation_fraction
    
    if train_fraction <= 0.0 or validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("Train, validation, and test fractions must be positive.")
    
    shuffled_model_indices = np.random.default_rng(random_seed).permutation(number_of_models)
    train_stop = int(number_of_models * train_fraction)
    validation_stop = train_stop + int(number_of_models * validation_fraction)
    candidate_split_indices = {
        "train": shuffled_model_indices[:train_stop],
        "validation": shuffled_model_indices[train_stop:validation_stop],
        "test": shuffled_model_indices[validation_stop:],
    }
    
    return validate_dataset_split_indices(
        candidate_split_indices,
        number_of_models,
    )
