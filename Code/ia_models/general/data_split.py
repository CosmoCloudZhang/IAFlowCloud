"""Reproducible train, validation, and test partitions for scientific samples."""

from __future__ import annotations

import numpy as np

__all__ = ["SPLIT_NAMES", "build_dataset_split_indices", "validate_dataset_split_indices"]

SPLIT_NAMES = ("train", "validation", "test")


def validate_dataset_split_indices(
    split_indices: dict[str, object], number_of_models: int
) -> dict[str, np.ndarray]:
    """Validate and normalize a complete, non-overlapping sample partition."""
    number_of_models = int(number_of_models)
    if number_of_models < len(SPLIT_NAMES):
        raise ValueError("At least three models are required for data splits.")
    if set(split_indices) != set(SPLIT_NAMES):
        raise ValueError(f"split_indices must contain exactly {list(SPLIT_NAMES)}.")

    normalized: dict[str, np.ndarray] = {}
    for split in SPLIT_NAMES:
        indices = np.asarray(split_indices[split], dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"The {split} split must be a non-empty vector.")
        if np.any(indices < 0) or np.any(indices >= number_of_models):
            raise ValueError(f"The {split} split contains an invalid index.")
        ordered = np.sort(indices)
        if np.any(ordered[1:] == ordered[:-1]):
            raise ValueError(f"The {split} split contains duplicate indices.")
        normalized[split] = ordered

    joined = np.concatenate([normalized[split] for split in SPLIT_NAMES])
    if len(joined) != number_of_models or not np.array_equal(
        np.sort(joined), np.arange(number_of_models, dtype=np.int64)
    ):
        raise ValueError("Dataset splits must cover every model exactly once.")
    return normalized


def build_dataset_split_indices(
    number_of_models: int,
    random_seed: int,
    *,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
) -> dict[str, np.ndarray]:
    """Return a reproducible, mutually exclusive train/validation/test partition."""
    number_of_models = int(number_of_models)
    train_fraction = float(train_fraction)
    validation_fraction = float(validation_fraction)
    test_fraction = 1.0 - train_fraction - validation_fraction
    if train_fraction <= 0.0 or validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("Train, validation, and test fractions must be positive.")

    permutation = np.random.default_rng(random_seed).permutation(number_of_models)
    train_end = int(number_of_models * train_fraction)
    validation_end = train_end + int(number_of_models * validation_fraction)
    return validate_dataset_split_indices(
        {
            "train": permutation[:train_end],
            "validation": permutation[train_end:validation_end],
            "test": permutation[validation_end:],
        },
        number_of_models,
    )
