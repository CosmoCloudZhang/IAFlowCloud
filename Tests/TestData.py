from pathlib import Path

import h5py
import numpy as np

from IAFlow.Config import DataConfig, ExperimentConfig
from IAFlow.Data import (
    CachedSurfaceDataset,
    NormalizationStats,
    prepare_surface_cache,
    validate_surface_cache,
)


def write_synthetic_source(path: Path, shape=(3, 9), number_of_rows=20) -> np.ndarray:
    rng = np.random.default_rng(21)
    log_values = rng.normal(scale=0.3, size=(number_of_rows, *shape)).astype(np.float32)
    values = np.power(10.0, log_values).astype(np.float32)
    train = np.arange(0, 12, dtype=np.int64)
    validation = np.arange(12, 16, dtype=np.int64)
    test = np.arange(16, 20, dtype=np.int64)
    with h5py.File(path, "w") as destination:
        destination.attrs["generation_complete"] = True
        destination.attrs["schema_version"] = "test"
        destination.create_dataset("components/A_theta", data=values)
        splits = destination.create_group("splits")
        splits.create_dataset("train", data=train)
        splits.create_dataset("validation", data=validation)
        splits.create_dataset("test", data=test)
    return values


def synthetic_config(tmp_path: Path, shape=(3, 9)) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            source_path="Source.hdf5",
            cache_directory="Cache",
            input_shape=shape,
            preparation_block_size=5,
            batch_size=4,
            evaluation_batch_size=4,
        ),
        project_root=tmp_path,
    )


def test_prepare_cache_preserves_splits_and_training_normalization(tmp_path):
    config = synthetic_config(tmp_path)
    physical = write_synthetic_source(tmp_path / "Source.hdf5")
    metadata = prepare_surface_cache(config)
    validate_surface_cache(config)

    cached_train = np.load(tmp_path / "Cache" / "Train.npy")
    expected_train = np.log10(physical[:12])
    np.testing.assert_allclose(cached_train, expected_train, rtol=1e-6, atol=1e-6)
    assert metadata["split_sizes"] == {"train": 12, "validation": 4, "test": 4}

    normalization = NormalizationStats.load(tmp_path / "Cache" / "Normalization.npz")
    np.testing.assert_allclose(normalization.mean, expected_train.mean(axis=0), atol=1e-6)
    expected_scale = np.sqrt(np.mean(np.square(expected_train - expected_train.mean(axis=0))))
    assert np.isclose(normalization.scale, expected_scale, rtol=1e-6)

    dataset = CachedSurfaceDataset(tmp_path / "Cache", "validation")
    assert tuple(dataset[0].shape) == (3, 9)
    assert dataset.source_indices.tolist() == [12, 13, 14, 15]
