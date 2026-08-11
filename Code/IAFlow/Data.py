"""Data preparation and PyTorch datasets for IA surface compression."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .Config import DataConfig, ExperimentConfig

__all__ = [
    "CACHE_FORMAT_VERSION",
    "SPLIT_NAMES",
    "CachedSurfaceDataset",
    "NormalizationStats",
    "build_dataloader",
    "load_cache_metadata",
    "prepare_surface_cache",
    "validate_surface_cache",
]

CACHE_FORMAT_VERSION = "1.0"
SPLIT_NAMES = ("train", "validation", "test")


def _split_stem(split: str) -> str:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split '{split}'; expected one of {SPLIT_NAMES}.")
    return split.capitalize()


@dataclass(slots=True)
class NormalizationStats:
    """Training-only centering surface and scalar RMS scale."""

    mean: np.ndarray
    scale: float
    count: int

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.scale = float(self.scale)
        self.count = int(self.count)
        if self.mean.ndim != 2 or not np.all(np.isfinite(self.mean)):
            raise ValueError("Normalization mean must be a finite two-dimensional array.")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("Normalization scale must be finite and positive.")
        if self.count <= 0:
            raise ValueError("Normalization sample count must be positive.")

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return (values - self.mean) / np.float32(self.scale)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return values * np.float32(self.scale) + self.mean

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                mean=self.mean,
                scale=np.asarray(self.scale, dtype=np.float64),
                count=np.asarray(self.count, dtype=np.int64),
            )
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        with np.load(Path(path), allow_pickle=False) as stored:
            return cls(
                mean=stored["mean"],
                scale=float(stored["scale"]),
                count=int(stored["count"]),
            )


class _RunningSurfaceStats:
    """Numerically stable, batch-wise Welford statistics over samples."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        block = np.asarray(values, dtype=np.float64)
        block_count = block.shape[0]
        block_mean = block.mean(axis=0)
        block_m2 = np.square(block - block_mean).sum(axis=0)
        if self.count == 0:
            self.count = block_count
            self.mean[...] = block_mean
            self.m2[...] = block_m2
            return

        total = self.count + block_count
        delta = block_mean - self.mean
        self.mean += delta * (block_count / total)
        self.m2 += block_m2 + np.square(delta) * (self.count * block_count / total)
        self.count = total

    def finalize(self) -> NormalizationStats:
        if self.count == 0:
            raise ValueError("Cannot finalize empty normalization statistics.")
        number_of_values = self.count * self.mean.size
        global_rms = np.sqrt(self.m2.sum() / number_of_values)
        return NormalizationStats(self.mean.astype(np.float32), global_rms, self.count)


def _read_source_description(
    source_path: Path,
    target_dataset: str,
    expected_shape: tuple[int, int],
) -> tuple[int, dict[str, np.ndarray], str]:
    with h5py.File(source_path, "r") as source:
        if target_dataset not in source:
            raise KeyError(f"HDF5 target '{target_dataset}' does not exist.")
        target = source[target_dataset]
        if target.ndim != 3 or tuple(target.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected target shape (N, {expected_shape[0]}, {expected_shape[1]}), "
                f"found {target.shape}."
            )
        if not bool(source.attrs.get("generation_complete", False)):
            raise ValueError("The source HDF5 dataset is not marked generation_complete.")

        split_indices: dict[str, np.ndarray] = {}
        for split in SPLIT_NAMES:
            dataset_name = f"splits/{split}"
            if dataset_name not in source:
                raise KeyError(f"Missing source split '{dataset_name}'.")
            indices = np.asarray(source[dataset_name][:], dtype=np.int64)
            if indices.ndim != 1 or len(indices) == 0:
                raise ValueError(f"Source split '{split}' must be a non-empty vector.")
            if np.any(indices[1:] <= indices[:-1]):
                raise ValueError(f"Source split '{split}' must be strictly increasing.")
            split_indices[split] = indices

        joined = np.concatenate(tuple(split_indices.values()))
        if len(joined) != target.shape[0] or not np.array_equal(
            np.sort(joined), np.arange(target.shape[0], dtype=np.int64)
        ):
            raise ValueError("Stored splits must cover each source row exactly once.")
        schema_version = str(source.attrs.get("schema_version", "unknown"))
        return int(target.shape[0]), split_indices, schema_version


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def prepare_surface_cache(
    config: ExperimentConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert the compressed HDF5 target into split-contiguous log10 arrays.

    The source target is read once in contiguous sample-axis blocks. Split rows
    are scattered into three NumPy ``.npy`` files that can subsequently be
    memory-mapped efficiently by PyTorch workers.
    """
    data = config.data
    source_path = config.resolve_path(data.source_path)
    cache_directory = config.resolve_path(data.cache_directory)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")
    cache_directory.mkdir(parents=True, exist_ok=True)

    metadata_path = cache_directory / "Metadata.json"
    if metadata_path.exists() and not overwrite:
        validate_surface_cache(config)
        return load_cache_metadata(cache_directory)

    number_of_rows, split_indices, schema_version = _read_source_description(
        source_path,
        data.target_dataset,
        data.input_shape,
    )

    data_paths = {
        split: cache_directory / f"{_split_stem(split)}.npy" for split in SPLIT_NAMES
    }
    index_paths = {
        split: cache_directory / f"{_split_stem(split)}Indices.npy"
        for split in SPLIT_NAMES
    }
    all_destinations = [*data_paths.values(), *index_paths.values()]
    if not overwrite and any(path.exists() for path in all_destinations):
        raise FileExistsError(
            f"Incomplete cache files already exist in {cache_directory}; "
            "inspect them or rerun with overwrite=True."
        )

    temporary_data_paths = {split: _temporary_path(path) for split, path in data_paths.items()}
    temporary_index_paths = {
        split: _temporary_path(path) for split, path in index_paths.items()
    }
    normalization_path = cache_directory / "Normalization.npz"
    temporary_normalization_path = _temporary_path(normalization_path)
    temporary_paths = [
        *temporary_data_paths.values(),
        *temporary_index_paths.values(),
        temporary_normalization_path,
    ]
    for path in temporary_paths:
        path.unlink(missing_ok=True)

    arrays: dict[str, np.memmap] = {}
    running_stats = _RunningSurfaceStats(data.input_shape)
    try:
        for split in SPLIT_NAMES:
            arrays[split] = np.lib.format.open_memmap(
                temporary_data_paths[split],
                mode="w+",
                dtype=np.float32,
                shape=(len(split_indices[split]), *data.input_shape),
            )
            with temporary_index_paths[split].open("wb") as stream:
                np.save(stream, split_indices[split], allow_pickle=False)

        with h5py.File(source_path, "r") as source:
            target = source[data.target_dataset]
            for start in range(0, number_of_rows, data.preparation_block_size):
                stop = min(start + data.preparation_block_size, number_of_rows)
                block = np.asarray(target[start:stop], dtype=np.float32)
                if not np.all(np.isfinite(block)) or np.any(block <= 0.0):
                    raise ValueError(
                        f"Target rows [{start}:{stop}] must be finite and strictly positive."
                    )
                np.log10(block, out=block)

                for split in SPLIT_NAMES:
                    indices = split_indices[split]
                    lower = int(np.searchsorted(indices, start, side="left"))
                    upper = int(np.searchsorted(indices, stop, side="left"))
                    if lower == upper:
                        continue
                    selected = block[indices[lower:upper] - start]
                    arrays[split][lower:upper] = selected
                    if split == "train":
                        running_stats.update(selected)

        for array in arrays.values():
            array.flush()
        arrays.clear()

        normalization = running_stats.finalize()
        normalization.save(temporary_normalization_path)

        source_stat = source_path.stat()
        metadata: dict[str, Any] = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "source_path": str(source_path),
            "source_size_bytes": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "source_schema_version": schema_version,
            "target_dataset": data.target_dataset,
            "transform": data.transform,
            "normalization": data.normalization,
            "input_shape": list(data.input_shape),
            "dtype": "float32",
            "split_sizes": {split: len(split_indices[split]) for split in SPLIT_NAMES},
            "data_files": {split: path.name for split, path in data_paths.items()},
            "index_files": {split: path.name for split, path in index_paths.items()},
            "normalization_file": normalization_path.name,
        }

        for split in SPLIT_NAMES:
            os.replace(temporary_data_paths[split], data_paths[split])
            os.replace(temporary_index_paths[split], index_paths[split])
        os.replace(temporary_normalization_path, normalization_path)
        temporary_metadata = _temporary_path(metadata_path)
        with temporary_metadata.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_metadata, metadata_path)
    except Exception:
        arrays.clear()
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise

    validate_surface_cache(config)
    return metadata


def load_cache_metadata(cache_directory: str | Path) -> dict[str, Any]:
    metadata_path = Path(cache_directory) / "Metadata.json"
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise TypeError(f"Cache metadata is not a mapping: {metadata_path}")
    return metadata


def validate_surface_cache(config: ExperimentConfig) -> dict[str, Any]:
    """Validate cache provenance, shapes, split indices, and normalization."""
    cache_directory = config.resolve_path(config.data.cache_directory)
    metadata = load_cache_metadata(cache_directory)
    if metadata.get("cache_format_version") != CACHE_FORMAT_VERSION:
        raise ValueError("Unsupported or stale surface-cache format.")
    if tuple(metadata.get("input_shape", ())) != config.data.input_shape:
        raise ValueError("Cached input shape does not match the experiment configuration.")
    if metadata.get("target_dataset") != config.data.target_dataset:
        raise ValueError("Cached target dataset does not match the experiment configuration.")
    if metadata.get("transform") != config.data.transform:
        raise ValueError("Cached transform does not match the experiment configuration.")

    for split in SPLIT_NAMES:
        data_path = cache_directory / metadata["data_files"][split]
        index_path = cache_directory / metadata["index_files"][split]
        values = np.load(data_path, mmap_mode="r", allow_pickle=False)
        indices = np.load(index_path, mmap_mode="r", allow_pickle=False)
        expected_size = int(metadata["split_sizes"][split])
        expected_shape = (expected_size, *config.data.input_shape)
        if values.shape != expected_shape or values.dtype != np.float32:
            raise ValueError(f"Invalid cached data array for split '{split}'.")
        if indices.shape != (expected_size,) or indices.dtype != np.int64:
            raise ValueError(f"Invalid cached index array for split '{split}'.")

    normalization = NormalizationStats.load(
        cache_directory / metadata["normalization_file"]
    )
    if normalization.mean.shape != config.data.input_shape:
        raise ValueError("Cached normalization shape is invalid.")
    if normalization.count != int(metadata["split_sizes"]["train"]):
        raise ValueError("Cached normalization was not computed from the full training split.")
    return metadata


class CachedSurfaceDataset(Dataset[torch.Tensor]):
    """Memory-mapped, normalized ``(channel=31, length=101)`` surfaces."""

    def __init__(self, cache_directory: str | Path, split: str) -> None:
        self.cache_directory = Path(cache_directory)
        self.split = split
        metadata = load_cache_metadata(self.cache_directory)
        stem = _split_stem(split)
        self.values = np.load(
            self.cache_directory / metadata["data_files"][split],
            mmap_mode="r",
            allow_pickle=False,
        )
        self.source_indices = np.load(
            self.cache_directory / metadata["index_files"][split],
            mmap_mode="r",
            allow_pickle=False,
        )
        self.normalization = NormalizationStats.load(
            self.cache_directory / metadata["normalization_file"]
        )
        if self.values.shape[1:] != self.normalization.mean.shape:
            raise ValueError(f"Data and normalization shapes disagree for split '{split}'.")

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        surface = np.asarray(self.values[index], dtype=np.float32)
        normalized = self.normalization.normalize(surface)
        return torch.from_numpy(normalized)


def build_dataloader(
    dataset: Dataset,
    data_config: DataConfig,
    *,
    training: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    """Build a reproducibly shuffled training loader or ordered evaluation loader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = data_config.num_workers
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": data_config.batch_size if training else data_config.evaluation_batch_size,
        "shuffle": training,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "generator": generator,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(**options)
