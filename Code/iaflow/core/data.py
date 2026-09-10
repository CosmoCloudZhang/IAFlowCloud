"""
Source-ordered cache preparation and PyTorch datasets for IA compression.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ia_models.utilities.data_schema import (
    SurfaceDatasetDescription,
    read_surface_dataset_description,
)
from ia_models.utilities.data_split import SPLIT_NAMES

__all__ = [
    "CACHE_FORMAT_VERSION",
    "SPLIT_NAMES",
    "CachedSurfaceDataset",
    "NormalizationStats",
    "build_dataloader",
    "check_surface_cache",
    "load_cache_metadata",
    "prepare_surface_cache",
]

CACHE_FORMAT_VERSION = "3.0"


@dataclass(slots=True)
class NormalizationStats:
    """
    Store the training-only centering surface and global RMS scale.
    """
    
    mean: np.ndarray
    scale: float
    count: int
    
    def __post_init__(self) -> None:
        """
        Normalize stored dtypes and check the normalization statistics.
        """
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
        """
        Center and scale one surface or a batch of surfaces.
        
        Arguments:
            values (numpy.ndarray):
                Log10-space surfaces whose trailing dimensions match mean.
        
        Returns:
            normalized (numpy.ndarray):
                Float32 surfaces in normalized training space.
        """
        array = self._surface_array(values)
        return (array - self.mean) / np.float32(self.scale)
    
    def denormalize(self, values: np.ndarray) -> np.ndarray:
        """
        Restore normalized surfaces to centered log10 space.
        
        Arguments:
            values (numpy.ndarray):
                Normalized surfaces whose trailing dimensions match mean.
        
        Returns:
            denormalized (numpy.ndarray):
                Float32 log10-space surfaces.
        """
        array = self._surface_array(values)
        return array * np.float32(self.scale) + self.mean
    
    def _surface_array(self, values: np.ndarray) -> np.ndarray:
        """
        Check the rank and trailing shape of surface values.
        
        Arguments:
            values (numpy.ndarray):
                One surface or a batch of surfaces.
        
        Returns:
            array (numpy.ndarray):
                Float32 surface array with shape matching the stored mean.
        """
        array = np.asarray(values, dtype=np.float32)
        if array.ndim not in {2, 3} or tuple(array.shape[-2:]) != self.mean.shape:
            raise ValueError(
                "Surface values must have shape "
                f"{self.mean.shape} or (batch, {self.mean.shape[0]}, {self.mean.shape[1]})."
            )
        return array
    
    def save(self, path: str | Path) -> None:
        """
        Atomically save the normalization arrays and sample count.
        
        Arguments:
            path (str or pathlib.Path):
                Final NPZ destination.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_path(destination)
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
        """
        Load and check normalization statistics from an NPZ artifact.
        
        Arguments:
            path (str or pathlib.Path):
                Stored normalization artifact.
        
        Returns:
            normalization (NormalizationStats):
                Checked normalization statistics.
        """
        with np.load(Path(path), allow_pickle=False) as stored:
            return cls(
                mean=stored["mean"],
                scale=float(stored["scale"]),
                count=int(stored["count"]),
            )


class _RunningSurfaceStats:
    """
    Accumulate numerically stable batch-wise Welford statistics over samples.
    """
    
    def __init__(self, shape: tuple[int, int]) -> None:
        """
        Initialize empty statistics for one fixed surface shape.
        
        Arguments:
            shape (tuple[int, int]):
                Channel and data dimensions of one surface.
        """
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)
    
    def update(self, values: np.ndarray) -> None:
        """
        Merge one non-empty sample batch into the running statistics.
        
        Arguments:
            values (numpy.ndarray):
                Batch with shape (N_samples, *surface_shape).
        """
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
        """
        Convert accumulated moments into the selected normalization.
        
        Returns:
            normalization (NormalizationStats):
                Training mean surface, global RMS, and sample count.
        """
        if self.count == 0:
            raise ValueError("Cannot finalize empty normalization statistics.")
        global_rms = np.sqrt(self.m2.sum() / (self.count * self.mean.size))
        return NormalizationStats(self.mean.astype(np.float32), global_rms, self.count)


def _temporary_path(
    path: Path,
) -> Path:
    """
    Return the hidden temporary sibling used for atomic promotion.
    
    Arguments:
        path (pathlib.Path):
            Final artifact path.
    
    Returns:
        temporary_path (pathlib.Path):
            Hidden temporary path beside the final artifact.
    """
    return path.with_name(f".{path.name}.tmp")


def _portable_path(
    path: Path,
    project_root: Path,
) -> str:
    """
    Represent provenance relative to the project rather than one machine.
    
    Arguments:
        path (pathlib.Path):
            Absolute path to represent.
        project_root (pathlib.Path):
            Repository root used as the relative-path anchor.
    
    Returns:
        relative_path (str):
            Portable repository-relative path.
    """
    return os.path.relpath(path, project_root)


def _source_structure_fingerprint(
    source_path: Path,
    target_dataset: str,
) -> str:
    """
    Hash the lightweight HDF5 structure that determines cache interpretation.
    
    Arguments:
        source_path (pathlib.Path):
            Authoritative HDF5 source dataset.
        target_dataset (str):
            HDF5 path of the cached target array.
    
    Returns:
        fingerprint (str):
            SHA-256 digest of target metadata, coordinates, names, and splits.
    """
    digest = hashlib.sha256()
    with h5py.File(source_path, "r") as source:
        target = source[target_dataset]
        digest.update(target_dataset.encode("utf-8"))
        digest.update(str(target.shape).encode("ascii"))
        digest.update(str(target.dtype).encode("ascii"))
        for dataset_name in (
            "coordinates/z",
            "coordinates/k",
            "parameters/names",
            *(f"splits/{split}" for split in SPLIT_NAMES),
        ):
            if dataset_name not in source:
                raise KeyError(f"Missing source dataset required for provenance: {dataset_name}")
            values = np.asarray(source[dataset_name][:])
            digest.update(dataset_name.encode("utf-8"))
            digest.update(str(values.shape).encode("ascii"))
            digest.update(str(values.dtype).encode("ascii"))
            if values.dtype.kind in {"O", "S", "U"}:
                for value in values.reshape(-1):
                    encoded = value if isinstance(value, bytes) else str(value).encode("utf-8")
                    digest.update(len(encoded).to_bytes(8, "little"))
                    digest.update(encoded)
            else:
                digest.update(values.tobytes())
    return digest.hexdigest()


def _stream_log10_surfaces(
    source_path: Path,
    target_dataset: str,
    temporary_surfaces: Path,
    description: SurfaceDatasetDescription,
    input_shape: tuple[int, int],
    block_size: int,
) -> NormalizationStats:
    """
    Transform source surfaces in blocks and accumulate training-only statistics.
    
    Arguments:
        source_path (pathlib.Path):
            Authoritative HDF5 source dataset.
        target_dataset (str):
            HDF5 path of the positive target surfaces.
        temporary_surfaces (pathlib.Path):
            Temporary NPY destination for the source-ordered log10 surfaces.
        description (SurfaceDatasetDescription):
            Checked source dimensions and authoritative split indices.
        input_shape (tuple[int, int]):
            Channel and wavenumber dimensions of one surface.
        block_size (int):
            Number of source rows transformed per HDF5 read.
    
    Returns:
        normalization (NormalizationStats):
            Training-only mean surface, global RMS, and sample count.
    """
    surfaces: np.memmap | None = None
    running_stats = _RunningSurfaceStats(input_shape)
    train_indices = description.split_indices["train"]
    try:
        surfaces = np.lib.format.open_memmap(
            temporary_surfaces,
            mode="w+",
            dtype=np.float32,
            shape=(description.number_of_models, *input_shape),
        )
        with h5py.File(source_path, "r") as source:
            target = source[target_dataset]
            for start in range(0, description.number_of_models, block_size):
                stop = min(start + block_size, description.number_of_models)
                block = np.asarray(target[start:stop], dtype=np.float32)
                if not np.all(np.isfinite(block)) or np.any(block <= 0.0):
                    raise ValueError(
                        f"Target rows [{start}:{stop}] must be finite and strictly positive."
                    )
                np.log10(block, out=block)
                surfaces[start:stop] = block
                lower = int(np.searchsorted(train_indices, start, side="left"))
                upper = int(np.searchsorted(train_indices, stop, side="left"))
                if lower != upper:
                    running_stats.update(block[train_indices[lower:upper] - start])
        surfaces.flush()
    finally:
        surfaces = None
    return running_stats.finalize()


def _build_cache_metadata(
    config: object,
    source_path: Path,
    description: SurfaceDatasetDescription,
    surfaces_path: Path,
    normalization_path: Path,
) -> dict[str, Any]:
    """
    Build the portable manifest for one completed surface cache.
    
    Arguments:
        config (object):
            Checked experiment configuration.
        source_path (pathlib.Path):
            Authoritative HDF5 source dataset.
        description (SurfaceDatasetDescription):
            Checked source dimensions and authoritative split indices.
        surfaces_path (pathlib.Path):
            Final source-ordered NPY destination.
        normalization_path (pathlib.Path):
            Final training-normalization NPZ destination.
    
    Returns:
        metadata (dict[str, Any]):
            JSON-safe cache provenance and artifact description.
    """
    data = config.data
    source_stat = source_path.stat()
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "source_path": _portable_path(source_path, config.project_root),
        "source_size_bytes": source_stat.st_size,
        "source_structure_sha256": _source_structure_fingerprint(
            source_path,
            data.target_dataset,
        ),
        "target_dataset": data.target_dataset,
        "transform": data.transform,
        "normalization": data.normalization,
        "input_shape": list(data.input_shape),
        "dtype": "float32",
        "source_size": description.number_of_models,
        "split_sizes": {
            split: len(description.split_indices[split]) for split in SPLIT_NAMES
        },
        "surface_file": surfaces_path.name,
        "normalization_file": normalization_path.name,
    }


def prepare_surface_cache(
    config: object,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Create one source-ordered log10 cache and training-only normalization.
    
    The HDF5 split arrays remain authoritative. Only their sizes are recorded
    in cache metadata, and surfaces are never copied into split-specific files.
    Without overwrite, a compatible cache is checked and reused while an
    incompatible or partial cache is left unchanged with rebuild guidance.
    
    Arguments:
        config (object):
            Checked data and cache configuration.
        overwrite (bool):
            Whether to rebuild and replace an existing cache.
    
    Returns:
        metadata (dict[str, Any]):
            Checked compact cache manifest.
    """
    data = config.data
    source_path = config.resolve_path(data.source_path)
    cache_directory = config.resolve_path(data.cache_directory)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")
    cache_directory.mkdir(parents=True, exist_ok=True)
    
    metadata_path = cache_directory / "Metadata.json"
    if metadata_path.exists() and not overwrite:
        try:
            return check_surface_cache(config)
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(
                f"Existing cache cannot be reused: {error} "
                "No cache files were modified. Rerun with --overwrite or call "
                "prepare_surface_cache(..., overwrite=True) to rebuild it."
            ) from error
    
    description = read_surface_dataset_description(
        source_path, data.target_dataset, data.input_shape
    )
    surfaces_path = cache_directory / "Surfaces.npy"
    normalization_path = cache_directory / "Normalization.npz"
    if not overwrite and any(path.exists() for path in (surfaces_path, normalization_path)):
        raise FileExistsError(
            f"Incomplete cache files already exist in {cache_directory}. "
            "No cache files were modified; rerun with --overwrite or call "
            "prepare_surface_cache(..., overwrite=True) to rebuild them."
        )
    
    temporary_surfaces = _temporary_path(surfaces_path)
    temporary_normalization = _temporary_path(normalization_path)
    temporary_metadata = _temporary_path(metadata_path)
    for path in (temporary_surfaces, temporary_normalization, temporary_metadata):
        path.unlink(missing_ok=True)
    
    try:
        normalization = _stream_log10_surfaces(
            source_path,
            data.target_dataset,
            temporary_surfaces,
            description,
            data.input_shape,
            data.preparation_block_size,
        )
        normalization.save(temporary_normalization)
        metadata = _build_cache_metadata(
            config,
            source_path,
            description,
            surfaces_path,
            normalization_path,
        )
        with temporary_metadata.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_surfaces, surfaces_path)
        os.replace(temporary_normalization, normalization_path)
        os.replace(temporary_metadata, metadata_path)
    except Exception:
        for path in (temporary_surfaces, temporary_normalization, temporary_metadata):
            path.unlink(missing_ok=True)
        raise
    
    check_surface_cache(config)
    return metadata


def load_cache_metadata(
    cache_directory: str | Path,
) -> dict[str, Any]:
    """
    Load the source-ordered cache manifest.
    
    Arguments:
        cache_directory (str or pathlib.Path):
            Directory containing Metadata.json.
    
    Returns:
        metadata (dict[str, Any]):
            Parsed cache manifest.
    """
    metadata_path = Path(cache_directory) / "Metadata.json"
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise TypeError(f"Cache metadata is not a mapping: {metadata_path}")
    return metadata


def _check_cache_configuration(
    metadata: dict[str, Any],
    config: object,
) -> None:
    """
    Check cache-format and experiment-configuration metadata.
    
    Arguments:
        metadata (dict[str, Any]):
            Stored cache manifest.
        config (object):
            Current experiment configuration.
    """
    if metadata.get("cache_format_version") != CACHE_FORMAT_VERSION:
        raise ValueError("Unsupported or stale surface-cache format.")
    
    if tuple(metadata.get("input_shape", ())) != config.data.input_shape:
        raise ValueError("Cached input shape does not match the experiment configuration.")
    
    if metadata.get("target_dataset") != config.data.target_dataset:
        raise ValueError("Cached target dataset does not match the experiment configuration.")
    
    if metadata.get("transform") != config.data.transform:
        raise ValueError("Cached transform does not match the experiment configuration.")
    
    if metadata.get("normalization") != config.data.normalization:
        raise ValueError("Cached normalization does not match the experiment configuration.")
    
    if metadata.get("dtype") != "float32":
        raise ValueError("Cached surface dtype metadata is invalid.")


def _check_cache_source(
    metadata: dict[str, Any],
    config: object,
) -> SurfaceDatasetDescription:
    """
    Check the manifest against the current authoritative HDF5 source.
    
    Arguments:
        metadata (dict[str, Any]):
            Stored cache manifest.
        config (object):
            Current experiment and source-data configuration.
    
    Returns:
        description (SurfaceDatasetDescription):
            Checked source dimensions and authoritative split indices.
    """
    source_path = config.resolve_path(config.data.source_path)
    description = read_surface_dataset_description(
        source_path,
        config.data.target_dataset,
        config.data.input_shape,
    )
    source_stat = source_path.stat()
    path_matches = metadata.get("source_path") == _portable_path(
        source_path,
        config.project_root,
    )
    size_matches = metadata.get("source_size_bytes") == source_stat.st_size
    current_structure_sha256 = _source_structure_fingerprint(
        source_path,
        config.data.target_dataset,
    )
    structure_matches = (
        metadata.get("source_structure_sha256") == current_structure_sha256
    )
    if not path_matches or not size_matches or not structure_matches:
        raise ValueError("The cache does not match the current source HDF5 file.")
    return description


def _check_cache_artifacts(
    metadata: dict[str, Any],
    config: object,
    description: SurfaceDatasetDescription,
) -> None:
    """
    Check cached arrays, split sizes, and training normalization.
    
    Arguments:
        metadata (dict[str, Any]):
            Stored cache manifest.
        config (object):
            Current experiment configuration.
        description (SurfaceDatasetDescription):
            Checked source dimensions and authoritative split indices.
    """
    cache_directory = config.resolve_path(config.data.cache_directory)
    try:
        surface_file = metadata["surface_file"]
        normalization_file = metadata["normalization_file"]
    except KeyError as error:
        raise ValueError(f"Cache metadata is missing {error.args[0]!r}.") from error
    if not isinstance(surface_file, str) or not isinstance(normalization_file, str):
        raise ValueError("Cache artifact filenames must be strings.")
    
    values = np.load(
        cache_directory / surface_file,
        mmap_mode="r",
        allow_pickle=False,
    )
    if int(metadata.get("source_size", -1)) != description.number_of_models:
        raise ValueError("Cached source size does not match the HDF5 target.")
    expected_shape = (description.number_of_models, *config.data.input_shape)
    if values.shape != expected_shape or values.dtype != np.float32:
        raise ValueError("Invalid source-ordered cached surface array.")
    expected_split_sizes = {
        split: len(description.split_indices[split]) for split in SPLIT_NAMES
    }
    if metadata.get("split_sizes") != expected_split_sizes:
        raise ValueError("Cached split sizes do not match the authoritative HDF5 splits.")
    normalization = NormalizationStats.load(cache_directory / normalization_file)
    if normalization.mean.shape != config.data.input_shape:
        raise ValueError("Cached normalization shape is invalid.")
    
    if normalization.count != len(description.split_indices["train"]):
        raise ValueError("Cached normalization was not computed from the full training split.")


def check_surface_cache(
    config: object,
) -> dict[str, Any]:
    """
    Check source-order cache provenance, shape, splits, and normalization.
    
    Arguments:
        config (object):
            Current experiment and source-data configuration.
    
    Returns:
        metadata (dict[str, Any]):
            Cache manifest after every compatibility check succeeds.
    """
    metadata = load_cache_metadata(config.resolve_path(config.data.cache_directory))
    _check_cache_configuration(metadata, config)
    description = _check_cache_source(metadata, config)
    _check_cache_artifacts(metadata, config, description)
    return metadata


class CachedSurfaceDataset(Dataset[torch.Tensor]):
    """
    Provide a normalized split view of one source-ordered ``(31, 101)`` cache.
    """
    
    def __init__(self, config: object, split: str) -> None:
        """
        Resolve the cache and select authoritative HDF5 indices for one split.
        
        Arguments:
            config (object):
                Checked source-data and cache configuration.
            split (str):
                One of train, validation, or test.
        """
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split '{split}'; expected one of {SPLIT_NAMES}.")
        self.cache_directory = config.resolve_path(config.data.cache_directory)
        self.split = split
        metadata = check_surface_cache(config)
        self.values = np.load(
            self.cache_directory / metadata["surface_file"], mmap_mode="r", allow_pickle=False
        )
        source_path = config.resolve_path(config.data.source_path)
        description = read_surface_dataset_description(
            source_path, config.data.target_dataset, config.data.input_shape
        )
        self.source_indices = description.split_indices[split]
        self.normalization = NormalizationStats.load(
            self.cache_directory / metadata["normalization_file"]
        )
        if self.values.shape[1:] != self.normalization.mean.shape:
            raise ValueError(f"Data and normalization shapes disagree for split '{split}'.")
    
    def __len__(self) -> int:
        """
        Return the number of surfaces in this split.
        
        Returns:
            length (int):
                Number of authoritative source indices.
        """
        return int(len(self.source_indices))
    
    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Load and normalize one source-ordered surface.
        
        Arguments:
            index (int):
                Position within this split.
        
        Returns:
            surface (torch.Tensor):
                Normalized float32 surface with shape (31, 101).
        """
        surface = np.asarray(self.values[self.source_indices[index]], dtype=np.float32)
        return torch.from_numpy(self.normalization.normalize(surface))


def build_dataloader(
    dataset: Dataset,
    data_config: Any,
    *,
    training: bool,
    seed: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """
    Build a reproducibly shuffled training or ordered evaluation loader.
    
    Arguments:
        dataset (torch.utils.data.Dataset):
            Dataset exposed by the loader.
        data_config (Any):
            Data configuration containing batch sizes and worker count.
        training (bool):
            Whether to shuffle with the reproducible generator.
        seed (int):
            Seed used when a generator is not supplied.
        device (torch.device):
            Training device used to select pinned-memory behavior.
        generator (torch.Generator or None):
            Optional generator whose state can be checkpointed and restored.
    
    Returns:
        loader (torch.utils.data.DataLoader):
            Configured training or evaluation data loader.
    """
    if generator is None:
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
