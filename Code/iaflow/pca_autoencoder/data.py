"""
Source-ordered coefficient cache and datasets for the additive PCA-AE model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ia_models.utilities.data_schema import read_surface_dataset_description

from ..core.artifacts import save_json
from ..core.data import NormalizationStats, check_surface_cache
from .config import PCAAEExperimentConfig, PCAAEExperimentTemplate
from .transform import (
    PortablePCATransform,
    file_sha256,
    load_portable_pca_transform,
)

__all__ = [
    "PCA_AE_CACHE_FORMAT_VERSION",
    "CoefficientNormalization",
    "CachedPCACoefficientDataset",
    "check_pca_ae_cache",
    "load_pca_ae_cache_metadata",
    "load_pca_ae_transform",
    "prepare_pca_ae_cache",
]

PCA_AE_CACHE_FORMAT_VERSION = "1.0"
COEFFICIENT_FILE = "Coefficients.npy"
COEFFICIENT_NORMALIZATION_FILE = "CoefficientNormalization.npz"
PROJECTION_RESIDUAL_FILE = "ProjectionResidualSquaredNorm.npy"
BASELINE_NORM_FILE = "NormalizedSurfaceSquaredNorm.npy"
METADATA_FILE = "Metadata.json"


@dataclass(slots=True)
class CoefficientNormalization:
    """
    Store training-only mean and scale for raw PCA coefficients.
    """
    
    mean: np.ndarray
    scale: np.ndarray
    count: int
    
    def __post_init__(self) -> None:
        """
        Normalize dtypes and check coefficient statistics.
        """
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.scale = np.asarray(self.scale, dtype=np.float32)
        self.count = int(self.count)
        if self.mean.ndim != 1 or self.scale.shape != self.mean.shape:
            raise ValueError("Coefficient mean and scale must be matching vectors.")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)):
            raise ValueError("Coefficient statistics must contain only finite values.")
        if np.any(self.scale <= 0.0):
            raise ValueError("Every coefficient scale must be positive.")
        if self.count <= 0:
            raise ValueError("Coefficient normalization count must be positive.")
    
    @property
    def rank(self) -> int:
        """
        Return the coefficient-vector dimension.
        
        Returns:
            rank (int):
                Number of retained PCA coefficients.
        """
        return int(len(self.mean))
    
    def normalize(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Standardize raw PCA coefficients.
        
        Arguments:
            values (numpy.ndarray):
                Coefficient vector or batch with final dimension rank.
        
        Returns:
            normalized (numpy.ndarray):
                Float32 standardized coefficients.
        """
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1:] != (self.rank,):
            raise ValueError("Coefficient values have an invalid final dimension.")
        return (array - self.mean) / self.scale
    
    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Atomically save coefficient normalization statistics.
        
        Arguments:
            path (str or pathlib.Path):
                Final NPZ destination.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                mean=self.mean,
                scale=self.scale,
                count=np.asarray(self.count, dtype=np.int64),
            )
        os.replace(temporary, destination)
    
    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "CoefficientNormalization":
        """
        Load one checked coefficient-normalization artifact.
        
        Arguments:
            path (str or pathlib.Path):
                Stored NPZ artifact.
        
        Returns:
            normalization (CoefficientNormalization):
                Checked coefficient statistics.
        """
        with np.load(Path(path), allow_pickle=False) as stored:
            if set(stored.files) != {"mean", "scale", "count"}:
                raise ValueError("Coefficient normalization artifact is malformed.")
            return cls(
                mean=stored["mean"],
                scale=stored["scale"],
                count=int(stored["count"]),
            )


def load_pca_ae_transform(
    config: PCAAEExperimentConfig | PCAAEExperimentTemplate,
) -> PortablePCATransform:
    """
    Load the PCA transform selected by one PCA-AE configuration.
    
    Arguments:
        config (PCAAEExperimentConfig or PCAAEExperimentTemplate):
            Checked PCA-AE configuration.
    
    Returns:
        transform (PortablePCATransform):
            Authenticated rank-matched frozen PCA transform.
    """
    return load_portable_pca_transform(
        config.resolve_path(config.model.pca_transform_path),
        config.resolve_path(config.model.pca_transform_metadata_path),
        expected_rank=config.model.pca_rank,
        expected_input_shape=config.data.input_shape,
    )


def _temporary_path(
    path: Path,
) -> Path:
    """
    Return a hidden temporary sibling path.
    
    Arguments:
        path (pathlib.Path):
            Final cache artifact path.
    
    Returns:
        temporary (pathlib.Path):
            Hidden temporary sibling.
    """
    return path.with_name(f".{path.name}.tmp")


def _coefficient_cache_metadata(
    config: PCAAEExperimentConfig | PCAAEExperimentTemplate,
    transform: PortablePCATransform,
    surface_metadata: dict[str, Any],
    split_sizes: dict[str, int],
) -> dict[str, Any]:
    """
    Build the portable PCA-AE cache manifest.
    
    Arguments:
        config (PCAAEExperimentConfig or PCAAEExperimentTemplate):
            Checked PCA-AE configuration.
        transform (PortablePCATransform):
            Frozen PCA basis used for projection.
        surface_metadata (dict[str, Any]):
            Checked source-surface cache manifest.
        split_sizes (dict[str, int]):
            Authoritative train, validation, and test counts.
    
    Returns:
        metadata (dict[str, Any]):
            JSON-safe coefficient-cache manifest.
    """
    surface_metadata_path = config.resolve_path(config.data.cache_directory) / METADATA_FILE
    transform_metadata_path = config.resolve_path(
        config.model.pca_transform_metadata_path
    )
    return {
        "cache_format_version": PCA_AE_CACHE_FORMAT_VERSION,
        "source_path": config.data.source_path,
        "target_dataset": config.data.target_dataset,
        "source_structure_sha256": surface_metadata["source_structure_sha256"],
        "surface_cache_metadata_sha256": file_sha256(surface_metadata_path),
        "pca_transform_sha256": transform.artifact_sha256,
        "pca_transform_metadata_sha256": file_sha256(transform_metadata_path),
        "pca_rank": transform.rank,
        "input_shape": list(transform.input_shape),
        "source_size": int(surface_metadata["source_size"]),
        "split_sizes": split_sizes,
        "coefficient_file": COEFFICIENT_FILE,
        "coefficient_normalization_file": COEFFICIENT_NORMALIZATION_FILE,
        "projection_residual_file": PROJECTION_RESIDUAL_FILE,
        "baseline_norm_file": BASELINE_NORM_FILE,
        "coefficient_shape": [int(surface_metadata["source_size"]), transform.rank],
        "coefficient_dtype": "float32",
        "norm_dtype": "float64",
        "coefficient_centering": "training coefficient mean",
        "coefficient_scaling": "training population standard deviation",
        "surface_transform": config.data.transform,
        "surface_normalization": config.data.normalization,
    }


def prepare_pca_ae_cache(
    config: PCAAEExperimentConfig | PCAAEExperimentTemplate,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Prepare raw PCA coefficients and exact surface-objective auxiliaries.
    
    Arguments:
        config (PCAAEExperimentConfig or PCAAEExperimentTemplate):
            Checked PCA-AE experiment or template.
        overwrite (bool):
            Whether an existing complete cache may be regenerated.
    
    Returns:
        metadata (dict[str, Any]):
            Checked coefficient-cache manifest.
    """
    cache_directory = config.resolve_path(config.model.coefficient_cache_directory)
    metadata_path = cache_directory / METADATA_FILE
    if metadata_path.is_file() and not overwrite:
        return check_pca_ae_cache(config)
    cache_directory.mkdir(parents=True, exist_ok=True)
    surface_metadata = check_surface_cache(config)
    transform = load_pca_ae_transform(config)
    surface_cache_directory = config.resolve_path(config.data.cache_directory)
    surfaces = np.load(
        surface_cache_directory / surface_metadata["surface_file"],
        mmap_mode="r",
        allow_pickle=False,
    )
    surface_normalization = NormalizationStats.load(
        surface_cache_directory / surface_metadata["normalization_file"]
    )
    description = read_surface_dataset_description(
        config.resolve_path(config.data.source_path),
        config.data.target_dataset,
        config.data.input_shape,
    )
    split_sizes = {
        name: int(len(indices))
        for name, indices in description.split_indices.items()
    }
    number_of_surfaces = description.number_of_models
    coefficient_path = cache_directory / COEFFICIENT_FILE
    projection_residual_path = cache_directory / PROJECTION_RESIDUAL_FILE
    baseline_norm_path = cache_directory / BASELINE_NORM_FILE
    temporary_coefficients = _temporary_path(coefficient_path)
    temporary_projection = _temporary_path(projection_residual_path)
    temporary_baseline = _temporary_path(baseline_norm_path)
    temporary_paths = (
        temporary_coefficients,
        temporary_projection,
        temporary_baseline,
    )
    for path in temporary_paths:
        path.unlink(missing_ok=True)
    coefficients: np.memmap | None = None
    projection_residual: np.memmap | None = None
    baseline_norm: np.memmap | None = None
    try:
        coefficients = np.lib.format.open_memmap(
            temporary_coefficients,
            mode="w+",
            dtype=np.float32,
            shape=(number_of_surfaces, transform.rank),
        )
        projection_residual = np.lib.format.open_memmap(
            temporary_projection,
            mode="w+",
            dtype=np.float64,
            shape=(number_of_surfaces,),
        )
        baseline_norm = np.lib.format.open_memmap(
            temporary_baseline,
            mode="w+",
            dtype=np.float64,
            shape=(number_of_surfaces,),
        )
        block_size = config.data.preparation_block_size
        flattened_mean = surface_normalization.mean.reshape(-1)
        for start in range(0, number_of_surfaces, block_size):
            stop = min(start + block_size, number_of_surfaces)
            block = np.asarray(surfaces[start:stop], dtype=np.float32)
            flattened = block.reshape(len(block), transform.number_of_features)
            block_coefficients = (flattened - transform.mean) @ transform.components.T
            reconstructed = block_coefficients @ transform.components + transform.mean
            residual = flattened.astype(np.float64) - reconstructed.astype(np.float64)
            centered = flattened.astype(np.float64) - flattened_mean.astype(np.float64)
            coefficients[start:stop] = block_coefficients
            projection_residual[start:stop] = np.sum(
                np.square(residual),
                axis=1,
            )
            baseline_norm[start:stop] = np.sum(
                np.square(centered / surface_normalization.scale),
                axis=1,
            )
        coefficients.flush()
        projection_residual.flush()
        baseline_norm.flush()
        train_indices = description.split_indices["train"]
        training_coefficients = np.asarray(
            coefficients[train_indices],
            dtype=np.float64,
        )
        coefficient_mean = np.mean(training_coefficients, axis=0)
        coefficient_scale = np.std(training_coefficients, axis=0, ddof=0)
        coefficient_normalization = CoefficientNormalization(
            coefficient_mean,
            coefficient_scale,
            len(train_indices),
        )
        coefficient_normalization.save(
            cache_directory / COEFFICIENT_NORMALIZATION_FILE
        )
        coefficients = None
        projection_residual = None
        baseline_norm = None
        os.replace(temporary_coefficients, coefficient_path)
        os.replace(temporary_projection, projection_residual_path)
        os.replace(temporary_baseline, baseline_norm_path)
        metadata = _coefficient_cache_metadata(
            config,
            transform,
            surface_metadata,
            split_sizes,
        )
        save_json(metadata, metadata_path)
    except Exception:
        coefficients = None
        projection_residual = None
        baseline_norm = None
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise
    return check_pca_ae_cache(config)


def load_pca_ae_cache_metadata(
    cache_directory: str | Path,
) -> dict[str, Any]:
    """
    Load the PCA-AE cache manifest.
    
    Arguments:
        cache_directory (str or pathlib.Path):
            Directory containing Metadata.json.
    
    Returns:
        metadata (dict[str, Any]):
            Parsed manifest mapping.
    """
    path = Path(cache_directory) / METADATA_FILE
    if not path.is_file():
        raise FileNotFoundError(f"PCA-AE cache metadata not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("PCA-AE cache metadata must contain a mapping.")
    return metadata


def check_pca_ae_cache(
    config: PCAAEExperimentConfig | PCAAEExperimentTemplate,
) -> dict[str, Any]:
    """
    Validate coefficient-cache identity, arrays, and training statistics.
    
    Arguments:
        config (PCAAEExperimentConfig or PCAAEExperimentTemplate):
            Checked PCA-AE experiment or template.
    
    Returns:
        metadata (dict[str, Any]):
            Validated cache manifest.
    """
    cache_directory = config.resolve_path(config.model.coefficient_cache_directory)
    metadata = load_pca_ae_cache_metadata(cache_directory)
    if metadata.get("cache_format_version") != PCA_AE_CACHE_FORMAT_VERSION:
        raise ValueError("Unsupported PCA-AE coefficient-cache format.")
    surface_metadata = check_surface_cache(config)
    transform = load_pca_ae_transform(config)
    expected_split_sizes = {
        name: int(count)
        for name, count in surface_metadata["split_sizes"].items()
    }
    expected = _coefficient_cache_metadata(
        config,
        transform,
        surface_metadata,
        expected_split_sizes,
    )
    identity_keys = (
        "source_path",
        "target_dataset",
        "source_structure_sha256",
        "surface_cache_metadata_sha256",
        "pca_transform_sha256",
        "pca_transform_metadata_sha256",
        "pca_rank",
        "input_shape",
        "source_size",
        "coefficient_shape",
        "surface_transform",
        "surface_normalization",
    )
    if any(metadata.get(name) != expected.get(name) for name in identity_keys):
        raise ValueError("PCA-AE cache provenance does not match the current config.")
    if metadata.get("split_sizes") != expected_split_sizes:
        raise ValueError("PCA-AE cache split sizes do not match the surface cache.")
    coefficient_path = cache_directory / metadata["coefficient_file"]
    projection_path = cache_directory / metadata["projection_residual_file"]
    baseline_path = cache_directory / metadata["baseline_norm_file"]
    coefficients = np.load(coefficient_path, mmap_mode="r", allow_pickle=False)
    projection = np.load(projection_path, mmap_mode="r", allow_pickle=False)
    baseline = np.load(baseline_path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(metadata["coefficient_shape"])
    if coefficients.shape != expected_shape or coefficients.dtype != np.float32:
        raise ValueError("PCA-AE coefficient array has an invalid shape or dtype.")
    if projection.shape != (expected_shape[0],) or projection.dtype != np.float64:
        raise ValueError("PCA projection residual array is invalid.")
    if baseline.shape != (expected_shape[0],) or baseline.dtype != np.float64:
        raise ValueError("PCA baseline-norm array is invalid.")
    if not np.all(np.isfinite(projection)) or np.any(projection < 0.0):
        raise ValueError("PCA projection residuals must be finite and non-negative.")
    if not np.all(np.isfinite(baseline)) or np.any(baseline <= 0.0):
        raise ValueError("PCA baseline norms must be finite and positive.")
    normalization = CoefficientNormalization.load(
        cache_directory / metadata["coefficient_normalization_file"]
    )
    if normalization.rank != transform.rank:
        raise ValueError("Coefficient normalization rank is incompatible.")
    if normalization.count != metadata["split_sizes"].get("train"):
        raise ValueError("Coefficient normalization was not fit on the training split.")
    return metadata


class CachedPCACoefficientDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """
    Provide standardized coefficients and exact objective auxiliaries by split.
    """
    
    def __init__(
        self,
        config: PCAAEExperimentConfig,
        split: str,
    ) -> None:
        """
        Load one authoritative split view of the PCA-AE cache.
        
        Arguments:
            config (PCAAEExperimentConfig):
                Checked resolved PCA-AE configuration.
            split (str):
                Train, validation, or test split name.
        """
        metadata = check_pca_ae_cache(config)
        if split not in metadata["split_sizes"]:
            raise ValueError(f"Unknown PCA-AE split: {split!r}.")
        cache_directory = config.resolve_path(config.model.coefficient_cache_directory)
        self.values = np.load(
            cache_directory / metadata["coefficient_file"],
            mmap_mode="r",
            allow_pickle=False,
        )
        self.projection_residual = np.load(
            cache_directory / metadata["projection_residual_file"],
            mmap_mode="r",
            allow_pickle=False,
        )
        self.baseline_norm = np.load(
            cache_directory / metadata["baseline_norm_file"],
            mmap_mode="r",
            allow_pickle=False,
        )
        self.normalization = CoefficientNormalization.load(
            cache_directory / metadata["coefficient_normalization_file"]
        )
        description = read_surface_dataset_description(
            config.resolve_path(config.data.source_path),
            config.data.target_dataset,
            config.data.input_shape,
        )
        self.source_indices = description.split_indices[split]
        self.split = split
    
    def __len__(self) -> int:
        """
        Return the number of stored rows in the split.
        
        Returns:
            length (int):
                Split sample count.
        """
        return int(len(self.source_indices))
    
    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return standardized coefficients and two scalar objective terms.
        
        Arguments:
            index (int):
                Position within the selected split.
        
        Returns:
            sample (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                Coefficients, raw log10 PCA residual SSE, and normalized target SSE.
        """
        source_index = int(self.source_indices[index])
        coefficients = self.normalization.normalize(self.values[source_index])
        return (
            torch.from_numpy(coefficients),
            torch.tensor(self.projection_residual[source_index], dtype=torch.float64),
            torch.tensor(self.baseline_norm[source_index], dtype=torch.float64),
        )
