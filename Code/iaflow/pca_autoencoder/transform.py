"""
Portable frozen PCA transformation for the PCA-AE workflow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "PCA_TRANSFORM_FORMAT_VERSION",
    "PortablePCATransform",
    "file_sha256",
    "load_portable_pca_transform",
]

PCA_TRANSFORM_FORMAT_VERSION = "1.0"


def file_sha256(
    path: str | Path,
) -> str:
    """
    Return the SHA-256 digest of one file.
    
    Arguments:
        path (str or pathlib.Path):
            File whose bytes are hashed.
    
    Returns:
        digest (str):
            Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class PortablePCATransform:
    """
    Store one checked orthonormal PCA basis without scikit-learn.
    """
    
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    input_shape: tuple[int, int]
    training_count: int
    metadata: dict[str, Any]
    artifact_sha256: str
    
    def __post_init__(self) -> None:
        """
        Normalize array dtypes and validate the complete frozen transform.
        """
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.components = np.asarray(self.components, dtype=np.float32)
        self.explained_variance = np.asarray(
            self.explained_variance,
            dtype=np.float32,
        )
        self.explained_variance_ratio = np.asarray(
            self.explained_variance_ratio,
            dtype=np.float32,
        )
        self.input_shape = tuple(int(value) for value in self.input_shape)
        self.training_count = int(self.training_count)
        if len(self.input_shape) != 2 or min(self.input_shape) <= 0:
            raise ValueError("PCA input_shape must contain two positive dimensions.")
        number_of_features = int(np.prod(self.input_shape))
        rank = self.components.shape[0] if self.components.ndim == 2 else -1
        if self.mean.shape != (number_of_features,):
            raise ValueError("PCA mean has an invalid feature shape.")
        if self.components.shape != (rank, number_of_features) or rank <= 0:
            raise ValueError("PCA components have an invalid shape.")
        if self.explained_variance.shape != (rank,) or (
            self.explained_variance_ratio.shape != (rank,)
        ):
            raise ValueError("PCA variance arrays must match the retained rank.")
        arrays = (
            self.mean,
            self.components,
            self.explained_variance,
            self.explained_variance_ratio,
        )
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("PCA transform arrays must contain only finite values.")
        if np.any(self.explained_variance <= 0.0) or np.any(
            self.explained_variance_ratio <= 0.0
        ):
            raise ValueError("PCA retained variances must be positive.")
        if self.training_count <= 0:
            raise ValueError("PCA training_count must be positive.")
        identity = np.eye(rank, dtype=np.float32)
        orthogonality_error = float(
            np.max(np.abs(self.components @ self.components.T - identity))
        )
        if orthogonality_error >= 1.0e-4:
            raise ValueError("PCA components are not sufficiently orthonormal.")
        if len(self.artifact_sha256) != 64:
            raise ValueError("PCA artifact SHA-256 is invalid.")
    
    @property
    def rank(self) -> int:
        """
        Return the retained PCA component count.
        
        Returns:
            rank (int):
                Number of frozen PCA components.
        """
        return int(self.components.shape[0])
    
    @property
    def number_of_features(self) -> int:
        """
        Return the flattened surface feature count.
        
        Returns:
            count (int):
                Product of the two surface dimensions.
        """
        return int(np.prod(self.input_shape))
    
    def encode_log10(
        self,
        surfaces: np.ndarray,
    ) -> np.ndarray:
        """
        Project log10 surfaces onto the frozen PCA basis.
        
        Arguments:
            surfaces (numpy.ndarray):
                One surface or batch with trailing shape input_shape.
        
        Returns:
            coefficients (numpy.ndarray):
                Raw PCA coefficients with final dimension rank.
        """
        values = np.asarray(surfaces, dtype=np.float32)
        was_single = values.ndim == 2
        if was_single:
            values = values[None, ...]
        if values.ndim != 3 or tuple(values.shape[1:]) != self.input_shape:
            raise ValueError("PCA surfaces have an invalid shape.")
        flattened = values.reshape(len(values), self.number_of_features)
        coefficients = (flattened - self.mean) @ self.components.T
        return coefficients[0] if was_single else coefficients
    
    def decode_log10(
        self,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct log10 surfaces from raw PCA coefficients.
        
        Arguments:
            coefficients (numpy.ndarray):
                One coefficient vector or batch with final dimension rank.
        
        Returns:
            surfaces (numpy.ndarray):
                Reconstructed log10 surfaces with trailing input_shape.
        """
        values = np.asarray(coefficients, dtype=np.float32)
        was_single = values.ndim == 1
        if was_single:
            values = values[None, ...]
        if values.ndim != 2 or values.shape[1] != self.rank:
            raise ValueError("PCA coefficients have an invalid shape.")
        flattened = values @ self.components + self.mean
        surfaces = flattened.reshape(len(values), *self.input_shape)
        return surfaces[0] if was_single else surfaces


def load_portable_pca_transform(
    artifact_path: str | Path,
    metadata_path: str | Path,
    *,
    expected_rank: int | None = None,
    expected_input_shape: tuple[int, int] | None = None,
) -> PortablePCATransform:
    """
    Load and authenticate one portable PCA transform and metadata pair.
    
    Arguments:
        artifact_path (str or pathlib.Path):
            NPZ file containing the frozen numerical arrays.
        metadata_path (str or pathlib.Path):
            JSON file containing artifact identity and scientific provenance.
        expected_rank (int or None):
            Optional required retained component count.
        expected_input_shape (tuple[int, int] or None):
            Optional required surface shape.
    
    Returns:
        transform (PortablePCATransform):
            Checked frozen PCA transform.
    """
    artifact = Path(artifact_path)
    metadata_file = Path(metadata_path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Portable PCA artifact not found: {artifact}")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"PCA transform metadata not found: {metadata_file}")
    with metadata_file.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("PCA transform metadata must contain a mapping.")
    if metadata.get("format_version") != PCA_TRANSFORM_FORMAT_VERSION:
        raise ValueError("Unsupported portable PCA transform format.")
    artifact_digest = file_sha256(artifact)
    if metadata.get("artifact_sha256") != artifact_digest:
        raise ValueError("Portable PCA artifact digest does not match metadata.")
    with np.load(artifact, allow_pickle=False) as stored:
        required_arrays = {
            "mean",
            "components",
            "explained_variance",
            "explained_variance_ratio",
            "input_shape",
            "training_count",
        }
        if set(stored.files) != required_arrays:
            raise ValueError(
                f"Portable PCA artifact must contain exactly {sorted(required_arrays)}."
            )
        transform = PortablePCATransform(
            mean=stored["mean"],
            components=stored["components"],
            explained_variance=stored["explained_variance"],
            explained_variance_ratio=stored["explained_variance_ratio"],
            input_shape=tuple(stored["input_shape"].tolist()),
            training_count=int(stored["training_count"]),
            metadata=metadata,
            artifact_sha256=artifact_digest,
        )
    if expected_rank is not None and transform.rank != int(expected_rank):
        raise ValueError("Portable PCA rank does not match the experiment config.")
    if expected_input_shape is not None and transform.input_shape != tuple(
        expected_input_shape
    ):
        raise ValueError("Portable PCA input shape does not match the data config.")
    if int(metadata.get("rank", -1)) != transform.rank:
        raise ValueError("PCA metadata rank does not match its arrays.")
    if tuple(metadata.get("input_shape", ())) != transform.input_shape:
        raise ValueError("PCA metadata input shape does not match its arrays.")
    if metadata.get("whiten") is not False:
        raise ValueError("PCA-AE requires the validated non-whitened PCA basis.")
    return transform
