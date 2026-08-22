"""
Frozen-PCA coefficient autoencoder implementation.
"""

from .model import PCAAutoEncoder
from .artifacts import (
    build_pca_autoencoder,
    load_compatible_pca_autoencoder_checkpoint,
)
from .config import (
    PCAAEExperimentConfig,
    PCAAEExperimentTemplate,
    load_pca_ae_experiment_template,
    load_resolved_pca_ae_config,
)
from .data import (
    CoefficientNormalization,
    check_pca_ae_cache,
    load_pca_ae_transform,
    prepare_pca_ae_cache,
)
from .training import fit_pca_autoencoder
from .transform import PortablePCATransform

__all__ = [
    "CoefficientNormalization",
    "PCAAEExperimentConfig",
    "PCAAEExperimentTemplate",
    "PCAAutoEncoder",
    "PortablePCATransform",
    "build_pca_autoencoder",
    "check_pca_ae_cache",
    "fit_pca_autoencoder",
    "load_compatible_pca_autoencoder_checkpoint",
    "load_pca_ae_experiment_template",
    "load_pca_ae_transform",
    "load_resolved_pca_ae_config",
    "prepare_pca_ae_cache",
]
