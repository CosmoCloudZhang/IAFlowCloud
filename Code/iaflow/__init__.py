"""
Data compression, training, evaluation, and inference for IA spectra.
"""

from .autoencoder import (
    AutoEncoder,
    Conv1dAutoEncoder,
    Conv2dAutoEncoder,
    ExperimentConfig,
    ExperimentTemplate,
    build_autoencoder,
    load_experiment_config,
    load_experiment_template,
    load_resolved_experiment_config,
)
from .core.data import NormalizationStats

__all__ = [
    "AutoEncoder",
    "Conv1dAutoEncoder",
    "Conv2dAutoEncoder",
    "build_autoencoder",
    "ExperimentConfig",
    "ExperimentTemplate",
    "NormalizationStats",
    "load_experiment_config",
    "load_experiment_template",
    "load_resolved_experiment_config",
]

__version__ = "0.1.0"
