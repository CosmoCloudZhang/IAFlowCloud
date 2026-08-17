"""
Data compression, training, evaluation, and inference for IA spectra.
"""

from .architectures import (
    AutoEncoder,
    Conv1dAutoEncoder,
    Conv2dAutoEncoder,
    build_autoencoder,
)
from .config import (
    ExperimentConfig,
    ExperimentTemplate,
    load_experiment_config,
    load_experiment_template,
    load_resolved_experiment_config,
)
from .data import NormalizationStats

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
