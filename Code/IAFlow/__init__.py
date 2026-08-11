"""Data compression, training, evaluation, and inference for IA spectra."""

from .Config import ExperimentConfig, load_experiment_config
from .Data import NormalizationStats
from .Models import Conv1dAutoEncoder, build_autoencoder

__all__ = [
    "Conv1dAutoEncoder",
    "build_autoencoder",
    "ExperimentConfig",
    "NormalizationStats",
    "load_experiment_config",
]

__version__ = "0.1.0"
