"""Data compression, training, evaluation, and inference for IA spectra."""

from .architectures import Conv1dAutoEncoder, build_autoencoder
from .config import ExperimentConfig, load_experiment_config
from .data import NormalizationStats

__all__ = [
    "Conv1dAutoEncoder",
    "build_autoencoder",
    "ExperimentConfig",
    "NormalizationStats",
    "load_experiment_config",
]

__version__ = "0.1.0"
