"""Reusable machine-learning components for IA surface compression."""

from .AutoEncoder import Conv1dAutoEncoder
from .Config import ExperimentConfig, load_experiment_config
from .Data import NormalizationStats

__all__ = [
    "Conv1dAutoEncoder",
    "ExperimentConfig",
    "NormalizationStats",
    "load_experiment_config",
]

__version__ = "0.1.0"
