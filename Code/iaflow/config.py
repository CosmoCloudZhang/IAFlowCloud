"""
Validated YAML configuration for IAFlow experiments.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

__all__ = ["ExperimentConfig", "config_to_dict", "load_experiment_config"]


class _Section(dict[str, Any]):
    """
    Provide attribute access over one YAML section without another schema class.
    """
    
    def __getattr__(self, name: str) -> Any:
        """
        Return a configuration value through attribute syntax.
        
        Arguments:
            name (str):
                Configuration key to retrieve.
        
        Returns:
            value (Any):
                Stored configuration value.
        """
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error
    
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Store a configuration value through attribute syntax.
        
        Arguments:
            name (str):
                Configuration key to update.
            value (Any):
                New configuration value.
        """
        self[name] = value


class ExperimentConfig:
    """
    Store one lightweight, validated authoritative experiment configuration.
    """
    
    def __init__(self, values: dict[str, Any], project_root: str | Path) -> None:
        """
        Construct and validate all configuration sections.
        
        Arguments:
            values (dict[str, Any]):
                Complete mapping loaded from the experiment YAML file.
            project_root (str or pathlib.Path):
                Repository root used to resolve portable paths.
        """
        self.project_root = Path(project_root).expanduser().resolve()
        self.data = _Section(values["data"])
        self.model = _Section(values["model"])
        self.training = _Section(values["training"])
        self.output = _Section(values["output"])
        self.validate()
    
    def resolve_path(self, path: str | Path) -> Path:
        """
        Resolve an absolute path or a path relative to the repository root.
        
        Arguments:
            path (str or pathlib.Path):
                Configured path to resolve.
        
        Returns:
            resolved_path (pathlib.Path):
                Expanded absolute path.
        """
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.project_root / candidate).resolve()
    
    def validate(self) -> None:
        """
        Normalize and validate every configured section.
        """
        data = self.data
        data["input_shape"] = tuple(int(value) for value in data["input_shape"])
        if len(data.input_shape) != 2 or min(data.input_shape) <= 0:
            raise ValueError("data.input_shape must contain two positive integers.")
        
        if data.transform != "log10":
            raise ValueError("Only the scientifically selected log10 transform is supported.")
        
        if data.normalization != "global_rms":
            raise ValueError("Only global_rms normalization is currently supported.")
        
        if int(data.preparation_block_size) <= 0:
            raise ValueError("data.preparation_block_size must be positive.")
        
        if int(data.batch_size) <= 0 or int(data.evaluation_batch_size) <= 0:
            raise ValueError("Data-loader batch sizes must be positive.")
        
        if int(data.num_workers) < 0:
            raise ValueError("data.num_workers cannot be negative.")
        
        from .architectures import validate_model_config
        
        validate_model_config(self.model)
        
        training = self.training
        for name in ("epochs", "seed", "scheduler_patience", "early_stopping_patience"):
            training[name] = int(training[name])
        for name in (
            "learning_rate",
            "weight_decay",
            "scheduler_factor",
            "minimum_learning_rate",
            "minimum_improvement",
            "target_variance_recovered",
        ):
            training[name] = float(training[name])
        if training.gradient_clip_norm is not None:
            training["gradient_clip_norm"] = float(training.gradient_clip_norm)
        
        if not isinstance(training.deterministic, bool) or not isinstance(
            training.mixed_precision,
            bool,
        ):
            raise ValueError(
                "training.deterministic and training.mixed_precision must be booleans."
            )
        
        if training.epochs <= 0 or training.early_stopping_patience <= 0:
            raise ValueError("Training epochs and early stopping patience must be positive.")
        
        if training.seed < 0:
            raise ValueError("training.seed cannot be negative.")
        
        if training.scheduler_patience < 0:
            raise ValueError("training.scheduler_patience cannot be negative.")
        
        if training.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("training.device must be auto, cpu, mps, or cuda.")
        
        if training.loss not in {"mse", "l1", "smooth_l1"}:
            raise ValueError("training.loss must be mse, l1, or smooth_l1.")
        
        if training.optimizer not in {"adam", "adamw"}:
            raise ValueError("training.optimizer must be adam or adamw.")
        
        if training.scheduler not in {"plateau", "cosine", "none"}:
            raise ValueError("training.scheduler must be plateau, cosine, or none.")
        
        if training.learning_rate <= 0.0 or training.weight_decay < 0.0:
            raise ValueError("Learning rate must be positive and weight decay non-negative.")
        
        if training.gradient_clip_norm is not None and training.gradient_clip_norm <= 0.0:
            raise ValueError("training.gradient_clip_norm must be positive or null.")
        
        if not 0.0 < training.scheduler_factor < 1.0:
            raise ValueError("training.scheduler_factor must lie in (0, 1).")
        
        if training.minimum_learning_rate <= 0.0 or training.minimum_improvement < 0.0:
            raise ValueError("Minimum learning rate must be positive and improvement non-negative.")
        
        if not 0.0 < training.target_variance_recovered <= 1.0:
            raise ValueError("training.target_variance_recovered must lie in (0, 1].")
        
        self.output["save_every_epochs"] = int(self.output.save_every_epochs)
        if self.output.save_every_epochs <= 0:
            raise ValueError("output.save_every_epochs must be positive.")


_SECTION_KEYS = {
    "data": {
        "source_path",
        "target_dataset",
        "cache_directory",
        "input_shape",
        "transform",
        "normalization",
        "preparation_block_size",
        "batch_size",
        "evaluation_batch_size",
        "num_workers",
    },
    "training": {
        "epochs", "seed", "device", "deterministic", "loss", "optimizer", "learning_rate",
        "weight_decay", "gradient_clip_norm", "mixed_precision", "scheduler", "scheduler_factor",
        "scheduler_patience", "minimum_learning_rate", "early_stopping_patience",
        "minimum_improvement", "target_variance_recovered",
    },
    "output": {"root_directory", "run_name", "save_every_epochs"},
}


def _repository_root_from_config(
    path: Path,
) -> Path:
    """
    Infer the repository root from a configuration-file location.
    
    Arguments:
        path (pathlib.Path):
            Absolute path of an experiment configuration file.
    
    Returns:
        project_root (pathlib.Path):
            Parent of the nearest Config directory, or the file's parent.
    """
    return next(
        (candidate.parent for candidate in path.parents if candidate.name == "Config"),
        path.parent,
    )


def load_experiment_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ExperimentConfig:
    """
    Load one complete YAML experiment configuration and validate its values.
    
    Arguments:
        path (str or pathlib.Path):
            YAML configuration file.
        project_root (str or pathlib.Path or None):
            Optional explicit repository root.
    
    Returns:
        config (ExperimentConfig):
            Validated experiment configuration.
    """
    configuration_path = Path(path).expanduser().resolve()
    with configuration_path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    required_sections = {*_SECTION_KEYS, "model"}
    if not isinstance(values, dict) or set(values) != required_sections:
        raise ValueError(f"Configuration must contain exactly {sorted(required_sections)}.")
    for section_name, expected_keys in _SECTION_KEYS.items():
        section = values[section_name]
        if not isinstance(section, dict) or set(section) != expected_keys:
            raise ValueError(
                f"'{section_name}' must contain exactly {sorted(expected_keys)}."
            )
    if not isinstance(values["model"], dict) or "name" not in values["model"]:
        raise ValueError("'model' must be a mapping containing a model name.")
    resolved_root = project_root or _repository_root_from_config(configuration_path)
    return ExperimentConfig(values, resolved_root)


def config_to_dict(
    config: ExperimentConfig,
) -> dict[str, Any]:
    """
    Return a JSON-safe snapshot of the YAML-backed experiment container.
    
    Arguments:
        config (ExperimentConfig):
            Validated experiment configuration.
    
    Returns:
        values (dict[str, Any]):
            Deep-copied configuration with a list-valued input shape.
    """
    values = {
        "data": copy.deepcopy(dict(config.data)),
        "model": copy.deepcopy(dict(config.model)),
        "training": copy.deepcopy(dict(config.training)),
        "output": copy.deepcopy(dict(config.output)),
    }
    values["data"]["input_shape"] = list(config.data.input_shape)
    return values
