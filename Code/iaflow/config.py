"""
YAML configuration loading and checking for IAFlow experiments.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ExperimentConfig",
    "check_input_shape",
    "check_model_config",
    "config_to_dict",
    "load_experiment_config",
]


_ROOT_KEYS = {"data", "model", "training", "output"}

_DATA_KEYS = {
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
}

_MODEL_KEYS = {
    "name",
    "latent_dim",
    "encoder_channels",
    "kernel_sizes",
    "strides",
    "dense_hidden",
    "activation",
    "normalization",
    "group_count",
    "dropout",
}

_TRAINING_KEYS = {
    "epochs",
    "seed",
    "device",
    "deterministic",
    "loss",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "mixed_precision",
    "scheduler",
    "scheduler_factor",
    "scheduler_patience",
    "minimum_learning_rate",
    "early_stopping_patience",
    "minimum_improvement",
    "target_variance_recovered",
}

_OUTPUT_KEYS = {"root_directory", "run_name", "save_every_epochs"}


def _check_mapping_keys(
    values: object,
    name: str,
    expected_keys: set[str],
) -> dict[str, Any]:
    """
    Return a copied mapping containing exactly the expected keys.
    
    Arguments:
        values (object):
            Candidate configuration mapping.
        name (str):
            Configuration mapping name used in error messages.
        expected_keys (set[str]):
            Complete permitted key set.
    
    Returns:
        mapping (dict[str, Any]):
            Shallow copy of the checked mapping.
    """
    if not isinstance(values, Mapping) or set(values) != expected_keys:
        raise ValueError(f"{name} must contain exactly {sorted(expected_keys)}.")
    return dict(values)


def _require_integer(
    value: Any,
    name: str,
) -> int:
    """
    Return a strictly integral configuration value.
    
    Arguments:
        value (Any):
            Candidate value. Booleans, strings, and fractional values are rejected.
        name (str):
            Fully qualified configuration key used in error messages.
    
    Returns:
        integer (int):
            Checked Python integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _require_finite_real(
    value: Any,
    name: str,
) -> float:
    """
    Return a finite real-valued configuration value.
    
    Arguments:
        value (Any):
            Candidate real value. Booleans and strings are rejected.
        name (str):
            Fully qualified configuration key used in error messages.
    
    Returns:
        scalar (float):
            Checked finite Python float.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number.")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _require_integer_sequence(
    values: Any,
    name: str,
) -> list[int]:
    """
    Return a list of strictly integral configuration values.
    
    Arguments:
        values (Any):
            Candidate list or tuple of integer values.
        name (str):
            Fully qualified configuration key used in error messages.
    
    Returns:
        integers (list[int]):
            Checked list of Python integers.
    """
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of integers.")
    return [
        _require_integer(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    ]


def check_input_shape(
    values: object,
    name: str,
) -> tuple[int, int]:
    """
    Check and normalize one configured two-dimensional input shape.
    
    Arguments:
        values (object):
            Candidate two-value shape.
        name (str):
            Fully qualified shape name used in error messages.
    
    Returns:
        input_shape (tuple[int, int]):
            Two positive integer dimensions.
    """
    try:
        shape_values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must contain two positive integers.") from error
    if len(shape_values) != 2:
        raise ValueError(f"{name} must contain two positive integers.")
    input_shape = tuple(
        _require_integer(value, f"{name}[{index}]")
        for index, value in enumerate(shape_values)
    )
    if min(input_shape) <= 0:
        raise ValueError(f"{name} must contain two positive integers.")
    return input_shape


def _check_data_config(
    values: object,
) -> dict[str, Any]:
    """
    Check and normalize the data configuration section.
    
    Arguments:
        values (object):
            Candidate data-section mapping.
    
    Returns:
        data (dict[str, Any]):
            Checked data configuration.
    """
    data = _check_mapping_keys(values, "'data'", _DATA_KEYS)
    data["input_shape"] = check_input_shape(data["input_shape"], "data.input_shape")
    
    if data["transform"] != "log10":
        raise ValueError("Only the scientifically selected log10 transform is supported.")
    
    if data["normalization"] != "global_rms":
        raise ValueError("Only global_rms normalization is currently supported.")
    
    for name in (
        "preparation_block_size",
        "batch_size",
        "evaluation_batch_size",
        "num_workers",
    ):
        data[name] = _require_integer(data[name], f"data.{name}")
    
    if data["preparation_block_size"] <= 0:
        raise ValueError("data.preparation_block_size must be positive.")
    
    if data["batch_size"] <= 0 or data["evaluation_batch_size"] <= 0:
        raise ValueError("Data-loader batch sizes must be positive.")
    
    if data["num_workers"] < 0:
        raise ValueError("data.num_workers cannot be negative.")
    return data


def check_model_config(
    values: object,
) -> dict[str, Any]:
    """
    Check and normalize the currently supported Conv1D model configuration.
    
    Arguments:
        values (object):
            Candidate model-section mapping.
    
    Returns:
        model (dict[str, Any]):
            Checked Conv1D model configuration.
    """
    model = _check_mapping_keys(values, "'model'", _MODEL_KEYS)
    if model["name"] != "Conv1D":
        raise ValueError(f"Unsupported model architecture: {model['name']!r}.")
    
    for name in ("latent_dim", "group_count"):
        model[name] = _require_integer(model[name], f"model.{name}")
    
    for name in ("encoder_channels", "kernel_sizes", "strides", "dense_hidden"):
        model[name] = _require_integer_sequence(model[name], f"model.{name}")
    
    if model["latent_dim"] <= 0 or model["group_count"] <= 0:
        raise ValueError("Model dimensions and group_count must be positive.")
    
    if not model["encoder_channels"]:
        raise ValueError("model.encoder_channels cannot be empty.")
    
    if len(model["kernel_sizes"]) != len(model["encoder_channels"]) or len(
        model["strides"]
    ) != len(model["encoder_channels"]):
        raise ValueError("encoder_channels, kernel_sizes, and strides must have equal lengths.")
    
    convolution_sizes = (
        model["encoder_channels"] + model["kernel_sizes"] + model["strides"]
    )
    if min(convolution_sizes) <= 0:
        raise ValueError("Convolution sizes must be positive.")
    
    if any(kernel % 2 == 0 for kernel in model["kernel_sizes"]):
        raise ValueError("Odd kernel sizes are required for symmetric padding.")
    
    if any(width <= 0 for width in model["dense_hidden"]):
        raise ValueError("All dense hidden widths must be positive.")
    
    if model["activation"] not in {"silu", "gelu", "relu"}:
        raise ValueError("model.activation must be silu, gelu, or relu.")
    
    if model["normalization"] not in {"group", "none"}:
        raise ValueError("model.normalization must be group or none.")
    model["dropout"] = _require_finite_real(model["dropout"], "model.dropout")
    
    if not 0.0 <= model["dropout"] < 1.0:
        raise ValueError("model.dropout must lie in [0, 1).")
    return model


def _check_training_config(
    values: object,
) -> dict[str, Any]:
    """
    Check and normalize the training configuration section.
    
    Arguments:
        values (object):
            Candidate training-section mapping.
    
    Returns:
        training (dict[str, Any]):
            Checked training configuration.
    """
    training = _check_mapping_keys(values, "'training'", _TRAINING_KEYS)
    for name in ("epochs", "seed", "scheduler_patience", "early_stopping_patience"):
        training[name] = _require_integer(training[name], f"training.{name}")
    for name in (
        "learning_rate",
        "weight_decay",
        "scheduler_factor",
        "minimum_learning_rate",
        "minimum_improvement",
        "target_variance_recovered",
    ):
        training[name] = _require_finite_real(training[name], f"training.{name}")
    if training["gradient_clip_norm"] is not None:
        training["gradient_clip_norm"] = _require_finite_real(
            training["gradient_clip_norm"],
            "training.gradient_clip_norm",
        )
    
    if not isinstance(training["deterministic"], bool) or not isinstance(
        training["mixed_precision"],
        bool,
    ):
        raise ValueError(
            "training.deterministic and training.mixed_precision must be booleans."
        )
    
    if training["epochs"] <= 0 or training["early_stopping_patience"] <= 0:
        raise ValueError("Training epochs and early stopping patience must be positive.")
    
    if training["seed"] < 0:
        raise ValueError("training.seed cannot be negative.")
    
    if training["scheduler_patience"] < 0:
        raise ValueError("training.scheduler_patience cannot be negative.")
    
    if training["device"] not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("training.device must be auto, cpu, mps, or cuda.")
    
    if training["loss"] not in {"mse", "l1", "smooth_l1"}:
        raise ValueError("training.loss must be mse, l1, or smooth_l1.")
    
    if training["optimizer"] not in {"adam", "adamw"}:
        raise ValueError("training.optimizer must be adam or adamw.")
    
    if training["scheduler"] not in {"plateau", "cosine", "none"}:
        raise ValueError("training.scheduler must be plateau, cosine, or none.")
    
    if training["learning_rate"] <= 0.0 or training["weight_decay"] < 0.0:
        raise ValueError("Learning rate must be positive and weight decay non-negative.")
    
    if (
        training["gradient_clip_norm"] is not None
        and training["gradient_clip_norm"] <= 0.0
    ):
        raise ValueError("training.gradient_clip_norm must be positive or null.")
    
    if not 0.0 < training["scheduler_factor"] < 1.0:
        raise ValueError("training.scheduler_factor must lie in (0, 1).")
    
    if (
        training["minimum_learning_rate"] <= 0.0
        or training["minimum_improvement"] < 0.0
    ):
        raise ValueError("Minimum learning rate must be positive and improvement non-negative.")
    
    if not 0.0 < training["target_variance_recovered"] <= 1.0:
        raise ValueError("training.target_variance_recovered must lie in (0, 1].")
    return training


def _check_output_config(
    values: object,
) -> dict[str, Any]:
    """
    Check and normalize the output configuration section.
    
    Arguments:
        values (object):
            Candidate output-section mapping.
    
    Returns:
        output (dict[str, Any]):
            Checked output configuration.
    """
    output = _check_mapping_keys(values, "'output'", _OUTPUT_KEYS)
    output["save_every_epochs"] = _require_integer(
        output["save_every_epochs"],
        "output.save_every_epochs",
    )
    if output["save_every_epochs"] <= 0:
        raise ValueError("output.save_every_epochs must be positive.")
    return output


def _check_experiment_values(
    values: object,
) -> dict[str, dict[str, Any]]:
    """
    Check and normalize every experiment configuration section.
    
    Arguments:
        values (object):
            Candidate complete experiment mapping.
    
    Returns:
        configuration (dict[str, dict[str, Any]]):
            Checked copies of all experiment sections.
    """
    configuration = _check_mapping_keys(values, "Configuration", _ROOT_KEYS)
    return {
        "data": _check_data_config(configuration["data"]),
        "model": check_model_config(configuration["model"]),
        "training": _check_training_config(configuration["training"]),
        "output": _check_output_config(configuration["output"]),
    }


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
    Store one lightweight, checked authoritative experiment configuration.
    """
    def __init__(self, values: dict[str, Any], project_root: str | Path) -> None:
        """
        Construct and check all configuration sections.
        
        Arguments:
            values (dict[str, Any]):
                Complete mapping loaded from the experiment YAML file.
            project_root (str or pathlib.Path):
                Repository root used to resolve portable paths.
        """
        checked_values = _check_experiment_values(values)
        self.project_root = Path(project_root).expanduser().resolve()
        self.data = _Section(checked_values["data"])
        self.model = _Section(checked_values["model"])
        self.training = _Section(checked_values["training"])
        self.output = _Section(checked_values["output"])
    
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
    
    def check(self) -> None:
        """
        Check and normalize every currently stored configuration section.
        """
        checked_values = _check_experiment_values(
            {
                "data": self.data,
                "model": self.model,
                "training": self.training,
                "output": self.output,
            }
        )
        for name, values in checked_values.items():
            section = getattr(self, name)
            section.clear()
            section.update(values)


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
    Load one complete YAML experiment configuration and check its values.
    
    Arguments:
        path (str or pathlib.Path):
            YAML configuration file.
        project_root (str or pathlib.Path or None):
            Optional explicit repository root.
    
    Returns:
        config (ExperimentConfig):
            Checked experiment configuration.
    """
    configuration_path = Path(path).expanduser().resolve()
    with configuration_path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    resolved_root = project_root or _repository_root_from_config(configuration_path)
    return ExperimentConfig(values, resolved_root)


def config_to_dict(
    config: ExperimentConfig,
) -> dict[str, Any]:
    """
    Return a JSON-safe snapshot of the YAML-backed experiment container.
    
    Arguments:
        config (ExperimentConfig):
            Checked experiment configuration.
    
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
