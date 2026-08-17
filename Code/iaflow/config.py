"""
Load, compose, resolve, and validate IAFlow experiment configurations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ExperimentConfig",
    "ExperimentTemplate",
    "check_input_shape",
    "check_model_config",
    "config_to_dict",
    "load_experiment_config",
    "load_experiment_template",
    "load_resolved_experiment_config",
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

_TEMPLATE_OUTPUT_KEYS = {"root_directory", "save_every_epochs"}
_RESOLVED_OUTPUT_KEYS = {
    "root_directory",
    "run_directory",
    "save_every_epochs",
}


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
            Candidate value. Booleans, strings, and fractions are rejected.
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


def _require_pair_sequence(
    values: Any,
    name: str,
) -> list[tuple[int, int]]:
    """
    Return a list of positive two-dimensional integer pairs.
    
    Arguments:
        values (Any):
            Candidate list of two-value convolution shapes.
        name (str):
            Fully qualified configuration key used in error messages.
    
    Returns:
        pairs (list[tuple[int, int]]):
            Checked positive integer pairs.
    """
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of integer pairs.")
    return [
        check_input_shape(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    ]


def _require_path_text(
    value: Any,
    name: str,
) -> str:
    """
    Return a non-empty portable path string.
    
    Arguments:
        value (Any):
            Candidate configured path.
        name (str):
            Fully qualified configuration key used in error messages.
    
    Returns:
        path (str):
            Non-empty path text.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string.")
    return value


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
    for name in ("source_path", "target_dataset", "cache_directory"):
        data[name] = _require_path_text(data[name], f"data.{name}")
    
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


def _check_model_values(
    values: object,
    *,
    require_latent_dim: bool,
) -> dict[str, Any]:
    """
    Check a Conv1D or Conv2D architecture section.
    
    Arguments:
        values (object):
            Candidate model-section mapping.
        require_latent_dim (bool):
            Whether the runtime latent dimension must be present.
    
    Returns:
        model (dict[str, Any]):
            Checked architecture configuration.
    """
    expected_keys = set(_MODEL_KEYS)
    if not require_latent_dim:
        expected_keys.remove("latent_dim")
    model = _check_mapping_keys(values, "'model'", expected_keys)
    if model["name"] not in {"Conv1D", "Conv2D"}:
        raise ValueError(f"Unsupported model architecture: {model['name']!r}.")
    
    model["group_count"] = _require_integer(
        model["group_count"],
        "model.group_count",
    )
    if require_latent_dim:
        model["latent_dim"] = _require_integer(
            model["latent_dim"],
            "model.latent_dim",
        )
        if model["latent_dim"] <= 0:
            raise ValueError("model.latent_dim must be positive.")
    
    model["encoder_channels"] = _require_integer_sequence(
        model["encoder_channels"],
        "model.encoder_channels",
    )
    model["dense_hidden"] = _require_integer_sequence(
        model["dense_hidden"],
        "model.dense_hidden",
    )
    if model["name"] == "Conv1D":
        model["kernel_sizes"] = _require_integer_sequence(
            model["kernel_sizes"],
            "model.kernel_sizes",
        )
        model["strides"] = _require_integer_sequence(
            model["strides"],
            "model.strides",
        )
        if any(kernel % 2 == 0 for kernel in model["kernel_sizes"]):
            raise ValueError("Conv1D kernel sizes must be odd for symmetric padding.")
    else:
        model["kernel_sizes"] = _require_pair_sequence(
            model["kernel_sizes"],
            "model.kernel_sizes",
        )
        model["strides"] = _require_pair_sequence(
            model["strides"],
            "model.strides",
        )
        if any(
            dimension % 2 == 0
            for kernel in model["kernel_sizes"]
            for dimension in kernel
        ):
            raise ValueError("Conv2D kernel dimensions must be odd for symmetric padding.")
    
    if model["group_count"] <= 0:
        raise ValueError("model.group_count must be positive.")
    
    if not model["encoder_channels"]:
        raise ValueError("model.encoder_channels cannot be empty.")
    
    if len(model["kernel_sizes"]) != len(model["encoder_channels"]) or len(
        model["strides"]
    ) != len(model["encoder_channels"]):
        raise ValueError("encoder_channels, kernel_sizes, and strides must have equal lengths.")
    
    if min(model["encoder_channels"]) <= 0:
        raise ValueError("All encoder channels must be positive.")
    
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


def check_model_config(
    values: object,
) -> dict[str, Any]:
    """
    Check and normalize a fully resolved autoencoder configuration.
    
    Arguments:
        values (object):
            Candidate resolved model-section mapping.
    
    Returns:
        model (dict[str, Any]):
            Checked Conv1D or Conv2D model configuration.
    """
    return _check_model_values(values, require_latent_dim=True)


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
    
    if training["seed"] < 0 or training["scheduler_patience"] < 0:
        raise ValueError("Training seed and scheduler patience cannot be negative.")
    
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
    *,
    resolved: bool,
) -> dict[str, Any]:
    """
    Check a template or resolved output section.
    
    Arguments:
        values (object):
            Candidate output-section mapping.
        resolved (bool):
            Whether a concrete run directory must be present.
    
    Returns:
        output (dict[str, Any]):
            Checked output configuration.
    """
    keys = _RESOLVED_OUTPUT_KEYS if resolved else _TEMPLATE_OUTPUT_KEYS
    output = _check_mapping_keys(values, "'output'", keys)
    output["root_directory"] = _require_path_text(
        output["root_directory"],
        "output.root_directory",
    )
    if resolved:
        output["run_directory"] = _require_path_text(
            output["run_directory"],
            "output.run_directory",
        )
    output["save_every_epochs"] = _require_integer(
        output["save_every_epochs"],
        "output.save_every_epochs",
    )
    if output["save_every_epochs"] <= 0:
        raise ValueError("output.save_every_epochs must be positive.")
    return output


class _Section(dict[str, Any]):
    """
    Provide attribute access over one checked configuration section.
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
    Store one fully resolved and strictly checked experiment configuration.
    """
    
    def __init__(
        self,
        values: dict[str, Any],
        project_root: str | Path,
        *,
        runtime: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Construct all resolved configuration sections.
        
        Arguments:
            values (dict[str, Any]):
                Complete resolved experiment mapping.
            project_root (str or pathlib.Path):
                Repository root used to resolve portable paths.
            runtime (collections.abc.Mapping or None):
                Optional JSON-safe launch and provenance metadata.
        """
        configuration = _check_mapping_keys(values, "Configuration", _ROOT_KEYS)
        self.project_root = Path(project_root).expanduser().resolve()
        self.data = _Section(_check_data_config(configuration["data"]))
        self.model = _Section(check_model_config(configuration["model"]))
        self.training = _Section(_check_training_config(configuration["training"]))
        self.output = _Section(
            _check_output_config(configuration["output"], resolved=True)
        )
        self.runtime = copy.deepcopy(dict(runtime or {}))
    
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
        Recheck and normalize every mutable resolved section.
        """
        checked = {
            "data": _check_data_config(self.data),
            "model": check_model_config(self.model),
            "training": _check_training_config(self.training),
            "output": _check_output_config(self.output, resolved=True),
        }
        for name, values in checked.items():
            section = getattr(self, name)
            section.clear()
            section.update(values)


class ExperimentTemplate:
    """
    Store reusable experiment choices before runtime axes are supplied.
    """
    
    def __init__(
        self,
        values: dict[str, Any],
        project_root: str | Path,
        *,
        configuration_sources: list[dict[str, str]],
    ) -> None:
        """
        Construct a reusable experiment template.
        
        Arguments:
            values (dict[str, Any]):
                Composed template mapping without latent_dim or run_directory.
            project_root (str or pathlib.Path):
                Repository root used to resolve paths.
            configuration_sources (list[dict[str, str]]):
                Portable source paths and SHA-256 hashes used in composition.
        """
        configuration = _check_mapping_keys(values, "Template", _ROOT_KEYS)
        self.project_root = Path(project_root).expanduser().resolve()
        self.data = _Section(_check_data_config(configuration["data"]))
        self.model = _Section(
            _check_model_values(configuration["model"], require_latent_dim=False)
        )
        self.training = _Section(_check_training_config(configuration["training"]))
        self.output = _Section(
            _check_output_config(configuration["output"], resolved=False)
        )
        self.configuration_sources = copy.deepcopy(configuration_sources)
    
    def resolve_path(self, path: str | Path) -> Path:
        """
        Resolve an absolute path or a repository-relative path.
        
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
    
    def resolve(
        self,
        latent_dim: int,
        run_directory: str | Path,
        *,
        runtime: Mapping[str, Any] | None = None,
    ) -> ExperimentConfig:
        """
        Resolve runtime axes into a strict experiment configuration.
        
        Arguments:
            latent_dim (int):
                Positive bottleneck dimension for this run.
            run_directory (str or pathlib.Path):
                Concrete directory receiving the run artifacts.
            runtime (collections.abc.Mapping or None):
                Optional additional launch metadata.
        
        Returns:
            config (ExperimentConfig):
                Fully resolved experiment configuration.
        """
        latent_dim = _require_integer(latent_dim, "latent_dim")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        resolved_runtime = copy.deepcopy(dict(runtime or {}))
        resolved_runtime["configuration_sources"] = copy.deepcopy(
            self.configuration_sources
        )
        values = {
            "data": copy.deepcopy(dict(self.data)),
            "model": {
                **copy.deepcopy(dict(self.model)),
                "latent_dim": latent_dim,
            },
            "training": copy.deepcopy(dict(self.training)),
            "output": {
                **copy.deepcopy(dict(self.output)),
                "run_directory": _portable_path(
                    self.resolve_path(run_directory),
                    self.project_root,
                ),
            },
        }
        return ExperimentConfig(
            values,
            self.project_root,
            runtime=resolved_runtime,
        )


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


def _project_root_from_artifact(
    path: Path,
) -> Path:
    """
    Locate the nearest repository root containing pyproject.toml.
    
    Arguments:
        path (pathlib.Path):
            File or directory within the repository.
    
    Returns:
        project_root (pathlib.Path):
            Nearest project root.
    """
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate the repository root from {path}.")


def _portable_path(
    path: Path,
    project_root: Path,
) -> str:
    """
    Represent a path relative to the repository root.
    
    Arguments:
        path (pathlib.Path):
            Absolute path to represent.
        project_root (pathlib.Path):
            Repository root used as the path anchor.
    
    Returns:
        relative_path (str):
            Portable repository-relative path.
    """
    return os.path.relpath(path, project_root)


def _load_mapping(
    path: Path,
) -> dict[str, Any]:
    """
    Load a YAML or JSON mapping from disk.
    
    Arguments:
        path (pathlib.Path):
            Configuration source file.
    
    Returns:
        values (dict[str, Any]):
            Parsed top-level mapping.
    """
    with path.open("r", encoding="utf-8") as stream:
        if path.suffix.lower() == ".json":
            values = json.load(stream)
        else:
            values = yaml.safe_load(stream)
    if not isinstance(values, Mapping):
        raise ValueError(f"Configuration source must contain a mapping: {path}")
    return dict(values)


def _source_record(
    path: Path,
    project_root: Path,
) -> dict[str, str]:
    """
    Build portable identity metadata for one configuration source.
    
    Arguments:
        path (pathlib.Path):
            Configuration source file.
        project_root (pathlib.Path):
            Repository root used as the path anchor.
    
    Returns:
        record (dict[str, str]):
            Portable path and SHA-256 content digest.
    """
    return {
        "path": _portable_path(path, project_root),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _compose_template_section(
    value: Any,
    name: str,
    project_root: Path,
    sources: list[dict[str, str]],
) -> dict[str, Any]:
    """
    Resolve one inline or file-backed template section.
    
    Arguments:
        value (Any):
            Inline mapping or repository-relative YAML path.
        name (str):
            Section name used in error messages.
        project_root (pathlib.Path):
            Repository root used to resolve shared files.
        sources (list[dict[str, str]]):
            Mutable source-identity registry.
    
    Returns:
        section (dict[str, Any]):
            Composed section mapping.
    """
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise ValueError(f"Template section {name!r} must be a mapping or YAML path.")
    path = (project_root / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Shared configuration section not found: {path}")
    sources.append(_source_record(path, project_root))
    return _load_mapping(path)


def load_experiment_template(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ExperimentTemplate:
    """
    Load and compose one reusable YAML experiment template.
    
    Arguments:
        path (str or pathlib.Path):
            Architecture-depth YAML template.
        project_root (str or pathlib.Path or None):
            Optional explicit repository root.
    
    Returns:
        template (ExperimentTemplate):
            Checked template without runtime axes.
    """
    configuration_path = Path(path).expanduser().resolve()
    resolved_root = Path(
        project_root or _repository_root_from_config(configuration_path)
    ).resolve()
    values = _load_mapping(configuration_path)
    root = _check_mapping_keys(values, "Template", _ROOT_KEYS)
    sources = [_source_record(configuration_path, resolved_root)]
    composed = {
        name: _compose_template_section(root[name], name, resolved_root, sources)
        for name in ("data", "model", "training", "output")
    }
    return ExperimentTemplate(
        composed,
        resolved_root,
        configuration_sources=sources,
    )


def load_experiment_config(
    path: str | Path,
    *,
    latent_dim: int,
    run_directory: str | Path,
    project_root: str | Path | None = None,
) -> ExperimentConfig:
    """
    Load a template and resolve its two runtime axes.
    
    Arguments:
        path (str or pathlib.Path):
            Architecture-depth YAML template.
        latent_dim (int):
            Positive bottleneck dimension for this run.
        run_directory (str or pathlib.Path):
            Concrete directory receiving run artifacts.
        project_root (str or pathlib.Path or None):
            Optional explicit repository root.
    
    Returns:
        config (ExperimentConfig):
            Fully resolved experiment configuration.
    """
    template = load_experiment_template(path, project_root=project_root)
    return template.resolve(latent_dim, run_directory)


def load_resolved_experiment_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ExperimentConfig:
    """
    Load a run's persisted resolved configuration.
    
    Legacy AutoEncoder artifacts are migrated in memory only. Their files and
    checkpoint provenance remain unchanged.
    
    Arguments:
        path (str or pathlib.Path):
            ResolvedConfig.json path or its containing run directory.
        project_root (str or pathlib.Path or None):
            Optional explicit repository root.
    
    Returns:
        config (ExperimentConfig):
            Fully resolved run configuration.
    """
    requested = Path(path).expanduser().resolve()
    configuration_path = (
        requested / "ResolvedConfig.json" if requested.is_dir() else requested
    )
    values = _load_mapping(configuration_path)
    runtime = values.pop("runtime", {})
    root = _check_mapping_keys(values, "Resolved configuration", _ROOT_KEYS)
    if project_root is not None:
        resolved_root = Path(project_root).expanduser().resolve()
    else:
        try:
            resolved_root = _project_root_from_artifact(configuration_path)
        except FileNotFoundError:
            resolved_root = _project_root_from_artifact(Path.cwd())
    output = dict(root["output"])
    output.pop("run_name", None)
    if "run_directory" not in output:
        output["run_directory"] = _portable_path(
            configuration_path.parent,
            resolved_root,
        )
    root["output"] = output
    data = dict(root["data"])
    if data.get("cache_directory") == "Data/NLA/Cache/Log10ATheta":
        relocated_cache = resolved_root / "Data" / "NLA" / "Cache" / "Surface"
        if relocated_cache.is_dir():
            data["cache_directory"] = "Data/NLA/Cache/Surface"
    root["data"] = data
    return ExperimentConfig(root, resolved_root, runtime=runtime)


def _json_safe(
    value: Any,
) -> Any:
    """
    Convert tuples and nested configuration containers to JSON-safe values.
    
    Arguments:
        value (Any):
            Nested configuration value.
    
    Returns:
        converted (Any):
            Deep-copied JSON-safe value.
    """
    if isinstance(value, Mapping):
        return {name: _json_safe(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return copy.deepcopy(value)


def config_to_dict(
    config: ExperimentConfig,
) -> dict[str, Any]:
    """
    Return a JSON-safe snapshot of one resolved experiment configuration.
    
    Arguments:
        config (ExperimentConfig):
            Checked resolved experiment configuration.
    
    Returns:
        values (dict[str, Any]):
            Deep-copied configuration including runtime provenance.
    """
    return {
        "data": _json_safe(config.data),
        "model": _json_safe(config.model),
        "training": _json_safe(config.training),
        "output": _json_safe(config.output),
        "runtime": _json_safe(config.runtime),
    }
