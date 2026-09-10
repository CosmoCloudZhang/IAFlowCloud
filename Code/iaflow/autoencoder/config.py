"""
Load, compose, resolve, and validate direct-AE experiment configurations.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..core.config import (
    _Section,
    _check_data_config,
    _check_mapping_keys,
    _check_output_config,
    _check_training_config,
    _compose_template_section,
    _load_mapping,
    _portable_path,
    _project_root_from_artifact,
    _repository_root_from_config,
    _require_finite_real,
    _require_integer,
    _require_integer_sequence,
    _source_record,
    check_input_shape,
    config_to_dict,
)

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


