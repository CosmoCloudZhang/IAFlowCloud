"""
Load and validate the additive PCA-AE experiment configuration.
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
    _require_path_text,
    _source_record,
)

__all__ = [
    "PCAAEExperimentConfig",
    "PCAAEExperimentTemplate",
    "check_pca_ae_model_config",
    "load_pca_ae_experiment_template",
    "load_resolved_pca_ae_config",
]


_ROOT_KEYS = {"data", "model", "training", "output"}
_MODEL_KEYS = {
    "name",
    "latent_dim",
    "pca_rank",
    "pca_transform_path",
    "pca_transform_metadata_path",
    "coefficient_cache_directory",
    "dense_hidden",
    "activation",
    "normalization",
    "dropout",
    "coefficient_loss",
}


def _check_pca_ae_model_values(
    values: object,
    *,
    require_latent_dim: bool,
) -> dict[str, Any]:
    """
    Check one PCA-AE architecture section without changing direct-AE schemas.
    
    Arguments:
        values (object):
            Candidate PCA-AE model mapping.
        require_latent_dim (bool):
            Whether the runtime bottleneck dimension must be present.
    
    Returns:
        model (dict[str, Any]):
            Checked PCA-AE architecture configuration.
    """
    expected_keys = set(_MODEL_KEYS)
    if not require_latent_dim:
        expected_keys.remove("latent_dim")
    model = _check_mapping_keys(values, "'model'", expected_keys)
    if model["name"] != "PCA_AE":
        raise ValueError("PCA-AE configurations must use model.name='PCA_AE'.")
    model["pca_rank"] = _require_integer(model["pca_rank"], "model.pca_rank")
    if model["pca_rank"] <= 0:
        raise ValueError("model.pca_rank must be positive.")
    if require_latent_dim:
        model["latent_dim"] = _require_integer(
            model["latent_dim"],
            "model.latent_dim",
        )
        if model["latent_dim"] <= 0:
            raise ValueError("model.latent_dim must be positive.")
        if model["latent_dim"] > model["pca_rank"]:
            raise ValueError("model.latent_dim cannot exceed model.pca_rank.")
    for name in (
        "pca_transform_path",
        "pca_transform_metadata_path",
        "coefficient_cache_directory",
    ):
        model[name] = _require_path_text(model[name], f"model.{name}")
    model["dense_hidden"] = _require_integer_sequence(
        model["dense_hidden"],
        "model.dense_hidden",
    )
    if not model["dense_hidden"] or any(
        width <= 0 for width in model["dense_hidden"]
    ):
        raise ValueError("model.dense_hidden must contain positive widths.")
    if model["activation"] not in {"silu", "gelu", "relu"}:
        raise ValueError("model.activation must be silu, gelu, or relu.")
    if model["normalization"] != "none":
        raise ValueError("The first PCA-AE comparison requires normalization='none'.")
    model["dropout"] = _require_finite_real(model["dropout"], "model.dropout")
    if model["dropout"] != 0.0:
        raise ValueError("The first PCA-AE comparison requires dropout=0.0.")
    if model["coefficient_loss"] != "surface_equivalent_mse":
        raise ValueError(
            "model.coefficient_loss must be 'surface_equivalent_mse'."
        )
    return model


def check_pca_ae_model_config(
    values: object,
) -> dict[str, Any]:
    """
    Check a fully resolved PCA-AE model configuration.
    
    Arguments:
        values (object):
            Candidate resolved model mapping.
    
    Returns:
        model (dict[str, Any]):
            Checked PCA-AE model configuration.
    """
    return _check_pca_ae_model_values(values, require_latent_dim=True)


class PCAAEExperimentConfig:
    """
    Store one resolved PCA-AE experiment independently of direct-AE config.
    """
    
    def __init__(
        self,
        values: dict[str, Any],
        project_root: str | Path,
        *,
        runtime: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Construct and check every resolved PCA-AE section.
        
        Arguments:
            values (dict[str, Any]):
                Complete resolved PCA-AE experiment mapping.
            project_root (str or pathlib.Path):
                Repository root used to resolve portable paths.
            runtime (collections.abc.Mapping or None):
                Optional launch and provenance metadata.
        """
        configuration = _check_mapping_keys(values, "Configuration", _ROOT_KEYS)
        self.project_root = Path(project_root).expanduser().resolve()
        self.data = _Section(_check_data_config(configuration["data"]))
        self.model = _Section(check_pca_ae_model_config(configuration["model"]))
        self.training = _Section(_check_training_config(configuration["training"]))
        self.output = _Section(
            _check_output_config(configuration["output"], resolved=True)
        )
        self.runtime = copy.deepcopy(dict(runtime or {}))
    
    def resolve_path(self, path: str | Path) -> Path:
        """
        Resolve an absolute or repository-relative path.
        
        Arguments:
            path (str or pathlib.Path):
                Path to resolve.
        
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
        Recheck every mutable PCA-AE configuration section.
        """
        checked = {
            "data": _check_data_config(self.data),
            "model": check_pca_ae_model_config(self.model),
            "training": _check_training_config(self.training),
            "output": _check_output_config(self.output, resolved=True),
        }
        for name, values in checked.items():
            section = getattr(self, name)
            section.clear()
            section.update(values)


class PCAAEExperimentTemplate:
    """
    Store one reusable PCA-AE template before latent resolution.
    """
    
    def __init__(
        self,
        values: dict[str, Any],
        project_root: str | Path,
        *,
        configuration_sources: list[dict[str, str]],
    ) -> None:
        """
        Construct the checked reusable template.
        
        Arguments:
            values (dict[str, Any]):
                Composed template mapping without runtime latent dimension.
            project_root (str or pathlib.Path):
                Repository root used to resolve portable paths.
            configuration_sources (list[dict[str, str]]):
                Portable source paths and SHA-256 hashes.
        """
        configuration = _check_mapping_keys(values, "Template", _ROOT_KEYS)
        self.project_root = Path(project_root).expanduser().resolve()
        self.data = _Section(_check_data_config(configuration["data"]))
        self.model = _Section(
            _check_pca_ae_model_values(
                configuration["model"],
                require_latent_dim=False,
            )
        )
        self.training = _Section(_check_training_config(configuration["training"]))
        self.output = _Section(
            _check_output_config(configuration["output"], resolved=False)
        )
        self.configuration_sources = copy.deepcopy(configuration_sources)
    
    def resolve_path(self, path: str | Path) -> Path:
        """
        Resolve an absolute or repository-relative path.
        
        Arguments:
            path (str or pathlib.Path):
                Path to resolve.
        
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
    ) -> PCAAEExperimentConfig:
        """
        Resolve the latent dimension and run directory.
        
        Arguments:
            latent_dim (int):
                Positive PCA-AE bottleneck dimension.
            run_directory (str or pathlib.Path):
                Concrete run directory.
            runtime (collections.abc.Mapping or None):
                Optional launch metadata.
        
        Returns:
            config (PCAAEExperimentConfig):
                Fully resolved PCA-AE experiment.
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
        return PCAAEExperimentConfig(
            values,
            self.project_root,
            runtime=resolved_runtime,
        )


def load_pca_ae_experiment_template(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> PCAAEExperimentTemplate:
    """
    Load and compose one PCA-AE depth template.
    
    Arguments:
        path (str or pathlib.Path):
            PCA-AE DepthXX YAML template.
        project_root (str or pathlib.Path or None):
            Optional explicit repository root.
    
    Returns:
        template (PCAAEExperimentTemplate):
            Checked reusable PCA-AE template.
    """
    configuration_path = Path(path).expanduser().resolve()
    resolved_root = Path(
        project_root or _repository_root_from_config(configuration_path)
    ).resolve()
    values = _load_mapping(configuration_path)
    root = _check_mapping_keys(values, "Template", _ROOT_KEYS)
    sources = [_source_record(configuration_path, resolved_root)]
    composed = {
        name: _compose_template_section(
            root[name],
            name,
            resolved_root,
            sources,
        )
        for name in ("data", "model", "training", "output")
    }
    return PCAAEExperimentTemplate(
        composed,
        resolved_root,
        configuration_sources=sources,
    )


def load_resolved_pca_ae_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> PCAAEExperimentConfig:
    """
    Load one persisted PCA-AE ResolvedConfig.json.
    
    Arguments:
        path (str or pathlib.Path):
            Resolved configuration or its containing run directory.
        project_root (str or pathlib.Path or None):
            Optional explicit repository root.
    
    Returns:
        config (PCAAEExperimentConfig):
            Checked resolved PCA-AE experiment.
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
    return PCAAEExperimentConfig(root, resolved_root, runtime=runtime)
