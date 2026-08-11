"""Typed YAML configuration for autoencoder experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "OutputConfig",
    "TrainingConfig",
    "config_to_dict",
    "load_experiment_config",
]


def _reject_unknown(section: str, values: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(values).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown keys in '{section}': {sorted(unknown)}")


@dataclass(slots=True)
class DataConfig:
    source_path: str = "Data/SAMPLE/nla_shape_final_prior.hdf5"
    target_dataset: str = "components/A_theta"
    cache_directory: str = "Data/ML/Log10ATheta"
    input_shape: tuple[int, int] = (31, 101)
    transform: str = "log10"
    normalization: str = "global_rms"
    preparation_block_size: int = 3125
    batch_size: int = 256
    evaluation_batch_size: int = 512
    num_workers: int = 0
    
    def __post_init__(self) -> None:
        self.input_shape = tuple(int(value) for value in self.input_shape)
        if len(self.input_shape) != 2 or min(self.input_shape) <= 0:
            raise ValueError("data.input_shape must contain two positive integers.")
        if self.transform != "log10":
            raise ValueError("Only the scientifically selected log10 transform is supported.")
        if self.normalization != "global_rms":
            raise ValueError("Only global_rms normalization is currently supported.")
        if self.preparation_block_size <= 0:
            raise ValueError("data.preparation_block_size must be positive.")
        if self.batch_size <= 0 or self.evaluation_batch_size <= 0:
            raise ValueError("Data-loader batch sizes must be positive.")
        if self.num_workers < 0:
            raise ValueError("data.num_workers cannot be negative.")


@dataclass(slots=True)
class ModelConfig:
    latent_dim: int = 2
    encoder_channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_sizes: list[int] = field(default_factory=lambda: [5, 5, 3])
    strides: list[int] = field(default_factory=lambda: [2, 2, 2])
    dense_hidden: list[int] = field(default_factory=lambda: [256, 64])
    activation: str = "silu"
    normalization: str = "group"
    group_count: int = 8
    dropout: float = 0.0
    
    def __post_init__(self) -> None:
        self.encoder_channels = [int(value) for value in self.encoder_channels]
        self.kernel_sizes = [int(value) for value in self.kernel_sizes]
        self.strides = [int(value) for value in self.strides]
        self.dense_hidden = [int(value) for value in self.dense_hidden]
        number_of_layers = len(self.encoder_channels)
        if number_of_layers == 0:
            raise ValueError("model.encoder_channels cannot be empty.")
        if len(self.kernel_sizes) != number_of_layers or len(self.strides) != number_of_layers:
            raise ValueError(
                "encoder_channels, kernel_sizes, and strides must have equal lengths."
            )
        if self.latent_dim <= 0:
            raise ValueError("model.latent_dim must be positive.")
        if min(self.encoder_channels + self.kernel_sizes + self.strides) <= 0:
            raise ValueError("Convolution sizes must be positive.")
        if any(kernel % 2 == 0 for kernel in self.kernel_sizes):
            raise ValueError("Odd kernel sizes are required for symmetric padding.")
        if any(width <= 0 for width in self.dense_hidden):
            raise ValueError("All dense hidden widths must be positive.")
        if self.activation not in {"silu", "gelu", "relu"}:
            raise ValueError("model.activation must be silu, gelu, or relu.")
        if self.normalization not in {"group", "none"}:
            raise ValueError("model.normalization must be group or none.")
        if self.group_count <= 0:
            raise ValueError("model.group_count must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must lie in [0, 1).")


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 300
    seed: int = 42
    device: str = "auto"
    deterministic: bool = False
    loss: str = "mse"
    optimizer: str = "adamw"
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float | None = 1.0
    mixed_precision: bool = True
    scheduler: str = "plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 12
    minimum_learning_rate: float = 1.0e-7
    early_stopping_patience: int = 35
    minimum_improvement: float = 1.0e-7
    target_variance_recovered: float = 0.999
    
    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("training.epochs must be positive.")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("training.device must be auto, cpu, mps, or cuda.")
        if self.loss not in {"mse", "l1", "smooth_l1"}:
            raise ValueError("training.loss must be mse, l1, or smooth_l1.")
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("training.optimizer must be adam or adamw.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Learning rate must be positive and weight decay non-negative.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("training.gradient_clip_norm must be positive or null.")
        if self.scheduler not in {"plateau", "cosine", "none"}:
            raise ValueError("training.scheduler must be plateau, cosine, or none.")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("training.scheduler_factor must lie in (0, 1).")
        if self.scheduler_patience < 0 or self.early_stopping_patience <= 0:
            raise ValueError("Scheduler patience cannot be negative; early stopping must be positive.")
        if self.minimum_learning_rate <= 0.0 or self.minimum_improvement < 0.0:
            raise ValueError("Minimum learning rate must be positive and improvement non-negative.")
        if not 0.0 < self.target_variance_recovered <= 1.0:
            raise ValueError("training.target_variance_recovered must lie in (0, 1].")


@dataclass(slots=True)
class OutputConfig:
    root_directory: str = "Runs/AutoEncoder"
    run_name: str | None = None
    save_every_epochs: int = 25
    
    def __post_init__(self) -> None:
        if self.save_every_epochs <= 0:
            raise ValueError("output.save_every_epochs must be positive.")


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    project_root: Path = field(default_factory=Path.cwd)
    
    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).expanduser().resolve()
    
    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (self.project_root / candidate).resolve()


def _build_section(cls: type, name: str, raw: Any):
    values = {} if raw is None else raw
    if not isinstance(values, dict):
        raise TypeError(f"The '{name}' configuration section must be a mapping.")
    allowed = set(cls.__dataclass_fields__)
    _reject_unknown(name, values, allowed)
    return cls(**values)


def load_experiment_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ExperimentConfig:
    """Load and validate an experiment YAML file.

    Relative data and output paths are resolved against the repository root,
    inferred as the parent of the directory containing the configuration file.
    """
    configuration_path = Path(path).expanduser().resolve()
    with configuration_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise TypeError("The experiment configuration must be a YAML mapping.")
    _reject_unknown("root", raw, {"data", "model", "training", "output"})

    inferred_root = configuration_path.parent.parent
    return ExperimentConfig(
        data=_build_section(DataConfig, "data", raw.get("data")),
        model=_build_section(ModelConfig, "model", raw.get("model")),
        training=_build_section(TrainingConfig, "training", raw.get("training")),
        output=_build_section(OutputConfig, "output", raw.get("output")),
        project_root=project_root or inferred_root,
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert a configuration to a JSON/YAML-safe dictionary."""
    values = asdict(config)
    values["project_root"] = str(config.project_root)
    values["data"]["input_shape"] = list(config.data.input_shape)
    return values
