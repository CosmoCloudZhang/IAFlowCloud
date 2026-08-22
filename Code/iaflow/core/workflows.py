"""
High-level workflow operations shared by direct AE and PCA-AE commands.
"""

from __future__ import annotations

from pathlib import Path

from .runs import propose_run_directory

__all__ = [
    "apply_training_overrides",
    "resolve_run_directory",
    "template_paths",
]


def resolve_run_directory(
    template: object,
    latent_dim: int,
    *,
    explicit: str | Path | None = None,
    resume: str | Path | None = None,
    family: str = "AE",
) -> Path:
    """
    Resolve a collision-free new run or the directory of a resumed run.
    
    Arguments:
        template (object):
            Checked model-family template with resolve_path and output sections.
        latent_dim (int):
            Positive bottleneck dimension.
        explicit (str or pathlib.Path or None):
            Optional explicit run directory.
        resume (str or pathlib.Path or None):
            Optional checkpoint used to continue an existing run.
        family (str):
            AE or PCA-AE label used to preserve command error contracts.
    
    Returns:
        directory (pathlib.Path):
            Concrete new or resumed run directory.
    """
    if resume is not None:
        checkpoint = template.resolve_path(resume)
        if not checkpoint.is_file():
            prefix = "PCA-AE resume" if family == "PCA-AE" else "Resume"
            raise FileNotFoundError(f"{prefix} checkpoint not found: {checkpoint}")
        if explicit is not None and template.resolve_path(explicit) != checkpoint.parent:
            checkpoint_name = (
                "the PCA-AE resume checkpoint"
                if family == "PCA-AE"
                else "the resume checkpoint"
            )
            raise ValueError(f"--run-directory must contain {checkpoint_name}.")
        return checkpoint.parent
    return propose_run_directory(
        template,
        latent_dim,
        explicit=explicit,
    )


def apply_training_overrides(
    config: object,
    *,
    device: str | None = None,
    epochs: int | None = None,
    seed: int | None = None,
) -> None:
    """
    Apply checked command-line training overrides and revalidate the config.
    
    Arguments:
        config (object):
            Resolved model-family configuration with training and check members.
        device (str or None):
            Optional explicit compute device.
        epochs (int or None):
            Optional positive total epoch count.
        seed (int or None):
            Optional non-negative reproducibility seed.
    """
    if device is not None:
        config.training.device = device
    
    if epochs is not None:
        if epochs <= 0:
            raise ValueError("--epochs must be positive.")
        config.training.epochs = epochs
    
    if seed is not None:
        if seed < 0:
            raise ValueError("--seed cannot be negative.")
        config.training.seed = seed
    config.check()


def template_paths(
    requested: list[Path],
    *,
    family: str,
) -> list[Path]:
    """
    Expand template files and directories into an ordered unique list.
    
    Arguments:
        requested (list[pathlib.Path]):
            Explicit template files or directories.
        family (str):
            Model-family label used in validation errors.
    
    Returns:
        paths (list[pathlib.Path]):
            Ordered unique DepthXX YAML template files.
    """
    paths: set[Path] = set()
    for value in requested:
        path = value.expanduser().resolve()
        if path.is_dir():
            paths.update(path.rglob("Depth*.yaml"))
        elif path.is_file():
            paths.add(path)
        else:
            if family == "PCA-AE":
                raise FileNotFoundError(f"PCA-AE configuration not found: {path}")
            raise FileNotFoundError(f"Configuration path not found: {path}")
    if not paths:
        if family == "PCA-AE":
            raise ValueError("No PCA-AE Depth*.yaml templates were selected.")
        raise ValueError("No Depth*.yaml experiment templates were selected.")
    return sorted(paths)
