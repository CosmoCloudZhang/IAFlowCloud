"""
Canonical run paths, discovery, status, and latest-run pointers.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from .config import ExperimentConfig, ExperimentTemplate

__all__ = [
    "completed_run",
    "discover_run_directories",
    "latest_run_directory",
    "latent_root_directory",
    "propose_run_directory",
    "write_latest_run",
]


def latent_root_directory(
    template: ExperimentTemplate,
    latent_dim: int,
) -> Path:
    """
    Return the canonical parent directory for one runtime latent dimension.
    
    Arguments:
        template (ExperimentTemplate):
            Reusable architecture-depth template.
        latent_dim (int):
            Positive bottleneck dimension.
    
    Returns:
        directory (pathlib.Path):
            Canonical ``LatentXX`` directory.
    """
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int):
        raise ValueError("latent_dim must be an integer.")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")
    return template.resolve_path(template.output.root_directory) / f"Latent{latent_dim:02d}"


def propose_run_directory(
    template: ExperimentTemplate,
    latent_dim: int,
    *,
    explicit: str | Path | None = None,
    timestamp: str | None = None,
) -> Path:
    """
    Resolve a collision-free run directory without creating it.
    
    Arguments:
        template (ExperimentTemplate):
            Reusable architecture-depth template.
        latent_dim (int):
            Positive bottleneck dimension.
        explicit (str or pathlib.Path or None):
            Optional explicit run directory.
        timestamp (str or None):
            Optional timestamp used instead of the current local time.
    
    Returns:
        directory (pathlib.Path):
            Proposed concrete run directory.
    """
    if explicit is not None:
        return template.resolve_path(explicit)
    base = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    parent = latent_root_directory(template, latent_dim)
    candidate = parent / base
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base}-{suffix:02d}"
        suffix += 1
    return candidate


def discover_run_directories(
    latent_root: str | Path,
) -> list[Path]:
    """
    Discover run directories ordered by name.
    
    Arguments:
        latent_root (str or pathlib.Path):
            Canonical ``LatentXX`` directory.
    
    Returns:
        directories (list[pathlib.Path]):
            Ordered child directories containing resolved configuration or results.
    """
    root = Path(latent_root).expanduser().resolve()
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and (
                (path / "ResolvedConfig.json").is_file()
                or (path / "Summary.json").is_file()
                or (path / "Best.pt").is_file()
            )
        ),
        key=lambda path: path.name,
    )


def completed_run(
    run_directory: str | Path,
) -> bool:
    """
    Report whether training produced its required selected artifacts.
    
    Arguments:
        run_directory (str or pathlib.Path):
            Candidate run directory.
    
    Returns:
        completed (bool):
            Whether Summary.json, Best.pt, and ResolvedConfig.json exist.
    """
    directory = Path(run_directory)
    return all(
        (directory / name).is_file()
        for name in ("Summary.json", "Best.pt", "ResolvedConfig.json")
    )


def latest_run_directory(
    latent_root: str | Path,
    *,
    project_root: str | Path,
    require_completed: bool = True,
) -> Path | None:
    """
    Resolve the latest run, tolerating stale legacy pointer paths.
    
    Arguments:
        latent_root (str or pathlib.Path):
            Canonical ``LatentXX`` directory.
        project_root (str or pathlib.Path):
            Repository root used to resolve portable pointer text.
        require_completed (bool):
            Whether the selected run must contain completed training artifacts.
    
    Returns:
        directory (pathlib.Path or None):
            Latest compatible run or None when no candidate exists.
    """
    root = Path(latent_root).expanduser().resolve()
    pointer = root / "LatestRun.txt"
    if pointer.is_file():
        text = pointer.read_text(encoding="utf-8").strip()
        if text:
            candidate = Path(text).expanduser()
            if not candidate.is_absolute():
                candidate = Path(project_root).expanduser().resolve() / candidate
            candidate = candidate.resolve()
            if candidate.is_dir() and (
                not require_completed or completed_run(candidate)
            ):
                return candidate
            relocated = root / candidate.name
            if relocated.is_dir() and (
                not require_completed or completed_run(relocated)
            ):
                return relocated
    candidates = discover_run_directories(root)
    if require_completed:
        candidates = [candidate for candidate in candidates if completed_run(candidate)]
    return candidates[-1] if candidates else None


def write_latest_run(
    config: ExperimentConfig,
    run_directory: str | Path,
) -> Path:
    """
    Atomically update the pointer beside one completed run.
    
    Arguments:
        config (ExperimentConfig):
            Resolved experiment configuration.
        run_directory (str or pathlib.Path):
            Completed run directory.
    
    Returns:
        pointer (pathlib.Path):
            Updated LatestRun.txt path.
    """
    directory = config.resolve_path(run_directory)
    pointer = directory.parent / "LatestRun.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_name(f".{pointer.name}.tmp")
    portable = os.path.relpath(directory, config.project_root)
    temporary.write_text(f"{portable}\n", encoding="utf-8")
    os.replace(temporary, pointer)
    return pointer
