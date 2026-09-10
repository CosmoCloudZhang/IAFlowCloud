"""
Shared atomic serialization helpers for IAFlow model families.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

__all__ = ["portable_path", "save_checkpoint", "save_json"]


def portable_path(
    path: str | Path,
    project_root: str | Path,
) -> str:
    """
    Return a path relative to the project for portable artifact metadata.
    
    Arguments:
        path (str or pathlib.Path):
            Path to represent in an artifact.
        project_root (str or pathlib.Path):
            Repository root used as the relative-path anchor.
    
    Returns:
        relative_path (str):
            Portable path relative to project_root.
    """
    return os.path.relpath(Path(path).resolve(), Path(project_root).resolve())


def save_json(
    values: dict[str, Any] | list[Any],
    path: str | Path,
) -> None:
    """
    Atomically save a mapping or list as strict, human-readable JSON.
    
    Arguments:
        values (dict or list):
            JSON-serializable values to store.
        path (str or pathlib.Path):
            Final JSON destination.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(values, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, destination)


def save_checkpoint(
    checkpoint: dict[str, Any],
    path: str | Path,
) -> None:
    """
    Atomically save a PyTorch checkpoint.
    
    Arguments:
        checkpoint (dict[str, Any]):
            Checkpoint payload to serialize.
        path (str or pathlib.Path):
            Final checkpoint destination.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)
