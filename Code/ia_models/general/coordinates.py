"""
Validation helpers for one-dimensional scientific coordinate grids.
"""

from __future__ import annotations

import numpy as np

__all__ = ["validate_coordinate_grid"]


def validate_coordinate_grid(
    values: object,
    name: str,
) -> np.ndarray:
    """
    Return a finite, strictly increasing one-dimensional float grid.
    
    Arguments:
        values (object):
            Values to convert to a scientific coordinate grid.
        name (str):
            Coordinate name used in validation errors.
    
    Returns:
        grid (numpy.ndarray):
            A finite, strictly increasing one-dimensional float array.
    """
    grid = np.asarray(values, dtype=float)
    
    if grid.ndim != 1 or len(grid) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must contain only finite values.")
    
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    
    return grid
