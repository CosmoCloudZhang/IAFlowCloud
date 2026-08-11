"""Loss construction shared by IAFlow training implementations."""

from __future__ import annotations

from torch import nn

__all__ = ["build_reconstruction_loss"]


def build_reconstruction_loss(name: str) -> nn.Module:
    """Return a configured reconstruction loss by its YAML name."""
    try:
        return {"mse": nn.MSELoss, "l1": nn.L1Loss, "smooth_l1": nn.SmoothL1Loss}[name]()
    except KeyError as error:
        raise ValueError(f"Unsupported reconstruction loss: {name}") from error
