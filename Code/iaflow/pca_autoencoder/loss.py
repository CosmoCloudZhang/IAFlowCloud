"""
Surface-equivalent weighted coefficient loss for PCA-AE training.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["SurfaceEquivalentCoefficientMSE"]


class SurfaceEquivalentCoefficientMSE(nn.Module):
    """
    Measure the trainable PCA-AE contribution to normalized surface MSE.
    """
    
    def __init__(
        self,
        coefficient_scale: torch.Tensor,
        number_of_surface_features: int,
        surface_scale: float,
    ) -> None:
        """
        Construct fixed weights from coefficient and surface normalization.
        
        Arguments:
            coefficient_scale (torch.Tensor):
                Training population standard deviations of raw PCA coefficients.
            number_of_surface_features (int):
                Flattened surface size, currently 3,131.
            surface_scale (float):
                Positive global RMS used to normalize log10 surfaces.
        """
        super().__init__()
        scales = torch.as_tensor(coefficient_scale, dtype=torch.float32)
        number_of_surface_features = int(number_of_surface_features)
        surface_scale = float(surface_scale)
        if scales.ndim != 1 or torch.any(~torch.isfinite(scales)):
            raise ValueError("coefficient_scale must be a finite vector.")
        if torch.any(scales <= 0.0):
            raise ValueError("coefficient_scale must be positive.")
        if number_of_surface_features <= 0 or surface_scale <= 0.0:
            raise ValueError("Surface feature count and scale must be positive.")
        weights = torch.square(scales) / (
            number_of_surface_features * surface_scale**2
        )
        self.register_buffer("weights", weights)
    
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the batch-mean weighted coefficient squared error.
        
        Arguments:
            prediction (torch.Tensor):
                Reconstructed standardized coefficients.
            target (torch.Tensor):
                Target standardized coefficients with matching shape.
        
        Returns:
            loss (torch.Tensor):
                Scalar normalized-surface-MSE contribution.
        """
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError("Coefficient loss expects matching two-dimensional tensors.")
        if prediction.shape[1] != len(self.weights):
            raise ValueError("Coefficient loss rank does not match its fixed weights.")
        squared_error = torch.square(prediction - target)
        return torch.mean(torch.sum(squared_error * self.weights, dim=1))
