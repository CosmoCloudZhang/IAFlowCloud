"""
Scientifically interpretable reconstruction metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["ReconstructionMetrics"]


@dataclass(slots=True)
class ReconstructionMetrics:
    """
    Stream normalized-space errors and physical-space tail diagnostics.
    
    The variance denominator is the held-out squared distance from the training
    mean surface. Because normalization uses one global scalar, this is exactly
    the total log10-space variance recovery ``1 - SSE / SST``.
    """
    
    normalization_scale: float
    squared_error: float = 0.0
    absolute_log_error: float = 0.0
    baseline_squared_error: float = 0.0
    relative_error: float = 0.0
    maximum_relative_error: float = 0.0
    number_of_values: int = 0
    surface_relative_rmse: list[np.ndarray] = field(default_factory=list)
    surface_relative_maximum: list[np.ndarray] = field(default_factory=list)
    
    def update(self, target: torch.Tensor, prediction: torch.Tensor) -> None:
        """
        Accumulate metrics from one normalized prediction batch.
        
        Arguments:
            target (torch.Tensor):
                Normalized target surfaces.
            prediction (torch.Tensor):
                Normalized reconstructions with the same shape as target.
        """
        if target.shape != prediction.shape:
            raise ValueError("Metric target and prediction shapes must match.")
        if target.ndim != 3:
            raise ValueError("Metrics expect tensors with shape (batch, channel, length).")
        
        residual = prediction.detach() - target.detach()
        self.squared_error += float(torch.sum(torch.square(residual)).item())
        self.baseline_squared_error += float(torch.sum(torch.square(target.detach())).item())
        log_residual = residual * self.normalization_scale
        self.absolute_log_error += float(torch.sum(torch.abs(log_residual)).item())
        self.number_of_values += residual.numel()
        
        # A_theta_pred / A_theta_true = 10**(pred_log10 - true_log10).
        # The clamp only prevents diagnostic overflow for an untrained model.
        ratio = torch.pow(10.0, torch.clamp(log_residual, min=-15.0, max=15.0))
        relative = torch.abs(ratio - 1.0)
        self.relative_error += float(torch.sum(relative).item())
        self.maximum_relative_error = max(
            self.maximum_relative_error,
            float(torch.max(relative).item()),
        )
        flattened = relative.flatten(start_dim=1)
        rms = torch.sqrt(torch.mean(torch.square(flattened), dim=1))
        maximum = torch.max(flattened, dim=1).values
        self.surface_relative_rmse.append(rms.cpu().numpy())
        self.surface_relative_maximum.append(maximum.cpu().numpy())
    
    def compute(self) -> dict[str, float]:
        """
        Reduce accumulated statistics into scalar reconstruction metrics.
        
        Returns:
            metrics (dict[str, float]):
                Normalized, log10-space, and physical-space error summaries.
        """
        if self.number_of_values == 0:
            raise ValueError("No samples were supplied to the metric accumulator.")
        if self.baseline_squared_error <= 0.0:
            raise ValueError("Variance denominator is zero; check normalization and data.")
        
        rms_values = np.concatenate(self.surface_relative_rmse)
        maximum_values = np.concatenate(self.surface_relative_maximum)
        normalized_mse = self.squared_error / self.number_of_values
        return {
            "normalized_mse": normalized_mse,
            "log10_rmse": self.normalization_scale * math.sqrt(normalized_mse),
            "log10_mae": self.absolute_log_error / self.number_of_values,
            "variance_recovered": 1.0
            - self.squared_error / self.baseline_squared_error,
            "mean_relative_error": self.relative_error / self.number_of_values,
            "maximum_relative_error": self.maximum_relative_error,
            "surface_relative_rmse_p95": float(np.quantile(rms_values, 0.95)),
            "surface_relative_rmse_p99": float(np.quantile(rms_values, 0.99)),
            "surface_relative_maximum_p95": float(
                np.quantile(maximum_values, 0.95)
            ),
            "surface_relative_maximum_p99": float(
                np.quantile(maximum_values, 0.99)
            ),
            "number_of_surfaces": int(len(rms_values)),
        }
