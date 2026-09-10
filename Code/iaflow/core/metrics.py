"""
Scientifically interpretable reconstruction metrics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
import torch

__all__ = [
    "RECONSTRUCTION_COMPARISON_METRIC_NAMES",
    "RECONSTRUCTION_FRACTIONAL_ERROR_METRIC_NAMES",
    "RECONSTRUCTION_METRIC_NAMES",
    "RELATIVE_ERROR_LOG10_LIMIT",
    "ReconstructionMetrics",
    "ReconstructionObjective",
    "check_reconstruction_metrics",
    "relative_error_from_log10_residual",
]


RECONSTRUCTION_METRIC_NAMES = (
    "normalized_mse",
    "log10_mse",
    "log10_rmse",
    "log10_mae",
    "variance_recovered",
    "mean_relative_error",
    "maximum_relative_error",
    "surface_relative_rmse_p95",
    "surface_relative_rmse_p99",
    "surface_relative_maximum_p95",
    "surface_relative_maximum_p99",
    "number_of_surfaces",
)
RECONSTRUCTION_COMPARISON_METRIC_NAMES = RECONSTRUCTION_METRIC_NAMES[:-1]
RECONSTRUCTION_FRACTIONAL_ERROR_METRIC_NAMES = (
    "log10_mse",
    "log10_rmse",
    "log10_mae",
    "mean_relative_error",
    "maximum_relative_error",
    "surface_relative_rmse_p95",
    "surface_relative_rmse_p99",
    "surface_relative_maximum_p95",
    "surface_relative_maximum_p99",
)
RELATIVE_ERROR_LOG10_LIMIT = 15.0


def relative_error_from_log10_residual(
    log10_residual: torch.Tensor,
) -> torch.Tensor:
    """
    Convert log10 residuals into absolute physical relative errors.
    
    The result is abs(10**(prediction_log10 - target_log10) - 1). Residuals
    are limited only to prevent numerical overflow for untrained models; the
    limit corresponds to relative errors far beyond any useful reconstruction.
    
    Arguments:
        log10_residual (torch.Tensor):
            Prediction minus target in log10 surface space.
    
    Returns:
        relative_error (torch.Tensor):
            Non-negative physical relative errors with the same shape.
    """
    limited_residual = torch.clamp(
        log10_residual,
        min=-RELATIVE_ERROR_LOG10_LIMIT,
        max=RELATIVE_ERROR_LOG10_LIMIT,
    )
    return torch.abs(torch.pow(10.0, limited_residual) - 1.0)


def check_reconstruction_metrics(
    values: object,
    *,
    name: str,
    normalization_scale: float,
) -> None:
    """
    Check the complete reconstruction-metric schema and scalar identities.
    
    Arguments:
        values (object):
            Candidate complete reconstruction-metric mapping.
        name (str):
            Artifact or calculation name used in validation errors.
        normalization_scale (float):
            Positive global RMS relating normalized and log10 surface space.
    """
    if not isinstance(values, Mapping) or set(values) != set(
        RECONSTRUCTION_METRIC_NAMES
    ):
        raise ValueError(
            f"{name} must contain exactly {list(RECONSTRUCTION_METRIC_NAMES)}."
        )
    
    if (
        isinstance(normalization_scale, bool)
        or not isinstance(normalization_scale, Real)
        or not math.isfinite(float(normalization_scale))
        or normalization_scale <= 0.0
    ):
        raise ValueError("normalization_scale must be finite and positive.")
    
    number_of_surfaces = values["number_of_surfaces"]
    if (
        isinstance(number_of_surfaces, bool)
        or not isinstance(number_of_surfaces, Integral)
        or number_of_surfaces <= 0
    ):
        raise ValueError(f"{name} number_of_surfaces must be a positive integer.")
    
    scalar_values: dict[str, float] = {}
    for metric_name in RECONSTRUCTION_COMPARISON_METRIC_NAMES:
        value = values[metric_name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} {metric_name} must be a finite real number.")
        scalar_values[metric_name] = float(value)
    
    nonnegative_names = (
        "normalized_mse",
        "log10_mse",
        "log10_rmse",
        "log10_mae",
        "mean_relative_error",
        "maximum_relative_error",
        "surface_relative_rmse_p95",
        "surface_relative_rmse_p99",
        "surface_relative_maximum_p95",
        "surface_relative_maximum_p99",
    )
    if any(scalar_values[metric_name] < 0.0 for metric_name in nonnegative_names):
        raise ValueError(f"{name} error metrics must be non-negative.")
    
    if scalar_values["variance_recovered"] > 1.0:
        raise ValueError(f"{name} variance_recovered cannot exceed one.")
    
    if not math.isclose(
        scalar_values["log10_rmse"] ** 2,
        scalar_values["log10_mse"],
        rel_tol=1.0e-7,
        abs_tol=1.0e-15,
    ):
        raise ValueError(f"{name} log10_rmse is inconsistent with log10_mse.")
    
    if not math.isclose(
        scalar_values["normalized_mse"] * float(normalization_scale) ** 2,
        scalar_values["log10_mse"],
        rel_tol=1.0e-7,
        abs_tol=1.0e-15,
    ):
        raise ValueError(f"{name} normalized_mse is inconsistent with log10_mse.")
    
    percentile_pairs = (
        ("surface_relative_rmse_p95", "surface_relative_rmse_p99"),
        ("surface_relative_maximum_p95", "surface_relative_maximum_p99"),
    )
    if any(
        scalar_values[p95_name] > scalar_values[p99_name]
        for p95_name, p99_name in percentile_pairs
    ):
        raise ValueError(f"{name} 95th percentiles cannot exceed 99th percentiles.")
    
    if (
        scalar_values["surface_relative_maximum_p99"]
        > scalar_values["maximum_relative_error"]
    ):
        raise ValueError(f"{name} maximum-relative-error summaries are inconsistent.")


@dataclass(slots=True)
class ReconstructionObjective:
    """
    Stream the reconstruction objective and held-out variance recovery.
    """
    
    normalization_scale: float
    squared_error: float = 0.0
    baseline_squared_error: float = 0.0
    number_of_values: int = 0
    
    def update(self, target: torch.Tensor, prediction: torch.Tensor) -> None:
        """
        Accumulate objective statistics from one normalized prediction batch.
        
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
        self.number_of_values += residual.numel()
    
    def compute(self) -> dict[str, float]:
        """
        Reduce accumulated objective statistics into scalar metrics.
        
        Returns:
            metrics (dict[str, float]):
                Normalized loss, log10-space loss, and variance recovery.
        """
        if self.number_of_values == 0:
            raise ValueError("No samples were supplied to the metric accumulator.")
        
        if self.baseline_squared_error <= 0.0:
            raise ValueError("Variance denominator is zero; check normalization and data.")
        
        normalized_mse = self.squared_error / self.number_of_values
        log10_mse = self.normalization_scale**2 * normalized_mse
        return {
            "normalized_mse": normalized_mse,
            "log10_mse": log10_mse,
            "log10_rmse": math.sqrt(log10_mse),
            "variance_recovered": 1.0
            - self.squared_error / self.baseline_squared_error,
        }


@dataclass(slots=True)
class ReconstructionMetrics(ReconstructionObjective):
    """
    Stream normalized-space errors and physical-space tail diagnostics.
    
    The variance denominator is the held-out squared distance from the training
    mean surface. Because normalization uses one global scalar, this is exactly
    the total log10-space variance recovery 1 - SSE / SST.
    """
    
    absolute_log_error: float = 0.0
    relative_error: float = 0.0
    maximum_relative_error: float = 0.0
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
        ReconstructionObjective.update(self, target, prediction)
        residual = prediction.detach() - target.detach()
        log_residual = residual * self.normalization_scale
        self.absolute_log_error += float(torch.sum(torch.abs(log_residual)).item())
        
        relative = relative_error_from_log10_residual(log_residual)
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
        metrics = ReconstructionObjective.compute(self)
        rms_values = np.concatenate(self.surface_relative_rmse)
        maximum_values = np.concatenate(self.surface_relative_maximum)
        metrics.update(
            {
                "log10_mae": self.absolute_log_error / self.number_of_values,
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
        )
        check_reconstruction_metrics(
            metrics,
            name="Computed reconstruction metrics",
            normalization_scale=self.normalization_scale,
        )
        return metrics
