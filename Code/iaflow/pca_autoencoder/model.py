"""
Dense autoencoder over a frozen rank-30 PCA coefficient representation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from ..core.data import NormalizationStats
from ..core.model import AutoEncoder
from .data import CoefficientNormalization
from .transform import PortablePCATransform

__all__ = ["PCAAutoEncoder"]


def _activation(
    name: str,
) -> nn.Module:
    """
    Construct one configured activation layer.
    
    Arguments:
        name (str):
            Supported activation name.
    
    Returns:
        activation (torch.nn.Module):
            Newly constructed stateless activation.
    """
    return {
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
    }[name]()


class _DenseBlock(nn.Sequential):
    """
    Combine one linear map with the configured nonlinear activation.
    """
    def __init__(
        self,
        input_features: int,
        output_features: int,
        activation: str,
    ) -> None:
        """
        Construct one PCA-AE hidden block.
        
        Arguments:
            input_features (int):
                Width entering the block.
            output_features (int):
                Width leaving the block.
            activation (str):
                Configured nonlinear activation.
        """
        super().__init__(
            nn.Linear(input_features, output_features),
            _activation(activation),
        )


def _dense_stack(
    input_features: int,
    hidden_features: Sequence[int],
    output_features: int,
    activation: str,
) -> nn.Sequential:
    """
    Build hidden blocks followed by one unconstrained linear output.
    
    Arguments:
        input_features (int):
            Width entering the dense stack.
        hidden_features (collections.abc.Sequence[int]):
            Ordered hidden-layer widths.
        output_features (int):
            Final linear output width.
        activation (str):
            Activation used after every hidden linear map.
    
    Returns:
        stack (torch.nn.Sequential):
            Configured dense network.
    """
    layers: list[nn.Module] = []
    previous_width = input_features
    for width in hidden_features:
        layers.append(_DenseBlock(previous_width, width, activation))
        previous_width = width
    layers.append(nn.Linear(previous_width, output_features))
    return nn.Sequential(*layers)


class PCAAutoEncoder(AutoEncoder):
    """
    Compress standardized PCA coefficients through a nonlinear bottleneck.
    
    The public autoencoder interface consumes and reconstructs normalized
    ``(N_z, N_k)`` surfaces. Dedicated coefficient methods allow the separate
    PCA-AE trainer to avoid repeated 3,131-feature projections during epochs.
    """
    def __init__(
        self,
        config: object,
        input_shape: tuple[int, int],
        transform: PortablePCATransform,
        surface_normalization: NormalizationStats,
        coefficient_normalization: CoefficientNormalization,
    ) -> None:
        """
        Construct the frozen codec and symmetric coefficient MLP.
        
        Arguments:
            config (object):
                Checked PCA-AE architecture configuration.
            input_shape (tuple[int, int]):
                Redshift and wavenumber surface dimensions.
            transform (PortablePCATransform):
                Frozen non-whitened PCA basis.
            surface_normalization (NormalizationStats):
                Training-only surface mean and global RMS.
            coefficient_normalization (CoefficientNormalization):
                Training-only coefficient mean and component scales.
        """
        super().__init__(config, input_shape)
        if transform.input_shape != self.input_shape:
            raise ValueError("PCA transform and model input shapes differ.")
        if transform.rank != int(config.pca_rank):
            raise ValueError("PCA transform and configured ranks differ.")
        if surface_normalization.mean.shape != self.input_shape:
            raise ValueError("Surface normalization and model shapes differ.")
        if coefficient_normalization.rank != transform.rank:
            raise ValueError("Coefficient normalization and PCA ranks differ.")
        self.register_buffer(
            "pca_mean",
            torch.from_numpy(np.array(transform.mean, copy=True)),
        )
        self.register_buffer(
            "pca_components",
            torch.from_numpy(np.array(transform.components, copy=True)),
        )
        self.register_buffer(
            "surface_mean",
            torch.from_numpy(np.array(surface_normalization.mean, copy=True)),
        )
        self.register_buffer(
            "surface_scale",
            torch.tensor(surface_normalization.scale, dtype=torch.float32),
        )
        self.register_buffer(
            "coefficient_mean",
            torch.from_numpy(np.array(coefficient_normalization.mean, copy=True)),
        )
        self.register_buffer(
            "coefficient_scale",
            torch.from_numpy(np.array(coefficient_normalization.scale, copy=True)),
        )
        self.encoder_network = _dense_stack(
            transform.rank,
            config.dense_hidden,
            config.latent_dim,
            config.activation,
        )
        self.decoder_network = _dense_stack(
            config.latent_dim,
            tuple(reversed(config.dense_hidden)),
            transform.rank,
            config.activation,
        )
        self.transform_sha256 = transform.artifact_sha256
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """
        Initialize dense layers and a near-zero coefficient output map.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        output_layer = self.decoder_network[-1]
        if not isinstance(output_layer, nn.Linear):
            raise RuntimeError("PCA-AE decoder output must be linear.")
        nn.init.normal_(output_layer.weight, mean=0.0, std=1.0e-3)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)
    
    def surfaces_to_coefficients(
        self,
        surfaces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert normalized surfaces to standardized PCA coefficients.
        
        Arguments:
            surfaces (torch.Tensor):
                Normalized surfaces with shape (batch, N_z, N_k).
        
        Returns:
            coefficients (torch.Tensor):
                Standardized coefficients with shape (batch, pca_rank).
        """
        if surfaces.ndim != 3 or tuple(surfaces.shape[1:]) != self.input_shape:
            raise ValueError("PCA-AE surfaces have an invalid shape.")
        log10_surfaces = surfaces * self.surface_scale + self.surface_mean
        flattened = log10_surfaces.flatten(start_dim=1)
        raw_coefficients = (
            flattened - self.pca_mean
        ) @ self.pca_components.T
        return (raw_coefficients - self.coefficient_mean) / self.coefficient_scale
    
    def coefficients_to_surfaces(
        self,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert standardized PCA coefficients to normalized surfaces.
        
        Arguments:
            coefficients (torch.Tensor):
                Standardized coefficients with shape (batch, pca_rank).
        
        Returns:
            surfaces (torch.Tensor):
                Normalized reconstructed surfaces with shape (batch, N_z, N_k).
        """
        if coefficients.ndim != 2 or coefficients.shape[1] != self.config.pca_rank:
            raise ValueError("PCA-AE coefficients have an invalid shape.")
        raw_coefficients = (
            coefficients * self.coefficient_scale + self.coefficient_mean
        )
        flattened = raw_coefficients @ self.pca_components + self.pca_mean
        log10_surfaces = flattened.reshape(len(coefficients), *self.input_shape)
        return (log10_surfaces - self.surface_mean) / self.surface_scale
    
    def encode_coefficients(
        self,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode standardized PCA coefficients into latent coordinates.
        
        Arguments:
            coefficients (torch.Tensor):
                Standardized coefficients with shape (batch, pca_rank).
        
        Returns:
            latent (torch.Tensor):
                Latent coordinates with shape (batch, latent_dim).
        """
        if coefficients.ndim != 2 or coefficients.shape[1] != self.config.pca_rank:
            raise ValueError("PCA-AE coefficients have an invalid shape.")
        return self.encoder_network(coefficients)
    
    def decode_coefficients(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent coordinates into standardized PCA coefficients.
        
        Arguments:
            latent (torch.Tensor):
                Latent coordinates with shape (batch, latent_dim).
        
        Returns:
            coefficients (torch.Tensor):
                Reconstructed standardized coefficients.
        """
        if latent.ndim != 2 or latent.shape[1] != self.config.latent_dim:
            raise ValueError("PCA-AE latent coordinates have an invalid shape.")
        return self.decoder_network(latent)
    
    def reconstruct_coefficients(
        self,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct standardized coefficients through the bottleneck.
        
        Arguments:
            coefficients (torch.Tensor):
                Standardized target coefficients.
        
        Returns:
            reconstructed (torch.Tensor):
                Standardized coefficient reconstruction.
        """
        return self.decode_coefficients(self.encode_coefficients(coefficients))
    
    def encode(
        self,
        surfaces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode normalized surfaces through the frozen PCA front end.
        
        Arguments:
            surfaces (torch.Tensor):
                Normalized surfaces with shape (batch, N_z, N_k).
        
        Returns:
            latent (torch.Tensor):
                Latent coordinates with shape (batch, latent_dim).
        """
        return self.encode_coefficients(self.surfaces_to_coefficients(surfaces))
    
    def decode(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent coordinates through the frozen inverse PCA map.
        
        Arguments:
            latent (torch.Tensor):
                Latent coordinates with shape (batch, latent_dim).
        
        Returns:
            surfaces (torch.Tensor):
                Normalized reconstructed surfaces.
        """
        return self.coefficients_to_surfaces(self.decode_coefficients(latent))
    
    def architecture_summary(self) -> dict[str, object]:
        """
        Return a JSON-safe PCA-AE tensor-flow summary.
        
        Returns:
            summary (dict[str, object]):
                Frozen rank, hidden widths, shapes, and parameter count.
        """
        return {
            "name": "PCA_AE",
            "input_shape": list(self.input_shape),
            "flattened_features": int(np.prod(self.input_shape)),
            "pca_rank": int(self.config.pca_rank),
            "dense_hidden": list(self.config.dense_hidden),
            "latent_dim": int(self.config.latent_dim),
            "activation": self.config.activation,
            "normalization": self.config.normalization,
            "dropout": float(self.config.dropout),
            "coefficient_loss": self.config.coefficient_loss,
            "number_of_parameters": self.number_of_parameters,
            "pca_transform_sha256": self.transform_sha256,
        }
